# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Allocation-free AdamW primitives for online native-MTP learning.

The ordinary PyTorch expression used by the online optimizer materializes one
FP32 copy of every accumulated gradient and several parameter-sized FP32
temporaries while forming the Adam update.  The CUDA path in this module keeps
the same FP32 Adam state and master weights, but performs each element's entire
update in registers.  Its only grad-norm workspace contains one FP32 partial
per 1024 gradient elements.

The prepared serving weight may alias the BF16 gradient buffer.  A kernel reads
each gradient element before writing the corresponding prepared element, so
this is safe and lets the runtime reuse two tensors whose lifetimes are
disjoint.  At a synchronized update boundary the destination may instead be
the graph-stable parameter itself, removing a redundant shadow publication
pass while inference is excluded from those addresses.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

_GRAD_NORM_BLOCK_SIZE = 1024
_ADAMW_BLOCK_SIZE = 1024


@triton.jit
def _scaled_grad_norm_partial_kernel(
    gradient,
    partials,
    num_elements,
    denominator,
    contribution_scale,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    value = tl.load(gradient + offsets, mask=mask, other=0.0).to(tl.float32)
    value = value / denominator
    partial = tl.sum(value * value, axis=0) * contribution_scale
    tl.store(partials + tl.program_id(0), partial)


@triton.jit
def _adamw_prepare_kernel(
    gradient,
    first_moment,
    second_moment,
    master_parameter,
    prepared_parameter,
    num_elements,
    denominator,
    clip_scale,
    beta1,
    beta2,
    correction1,
    correction2,
    epsilon,
    learning_rate,
    decay_multiplier,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT_DECAY: tl.constexpr,
    CLEAR_GRADIENT: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    gradient_value = tl.load(gradient + offsets, mask=mask, other=0.0).to(tl.float32)
    gradient_value = (gradient_value / denominator) * clip_scale
    first_value = tl.load(first_moment + offsets, mask=mask, other=0.0).to(tl.float32)
    second_value = tl.load(second_moment + offsets, mask=mask, other=0.0).to(tl.float32)
    master_value = tl.load(master_parameter + offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    first_value = beta1 * first_value + (1.0 - beta1) * gradient_value
    second_value = beta2 * second_value + (1.0 - beta2) * (
        gradient_value * gradient_value
    )
    denominator_value = tl.sqrt(second_value / correction2) + epsilon
    update = (first_value / correction1) / denominator_value
    if HAS_WEIGHT_DECAY:
        master_value *= decay_multiplier
    master_value -= learning_rate * update

    tl.store(first_moment + offsets, first_value, mask=mask)
    tl.store(second_moment + offsets, second_value, mask=mask)
    tl.store(master_parameter + offsets, master_value, mask=mask)
    tl.store(prepared_parameter + offsets, master_value, mask=mask)
    if CLEAR_GRADIENT:
        tl.store(gradient + offsets, 0.0, mask=mask)


def grad_norm_num_partials(gradients: Sequence[torch.Tensor]) -> int:
    """Return the exact FP32 partial count required by the Triton reduction."""

    return sum(
        triton.cdiv(gradient.numel(), _GRAD_NORM_BLOCK_SIZE) for gradient in gradients
    )


def _validate_grad_norm_inputs(
    gradients: Sequence[torch.Tensor],
    denominator: float,
    contribution_scales: Sequence[float],
    partials_out: Optional[torch.Tensor],
) -> None:
    if not gradients:
        raise ValueError("at least one gradient is required")
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("denominator must be finite and positive")
    if len(contribution_scales) != len(gradients):
        raise ValueError("one contribution scale is required per gradient")
    device = gradients[0].device
    for gradient, contribution_scale in zip(gradients, contribution_scales):
        if not gradient.is_floating_point():
            raise ValueError("gradients must be floating-point tensors")
        if gradient.device != device:
            raise ValueError("all gradients must be on the same device")
        if not math.isfinite(contribution_scale) or contribution_scale < 0.0:
            raise ValueError("contribution scales must be finite and nonnegative")
    if partials_out is not None:
        required = grad_norm_num_partials(gradients)
        if partials_out.device != device:
            raise ValueError("partials_out must be on the gradient device")
        if partials_out.dtype != torch.float32:
            raise ValueError("partials_out must have dtype torch.float32")
        if not partials_out.is_contiguous() or partials_out.ndim != 1:
            raise ValueError("partials_out must be a contiguous vector")
        if partials_out.numel() < required:
            raise ValueError(
                f"partials_out needs {required} elements, got "
                f"{partials_out.numel()}"
            )
        if any(torch._C._overlaps(partials_out, gradient) for gradient in gradients):
            raise ValueError("partials_out must not overlap any gradient")


def _is_exact_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two contiguous tensors describe the same typed region."""

    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.storage_offset() == right.storage_offset()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _can_use_triton_grad_norm(gradients: Sequence[torch.Tensor]) -> bool:
    device = gradients[0].device
    return (
        device.type == "cuda"
        and torch.version.hip is None
        and all(
            gradient.device == device
            and gradient.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and gradient.is_contiguous()
            for gradient in gradients
        )
    )


@torch.no_grad()
def online_mtp_scaled_grad_norm_sq(
    gradients: Sequence[torch.Tensor],
    *,
    denominator: float,
    contribution_scales: Optional[Sequence[float]] = None,
    partials_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ``sum_i scale_i * ||gradient_i / denominator||^2``.

    CUDA tensors use a two-stage reduction.  The first stage writes one FP32
    scalar per 1024 elements into ``partials_out``; PyTorch then reduces this
    compact vector to one scalar.  No gradient-sized FP32 tensor is created.
    The fallback intentionally mirrors the former ATen expression.
    """

    gradients = tuple(gradients)
    if contribution_scales is None:
        contribution_scales = (1.0,) * len(gradients)
    else:
        contribution_scales = tuple(contribution_scales)
    _validate_grad_norm_inputs(
        gradients, denominator, contribution_scales, partials_out
    )

    if not _can_use_triton_grad_norm(gradients):
        norm_sq = torch.zeros((), dtype=torch.float32, device=gradients[0].device)
        for gradient, contribution_scale in zip(gradients, contribution_scales):
            contribution = (gradient.float() / denominator).square().sum()
            norm_sq.add_(contribution, alpha=contribution_scale)
        return norm_sq

    required = grad_norm_num_partials(gradients)
    if partials_out is None:
        partials_out = torch.empty(
            required, dtype=torch.float32, device=gradients[0].device
        )
    used = partials_out[:required]
    partial_start = 0
    for gradient, contribution_scale in zip(gradients, contribution_scales):
        num_partials = triton.cdiv(gradient.numel(), _GRAD_NORM_BLOCK_SIZE)
        if num_partials == 0:
            continue
        partial_stop = partial_start + num_partials
        _scaled_grad_norm_partial_kernel[(num_partials,)](
            gradient,
            used[partial_start:partial_stop],
            gradient.numel(),
            denominator,
            contribution_scale,
            BLOCK_SIZE=_GRAD_NORM_BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
        )
        partial_start = partial_stop
    return used.sum()


def _validate_adamw_inputs(
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    master_parameter: torch.Tensor,
    prepared_parameter: torch.Tensor,
    *,
    denominator: float,
    clip_scale: float,
    betas: tuple[float, float],
    corrections: tuple[float, float],
    epsilon: float,
    learning_rate: float,
    weight_decay: float,
) -> None:
    tensors = (
        first_moment,
        second_moment,
        master_parameter,
        prepared_parameter,
    )
    for tensor in tensors:
        if tensor.shape != gradient.shape:
            raise ValueError("all AdamW tensors must have the gradient shape")
        if tensor.device != gradient.device:
            raise ValueError("all AdamW tensors must be on the same device")
        if not tensor.is_contiguous():
            raise ValueError("all AdamW tensors must be contiguous")
    if not gradient.is_contiguous():
        raise ValueError("gradient must be contiguous")
    if not gradient.is_floating_point() or not prepared_parameter.is_floating_point():
        raise ValueError("gradient and prepared parameter must be floating-point")
    if any(
        tensor.dtype != torch.float32
        for tensor in (first_moment, second_moment, master_parameter)
    ):
        raise ValueError("Adam moments and master parameter must be FP32")

    # The fused kernel supports exactly one aliasing pattern: prepared output
    # may be the same typed tensor region as the consumed gradient.  A shifted
    # overlap is unsafe because one program can overwrite a value before the
    # neighboring program loads it.  Optimizer state must remain independent
    # from both inputs/outputs and from the other state tensors.
    if torch._C._overlaps(gradient, prepared_parameter) and not _is_exact_alias(
        gradient, prepared_parameter
    ):
        raise ValueError(
            "gradient and prepared_parameter must be exactly aliased or disjoint"
        )
    named_tensors = (
        ("gradient", gradient),
        ("prepared_parameter", prepared_parameter),
        ("first_moment", first_moment),
        ("second_moment", second_moment),
        ("master_parameter", master_parameter),
    )
    for state_index in range(2, len(named_tensors)):
        state_name, state = named_tensors[state_index]
        for other_name, other in named_tensors[:state_index]:
            if torch._C._overlaps(state, other):
                raise ValueError(
                    f"{state_name} must not overlap {other_name}"
                )
    scalars = (
        denominator,
        clip_scale,
        *betas,
        *corrections,
        epsilon,
        learning_rate,
        weight_decay,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("AdamW scalar arguments must be finite")
    if denominator <= 0.0:
        raise ValueError("denominator must be positive")
    if clip_scale < 0.0:
        raise ValueError("clip_scale must be nonnegative")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    if corrections[0] <= 0.0 or corrections[1] <= 0.0:
        raise ValueError("bias corrections must be positive")


def _can_use_triton_adamw(
    gradient: torch.Tensor, prepared_parameter: torch.Tensor
) -> bool:
    return (
        gradient.device.type == "cuda"
        and torch.version.hip is None
        and gradient.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and prepared_parameter.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )


@torch.no_grad()
def online_mtp_adamw_prepare_(
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    master_parameter: torch.Tensor,
    prepared_parameter: torch.Tensor,
    *,
    denominator: float,
    clip_scale: float,
    betas: tuple[float, float],
    corrections: tuple[float, float],
    epsilon: float,
    learning_rate: float,
    weight_decay: float,
    clear_gradient: bool = False,
) -> None:
    """Prepare one AdamW parameter in place without full-size temporaries.

    ``prepared_parameter`` may be the same tensor as ``gradient``.  This is the
    intended serving configuration: every gradient element is consumed before
    its location is overwritten with the BF16 shadow weight.  ``clear_gradient``
    is only valid for distinct storage and fuses the ordinary post-update zero.
    """

    _validate_adamw_inputs(
        gradient,
        first_moment,
        second_moment,
        master_parameter,
        prepared_parameter,
        denominator=denominator,
        clip_scale=clip_scale,
        betas=betas,
        corrections=corrections,
        epsilon=epsilon,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    beta1, beta2 = betas
    correction1, correction2 = corrections
    if clear_gradient and torch._C._overlaps(gradient, prepared_parameter):
        raise ValueError(
            "clear_gradient requires distinct gradient and prepared storage"
        )
    if not _can_use_triton_adamw(gradient, prepared_parameter):
        scaled_gradient = gradient.float() / denominator
        scaled_gradient.mul_(clip_scale)
        first_moment.mul_(beta1).add_(scaled_gradient, alpha=1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(
            scaled_gradient, scaled_gradient, value=1.0 - beta2
        )
        update = (first_moment / correction1) / (
            (second_moment / correction2).sqrt().add_(epsilon)
        )
        if weight_decay:
            master_parameter.mul_(1.0 - learning_rate * weight_decay)
        master_parameter.add_(update, alpha=-learning_rate)
        prepared_parameter.copy_(master_parameter)
        if clear_gradient:
            gradient.zero_()
        return

    num_elements = gradient.numel()
    if num_elements == 0:
        return
    _adamw_prepare_kernel[(triton.cdiv(num_elements, _ADAMW_BLOCK_SIZE),)](
        gradient,
        first_moment,
        second_moment,
        master_parameter,
        prepared_parameter,
        num_elements,
        denominator,
        clip_scale,
        beta1,
        beta2,
        correction1,
        correction2,
        epsilon,
        learning_rate,
        1.0 - learning_rate * weight_decay,
        BLOCK_SIZE=_ADAMW_BLOCK_SIZE,
        HAS_WEIGHT_DECAY=weight_decay != 0.0,
        CLEAR_GRADIENT=clear_gradient,
        num_warps=4,
        num_stages=1,
    )
