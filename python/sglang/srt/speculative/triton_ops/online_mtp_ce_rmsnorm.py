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

"""Fused hard-CE tail and Gemma RMSNorm backward for online native MTP.

The online learner receives compact, label-independent CE statistics from the
real inference forward.  Once target verification supplies the hard label row,
the remaining work is

* reconstructing the normalized hidden state for loss reporting,
* forming ``E_p[W] - W_target``,
* backpropagating it through the final Gemma RMSNorm, and
* accumulating the final-norm weight gradient.

Expressing that chain as ordinary ATen operations launches many small kernels
and materializes several FP32 hidden-sized temporaries.  The CUDA BF16 fast path
below uses two Triton kernels: a row-wise input-gradient/loss kernel and a
column-wise weight-gradient reduction.  The latter avoids hidden-wide atomics.

The explicit casts are part of the operation's contract.  They preserve the
rounding points of the existing mixed-precision reference path: normalized
hidden values are rounded to BF16 before the target-logit dot, RMSNorm input
gradients are rounded to BF16, and each FP32 batch weight gradient is rounded to
BF16 before being added to the persistent BF16 accumulator.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_MAX_TRITON_HIDDEN_SIZE = 8192
_DWEIGHT_BLOCK_HIDDEN = 128
_DWEIGHT_BLOCK_ROWS = 32


@triton.jit
def _pack_local_ce_for_tp_reduce_kernel(
    local_expected,
    norm_input,
    raw_norm_weight,
    local_lm_head_weight,
    labels,
    packed_output,
    num_rows,
    SHARD_START: tl.constexpr,
    SHARD_STOP: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    """Pack local CE gradient contribution and one-owner target logit."""

    row = tl.program_id(0)
    hidden = tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = hidden < HIDDEN_SIZE
    offsets = row * HIDDEN_SIZE + hidden

    expected = tl.load(
        local_expected + offsets, mask=hidden_mask, other=0.0
    ).to(tl.float32)
    label = tl.load(labels + row).to(tl.int64)
    owns_target = (label >= SHARD_START) & (label < SHARD_STOP)
    local_row = tl.minimum(
        tl.maximum(label - SHARD_START, 0), SHARD_STOP - SHARD_START - 1
    )
    target = tl.load(
        local_lm_head_weight + local_row * HIDDEN_SIZE + hidden,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    target = tl.where(owns_target, target, 0.0)

    # The first N*H elements form a contiguous FP32 gradient matrix.  Target
    # logits follow it, then the Python wrapper adds at most three zero padding
    # elements so every custom-all-reduce input is 16-byte aligned.
    tl.store(packed_output + offsets, expected - target, mask=hidden_mask)

    x = tl.load(norm_input + offsets, mask=hidden_mask, other=0.0).to(tl.float32)
    effective_weight = (
        tl.load(raw_norm_weight + hidden, mask=hidden_mask, other=0.0).to(
            tl.float32
        )
        + 1.0
    )
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN_SIZE + EPSILON)
    # Match the existing CE/RMSNorm kernel's BF16 serving boundary and
    # reduction geometry exactly.  Nonowners contribute an exact zero scalar.
    final_hidden = (x * inv_rms * effective_weight).to(tl.bfloat16)
    target_logit = tl.sum(final_hidden.to(tl.float32) * target, axis=0)
    tl.store(packed_output + num_rows * HIDDEN_SIZE + row, target_logit)


@torch.no_grad()
def pack_local_ce_for_tp_reduce(
    local_expected: torch.Tensor,
    norm_input: torch.Tensor,
    raw_norm_weight: torch.Tensor,
    local_lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    shard_start: int,
    shard_stop: int,
    epsilon: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """Pack one rank's CE contribution for a single aligned FP32 TP sum.

    Layout is ``[N*H gradient values][N target logits][0--3 pad values]``.
    The returned one-dimensional view is always 16-byte aligned in size, so
    SGLang's CUDA custom all-reduce remains eligible for every row count.
    """

    if local_expected.ndim != 2 or norm_input.shape != local_expected.shape:
        raise ValueError("local_expected and norm_input must be equal-shape matrices")
    rows, hidden_size = local_expected.shape
    if rows <= 0 or hidden_size <= 0:
        raise ValueError("local CE inputs must be nonempty")
    if labels.numel() != rows:
        raise ValueError("labels must contain one id per local CE row")
    if raw_norm_weight.shape != (hidden_size,):
        raise ValueError("raw_norm_weight shape does not match hidden size")
    if shard_stop <= shard_start:
        raise ValueError("local vocabulary shard must be nonempty")
    if local_lm_head_weight.ndim != 2 or local_lm_head_weight.shape[1] != hidden_size:
        raise ValueError("local LM-head weight shape does not match hidden size")
    if local_lm_head_weight.shape[0] < shard_stop - shard_start:
        raise ValueError("local LM-head weight does not cover its vocabulary shard")
    if local_expected.dtype != local_lm_head_weight.dtype:
        raise ValueError("local expected vector must preserve LM-head GEMM dtype")
    if norm_input.dtype != local_lm_head_weight.dtype:
        raise ValueError("norm input and LM-head weight dtypes must match")
    if out.dtype != torch.float32 or out.device != local_expected.device:
        raise ValueError("packed CE output must be FP32 on the input device")
    logical_elements = rows * hidden_size + rows
    aligned_elements = (logical_elements + 3) // 4 * 4
    if out.ndim != 1 or out.numel() < aligned_elements or not out.is_contiguous():
        raise ValueError(
            f"packed CE output needs {aligned_elements} contiguous elements"
        )
    packed = out[:aligned_elements]
    if aligned_elements > logical_elements:
        packed[logical_elements:aligned_elements].zero_()

    can_use_triton = (
        local_expected.device.type == "cuda"
        and torch.version.hip is None
        and local_expected.dtype == torch.bfloat16
        and all(
            tensor.is_contiguous()
            for tensor in (
                local_expected,
                norm_input,
                raw_norm_weight,
                local_lm_head_weight,
                labels,
            )
        )
        and hidden_size <= _MAX_TRITON_HIDDEN_SIZE
    )
    if can_use_triton:
        block_hidden = triton.next_power_of_2(hidden_size)
        _pack_local_ce_for_tp_reduce_kernel[(rows,)](
            local_expected,
            norm_input,
            raw_norm_weight,
            local_lm_head_weight,
            labels.reshape(-1),
            packed,
            rows,
            SHARD_START=shard_start,
            SHARD_STOP=shard_stop,
            HIDDEN_SIZE=hidden_size,
            EPSILON=epsilon,
            BLOCK_HIDDEN=block_hidden,
            num_warps=8,
        )
        return packed

    labels = labels.reshape(-1).long()
    owns_target = (labels >= shard_start) & (labels < shard_stop)
    local_rows = (labels - shard_start).clamp(0, shard_stop - shard_start - 1)
    target = local_lm_head_weight[local_rows].clone()
    target.mul_(owns_target[:, None])
    packed[: rows * hidden_size].view(rows, hidden_size).copy_(
        local_expected.float() - target.float()
    )
    x = norm_input.float()
    inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon)
    final_hidden = (
        x * inv_rms * (raw_norm_weight.float() + 1.0)
    ).to(norm_input.dtype)
    packed[rows * hidden_size : logical_elements].copy_(
        (final_hidden.float() * target.float()).sum(dim=-1)
    )
    return packed


@triton.jit
def _online_mtp_ce_rmsnorm_dx_kernel(
    norm_input,
    expected_weight,
    target_weight,
    raw_norm_weight,
    logsumexp,
    loss_mask,
    grad_input,
    inv_rms_output,
    loss_output,
    HIDDEN_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    HAS_MASK: tl.constexpr,
    INPLACE_CE_BUFFERS: tl.constexpr,
):
    row = tl.program_id(0)
    hidden = tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = hidden < HIDDEN_SIZE
    offsets = row * HIDDEN_SIZE + hidden

    x = tl.load(norm_input + offsets, mask=hidden_mask, other=0.0).to(tl.float32)
    expected = tl.load(expected_weight + offsets, mask=hidden_mask, other=0.0).to(
        tl.float32
    )
    target = tl.load(target_weight + offsets, mask=hidden_mask, other=0.0).to(
        tl.float32
    )
    effective_weight = (
        tl.load(raw_norm_weight + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
        + 1.0
    )
    if HAS_MASK:
        row_scale = tl.load(loss_mask + row).to(tl.float32)
    else:
        row_scale = 1.0

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN_SIZE + EPSILON)

    # Match the serving/reference path's BF16 rounding before the target dot.
    final_hidden = (x * inv_rms * effective_weight).to(tl.bfloat16)
    target_logit = tl.sum(final_hidden.to(tl.float32) * target, axis=0)
    row_logsumexp = tl.load(logsumexp + row).to(tl.float32)
    loss = tl.maximum(row_logsumexp - target_logit, 0.0) * row_scale

    grad_hidden = (expected - target) * row_scale
    weighted_grad = grad_hidden * effective_weight
    projection = tl.sum(weighted_grad * x, axis=0) / HIDDEN_SIZE
    inv_cubed = inv_rms * inv_rms * inv_rms
    dx = inv_rms * weighted_grad - x * inv_cubed * projection

    # The serving integration no longer needs either compact-CE input after
    # this kernel.  Reusing them avoids a second hidden-sized BF16 allocation
    # for ``dx`` and leaves the FP32 quantity consumed by the reduction kernel
    # in ``expected_weight``.  All target-dependent work above is complete
    # before the destructive stores.
    if INPLACE_CE_BUFFERS:
        tl.store(expected_weight + offsets, grad_hidden, mask=hidden_mask)
    tl.store(grad_input + offsets, dx, mask=hidden_mask)
    tl.store(inv_rms_output + row, inv_rms)
    tl.store(loss_output + row, loss)


@triton.jit
def _online_mtp_precomputed_ce_rmsnorm_dx_kernel(
    norm_input,
    grad_hidden,
    raw_norm_weight,
    logsumexp,
    target_logits,
    loss_mask,
    grad_input,
    inv_rms_output,
    loss_output,
    HIDDEN_SIZE: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """RMSNorm backward when TP reduction already formed CE hidden gradient."""

    row = tl.program_id(0)
    hidden = tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = hidden < HIDDEN_SIZE
    offsets = row * HIDDEN_SIZE + hidden

    x = tl.load(norm_input + offsets, mask=hidden_mask, other=0.0).to(tl.float32)
    gradient = tl.load(grad_hidden + offsets, mask=hidden_mask, other=0.0).to(
        tl.float32
    )
    effective_weight = (
        tl.load(raw_norm_weight + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
        + 1.0
    )
    if HAS_MASK:
        row_scale = tl.load(loss_mask + row).to(tl.float32)
        gradient *= row_scale
        # The column reduction runs after this kernel on the same stream.  Keep
        # its input masked in place rather than allocating another FP32 matrix.
        tl.store(grad_hidden + offsets, gradient, mask=hidden_mask)
    else:
        row_scale = 1.0

    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN_SIZE + EPSILON)
    weighted_grad = gradient * effective_weight
    projection = tl.sum(weighted_grad * x, axis=0) / HIDDEN_SIZE
    inv_cubed = inv_rms * inv_rms * inv_rms
    dx = inv_rms * weighted_grad - x * inv_cubed * projection

    row_logsumexp = tl.load(logsumexp + row).to(tl.float32)
    target_logit = tl.load(target_logits + row).to(tl.float32)
    loss = tl.maximum(row_logsumexp - target_logit, 0.0) * row_scale

    tl.store(grad_input + offsets, dx, mask=hidden_mask)
    tl.store(inv_rms_output + row, inv_rms)
    tl.store(loss_output + row, loss)


@triton.jit
def _online_mtp_rmsnorm_dweight_kernel(
    norm_input,
    expected_weight,
    target_weight,
    inv_rms,
    loss_mask,
    norm_gradient_accumulator,
    num_rows,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    PRECOMPUTED_GRAD_HIDDEN: tl.constexpr,
):
    hidden = tl.program_id(0) * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    hidden_mask = hidden < HIDDEN_SIZE
    gradient_sum = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)

    for row_start in tl.range(0, num_rows, BLOCK_ROWS):
        rows = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < num_rows
        offsets = rows[:, None] * HIDDEN_SIZE + hidden[None, :]
        mask = row_mask[:, None] & hidden_mask[None, :]

        x = tl.load(norm_input + offsets, mask=mask, other=0.0).to(tl.float32)
        expected = tl.load(expected_weight + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        row_inv_rms = tl.load(inv_rms + rows, mask=row_mask, other=0.0).to(tl.float32)

        if PRECOMPUTED_GRAD_HIDDEN:
            # The row kernel has already formed
            # ``(expected_weight - target_weight) * loss_mask`` exactly once.
            contribution = expected
        else:
            target = tl.load(target_weight + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            contribution = expected - target
            if HAS_MASK:
                row_scale = tl.load(loss_mask + rows, mask=row_mask, other=0).to(
                    tl.float32
                )
                contribution *= row_scale[:, None]

        contribution = contribution * x * row_inv_rms[:, None]
        gradient_sum += tl.sum(contribution, axis=0)

    accumulator = tl.load(
        norm_gradient_accumulator + hidden,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    # Match ``grad_weight.float().to(bfloat16)`` followed by BF16 ``add_``.
    rounded_batch_gradient = gradient_sum.to(tl.bfloat16).to(tl.float32)
    tl.store(
        norm_gradient_accumulator + hidden,
        accumulator + rounded_batch_gradient,
        mask=hidden_mask,
    )


def _validate_output(
    name: str,
    tensor: Optional[torch.Tensor],
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor is None:
        return
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _storage_ranges_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Exact byte-range overlap for contiguous views, conservative otherwise.

    ``torch._C._overlaps`` reports true for any two views backed by the same
    storage, even when their byte intervals are disjoint.  Online MTP dtype
    slabs intentionally create such disjoint contiguous views.  Preserve the
    conservative answer for arbitrary strided tensors, while allowing ranges
    that can be proven disjoint.
    """

    if not torch._C._overlaps(lhs, rhs):
        return False
    if not lhs.is_contiguous() or not rhs.is_contiguous():
        return True
    if lhs.untyped_storage().data_ptr() != rhs.untyped_storage().data_ptr():
        return False
    # Absolute addresses also handle views that reinterpret one untyped
    # storage through different dtypes; storage_offset is dtype-relative.
    lhs_start = lhs.data_ptr()
    rhs_start = rhs.data_ptr()
    lhs_end = lhs_start + lhs.numel() * lhs.element_size()
    rhs_end = rhs_start + rhs.numel() * rhs.element_size()
    return lhs_start < rhs_end and rhs_start < lhs_end


def _validate_inputs(
    norm_input: torch.Tensor,
    expected_weight: torch.Tensor,
    target_weight: torch.Tensor,
    raw_norm_weight: torch.Tensor,
    logsumexp: torch.Tensor,
    norm_gradient_accumulator: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    grad_input_out: Optional[torch.Tensor],
    inv_rms_out: Optional[torch.Tensor],
    losses_out: Optional[torch.Tensor],
    epsilon: float,
) -> Tuple[int, int]:
    if norm_input.ndim != 2:
        raise ValueError(
            f"norm_input must be a matrix, got shape {tuple(norm_input.shape)}"
        )
    rows, hidden_size = norm_input.shape
    if hidden_size <= 0:
        raise ValueError("norm_input must have a nonempty hidden dimension")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")

    same_shape = {
        "expected_weight": expected_weight,
        "target_weight": target_weight,
    }
    for name, tensor in same_shape.items():
        if tuple(tensor.shape) != tuple(norm_input.shape):
            raise ValueError(
                f"{name} must have shape {tuple(norm_input.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
    vectors = {
        "raw_norm_weight": (raw_norm_weight, (hidden_size,)),
        "logsumexp": (logsumexp, (rows,)),
        "norm_gradient_accumulator": (
            norm_gradient_accumulator,
            (hidden_size,),
        ),
    }
    for name, (tensor, shape) in vectors.items():
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )

    tensors = {
        "expected_weight": expected_weight,
        "target_weight": target_weight,
        "raw_norm_weight": raw_norm_weight,
        "logsumexp": logsumexp,
        "norm_gradient_accumulator": norm_gradient_accumulator,
    }
    if not norm_input.is_floating_point():
        raise ValueError("norm_input must be floating-point")
    for name, tensor in tensors.items():
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating-point")
        if tensor.device != norm_input.device:
            raise ValueError(
                f"{name} must be on {norm_input.device}, got {tensor.device}"
            )
    if loss_mask is not None:
        if tuple(loss_mask.shape) != (rows,):
            raise ValueError(
                f"loss_mask must have shape {(rows,)}, " f"got {tuple(loss_mask.shape)}"
            )
        if loss_mask.device != norm_input.device:
            raise ValueError(
                f"loss_mask must be on {norm_input.device}, got {loss_mask.device}"
            )
        if loss_mask.dtype != torch.bool:
            raise ValueError(
                f"loss_mask must have dtype torch.bool, got {loss_mask.dtype}"
            )

    _validate_output(
        "grad_input_out",
        grad_input_out,
        shape=(rows, hidden_size),
        dtype=norm_input.dtype,
        device=norm_input.device,
    )
    _validate_output(
        "inv_rms_out",
        inv_rms_out,
        shape=(rows,),
        dtype=torch.float32,
        device=norm_input.device,
    )
    _validate_output(
        "losses_out",
        losses_out,
        shape=(rows,),
        dtype=torch.float32,
        device=norm_input.device,
    )

    if torch._C._overlaps(raw_norm_weight, norm_gradient_accumulator):
        raise ValueError(
            "raw_norm_weight must not overlap norm_gradient_accumulator"
        )
    if (
        inv_rms_out is not None
        and losses_out is not None
        and torch._C._overlaps(inv_rms_out, losses_out)
    ):
        raise ValueError("inv_rms_out must not overlap losses_out")

    if grad_input_out is not None:
        for name, source in (
            ("norm_input", norm_input),
            ("expected_weight", expected_weight),
            ("target_weight", target_weight),
        ):
            if torch._C._overlaps(grad_input_out, source):
                raise ValueError(
                    f"grad_input_out must not overlap {name}; the second fused "
                    "kernel still consumes all inputs"
                )
    return rows, hidden_size


def _validate_inplace_ce_buffers(
    norm_input: torch.Tensor,
    expected_weight: torch.Tensor,
    target_weight: torch.Tensor,
    raw_norm_weight: torch.Tensor,
    logsumexp: torch.Tensor,
    norm_gradient_accumulator: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    inv_rms_out: Optional[torch.Tensor],
    losses_out: Optional[torch.Tensor],
) -> None:
    """Validate the stronger contract of the destructive fast path."""

    if expected_weight.dtype != torch.float32:
        raise ValueError(
            "inplace_ce_buffers requires expected_weight to have dtype "
            f"torch.float32, got {expected_weight.dtype}"
        )
    if target_weight.dtype != norm_input.dtype:
        raise ValueError(
            "inplace_ce_buffers requires target_weight and norm_input to have "
            f"the same dtype, got {target_weight.dtype} and {norm_input.dtype}"
        )

    # Both CE tensors become outputs.  Reject aliases with every other live
    # operand so that the destructive contract stays valid on both the Triton
    # and ATen paths (including unusual tensor views supplied by callers).
    other_tensors = (
        ("norm_input", norm_input),
        ("raw_norm_weight", raw_norm_weight),
        ("logsumexp", logsumexp),
        ("norm_gradient_accumulator", norm_gradient_accumulator),
        ("loss_mask", loss_mask),
        ("inv_rms_out", inv_rms_out),
        ("losses_out", losses_out),
    )
    for buffer_name, buffer in (
        ("expected_weight", expected_weight),
        ("target_weight", target_weight),
    ):
        peer_name = (
            "target_weight" if buffer_name == "expected_weight" else "expected_weight"
        )
        peer = target_weight if buffer_name == "expected_weight" else expected_weight
        if torch._C._overlaps(buffer, peer):
            raise ValueError(
                f"inplace_ce_buffers requires {buffer_name} not to overlap {peer_name}"
            )
        for other_name, other in other_tensors:
            if other is not None and torch._C._overlaps(buffer, other):
                raise ValueError(
                    "inplace_ce_buffers requires "
                    f"{buffer_name} not to overlap {other_name}"
                )


def _aten_fallback(
    norm_input: torch.Tensor,
    expected_weight: torch.Tensor,
    target_weight: torch.Tensor,
    raw_norm_weight: torch.Tensor,
    logsumexp: torch.Tensor,
    norm_gradient_accumulator: torch.Tensor,
    *,
    epsilon: float,
    loss_mask: Optional[torch.Tensor],
    grad_input_out: torch.Tensor,
    inv_rms_out: torch.Tensor,
    losses_out: torch.Tensor,
    inplace_ce_buffers: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    effective_weight = raw_norm_weight.detach().float() + 1.0
    inv_rms = torch.rsqrt(
        norm_input.float().square().mean(dim=-1, keepdim=True) + epsilon
    )
    inv_rms_out.copy_(inv_rms.reshape(-1))

    # Keep the same BF16 final-hidden rounding point as serving.
    final_hidden = (norm_input.float() * inv_rms * effective_weight).to(
        norm_input.dtype
    )
    target_logits = (final_hidden.float() * target_weight.float()).sum(dim=-1)
    losses = (logsumexp.float() - target_logits).clamp_min_(0.0)
    if loss_mask is None:
        row_scale = torch.ones_like(losses)
    else:
        row_scale = loss_mask.float()
        losses = losses * loss_mask
    losses_out.copy_(losses)

    grad_hidden = (expected_weight.float() - target_weight.float()) * row_scale[:, None]
    x_fp32 = norm_input.float()
    weighted_grad = grad_hidden.float() * effective_weight
    projection = (weighted_grad * x_fp32).mean(dim=-1, keepdim=True)
    grad_input = inv_rms * weighted_grad - x_fp32 * inv_rms.float().pow(3) * projection
    grad_input_out.copy_(grad_input)
    grad_weight = (grad_hidden.float() * x_fp32 * inv_rms.float()).sum(dim=0)
    norm_gradient_accumulator.add_(grad_weight.to(norm_gradient_accumulator.dtype))
    if inplace_ce_buffers:
        expected_weight.copy_(grad_hidden)
    return grad_input_out, losses_out


def _can_use_triton(
    norm_input: torch.Tensor,
    expected_weight: torch.Tensor,
    target_weight: torch.Tensor,
    raw_norm_weight: torch.Tensor,
    logsumexp: torch.Tensor,
    norm_gradient_accumulator: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    grad_input_out: torch.Tensor,
    inv_rms_out: torch.Tensor,
    losses_out: torch.Tensor,
) -> bool:
    return (
        norm_input.device.type == "cuda"
        and torch.version.hip is None
        and norm_input.dtype == torch.bfloat16
        and expected_weight.dtype == torch.float32
        and target_weight.dtype == torch.bfloat16
        and raw_norm_weight.dtype == torch.bfloat16
        and logsumexp.dtype == torch.float32
        and norm_gradient_accumulator.dtype == torch.bfloat16
        and grad_input_out.dtype == torch.bfloat16
        and inv_rms_out.dtype == torch.float32
        and losses_out.dtype == torch.float32
        and norm_input.shape[1] <= _MAX_TRITON_HIDDEN_SIZE
        and all(
            tensor.is_contiguous()
            for tensor in (
                norm_input,
                expected_weight,
                target_weight,
                raw_norm_weight,
                logsumexp,
                norm_gradient_accumulator,
                grad_input_out,
                inv_rms_out,
                losses_out,
            )
        )
        and (loss_mask is None or loss_mask.is_contiguous())
    )


@torch.no_grad()
def online_mtp_ce_rmsnorm_backward(
    norm_input: torch.Tensor,
    expected_weight: torch.Tensor,
    target_weight: torch.Tensor,
    raw_norm_weight: torch.Tensor,
    logsumexp: torch.Tensor,
    norm_gradient_accumulator: torch.Tensor,
    *,
    epsilon: float,
    loss_mask: Optional[torch.Tensor] = None,
    grad_input_out: Optional[torch.Tensor] = None,
    inv_rms_out: Optional[torch.Tensor] = None,
    losses_out: Optional[torch.Tensor] = None,
    inplace_ce_buffers: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finish compact hard CE and backpropagate through Gemma RMSNorm.

    ``norm_gradient_accumulator`` is updated in place.  The returned tensors are
    ``(grad_input, per_row_losses)``.  Supplying the three optional output/scratch
    tensors removes all allocations from the default fast path.

    Setting ``inplace_ce_buffers=True`` enables the destructive serving fast
    path.  It overwrites FP32 ``expected_weight`` with the masked CE hidden
    gradient and BF16 ``target_weight`` with the returned RMSNorm input
    gradient.  In that mode ``grad_input_out`` must be omitted.  This removes a
    separate hidden-sized gradient scratch tensor; callers must not use either
    CE input after the call.

    The optimized path is specialized for contiguous CUDA tensors with the
    production mixed-precision dtypes.  Other devices, layouts and floating
    dtypes use a semantically equivalent ATen implementation.
    """

    if inplace_ce_buffers and grad_input_out is not None:
        raise ValueError(
            "grad_input_out must be omitted when inplace_ce_buffers=True; "
            "target_weight is used as the output buffer"
        )

    rows, hidden_size = _validate_inputs(
        norm_input,
        expected_weight,
        target_weight,
        raw_norm_weight,
        logsumexp,
        norm_gradient_accumulator,
        loss_mask,
        grad_input_out,
        inv_rms_out,
        losses_out,
        epsilon,
    )
    if inplace_ce_buffers:
        _validate_inplace_ce_buffers(
            norm_input,
            expected_weight,
            target_weight,
            raw_norm_weight,
            logsumexp,
            norm_gradient_accumulator,
            loss_mask,
            inv_rms_out,
            losses_out,
        )
        grad_input_out = target_weight
    elif grad_input_out is None:
        grad_input_out = torch.empty_like(norm_input)
    if inv_rms_out is None:
        inv_rms_out = torch.empty(rows, dtype=torch.float32, device=norm_input.device)
    if losses_out is None:
        losses_out = torch.empty(rows, dtype=torch.float32, device=norm_input.device)
    if rows == 0:
        return grad_input_out, losses_out

    if not _can_use_triton(
        norm_input,
        expected_weight,
        target_weight,
        raw_norm_weight,
        logsumexp,
        norm_gradient_accumulator,
        loss_mask,
        grad_input_out,
        inv_rms_out,
        losses_out,
    ):
        return _aten_fallback(
            norm_input,
            expected_weight,
            target_weight,
            raw_norm_weight,
            logsumexp,
            norm_gradient_accumulator,
            epsilon=epsilon,
            loss_mask=loss_mask,
            grad_input_out=grad_input_out,
            inv_rms_out=inv_rms_out,
            losses_out=losses_out,
            inplace_ce_buffers=inplace_ce_buffers,
        )

    has_mask = loss_mask is not None
    mask_argument = loss_mask if loss_mask is not None else norm_input
    block_hidden = triton.next_power_of_2(hidden_size)
    _online_mtp_ce_rmsnorm_dx_kernel[(rows,)](
        norm_input,
        expected_weight,
        target_weight,
        raw_norm_weight,
        logsumexp,
        mask_argument,
        grad_input_out,
        inv_rms_out,
        losses_out,
        HIDDEN_SIZE=hidden_size,
        EPSILON=epsilon,
        BLOCK_HIDDEN=block_hidden,
        HAS_MASK=has_mask,
        INPLACE_CE_BUFFERS=inplace_ce_buffers,
        num_warps=8,
        num_stages=1,
    )
    _online_mtp_rmsnorm_dweight_kernel[
        (triton.cdiv(hidden_size, _DWEIGHT_BLOCK_HIDDEN),)
    ](
        norm_input,
        expected_weight,
        target_weight,
        inv_rms_out,
        mask_argument,
        norm_gradient_accumulator,
        rows,
        HIDDEN_SIZE=hidden_size,
        BLOCK_HIDDEN=_DWEIGHT_BLOCK_HIDDEN,
        BLOCK_ROWS=_DWEIGHT_BLOCK_ROWS,
        HAS_MASK=has_mask,
        PRECOMPUTED_GRAD_HIDDEN=inplace_ce_buffers,
        num_warps=4,
        num_stages=1,
    )
    return grad_input_out, losses_out


@torch.no_grad()
def online_mtp_precomputed_ce_rmsnorm_backward(
    norm_input: torch.Tensor,
    grad_hidden: torch.Tensor,
    raw_norm_weight: torch.Tensor,
    logsumexp: torch.Tensor,
    target_logits: torch.Tensor,
    norm_gradient_accumulator: torch.Tensor,
    *,
    epsilon: float,
    loss_mask: Optional[torch.Tensor] = None,
    grad_input_out: Optional[torch.Tensor] = None,
    inv_rms_out: Optional[torch.Tensor] = None,
    losses_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backpropagate a TP-reduced CE hidden gradient through Gemma RMSNorm.

    ``grad_hidden`` is the already reduced FP32 value ``E_p[W] - W_target``;
    ``target_logits`` contains the matching reduced hard-label logits.  This
    specialized entry point avoids rebuilding a zero target matrix, rewriting
    logsumexp, and copying a temporary final-norm gradient into the persistent
    accumulator.  Every row still contributes to backward.  With a loss mask,
    the FP32 gradient is masked in place for the following dweight reduction.
    """

    if norm_input.ndim != 2:
        raise ValueError("norm_input must be a matrix")
    rows, hidden_size = norm_input.shape
    if grad_hidden.shape != norm_input.shape or grad_hidden.dtype != torch.float32:
        raise ValueError("grad_hidden must be an equal-shape FP32 matrix")
    if raw_norm_weight.shape != (hidden_size,):
        raise ValueError("raw_norm_weight shape does not match hidden size")
    if logsumexp.shape != (rows,) or logsumexp.dtype != torch.float32:
        raise ValueError("logsumexp must be an FP32 row vector")
    if target_logits.shape != (rows,) or target_logits.dtype != torch.float32:
        raise ValueError("target_logits must be an FP32 row vector")
    if norm_gradient_accumulator.shape != (hidden_size,):
        raise ValueError("norm gradient accumulator shape does not match hidden size")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    inputs = (
        grad_hidden,
        raw_norm_weight,
        logsumexp,
        target_logits,
        norm_gradient_accumulator,
    )
    if any(tensor.device != norm_input.device for tensor in inputs):
        raise ValueError("all precomputed CE inputs must share one device")
    if loss_mask is not None:
        if loss_mask.shape != (rows,) or loss_mask.dtype != torch.bool:
            raise ValueError("loss_mask must be a boolean row vector")
        if loss_mask.device != norm_input.device:
            raise ValueError("loss_mask must share the input device")

    if grad_input_out is None:
        grad_input_out = torch.empty_like(norm_input)
    _validate_output(
        "grad_input_out",
        grad_input_out,
        shape=(rows, hidden_size),
        dtype=norm_input.dtype,
        device=norm_input.device,
    )
    if _storage_ranges_overlap(
        grad_input_out, norm_input
    ) or _storage_ranges_overlap(grad_input_out, grad_hidden):
        raise ValueError("grad_input_out must not overlap live RMSNorm inputs")
    if inv_rms_out is None:
        inv_rms_out = torch.empty(rows, dtype=torch.float32, device=norm_input.device)
    if losses_out is None:
        losses_out = torch.empty(rows, dtype=torch.float32, device=norm_input.device)
    _validate_output(
        "inv_rms_out",
        inv_rms_out,
        shape=(rows,),
        dtype=torch.float32,
        device=norm_input.device,
    )
    _validate_output(
        "losses_out",
        losses_out,
        shape=(rows,),
        dtype=torch.float32,
        device=norm_input.device,
    )
    if rows == 0:
        return grad_input_out, losses_out

    can_use_triton = (
        norm_input.device.type == "cuda"
        and torch.version.hip is None
        and norm_input.dtype == torch.bfloat16
        and raw_norm_weight.dtype == torch.bfloat16
        and norm_gradient_accumulator.dtype == torch.bfloat16
        and hidden_size <= _MAX_TRITON_HIDDEN_SIZE
        and all(
            tensor.is_contiguous()
            for tensor in (
                norm_input,
                grad_hidden,
                raw_norm_weight,
                logsumexp,
                target_logits,
                norm_gradient_accumulator,
                grad_input_out,
                inv_rms_out,
                losses_out,
            )
        )
        and (loss_mask is None or loss_mask.is_contiguous())
    )
    if not can_use_triton:
        x = norm_input.float()
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon)
        inv_rms_out.copy_(inv_rms.reshape(-1))
        if loss_mask is None:
            row_scale = torch.ones_like(logsumexp)
        else:
            row_scale = loss_mask.float()
            grad_hidden.mul_(row_scale[:, None])
        losses_out.copy_(
            (logsumexp - target_logits).clamp_min(0.0) * row_scale
        )
        effective_weight = raw_norm_weight.float() + 1.0
        weighted_grad = grad_hidden * effective_weight
        projection = (weighted_grad * x).mean(dim=-1, keepdim=True)
        grad_input_out.copy_(
            inv_rms * weighted_grad - x * inv_rms.pow(3) * projection
        )
        grad_weight = (grad_hidden * x * inv_rms).sum(dim=0)
        norm_gradient_accumulator.add_(
            grad_weight.to(norm_gradient_accumulator.dtype)
        )
        return grad_input_out, losses_out

    has_mask = loss_mask is not None
    mask_argument = loss_mask if loss_mask is not None else norm_input
    block_hidden = triton.next_power_of_2(hidden_size)
    _online_mtp_precomputed_ce_rmsnorm_dx_kernel[(rows,)](
        norm_input,
        grad_hidden,
        raw_norm_weight,
        logsumexp,
        target_logits,
        mask_argument,
        grad_input_out,
        inv_rms_out,
        losses_out,
        HIDDEN_SIZE=hidden_size,
        EPSILON=epsilon,
        BLOCK_HIDDEN=block_hidden,
        HAS_MASK=has_mask,
        num_warps=8,
        num_stages=1,
    )
    _online_mtp_rmsnorm_dweight_kernel[
        (triton.cdiv(hidden_size, _DWEIGHT_BLOCK_HIDDEN),)
    ](
        norm_input,
        grad_hidden,
        norm_input,
        inv_rms_out,
        mask_argument,
        norm_gradient_accumulator,
        rows,
        HIDDEN_SIZE=hidden_size,
        BLOCK_HIDDEN=_DWEIGHT_BLOCK_HIDDEN,
        BLOCK_ROWS=_DWEIGHT_BLOCK_ROWS,
        HAS_MASK=has_mask,
        PRECOMPUTED_GRAD_HIDDEN=True,
        num_warps=4,
        num_stages=1,
    )
    return grad_input_out, losses_out


__all__ = [
    "online_mtp_ce_rmsnorm_backward",
    "online_mtp_precomputed_ce_rmsnorm_backward",
    "pack_local_ce_for_tp_reduce",
]
