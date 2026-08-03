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

"""Fused SwiGLU backward for native-MTP online learning.

The manual online-training path only needs the SwiGLU input gradient in order
to form the gate/up weight gradient.  Expressing it with ordinary ATen ops
launches separate multiply, SiLU-backward, SiLU, multiply, and concatenate
kernels.  This module performs the same BF16 operations in one Triton launch.

The explicit BF16 casts below are intentional.  They preserve the rounding
boundaries of :func:`torch.ops.aten.silu_backward` and the surrounding BF16
multiplications used by the reference mixed-precision backward.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _online_mtp_swiglu_backward_kernel(
    grad_output,
    gate_up,
    output,
    num_elements,
    INTERMEDIATE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    element = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element < num_elements

    row = element // INTERMEDIATE_SIZE
    column = element - row * INTERMEDIATE_SIZE
    gate_offset = row * (2 * INTERMEDIATE_SIZE) + column
    up_offset = gate_offset + INTERMEDIATE_SIZE

    gate = tl.load(gate_up + gate_offset, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up + up_offset, mask=mask, other=0.0)
    grad = tl.load(grad_output + element, mask=mask, other=0.0)

    # Match ``grad * up`` in the reference BF16 path before silu_backward.
    grad_times_up = (grad * up).to(tl.bfloat16)
    sigmoid = 1.0 / (1.0 + libdevice.exp(-gate))

    # F.silu(gate) returns BF16 before the following multiplication.
    silu_gate = (gate * sigmoid).to(tl.bfloat16)
    grad_up = (grad * silu_gate).to(tl.bfloat16)

    # aten.silu_backward consumes the already-rounded BF16 grad_times_up and
    # evaluates sigmoid(x) * (1 + x * (1 - sigmoid(x))) in FP32 internally.
    grad_gate = (
        grad_times_up.to(tl.float32) * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    ).to(tl.bfloat16)

    tl.store(output + gate_offset, grad_gate, mask=mask)
    tl.store(output + up_offset, grad_up, mask=mask)


def _reference_swiglu_backward(
    grad_output: torch.Tensor,
    gate_up: torch.Tensor,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """ATen fallback with the same mixed-precision rounding semantics."""

    gate, up = gate_up.chunk(2, dim=-1)
    grad = grad_output.to(gate_up.dtype)
    grad_gate = torch.ops.aten.silu_backward(grad * up, gate)
    grad_up = grad * torch.nn.functional.silu(gate)
    result = torch.cat((grad_gate, grad_up), dim=-1)
    if out is None:
        return result
    out.copy_(result)
    return out


def _validate_inputs(
    grad_output: torch.Tensor,
    gate_up: torch.Tensor,
    out: Optional[torch.Tensor],
) -> int:
    if gate_up.ndim == 0 or gate_up.shape[-1] == 0:
        raise ValueError("gate_up must have a nonempty final dimension")
    if gate_up.shape[-1] % 2:
        raise ValueError(
            "gate_up final dimension must contain equal gate and up halves"
        )
    intermediate_size = gate_up.shape[-1] // 2
    expected_shape = (*gate_up.shape[:-1], intermediate_size)
    if tuple(grad_output.shape) != expected_shape:
        raise ValueError(
            "grad_output shape must equal gate_up.shape with the final "
            f"dimension halved; got grad_output={tuple(grad_output.shape)}, "
            f"gate_up={tuple(gate_up.shape)}"
        )
    if grad_output.device != gate_up.device:
        raise ValueError("grad_output and gate_up must be on the same device")
    if not gate_up.is_floating_point() or not grad_output.is_floating_point():
        raise ValueError("grad_output and gate_up must be floating-point tensors")
    if out is not None:
        if out.shape != gate_up.shape:
            raise ValueError("out must have the same shape as gate_up")
        if out.dtype != gate_up.dtype:
            raise ValueError("out must have the same dtype as gate_up")
        if out.device != gate_up.device:
            raise ValueError("out must be on the same device as gate_up")
    return intermediate_size


@torch.no_grad()
def online_mtp_swiglu_backward(
    grad_output: torch.Tensor,
    gate_up: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """Return the mixed-precision SwiGLU input gradient.

    Args:
        grad_output: Gradient of ``silu(gate) * up``, with shape
            ``[..., intermediate_size]``.
        gate_up: Concatenated gate/up input with shape
            ``[..., 2 * intermediate_size]``.
        out: Optional destination with the same shape, dtype, and device as
            ``gate_up``.  Passing ``out=gate_up`` is safe.
        inplace: Convenience form of ``out=gate_up``.  It cannot be combined
            with a distinct ``out`` tensor.

    The optimized path requires contiguous CUDA BF16 tensors.  Other devices,
    dtypes, and layouts use the ATen reference implementation so the public
    operation retains the same semantics outside the serving fast path.
    """

    if inplace:
        if out is not None and out is not gate_up:
            raise ValueError("inplace=True cannot be combined with a distinct out")
        out = gate_up

    intermediate_size = _validate_inputs(grad_output, gate_up, out)
    use_triton = (
        gate_up.device.type == "cuda"
        and gate_up.dtype == torch.bfloat16
        and grad_output.dtype == torch.bfloat16
        and gate_up.is_contiguous()
        and grad_output.is_contiguous()
        and (out is None or out.is_contiguous())
    )
    if not use_triton:
        return _reference_swiglu_backward(grad_output, gate_up, out)

    if out is None:
        out = torch.empty_like(gate_up)
    num_elements = grad_output.numel()
    if num_elements == 0:
        return out

    block_size = 256
    _online_mtp_swiglu_backward_kernel[(triton.cdiv(num_elements, block_size),)](
        grad_output,
        gate_up,
        out,
        num_elements,
        INTERMEDIATE_SIZE=intermediate_size,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return out
