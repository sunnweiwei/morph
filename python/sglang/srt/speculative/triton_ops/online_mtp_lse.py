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

"""Specialized full-vocabulary CE producers for online native-MTP training.

SGLang places sampled logits in an FP32 ``[tokens, vocabulary]`` graph buffer.
The two-stage LSE reduction keeps only two FP32 scalars per vocabulary chunk;
the probability producer fuses subtract, exponential, and the BF16 GEMM-input
cast.  Both support BF16 logits as well, but FP32 is the production path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _online_mtp_lse_partial_kernel(
    logits,
    partial_max,
    partial_sum,
    vocab_size: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    columns = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    values = tl.load(
        logits + row * vocab_size + columns,
        mask=columns < vocab_size,
        other=-float("inf"),
    ).to(tl.float32)
    block_max = tl.max(values, axis=0)
    block_sum = tl.sum(libdevice.exp(values - block_max), axis=0)
    output_offset = row * num_chunks + chunk
    tl.store(partial_max + output_offset, block_max)
    tl.store(partial_sum + output_offset, block_sum)


@triton.jit
def _online_mtp_lse_argmax_partial_kernel(
    logits,
    partial_max,
    partial_sum,
    partial_argmax,
    vocab_size: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Compute the ordinary LSE partials and greedy index in one read."""

    row = tl.program_id(0)
    chunk = tl.program_id(1)
    columns = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    values = tl.load(
        logits + row * vocab_size + columns,
        mask=columns < vocab_size,
        other=-float("inf"),
    ).to(tl.float32)
    block_max = tl.max(values, axis=0)
    block_argmax = tl.argmax(values, axis=0, tie_break_left=True)
    block_sum = tl.sum(libdevice.exp(values - block_max), axis=0)
    output_offset = row * num_chunks + chunk
    tl.store(partial_max + output_offset, block_max)
    tl.store(partial_sum + output_offset, block_sum)
    tl.store(partial_argmax + output_offset, chunk * BLOCK_V + block_argmax)


@triton.jit
def _online_mtp_grouped3_lse_partial_kernel(
    logits_0,
    logits_1,
    logits_2,
    partial_max,
    partial_sum,
    vocab_size: tl.constexpr,
    rows_per_step: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """The ordinary row reduction with three draft-step inputs in one grid.

    Each output row uses exactly the same chunk width and arithmetic as
    ``_online_mtp_lse_partial_kernel``.  Grouping only removes two kernel
    submissions and exposes all three steps' CTAs to the GPU scheduler at
    once; it does not concatenate or round the FP32 inference logits.
    """

    grouped_row = tl.program_id(0)
    chunk = tl.program_id(1)
    step = grouped_row // rows_per_step
    row = grouped_row - step * rows_per_step
    columns = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    column_mask = columns < vocab_size
    values_0 = tl.load(
        logits_0 + row * vocab_size + columns,
        mask=column_mask & (step == 0),
        other=-float("inf"),
    ).to(tl.float32)
    values_1 = tl.load(
        logits_1 + row * vocab_size + columns,
        mask=column_mask & (step == 1),
        other=-float("inf"),
    ).to(tl.float32)
    values_2 = tl.load(
        logits_2 + row * vocab_size + columns,
        mask=column_mask & (step == 2),
        other=-float("inf"),
    ).to(tl.float32)
    values = tl.where(step == 0, values_0, tl.where(step == 1, values_1, values_2))
    block_max = tl.max(values, axis=0)
    block_sum = tl.sum(libdevice.exp(values - block_max), axis=0)
    output_offset = grouped_row * num_chunks + chunk
    tl.store(partial_max + output_offset, block_max)
    tl.store(partial_sum + output_offset, block_sum)


@triton.jit
def _online_mtp_lse_finalize_kernel(
    partial_max,
    partial_sum,
    output,
    num_chunks: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_C)
    mask = chunks < num_chunks
    maxima = tl.load(
        partial_max + row * num_chunks + chunks,
        mask=mask,
        other=-float("inf"),
    )
    sums = tl.load(partial_sum + row * num_chunks + chunks, mask=mask, other=0.0)
    global_max = tl.max(maxima, axis=0)
    global_sum = tl.sum(sums * libdevice.exp(maxima - global_max), axis=0)
    tl.store(output + row, global_max + libdevice.log(global_sum))


@triton.jit
def _online_mtp_lse_argmax_finalize_kernel(
    partial_max,
    partial_sum,
    partial_argmax,
    output,
    argmax_output,
    num_chunks: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_C)
    mask = chunks < num_chunks
    maxima = tl.load(
        partial_max + row * num_chunks + chunks,
        mask=mask,
        other=-float("inf"),
    )
    sums = tl.load(partial_sum + row * num_chunks + chunks, mask=mask, other=0.0)
    indices = tl.load(
        partial_argmax + row * num_chunks + chunks,
        mask=mask,
        other=0x7FFFFFFF,
    )
    global_max = tl.max(maxima, axis=0)
    global_sum = tl.sum(sums * libdevice.exp(maxima - global_max), axis=0)
    # Match torch.argmax's first-index tie break across vocabulary chunks.
    candidate = tl.where(mask & (maxima == global_max), indices, 0x7FFFFFFF)
    global_argmax = tl.min(candidate, axis=0)
    tl.store(output + row, global_max + libdevice.log(global_sum))
    tl.store(argmax_output + row, global_argmax)


@triton.jit
def _online_mtp_local_stats_finalize_kernel(
    partial_max,
    partial_sum,
    partial_argmax,
    packed_output,
    global_vocab_start: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Finalize one TP vocabulary shard directly into a transport packet."""

    row = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_C)
    mask = chunks < num_chunks
    maxima = tl.load(
        partial_max + row * num_chunks + chunks,
        mask=mask,
        other=-float("inf"),
    )
    sums = tl.load(partial_sum + row * num_chunks + chunks, mask=mask, other=0.0)
    indices = tl.load(
        partial_argmax + row * num_chunks + chunks,
        mask=mask,
        other=0x7FFFFFFF,
    )
    local_max = tl.max(maxima, axis=0)
    local_sum = tl.sum(sums * libdevice.exp(maxima - local_max), axis=0)
    candidate = tl.where(mask & (maxima == local_max), indices, 0x7FFFFFFF)
    global_argmax = tl.min(candidate, axis=0) + global_vocab_start
    output = packed_output + row * 4
    tl.store(output, local_max + libdevice.log(local_sum))
    tl.store(output + 1, local_max)
    # Qwen's vocabulary is below 2**24, so FP32 transports every token id
    # exactly. The fourth word pads each rank to one 128-bit packet.
    tl.store(output + 2, global_argmax.to(tl.float32))
    tl.store(output + 3, 0.0)


@triton.jit
def _online_mtp_global_stats_finalize_kernel(
    gathered_stats,
    output,
    argmax_output,
    tp_size: tl.constexpr,
    stats_stride: tl.constexpr,
    BLOCK_TP: tl.constexpr,
):
    """Finalize row-major rank packets after a last-dimension all-gather."""

    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK_TP)
    mask = ranks < tp_size
    offsets = row * tp_size * stats_stride + ranks * stats_stride
    local_lse = tl.load(
        gathered_stats + offsets, mask=mask, other=-float("inf")
    )
    local_max = tl.load(
        gathered_stats + offsets + 1, mask=mask, other=-float("inf")
    )
    token_ids = tl.load(
        gathered_stats + offsets + 2, mask=mask, other=float(0x7FFFFF)
    )
    max_lse = tl.max(local_lse, axis=0)
    global_lse = max_lse + libdevice.log(
        tl.sum(libdevice.exp(local_lse - max_lse), axis=0)
    )
    global_max = tl.max(local_max, axis=0)
    # Global token ids are monotonically ordered across ordinary vocabulary
    # shards, so minimum id exactly matches torch.argmax's first-index tie.
    candidate = tl.where(mask & (local_max == global_max), token_ids, 0x7FFFFF)
    global_argmax = tl.min(candidate, axis=0)
    tl.store(output + row, global_lse)
    tl.store(argmax_output + row, global_argmax.to(tl.int64))


@triton.jit
def _online_mtp_probability_kernel(
    logits,
    logsumexp,
    output,
    logits_stride,
    LOCAL_START: tl.constexpr,
    LOCAL_VOCAB: tl.constexpr,
    STORE_SCALE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    columns = block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = columns < LOCAL_VOCAB
    values = tl.load(
        logits + row * logits_stride + LOCAL_START + columns,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    normalizer = tl.load(logsumexp + row).to(tl.float32)
    probability = libdevice.exp(values - normalizer) * STORE_SCALE
    tl.store(output + row * LOCAL_VOCAB + columns, probability, mask=mask)


@triton.jit
def _online_mtp_grouped3_probability_kernel(
    logits_0,
    logits_1,
    logits_2,
    logsumexp,
    output,
    logits_stride,
    rows_per_step: tl.constexpr,
    LOCAL_START: tl.constexpr,
    LOCAL_VOCAB: tl.constexpr,
    STORE_SCALE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Produce three adjacent probability blocks with one kernel launch."""

    grouped_row = tl.program_id(0)
    block = tl.program_id(1)
    step = grouped_row // rows_per_step
    row = grouped_row - step * rows_per_step
    columns = block * BLOCK_V + tl.arange(0, BLOCK_V)
    column_mask = columns < LOCAL_VOCAB
    input_offsets = row * logits_stride + LOCAL_START + columns
    values_0 = tl.load(
        logits_0 + input_offsets,
        mask=column_mask & (step == 0),
        other=-float("inf"),
    ).to(tl.float32)
    values_1 = tl.load(
        logits_1 + input_offsets,
        mask=column_mask & (step == 1),
        other=-float("inf"),
    ).to(tl.float32)
    values_2 = tl.load(
        logits_2 + input_offsets,
        mask=column_mask & (step == 2),
        other=-float("inf"),
    ).to(tl.float32)
    values = tl.where(step == 0, values_0, tl.where(step == 1, values_1, values_2))
    normalizer = tl.load(logsumexp + grouped_row).to(tl.float32)
    probability = libdevice.exp(values - normalizer) * STORE_SCALE
    tl.store(
        output + grouped_row * LOCAL_VOCAB + columns,
        probability,
        mask=column_mask,
    )


def _validate_grouped3_logits(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[int, int]:
    if len(logits) != 3:
        raise ValueError("grouped online MTP CE requires exactly three draft steps")
    reference = logits[0]
    if reference.ndim != 2:
        raise ValueError("grouped online MTP logits must be matrices")
    if reference.shape[0] <= 0 or reference.shape[1] <= 0:
        raise ValueError("grouped online MTP logits must have nonzero dimensions")
    if any(
        value.shape != reference.shape
        or value.dtype != reference.dtype
        or value.device != reference.device
        or value.stride() != reference.stride()
        for value in logits[1:]
    ):
        raise ValueError("grouped online MTP logits have incompatible layouts")
    if not all(value.is_contiguous() for value in logits):
        raise ValueError("grouped online MTP logits must be contiguous")
    return reference.shape


def compact_vocab_logsumexp(logits: torch.Tensor) -> torch.Tensor:
    """Return FP32 row logsumexp without a full-vocabulary FP32 temporary."""

    if logits.ndim != 2:
        raise ValueError(f"online MTP logits must be a matrix, got {logits.shape}")
    if not logits.is_cuda:
        return torch.logsumexp(logits.float(), dim=-1)
    if not logits.is_contiguous():
        raise ValueError("online MTP compact logsumexp requires contiguous logits")
    rows, vocab_size = logits.shape
    if rows <= 0 or vocab_size <= 0:
        raise ValueError("online MTP logits must have nonzero dimensions")

    # FP32 production logits favor this width across the serving row range; it
    # also keeps one compiled graph kernel shape instead of a batch-size branch.
    block_v = 8192
    num_chunks = triton.cdiv(vocab_size, block_v)
    partial_max = torch.empty(
        (rows, num_chunks), dtype=torch.float32, device=logits.device
    )
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty(rows, dtype=torch.float32, device=logits.device)
    _online_mtp_lse_partial_kernel[(rows, num_chunks)](
        logits,
        partial_max,
        partial_sum,
        vocab_size=vocab_size,
        num_chunks=num_chunks,
        BLOCK_V=block_v,
        num_warps=8,
        num_stages=1,
    )
    _online_mtp_lse_finalize_kernel[(rows,)](
        partial_max,
        partial_sum,
        output,
        num_chunks=num_chunks,
        BLOCK_C=triton.next_power_of_2(num_chunks),
        num_warps=1,
        num_stages=1,
    )
    return output


def compact_vocab_logsumexp_argmax(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact FP32 LSE and greedy token while reading logits once."""

    if logits.ndim != 2:
        raise ValueError(f"online MTP logits must be a matrix, got {logits.shape}")
    if not logits.is_cuda:
        return torch.logsumexp(logits.float(), dim=-1), torch.argmax(logits, dim=-1)
    if not logits.is_contiguous():
        raise ValueError("online MTP compact logsumexp requires contiguous logits")
    rows, vocab_size = logits.shape
    if rows <= 0 or vocab_size <= 0:
        raise ValueError("online MTP logits must have nonzero dimensions")

    block_v = 8192
    num_chunks = triton.cdiv(vocab_size, block_v)
    partial_max = torch.empty(
        (rows, num_chunks), dtype=torch.float32, device=logits.device
    )
    partial_sum = torch.empty_like(partial_max)
    partial_argmax = torch.empty(
        (rows, num_chunks), dtype=torch.int32, device=logits.device
    )
    output = torch.empty(rows, dtype=torch.float32, device=logits.device)
    argmax_output = torch.empty(rows, dtype=torch.int64, device=logits.device)
    _online_mtp_lse_argmax_partial_kernel[(rows, num_chunks)](
        logits,
        partial_max,
        partial_sum,
        partial_argmax,
        vocab_size=vocab_size,
        num_chunks=num_chunks,
        BLOCK_V=block_v,
        num_warps=8,
        num_stages=1,
    )
    _online_mtp_lse_argmax_finalize_kernel[(rows,)](
        partial_max,
        partial_sum,
        partial_argmax,
        output,
        argmax_output,
        num_chunks=num_chunks,
        BLOCK_C=triton.next_power_of_2(num_chunks),
        num_warps=1,
        num_stages=1,
    )
    return output, argmax_output


def compact_local_vocab_stats(
    logits: torch.Tensor,
    *,
    global_vocab_start: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce one contiguous TP vocabulary shard to four FP32 row words.

    The packet layout is ``[local_lse, local_max, global_argmax, padding]``.
    Reinterpreting it as BF16 produces one aligned eight-element transport
    shard without rounding any FP32 statistic.
    """

    if logits.ndim != 2:
        raise ValueError(f"online MTP local logits must be a matrix, got {logits.shape}")
    if not logits.is_contiguous():
        raise ValueError("online MTP local logits must be contiguous")
    rows, vocab_size = logits.shape
    if rows <= 0 or vocab_size <= 0:
        raise ValueError("online MTP local logits must have nonzero dimensions")
    if global_vocab_start < 0 or global_vocab_start + vocab_size >= 2**24:
        raise ValueError("online MTP local vocabulary ids must be exactly FP32 representable")
    if out is None:
        out = torch.empty((rows, 4), dtype=torch.float32, device=logits.device)
    elif (
        out.shape != (rows, 4)
        or out.dtype != torch.float32
        or out.device != logits.device
        or not out.is_contiguous()
    ):
        raise ValueError("online MTP local statistics output has an invalid layout")
    if not logits.is_cuda:
        values = logits.float()
        local_max, local_argmax = torch.max(values, dim=-1)
        out[:, 0] = torch.logsumexp(values, dim=-1)
        out[:, 1] = local_max
        out[:, 2] = local_argmax.float() + global_vocab_start
        out[:, 3].zero_()
        return out

    block_v = 8192
    num_chunks = triton.cdiv(vocab_size, block_v)
    partial_max = torch.empty(
        (rows, num_chunks), dtype=torch.float32, device=logits.device
    )
    partial_sum = torch.empty_like(partial_max)
    partial_argmax = torch.empty(
        (rows, num_chunks), dtype=torch.int32, device=logits.device
    )
    _online_mtp_lse_argmax_partial_kernel[(rows, num_chunks)](
        logits,
        partial_max,
        partial_sum,
        partial_argmax,
        vocab_size=vocab_size,
        num_chunks=num_chunks,
        BLOCK_V=block_v,
        num_warps=8,
        num_stages=1,
    )
    _online_mtp_local_stats_finalize_kernel[(rows,)](
        partial_max,
        partial_sum,
        partial_argmax,
        out,
        global_vocab_start=global_vocab_start,
        num_chunks=num_chunks,
        BLOCK_C=triton.next_power_of_2(num_chunks),
        num_warps=1,
        num_stages=1,
    )
    return out


def compact_global_vocab_stats(
    gathered_stats: torch.Tensor,
    *,
    tp_size: int,
    out_lse: torch.Tensor | None = None,
    out_argmax: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finalize row-major rank packets into global LSE and greedy token."""

    if (
        gathered_stats.ndim != 2
        or gathered_stats.dtype != torch.float32
        or not gathered_stats.is_contiguous()
        or tp_size <= 0
        or gathered_stats.shape[1] != tp_size * 4
    ):
        raise ValueError("gathered online MTP statistics have an invalid layout")
    rows = gathered_stats.shape[0]
    if out_lse is not None and (
        out_lse.shape != (rows,)
        or out_lse.dtype != torch.float32
        or out_lse.device != gathered_stats.device
        or not out_lse.is_contiguous()
    ):
        raise ValueError("global online MTP LSE output has an invalid layout")
    if out_argmax is not None and (
        out_argmax.shape != (rows,)
        or out_argmax.dtype != torch.int64
        or out_argmax.device != gathered_stats.device
        or not out_argmax.is_contiguous()
    ):
        raise ValueError("global online MTP argmax output has an invalid layout")
    if not gathered_stats.is_cuda:
        packets = gathered_stats.view(rows, tp_size, 4)
        output = torch.logsumexp(packets[:, :, 0], dim=-1)
        global_max = packets[:, :, 1].max(dim=-1, keepdim=True).values
        candidates = torch.where(
            packets[:, :, 1] == global_max,
            packets[:, :, 2],
            torch.full_like(packets[:, :, 2], float(0x7FFFFF)),
        )
        argmax_output = candidates.min(dim=-1).values.long()
        if out_lse is not None:
            out_lse.copy_(output)
            output = out_lse
        if out_argmax is not None:
            out_argmax.copy_(argmax_output)
            argmax_output = out_argmax
        return output, argmax_output
    output = (
        out_lse
        if out_lse is not None
        else torch.empty(rows, dtype=torch.float32, device=gathered_stats.device)
    )
    argmax_output = (
        out_argmax
        if out_argmax is not None
        else torch.empty(rows, dtype=torch.int64, device=gathered_stats.device)
    )
    _online_mtp_global_stats_finalize_kernel[(rows,)](
        gathered_stats,
        output,
        argmax_output,
        tp_size=tp_size,
        stats_stride=4,
        BLOCK_TP=triton.next_power_of_2(tp_size),
        num_warps=1,
        num_stages=1,
    )
    return output, argmax_output


def compact_vocab_logsumexp_grouped3(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Return three step-major LSE vectors from one launch group.

    The returned layout is ``[step0 rows][step1 rows][step2 rows]``.  CUDA uses
    the same per-row partial and finalize reductions as
    :func:`compact_vocab_logsumexp`, making this an exact scheduling
    transformation rather than a different softmax implementation.
    """

    rows, vocab_size = _validate_grouped3_logits(logits)
    if not logits[0].is_cuda:
        return torch.cat(
            [torch.logsumexp(value.float(), dim=-1) for value in logits]
        )
    block_v = 8192
    num_chunks = triton.cdiv(vocab_size, block_v)
    grouped_rows = 3 * rows
    partial_max = torch.empty(
        (grouped_rows, num_chunks), dtype=torch.float32, device=logits[0].device
    )
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty(grouped_rows, dtype=torch.float32, device=logits[0].device)
    _online_mtp_grouped3_lse_partial_kernel[(grouped_rows, num_chunks)](
        logits[0],
        logits[1],
        logits[2],
        partial_max,
        partial_sum,
        vocab_size=vocab_size,
        rows_per_step=rows,
        num_chunks=num_chunks,
        BLOCK_V=block_v,
        num_warps=8,
        num_stages=1,
    )
    _online_mtp_lse_finalize_kernel[(grouped_rows,)](
        partial_max,
        partial_sum,
        output,
        num_chunks=num_chunks,
        BLOCK_C=triton.next_power_of_2(num_chunks),
        num_warps=1,
        num_stages=1,
    )
    return output


def compact_vocab_probability(
    logits: torch.Tensor,
    logsumexp: torch.Tensor,
    *,
    local_start: int,
    local_stop: int,
    output_dtype: torch.dtype,
    out: torch.Tensor | None = None,
    store_scale: float = 1.0,
) -> torch.Tensor:
    """Fuse FP32 subtract/exp plus the serving BF16 probability cast.

    ``out`` lets the grouped-CE path write each draft step directly into its
    final packed workspace.  Besides avoiding a later ``torch.cat``, retaining
    that workspace gives CUDA-graph replay a stable output address.
    """

    if logits.ndim != 2 or logsumexp.ndim != 1:
        raise ValueError("online MTP probability expects matrix logits and vector LSE")
    if logsumexp.shape[0] != logits.shape[0]:
        raise ValueError("online MTP probability logits/LSE row counts differ")
    if local_start < 0 or local_stop <= local_start or local_stop > logits.shape[1]:
        raise ValueError("online MTP local vocabulary range is invalid")
    if logits.device != logsumexp.device:
        raise ValueError("online MTP probability logits/LSE devices differ")
    rows = logits.shape[0]
    local_vocab = local_stop - local_start
    if out is not None:
        if out.shape != (rows, local_vocab):
            raise ValueError(
                "online MTP probability output has the wrong shape: "
                f"expected {(rows, local_vocab)}, got {tuple(out.shape)}"
            )
        if out.device != logits.device:
            raise ValueError("online MTP probability output is on the wrong device")
        if out.dtype != output_dtype:
            raise ValueError(
                "online MTP probability output has the wrong dtype: "
                f"expected {output_dtype}, got {out.dtype}"
            )
        if not out.is_contiguous():
            raise ValueError("online MTP probability output must be contiguous")
    if not logits.is_cuda:
        probability = torch.exp(
            logits[:, local_start:local_stop].float() - logsumexp.float()[:, None]
        ).mul_(store_scale).to(output_dtype)
        if out is None:
            return probability
        out.copy_(probability)
        return out
    if not logits.is_contiguous() or not logsumexp.is_contiguous():
        raise ValueError("online MTP compact probability requires contiguous inputs")

    output = (
        torch.empty((rows, local_vocab), dtype=output_dtype, device=logits.device)
        if out is None
        else out
    )
    block_v = 1024
    _online_mtp_probability_kernel[(rows, triton.cdiv(local_vocab, block_v))](
        logits,
        logsumexp,
        output,
        logits.stride(0),
        LOCAL_START=local_start,
        LOCAL_VOCAB=local_vocab,
        STORE_SCALE=store_scale,
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=1,
    )
    return output


def compact_vocab_probability_grouped3(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    logsumexp: torch.Tensor,
    *,
    local_start: int,
    local_stop: int,
    output_dtype: torch.dtype,
    out: torch.Tensor | None = None,
    store_scale: float = 1.0,
) -> torch.Tensor:
    """Produce three step-major local probability shards in one grid."""

    rows, vocab_size = _validate_grouped3_logits(logits)
    grouped_rows = 3 * rows
    if logsumexp.shape != (grouped_rows,) or logsumexp.device != logits[0].device:
        raise ValueError("grouped online MTP LSE has an incompatible layout")
    if local_start < 0 or local_stop <= local_start or local_stop > vocab_size:
        raise ValueError("grouped online MTP local vocabulary range is invalid")
    local_vocab = local_stop - local_start
    if out is not None and (
        out.shape != (grouped_rows, local_vocab)
        or out.dtype != output_dtype
        or out.device != logits[0].device
        or not out.is_contiguous()
    ):
        raise ValueError("grouped online MTP probability output is incompatible")
    if not logits[0].is_cuda:
        probability = torch.cat(
            [
                torch.exp(
                    value.float()
                    - logsumexp[step * rows : (step + 1) * rows, None].float()
                )[:, local_start:local_stop]
                for step, value in enumerate(logits)
            ]
        ).mul_(store_scale).to(output_dtype)
        if out is None:
            return probability
        out.copy_(probability)
        return out
    output = (
        torch.empty(
            (grouped_rows, local_vocab),
            dtype=output_dtype,
            device=logits[0].device,
        )
        if out is None
        else out
    )
    block_v = 1024
    _online_mtp_grouped3_probability_kernel[
        (grouped_rows, triton.cdiv(local_vocab, block_v))
    ](
        logits[0],
        logits[1],
        logits[2],
        logsumexp,
        output,
        logits[0].stride(0),
        rows_per_step=rows,
        LOCAL_START=local_start,
        LOCAL_VOCAB=local_vocab,
        STORE_SCALE=store_scale,
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=1,
    )
    return output
