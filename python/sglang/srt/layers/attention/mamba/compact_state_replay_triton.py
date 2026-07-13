import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fused_linear_compact_state_replay_with_mask_kernel(
    dst_ptr,
    k_norm_ptr,
    delta_v_ptr,
    decay_ptr,
    base_indices_raw_ptr,
    dst_indices_raw_ptr,
    step_indices_raw_ptr,
    total_requests,
    src_req_size,
    src_step_size,
    dst_req_size,
    NUM_HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_head_v = tl.program_id(2).to(tl.int64)
    pid_head = pid_head_v // NV
    pid_v = pid_head_v % NV

    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)
    if step_idx < 0:
        return

    base_idx = tl.load(base_indices_raw_ptr + pid_req).to(tl.int64)
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)
    if not (
        (base_idx >= 0)
        & (base_idx < dst_req_size)
        & (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (pid_req < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    dst_base = (
        ((pid_layer * dst_req_size + base_idx) * NUM_HEADS + pid_head) * V * K
        + offs_v[None, :] * K
        + offs_k[:, None]
    )
    b_h = tl.load(dst_ptr + dst_base, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(0, src_step_size):
        if t <= step_idx:
            k_norm_base = (
                ((pid_layer * src_req_size + pid_req) * src_step_size + t) * NUM_HEADS
                + pid_head
            ) * K + offs_k
            delta_v_base = (
                ((pid_layer * src_req_size + pid_req) * src_step_size + t) * NUM_HEADS
                + pid_head
            ) * V + offs_v

            b_k_norm = tl.load(k_norm_ptr + k_norm_base, mask=mask_k, other=0.0).to(
                tl.float32
            )
            b_delta_v = tl.load(delta_v_ptr + delta_v_base, mask=mask_v, other=0.0).to(
                tl.float32
            )
            b_decay = tl.load(decay_ptr + k_norm_base, mask=mask_k, other=1.0).to(
                tl.float32
            )

            b_h = _mul_rn_f32(b_h, b_decay[:, None])
            b_h = _fma_rn_f32(b_k_norm[:, None], b_delta_v[None, :], b_h)

    dst_out = (
        ((pid_layer * dst_req_size + dst_idx) * NUM_HEADS + pid_head) * V * K
        + offs_v[None, :] * K
        + offs_k[:, None]
    )
    dst_store_ptr = dst_ptr + dst_out
    tl.store(dst_store_ptr, b_h.to(dst_store_ptr.dtype.element_ty), mask=mask_h)


@triton.jit
def _fused_linear_compact_state_replay_with_track_kernel(
    dst_ptr,
    k_norm_ptr,
    delta_v_ptr,
    decay_ptr,
    base_indices_raw_ptr,
    accepted_dst_indices_raw_ptr,
    accepted_step_indices_raw_ptr,
    track_dst_indices_raw_ptr,
    track_step_indices_raw_ptr,
    total_requests,
    src_req_size,
    src_step_size,
    dst_req_size,
    NUM_HEADS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_head_v = tl.program_id(2).to(tl.int64)
    pid_head = pid_head_v // NV
    pid_v = pid_head_v % NV

    accepted_step = tl.load(accepted_step_indices_raw_ptr + pid_req).to(tl.int64)
    track_step = tl.load(track_step_indices_raw_ptr + pid_req).to(tl.int64)
    accepted_dst_idx = tl.load(accepted_dst_indices_raw_ptr + pid_req).to(tl.int64)
    track_dst_idx = tl.load(track_dst_indices_raw_ptr + pid_req).to(tl.int64)

    accepted_valid = (
        (accepted_step >= 0)
        & (accepted_step < src_step_size)
        & (accepted_dst_idx >= 0)
        & (accepted_dst_idx < dst_req_size)
    )
    track_valid = (
        (track_step >= 0)
        & (track_step <= accepted_step)
        & (track_step < src_step_size)
        & (track_dst_idx >= 0)
        & (track_dst_idx < dst_req_size)
    )
    if not accepted_valid:
        return

    base_idx = tl.load(base_indices_raw_ptr + pid_req).to(tl.int64)
    if not ((base_idx >= 0) & (base_idx < dst_req_size) & (pid_req < src_req_size)):
        return

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    dst_base = (
        ((pid_layer * dst_req_size + base_idx) * NUM_HEADS + pid_head) * V * K
        + offs_v[None, :] * K
        + offs_k[:, None]
    )
    b_h = tl.load(dst_ptr + dst_base, mask=mask_h, other=0.0).to(tl.float32)

    if track_valid:
        for t in tl.range(0, track_step + 1):
            k_norm_base = (
                ((pid_layer * src_req_size + pid_req) * src_step_size + t) * NUM_HEADS
                + pid_head
            ) * K + offs_k
            delta_v_base = (
                ((pid_layer * src_req_size + pid_req) * src_step_size + t) * NUM_HEADS
                + pid_head
            ) * V + offs_v

            b_k_norm = tl.load(k_norm_ptr + k_norm_base, mask=mask_k, other=0.0).to(
                tl.float32
            )
            b_delta_v = tl.load(delta_v_ptr + delta_v_base, mask=mask_v, other=0.0).to(
                tl.float32
            )
            b_decay = tl.load(decay_ptr + k_norm_base, mask=mask_k, other=1.0).to(
                tl.float32
            )

            b_h = _mul_rn_f32(b_h, b_decay[:, None])
            b_h = _fma_rn_f32(b_k_norm[:, None], b_delta_v[None, :], b_h)

        track_out = (
            ((pid_layer * dst_req_size + track_dst_idx) * NUM_HEADS + pid_head) * V * K
            + offs_v[None, :] * K
            + offs_k[:, None]
        )
        track_store_ptr = dst_ptr + track_out
        tl.store(
            track_store_ptr,
            b_h.to(track_store_ptr.dtype.element_ty),
            mask=mask_h,
        )

        for t in tl.range(track_step + 1, accepted_step + 1):
            k_norm_base = (
                ((pid_layer * src_req_size + pid_req) * src_step_size + t) * NUM_HEADS
                + pid_head
            ) * K + offs_k
            delta_v_base = (
                ((pid_layer * src_req_size + pid_req) * src_step_size + t) * NUM_HEADS
                + pid_head
            ) * V + offs_v

            b_k_norm = tl.load(k_norm_ptr + k_norm_base, mask=mask_k, other=0.0).to(
                tl.float32
            )
            b_delta_v = tl.load(delta_v_ptr + delta_v_base, mask=mask_v, other=0.0).to(
                tl.float32
            )
            b_decay = tl.load(decay_ptr + k_norm_base, mask=mask_k, other=1.0).to(
                tl.float32
            )

            b_h = _mul_rn_f32(b_h, b_decay[:, None])
            b_h = _fma_rn_f32(b_k_norm[:, None], b_delta_v[None, :], b_h)
    else:
        for t in tl.range(0, accepted_step + 1):
            k_norm_base = (
                ((pid_layer * src_req_size + pid_req) * src_step_size + t) * NUM_HEADS
                + pid_head
            ) * K + offs_k
            delta_v_base = (
                ((pid_layer * src_req_size + pid_req) * src_step_size + t) * NUM_HEADS
                + pid_head
            ) * V + offs_v

            b_k_norm = tl.load(k_norm_ptr + k_norm_base, mask=mask_k, other=0.0).to(
                tl.float32
            )
            b_delta_v = tl.load(delta_v_ptr + delta_v_base, mask=mask_v, other=0.0).to(
                tl.float32
            )
            b_decay = tl.load(decay_ptr + k_norm_base, mask=mask_k, other=1.0).to(
                tl.float32
            )

            b_h = _mul_rn_f32(b_h, b_decay[:, None])
            b_h = _fma_rn_f32(b_k_norm[:, None], b_delta_v[None, :], b_h)

    accepted_out = (
        ((pid_layer * dst_req_size + accepted_dst_idx) * NUM_HEADS + pid_head) * V * K
        + offs_v[None, :] * K
        + offs_k[:, None]
    )
    accepted_store_ptr = dst_ptr + accepted_out
    tl.store(
        accepted_store_ptr,
        b_h.to(accepted_store_ptr.dtype.element_ty),
        mask=mask_h,
    )


def fused_linear_compact_state_replay_with_mask(
    dst: torch.Tensor,  # [num_layers, cache_size, num_heads, V, K]
    k_norm: torch.Tensor,  # [num_layers, spec_req_slots, draft_tokens, num_heads, K]
    delta_v: torch.Tensor,  # [num_layers, spec_req_slots, draft_tokens, num_heads, V]
    decay: torch.Tensor,  # exp(gate), same shape as k_norm
    base_indices_raw: torch.Tensor,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
):
    # KDA/GDN compact replay: the generic compact K/V cache stores normalized K
    # and post-beta delta V before entering this kernel.
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
        return

    if not all(t.is_cuda for t in (dst, k_norm, delta_v, decay)):
        raise ValueError(
            "fused_linear_compact_state_replay_with_mask only supports CUDA tensors."
        )
    if dst.ndim != 5 or k_norm.ndim != 5 or delta_v.ndim != 5 or decay.ndim != 5:
        raise ValueError(
            f"Unexpected ranks: {dst.ndim=} {k_norm.ndim=} {delta_v.ndim=} {decay.ndim=}"
        )
    if dst.shape[0] != k_norm.shape[0] or dst.shape[0] != delta_v.shape[0]:
        raise ValueError("Layer dimension mismatch in compact replay buffers.")
    if k_norm.shape != decay.shape or k_norm.shape[:3] != delta_v.shape[:3]:
        raise ValueError("Compact replay buffer prefix shapes do not match.")
    if dst.shape[2] != k_norm.shape[3] or dst.shape[2] != delta_v.shape[3]:
        raise ValueError("Head dimension mismatch in compact replay buffers.")
    if dst.shape[3] != delta_v.shape[4] or dst.shape[4] != k_norm.shape[4]:
        raise ValueError("k_norm/delta_v dimension mismatch in compact replay buffers.")

    for name, tensor in (
        ("dst", dst),
        ("k_norm", k_norm),
        ("delta_v", delta_v),
        ("decay", decay),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} tensor must be contiguous")

    base_indices_raw = base_indices_raw.to(torch.int32).contiguous()
    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()

    num_layers = dst.shape[0]
    src_req_size = k_norm.shape[1]
    src_step_size = k_norm.shape[2]
    dst_req_size = dst.shape[1]
    num_heads = dst.shape[2]
    V = dst.shape[3]
    K = dst.shape[4]
    BK = triton.next_power_of_2(K)
    # Match the target-verify recurrence tile exactly. Besides making the
    # comparison controlled, this avoids changing FP32 instruction scheduling
    # merely because the accepted state is reconstructed in another kernel.
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    num_warps = 1
    grid = (total_requests, num_layers, num_heads * NV)

    _fused_linear_compact_state_replay_with_mask_kernel[grid](
        dst,
        k_norm,
        delta_v,
        decay,
        base_indices_raw,
        dst_indices_raw,
        step_indices_raw,
        total_requests,
        src_req_size,
        src_step_size,
        dst_req_size,
        NUM_HEADS=num_heads,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NV=NV,
        num_warps=num_warps,
        num_stages=3,
    )


def fused_linear_compact_state_replay_with_optional_track(
    dst: torch.Tensor,  # [num_layers, cache_size, num_heads, V, K]
    k_norm: torch.Tensor,  # [num_layers, spec_req_slots, draft_tokens, num_heads, K]
    delta_v: torch.Tensor,  # [num_layers, spec_req_slots, draft_tokens, num_heads, V]
    decay: torch.Tensor,  # exp(gate), same shape as k_norm
    base_indices_raw: torch.Tensor,
    accepted_dst_indices_raw: torch.Tensor,
    accepted_step_indices_raw: torch.Tensor,
    track_dst_indices_raw: torch.Tensor | None = None,
    track_step_indices_raw: torch.Tensor | None = None,
):
    # Current compact spec replay is restricted to topk=1, so every valid
    # prefix-cache track step is on the same linear path and must be <= the
    # accepted step. Tree/topk>1 support needs path-aware replay instead.
    # Reuse the single-destination kernel for the common case without a
    # prefix-cache track snapshot.
    if track_dst_indices_raw is None:
        assert track_step_indices_raw is None
        fused_linear_compact_state_replay_with_mask(
            dst,
            k_norm,
            delta_v,
            decay,
            base_indices_raw,
            accepted_dst_indices_raw,
            accepted_step_indices_raw,
        )
        return
    assert track_step_indices_raw is not None

    total_requests = accepted_step_indices_raw.shape[0]
    if total_requests == 0:
        return

    base_indices_raw = base_indices_raw.to(torch.int32).contiguous()
    accepted_dst_indices_raw = accepted_dst_indices_raw.to(torch.int32).contiguous()
    accepted_step_indices_raw = accepted_step_indices_raw.to(torch.int32).contiguous()
    track_dst_indices_raw = track_dst_indices_raw.to(torch.int32).contiguous()
    track_step_indices_raw = track_step_indices_raw.to(torch.int32).contiguous()

    num_layers = dst.shape[0]
    src_req_size = k_norm.shape[1]
    src_step_size = k_norm.shape[2]
    dst_req_size = dst.shape[1]
    num_heads = dst.shape[2]
    V = dst.shape[3]
    K = dst.shape[4]
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    num_warps = 1
    grid = (total_requests, num_layers, num_heads * NV)

    _fused_linear_compact_state_replay_with_track_kernel[grid](
        dst,
        k_norm,
        delta_v,
        decay,
        base_indices_raw,
        accepted_dst_indices_raw,
        accepted_step_indices_raw,
        track_dst_indices_raw,
        track_step_indices_raw,
        total_requests,
        src_req_size,
        src_step_size,
        dst_req_size,
        NUM_HEADS=num_heads,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NV=NV,
        num_warps=num_warps,
        num_stages=3,
    )
