# Morph: Inference That Learns

Morph is an experimental SGLang branch for continuous learning of the native
Qwen3.6 MTP draft model during serving. It captures activations from the actual
inference forward, attaches labels after target-model verification, computes an
exact in-engine backward, accumulates gradients, and publishes AdamW updates
without moving draft training into a separate framework.

The implementation is based on SGLang `release/v0.5.15`, with the Qwen3.6
compact linear-attention replay and exact FP32 recurrence fixes in this branch.
Normal serving is unchanged unless online MTP training is explicitly enabled.

## Current training slice

The current safe slice trains the dense SwiGLU MLP and final Gemma RMSNorm of
the native MTP layer. Input fusion, early normalization, draft attention, and
the frozen LM head remain frozen. Restricting updates to tensors downstream of
the persistent draft attention state avoids invalidating the serving cache.

The path uses:

- BF16 serving activations with bounded ticket-owned snapshots;
- compact hard cross entropy without materializing a token-by-vocabulary
  gradient;
- TP-global gradient clipping;
- FP32 Adam moments and master weights;
- graph-stable BF16 parameter publication;
- asynchronous backward and optimizer preparation overlapped with serving;
- every eligible draft token and batch, with no traffic subsampling.

## Code map

- `python/sglang/srt/speculative/online_mtp_training.py`: activation tickets,
  exact backward, gradient accumulation, optimizer state, and publication.
- `python/sglang/srt/speculative/triton_ops/online_mtp_*.py`: fused LSE/CE,
  RMSNorm, SwiGLU, and AdamW kernels.
- `python/sglang/srt/speculative/eagle_worker_v2.py`: serving scheduler and
  verification integration.
- `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`: CUDA Graph
  activation capture and lifetime handling.
- `python/sglang/srt/models/qwen3_5_mtp.py`: native-MTP forward hooks.
- `test/registered/spec/eagle/test_online_mtp_*.py`: kernel, optimizer,
  ownership, ordering, and parity tests.

## Current overhead result

On one 8xH100 node with Qwen3.6-27B, TP=8, native MTP draft depth 4,
concurrency 128, 256 fixed random 512-token prompts and 2,048-token greedy
outputs, three counterbalanced independent rounds measured:

| Configuration | Output throughput | Paired change vs. no backward |
| --- | ---: | ---: |
| No backward | 9,456.41 +/- 13.07 token/s | baseline |
| Full backward and optimizer, LR=0 | 9,353.33 +/- 63.40 token/s | -1.09% +/- 0.55% |
| Continuous learning, LR=1e-5 | 9,356.29 +/- 95.92 token/s | -1.06% +/- 1.13% |

The nonzero-learning-rate arm performed real parameter updates. Its paired
difference from the LR=0 arm was +0.04% +/- 1.69%, so the experiment does not
resolve any additional optimizer cost beyond the approximately 1.1% online
backward-path overhead.

## Status

This is research code, not an upstream-supported SGLang feature. The current
implementation and tests are intentionally kept in a reviewable feature branch
while the learning objective, trainable parameter slice, and long-horizon
quality behavior are evaluated.
