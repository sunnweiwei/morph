"""Experimental primitives for in-engine online MTP training.

The serving path remains inference-only.  Learning-enabled draft forwards may
stage a bounded set of tensors in :class:`ActivationTicketRing`; after target
verification a manual backward path consumes the ticket and accumulates only
parameter gradients.  Optimizer updates are deliberately separate from
backward so all gradients in one accumulation window are computed at one
immutable weight version.

This module contains no scheduler policy and is inert unless explicitly used.
The small, pure-torch kernels are also the correctness oracle for future
Triton/CUTLASS implementations.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import torch

_ONLINE_MTP_BASE_BACKWARD_BATCH_TOKENS = 24
_ONLINE_MTP_FP8_MAX = 448.0


class OnlineMTPActivationTap:
    """Collect per-step tensors from the *actual* eager inference forward.

    The first implementation clones eagerly for unambiguous ownership.  A
    CUDA-graph-aware implementation can later replace these clones with
    preallocated ring slots without changing the ticket/backward contract.
    """

    _MLP_TENSORS = frozenset(
        {
            "mlp_input",
            "mlp_gate_up",
            "final_norm_input",
            "ce_logsumexp",
            "ce_expected_weight",
            "ce_local_expected_weight",
            "ce_probability",
            "ce_logits",
        }
    )

    def __init__(
        self,
        *,
        weight_version: int,
        warmup_only: bool = False,
        defer_ce: bool = False,
        clone_tensors: bool = True,
        recompute_gate_up: bool = False,
        use_triton_lse: bool = False,
        use_triton_expcast: bool = False,
        group_ce_projection: bool = False,
        group_ce_steps: bool = False,
        fuse_ce_argmax: bool = False,
        local_vocab_ce: bool = False,
        defer_grouped_ce_projection: bool = False,
        defer_grouped_ce_projection_min_rows: Optional[int] = None,
        defer_grouped_ce_projection_include_base: bool = False,
        fused_ce_reduction: bool = False,
        async_ce_producer: bool = False,
        fp8_ce_weight: Optional[torch.Tensor] = None,
        fp8_ce_weight_scale: Optional[torch.Tensor] = None,
        fp8_ce_probability_scale: Optional[torch.Tensor] = None,
        expected_ce_steps: Optional[int] = None,
        ce_probability_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        if expected_ce_steps is not None and expected_ce_steps <= 0:
            raise ValueError("expected_ce_steps must be positive when provided")
        if fused_ce_reduction and not group_ce_projection:
            raise ValueError("fused CE reduction requires grouped CE projection")
        if defer_grouped_ce_projection and not group_ce_projection:
            raise ValueError(
                "deferred grouped CE projection requires grouped CE projection"
            )
        if group_ce_steps and not group_ce_projection:
            raise ValueError("step-grouped CE requires grouped CE projection")
        if group_ce_steps and (not use_triton_lse or not use_triton_expcast):
            raise ValueError(
                "step-grouped CE requires the exact Triton LSE/probability kernels"
            )
        if group_ce_steps and async_ce_producer:
            raise ValueError("step-grouped CE cannot use the async logits producer")
        if group_ce_steps and expected_ce_steps != 3:
            raise ValueError("step-grouped CE currently requires exactly three steps")
        if fuse_ce_argmax:
            if not group_ce_projection or not use_triton_lse:
                raise ValueError(
                    "fused CE argmax requires grouped CE and the Triton LSE kernel"
                )
            if group_ce_steps:
                raise ValueError(
                    "fused CE argmax must produce each draft step before sampling"
                )
        if local_vocab_ce:
            if not fuse_ce_argmax or not use_triton_expcast:
                raise ValueError(
                    "local-vocabulary CE requires fused argmax and Triton probability"
                )
            if defer_ce or group_ce_steps or async_ce_producer:
                raise ValueError(
                    "local-vocabulary CE cannot retain or group full-vocabulary logits"
                )
        if defer_grouped_ce_projection and not fused_ce_reduction:
            raise ValueError(
                "deferred grouped CE projection requires fused CE reduction"
            )
        fp8_ce_values = (
            fp8_ce_weight,
            fp8_ce_weight_scale,
            fp8_ce_probability_scale,
        )
        if any(value is not None for value in fp8_ce_values) and not all(
            value is not None for value in fp8_ce_values
        ):
            raise ValueError("FP8 CE projection requires weight and both scales")
        if fp8_ce_weight is not None and not group_ce_projection:
            raise ValueError("FP8 CE projection requires grouped CE projection")
        if fp8_ce_weight is not None and defer_grouped_ce_projection:
            raise ValueError("FP8 CE projection cannot defer its grouped projection")
        if (
            defer_grouped_ce_projection_min_rows is not None
            and defer_grouped_ce_projection_min_rows <= 0
        ):
            raise ValueError(
                "deferred grouped CE projection row threshold must be positive"
            )
        if ce_probability_buffer is not None:
            if not group_ce_projection:
                raise ValueError(
                    "an external CE probability buffer requires grouped CE"
                )
            if ce_probability_buffer.ndim != 2:
                raise ValueError("external CE probability buffer must be a matrix")
            if not ce_probability_buffer.is_contiguous():
                raise ValueError("external CE probability buffer must be contiguous")
        self.weight_version = weight_version
        self.warmup_only = warmup_only
        self.defer_ce = defer_ce
        self.clone_tensors = clone_tensors
        self.recompute_gate_up = recompute_gate_up
        self.use_triton_lse = use_triton_lse
        self.use_triton_expcast = use_triton_expcast
        self.group_ce_projection = group_ce_projection
        self.group_ce_steps = group_ce_steps
        self.fuse_ce_argmax = fuse_ce_argmax
        self.local_vocab_ce = local_vocab_ce
        self.defer_grouped_ce_projection = defer_grouped_ce_projection
        self.defer_grouped_ce_projection_min_rows = (
            defer_grouped_ce_projection_min_rows
        )
        self.defer_grouped_ce_projection_include_base = (
            defer_grouped_ce_projection_include_base
        )
        self.deferred_grouped_ce_projection_active = False
        self.fused_ce_reduction = fused_ce_reduction
        self.async_ce_producer = async_ce_producer
        self.fp8_ce_weight = fp8_ce_weight
        self.fp8_ce_weight_scale = fp8_ce_weight_scale
        self.fp8_ce_probability_scale = fp8_ce_probability_scale
        self.expected_ce_steps = expected_ce_steps
        self.current_step: Optional[int] = None
        self.steps: Dict[int, Dict[str, torch.Tensor]] = {}
        self.step_rows: Dict[int, int] = {}
        self._ce_probabilities: Dict[int, torch.Tensor] = {}
        self._ce_grouped_logits: Dict[int, torch.Tensor] = {}
        self._ce_argmax: Dict[int, torch.Tensor] = {}
        self._ce_probability_ranges: Dict[int, Tuple[int, int]] = {}
        self._ce_probability_buffer = ce_probability_buffer
        self._ce_probability_buffer_is_external = ce_probability_buffer is not None
        self._ce_probability_used_rows = 0
        self.ready_event: Optional[torch.cuda.Event] = None

    def reset(self) -> None:
        """Forget references recorded by an earlier graph warmup/capture."""

        self.current_step = None
        self.steps.clear()
        self.step_rows.clear()
        self._ce_probabilities.clear()
        self._ce_grouped_logits.clear()
        self._ce_argmax.clear()
        self._ce_probability_ranges.clear()
        self._ce_probability_used_rows = 0
        self.deferred_grouped_ce_projection_active = False
        self.ready_event = None

    def begin_step(
        self, step: int, *, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> None:
        if step in self.steps:
            raise RuntimeError(f"online MTP step {step} was recorded twice")
        self.current_step = step
        self.steps[step] = {}
        self.step_rows[step] = input_ids.numel()

    def record(self, name: str, tensor: torch.Tensor) -> None:
        # The first online-training slice updates only the dense MLP and final
        # norm.  Ignore the broader instrumentation points so the hot path does
        # not retain tensors that this backward cannot consume yet.
        if name not in self._MLP_TENSORS:
            return
        if name == "mlp_gate_up" and self.recompute_gate_up:
            return
        if self.current_step is None:
            raise RuntimeError("begin_step must be called before recording tensors")
        step = self.steps[self.current_step]
        if name in step:
            raise RuntimeError(
                f"online MTP tensor {name!r} was recorded twice at step "
                f"{self.current_step}"
            )
        step[name] = tensor.detach().clone() if self.clone_tensors else tensor.detach()

    def record_if_absent(self, name: str, tensor: torch.Tensor) -> None:
        if self.current_step is None:
            raise RuntimeError("begin_step must be called before recording tensors")
        if name in self.steps[self.current_step]:
            return
        self.record(name, tensor)

    def reserve_ce_probability(
        self,
        rows: int,
        local_vocab: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Reserve the current step's rows in one grow-only packed workspace.

        The first reservation uses ``expected_ce_steps`` to allocate the whole
        common fixed-row draft loop at once.  ``reset`` deliberately retains
        this allocation, so graph warmup/capture/replay always sees the same
        addresses.  Irregular eager shapes can still grow; already-written
        rows are copied once and their views are rebound to the new owner.
        """

        if not self.group_ce_projection:
            raise RuntimeError("grouped CE probability reserved while disabled")
        if self.current_step is None:
            raise RuntimeError("begin_step must precede grouped CE reservation")
        if self.current_step in self._ce_probabilities:
            raise RuntimeError(
                f"online MTP CE probability for step {self.current_step} "
                "was reserved twice"
            )
        if rows <= 0 or local_vocab <= 0:
            raise ValueError("grouped CE probability dimensions must be positive")
        if self.step_rows[self.current_step] != rows:
            raise ValueError(
                f"online MTP CE step {self.current_step} reserved {rows} rows, "
                f"expected {self.step_rows[self.current_step]}"
            )

        required_rows = self._ce_probability_used_rows + rows
        buffer = self._ce_probability_buffer
        if buffer is not None and (
            buffer.shape[1] != local_vocab
            or buffer.dtype != dtype
            or buffer.device != device
        ):
            raise RuntimeError(
                "grouped CE probability workspace layout changed without a new tap"
            )
        if buffer is None or buffer.shape[0] < required_rows:
            if buffer is not None and self._ce_probability_buffer_is_external:
                raise RuntimeError(
                    "external grouped CE probability workspace capacity exceeded: "
                    f"required {required_rows} rows, capacity {buffer.shape[0]}"
                )
            if buffer is None:
                capacity_rows = required_rows
                if self.expected_ce_steps is not None:
                    capacity_rows = max(capacity_rows, self.expected_ce_steps * rows)
            else:
                capacity_rows = max(required_rows, 2 * buffer.shape[0])
            grown = torch.empty(
                (capacity_rows, local_vocab), dtype=dtype, device=device
            )
            if buffer is not None and self._ce_probability_used_rows:
                grown[: self._ce_probability_used_rows].copy_(
                    buffer[: self._ce_probability_used_rows]
                )
            self._ce_probability_buffer = grown
            buffer = grown
            for step_id, (start, stop) in self._ce_probability_ranges.items():
                self._ce_probabilities[step_id] = buffer[start:stop]

        start = self._ce_probability_used_rows
        stop = start + rows
        probability = buffer[start:stop]
        self._ce_probability_used_rows = stop
        self._ce_probability_ranges[self.current_step] = (start, stop)
        self._ce_probabilities[self.current_step] = probability
        return probability

    def record_ce_probability(self, probability: torch.Tensor) -> None:
        """Retain a short-lived local-vocabulary probability for grouped CE.

        The optimized producer reserves its destination first and calls this
        with that same view, making this method a validation-only no-op.  The
        copy fallback preserves callers that supply a standalone probability.
        """

        if not self.group_ce_projection:
            raise RuntimeError("grouped CE probability recorded while disabled")
        if self.current_step is None:
            raise RuntimeError("begin_step must precede grouped CE recording")
        if probability.ndim != 2:
            raise ValueError("grouped CE probability must be a matrix")
        target = self._ce_probabilities.get(self.current_step)
        if target is None:
            target = self.reserve_ce_probability(
                probability.shape[0],
                probability.shape[1],
                dtype=probability.dtype,
                device=probability.device,
            )
        elif (
            target.shape != probability.shape
            or target.dtype != probability.dtype
            or target.device != probability.device
        ):
            raise ValueError(
                "recorded grouped CE probability does not match reservation"
            )
        if target.data_ptr() != probability.data_ptr():
            target.copy_(probability.detach())

    def record_grouped_ce_logits(self, logits: torch.Tensor) -> None:
        """Retain one step's graph-local FP32 logits until grouped CE runs.

        ``finalize_grouped_ce`` executes before the draft graph ends, so no
        ticket retains this full-vocabulary tensor.  Keeping the three live
        graph references merely prevents allocator reuse between draft steps;
        it does not copy or snapshot logits into the activation ring.
        """

        if not self.group_ce_steps:
            raise RuntimeError("grouped CE logits recorded while disabled")
        if self.current_step is None:
            raise RuntimeError("begin_step must precede grouped CE logits")
        if self.current_step in self._ce_grouped_logits:
            raise RuntimeError(
                f"online MTP grouped CE logits for step {self.current_step} "
                "were recorded twice"
            )
        if logits.ndim != 2 or logits.shape[0] != self.step_rows[self.current_step]:
            raise ValueError("grouped CE logits do not match the current step")
        if not logits.is_contiguous():
            raise ValueError("grouped CE logits must be contiguous")
        self._ce_grouped_logits[self.current_step] = logits.detach()

    def record_ce_argmax(self, argmax: torch.Tensor) -> None:
        """Publish the greedy index already produced by the CE LSE scan."""

        if not self.fuse_ce_argmax:
            raise RuntimeError("fused CE argmax recorded while disabled")
        if self.current_step is None:
            raise RuntimeError("begin_step must precede fused CE argmax")
        if self.current_step in self._ce_argmax:
            raise RuntimeError(
                f"online MTP CE argmax for step {self.current_step} was recorded twice"
            )
        if (
            argmax.shape != (self.step_rows[self.current_step],)
            or argmax.dtype != torch.int64
            or not argmax.is_contiguous()
        ):
            raise ValueError("fused CE argmax has an incompatible layout")
        self._ce_argmax[self.current_step] = argmax.detach()

    def take_ce_argmax(self, step: int) -> torch.Tensor:
        """Consume one per-step greedy index inside the draft loop."""

        try:
            return self._ce_argmax.pop(step)
        except KeyError as exc:
            raise RuntimeError(f"online MTP CE argmax for step {step} is missing") from exc

    def finalize_grouped_ce(self, lm_head) -> None:
        """Finalize D4 probability ownership or project it with one GEMM."""

        if self.async_ce_producer:
            return
        if not self.group_ce_projection:
            return
        step_ids = sorted(self.steps)
        if not step_ids:
            return
        if set(step_ids) != set(self._ce_probabilities):
            raise RuntimeError(
                "grouped online MTP CE is missing step probabilities: "
                f"steps={step_ids}, probabilities={sorted(self._ce_probabilities)}"
            )

        if self.group_ce_steps:
            if set(step_ids) != set(self._ce_grouped_logits):
                raise RuntimeError(
                    "grouped online MTP CE is missing step logits: "
                    f"steps={step_ids}, logits={sorted(self._ce_grouped_logits)}"
                )
            if len(step_ids) != 3:
                raise RuntimeError("step-grouped online MTP CE requires three steps")
            from sglang.srt.speculative.triton_ops.online_mtp_lse import (
                compact_vocab_logsumexp_grouped3,
                compact_vocab_probability_grouped3,
            )

            grouped_logits = tuple(self._ce_grouped_logits[step] for step in step_ids)
            if len({value.shape[0] for value in grouped_logits}) != 1:
                raise RuntimeError("step-grouped CE requires equal rows per draft step")
            indices = lm_head.shard_indices
            grouped_lse = compact_vocab_logsumexp_grouped3(grouped_logits)
            if self._ce_probability_buffer is None:
                raise RuntimeError("grouped online MTP CE has no probability workspace")
            probability_dtype = (
                torch.float8_e4m3fn
                if self.fp8_ce_weight is not None
                else lm_head.weight.dtype
            )
            compact_vocab_probability_grouped3(
                grouped_logits,
                grouped_lse,
                local_start=indices.org_vocab_start_index,
                local_stop=indices.org_vocab_end_index,
                output_dtype=probability_dtype,
                out=self._ce_probability_buffer[: self._ce_probability_used_rows],
                store_scale=(448.0 if self.fp8_ce_weight is not None else 1.0),
            )
            offset = 0
            for step_id in step_ids:
                rows = self.step_rows[step_id]
                self.steps[step_id]["ce_logsumexp"] = grouped_lse[
                    offset : offset + rows
                ].detach()
                offset += rows
            if offset != grouped_lse.shape[0]:
                raise RuntimeError("step-grouped CE LSE row packing failed")
            self._ce_grouped_logits.clear()

        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        indices = lm_head.shard_indices
        local_weight = lm_head.weight[
            : indices.org_vocab_end_index - indices.org_vocab_start_index
        ]
        if self._ce_probability_buffer is None:
            raise RuntimeError("grouped online MTP CE has no probability workspace")
        packed_rows = 0
        for step_id in step_ids:
            start, stop = self._ce_probability_ranges[step_id]
            if start != packed_rows or stop - start != self.step_rows[step_id]:
                raise RuntimeError(
                    "grouped online MTP CE probability rows are not step-major"
                )
            packed_rows = stop
        if packed_rows != self._ce_probability_used_rows:
            raise RuntimeError("grouped online MTP CE workspace row count is invalid")
        probabilities = self._ce_probability_buffer[:packed_rows]
        # Deferral is useful only when several small scheduler tickets will be
        # combined into one backward group.  A ticket that already fills the
        # established 24-row launch-balanced group gets no GEMM amortization;
        # moving its projection out of the draft graph merely creates a later
        # serving-side burst (and regresses C>=8 in the production curve).
        should_defer_projection = (
            self.defer_grouped_ce_projection
            and (
                packed_rows < _ONLINE_MTP_BASE_BACKWARD_BATCH_TOKENS
                or (
                    self.defer_grouped_ce_projection_include_base
                    and packed_rows == _ONLINE_MTP_BASE_BACKWARD_BATCH_TOKENS
                )
            )
            and (
                self.defer_grouped_ce_projection_min_rows is None
                or packed_rows < self.defer_grouped_ce_projection_min_rows
            )
        )
        if should_defer_projection:
            self.deferred_grouped_ce_projection_active = True
            offset = 0
            for step_id in step_ids:
                rows = self.step_rows[step_id]
                self.steps[step_id]["ce_probability"] = probabilities[
                    offset : offset + rows
                ].detach()
                offset += rows
            if offset != probabilities.shape[0]:
                raise RuntimeError(
                    "grouped CE probability row count does not match draft steps"
                )
            self._ce_probabilities.clear()
            self._ce_probability_ranges.clear()
            self._ce_probability_used_rows = 0
            return
        # The opt-in fused reduction retains the BF16 local GEMM result.  Its
        # FP32 cast, target-row subtraction and sole TP reduction happen only
        # after verification supplies labels.  This preserves the existing
        # local GEMM rounding while removing this collective from the draft
        # CUDA graph and halving retained expected-vector bytes.
        if self.fp8_ce_weight is None:
            local_expected = probabilities @ local_weight
        else:
            assert self.fp8_ce_weight_scale is not None
            assert self.fp8_ce_probability_scale is not None
            if probabilities.dtype != torch.float8_e4m3fn:
                raise RuntimeError("FP8 CE projection received non-E4M3 probabilities")
            if self.fp8_ce_weight.shape != local_weight.shape:
                raise RuntimeError("FP8 CE projection weight shape changed")
            local_expected = torch._scaled_mm(
                probabilities,
                self.fp8_ce_weight,
                self.fp8_ce_probability_scale,
                self.fp8_ce_weight_scale,
                out_dtype=local_weight.dtype,
                use_fast_accum=True,
            )
        if self.fused_ce_reduction:
            expected = local_expected
            expected_name = "ce_local_expected_weight"
        else:
            expected = tensor_model_parallel_all_reduce(local_expected.float())
            expected_name = "ce_expected_weight"

        offset = 0
        for step_id in step_ids:
            rows = self.step_rows[step_id]
            # Adjacent views preserve one packed owner for eager execution and
            # let the graph snapshot copy all steps into one ticket allocation.
            self.steps[step_id][expected_name] = expected[
                offset : offset + rows
            ].detach()
            offset += rows
        if offset != expected.shape[0]:
            raise RuntimeError("grouped CE output row count does not match draft steps")
        self._ce_probabilities.clear()
        self._ce_probability_ranges.clear()
        self._ce_probability_used_rows = 0

    def validate_mlp_complete(self) -> None:
        if self._ce_probabilities:
            raise RuntimeError("grouped online MTP CE was not finalized")
        if self._ce_grouped_logits:
            raise RuntimeError("step-grouped online MTP CE was not finalized")
        if self._ce_argmax:
            raise RuntimeError("fused online MTP CE argmax was not consumed")
        required = {
            "mlp_input",
            "final_norm_input",
        }
        if not self.recompute_gate_up:
            required.add("mlp_gate_up")
        if self.async_ce_producer or self.defer_ce:
            required.add("ce_logits")
        else:
            required.add("ce_logsumexp")
            required.add(
                "ce_probability"
                if self.deferred_grouped_ce_projection_active
                else (
                    "ce_local_expected_weight"
                    if self.fused_ce_reduction
                    else "ce_expected_weight"
                )
            )
        for step_id, step in self.steps.items():
            missing = required - set(step)
            if missing:
                raise RuntimeError(
                    f"online MTP step {step_id} is missing activations: "
                    f"{sorted(missing)}; captured={sorted(step)}"
                )

    @property
    def num_tokens(self) -> int:
        return sum(self.step_rows.values())

    def flatten(self) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for step_id, step in sorted(self.steps.items()):
            for name, value in step.items():
                tensors[f"step_{step_id}.{name}"] = value
        return tensors

    def copy_from_graph_template(
        self,
        template: "OnlineMTPActivationTap",
        *,
        num_rows: int,
        copy_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Asynchronously snapshot one replay of a static CUDA-graph tap.

        CUDA graphs reuse their output addresses.  A learning replay therefore
        cannot hand those addresses directly to a delayed backward.  Instead,
        this method copies only the real (unpadded) rows into ticket-owned
        tensors before the serving stream can launch the next graph replay.

        All steps of one activation kind are packed into one allocation.  The
        per-step entries are adjacent views, allowing backward to consume the
        packed storage without a later ``torch.cat`` allocation.

        Before returning, a device-side wait is inserted on the serving stream.
        The production path queues the snapshot on the serving stream.  Graph
        replay, snapshot and the next graph replay are therefore ordered
        without per-ticket CUDA events.  Tensor kinds sharing a dtype use one
        slab allocation and one foreach-copy submission, reducing the standard
        D=4 snapshot from five allocations and fifteen individual copies to two
        allocations and two foreach submissions.

        ``copy_stream`` retains the older asynchronous form for diagnostics.
        It records a device-side dependency and makes the serving stream wait
        before a graph sharing the same memory pool can overwrite the source.
        """

        if self.steps:
            raise RuntimeError("graph snapshot destination tap is not empty")
        if num_rows <= 0:
            raise ValueError("graph snapshot must contain at least one row")
        self.deferred_grouped_ce_projection_active = (
            template.deferred_grouped_ce_projection_active
        )
        self.async_ce_producer = template.async_ce_producer
        source_steps = sorted(template.steps.items())
        expected_names = set(source_steps[0][1])
        for step_id, source_step in source_steps:
            if set(source_step) != expected_names:
                raise RuntimeError(
                    "graph tap tensor names differ between speculative steps"
                )
            self.steps[step_id] = {}
            self.step_rows[step_id] = num_rows

        def queue_snapshot() -> None:
            descriptions = []
            grouped_numel: Dict[Tuple[torch.dtype, torch.device], int] = {}
            for name in sorted(expected_names):
                sources = [source_step[name] for _, source_step in source_steps]
                reference = sources[0]
                if reference.ndim == 0 or reference.shape[0] < num_rows:
                    raise RuntimeError(
                        f"graph tap tensor name={name} has shape="
                        f"{tuple(reference.shape)}, cannot slice {num_rows} rows"
                    )
                for source in sources[1:]:
                    if (
                        source.ndim == 0
                        or source.shape[0] < num_rows
                        or source.shape[1:] != reference.shape[1:]
                        or source.dtype != reference.dtype
                        or source.device != reference.device
                    ):
                        raise RuntimeError(
                            f"graph tap tensor {name} has incompatible step shapes"
                        )
                packed_numel = (
                    len(sources)
                    * num_rows
                    * math.prod(reference.shape[1:])
                )
                key = (reference.dtype, reference.device)
                offset = grouped_numel.get(key, 0)
                grouped_numel[key] = offset + packed_numel
                descriptions.append(
                    (name, sources, reference, key, offset, packed_numel)
                )

            # One owner per dtype/device replaces one owner per activation kind.
            owners = {
                key: torch.empty(elements, dtype=key[0], device=key[1])
                for key, elements in grouped_numel.items()
            }
            copies: Dict[
                Tuple[torch.dtype, torch.device],
                Tuple[list[torch.Tensor], list[torch.Tensor]],
            ] = {}
            for name, sources, reference, key, offset, packed_numel in descriptions:
                packed = owners[key][offset : offset + packed_numel].view(
                    len(sources) * num_rows, *reference.shape[1:]
                )
                destinations, copy_sources = copies.setdefault(key, ([], []))
                for step_index, ((step_id, _), source) in enumerate(
                    zip(source_steps, sources)
                ):
                    start = step_index * num_rows
                    destination = packed[start : start + num_rows]
                    destinations.append(destination)
                    copy_sources.append(source[:num_rows])
                    self.steps[step_id][name] = destination

            for destinations, sources in copies.values():
                torch._foreach_copy_(destinations, sources)

        if copy_stream is None:
            queue_snapshot()
            self.ready_event = None
            return

        graph_done = torch.cuda.Event()
        graph_done.record(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(graph_done)
            queue_snapshot()
            ready = torch.cuda.Event()
            ready.record(copy_stream)
        self.ready_event = ready
        torch.cuda.current_stream().wait_event(ready)


class TicketState(enum.Enum):
    DRAFTED = "drafted"
    VERIFIED = "verified"
    BACKWARDED = "backwarded"
    RELEASED = "released"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def fused_ce_transient_reservation_bytes(
    *,
    num_tokens: int,
    hidden_size: int,
    activation_element_size: int,
) -> int:
    """Conservative per-ticket budget for the asynchronous fused-CE tail.

    Tickets can later be combined into one backward group.  Reserving each
    ticket's proportional share independently is conservative because the sum
    of individually 16-byte-aligned reduction buffers is never smaller than
    the aligned buffer for their combined rows.  The two hidden-sized copies
    cover worst-case non-adjacent ``norm_input`` and ``local_expected``
    concatenations.  The precomputed-gradient CE tail writes RMSNorm dweight
    directly into its persistent accumulator, so it has no per-group norm
    gradient allocation.

    A bool loss mask and one retained FP32 mean-loss scalar cover the remaining
    ticket-owned cross-stream outputs.  This function performs accounting only;
    none of these buffers are preallocated by admission.
    """

    if num_tokens <= 0 or hidden_size <= 0:
        raise ValueError("fused CE reservation dimensions must be positive")
    if activation_element_size <= 0:
        raise ValueError("fused CE reservation element size must be positive")

    logical_reduction_elements = num_tokens * (hidden_size + 1)
    # The custom all-reduce input/output is FP32 and padded to 16-byte size
    # alignment, equivalently four FP32 elements.
    reduced_bytes = 16 * ((logical_reduction_elements + 3) // 4)
    concatenated_hidden_bytes = (
        2 * num_tokens * hidden_size * activation_element_size
    )
    loss_mask_bytes = num_tokens  # torch.bool is one byte.
    retained_mean_loss_bytes = 4  # One FP32 scalar.
    return (
        reduced_bytes
        + concatenated_hidden_bytes
        + loss_mask_bytes
        + retained_mean_loss_bytes
    )


def concatenate_adjacent_rows(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Return a zero-copy view when row tensors occupy adjacent storage."""

    tensors = list(tensors)
    if not tensors:
        raise ValueError("cannot concatenate an empty tensor list")
    if len(tensors) == 1:
        return tensors[0]
    first = tensors[0]
    can_view = first.ndim > 0 and first.is_contiguous()
    expected_offset = first.storage_offset()
    storage_pointer = first.untyped_storage().data_ptr()
    for tensor in tensors:
        can_view = can_view and (
            tensor.ndim == first.ndim
            and tensor.shape[1:] == first.shape[1:]
            and tensor.dtype == first.dtype
            and tensor.device == first.device
            and tensor.is_contiguous()
            and tensor.stride() == first.stride()
            and tensor.untyped_storage().data_ptr() == storage_pointer
            and tensor.storage_offset() == expected_offset
        )
        expected_offset += tensor.numel()
    if can_view:
        return torch.as_strided(
            first,
            size=(sum(tensor.shape[0] for tensor in tensors), *first.shape[1:]),
            stride=first.stride(),
            storage_offset=first.storage_offset(),
        )
    return torch.cat(tensors)


@dataclass
class OnlineMTPActivationTicket:
    """One speculative block whose weights are pinned to ``weight_version``."""

    ticket_id: int
    weight_version: int
    num_tokens: int
    tensors: Dict[str, torch.Tensor]
    bytes_reserved: int
    state: TicketState = TicketState.DRAFTED
    labels: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    ready_event: Optional[torch.cuda.Event] = None
    ce_ready_event: Optional[torch.cuda.Event] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def mark_verified(
        self, labels: torch.Tensor, loss_mask: Optional[torch.Tensor] = None
    ) -> None:
        if self.state is not TicketState.DRAFTED:
            raise RuntimeError(
                f"ticket {self.ticket_id} cannot verify from {self.state.value}"
            )
        if labels.numel() != self.num_tokens:
            raise ValueError(
                f"ticket {self.ticket_id}: labels contain {labels.numel()} values, "
                f"expected {self.num_tokens}"
            )
        if loss_mask is not None and loss_mask.numel() != self.num_tokens:
            raise ValueError(
                f"ticket {self.ticket_id}: loss mask contains {loss_mask.numel()} "
                f"values, expected {self.num_tokens}"
            )
        self.labels = labels
        self.loss_mask = loss_mask
        self.state = TicketState.VERIFIED


@dataclass
class PendingBackward:
    """A local-only backward queued on the training CUDA stream."""

    ticket_ids: Tuple[int, ...]
    training_batch_id: int
    done_event: torch.cuda.Event
    mean_loss: torch.Tensor


@dataclass
class PendingUpdate:
    """AdamW work prepared off the serving stream and awaiting publication."""

    done_event: torch.cuda.Event
    grad_norm: float
    update_tokens: int


@dataclass
class PreparedCELocal:
    """Local, label-dependent CE work awaiting the ordered TP reduction."""

    packed: torch.Tensor
    norm_input: torch.Tensor
    local_expected: torch.Tensor
    rows: int
    hidden_size: int


@dataclass
class PreparedCEReduction:
    """Owned output of the label-dependent TP reduction.

    Packing and the collective run on the ordered serving stream.  The
    remaining fields are retained so the single training stream can consume
    the reduction without reconstructing hidden-sized concatenations.
    """

    reduced: torch.Tensor
    norm_input: torch.Tensor
    local_expected: torch.Tensor
    rows: int
    hidden_size: int


@dataclass
class PendingCEProjection:
    """A local CE projection overlapped with the following inference step."""

    tickets: Tuple[OnlineMTPActivationTicket, ...]
    combined: OnlineMTPActivationTicket
    training_batch_id: int
    ready_event: torch.cuda.Event
    local: PreparedCELocal


@dataclass
class PreparedCETail:
    """Local CE/RMSNorm outputs produced on the backward stream."""

    grad_mlp_output: torch.Tensor
    mean_loss: torch.Tensor
    norm_gradient: Optional[torch.Tensor]
    active_tokens: int


class ActivationTicketRing:
    """Bounded ticket-admission accounting used by the complete-data runtime.

    The budget covers ticket-owned snapshots and explicitly reserved async CE
    temporaries.  It is not a hard cap on total CUDA memory: shared grow-only
    workspaces, allocator/communicator state, optimizer state, and short-lived
    MLP concatenations remain outside it.  :class:`OnlineMTPRuntime` turns
    admission pressure into serving backpressure so no eligible training batch
    is dropped.
    """

    def __init__(self, *, max_bytes: int, max_tickets: int) -> None:
        if max_bytes <= 0 or max_tickets <= 0:
            raise ValueError("activation ring budgets must be positive")
        self.max_bytes = max_bytes
        self.max_tickets = max_tickets
        self.live_bytes = 0
        self._next_ticket_id = 0
        self._tickets: MutableMapping[int, OnlineMTPActivationTicket] = {}
        self._ready: deque[int] = deque()

    @property
    def live_tickets(self) -> int:
        return len(self._tickets)

    @property
    def ready_tickets(self) -> int:
        return len(self._ready)

    @property
    def ready_tokens(self) -> int:
        return sum(self._tickets[ticket_id].num_tokens for ticket_id in self._ready)

    def get(self, ticket_id: int) -> OnlineMTPActivationTicket:
        return self._tickets[ticket_id]

    def try_stage(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        weight_version: int,
        num_tokens: int,
        clone: bool = True,
        reserved_extra_bytes: int = 0,
        ready_event: Optional[torch.cuda.Event] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Optional[OnlineMTPActivationTicket]:
        """Stage a ticket without blocking; return ``None`` on budget pressure."""

        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if reserved_extra_bytes < 0:
            raise ValueError("reserved_extra_bytes must be non-negative")
        requested = sum(tensor_nbytes(value) for value in tensors.values())
        requested += reserved_extra_bytes
        if (
            requested > self.max_bytes - self.live_bytes
            or len(self._tickets) >= self.max_tickets
        ):
            return None

        staged = {
            name: value.detach().clone() if clone else value.detach()
            for name, value in tensors.items()
        }
        ticket = OnlineMTPActivationTicket(
            ticket_id=self._next_ticket_id,
            weight_version=weight_version,
            num_tokens=num_tokens,
            tensors=staged,
            bytes_reserved=requested,
            ready_event=ready_event,
            metadata=dict(metadata or {}),
        )
        self._next_ticket_id += 1
        self._tickets[ticket.ticket_id] = ticket
        self.live_bytes += requested
        return ticket

    def can_stage(self, *, bytes_reserved: int) -> bool:
        """Return whether one ticket fits without changing ring accounting."""

        if bytes_reserved < 0:
            raise ValueError("bytes_reserved must be non-negative")
        return (
            self.live_tickets < self.max_tickets
            and self.live_bytes + bytes_reserved <= self.max_bytes
        )

    def mark_verified(
        self,
        ticket_id: int,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> None:
        ticket = self._tickets[ticket_id]
        ticket.mark_verified(labels, loss_mask)
        self._ready.append(ticket_id)

    def pop_verified(self) -> Optional[OnlineMTPActivationTicket]:
        if not self._ready:
            return None
        return self._tickets[self._ready.popleft()]

    def pop_all_verified(self) -> list[OnlineMTPActivationTicket]:
        tickets = [self._tickets[ticket_id] for ticket_id in self._ready]
        self._ready.clear()
        return tickets

    def release(self, ticket_id: int, *, backwarded: bool = False) -> None:
        ticket = self._tickets.pop(ticket_id)
        if backwarded and ticket.state is not TicketState.VERIFIED:
            raise RuntimeError(
                f"ticket {ticket_id} cannot complete backward from {ticket.state.value}"
            )
        ticket.state = TicketState.BACKWARDED if backwarded else TicketState.RELEASED
        self.live_bytes -= ticket.bytes_reserved
        ticket.tensors.clear()
        ticket.labels = None
        ticket.loss_mask = None
        ticket.ready_event = None
        ticket.ce_ready_event = None


def combine_verified_tickets(
    tickets: Iterable[OnlineMTPActivationTicket],
) -> OnlineMTPActivationTicket:
    """Create a zero-copy logical ticket for one larger backward GEMM.

    Combining tickets is exact because serving weights remain immutable until
    every live ticket has completed.  Original activation tensors stay owned
    by the ring and are released only after the combined backward event.
    """

    tickets = list(tickets)
    if not tickets:
        raise ValueError("cannot combine an empty ticket list")
    weight_version = tickets[0].weight_version
    tensors: Dict[str, torch.Tensor] = {}
    labels = []
    masks = []
    has_any_mask = any(ticket.loss_mask is not None for ticket in tickets)
    merged_step = 0
    for ticket in tickets:
        if ticket.state is not TicketState.VERIFIED:
            raise RuntimeError("only verified tickets can be combined")
        if ticket.weight_version != weight_version:
            raise RuntimeError("cannot combine tickets from different weight versions")
        assert ticket.labels is not None
        labels.append(ticket.labels.reshape(-1))
        if has_any_mask:
            masks.append(
                torch.ones_like(ticket.labels, dtype=torch.bool).reshape(-1)
                if ticket.loss_mask is None
                else ticket.loss_mask.reshape(-1).to(torch.bool)
            )
        step_ids = sorted(
            int(name.split(".", 1)[0][5:])
            for name in ticket.tensors
            if name.endswith(".mlp_input")
        )
        for step_id in step_ids:
            source_prefix = f"step_{step_id}."
            target_prefix = f"step_{merged_step}."
            for name, tensor in ticket.tensors.items():
                if name.startswith(source_prefix):
                    tensors[target_prefix + name[len(source_prefix) :]] = tensor
            merged_step += 1

    return OnlineMTPActivationTicket(
        ticket_id=tickets[0].ticket_id,
        weight_version=weight_version,
        num_tokens=sum(ticket.num_tokens for ticket in tickets),
        tensors=tensors,
        bytes_reserved=0,
        state=TicketState.VERIFIED,
        labels=torch.cat(labels),
        loss_mask=torch.cat(masks) if has_any_mask else None,
        metadata={
            "combined_tickets": len(tickets),
            "defer_ce": all(
                bool(ticket.metadata.get("defer_ce", False)) for ticket in tickets
            ),
            "fused_ce_reduction": all(
                bool(ticket.metadata.get("fused_ce_reduction", False))
                for ticket in tickets
            ),
            # A live scheduler group may mix small tickets whose probability
            # projection was deferred with full tickets that projected inline.
            # Preserve that fact so prepare_fused_ce_reduction can normalize
            # the heterogeneous per-step representation.
            "defer_grouped_ce_projection": any(
                bool(ticket.metadata.get("defer_grouped_ce_projection", False))
                for ticket in tickets
            ),
            "async_ce_producer": all(
                bool(ticket.metadata.get("async_ce_producer", False))
                for ticket in tickets
            ),
        },
    )


def rms_norm_forward(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference RMSNorm returning the inverse RMS needed by backward."""

    stats_dtype = (
        torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    )
    inv_rms = torch.rsqrt(x.to(stats_dtype).square().mean(dim=-1, keepdim=True) + eps)
    output = x * inv_rms.to(x.dtype) * weight
    return output, inv_rms


def rms_norm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    inv_rms: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact RMSNorm backward with FP32 reductions for low-precision inputs."""

    grad = grad_output.float()
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    inv = inv_rms.float()
    weighted_grad = grad * weight_fp32
    projection = (weighted_grad * x_fp32).mean(dim=-1, keepdim=True)
    grad_x = inv * weighted_grad - x_fp32 * inv.pow(3) * projection
    reduce_dims = tuple(range(grad.ndim - 1))
    grad_weight = (grad * x_fp32 * inv).sum(dim=reduce_dims)
    return grad_x.to(x.dtype), grad_weight


def silu_and_mul_forward(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def silu_and_mul_forward_out(
    gate_up: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Inference-equivalent SwiGLU into caller-owned contiguous storage."""

    expected_shape = (*gate_up.shape[:-1], gate_up.shape[-1] // 2)
    if out.shape != expected_shape:
        raise ValueError(f"SwiGLU output shape {out.shape} != {expected_shape}")
    if out.dtype != gate_up.dtype or out.device != gate_up.device:
        raise ValueError("SwiGLU output dtype/device must match its input")
    if not out.is_contiguous():
        raise ValueError("SwiGLU output storage must be contiguous")
    if gate_up.is_cuda:
        # This is the same CUDA implementation used by SiluAndMul.forward_cuda,
        # but supplies dead ticket storage instead of allocating a new tensor.
        from sglang.jit_kernel.activation import silu_and_mul

        return silu_and_mul(gate_up, out)
    out.copy_(silu_and_mul_forward(gate_up))
    return out


def silu_and_mul_backward(
    grad_output: torch.Tensor, gate_up: torch.Tensor
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    sigmoid = torch.sigmoid(gate.float())
    silu = gate.float() * sigmoid
    silu_grad = sigmoid * (1.0 + gate.float() * (1.0 - sigmoid))
    grad = grad_output.float()
    return torch.cat((grad * up.float() * silu_grad, grad * silu), dim=-1).to(
        gate_up.dtype
    )


def silu_and_mul_backward_mixed_precision(
    grad_output: torch.Tensor, gate_up: torch.Tensor
) -> torch.Tensor:
    """SwiGLU backward with the same BF16 rounding points as autograd."""

    gate, up = gate_up.chunk(2, dim=-1)
    grad = grad_output.to(gate_up.dtype)
    grad_gate = torch.ops.aten.silu_backward(grad * up, gate)
    grad_up = grad * torch.nn.functional.silu(gate)
    return torch.cat((grad_gate, grad_up), dim=-1)


def linear_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    need_input_grad: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Backward for ``y = x @ weight.T`` using FP32 gradient accumulation."""

    x_2d = x.reshape(-1, x.shape[-1])
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
    grad_weight = grad_2d.float().transpose(0, 1) @ x_2d.float()
    grad_input = None
    if need_input_grad:
        grad_input = (grad_2d.float() @ weight.float()).reshape_as(x).to(x.dtype)
    return grad_input, grad_weight


@dataclass
class DenseSwiGLUContext:
    x: torch.Tensor
    gate_up: torch.Tensor
    activated: torch.Tensor


def dense_swiglu_forward(
    x: torch.Tensor, gate_up_weight: torch.Tensor, down_weight: torch.Tensor
) -> Tuple[torch.Tensor, DenseSwiGLUContext]:
    gate_up = x @ gate_up_weight.transpose(0, 1)
    activated = silu_and_mul_forward(gate_up)
    output = activated @ down_weight.transpose(0, 1)
    return output, DenseSwiGLUContext(x=x, gate_up=gate_up, activated=activated)


def dense_swiglu_backward(
    grad_output: torch.Tensor,
    context: DenseSwiGLUContext,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_activated, grad_down_weight = linear_backward(
        grad_output, context.activated, down_weight
    )
    assert grad_activated is not None
    grad_gate_up = silu_and_mul_backward(grad_activated, context.gate_up)
    grad_x, grad_gate_up_weight = linear_backward(
        grad_gate_up, context.x, gate_up_weight
    )
    assert grad_x is not None
    return grad_x, grad_gate_up_weight, grad_down_weight


def dense_swiglu_backward_mixed_precision(
    grad_output: torch.Tensor,
    context: DenseSwiGLUContext,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    need_input_grad: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Tensor-Core backward matching ordinary BF16 mixed-precision training.

    Elementwise nonlinear derivatives retain FP32 intermediates, while the
    three GEMMs use the inference weight dtype.  The FP32 reference above is
    intentionally kept as the numerical oracle.
    """

    compute_dtype = down_weight.dtype
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).to(compute_dtype)
    activated_2d = context.activated.reshape(-1, context.activated.shape[-1]).to(
        compute_dtype
    )
    grad_down_weight = grad_2d.transpose(0, 1) @ activated_2d
    grad_activated = grad_2d @ down_weight
    grad_gate_up = silu_and_mul_backward_mixed_precision(
        grad_activated.reshape_as(context.activated), context.gate_up
    )
    grad_gate_2d = grad_gate_up.reshape(-1, grad_gate_up.shape[-1]).to(
        gate_up_weight.dtype
    )
    x_2d = context.x.reshape(-1, context.x.shape[-1]).to(gate_up_weight.dtype)
    grad_gate_up_weight = grad_gate_2d.transpose(0, 1) @ x_2d
    grad_x = None
    if need_input_grad:
        grad_x = (grad_gate_2d @ gate_up_weight).reshape_as(context.x)
    return grad_x, grad_gate_up_weight, grad_down_weight


def dense_swiglu_backward_accumulate_mixed_precision(
    grad_output: torch.Tensor,
    context: DenseSwiGLUContext,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_accumulator: torch.Tensor,
    down_accumulator: torch.Tensor,
    *,
    use_triton_swiglu: bool = False,
) -> None:
    """Parameter-only SwiGLU backward with in-GEMM accumulation.

    ``beta=1`` makes cuBLAS accumulate directly into the persistent gradient
    buffers.  Compared with returning two full weight-gradient tensors and
    launching separate ``add_`` kernels, this removes one materialization and
    one extra read/write of every trainable MLP parameter.
    """

    if gate_up_accumulator.shape != gate_up_weight.shape:
        raise ValueError("gate/up accumulator shape does not match its weight")
    if down_accumulator.shape != down_weight.shape:
        raise ValueError("down accumulator shape does not match its weight")
    if gate_up_accumulator.dtype != gate_up_weight.dtype:
        raise ValueError("gate/up accumulator dtype does not match its weight")
    if down_accumulator.dtype != down_weight.dtype:
        raise ValueError("down accumulator dtype does not match its weight")

    grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).to(down_weight.dtype)
    activated_2d = context.activated.reshape(-1, context.activated.shape[-1]).to(
        down_weight.dtype
    )
    torch.addmm(
        down_accumulator,
        grad_2d.transpose(0, 1),
        activated_2d,
        beta=1.0,
        alpha=1.0,
        out=down_accumulator,
    )
    # The preceding dW GEMM has consumed ``activated_2d`` on this same stream.
    # Reuse that now-dead [rows, intermediate] buffer as the dActivation GEMM
    # destination.  Stream ordering keeps the read-before-write dependency
    # exact and removes one BF16 intermediate allocation per backward group.
    torch.mm(grad_2d, down_weight, out=activated_2d)
    grad_activated = activated_2d
    if use_triton_swiglu:
        from sglang.srt.speculative.triton_ops.online_mtp_swiglu import (
            online_mtp_swiglu_backward,
        )

        # ``activated`` has already been reconstructed.  No later operation
        # needs the saved gate/up value, so its ticket storage becomes the
        # gradient destination and avoids another allocation.
        grad_gate_up = online_mtp_swiglu_backward(
            grad_activated.reshape_as(context.activated),
            context.gate_up,
            inplace=True,
        )
    else:
        grad_gate_up = silu_and_mul_backward_mixed_precision(
            grad_activated.reshape_as(context.activated), context.gate_up
        )
    grad_gate_2d = grad_gate_up.reshape(-1, grad_gate_up.shape[-1]).to(
        gate_up_weight.dtype
    )
    x_2d = context.x.reshape(-1, context.x.shape[-1]).to(gate_up_weight.dtype)
    torch.addmm(
        gate_up_accumulator,
        grad_gate_2d.transpose(0, 1),
        x_2d,
        beta=1.0,
        alpha=1.0,
        out=gate_up_accumulator,
    )


def dense_swiglu_backward_factors_mixed_precision(
    grad_output: torch.Tensor,
    context: DenseSwiGLUContext,
    down_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the two left factors needed by dense SwiGLU weight gradients.

    For a microbatch, both weight gradients are low-rank outer-product sums::

        dW_down = grad_output.T @ activated
        dW_gate = grad_gate_up.T @ x

    Retaining these factors allows an exact accumulated gradient to be
    materialized once per optimizer window instead of writing two full-sized
    gradient matrices for every sampled speculative block.
    """

    grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).to(down_weight.dtype)
    grad_activated = grad_2d @ down_weight
    grad_gate_up = silu_and_mul_backward_mixed_precision(
        grad_activated.reshape_as(context.activated), context.gate_up
    )
    return grad_2d, grad_gate_up.reshape(-1, grad_gate_up.shape[-1])


class FactorizedMLPGradientAccumulator:
    """Preallocated exact outer-product factors for one optimizer window."""

    def __init__(
        self,
        *,
        capacity_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if capacity_tokens <= 0:
            raise ValueError("factor capacity must be positive")
        self.capacity_tokens = capacity_tokens
        self.rows = 0
        self.x = torch.empty(capacity_tokens, hidden_size, dtype=dtype, device=device)
        self.activated = torch.empty(
            capacity_tokens, intermediate_size, dtype=dtype, device=device
        )
        self.grad_output = torch.empty(
            capacity_tokens, hidden_size, dtype=dtype, device=device
        )
        self.grad_gate_up = torch.empty(
            capacity_tokens, 2 * intermediate_size, dtype=dtype, device=device
        )

    def append(
        self,
        context: DenseSwiGLUContext,
        grad_output: torch.Tensor,
        grad_gate_up: torch.Tensor,
        *,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> int:
        x = context.x.reshape(-1, context.x.shape[-1])
        activated = context.activated.reshape(-1, context.activated.shape[-1])
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        grad_gate_up = grad_gate_up.reshape(-1, grad_gate_up.shape[-1])
        if loss_mask is not None:
            active = loss_mask.reshape(-1).to(torch.bool)
            x = x[active]
            activated = activated[active]
            grad_output = grad_output[active]
            grad_gate_up = grad_gate_up[active]
        num_rows = x.shape[0]
        stop = self.rows + num_rows
        if stop > self.capacity_tokens:
            raise RuntimeError(
                "factorized MTP gradient window exceeded its preallocated "
                f"capacity: required={stop}, capacity={self.capacity_tokens}"
            )
        destination = slice(self.rows, stop)
        self.x[destination].copy_(x)
        self.activated[destination].copy_(activated)
        self.grad_output[destination].copy_(grad_output)
        self.grad_gate_up[destination].copy_(grad_gate_up)
        self.rows = stop
        return num_rows

    def materialize_into_(
        self,
        gate_up_accumulator: torch.Tensor,
        down_accumulator: torch.Tensor,
    ) -> None:
        if self.rows == 0:
            return
        rows = slice(0, self.rows)
        torch.addmm(
            down_accumulator,
            self.grad_output[rows].transpose(0, 1),
            self.activated[rows],
            beta=1.0,
            alpha=1.0,
            out=down_accumulator,
        )
        torch.addmm(
            gate_up_accumulator,
            self.grad_gate_up[rows].transpose(0, 1),
            self.x[rows],
            beta=1.0,
            alpha=1.0,
            out=gate_up_accumulator,
        )
        self.rows = 0

    def zero(self) -> None:
        self.rows = 0


def vocab_ce_hidden_backward_from_logits(
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 8192,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute hard-CE loss and hidden gradient without materializing ``dlogits``.

    ``lm_head_weight`` is frozen.  Logits already produced by inference are
    consumed chunk by chunk, limiting temporary storage to ``tokens * chunk``.
    The returned hidden gradient and loss are FP32.
    """

    if logits.ndim != 2 or lm_head_weight.ndim != 2:
        raise ValueError("logits and lm_head_weight must be matrices")
    if logits.shape[1] != lm_head_weight.shape[0]:
        raise ValueError("vocabulary dimensions do not match")
    labels = labels.reshape(-1).long()
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must contain one id per logits row")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    logits_fp32 = logits.float()
    logsumexp = torch.logsumexp(logits_fp32, dim=-1)
    target_logits = logits_fp32.gather(1, labels[:, None]).squeeze(1)
    losses = logsumexp - target_logits

    if loss_mask is None:
        row_scale = torch.full_like(losses, 1.0 / max(1, losses.numel()))
    else:
        mask = loss_mask.reshape(-1).to(dtype=torch.bool, device=logits.device)
        if mask.shape[0] != logits.shape[0]:
            raise ValueError("loss_mask must contain one value per logits row")
        denom = mask.sum().clamp_min(1).float()
        row_scale = mask.float() / denom
        losses = losses * mask

    grad_hidden = torch.zeros(
        (logits.shape[0], lm_head_weight.shape[1]),
        dtype=torch.float32,
        device=logits.device,
    )
    for start in range(0, logits.shape[1], chunk_size):
        stop = min(start + chunk_size, logits.shape[1])
        probabilities = torch.exp(logits_fp32[:, start:stop] - logsumexp[:, None])
        probabilities.mul_(row_scale[:, None])
        grad_hidden.add_(probabilities @ lm_head_weight[start:stop].float())

    grad_hidden.sub_(lm_head_weight[labels].float() * row_scale[:, None])
    return losses.sum() / row_scale.ne(0).sum().clamp_min(1), grad_hidden


def vocab_ce_forward_stats(
    logits: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute label-independent CE statistics while inference logits live.

    Returns ``(logsumexp, E_p[W])``.  The second tensor is exactly the positive
    term of the hidden-state CE gradient.  Keeping these two small tensors
    avoids retaining a ``tokens * vocab`` logits buffer until verification.

    This reference accepts an unsharded LM head.  The serving integration uses
    the same operation on the local vocabulary shard followed by a TP sum.
    """

    if logits.ndim != 2 or lm_head_weight.ndim != 2:
        raise ValueError("logits and lm_head_weight must be matrices")
    if logits.shape[1] != lm_head_weight.shape[0]:
        raise ValueError("vocabulary dimensions do not match")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    logits_fp32 = logits.float()
    logsumexp = torch.logsumexp(logits_fp32, dim=-1)
    expected_weight = torch.zeros(
        (logits.shape[0], lm_head_weight.shape[1]),
        dtype=torch.float32,
        device=logits.device,
    )
    for start in range(0, logits.shape[1], chunk_size):
        stop = min(start + chunk_size, logits.shape[1])
        probabilities = torch.exp(logits_fp32[:, start:stop] - logsumexp[:, None])
        expected_weight.add_(probabilities @ lm_head_weight[start:stop].float())
    return logsumexp, expected_weight


def vocab_ce_hidden_backward_from_stats(
    logsumexp: torch.Tensor,
    expected_weight: torch.Tensor,
    final_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finish hard CE after verification using compact forward statistics."""

    labels = labels.reshape(-1).long()
    if labels.shape[0] != final_hidden.shape[0]:
        raise ValueError("labels must contain one id per hidden-state row")
    if expected_weight.shape != final_hidden.shape:
        raise ValueError("expected_weight and final_hidden shapes must match")
    target_weight = lm_head_weight[labels].float()
    target_logits = (final_hidden.float() * target_weight).sum(dim=-1)
    losses = logsumexp.reshape(-1).float() - target_logits
    if loss_mask is None:
        row_scale = torch.full_like(losses, 1.0 / max(1, losses.numel()))
    else:
        mask = loss_mask.reshape(-1).to(dtype=torch.bool, device=losses.device)
        if mask.shape[0] != labels.shape[0]:
            raise ValueError("loss_mask must contain one value per hidden-state row")
        row_scale = mask.float() / mask.sum().clamp_min(1).float()
        losses = losses * mask
    grad_hidden = (expected_weight.float() - target_weight) * row_scale[:, None]
    return losses.sum() / row_scale.ne(0).sum().clamp_min(1), grad_hidden


class FlatAdamWAccumulator:
    """Delayed in-place AdamW with configurable gradient-buffer precision.

    The serving integration uses BF16 accumulation buffers to keep the idle
    memory tax small.  FP32 moments and master weights are allocated lazily at
    the first optimizer step, so capture-only deployments do not pay for them.
    """

    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
        *,
        accumulation_tokens: int,
        gradient_dtype: torch.dtype = torch.float32,
        use_triton_optimizer: bool = False,
        direct_parameter_update: bool = False,
    ) -> None:
        if accumulation_tokens <= 0:
            raise ValueError("accumulation_tokens must be positive")
        self.parameters = {name: parameter for name, parameter in named_parameters}
        if not self.parameters:
            raise ValueError("no trainable parameters")
        self.gradients = {
            name: torch.zeros_like(parameter, dtype=gradient_dtype)
            for name, parameter in self.parameters.items()
        }
        # Serving can accumulate for a long time without applying an update.
        # Allocate the three large FP32 optimizer tensors only at first apply.
        self.first_moments: Optional[Dict[str, torch.Tensor]] = None
        self.second_moments: Optional[Dict[str, torch.Tensor]] = None
        self.master_parameters: Optional[Dict[str, torch.Tensor]] = None
        self.prepared_parameters: Optional[Dict[str, torch.Tensor]] = None
        self._grad_norm_partials: Optional[torch.Tensor] = None
        self.has_prepared_update = False
        if direct_parameter_update and not use_triton_optimizer:
            raise ValueError("direct parameter update requires the Triton optimizer")
        self.use_triton_optimizer = use_triton_optimizer
        self.direct_parameter_update = direct_parameter_update
        # The production gradient dtype equals the serving parameter dtype.
        # In the shadow-publication mode, the fused optimizer can overwrite
        # each consumed gradient element with the shadow weight and publish it
        # later. Direct publication instead keeps gradients disjoint so the
        # same kernel can clear them while writing graph-stable parameters.
        self.reuse_gradient_for_prepared = (
            use_triton_optimizer
            and not direct_parameter_update
            and all(
                self.gradients[name].dtype == parameter.dtype
                for name, parameter in self.parameters.items()
            )
        )
        self.accumulation_tokens = accumulation_tokens
        self.tokens = 0
        self.step = 0

    @property
    def ready(self) -> bool:
        return self.tokens >= self.accumulation_tokens

    def accumulate(
        self, gradients: Mapping[str, torch.Tensor], *, num_tokens: int
    ) -> None:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        missing = set(gradients) - set(self.gradients)
        if missing:
            raise KeyError(f"unknown gradient parameters: {sorted(missing)}")
        if self.has_prepared_update and self.reuse_gradient_for_prepared:
            raise RuntimeError(
                "cannot accumulate while gradient buffers hold prepared weights"
            )
        for name, gradient in gradients.items():
            if gradient.shape != self.gradients[name].shape:
                raise ValueError(
                    f"gradient shape mismatch for {name}: {gradient.shape} != "
                    f"{self.gradients[name].shape}"
                )
            self.gradients[name].add_(gradient.to(self.gradients[name].dtype))
        self.tokens += num_tokens

    def zero(self) -> None:
        if self.has_prepared_update and self.reuse_gradient_for_prepared:
            raise RuntimeError(
                "cannot zero gradient buffers before prepared weights publish"
            )
        for gradient in self.gradients.values():
            gradient.zero_()
        self.tokens = 0

    @torch.no_grad()
    def materialize_optimizer_state(self) -> None:
        """Allocate persistent Adam state outside the serving request path."""

        if self.first_moments is not None:
            return
        self.first_moments = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in self.parameters.items()
        }
        self.second_moments = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in self.parameters.items()
        }
        self.master_parameters = {
            name: parameter.detach().float().clone()
            for name, parameter in self.parameters.items()
        }
        if self.direct_parameter_update:
            # The synchronous online-update boundary already keeps inference
            # off these graph-stable addresses until the optimizer event has
            # completed.  Let the fused kernel publish directly and avoid a
            # second parameter-sized BF16 copy.
            self.prepared_parameters = dict(self.parameters)
        elif self.reuse_gradient_for_prepared:
            self.prepared_parameters = dict(self.gradients)
        else:
            self.prepared_parameters = {
                name: torch.empty_like(parameter)
                for name, parameter in self.parameters.items()
            }

    @torch.no_grad()
    def scaled_gradient_norm_sq(
        self,
        contribution_scales: Optional[Mapping[str, float]] = None,
    ) -> torch.Tensor:
        """Return the squared norm of token-averaged gradients.

        The opt-in Triton path reduces directly from the persistent gradient
        buffers into a compact reusable partial vector.  The default path is
        intentionally the original ATen expression.
        """

        device = next(iter(self.gradients.values())).device
        if self.tokens == 0:
            return torch.zeros((), dtype=torch.float32, device=device)
        contribution_scales = contribution_scales or {}
        unknown = set(contribution_scales) - set(self.gradients)
        if unknown:
            raise KeyError(
                f"unknown norm contribution parameters: {sorted(unknown)}"
            )
        if not self.use_triton_optimizer:
            norm_sq = torch.zeros((), dtype=torch.float32, device=device)
            for name, gradient in self.gradients.items():
                contribution = (
                    gradient.float() / float(self.tokens)
                ).square().sum()
                norm_sq.add_(
                    contribution,
                    alpha=float(contribution_scales.get(name, 1.0)),
                )
            return norm_sq

        from sglang.srt.speculative.triton_ops.online_mtp_adamw import (
            grad_norm_num_partials,
            online_mtp_scaled_grad_norm_sq,
        )

        gradients = tuple(self.gradients.values())
        required = grad_norm_num_partials(gradients)
        if (
            self._grad_norm_partials is None
            or self._grad_norm_partials.numel() < required
            or self._grad_norm_partials.device != device
        ):
            self._grad_norm_partials = torch.empty(
                required, dtype=torch.float32, device=device
            )
        return online_mtp_scaled_grad_norm_sq(
            gradients,
            denominator=float(self.tokens),
            contribution_scales=tuple(
                float(contribution_scales.get(name, 1.0))
                for name in self.gradients
            ),
            partials_out=self._grad_norm_partials,
        )

    @torch.no_grad()
    def warmup_optimizer_kernels_(self, *, weight_decay: float = 0.0) -> None:
        """Touch the full optimizer path without changing weights or state."""

        if self.tokens:
            raise RuntimeError("optimizer warmup requires empty gradient buffers")
        self.materialize_optimizer_state()
        self.tokens = 1
        if self.use_triton_optimizer:
            # Materialize/JIT the compact partial workspace as well as AdamW.
            # The live update supplies a global override, so apply_ alone would
            # otherwise leave this startup-sensitive kernel cold.
            self.scaled_gradient_norm_sq()
        self.apply_(
            learning_rate=0.0,
            weight_decay=weight_decay,
            grad_norm_override=0.0,
        )
        # Zero gradients leave moments/master weights unchanged.  Rewind the
        # logical Adam step so the first real update still uses t=1.
        self.step = 0

    @torch.no_grad()
    def apply_(
        self,
        *,
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        grad_norm_override: Optional[float] = None,
    ) -> float:
        grad_norm = self.prepare_(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            grad_norm_override=grad_norm_override,
        )
        if self.has_prepared_update:
            self.publish_()
        return grad_norm

    @torch.no_grad()
    def prepare_(
        self,
        *,
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: Optional[float] = None,
        grad_norm_override: Optional[float] = None,
    ) -> float:
        """Update Adam state and materialize shadow serving weights.

        Active parameters remain immutable until :meth:`publish_`, so this
        expensive FP32 work may execute on the training stream while serving
        continues with the previous complete weight version.
        """

        if self.tokens == 0:
            return 0.0
        if self.has_prepared_update:
            raise RuntimeError("cannot prepare a second AdamW update before publish")
        self.materialize_optimizer_state()
        assert self.first_moments is not None
        assert self.second_moments is not None
        assert self.master_parameters is not None
        assert self.prepared_parameters is not None
        beta1, beta2 = betas
        if grad_norm_override is None:
            norm_sq = self.scaled_gradient_norm_sq()
            grad_norm = math.sqrt(float(norm_sq.item()))
        else:
            grad_norm = grad_norm_override
        clip_scale = 1.0
        if max_grad_norm is not None and grad_norm > max_grad_norm:
            clip_scale = max_grad_norm / (grad_norm + 1e-12)

        self.step += 1
        correction1 = 1.0 - beta1**self.step
        correction2 = 1.0 - beta2**self.step
        if self.use_triton_optimizer:
            from sglang.srt.speculative.triton_ops.online_mtp_adamw import (
                online_mtp_adamw_prepare_,
            )

            for name in self.parameters:
                online_mtp_adamw_prepare_(
                    self.gradients[name],
                    self.first_moments[name],
                    self.second_moments[name],
                    self.master_parameters[name],
                    self.prepared_parameters[name],
                    denominator=float(self.tokens),
                    clip_scale=clip_scale,
                    betas=betas,
                    corrections=(correction1, correction2),
                    epsilon=eps,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    clear_gradient=not self.reuse_gradient_for_prepared,
                )
            # Aliased buffers now hold the complete prepared version and must
            # remain untouched until publish_.  Distinct buffers were zeroed
            # by the fused kernels themselves.
            self.tokens = 0
            self.has_prepared_update = True
            return grad_norm

        scaled_gradients = [
            gradient.float() / float(self.tokens)
            for gradient in self.gradients.values()
        ]
        for (name, _parameter), gradient in zip(
            self.parameters.items(), scaled_gradients
        ):
            gradient.mul_(clip_scale)
            first = self.first_moments[name]
            second = self.second_moments[name]
            master = self.master_parameters[name]
            first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
            update = (first / correction1) / ((second / correction2).sqrt().add_(eps))
            if weight_decay:
                master.mul_(1.0 - learning_rate * weight_decay)
            master.add_(update, alpha=-learning_rate)
            self.prepared_parameters[name].copy_(master)

        self.zero()
        self.has_prepared_update = True
        return grad_norm

    @torch.no_grad()
    def publish_(self) -> None:
        """Publish one complete update into graph-stable parameters."""

        if not self.has_prepared_update or self.prepared_parameters is None:
            raise RuntimeError("no prepared AdamW update is available to publish")
        if not self.direct_parameter_update:
            for name, parameter in self.parameters.items():
                parameter.copy_(self.prepared_parameters[name])
        self.has_prepared_update = False
        if self.reuse_gradient_for_prepared:
            for gradient in self.gradients.values():
                gradient.zero_()


def vocab_parallel_ce_forward_stats(
    logits: torch.Tensor,
    lm_head,
    *,
    chunk_size: int = 32768,
    use_triton_lse: bool = False,
    use_triton_expcast: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TP-aware variant of :func:`vocab_ce_forward_stats` for SGLang heads."""

    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    indices = lm_head.shard_indices
    start = indices.org_vocab_start_index
    stop = indices.org_vocab_end_index
    local_weight = lm_head.weight[: stop - start]
    if use_triton_lse:
        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
            compact_vocab_logsumexp,
        )

        logsumexp = compact_vocab_logsumexp(logits)
    else:
        logsumexp = torch.logsumexp(logits.float(), dim=-1)
    local_expected = torch.zeros(
        (logits.shape[0], local_weight.shape[1]),
        dtype=torch.float32,
        device=logits.device,
    )
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        local_start = chunk_start - start
        local_stop = chunk_stop - start
        if use_triton_expcast:
            from sglang.srt.speculative.triton_ops.online_mtp_lse import (
                compact_vocab_probability,
            )

            probabilities = compact_vocab_probability(
                logits,
                logsumexp,
                local_start=chunk_start,
                local_stop=chunk_stop,
                output_dtype=local_weight.dtype,
            )
        else:
            probabilities = torch.exp(
                logits[:, chunk_start:chunk_stop].float() - logsumexp[:, None]
            ).to(local_weight.dtype)
        local_expected.add_(
            (probabilities @ local_weight[local_start:local_stop]).float()
        )
    expected = tensor_model_parallel_all_reduce(local_expected)
    return logsumexp, expected


def vocab_parallel_ce_forward_probability(
    logits: torch.Tensor,
    lm_head,
    *,
    use_triton_lse: bool = False,
    use_triton_expcast: bool = False,
    out: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
    probability_store_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return LSE plus this TP rank's BF16 probability shard.

    The probability is intentionally ephemeral: D4 combines the three inner
    steps before projecting through the local LM-head shard.
    """

    indices = lm_head.shard_indices
    start = indices.org_vocab_start_index
    stop = indices.org_vocab_end_index
    if output_dtype is None:
        output_dtype = lm_head.weight.dtype
    if probability_store_scale != 1.0 and output_dtype != torch.float8_e4m3fn:
        raise ValueError("scaled CE probabilities currently require FP8 E4M3 output")
    if use_triton_lse:
        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
            compact_vocab_logsumexp,
        )

        logsumexp = compact_vocab_logsumexp(logits)
    else:
        logsumexp = torch.logsumexp(logits.float(), dim=-1)
    if use_triton_expcast:
        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
            compact_vocab_probability,
        )

        probability = compact_vocab_probability(
            logits,
            logsumexp,
            local_start=start,
            local_stop=stop,
            output_dtype=output_dtype,
            out=out,
            store_scale=probability_store_scale,
        )
    else:
        probability = (
            torch.exp(logits[:, start:stop].float() - logsumexp[:, None])
            .mul_(probability_store_scale)
            .to(output_dtype)
        )
        if out is not None:
            out.copy_(probability)
            probability = out
    return logsumexp, probability


@torch.no_grad()
def vocab_parallel_ce_forward_local_grouped(
    step_logits: Iterable[torch.Tensor],
    lm_head,
    *,
    use_triton_lse: bool,
    use_triton_expcast: bool,
) -> Tuple[list[torch.Tensor], torch.Tensor]:
    """Produce exact BF16 local CE statistics for several draft steps.

    This is the training-stream counterpart of the in-graph grouped producer:
    every step keeps the same FP32 LSE and BF16 probability boundary, while one
    packed GEMM projects all local-vocabulary probabilities through the frozen
    LM-head shard.  No TP collective is issued here, so the producer can run as
    soon as graph-stable logits have been snapshotted and overlap target verify.
    """

    logits = list(step_logits)
    if not logits:
        raise ValueError("async CE producer requires at least one draft step")
    reference = logits[0]
    if reference.ndim != 2 or not reference.is_contiguous():
        raise ValueError("async CE logits must be contiguous matrices")
    if any(
        value.ndim != 2
        or value.shape[1] != reference.shape[1]
        or value.dtype != reference.dtype
        or value.device != reference.device
        or not value.is_contiguous()
        for value in logits[1:]
    ):
        raise ValueError("async CE step logits have incompatible layouts")

    indices = lm_head.shard_indices
    start = indices.org_vocab_start_index
    stop = indices.org_vocab_end_index
    local_weight = lm_head.weight[: stop - start]
    total_rows = sum(value.shape[0] for value in logits)
    probability = torch.empty(
        (total_rows, stop - start),
        dtype=local_weight.dtype,
        device=reference.device,
    )
    logsumexp = []
    offset = 0
    for value in logits:
        rows = value.shape[0]
        step_lse, _ = vocab_parallel_ce_forward_probability(
            value,
            lm_head,
            use_triton_lse=use_triton_lse,
            use_triton_expcast=use_triton_expcast,
            out=probability[offset : offset + rows],
            output_dtype=local_weight.dtype,
        )
        logsumexp.append(step_lse)
        offset += rows
    if offset != total_rows:
        raise RuntimeError("async CE probability row packing failed")
    local_expected = probability @ local_weight
    return logsumexp, local_expected


@torch.no_grad()
def prepare_fp8_ce_projection(
    lm_head,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one frozen TP LM-head shard for the Hopper FP8 CE producer.

    The column-major view is the layout required by ``torch._scaled_mm``.  CE
    probabilities are stored after multiplication by 448, so their matching
    dequantization scale is constant and graph-stable.
    """

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("online MTP FP8 CE projection requires Hopper or newer")
    indices = lm_head.shard_indices
    local_weight = lm_head.weight[
        : indices.org_vocab_end_index - indices.org_vocab_start_index
    ]
    if local_weight.dtype != torch.bfloat16:
        raise RuntimeError("online MTP FP8 CE projection requires a BF16 LM-head")
    weight_scale = (local_weight.float().abs().max() / _ONLINE_MTP_FP8_MAX).clamp_min_(
        torch.finfo(torch.float32).tiny
    )
    weight_fp8 = (
        (local_weight / weight_scale)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
        .contiguous()
        .transpose(0, 1)
    )
    probability_scale = torch.tensor(
        1.0 / _ONLINE_MTP_FP8_MAX,
        dtype=torch.float32,
        device=local_weight.device,
    )
    return weight_fp8, weight_scale, probability_scale


def vocab_parallel_target_weight(lm_head, labels: torch.Tensor) -> torch.Tensor:
    """Gather only the target rows needed by hard CE across the TP group."""

    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    labels = labels.reshape(-1).long()
    indices = lm_head.shard_indices
    start = indices.org_vocab_start_index
    stop = indices.org_vocab_end_index
    local_mask = (labels >= start) & (labels < stop)
    # Avoid ``local_mask.any()``: branching on a CUDA scalar synchronizes the
    # serving thread on every learned block.  Clamp gives every row a valid
    # local address and the mask zeroes non-owned rows before the TP sum.
    local_rows = (labels - start).clamp_(0, stop - start - 1)
    # Exactly one rank owns each vocabulary row; all other ranks contribute
    # zero. Reducing in the serving BF16 weight dtype is therefore exact, while
    # halving both communication and retained target-row storage versus FP32.
    output = lm_head.weight[local_rows]
    output.mul_(local_mask[:, None])
    return tensor_model_parallel_all_reduce(output)


class OnlineMTPMLPTrainer:
    """First in-engine backward slice: final norm plus the dense MTP MLP.

    Attention and input projection remain frozen in this initial slice.  The
    class consumes tensors recorded by the real inference forward, so it is a
    useful end-to-end scheduling and numerical-parity milestone before adding
    paged-attention backward.
    """

    def __init__(
        self,
        model,
        *,
        accumulation_tokens: int,
        factorized_gradient_accumulation: bool = False,
        recompute_gate_up: bool = False,
        use_triton_swiglu: bool = False,
        use_triton_ce_rmsnorm: bool = False,
        use_triton_optimizer: bool = False,
        direct_optimizer_publish: bool = False,
    ) -> None:
        if len(model.model.layers) != 1:
            raise ValueError("online Qwen MTP training currently requires one layer")
        layer = model.model.layers[0]
        mlp = layer.mlp
        self.model = model
        self.mlp = mlp
        self.final_norm = model.model.norm
        self.lm_head = model.lm_head
        self.accumulator = FlatAdamWAccumulator(
            (
                ("mlp.gate_up_proj.weight", mlp.gate_up_proj.weight),
                ("mlp.down_proj.weight", mlp.down_proj.weight),
                ("final_norm.weight", self.final_norm.weight),
            ),
            accumulation_tokens=accumulation_tokens,
            gradient_dtype=mlp.gate_up_proj.weight.dtype,
            use_triton_optimizer=use_triton_optimizer,
            direct_parameter_update=direct_optimizer_publish,
        )
        self.factorized_gradient_accumulation = factorized_gradient_accumulation
        self.recompute_gate_up = recompute_gate_up
        self.use_triton_swiglu = use_triton_swiglu
        self.use_triton_ce_rmsnorm = use_triton_ce_rmsnorm
        self.use_triton_optimizer = use_triton_optimizer
        self.direct_optimizer_publish = direct_optimizer_publish
        self._ce_tail_inv_rms: Optional[torch.Tensor] = None
        self._ce_tail_losses: Optional[torch.Tensor] = None
        self._fused_ce_packed: Optional[torch.Tensor] = None
        self.factor_accumulator = None
        if factorized_gradient_accumulation:
            hidden_size = mlp.down_proj.weight.shape[0]
            intermediate_size = mlp.down_proj.weight.shape[1]
            # One live scheduler batch can cross the nominal optimizer token
            # boundary. A 50% overshoot covers the standard TP8 D4 graph up to
            # batch 512 while keeping this bounded and fully preallocated.
            factor_capacity = accumulation_tokens + max(2048, accumulation_tokens // 2)
            self.factor_accumulator = FactorizedMLPGradientAccumulator(
                capacity_tokens=factor_capacity,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=mlp.down_proj.weight.dtype,
                device=mlp.down_proj.weight.device,
            )

    def materialize_factorized_gradients(self) -> None:
        if self.factor_accumulator is None:
            return
        self.factor_accumulator.materialize_into_(
            self.accumulator.gradients["mlp.gate_up_proj.weight"],
            self.accumulator.gradients["mlp.down_proj.weight"],
        )

    def zero_accumulation(self) -> None:
        self.accumulator.zero()
        if self.factor_accumulator is not None:
            self.factor_accumulator.zero()

    def _ce_tail_scratch(
        self, norm_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return grow-only scalar scratch views for the fused verified CE tail."""

        rows = norm_input.shape[0]
        capacity = 1 << max(0, rows - 1).bit_length()
        needs_growth = (
            self._ce_tail_inv_rms is None
            or self._ce_tail_inv_rms.shape[0] < rows
        )
        if needs_growth:
            self._ce_tail_inv_rms = torch.empty(
                capacity, dtype=torch.float32, device=norm_input.device
            )
            self._ce_tail_losses = torch.empty(
                capacity, dtype=torch.float32, device=norm_input.device
            )
        assert self._ce_tail_inv_rms is not None
        assert self._ce_tail_losses is not None
        return (
            self._ce_tail_inv_rms[:rows],
            self._ce_tail_losses[:rows],
        )

    def _fused_ce_packed_scratch(self, rows: int, hidden_size: int) -> torch.Tensor:
        """Return a grow-only, 16-byte-aligned FP32 TP-reduction buffer."""

        logical_elements = rows * hidden_size + rows
        aligned_elements = (logical_elements + 3) // 4 * 4
        if (
            self._fused_ce_packed is None
            or self._fused_ce_packed.numel() < aligned_elements
        ):
            capacity = 1 << max(0, aligned_elements - 1).bit_length()
            self._fused_ce_packed = torch.empty(
                capacity,
                dtype=torch.float32,
                device=self.lm_head.weight.device,
            )
        return self._fused_ce_packed[:aligned_elements]

    def prepare_fused_ce_local(
        self,
        ticket: OnlineMTPActivationTicket,
    ) -> PreparedCELocal:
        """Project and pack the rank-local, label-dependent CE contribution."""

        if ticket.state is not TicketState.VERIFIED:
            raise RuntimeError("only verified tickets can prepare fused CE")
        if not self.use_triton_ce_rmsnorm:
            raise RuntimeError("fused CE reduction requires Triton CE/RMSNorm")
        assert ticket.labels is not None
        labels = ticket.labels.reshape(-1)
        step_ids = sorted(
            int(name.split(".", 1)[0][5:])
            for name in ticket.tensors
            if name.endswith(".mlp_input")
        )
        prefixes = [f"step_{step_id}." for step_id in step_ids]

        def concat(name: str) -> torch.Tensor:
            return concatenate_adjacent_rows(
                ticket.tensors[prefix + name] for prefix in prefixes
            )

        norm_input = concat("final_norm_input")
        rows, hidden_size = norm_input.shape
        if labels.numel() != rows:
            raise ValueError("fused CE ticket row count is invalid")

        from sglang.srt.speculative.triton_ops.online_mtp_ce_rmsnorm import (
            pack_local_ce_for_tp_reduce,
        )

        indices = self.lm_head.shard_indices
        shard_start = indices.org_vocab_start_index
        shard_stop = indices.org_vocab_end_index
        local_weight = self.lm_head.weight[: shard_stop - shard_start]
        deferred_prefixes = [
            prefix for prefix in prefixes if prefix + "ce_probability" in ticket.tensors
        ]
        if deferred_prefixes:
            probabilities = concatenate_adjacent_rows(
                ticket.tensors[prefix + "ce_probability"]
                for prefix in deferred_prefixes
            )
            if probabilities.shape[1] != shard_stop - shard_start:
                raise ValueError("deferred grouped CE probability shape is invalid")
            projected = probabilities @ local_weight
            projected_offset = 0
            expected_chunks = []
            for prefix in prefixes:
                probability_name = prefix + "ce_probability"
                if probability_name in ticket.tensors:
                    step_rows = ticket.tensors[probability_name].shape[0]
                    expected_chunks.append(
                        projected[
                            projected_offset : projected_offset + step_rows
                        ]
                    )
                    projected_offset += step_rows
                else:
                    expected_chunks.append(
                        ticket.tensors[prefix + "ce_local_expected_weight"]
                    )
            if projected_offset != projected.shape[0]:
                raise RuntimeError("deferred CE projection row packing failed")
            local_expected = concatenate_adjacent_rows(expected_chunks)
        else:
            local_expected = concat("ce_local_expected_weight")
        if local_expected.shape != norm_input.shape:
            raise ValueError("fused CE hidden shape is invalid")
        packed = pack_local_ce_for_tp_reduce(
            local_expected,
            norm_input,
            self.final_norm.weight.detach(),
            local_weight,
            labels,
            shard_start=shard_start,
            shard_stop=shard_stop,
            epsilon=self.final_norm.variance_epsilon,
            out=self._fused_ce_packed_scratch(rows, hidden_size),
        )
        return PreparedCELocal(
            packed=packed,
            norm_input=norm_input,
            local_expected=local_expected,
            rows=rows,
            hidden_size=hidden_size,
        )

    def reduce_fused_ce_local(
        self,
        local: PreparedCELocal,
        *,
        process_group=None,
    ) -> PreparedCEReduction:
        """Reduce a prepared local CE contribution in caller stream order.

        The production TP helper is out-of-place. Generic fallbacks may return
        the packed input, so an alias is cloned before the grow-only scratch is
        reused by a later projection.
        """

        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        packed = local.packed
        if process_group is None:
            reduced = tensor_model_parallel_all_reduce(packed)
        else:
            # A dedicated online-learning communicator permits this collective
            # to run on the training stream without interleaving with serving
            # collectives.  Clone before the in-place NCCL sum so the grow-only
            # pack scratch can be reused by the next queued training block.
            reduced = packed.clone()
            torch.distributed.all_reduce(reduced, group=process_group)
        if (
            reduced.shape != packed.shape
            or reduced.dtype != packed.dtype
            or reduced.device != packed.device
            or not reduced.is_contiguous()
        ):
            raise RuntimeError(
                "fused CE TP reduction returned an incompatible tensor"
            )
        # In-place process-group fallbacks return ``packed`` itself.  Clone on
        # the ordered serving stream so the next pack cannot overwrite data
        # that the asynchronous training stream has not consumed yet.
        if torch._C._overlaps(reduced, packed):
            reduced = reduced.clone()
        return PreparedCEReduction(
            reduced=reduced,
            norm_input=local.norm_input,
            local_expected=local.local_expected,
            rows=local.rows,
            hidden_size=local.hidden_size,
        )

    def prepare_fused_ce_reduction(
        self,
        ticket: OnlineMTPActivationTicket,
        *,
        process_group=None,
    ) -> PreparedCEReduction:
        """Prepare and reduce CE without an inter-step pipeline."""

        return self.reduce_fused_ce_local(
            self.prepare_fused_ce_local(ticket),
            process_group=process_group,
        )

    def finish_fused_ce_tail(
        self,
        ticket: OnlineMTPActivationTicket,
        reduction: PreparedCEReduction,
    ) -> PreparedCETail:
        """Finish CE/RMSNorm locally on the caller's backward stream."""

        if ticket.state is not TicketState.VERIFIED:
            raise RuntimeError("only verified tickets can finish fused CE")
        if not self.use_triton_ce_rmsnorm:
            raise RuntimeError("fused CE reduction requires Triton CE/RMSNorm")
        from sglang.srt.speculative.triton_ops.online_mtp_ce_rmsnorm import (
            online_mtp_precomputed_ce_rmsnorm_backward,
        )

        mask = (
            None
            if ticket.loss_mask is None
            else ticket.loss_mask.reshape(-1).to(torch.bool)
        )
        step_ids = sorted(
            int(name.split(".", 1)[0][5:])
            for name in ticket.tensors
            if name.endswith(".mlp_input")
        )
        prefixes = [f"step_{step_id}." for step_id in step_ids]

        def concat(name: str) -> torch.Tensor:
            return concatenate_adjacent_rows(
                ticket.tensors[prefix + name] for prefix in prefixes
            )

        rows = reduction.rows
        hidden_size = reduction.hidden_size
        norm_input = reduction.norm_input
        local_expected = reduction.local_expected
        if (
            norm_input.shape != (rows, hidden_size)
            or local_expected.shape != norm_input.shape
        ):
            raise ValueError("fused CE reduction hidden inputs are invalid")
        logical_elements = rows * hidden_size + rows
        if reduction.reduced.numel() < logical_elements:
            raise ValueError("fused CE reduction output is too small")
        grad_hidden = reduction.reduced[: rows * hidden_size].view(
            rows, hidden_size
        )
        target_logits = reduction.reduced[
            rows * hidden_size : logical_elements
        ]
        logsumexp = concat("ce_logsumexp")
        if logsumexp.shape != (rows,):
            raise ValueError("fused CE logsumexp row count is invalid")

        # Both CE buffers are dead after this point.  The specialized tail
        # consumes the already reduced FP32 hidden gradient and target logits
        # directly, writes RMSNorm dx into the BF16 local-expected storage, and
        # accumulates dweight into the persistent gradient.  This removes the
        # former LSE rewrite, target zero-fill, temporary dweight allocation,
        # and follow-up accumulator add without changing any training rows.
        inv_out, loss_out = self._ce_tail_scratch(norm_input)
        grad_mlp_output, losses = online_mtp_precomputed_ce_rmsnorm_backward(
            norm_input,
            grad_hidden,
            self.final_norm.weight.detach(),
            logsumexp,
            target_logits,
            self.accumulator.gradients["final_norm.weight"],
            epsilon=self.final_norm.variance_epsilon,
            loss_mask=mask,
            grad_input_out=local_expected,
            inv_rms_out=inv_out,
            losses_out=loss_out,
        )
        active_tokens = rows if mask is None else int(mask.sum().item())
        return PreparedCETail(
            grad_mlp_output=grad_mlp_output,
            mean_loss=losses.sum() / max(1, active_tokens),
            norm_gradient=None,
            active_tokens=active_tokens,
        )

    def backward_ticket(
        self,
        ticket: OnlineMTPActivationTicket,
        *,
        target_weight: Optional[torch.Tensor] = None,
        ce_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prepared_ce_reduction: Optional[PreparedCEReduction] = None,
    ) -> torch.Tensor:
        if ticket.state is not TicketState.VERIFIED:
            raise RuntimeError("only verified tickets can run backward")
        assert ticket.labels is not None
        labels = ticket.labels.reshape(-1)
        mask = (
            None
            if ticket.loss_mask is None
            else ticket.loss_mask.reshape(-1).to(torch.bool)
        )
        step_ids = sorted(
            int(name.split(".", 1)[0][5:])
            for name in ticket.tensors
            if name.endswith(".mlp_input")
        )
        prefixes = [f"step_{step_id}." for step_id in step_ids]

        def concat(name: str) -> torch.Tensor:
            return concatenate_adjacent_rows(
                ticket.tensors[prefix + name] for prefix in prefixes
            )

        num_rows = sum(
            ticket.tensors[prefix + "mlp_input"].shape[0] for prefix in prefixes
        )
        if num_rows != labels.numel():
            raise RuntimeError(
                f"ticket has {num_rows} rows but {labels.numel()} labels"
            )

        grad_norm_weight = None
        prepared_ce: Optional[PreparedCETail] = None
        activation_workspace: Optional[torch.Tensor] = None
        if prepared_ce_reduction is not None:
            if target_weight is not None or ce_stats is not None:
                raise ValueError("prepared CE cannot be combined with CE inputs")
            prepared_ce = self.finish_fused_ce_tail(
                ticket, prepared_ce_reduction
            )
            grad_mlp_output = prepared_ce.grad_mlp_output
            active_tokens = prepared_ce.active_tokens
            grad_norm_weight = prepared_ce.norm_gradient
            mean_loss = prepared_ce.mean_loss
            # The CE/RMSNorm tail has consumed this ticket tensor completely.
            # Its contiguous storage can hold the smaller SwiGLU activation.
            activation_workspace = prepared_ce_reduction.norm_input
            hidden_size = ticket.tensors[prefixes[0] + "mlp_input"].shape[-1]
            if grad_mlp_output.shape != (num_rows, hidden_size):
                raise ValueError("prepared CE gradient shape does not match ticket")
        else:
            if target_weight is None:
                target_weight = vocab_parallel_target_weight(self.lm_head, labels)
            norm_input = concat("final_norm_input")
            activation_workspace = norm_input
            if ce_stats is None:
                logsumexp = concat("ce_logsumexp")
                expected_weight = concat("ce_expected_weight")
            else:
                logsumexp, expected_weight = ce_stats
                if (
                    logsumexp.numel() != num_rows
                    or expected_weight.shape != norm_input.shape
                ):
                    raise ValueError("deferred CE statistics do not match ticket rows")
            if mask is None:
                active_tokens = num_rows
            else:
                active_tokens = int(mask.sum().item())
        if prepared_ce is None and self.use_triton_ce_rmsnorm:
            from sglang.srt.speculative.triton_ops.online_mtp_ce_rmsnorm import (
                online_mtp_ce_rmsnorm_backward,
            )

            inv_out, loss_out = self._ce_tail_scratch(norm_input)
            grad_mlp_output, losses = online_mtp_ce_rmsnorm_backward(
                norm_input,
                expected_weight,
                target_weight,
                self.final_norm.weight.detach(),
                logsumexp,
                self.accumulator.gradients["final_norm.weight"],
                epsilon=self.final_norm.variance_epsilon,
                loss_mask=mask,
                inv_rms_out=inv_out,
                losses_out=loss_out,
                inplace_ce_buffers=True,
            )
        elif prepared_ce is None:
            effective_norm_weight = self.final_norm.weight.detach().float() + 1.0
            inv_rms = torch.rsqrt(
                norm_input.float().square().mean(dim=-1, keepdim=True)
                + self.final_norm.variance_epsilon
            )
            # Reconstruct only the BF16 hidden needed for the reported loss.
            final_hidden = (
                norm_input.float() * inv_rms * effective_norm_weight
            ).to(norm_input.dtype)
            target_logits = (
                final_hidden.float() * target_weight.float()
            ).sum(dim=-1)
            losses = (logsumexp.float() - target_logits).clamp_min_(0.0)
            row_scale = (
                torch.ones_like(losses) if mask is None else mask.float()
            )
            if mask is not None:
                losses = losses * mask
            grad_final_hidden = (
                expected_weight.float() - target_weight.float()
            ) * row_scale[:, None]
            grad_mlp_output, grad_norm_weight = rms_norm_backward(
                grad_final_hidden,
                norm_input,
                effective_norm_weight,
                inv_rms,
            )

        mlp_input = concat("mlp_input")
        if self.recompute_gate_up:
            # The Qwen3.6 TP8 dense MLP projection is bitwise row/batch
            # invariant on the production cuBLAS path. Recompute one combined
            # projection instead of retaining one BF16 intermediate per token.
            gate_up, _ = self.mlp.gate_up_proj(mlp_input)
        else:
            gate_up = concat("mlp_gate_up")
        # Reuse the same fused inference activation when available. This
        # removes another saved BF16 tensor while preserving the forward's
        # rounding behavior.
        intermediate_size = gate_up.shape[-1] // 2
        activation_elements = num_rows * intermediate_size
        if (
            activation_workspace is not None
            and activation_workspace.is_contiguous()
            and activation_workspace.dtype == gate_up.dtype
            and activation_workspace.device == gate_up.device
            and activation_workspace.numel() >= activation_elements
            and activation_workspace.data_ptr() % 16 == 0
            # The CUDA inference kernel vectorizes one 16-byte lane.  Tiny
            # synthetic unit-test shapes use the ordinary allocating fallback.
            and (
                not gate_up.is_cuda
                or intermediate_size % (16 // gate_up.element_size()) == 0
            )
        ):
            activated = activation_workspace.reshape(-1)[
                :activation_elements
            ].view(num_rows, intermediate_size)
            silu_and_mul_forward_out(gate_up, activated)
        else:
            activated = (
                self.mlp.act_fn(gate_up)
                if hasattr(self.mlp, "act_fn")
                else silu_and_mul_forward(gate_up)
            )
        context = DenseSwiGLUContext(
            x=mlp_input,
            gate_up=gate_up,
            activated=activated,
        )
        if self.factor_accumulator is None:
            dense_swiglu_backward_accumulate_mixed_precision(
                grad_mlp_output,
                context,
                self.mlp.gate_up_proj.weight,
                self.mlp.down_proj.weight,
                self.accumulator.gradients["mlp.gate_up_proj.weight"],
                self.accumulator.gradients["mlp.down_proj.weight"],
                use_triton_swiglu=self.use_triton_swiglu,
            )
        else:
            factor_grad_output, factor_grad_gate_up = (
                dense_swiglu_backward_factors_mixed_precision(
                    grad_mlp_output,
                    context,
                    self.mlp.down_proj.weight,
                )
            )
            factor_rows = self.factor_accumulator.append(
                context,
                factor_grad_output,
                factor_grad_gate_up,
                loss_mask=mask,
            )
            if factor_rows != active_tokens:
                raise RuntimeError(
                    f"stored {factor_rows} gradient-factor rows for "
                    f"{active_tokens} active tokens"
                )
        if active_tokens:
            if grad_norm_weight is not None:
                self.accumulator.gradients["final_norm.weight"].add_(
                    grad_norm_weight.to(
                        self.accumulator.gradients["final_norm.weight"].dtype
                    )
                )
            self.accumulator.tokens += active_tokens
        return mean_loss if prepared_ce is not None else losses.sum() / max(1, active_tokens)

    def tensor_parallel_grad_norm(self) -> float:
        """Global norm of sharded MLP grads plus one replicated norm grad."""

        from sglang.srt.distributed import (
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_reduce,
        )

        tp_size = get_tensor_model_parallel_world_size()
        norm_sq = self.accumulator.scaled_gradient_norm_sq(
            {"final_norm.weight": 1.0 / tp_size}
        )
        norm_sq = tensor_model_parallel_all_reduce(norm_sq)
        return math.sqrt(float(norm_sq.item()))


class OnlineMTPRuntime:
    """Non-blocking admission and verified-gradient scheduling for one worker."""

    def __init__(
        self,
        model,
        *,
        activation_bytes: int,
        max_tickets: int,
        backward_batch_tokens: int,
        accumulation_tokens: int,
        backward_max_group_tickets: Optional[int] = None,
        defer_ce: bool = False,
        eager_only: bool = False,
        cuda_graph_capture: bool = False,
        factorized_gradient_accumulation: bool = False,
        recompute_gate_up: bool = False,
        use_triton_lse: bool = False,
        use_triton_expcast: bool = False,
        group_ce_projection: bool = False,
        group_ce_steps: bool = False,
        fuse_ce_argmax: bool = False,
        local_vocab_ce: bool = False,
        defer_grouped_ce_projection: bool = False,
        fused_ce_reduction: bool = False,
        async_ce_producer_max_rows: int = 0,
        async_deferred_ce_reduction: bool = False,
        pipelined_ce_group_tokens: int = 0,
        fp8_ce_projection: bool = False,
        use_triton_swiglu: bool = False,
        use_triton_ce_rmsnorm: bool = False,
        use_triton_optimizer: bool = False,
        direct_optimizer_publish: bool = False,
        apply_updates: bool = False,
        async_backward: bool = True,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
    ) -> None:
        if backward_batch_tokens <= 0:
            raise ValueError("backward_batch_tokens must be positive")
        if async_ce_producer_max_rows < 0:
            raise ValueError("async CE producer max rows cannot be negative")
        if pipelined_ce_group_tokens < 0:
            raise ValueError("pipelined CE group tokens cannot be negative")
        if (
            backward_max_group_tickets is not None
            and backward_max_group_tickets <= 0
        ):
            raise ValueError("backward_max_group_tickets must be positive")
        if defer_ce and group_ce_projection:
            raise ValueError("deferred CE and grouped CE cannot be enabled together")
        if fused_ce_reduction and not group_ce_projection:
            raise ValueError("fused CE reduction requires grouped CE projection")
        if group_ce_steps:
            if not group_ce_projection:
                raise ValueError("step-grouped CE requires grouped CE projection")
            if not use_triton_lse or not use_triton_expcast:
                raise ValueError(
                    "step-grouped CE requires Triton LSE and probability"
                )
            if defer_ce or async_ce_producer_max_rows:
                raise ValueError(
                    "step-grouped CE cannot combine with a logits CE producer"
                )
        if fuse_ce_argmax:
            if not group_ce_projection or not use_triton_lse:
                raise ValueError(
                    "fused CE argmax requires grouped CE and Triton LSE"
                )
            if group_ce_steps:
                raise ValueError("fused CE argmax cannot combine with step grouping")
        if local_vocab_ce:
            if not fuse_ce_argmax or not use_triton_expcast:
                raise ValueError(
                    "local-vocabulary CE requires fused argmax and Triton probability"
                )
            if defer_ce or group_ce_steps or async_ce_producer_max_rows:
                raise ValueError(
                    "local-vocabulary CE cannot combine with deferred/async logits CE"
                )
        if defer_grouped_ce_projection and not group_ce_projection:
            raise ValueError(
                "deferred grouped CE projection requires grouped CE projection"
            )
        if defer_grouped_ce_projection and not fused_ce_reduction:
            raise ValueError(
                "deferred grouped CE projection requires fused CE reduction"
            )
        if fused_ce_reduction and not use_triton_ce_rmsnorm:
            raise ValueError("fused CE reduction requires Triton CE/RMSNorm")
        if fp8_ce_projection and not group_ce_projection:
            raise ValueError("FP8 CE projection requires grouped CE projection")
        if fp8_ce_projection and not use_triton_expcast:
            raise ValueError("FP8 CE projection requires the Triton probability producer")
        if fp8_ce_projection and defer_grouped_ce_projection:
            raise ValueError("FP8 CE projection cannot defer its grouped projection")
        if async_ce_producer_max_rows:
            if not async_backward:
                raise ValueError("async CE producer requires async backward")
            if defer_ce or defer_grouped_ce_projection:
                raise ValueError("async CE producer cannot combine with deferred CE modes")
            if not group_ce_projection or not fused_ce_reduction:
                raise ValueError(
                    "async CE producer requires grouped CE and fused CE reduction"
                )
            if not use_triton_lse or not use_triton_expcast:
                raise ValueError("async CE producer requires Triton LSE and probability")
            if fp8_ce_projection:
                raise ValueError("async CE producer is strict-BF16 only")
        if async_deferred_ce_reduction:
            if not async_backward:
                raise ValueError("async deferred CE reduction requires async backward")
            if not defer_grouped_ce_projection or not fused_ce_reduction:
                raise ValueError(
                    "async deferred CE reduction requires deferred grouped CE "
                    "projection and fused CE reduction"
                )
            if fp8_ce_projection:
                raise ValueError("async deferred CE reduction is strict-BF16 only")
        if pipelined_ce_group_tokens:
            if not async_deferred_ce_reduction:
                raise ValueError(
                    "pipelined CE grouping requires async deferred CE reduction"
                )
            if pipelined_ce_group_tokens < _ONLINE_MTP_BASE_BACKWARD_BATCH_TOKENS:
                raise ValueError(
                    "pipelined CE grouping cannot be below the base threshold"
                )
        self.ring = ActivationTicketRing(
            max_bytes=activation_bytes, max_tickets=max_tickets
        )
        self.trainer = OnlineMTPMLPTrainer(
            model,
            accumulation_tokens=accumulation_tokens,
            factorized_gradient_accumulation=factorized_gradient_accumulation,
            recompute_gate_up=recompute_gate_up,
            use_triton_swiglu=use_triton_swiglu,
            use_triton_ce_rmsnorm=use_triton_ce_rmsnorm,
            use_triton_optimizer=use_triton_optimizer,
            direct_optimizer_publish=direct_optimizer_publish,
        )
        if fp8_ce_projection:
            (
                self.fp8_ce_weight,
                self.fp8_ce_weight_scale,
                self.fp8_ce_probability_scale,
            ) = prepare_fp8_ce_projection(self.trainer.lm_head)
        else:
            self.fp8_ce_weight = None
            self.fp8_ce_weight_scale = None
            self.fp8_ce_probability_scale = None
        self.backward_batch_tokens = backward_batch_tokens
        self.backward_max_group_tickets = backward_max_group_tickets
        self.defer_ce = defer_ce
        self.eager_only = eager_only
        self.cuda_graph_capture = cuda_graph_capture
        self.factorized_gradient_accumulation = factorized_gradient_accumulation
        self.recompute_gate_up = recompute_gate_up
        self.use_triton_lse = use_triton_lse
        self.use_triton_expcast = use_triton_expcast
        self.group_ce_projection = group_ce_projection
        self.group_ce_steps = group_ce_steps
        self.fuse_ce_argmax = fuse_ce_argmax
        self.local_vocab_ce = local_vocab_ce
        self.defer_grouped_ce_projection = defer_grouped_ce_projection
        self.fused_ce_reduction = fused_ce_reduction
        self.async_ce_producer_max_rows = async_ce_producer_max_rows
        self.async_deferred_ce_reduction = async_deferred_ce_reduction
        self.pipelined_ce_group_tokens = pipelined_ce_group_tokens
        self.fp8_ce_projection = fp8_ce_projection
        self.use_triton_swiglu = use_triton_swiglu
        self.use_triton_ce_rmsnorm = use_triton_ce_rmsnorm
        self.use_triton_optimizer = use_triton_optimizer
        self.direct_optimizer_publish = direct_optimizer_publish
        self.apply_updates = apply_updates
        self.async_backward = async_backward
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_version = 0
        self.backward_batches = 0
        self.backward_tickets = 0
        self.max_verified_ticket_tokens = 0
        self.backpressure_events = 0
        self.optimizer_steps = 0
        self.last_grad_norm: Optional[float] = None
        self.last_loss: Optional[float] = None
        self.last_ticket_tokens = 0
        self.last_ticket_bytes = 0
        self.last_update_tokens = 0
        self._training_stream = (
            torch.cuda.Stream(device=self.trainer.lm_head.weight.device)
            if async_backward
            else None
        )
        self._online_ce_process_group = None
        if async_ce_producer_max_rows:
            from sglang.srt.distributed.parallel_state import (
                create_custom_parallel_group,
                get_tp_group,
            )

            self._online_ce_process_group = create_custom_parallel_group(
                get_tp_group().ranks, backend="nccl"
            )
            if self._online_ce_process_group is None:
                raise RuntimeError("failed to create dedicated online CE process group")
        self._pending: deque[PendingBackward] = deque()
        self._pending_ce_projection: Optional[PendingCEProjection] = None
        self._pending_update: Optional[PendingUpdate] = None
        self._update_requested = False
        self._completed_metrics: deque[Tuple[int, float]] = deque()
        self._warmup_admitted = False
        self.last_was_warmup = False
        self.last_backward_launched = False
        if local_vocab_ce:
            model.prepare_online_local_vocab_ce()
        if apply_updates:
            # Move one-off allocation, FP32 conversion and optimizer kernel
            # initialization out of the first live request.  The TP norm also
            # warms the scalar collective in the same rank order on all ranks.
            self.trainer.accumulator.materialize_optimizer_state()
            self.trainer.tensor_parallel_grad_norm()
            self.trainer.accumulator.warmup_optimizer_kernels_(
                weight_decay=self.weight_decay
            )
            torch.cuda.synchronize(self.trainer.lm_head.weight.device)

    def _reap_completed(self, *, wait: bool = False) -> None:
        """Release completed tickets without making serving wait by default."""

        while self._pending:
            pending = self._pending[0]
            if wait:
                pending.done_event.synchronize()
            elif not pending.done_event.query():
                break
            self._pending.popleft()
            self.last_loss = float(pending.mean_loss.item())
            if pending.training_batch_id > 0:
                self._completed_metrics.append(
                    (pending.training_batch_id, self.last_loss)
                )
            for ticket_id in pending.ticket_ids:
                self.ring.release(ticket_id, backwarded=True)

    def pop_completed_metrics(self) -> list[Tuple[int, float]]:
        completed = list(self._completed_metrics)
        self._completed_metrics.clear()
        return completed

    def use_async_ce_producer(self, rows: Optional[int]) -> bool:
        return bool(
            self.async_ce_producer_max_rows
            and rows is not None
            and rows <= self.async_ce_producer_max_rows
        )

    def begin_tap(
        self,
        *,
        expected_ce_steps: Optional[int] = None,
        expected_ce_rows: Optional[int] = None,
    ) -> OnlineMTPActivationTap:
        """Admit every eligible batch, applying backpressure when required."""

        self._reap_completed()
        while not self._try_publish_update():
            # A complete-data online learner cannot skip the next forward while
            # changing weight versions. Drain local backward and optimizer work,
            # then publish one complete version before admitting the batch.
            self._reap_completed(wait=True)
            if self._pending_update is not None:
                self._pending_update.done_event.synchronize()
        if self.ring.live_tickets >= self.ring.max_tickets:
            self.backpressure_events += 1
            self._reap_completed(wait=True)
        if self.ring.live_tickets >= self.ring.max_tickets:
            raise RuntimeError("online MTP ticket ring cannot admit every batch")
        return OnlineMTPActivationTap(
            weight_version=self.weight_version,
            defer_ce=self.defer_ce,
            recompute_gate_up=self.recompute_gate_up,
            use_triton_lse=self.use_triton_lse,
            use_triton_expcast=self.use_triton_expcast,
            group_ce_projection=self.group_ce_projection,
            group_ce_steps=self.group_ce_steps,
            fuse_ce_argmax=self.fuse_ce_argmax,
            local_vocab_ce=self.local_vocab_ce,
            defer_grouped_ce_projection=self.defer_grouped_ce_projection,
            # The ordinary deferred path keeps a complete 24-row ticket inline.
            # The inter-verify pipeline can also hide that full projection, so
            # include the equality boundary only for the pipelined experiment.
            defer_grouped_ce_projection_min_rows=(
                self.backward_batch_tokens
                + int(self.async_deferred_ce_reduction)
            ),
            defer_grouped_ce_projection_include_base=(
                self.async_deferred_ce_reduction
            ),
            fused_ce_reduction=self.fused_ce_reduction,
            async_ce_producer=self.use_async_ce_producer(expected_ce_rows),
            fp8_ce_weight=self.fp8_ce_weight,
            fp8_ce_weight_scale=self.fp8_ce_weight_scale,
            fp8_ce_probability_scale=self.fp8_ce_probability_scale,
            expected_ce_steps=expected_ce_steps,
        )

    def maybe_begin_warmup_tap(
        self,
        *,
        expected_ce_steps: Optional[int] = None,
        expected_ce_rows: Optional[int] = None,
    ) -> Optional[OnlineMTPActivationTap]:
        """Capture exactly one startup block for kernel warmup, never learning."""

        self._reap_completed()
        if self._warmup_admitted or self.ring.live_tickets >= self.ring.max_tickets:
            return None
        self._warmup_admitted = True
        return OnlineMTPActivationTap(
            weight_version=self.weight_version,
            warmup_only=True,
            defer_ce=self.defer_ce,
            recompute_gate_up=self.recompute_gate_up,
            use_triton_lse=self.use_triton_lse,
            use_triton_expcast=self.use_triton_expcast,
            group_ce_projection=self.group_ce_projection,
            group_ce_steps=self.group_ce_steps,
            fuse_ce_argmax=self.fuse_ce_argmax,
            local_vocab_ce=self.local_vocab_ce,
            defer_grouped_ce_projection=self.defer_grouped_ce_projection,
            defer_grouped_ce_projection_min_rows=(
                self.backward_batch_tokens
                + int(self.async_deferred_ce_reduction)
            ),
            defer_grouped_ce_projection_include_base=(
                self.async_deferred_ce_reduction
            ),
            fused_ce_reduction=self.fused_ce_reduction,
            async_ce_producer=self.use_async_ce_producer(expected_ce_rows),
            fp8_ce_weight=self.fp8_ce_weight,
            fp8_ce_weight_scale=self.fp8_ce_weight_scale,
            fp8_ce_probability_scale=self.fp8_ce_probability_scale,
            expected_ce_steps=expected_ce_steps,
        )

    def _try_publish_update(self) -> bool:
        """Advance an optimizer update without blocking the serving CPU."""

        pending = self._pending_update
        if pending is not None:
            if not pending.done_event.query():
                return False
            if self.ring.live_tickets:
                raise RuntimeError("cannot publish MTP weights with live tickets")
            # The query avoids blocking the CPU; the device-side wait is kept
            # as the formal cross-stream dependency for parameter copies.
            torch.cuda.current_stream().wait_event(pending.done_event)
            self.trainer.accumulator.publish_()
            self._refresh_derived_weights()
            self.weight_version += 1
            self.optimizer_steps += 1
            self.last_grad_norm = pending.grad_norm
            self.last_update_tokens = pending.update_tokens
            self._pending_update = None
            return True

        if not self._update_requested:
            return True
        # Every TP rank enters this scheduler boundary in the same model-step
        # order.  A locally unfinished rank returns to begin_tap(), which waits
        # for its training event before retrying; a faster rank can only reach
        # the following TP grad-norm collective and wait there.  No inference
        # collective can interleave at this boundary, so a separate CPU/Gloo
        # readiness consensus duplicated synchronization without adding an
        # ordering guarantee.
        if self._pending:
            return False
        if self.ring.live_tickets:
            raise RuntimeError("cannot prepare MTP weights with live tickets")

        self.trainer.materialize_factorized_gradients()
        global_grad_norm = self.trainer.tensor_parallel_grad_norm()
        update_tokens = self.trainer.accumulator.tokens
        self._update_requested = False
        if self._training_stream is None:
            grad_norm = self.trainer.accumulator.apply_(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                max_grad_norm=1.0,
                grad_norm_override=global_grad_norm,
            )
            self._refresh_derived_weights()
            self.weight_version += 1
            self.optimizer_steps += 1
            self.last_grad_norm = grad_norm
            self.last_update_tokens = update_tokens
            return True

        serving_done = torch.cuda.Event()
        serving_done.record(torch.cuda.current_stream())
        with torch.cuda.stream(self._training_stream):
            self._training_stream.wait_event(serving_done)
            grad_norm = self.trainer.accumulator.prepare_(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                max_grad_norm=1.0,
                grad_norm_override=global_grad_norm,
            )
            done_event = torch.cuda.Event()
            done_event.record(self._training_stream)
        self._pending_update = PendingUpdate(
            done_event=done_event,
            grad_norm=grad_norm,
            update_tokens=update_tokens,
        )
        return False

    def _refresh_derived_weights(self) -> None:
        """Refresh inference-only derived tensors after either publish path."""

        gemma_weight = getattr(self.trainer.final_norm, "gemma_weight", None)
        if gemma_weight is not None:
            torch.add(
                self.trainer.final_norm.weight,
                1.0,
                out=gemma_weight,
            )

    def _launch_async_ce_producer(self, ticket: OnlineMTPActivationTicket) -> None:
        """Queue label-independent strict-BF16 CE work during target verify."""

        if not bool(ticket.metadata.get("async_ce_producer", False)):
            return
        if self._training_stream is None or self._online_ce_process_group is None:
            raise RuntimeError("async CE producer is missing its training stream/group")
        step_ids = sorted(
            int(name.split(".", 1)[0][5:])
            for name in ticket.tensors
            if name.endswith(".mlp_input")
        )
        logits = [ticket.tensors[f"step_{step_id}.ce_logits"] for step_id in step_ids]
        serving_done = torch.cuda.Event()
        serving_done.record(torch.cuda.current_stream())
        with torch.cuda.stream(self._training_stream):
            self._training_stream.wait_event(serving_done)
            logsumexp, local_expected = vocab_parallel_ce_forward_local_grouped(
                logits,
                self.trainer.lm_head,
                use_triton_lse=self.use_triton_lse,
                use_triton_expcast=self.use_triton_expcast,
            )
            offset = 0
            for step_id, step_lse in zip(step_ids, logsumexp):
                rows = ticket.tensors[f"step_{step_id}.mlp_input"].shape[0]
                ticket.tensors[f"step_{step_id}.ce_logsumexp"] = step_lse
                ticket.tensors[f"step_{step_id}.ce_local_expected_weight"] = (
                    local_expected[offset : offset + rows]
                )
                offset += rows
            if offset != local_expected.shape[0]:
                raise RuntimeError("async CE expected-vector row packing failed")
            ready = torch.cuda.Event()
            ready.record(self._training_stream)
        ticket.ce_ready_event = ready

    @staticmethod
    def _wait_and_retire_async_ce_logits(
        tickets: Iterable[OnlineMTPActivationTicket],
    ) -> None:
        async_tickets = [
            ticket
            for ticket in tickets
            if bool(ticket.metadata.get("async_ce_producer", False))
        ]
        if not async_tickets:
            return
        serving_stream = torch.cuda.current_stream()
        for ticket in async_tickets:
            if ticket.ce_ready_event is None:
                raise RuntimeError("async CE ticket has no producer event")
            serving_stream.wait_event(ticket.ce_ready_event)
            for name in [name for name in ticket.tensors if name.endswith(".ce_logits")]:
                del ticket.tensors[name]

    def stage(self, tap: OnlineMTPActivationTap) -> Optional[int]:
        if not tap.steps:
            return None
        tap.validate_mlp_complete()
        tensors = tap.flatten()
        # Reserve the fused-CE-specific asynchronous working set at admission.
        # This is accounting only: buffers are allocated lazily by
        # prepare/backward.  It conservatively covers the retained reduction
        # and the two possible hidden concatenations, but ONLINE_ACTIVATION_MB
        # remains a ticket-admission budget rather than a total CUDA-memory cap.
        if tap.fused_ce_reduction:
            reserved_extra_bytes = (
                fused_ce_transient_reservation_bytes(
                    num_tokens=tap.num_tokens,
                    hidden_size=self.trainer.lm_head.weight.shape[1],
                    activation_element_size=(
                        self.trainer.lm_head.weight.element_size()
                    ),
                )
            )
        elif tap.defer_ce:
            # Deferred CE creates one FP32 expected and target hidden vector
            # per row after verification.
            reserved_extra_bytes = (
                tap.num_tokens
                * self.trainer.lm_head.weight.shape[1]
                * self.trainer.lm_head.weight.element_size()
            )
        else:
            reserved_extra_bytes = (
                tap.num_tokens
                * self.trainer.lm_head.weight.shape[1]
                * self.trainer.lm_head.weight.element_size()
            )
        bytes_reserved = sum(tensor_nbytes(value) for value in tensors.values())
        bytes_reserved += reserved_extra_bytes
        self.last_ticket_bytes = bytes_reserved
        if bytes_reserved > self.ring.max_bytes:
            raise RuntimeError(
                "one online MTP batch requires "
                f"{bytes_reserved} activation bytes, exceeding the configured "
                f"ring capacity {self.ring.max_bytes}"
            )
        while not self.ring.can_stage(bytes_reserved=bytes_reserved):
            if not self._pending:
                raise RuntimeError(
                    "online MTP activation ring is full without pending backward work"
                )
            # Preserve every token. Memory pressure becomes measured serving
            # backpressure instead of silently dropping the learning batch.
            self.backpressure_events += 1
            self._reap_completed(wait=True)
        ticket = self.ring.try_stage(
            tensors,
            weight_version=tap.weight_version,
            num_tokens=tap.num_tokens,
            # The tap already owns clones from the inference forward.
            clone=False,
            reserved_extra_bytes=reserved_extra_bytes,
            ready_event=tap.ready_event,
            metadata={
                "num_steps": len(tap.steps),
                "warmup_only": tap.warmup_only,
                "defer_ce": tap.defer_ce,
                "fused_ce_reduction": tap.fused_ce_reduction,
                "defer_grouped_ce_projection": (
                    tap.deferred_grouped_ce_projection_active
                ),
                "async_ce_producer": tap.async_ce_producer,
            },
        )
        if ticket is None:
            raise RuntimeError("online MTP admission changed after capacity check")
        self._launch_async_ce_producer(ticket)
        return ticket.ticket_id

    def snapshot_cuda_graph_tap(
        self,
        destination: OnlineMTPActivationTap,
        template: OnlineMTPActivationTap,
        *,
        num_rows: int,
    ) -> None:
        """Queue a ticket-owned snapshot of one learning-graph replay."""

        if not self.cuda_graph_capture:
            raise RuntimeError("online MTP CUDA-graph capture is not enabled")
        destination.copy_from_graph_template(
            template,
            num_rows=num_rows,
        )

    def verify_and_backward(
        self,
        ticket_id: int,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Optional[float]:
        # Target verification has now completed on the ordered serving stream.
        # Finish the previous block's local projection here: its large frozen
        # LM-head read has overlapped one complete draft+target iteration, but
        # its TP collective still retains the normal inference communicator
        # order.  No second NCCL communicator is needed.
        self._finish_pending_ce_projection()
        ticket = self.ring.get(ticket_id)
        if ticket.weight_version != self.weight_version:
            self.ring.release(ticket.ticket_id)
            raise RuntimeError(
                "online MTP weights changed between draft forward and backward"
            )
        warmup_only = bool(ticket.metadata.get("warmup_only", False))
        self.ring.mark_verified(ticket_id, labels, loss_mask)
        if not warmup_only:
            self.max_verified_ticket_tokens = max(
                self.max_verified_ticket_tokens,
                ticket.num_tokens,
            )
        self.last_was_warmup = warmup_only
        self.last_backward_launched = False
        loss = self._maybe_launch_backward_group(
            force=warmup_only or self.trainer.accumulator.ready
        )
        if warmup_only and not self.last_backward_launched:
            raise RuntimeError("online MTP startup warmup did not launch backward")
        if warmup_only:
            # Complete the real-shape training-stream work before the server is
            # declared ready, then discard every accumulated gradient.  This
            # warms the exact backward without learning from synthetic startup
            # traffic or advancing the optimizer/weight version.
            self._reap_completed(wait=True)
            self.trainer.zero_accumulation()
            self._completed_metrics.clear()
            self.last_loss = None
            return None
        return loss

    def _finish_pending_ce_projection(self) -> None:
        pending = self._pending_ce_projection
        if pending is None:
            return
        if self._training_stream is None:
            raise RuntimeError("pipelined CE projection requires async backward")

        serving_stream = torch.cuda.current_stream()
        serving_stream.wait_event(pending.ready_event)
        reduction = self.trainer.reduce_fused_ce_local(pending.local)
        owner = pending.tickets[0]
        owner.tensors["fused_ce_reduced"] = reduction.reduced
        owner.tensors["fused_ce_norm_input"] = reduction.norm_input
        owner.tensors["fused_ce_local_expected"] = reduction.local_expected

        producer_done = torch.cuda.Event()
        producer_done.record(serving_stream)
        with torch.cuda.stream(self._training_stream):
            self._training_stream.wait_event(producer_done)
            loss_tensor = self.trainer.backward_ticket(
                pending.combined,
                prepared_ce_reduction=reduction,
            )
            done_event = torch.cuda.Event()
            done_event.record(self._training_stream)
        self._pending.append(
            PendingBackward(
                tuple(ticket.ticket_id for ticket in pending.tickets),
                pending.training_batch_id,
                done_event,
                loss_tensor,
            )
        )
        self._pending_ce_projection = None

    def _maybe_launch_backward_group(self, *, force: bool = False) -> Optional[float]:
        if not self.ring.ready_tickets:
            return None
        effective_batch_tokens = self.backward_batch_tokens
        use_base_threshold = (
            self.max_verified_ticket_tokens <= _ONLINE_MTP_BASE_BACKWARD_BATCH_TOKENS
            or self.max_verified_ticket_tokens >= self.backward_batch_tokens
        )
        if use_base_threshold:
            # Tiny tickets are already launch-balanced at the established
            # 24-token threshold, while a single large steady-state ticket
            # already fills the configured target. Cross-ticket grouping is
            # useful only between those boundaries (C=32 in the standard
            # curve), and otherwise creates avoidable decode-facing bursts.
            if (
                self.pipelined_ce_group_tokens
                and self.max_verified_ticket_tokens
                < _ONLINE_MTP_BASE_BACKWARD_BATCH_TOKENS
            ):
                effective_batch_tokens = self.pipelined_ce_group_tokens
            else:
                effective_batch_tokens = min(
                    effective_batch_tokens,
                    _ONLINE_MTP_BASE_BACKWARD_BATCH_TOKENS,
                )
        # The ticket cap belongs only to the larger cross-ticket grouping path.
        # At the base threshold, token count alone reproduces the established
        # C=1/C=8 scheduling and cannot fire early for an unusual ticket size.
        effective_ticket_limit = (
            None if use_base_threshold else self.backward_max_group_tickets
        )
        ticket_limit_reached = (
            effective_ticket_limit is not None
            and self.ring.ready_tickets >= effective_ticket_limit
        )
        if (
            not force
            and self.ring.ready_tokens < effective_batch_tokens
            and self.ring.ready_tickets < self.ring.max_tickets
            and not ticket_limit_reached
        ):
            return None

        tickets = self.ring.pop_all_verified()
        self._wait_and_retire_async_ce_logits(tickets)
        combined = combine_verified_tickets(tickets)
        # Production snapshots are already ordered on the serving stream.  A
        # diagnostic caller may instead supply a dedicated copy stream; put
        # that optional dependency on the serving stream before *any* grouped
        # concatenation, CE projection or TP collective reads ticket data.
        # The later producer event then transfers this whole dependency chain
        # to the single training stream without a host synchronization.
        ready_events = [
            ticket.ready_event
            for ticket in tickets
            if ticket.ready_event is not None
        ]
        if ready_events:
            serving_stream = torch.cuda.current_stream()
            for ready_event in ready_events:
                serving_stream.wait_event(ready_event)
        warmup_only = bool(tickets[0].metadata.get("warmup_only", False))
        if any(
            bool(ticket.metadata.get("warmup_only", False)) != warmup_only
            for ticket in tickets
        ):
            raise RuntimeError("startup warmup cannot share a backward group")
        training_batch_id = 0 if warmup_only else self.backward_batches + 1
        self.last_ticket_tokens = combined.num_tokens
        self.last_backward_launched = True
        ce_stats = None
        prepared_ce_reduction = None
        if bool(combined.metadata.get("defer_ce", False)):
            step_ids = sorted(
                int(name.split(".", 1)[0][5:])
                for name in combined.tensors
                if name.endswith(".mlp_input")
            )
            logits = torch.cat(
                [combined.tensors[f"step_{step_id}.ce_logits"] for step_id in step_ids]
            )
            ce_stats = vocab_parallel_ce_forward_stats(
                logits,
                self.trainer.lm_head,
                use_triton_lse=self.use_triton_lse,
                use_triton_expcast=self.use_triton_expcast,
            )
            tickets[0].tensors["deferred_ce_logsumexp"] = ce_stats[0]
            tickets[0].tensors["deferred_ce_expected_weight"] = ce_stats[1]
        async_ce_producer = bool(
            combined.metadata.get("async_ce_producer", False)
        )
        # Tiny tickets can retain their BF16 probability shards until several
        # tickets form one launch-balanced group. Pipeline that group's local
        # 318-MB LM-head read with the next inference iteration; the following
        # verify callback performs the small TP sum in normal serving order.
        pipelined_deferred_ce = (
            self.async_deferred_ce_reduction
            and bool(
                combined.metadata.get("defer_grouped_ce_projection", False)
            )
            and not warmup_only
            and not self.trainer.accumulator.ready
        )
        async_ce_reduction = async_ce_producer
        if (
            bool(combined.metadata.get("fused_ce_reduction", False))
            and not async_ce_reduction
            and not pipelined_deferred_ce
        ):
            prepared_ce_reduction = self.trainer.prepare_fused_ce_reduction(
                combined
            )
        if self._training_stream is None:
            loss_tensor = self.trainer.backward_ticket(
                combined,
                ce_stats=ce_stats,
                prepared_ce_reduction=prepared_ce_reduction,
            )
            loss = float(loss_tensor.item())
            self.last_loss = loss
            if training_batch_id:
                self._completed_metrics.append((training_batch_id, loss))
            for ticket in tickets:
                self.ring.release(ticket.ticket_id, backwarded=True)
        else:
            # TP collectives retain their normal serving-stream order.  The
            # queued backward itself is local-only and can safely overlap the
            # next inference batch.  The ticket remains alive until its event
            # completes, so the allocator cannot recycle captured activations.
            target_weight = None
            if combined.loss_mask is not None:
                # ``combine_verified_tickets`` creates this tensor on the
                # serving stream, while both CE and factor accumulation may
                # consume it on the training stream.
                tickets[0].tensors["combined_loss_mask"] = combined.loss_mask
            if (
                prepared_ce_reduction is None
                and not async_ce_reduction
                and not pipelined_deferred_ce
            ):
                target_weight = vocab_parallel_target_weight(
                    self.trainer.lm_head, combined.labels
                )
                # This tensor is allocated on the serving stream and consumed
                # on the training stream. Retain it until the done event.
                tickets[0].tensors["target_weight"] = target_weight
            elif not async_ce_reduction and not pipelined_deferred_ce:
                # The reduction and any combined hidden owners were allocated
                # on the serving stream and are read/written by the training
                # stream.  Bind every cross-stream allocation to a live ring
                # ticket; release happens only after ``done_event`` completes.
                tickets[0].tensors["fused_ce_reduced"] = (
                    prepared_ce_reduction.reduced
                )
                tickets[0].tensors["fused_ce_norm_input"] = (
                    prepared_ce_reduction.norm_input
                )
                tickets[0].tensors["fused_ce_local_expected"] = (
                    prepared_ce_reduction.local_expected
                )
            producer_done = torch.cuda.Event()
            producer_done.record(torch.cuda.current_stream())
            with torch.cuda.stream(self._training_stream):
                self._training_stream.wait_event(producer_done)
                if pipelined_deferred_ce:
                    if self._pending_ce_projection is not None:
                        raise RuntimeError("only one CE projection may be in flight")
                    prepared_ce_local = self.trainer.prepare_fused_ce_local(combined)
                    projection_ready = torch.cuda.Event()
                    projection_ready.record(self._training_stream)
                    loss_tensor = None
                    done_event = None
                else:
                    if async_ce_reduction:
                        if self._online_ce_process_group is None:
                            raise RuntimeError(
                                "async CE reduction has no dedicated process group"
                            )
                        prepared_ce_reduction = (
                            self.trainer.prepare_fused_ce_reduction(
                                combined,
                                process_group=self._online_ce_process_group,
                            )
                        )
                        # These tensors are created on the training stream. Keep
                        # explicit owners until queued backward has completed.
                        tickets[0].tensors["fused_ce_reduced"] = (
                            prepared_ce_reduction.reduced
                        )
                        tickets[0].tensors["fused_ce_norm_input"] = (
                            prepared_ce_reduction.norm_input
                        )
                        tickets[0].tensors["fused_ce_local_expected"] = (
                            prepared_ce_reduction.local_expected
                        )
                    loss_tensor = self.trainer.backward_ticket(
                        combined,
                        target_weight=target_weight,
                        ce_stats=ce_stats,
                        prepared_ce_reduction=prepared_ce_reduction,
                    )
                    done_event = torch.cuda.Event()
                    done_event.record(self._training_stream)
            if pipelined_deferred_ce:
                self._pending_ce_projection = PendingCEProjection(
                    tickets=tuple(tickets),
                    combined=combined,
                    training_batch_id=training_batch_id,
                    ready_event=projection_ready,
                    local=prepared_ce_local,
                )
            else:
                assert done_event is not None
                assert loss_tensor is not None
                self._pending.append(
                    PendingBackward(
                        tuple(ticket.ticket_id for ticket in tickets),
                        training_batch_id,
                        done_event,
                        loss_tensor,
                    )
                )
            loss = None
        if not warmup_only:
            self.backward_batches += 1
            self.backward_tickets += len(tickets)
        return loss

    def maybe_apply_update(self) -> Optional[float]:
        """Apply one cache-safe MLP/final-norm update at a scheduler boundary.

        The trainable tensors are all downstream of the draft attention KV
        write, so their update does not invalidate persistent MTP KV entries.
        Attention/input-projection parameters deliberately remain frozen.
        """

        if self._pending_update is not None or self._update_requested:
            return None
        if not self.trainer.accumulator.ready:
            self._reap_completed()
            return None
        if not self.apply_updates:
            # Backward-only safety mode must not accumulate forever.  Drain the
            # asynchronous writers and recycle the window at the same token
            # boundary, without allocating optimizer state or changing weights.
            self._reap_completed(wait=True)
            if self.ring.live_tickets:
                raise RuntimeError(
                    "cannot discard MTP gradients with live activation tickets"
                )
            self.last_update_tokens = self.trainer.accumulator.tokens
            self.trainer.zero_accumulation()
            return None
        # Do not synchronize the host with the just-launched backward.  The
        # next scheduler boundary polls its event, then starts TP norm/AdamW
        # only after every ticket has completed.  Until publication, draft
        # inference remains on the previous complete weight version.
        self._update_requested = True
        return None
