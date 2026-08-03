import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.online_mtp_training import (
    ActivationTicketRing,
    FlatAdamWAccumulator,
    OnlineMTPActivationTicket,
    OnlineMTPActivationTap,
    OnlineMTPMLPTrainer,
    OnlineMTPRuntime,
    TicketState,
    combine_verified_tickets,
    concatenate_adjacent_rows,
    dense_swiglu_backward,
    dense_swiglu_backward_accumulate_mixed_precision,
    dense_swiglu_backward_mixed_precision,
    dense_swiglu_forward,
    fused_ce_transient_reservation_bytes,
    rms_norm_backward,
    rms_norm_forward,
    silu_and_mul_backward_mixed_precision,
    silu_and_mul_forward_out,
    vocab_ce_forward_stats,
    vocab_ce_hidden_backward_from_stats,
    vocab_ce_hidden_backward_from_logits,
    vocab_parallel_ce_forward_local_grouped,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


class TestOnlineMTPTraining(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(7)

    def test_async_local_grouped_ce_matches_strict_reference(self):
        rows, vocab_size, hidden_size = 3, 23, 16
        logits = [
            torch.randn(rows, vocab_size, device=self.device, dtype=torch.float32)
            for _ in range(3)
        ]
        weight_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        weight = torch.randn(
            vocab_size, hidden_size, device=self.device, dtype=weight_dtype
        )
        lm_head = SimpleNamespace(
            weight=weight,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0,
                org_vocab_end_index=vocab_size,
            ),
        )
        logsumexp, expected = vocab_parallel_ce_forward_local_grouped(
            logits,
            lm_head,
            use_triton_lse=False,
            use_triton_expcast=False,
        )
        reference_lse = [torch.logsumexp(value, dim=-1) for value in logits]
        probability = torch.cat(
            [
                torch.exp(value - value_lse[:, None]).to(weight_dtype)
                for value, value_lse in zip(logits, reference_lse)
            ]
        )
        reference_expected = probability @ weight
        for actual, reference in zip(logsumexp, reference_lse):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(expected, reference_expected, rtol=0, atol=0)

    def _fused_ce_fixture(
        self,
        *,
        device=None,
        dtype=None,
        rows=5,
        hidden_size=16,
        intermediate_size=12,
        vocab_size=23,
    ):
        device = self.device if device is None else device
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                weight=torch.randn(
                    2 * intermediate_size,
                    hidden_size,
                    device=device,
                    dtype=dtype,
                )
            ),
            down_proj=SimpleNamespace(
                weight=torch.randn(
                    hidden_size,
                    intermediate_size,
                    device=device,
                    dtype=dtype,
                )
            ),
        )
        final_norm = SimpleNamespace(
            weight=torch.randn(hidden_size, device=device, dtype=dtype),
            variance_epsilon=1e-6,
        )
        lm_head = SimpleNamespace(
            weight=torch.randn(
                vocab_size, hidden_size, device=device, dtype=dtype
            ),
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0,
                org_vocab_end_index=vocab_size,
            ),
        )
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(mlp=mlp)], norm=final_norm
            ),
            lm_head=lm_head,
        )
        trainer = OnlineMTPMLPTrainer(
            model,
            accumulation_tokens=4 * rows,
            use_triton_ce_rmsnorm=True,
        )
        tensors = {
            "step_0.mlp_input": torch.randn(
                rows, hidden_size, device=device, dtype=dtype
            ),
            "step_0.mlp_gate_up": torch.randn(
                rows, 2 * intermediate_size, device=device, dtype=dtype
            ),
            "step_0.final_norm_input": torch.randn(
                rows, hidden_size, device=device, dtype=dtype
            ),
            "step_0.ce_logsumexp": torch.rand(
                rows, device=device, dtype=torch.float32
            )
            + 8.0,
            "step_0.ce_local_expected_weight": torch.randn(
                rows, hidden_size, device=device, dtype=dtype
            ),
        }
        ticket = OnlineMTPActivationTicket(
            ticket_id=0,
            weight_version=0,
            num_tokens=rows,
            tensors=tensors,
            bytes_reserved=0,
            state=TicketState.VERIFIED,
            labels=torch.randint(0, vocab_size, (rows,), device=device),
            metadata={"fused_ce_reduction": True},
        )
        return model, trainer, ticket

    def _tap_from_fused_ce_fixture(self, fixture):
        tap = OnlineMTPActivationTap(
            weight_version=fixture.weight_version,
            clone_tensors=False,
            group_ce_projection=True,
            fused_ce_reduction=True,
        )
        ids = torch.arange(
            fixture.num_tokens,
            device=fixture.labels.device,
        )
        tap.begin_step(0, input_ids=ids, positions=ids)
        for qualified_name, tensor in fixture.tensors.items():
            tap.record(qualified_name.split(".", 1)[1], tensor)
        return tap

    def test_recompute_tap_does_not_retain_gate_up(self):
        tap = OnlineMTPActivationTap(weight_version=0, recompute_gate_up=True)
        ids = torch.arange(2, device=self.device)
        tap.begin_step(0, input_ids=ids, positions=ids)
        tap.record("mlp_input", torch.randn(2, 4, device=self.device))
        tap.record("mlp_gate_up", torch.randn(2, 6, device=self.device))
        tap.record("final_norm_input", torch.randn(2, 4, device=self.device))
        tap.record("ce_logsumexp", torch.randn(2, device=self.device))
        tap.record("ce_expected_weight", torch.randn(2, 4, device=self.device))
        tap.validate_mlp_complete()
        self.assertNotIn("mlp_gate_up", tap.steps[0])

    def test_grouped_ce_projects_all_steps_once_and_keeps_adjacent_views(self):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        local_vocab, hidden_size = 257, 64
        weight = torch.randn(
            local_vocab, hidden_size, device=self.device, dtype=dtype
        )
        indices = SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=local_vocab,
        )
        lm_head = SimpleNamespace(weight=weight, shard_indices=indices)
        tap = OnlineMTPActivationTap(
            weight_version=0,
            group_ce_projection=True,
            expected_ce_steps=3,
        )
        probabilities = []
        for step, rows in enumerate((3, 5, 2)):
            ids = torch.arange(rows, device=self.device)
            tap.begin_step(step, input_ids=ids, positions=ids)
            probability = torch.rand(
                rows, local_vocab, device=self.device, dtype=dtype
            )
            probabilities.append(probability.clone())
            tap.record_ce_probability(probability)

        with (
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                side_effect=lambda value: value,
            ),
            patch("torch.cat", side_effect=AssertionError("unexpected torch.cat")),
        ):
            tap.finalize_grouped_ce(lm_head)

        grouped = torch.cat(
            [tap.steps[step]["ce_expected_weight"] for step in range(3)]
        )
        separate = torch.cat(
            [(probability @ weight).float() for probability in probabilities]
        )
        torch.testing.assert_close(grouped, separate, rtol=2e-3, atol=3e-4)
        self.assertFalse(tap._ce_probabilities)
        storage = tap.steps[0]["ce_expected_weight"].untyped_storage().data_ptr()
        for step in range(1, 3):
            self.assertEqual(
                tap.steps[step]["ce_expected_weight"].untyped_storage().data_ptr(),
                storage,
            )

    def test_step_grouped_ce_finalizer_matches_separate_steps(self):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        rows, vocab_size, local_vocab, hidden_size = 3, 41, 17, 8
        local_start = 7
        weight = torch.randn(
            local_vocab, hidden_size, device=self.device, dtype=dtype
        )
        lm_head = SimpleNamespace(
            weight=weight,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=local_start,
                org_vocab_end_index=local_start + local_vocab,
            ),
        )
        tap = OnlineMTPActivationTap(
            weight_version=0,
            use_triton_lse=True,
            use_triton_expcast=True,
            group_ce_projection=True,
            group_ce_steps=True,
            fused_ce_reduction=True,
            expected_ce_steps=3,
        )
        logits = []
        for step in range(3):
            ids = torch.arange(rows, device=self.device)
            tap.begin_step(step, input_ids=ids, positions=ids)
            value = torch.randn(
                rows, vocab_size, device=self.device, dtype=torch.float32
            )
            logits.append(value)
            tap.reserve_ce_probability(
                rows, local_vocab, dtype=dtype, device=value.device
            )
            tap.record_grouped_ce_logits(value)

        tap.finalize_grouped_ce(lm_head)
        for step, value in enumerate(logits):
            reference_lse = torch.logsumexp(value.float(), dim=-1)
            reference_probability = torch.exp(
                value[:, local_start : local_start + local_vocab].float()
                - reference_lse[:, None]
            ).to(dtype)
            torch.testing.assert_close(
                tap.steps[step]["ce_logsumexp"], reference_lse, rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                tap.steps[step]["ce_local_expected_weight"],
                reference_probability @ weight,
                rtol=2e-3 if dtype == torch.bfloat16 else 1e-6,
                atol=3e-4 if dtype == torch.bfloat16 else 1e-6,
            )
        self.assertFalse(tap._ce_grouped_logits)

    def test_fused_grouped_ce_retains_local_gemm_without_collective(self):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        local_vocab, hidden_size = 19, 8
        weight = torch.randn(
            local_vocab, hidden_size, device=self.device, dtype=dtype
        )
        lm_head = SimpleNamespace(
            weight=weight,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=37,
                org_vocab_end_index=37 + local_vocab,
            ),
        )
        tap = OnlineMTPActivationTap(
            weight_version=0,
            group_ce_projection=True,
            fused_ce_reduction=True,
            expected_ce_steps=2,
        )
        probabilities = []
        for step, rows in enumerate((2, 3)):
            ids = torch.arange(rows, device=self.device)
            tap.begin_step(step, input_ids=ids, positions=ids)
            probability = torch.rand(
                rows, local_vocab, device=self.device, dtype=dtype
            )
            probabilities.append(probability.clone())
            tap.record_ce_probability(probability)

        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_reduce",
            side_effect=AssertionError("draft path must not reduce local expected"),
        ):
            tap.finalize_grouped_ce(lm_head)

        actual = torch.cat(
            [tap.steps[step]["ce_local_expected_weight"] for step in range(2)]
        )
        reference = torch.cat(probabilities) @ weight
        self.assertEqual(actual.dtype, dtype)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertNotIn("ce_expected_weight", tap.steps[0])

    def test_deferred_grouped_ce_retains_adjacent_probability_views(self):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        local_vocab, hidden_size = 19, 8
        lm_head = SimpleNamespace(
            weight=torch.randn(
                local_vocab, hidden_size, device=self.device, dtype=dtype
            ),
            shard_indices=SimpleNamespace(
                org_vocab_start_index=37,
                org_vocab_end_index=37 + local_vocab,
            ),
        )
        tap = OnlineMTPActivationTap(
            weight_version=0,
            group_ce_projection=True,
            defer_grouped_ce_projection=True,
            fused_ce_reduction=True,
            expected_ce_steps=2,
        )
        probabilities = []
        for step, rows in enumerate((2, 3)):
            ids = torch.arange(rows, device=self.device)
            tap.begin_step(step, input_ids=ids, positions=ids)
            probability = torch.rand(
                rows, local_vocab, device=self.device, dtype=dtype
            )
            probabilities.append(probability.clone())
            tap.record_ce_probability(probability)

        tap.finalize_grouped_ce(lm_head)

        actual = torch.cat(
            [tap.steps[step]["ce_probability"] for step in range(2)]
        )
        torch.testing.assert_close(actual, torch.cat(probabilities), rtol=0, atol=0)
        self.assertNotIn("ce_local_expected_weight", tap.steps[0])
        storage = tap.steps[0]["ce_probability"].untyped_storage().data_ptr()
        self.assertEqual(
            tap.steps[1]["ce_probability"].untyped_storage().data_ptr(), storage
        )
        self.assertFalse(tap._ce_probabilities)

    def test_deferred_grouped_ce_keeps_full_base_ticket_inline(self):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        rows, local_vocab, hidden_size = 24, 19, 8
        weight = torch.randn(
            local_vocab, hidden_size, device=self.device, dtype=dtype
        )
        lm_head = SimpleNamespace(
            weight=weight,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=37,
                org_vocab_end_index=37 + local_vocab,
            ),
        )
        tap = OnlineMTPActivationTap(
            weight_version=0,
            group_ce_projection=True,
            defer_grouped_ce_projection=True,
            defer_grouped_ce_projection_min_rows=96,
            fused_ce_reduction=True,
            expected_ce_steps=1,
        )
        ids = torch.arange(rows, device=self.device)
        tap.begin_step(0, input_ids=ids, positions=ids)
        probability = torch.rand(
            rows, local_vocab, device=self.device, dtype=dtype
        )
        tap.record_ce_probability(probability)

        tap.finalize_grouped_ce(lm_head)

        self.assertFalse(tap.deferred_grouped_ce_projection_active)
        self.assertNotIn("ce_probability", tap.steps[0])
        torch.testing.assert_close(
            tap.steps[0]["ce_local_expected_weight"],
            probability @ weight,
            rtol=0,
            atol=0,
        )

    def test_pipelined_deferred_ce_includes_full_base_ticket(self):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        rows, local_vocab, hidden_size = 24, 19, 8
        lm_head = SimpleNamespace(
            weight=torch.randn(
                local_vocab, hidden_size, device=self.device, dtype=dtype
            ),
            shard_indices=SimpleNamespace(
                org_vocab_start_index=37,
                org_vocab_end_index=37 + local_vocab,
            ),
        )
        tap = OnlineMTPActivationTap(
            weight_version=0,
            group_ce_projection=True,
            defer_grouped_ce_projection=True,
            # Runtime uses backward_batch_tokens + 1 when the inter-verify
            # pipeline is enabled, making the 24-row equality boundary defer.
            defer_grouped_ce_projection_min_rows=25,
            defer_grouped_ce_projection_include_base=True,
            fused_ce_reduction=True,
            expected_ce_steps=1,
        )
        ids = torch.arange(rows, device=self.device)
        tap.begin_step(0, input_ids=ids, positions=ids)
        probability = torch.rand(
            rows, local_vocab, device=self.device, dtype=dtype
        )
        tap.record_ce_probability(probability)

        tap.finalize_grouped_ce(lm_head)

        self.assertTrue(tap.deferred_grouped_ce_projection_active)
        self.assertIn("ce_probability", tap.steps[0])
        self.assertNotIn("ce_local_expected_weight", tap.steps[0])

    def test_fused_ce_prepare_projects_deferred_probabilities_once(self):
        model, trainer, ticket = self._fused_ce_fixture(device="cpu")
        local_weight = model.lm_head.weight
        probability = torch.rand(ticket.num_tokens, local_weight.shape[0])
        expected = probability @ local_weight
        ticket.tensors.pop("step_0.ce_local_expected_weight")
        ticket.tensors["step_0.ce_probability"] = probability
        ticket.metadata["defer_grouped_ce_projection"] = True

        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value.clone(),
        ):
            reduction = trainer.prepare_fused_ce_reduction(ticket)

        torch.testing.assert_close(
            reduction.local_expected, expected, rtol=0, atol=0
        )

    def test_fused_ce_prepare_handles_mixed_inline_and_deferred_tickets(self):
        model, trainer, inline = self._fused_ce_fixture(device="cpu", rows=3)
        deferred_tensors = {
            name: torch.randn_like(value)
            for name, value in inline.tensors.items()
            if not name.endswith("ce_local_expected_weight")
        }
        probability = torch.rand(3, model.lm_head.weight.shape[0])
        deferred_tensors["step_0.ce_probability"] = probability
        deferred = OnlineMTPActivationTicket(
            ticket_id=1,
            weight_version=inline.weight_version,
            num_tokens=3,
            tensors=deferred_tensors,
            bytes_reserved=0,
            state=TicketState.VERIFIED,
            labels=torch.randint(0, model.lm_head.weight.shape[0], (3,)),
            metadata={
                "fused_ce_reduction": True,
                "defer_grouped_ce_projection": True,
            },
        )
        inline.metadata["defer_grouped_ce_projection"] = False
        expected = torch.cat(
            [
                inline.tensors["step_0.ce_local_expected_weight"],
                probability @ model.lm_head.weight,
            ]
        )
        combined = combine_verified_tickets([inline, deferred])

        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value.clone(),
        ):
            reduction = trainer.prepare_fused_ce_reduction(combined)

        self.assertTrue(combined.metadata["defer_grouped_ce_projection"])
        torch.testing.assert_close(
            reduction.local_expected, expected, rtol=0, atol=0
        )

    def test_local_ce_pack_handles_masks_and_vocab_shard_boundaries(self):
        from sglang.srt.speculative.triton_ops.online_mtp_ce_rmsnorm import (
            pack_local_ce_for_tp_reduce,
        )

        rows, hidden_size = 4, 8
        shard_start, shard_stop = 11, 16
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        local_weight = torch.randn(
            shard_stop - shard_start,
            hidden_size,
            device=self.device,
            dtype=dtype,
        )
        local_expected = torch.randn(
            rows, hidden_size, device=self.device, dtype=dtype
        )
        norm_input = torch.randn_like(local_expected)
        raw_norm_weight = torch.randn(
            hidden_size, device=self.device, dtype=dtype
        )
        # Cover the first and last owned rows plus both neighboring nonowners.
        labels = torch.tensor(
            [shard_start, shard_stop - 1, shard_start - 1, shard_stop],
            device=self.device,
        )
        logical = rows * hidden_size + rows
        out = torch.empty((logical + 3) // 4 * 4, device=self.device)
        packed = pack_local_ce_for_tp_reduce(
            local_expected,
            norm_input,
            raw_norm_weight,
            local_weight,
            labels,
            shard_start=shard_start,
            shard_stop=shard_stop,
            epsilon=1e-6,
            out=out,
        )

        target = torch.zeros_like(local_expected)
        target[0] = local_weight[0]
        target[1] = local_weight[-1]
        expected_grad = local_expected.float() - target.float()
        inv_rms = torch.rsqrt(
            norm_input.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        final_hidden = (
            norm_input.float()
            * inv_rms
            * (raw_norm_weight.float() + 1.0)
        ).to(dtype)
        expected_logits = (final_hidden.float() * target.float()).sum(dim=-1)
        torch.testing.assert_close(
            packed[: rows * hidden_size].view(rows, hidden_size),
            expected_grad,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            packed[rows * hidden_size : logical],
            expected_logits,
            rtol=0,
            atol=0,
        )
        self.assertEqual(packed.numel() * packed.element_size() % 16, 0)
        self.assertEqual(expected_logits[2:].count_nonzero().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires Triton CUDA path")
    def test_precomputed_ce_tail_is_bitwise_equal_to_existing_exact_path(self):
        from sglang.srt.speculative.triton_ops.online_mtp_ce_rmsnorm import (
            online_mtp_ce_rmsnorm_backward,
            online_mtp_precomputed_ce_rmsnorm_backward,
        )

        hidden_size = 64
        for rows in (1, 7, 24):
            for use_mask in (False, True):
                with self.subTest(rows=rows, use_mask=use_mask):
                    torch.manual_seed(1700 + rows)
                    norm_input = torch.randn(
                        rows,
                        hidden_size,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    grad_hidden = torch.randn(
                        rows, hidden_size, device="cuda", dtype=torch.float32
                    )
                    raw_weight = torch.randn(
                        hidden_size, device="cuda", dtype=torch.bfloat16
                    )
                    logsumexp = torch.rand(rows, device="cuda") + 8.0
                    target_logits = torch.randn(rows, device="cuda")
                    mask = (
                        torch.arange(rows, device="cuda").remainder(3).ne(1)
                        if use_mask
                        else None
                    )
                    initial_norm_grad = torch.randn(
                        hidden_size, device="cuda", dtype=torch.bfloat16
                    )

                    reference_grad = grad_hidden.clone()
                    zero_target_and_dx = torch.zeros_like(norm_input)
                    adjusted_lse = logsumexp - target_logits
                    batch_norm_grad = torch.zeros_like(initial_norm_grad)
                    reference_dx, reference_losses = (
                        online_mtp_ce_rmsnorm_backward(
                            norm_input,
                            reference_grad,
                            zero_target_and_dx,
                            raw_weight,
                            adjusted_lse,
                            batch_norm_grad,
                            epsilon=1e-6,
                            loss_mask=mask,
                            inplace_ce_buffers=True,
                        )
                    )
                    reference_norm_grad = initial_norm_grad.clone()
                    reference_norm_grad.add_(batch_norm_grad)

                    actual_grad = grad_hidden.clone()
                    actual_norm_grad = initial_norm_grad.clone()
                    actual_dx_storage = torch.empty_like(norm_input)
                    actual_dx, actual_losses = (
                        online_mtp_precomputed_ce_rmsnorm_backward(
                            norm_input,
                            actual_grad,
                            raw_weight,
                            logsumexp,
                            target_logits,
                            actual_norm_grad,
                            epsilon=1e-6,
                            loss_mask=mask,
                            grad_input_out=actual_dx_storage,
                        )
                    )
                    torch.cuda.synchronize()

                    self.assertTrue(torch.equal(actual_dx, reference_dx))
                    self.assertTrue(torch.equal(actual_losses, reference_losses))
                    self.assertTrue(torch.equal(actual_grad, reference_grad))
                    self.assertTrue(
                        torch.equal(actual_norm_grad, reference_norm_grad)
                    )

                    # Disjoint contiguous views in one dtype slab are safe,
                    # even though torch._C._overlaps conservatively reports
                    # shared storage for the pair.
                    slab = torch.empty(
                        2 * norm_input.numel(),
                        device="cuda",
                        dtype=norm_input.dtype,
                    )
                    slab_norm = slab[: norm_input.numel()].view_as(norm_input)
                    slab_dx = slab[norm_input.numel() :].view_as(norm_input)
                    slab_norm.copy_(norm_input)
                    slab_grad = grad_hidden.clone()
                    slab_norm_grad = initial_norm_grad.clone()
                    actual_slab_dx, actual_slab_losses = (
                        online_mtp_precomputed_ce_rmsnorm_backward(
                            slab_norm,
                            slab_grad,
                            raw_weight,
                            logsumexp,
                            target_logits,
                            slab_norm_grad,
                            epsilon=1e-6,
                            loss_mask=mask,
                            grad_input_out=slab_dx,
                        )
                    )
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(actual_slab_dx, reference_dx))
                    self.assertTrue(
                        torch.equal(actual_slab_losses, reference_losses)
                    )
                    self.assertTrue(torch.equal(slab_grad, reference_grad))
                    self.assertTrue(
                        torch.equal(slab_norm_grad, reference_norm_grad)
                    )

                    overlap_slab = torch.empty(
                        norm_input.numel() + norm_input.numel() // 2,
                        device="cuda",
                        dtype=norm_input.dtype,
                    )
                    overlap_norm = overlap_slab[: norm_input.numel()].view_as(
                        norm_input
                    )
                    overlap_dx = overlap_slab[
                        norm_input.numel() // 2 :
                    ].view_as(norm_input)
                    with self.assertRaisesRegex(
                        ValueError, "must not overlap live RMSNorm inputs"
                    ):
                        online_mtp_precomputed_ce_rmsnorm_backward(
                            overlap_norm,
                            grad_hidden.clone(),
                            raw_weight,
                            logsumexp,
                            target_logits,
                            initial_norm_grad.clone(),
                            epsilon=1e-6,
                            loss_mask=mask,
                            grad_input_out=overlap_dx,
                        )

    def test_fused_ce_reduction_owns_inplace_and_outofplace_results(self):
        """The training stream must never retain the reusable pack scratch."""

        for aliases_input in (False, True):
            with self.subTest(aliases_input=aliases_input):
                _, trainer, ticket = self._fused_ce_fixture(device="cpu")

                def fake_all_reduce(value):
                    return value if aliases_input else value.clone()

                with patch(
                    "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                    side_effect=fake_all_reduce,
                ):
                    reduction = trainer.prepare_fused_ce_reduction(ticket)

                scratch = trainer._fused_ce_packed
                self.assertIsNotNone(scratch)
                self.assertFalse(torch._C._overlaps(reduction.reduced, scratch))
                expected = reduction.reduced.clone()
                scratch.fill_(float("nan"))
                torch.testing.assert_close(
                    reduction.reduced, expected, rtol=0, atol=0
                )

    def test_fused_ce_sync_path_finishes_tail_and_backward(self):
        model, _, fixture = self._fused_ce_fixture(device="cpu")
        runtime = OnlineMTPRuntime(
            model,
            activation_bytes=1 << 20,
            max_tickets=4,
            backward_batch_tokens=1,
            accumulation_tokens=64,
            group_ce_projection=True,
            fused_ce_reduction=True,
            use_triton_ce_rmsnorm=True,
            async_backward=False,
        )
        tap = OnlineMTPActivationTap(
            weight_version=0,
            clone_tensors=False,
            group_ce_projection=True,
            fused_ce_reduction=True,
        )
        ids = torch.arange(fixture.num_tokens)
        tap.begin_step(0, input_ids=ids, positions=ids)
        for qualified_name, tensor in fixture.tensors.items():
            tap.record(qualified_name.split(".", 1)[1], tensor)
        ticket_id = runtime.stage(tap)
        self.assertIsNotNone(ticket_id)
        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value,
        ):
            loss = runtime.verify_and_backward(ticket_id, fixture.labels)

        self.assertIsInstance(loss, float)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(runtime.trainer.accumulator.tokens, fixture.num_tokens)
        self.assertEqual(runtime.ring.live_tickets, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_ce_dtype_slab_snapshot_runs_complete_backward(self):
        model, _, fixture = self._fused_ce_fixture(device="cuda")
        runtime = OnlineMTPRuntime(
            model,
            activation_bytes=1 << 24,
            max_tickets=4,
            backward_batch_tokens=1,
            accumulation_tokens=64,
            group_ce_projection=True,
            fused_ce_reduction=True,
            use_triton_ce_rmsnorm=True,
            async_backward=False,
        )
        template = OnlineMTPActivationTap(
            weight_version=0,
            clone_tensors=False,
            group_ce_projection=True,
            fused_ce_reduction=True,
        )
        ids = torch.arange(fixture.num_tokens, device="cuda")
        template.begin_step(0, input_ids=ids, positions=ids)
        for qualified_name, tensor in fixture.tensors.items():
            template.record(qualified_name.split(".", 1)[1], tensor)
        tap = OnlineMTPActivationTap(
            weight_version=0,
            group_ce_projection=True,
            fused_ce_reduction=True,
        )
        tap.copy_from_graph_template(template, num_rows=fixture.num_tokens)

        self.assertEqual(
            len(
                {
                    tensor.untyped_storage().data_ptr()
                    for step in tap.steps.values()
                    for tensor in step.values()
                }
            ),
            2,
        )
        ticket_id = runtime.stage(tap)
        self.assertIsNotNone(ticket_id)
        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value,
        ):
            loss = runtime.verify_and_backward(ticket_id, fixture.labels)
        torch.cuda.synchronize()

        self.assertIsInstance(loss, float)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(runtime.trainer.accumulator.tokens, fixture.num_tokens)
        self.assertEqual(runtime.ring.live_tickets, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA streams")
    def test_fused_ce_async_tail_waits_for_serving_reduction_and_owns_inputs(self):
        model, _, first_fixture = self._fused_ce_fixture(device="cuda")
        _, _, second_fixture = self._fused_ce_fixture(device="cuda")
        fixtures = (first_fixture, second_fixture)
        runtime = OnlineMTPRuntime(
            model,
            activation_bytes=1 << 24,
            max_tickets=4,
            backward_batch_tokens=sum(fixture.num_tokens for fixture in fixtures),
            accumulation_tokens=64,
            group_ce_projection=True,
            fused_ce_reduction=True,
            use_triton_ce_rmsnorm=True,
            async_backward=True,
        )
        taps = []
        expected_local = []
        for fixture in fixtures:
            tap = OnlineMTPActivationTap(
                weight_version=0,
                clone_tensors=False,
                group_ce_projection=True,
                fused_ce_reduction=True,
            )
            ids = torch.arange(fixture.num_tokens, device="cuda")
            tap.begin_step(0, input_ids=ids, positions=ids)
            for qualified_name, tensor in fixture.tensors.items():
                tap.record(qualified_name.split(".", 1)[1], tensor)
            local_expected = fixture.tensors[
                "step_0.ce_local_expected_weight"
            ]
            expected_local.append(local_expected.clone())
            local_expected.fill_(float("nan"))
            taps.append(tap)

        # Model a graph snapshot that is not ready when verification reaches
        # the runtime.  The serving stream must wait before pack/all-reduce;
        # producer_done then carries that dependency to the training stream.
        torch.cuda.current_stream().synchronize()
        snapshot_stream = torch.cuda.Stream()
        with torch.cuda.stream(snapshot_stream):
            torch.cuda._sleep(10_000_000)
            for fixture, expected in zip(fixtures, expected_local):
                fixture.tensors["step_0.ce_local_expected_weight"].copy_(expected)
            snapshot_ready = torch.cuda.Event()
            snapshot_ready.record(snapshot_stream)
        ticket_ids = []
        for tap in taps:
            tap.ready_event = snapshot_ready
            ticket_id = runtime.stage(tap)
            self.assertIsNotNone(ticket_id)
            ticket_ids.append(ticket_id)

        loss_masks = (
            torch.tensor([True, False, True, True, False], device="cuda"),
            torch.tensor([False, True, True, False, True], device="cuda"),
        )

        seen_streams = {}
        serving_stream = torch.cuda.current_stream().cuda_stream
        original_prepare = runtime.trainer.prepare_fused_ce_reduction
        original_finish = runtime.trainer.finish_fused_ce_tail

        def observed_prepare(*args, **kwargs):
            seen_streams["prepare"] = torch.cuda.current_stream().cuda_stream
            return original_prepare(*args, **kwargs)

        def observed_finish(*args, **kwargs):
            seen_streams["finish"] = torch.cuda.current_stream().cuda_stream
            return original_finish(*args, **kwargs)

        def delayed_outofplace_all_reduce(value):
            output = torch.full_like(value, float("nan"))
            # If the training stream misses its producer-event dependency, the
            # CE tail can observe NaNs before this serving-stream copy.
            torch.cuda._sleep(10_000_000)
            output.copy_(value)
            return output

        with (
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                side_effect=delayed_outofplace_all_reduce,
            ),
            patch.object(
                runtime.trainer,
                "prepare_fused_ce_reduction",
                side_effect=observed_prepare,
            ),
            patch.object(
                runtime.trainer,
                "finish_fused_ce_tail",
                side_effect=observed_finish,
            ),
        ):
            first_loss = runtime.verify_and_backward(
                ticket_ids[0], first_fixture.labels, loss_masks[0]
            )
            loss = runtime.verify_and_backward(
                ticket_ids[1], second_fixture.labels, loss_masks[1]
            )

        self.assertIsNone(first_loss)
        self.assertIsNone(loss)
        live = runtime.ring.get(ticket_ids[0])
        for name in (
            "fused_ce_reduced",
            "fused_ce_norm_input",
            "fused_ce_local_expected",
            "combined_loss_mask",
        ):
            self.assertIn(name, live.tensors)
        self.assertEqual(
            live.tensors["combined_loss_mask"].numel(),
            sum(fixture.num_tokens for fixture in fixtures),
        )
        self.assertFalse(
            torch._C._overlaps(
                live.tensors["fused_ce_reduced"],
                runtime.trainer._fused_ce_packed,
            )
        )
        runtime._reap_completed(wait=True)

        self.assertEqual(seen_streams["prepare"], serving_stream)
        self.assertEqual(
            seen_streams["finish"], runtime._training_stream.cuda_stream
        )
        self.assertTrue(math.isfinite(runtime.last_loss))
        self.assertEqual(runtime.ring.live_tickets, 0)
        self.assertEqual(
            runtime.trainer.accumulator.tokens,
            sum(int(mask.sum().item()) for mask in loss_masks),
        )
        self.assertTrue(
            all(
                bool(torch.isfinite(gradient).all())
                for gradient in runtime.trainer.accumulator.gradients.values()
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA streams")
    def test_deferred_ce_pipeline_overlaps_local_work_but_orders_tp_reduce(self):
        model, _, first_fixture = self._fused_ce_fixture(device="cuda")
        _, _, second_fixture = self._fused_ce_fixture(device="cuda")
        fixtures = (first_fixture, second_fixture)
        runtime = OnlineMTPRuntime(
            model,
            activation_bytes=1 << 24,
            max_tickets=4,
            backward_batch_tokens=first_fixture.num_tokens,
            accumulation_tokens=64,
            group_ce_projection=True,
            defer_grouped_ce_projection=True,
            fused_ce_reduction=True,
            async_deferred_ce_reduction=True,
            use_triton_ce_rmsnorm=True,
            async_backward=True,
        )
        self.assertIsNone(runtime._online_ce_process_group)

        ticket_ids = []
        for fixture in fixtures:
            tap = self._tap_from_fused_ce_fixture(fixture)
            step = tap.steps[0]
            step.pop("ce_local_expected_weight")
            step["ce_probability"] = torch.softmax(
                torch.randn(
                    fixture.num_tokens,
                    model.lm_head.weight.shape[0],
                    device="cuda",
                    dtype=model.lm_head.weight.dtype,
                ),
                dim=-1,
            )
            tap.deferred_grouped_ce_projection_active = True
            ticket_id = runtime.stage(tap)
            self.assertIsNotNone(ticket_id)
            ticket_ids.append(ticket_id)

        serving_stream = torch.cuda.current_stream().cuda_stream
        observed = {"prepare": [], "reduce": []}
        original_prepare = runtime.trainer.prepare_fused_ce_local
        original_reduce = runtime.trainer.reduce_fused_ce_local

        def observed_prepare(*args, **kwargs):
            observed["prepare"].append(torch.cuda.current_stream().cuda_stream)
            return original_prepare(*args, **kwargs)

        def observed_reduce(*args, **kwargs):
            observed["reduce"].append(torch.cuda.current_stream().cuda_stream)
            return original_reduce(*args, **kwargs)

        with (
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                side_effect=lambda value: value.clone(),
            ),
            patch.object(
                runtime.trainer,
                "prepare_fused_ce_local",
                side_effect=observed_prepare,
            ),
            patch.object(
                runtime.trainer,
                "reduce_fused_ce_local",
                side_effect=observed_reduce,
            ),
        ):
            self.assertIsNone(
                runtime.verify_and_backward(ticket_ids[0], first_fixture.labels)
            )
            self.assertIsNotNone(runtime._pending_ce_projection)
            self.assertFalse(runtime._pending)

            self.assertIsNone(
                runtime.verify_and_backward(ticket_ids[1], second_fixture.labels)
            )
            self.assertIsNotNone(runtime._pending_ce_projection)
            self.assertTrue(runtime._pending)
            runtime._finish_pending_ce_projection()
            runtime._reap_completed(wait=True)

        self.assertEqual(
            observed["prepare"],
            [runtime._training_stream.cuda_stream] * 2,
        )
        self.assertEqual(observed["reduce"], [serving_stream] * 2)
        self.assertEqual(runtime.ring.live_tickets, 0)
        self.assertEqual(
            runtime.trainer.accumulator.tokens,
            sum(fixture.num_tokens for fixture in fixtures),
        )
        self.assertTrue(
            all(
                bool(torch.isfinite(gradient).all())
                for gradient in runtime.trainer.accumulator.gradients.values()
            )
        )

    def test_grouped_ce_probability_workspace_is_grow_only(self):
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        tap = OnlineMTPActivationTap(
            weight_version=0,
            clone_tensors=False,
            group_ce_projection=True,
            expected_ce_steps=3,
        )
        views = []
        for step in range(3):
            ids = torch.arange(4, device=self.device)
            tap.begin_step(step, input_ids=ids, positions=ids)
            view = tap.reserve_ce_probability(4, 17, dtype=dtype, device=ids.device)
            view.fill_(step + 1)
            tap.record_ce_probability(view)
            views.append(view)

        owner = tap._ce_probability_buffer
        self.assertEqual(owner.shape, (12, 17))
        self.assertTrue(
            all(
                view.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
                for view in views
            )
        )
        torch.testing.assert_close(
            owner[:, 0].float(),
            torch.tensor(
                [1] * 4 + [2] * 4 + [3] * 4,
                device=self.device,
                dtype=torch.float32,
            ),
        )

        owner_pointer = owner.data_ptr()
        tap.reset()
        for step in range(3):
            ids = torch.arange(3, device=self.device)
            tap.begin_step(step, input_ids=ids, positions=ids)
            tap.reserve_ce_probability(3, 17, dtype=dtype, device=ids.device)
        self.assertEqual(tap._ce_probability_buffer.data_ptr(), owner_pointer)
        self.assertEqual(tap._ce_probability_buffer.shape, (12, 17))

        fixed = torch.empty(5, 17, device=self.device, dtype=dtype)
        fixed_tap = OnlineMTPActivationTap(
            weight_version=0,
            group_ce_projection=True,
            expected_ce_steps=2,
            ce_probability_buffer=fixed,
        )
        for step in range(2):
            ids = torch.arange(3, device=self.device)
            fixed_tap.begin_step(step, input_ids=ids, positions=ids)
            if step == 0:
                fixed_tap.reserve_ce_probability(
                    3, 17, dtype=dtype, device=ids.device
                )
            else:
                with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
                    fixed_tap.reserve_ce_probability(
                        3, 17, dtype=dtype, device=ids.device
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graphs")
    def test_grouped_ce_probability_workspace_replays_in_cuda_graph(self):
        rows, local_vocab, hidden_size, steps = 4, 32, 16, 3
        dtype = torch.bfloat16
        sources = [
            torch.rand(rows, local_vocab, device="cuda", dtype=dtype)
            for _ in range(steps)
        ]
        weight = torch.randn(
            local_vocab, hidden_size, device="cuda", dtype=dtype
        )
        indices = SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=local_vocab,
        )
        lm_head = SimpleNamespace(weight=weight, shard_indices=indices)
        tap = OnlineMTPActivationTap(
            weight_version=0,
            clone_tensors=False,
            group_ce_projection=True,
            expected_ce_steps=steps,
        )

        def run_once():
            tap.reset()
            for step, source in enumerate(sources):
                ids = torch.arange(rows, device="cuda")
                tap.begin_step(step, input_ids=ids, positions=ids)
                probability = tap.reserve_ce_probability(
                    rows, local_vocab, dtype=dtype, device=source.device
                )
                probability.copy_(source)
                tap.record_ce_probability(probability)
            tap.finalize_grouped_ce(lm_head)

        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value,
        ):
            warmup_stream = torch.cuda.Stream()
            with torch.cuda.stream(warmup_stream):
                run_once()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()
            owner_pointer = tap._ce_probability_buffer.data_ptr()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_once()

        self.assertEqual(tap._ce_probability_buffer.data_ptr(), owner_pointer)
        for source in sources:
            source.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        actual = torch.cat(
            [tap.steps[step]["ce_expected_weight"] for step in range(steps)]
        )
        reference = torch.cat(sources) @ weight
        torch.testing.assert_close(actual, reference.float(), rtol=2e-3, atol=3e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graphs")
    def test_two_cuda_graph_shapes_share_ce_probability_workspace(self):
        local_vocab, hidden_size, steps = 32, 16, 3
        dtype = torch.bfloat16
        shared = torch.empty(steps * 7, local_vocab, device="cuda", dtype=dtype)
        weight = torch.randn(
            local_vocab, hidden_size, device="cuda", dtype=dtype
        )
        indices = SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=local_vocab,
        )
        lm_head = SimpleNamespace(weight=weight, shard_indices=indices)

        def make_shape(rows):
            sources = [
                torch.rand(rows, local_vocab, device="cuda", dtype=dtype)
                for _ in range(steps)
            ]
            tap = OnlineMTPActivationTap(
                weight_version=0,
                clone_tensors=False,
                group_ce_projection=True,
                expected_ce_steps=steps,
                ce_probability_buffer=shared,
            )

            def run_once():
                tap.reset()
                for step, source in enumerate(sources):
                    ids = torch.arange(rows, device="cuda")
                    tap.begin_step(step, input_ids=ids, positions=ids)
                    probability = tap.reserve_ce_probability(
                        rows, local_vocab, dtype=dtype, device=source.device
                    )
                    probability.copy_(source)
                    tap.record_ce_probability(probability)
                tap.finalize_grouped_ce(lm_head)

            warmup_stream = torch.cuda.Stream()
            with torch.cuda.stream(warmup_stream):
                run_once()
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_once()
            return tap, sources, graph

        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_reduce",
            side_effect=lambda value: value,
        ):
            small_tap, small_sources, small_graph = make_shape(3)
            large_tap, large_sources, large_graph = make_shape(7)

        shared_pointer = shared.untyped_storage().data_ptr()
        self.assertEqual(
            small_tap._ce_probability_buffer.untyped_storage().data_ptr(),
            shared_pointer,
        )
        self.assertEqual(
            large_tap._ce_probability_buffer.untyped_storage().data_ptr(),
            shared_pointer,
        )

        for tap, sources, graph, scale in (
            (small_tap, small_sources, small_graph, 0.5),
            (large_tap, large_sources, large_graph, 0.25),
            (small_tap, small_sources, small_graph, 0.75),
        ):
            for source in sources:
                source.mul_(scale)
            graph.replay()
            torch.cuda.synchronize()
            actual = torch.cat(
                [tap.steps[step]["ce_expected_weight"] for step in range(steps)]
            )
            reference = torch.cat(sources) @ weight
            torch.testing.assert_close(
                actual, reference.float(), rtol=2e-3, atol=3e-4
            )

    def test_rms_norm_backward_matches_autograd(self):
        x_ref = torch.randn(7, 32, device=self.device, dtype=torch.float32)
        weight_ref = torch.randn(32, device=self.device, dtype=torch.float32)
        grad_output = torch.randn_like(x_ref)
        x = x_ref.detach().requires_grad_(True)
        weight = weight_ref.detach().requires_grad_(True)
        inv = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
        output = x * inv * weight
        output.backward(grad_output)

        _, saved_inv = rms_norm_forward(x_ref, weight_ref, 1e-6)
        grad_x, grad_weight = rms_norm_backward(
            grad_output, x_ref, weight_ref, saved_inv
        )
        torch.testing.assert_close(grad_x, x.grad, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(grad_weight, weight.grad, rtol=2e-5, atol=2e-5)

    def test_dense_swiglu_backward_matches_autograd(self):
        x_ref = torch.randn(11, 16, device=self.device, dtype=torch.float32)
        gate_up_ref = torch.randn(48, 16, device=self.device, dtype=torch.float32)
        down_ref = torch.randn(16, 24, device=self.device, dtype=torch.float32)
        grad_output = torch.randn(11, 16, device=self.device, dtype=torch.float32)

        x = x_ref.detach().requires_grad_(True)
        gate_up = gate_up_ref.detach().requires_grad_(True)
        down = down_ref.detach().requires_grad_(True)
        projected = x @ gate_up.transpose(0, 1)
        gate, up = projected.chunk(2, dim=-1)
        output = (torch.nn.functional.silu(gate) * up) @ down.transpose(0, 1)
        output.backward(grad_output)

        _, context = dense_swiglu_forward(x_ref, gate_up_ref, down_ref)
        grad_x, grad_gate_up, grad_down = dense_swiglu_backward(
            grad_output, context, gate_up_ref, down_ref
        )
        torch.testing.assert_close(grad_x, x.grad, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(grad_gate_up, gate_up.grad, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(grad_down, down.grad, rtol=3e-5, atol=3e-5)

    def test_mixed_precision_swiglu_backward_matches_bf16_autograd(self):
        x_ref = torch.randn(13, 32, device=self.device, dtype=torch.bfloat16)
        gate_up_ref = torch.randn(64, 32, device=self.device, dtype=torch.bfloat16)
        down_ref = torch.randn(32, 32, device=self.device, dtype=torch.bfloat16)
        grad_output = torch.randn(13, 32, device=self.device, dtype=torch.bfloat16)

        x = x_ref.detach().requires_grad_(True)
        gate_up = gate_up_ref.detach().requires_grad_(True)
        down = down_ref.detach().requires_grad_(True)
        projected = x @ gate_up.transpose(0, 1)
        gate, up = projected.chunk(2, dim=-1)
        output = (torch.nn.functional.silu(gate) * up) @ down.transpose(0, 1)
        output.backward(grad_output)

        _, context = dense_swiglu_forward(x_ref, gate_up_ref, down_ref)
        grad_x, grad_gate_up, grad_down = dense_swiglu_backward_mixed_precision(
            grad_output, context, gate_up_ref, down_ref
        )
        torch.testing.assert_close(grad_x, x.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(grad_gate_up, gate_up.grad, rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(grad_down, down.grad, rtol=3e-2, atol=3e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Triton kernels")
    def test_triton_swiglu_backward_is_bitwise_equal_and_inplace_safe(self):
        from sglang.srt.speculative.triton_ops.online_mtp_swiglu import (
            online_mtp_swiglu_backward,
        )

        grad_output = torch.randn(96, 2176, device="cuda", dtype=torch.bfloat16)
        gate_up = torch.randn(96, 4352, device="cuda", dtype=torch.bfloat16)
        reference = silu_and_mul_backward_mixed_precision(grad_output, gate_up)
        actual = online_mtp_swiglu_backward(grad_output, gate_up)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

        inplace_input = gate_up.clone()
        inplace_output = online_mtp_swiglu_backward(
            grad_output, inplace_input, inplace=True
        )
        self.assertIs(inplace_output, inplace_input)
        torch.testing.assert_close(inplace_output, reference, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Triton kernels")
    def test_triton_ce_rmsnorm_backward_matches_mixed_precision_reference(self):
        from sglang.srt.speculative.triton_ops.online_mtp_ce_rmsnorm import (
            online_mtp_ce_rmsnorm_backward,
        )

        rows, hidden_size = 96, 5120
        norm_input = (
            0.1
            * torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        expected_weight = 0.01 * torch.randn(
            rows, hidden_size, device="cuda", dtype=torch.float32
        )
        target_weight = (
            0.01
            * torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        raw_norm_weight = (
            0.01 * torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        )
        logsumexp = torch.rand(rows, device="cuda", dtype=torch.float32) + 8.0
        loss_mask = torch.rand(rows, device="cuda") > 0.2
        initial_accumulator = (
            0.01 * torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        )

        effective_weight = raw_norm_weight.float() + 1.0
        inv_rms = torch.rsqrt(
            norm_input.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        final_hidden = (
            norm_input.float() * inv_rms * effective_weight
        ).to(torch.bfloat16)
        target_logits = (final_hidden.float() * target_weight.float()).sum(dim=-1)
        reference_losses = (
            (logsumexp - target_logits).clamp_min_(0.0) * loss_mask
        )
        grad_hidden = (
            expected_weight - target_weight.float()
        ) * loss_mask[:, None]
        reference_grad, reference_norm_grad = rms_norm_backward(
            grad_hidden,
            norm_input,
            effective_weight,
            inv_rms,
        )
        reference_accumulator = initial_accumulator.clone()
        reference_accumulator.add_(reference_norm_grad.to(torch.bfloat16))

        baseline_expected_weight = expected_weight.clone()
        baseline_target_weight = target_weight.clone()
        actual_accumulator = initial_accumulator.clone()
        actual_grad, actual_losses = online_mtp_ce_rmsnorm_backward(
            norm_input,
            baseline_expected_weight,
            baseline_target_weight,
            raw_norm_weight,
            logsumexp,
            actual_accumulator,
            epsilon=1e-6,
            loss_mask=loss_mask,
        )
        torch.testing.assert_close(
            actual_grad, reference_grad, rtol=5e-2, atol=2e-3
        )
        torch.testing.assert_close(
            actual_losses, reference_losses, rtol=3e-4, atol=2e-4
        )
        torch.testing.assert_close(
            actual_accumulator, reference_accumulator, rtol=5e-2, atol=2e-3
        )
        # The default API remains non-destructive.
        torch.testing.assert_close(
            baseline_expected_weight, expected_weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            baseline_target_weight, target_weight, rtol=0, atol=0
        )

        inplace_expected_weight = expected_weight.clone()
        inplace_target_weight = target_weight.clone()
        inplace_accumulator = initial_accumulator.clone()
        inplace_grad, inplace_losses = online_mtp_ce_rmsnorm_backward(
            norm_input,
            inplace_expected_weight,
            inplace_target_weight,
            raw_norm_weight,
            logsumexp,
            inplace_accumulator,
            epsilon=1e-6,
            loss_mask=loss_mask,
            inplace_ce_buffers=True,
        )
        self.assertIs(inplace_grad, inplace_target_weight)
        torch.testing.assert_close(inplace_grad, actual_grad, rtol=0, atol=0)
        torch.testing.assert_close(inplace_losses, actual_losses, rtol=0, atol=0)
        torch.testing.assert_close(
            inplace_accumulator, actual_accumulator, rtol=0, atol=0
        )
        torch.testing.assert_close(
            inplace_expected_weight, grad_hidden, rtol=0, atol=0
        )

    def test_ce_rmsnorm_inplace_buffers_match_aten_fallback(self):
        from sglang.srt.speculative.triton_ops.online_mtp_ce_rmsnorm import (
            online_mtp_ce_rmsnorm_backward,
        )

        rows, hidden_size = 7, 32
        norm_input = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
        expected_weight = torch.randn(rows, hidden_size, dtype=torch.float32)
        target_weight = torch.randn(rows, hidden_size, dtype=torch.bfloat16)
        raw_norm_weight = torch.randn(hidden_size, dtype=torch.bfloat16)
        logsumexp = torch.randn(rows, dtype=torch.float32)
        loss_mask = torch.tensor(
            [True, False, True, True, False, True, True], dtype=torch.bool
        )
        initial_accumulator = torch.randn(hidden_size, dtype=torch.bfloat16)

        baseline_accumulator = initial_accumulator.clone()
        baseline_grad, baseline_losses = online_mtp_ce_rmsnorm_backward(
            norm_input,
            expected_weight.clone(),
            target_weight.clone(),
            raw_norm_weight,
            logsumexp,
            baseline_accumulator,
            epsilon=1e-6,
            loss_mask=loss_mask,
        )

        inplace_expected = expected_weight.clone()
        inplace_target = target_weight.clone()
        inplace_accumulator = initial_accumulator.clone()
        inplace_grad, inplace_losses = online_mtp_ce_rmsnorm_backward(
            norm_input,
            inplace_expected,
            inplace_target,
            raw_norm_weight,
            logsumexp,
            inplace_accumulator,
            epsilon=1e-6,
            loss_mask=loss_mask,
            inplace_ce_buffers=True,
        )

        self.assertIs(inplace_grad, inplace_target)
        torch.testing.assert_close(inplace_grad, baseline_grad, rtol=0, atol=0)
        torch.testing.assert_close(inplace_losses, baseline_losses, rtol=0, atol=0)
        torch.testing.assert_close(
            inplace_accumulator, baseline_accumulator, rtol=0, atol=0
        )
        expected_grad_hidden = (
            expected_weight - target_weight.float()
        ) * loss_mask[:, None]
        torch.testing.assert_close(
            inplace_expected, expected_grad_hidden, rtol=0, atol=0
        )

        explicit_grad = torch.empty_like(norm_input)
        with self.assertRaisesRegex(ValueError, "grad_input_out must be omitted"):
            online_mtp_ce_rmsnorm_backward(
                norm_input,
                expected_weight.clone(),
                target_weight.clone(),
                raw_norm_weight,
                logsumexp,
                initial_accumulator.clone(),
                epsilon=1e-6,
                grad_input_out=explicit_grad,
                inplace_ce_buffers=True,
            )

        shared_scalar_outputs = torch.empty(rows, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "inv_rms_out must not overlap"):
            online_mtp_ce_rmsnorm_backward(
                norm_input,
                expected_weight.clone(),
                target_weight.clone(),
                raw_norm_weight,
                logsumexp,
                initial_accumulator.clone(),
                epsilon=1e-6,
                inv_rms_out=shared_scalar_outputs,
                losses_out=shared_scalar_outputs,
            )

        shared_weight = raw_norm_weight.clone()
        with self.assertRaisesRegex(ValueError, "raw_norm_weight must not overlap"):
            online_mtp_ce_rmsnorm_backward(
                norm_input,
                expected_weight.clone(),
                target_weight.clone(),
                shared_weight,
                logsumexp,
                shared_weight,
                epsilon=1e-6,
            )

    def test_single_reduce_ce_tail_matches_current_loss_and_grad_with_mask(self):
        from sglang.srt.speculative.triton_ops.online_mtp_ce_rmsnorm import (
            online_mtp_ce_rmsnorm_backward,
            pack_local_ce_for_tp_reduce,
        )

        rows, hidden_size, vocab_size = 7, 32, 41
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        norm_input = torch.randn(
            rows, hidden_size, device=self.device, dtype=dtype
        )
        local_expected = torch.randn(
            rows, hidden_size, device=self.device, dtype=dtype
        )
        raw_norm_weight = torch.randn(
            hidden_size, device=self.device, dtype=dtype
        )
        lm_head_weight = torch.randn(
            vocab_size, hidden_size, device=self.device, dtype=dtype
        )
        labels = torch.tensor(
            [0, vocab_size - 1, 3, 7, 11, 19, 23], device=self.device
        )
        logsumexp = torch.rand(rows, device=self.device) + 12.0
        loss_mask = torch.tensor(
            [True, False, True, True, False, True, True],
            device=self.device,
        )
        initial_accumulator = torch.randn(
            hidden_size, device=self.device, dtype=dtype
        )

        current_accumulator = initial_accumulator.clone()
        current_grad, current_losses = online_mtp_ce_rmsnorm_backward(
            norm_input,
            local_expected.float(),
            lm_head_weight[labels],
            raw_norm_weight,
            logsumexp.clone(),
            current_accumulator,
            epsilon=1e-6,
            loss_mask=loss_mask,
        )

        logical = rows * hidden_size + rows
        packed = torch.empty(
            (logical + 3) // 4 * 4, device=self.device, dtype=torch.float32
        )
        packed = pack_local_ce_for_tp_reduce(
            local_expected,
            norm_input,
            raw_norm_weight,
            lm_head_weight,
            labels,
            shard_start=0,
            shard_stop=vocab_size,
            epsilon=1e-6,
            out=packed,
        )
        reduced_grad = packed[: rows * hidden_size].view(rows, hidden_size)
        adjusted_lse = logsumexp - packed[rows * hidden_size : logical]
        zero_target = torch.zeros_like(local_expected)
        fused_accumulator = initial_accumulator.clone()
        fused_grad, fused_losses = online_mtp_ce_rmsnorm_backward(
            norm_input,
            reduced_grad,
            zero_target,
            raw_norm_weight,
            adjusted_lse,
            fused_accumulator,
            epsilon=1e-6,
            loss_mask=loss_mask,
        )

        torch.testing.assert_close(fused_losses, current_losses, rtol=0, atol=0)
        torch.testing.assert_close(fused_grad, current_grad, rtol=0, atol=0)
        torch.testing.assert_close(
            fused_accumulator, current_accumulator, rtol=0, atol=0
        )

    def test_fused_swiglu_accumulation_matches_materialized_gradients(self):
        x = torch.randn(13, 32, device=self.device, dtype=torch.bfloat16)
        gate_up_weight = torch.randn(64, 32, device=self.device, dtype=torch.bfloat16)
        down_weight = torch.randn(32, 32, device=self.device, dtype=torch.bfloat16)
        grad_output = torch.randn(13, 32, device=self.device, dtype=torch.bfloat16)
        _, context = dense_swiglu_forward(x, gate_up_weight, down_weight)
        _, grad_gate_up, grad_down = dense_swiglu_backward_mixed_precision(
            grad_output,
            context,
            gate_up_weight,
            down_weight,
            need_input_grad=False,
        )
        gate_accumulator = torch.randn_like(gate_up_weight)
        down_accumulator = torch.randn_like(down_weight)
        expected_gate = gate_accumulator + grad_gate_up
        expected_down = down_accumulator + grad_down

        dense_swiglu_backward_accumulate_mixed_precision(
            grad_output,
            context,
            gate_up_weight,
            down_weight,
            gate_accumulator,
            down_accumulator,
        )
        # beta=1 fuses accumulation into the GEMM epilogue, which has one
        # fewer BF16 rounding point than materialize-then-add.
        torch.testing.assert_close(
            gate_accumulator, expected_gate, rtol=3e-2, atol=3e-2
        )
        torch.testing.assert_close(
            down_accumulator, expected_down, rtol=3e-2, atol=3e-2
        )

    def test_vocab_ce_hidden_backward_matches_autograd(self):
        hidden_ref = torch.randn(9, 13, device=self.device, dtype=torch.float32)
        weight = torch.randn(37, 13, device=self.device, dtype=torch.float32)
        labels = torch.randint(0, 37, (9,), device=self.device)
        mask = torch.tensor(
            [1, 0, 1, 1, 0, 1, 1, 1, 0],
            dtype=torch.bool,
            device=self.device,
        )
        hidden = hidden_ref.detach().requires_grad_(True)
        logits = hidden @ weight.transpose(0, 1)
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        reference_loss = losses[mask].mean()
        reference_loss.backward()

        raw_logits = hidden_ref @ weight.transpose(0, 1)
        loss, grad_hidden = vocab_ce_hidden_backward_from_logits(
            raw_logits, weight, labels, chunk_size=11, loss_mask=mask
        )
        torch.testing.assert_close(loss, reference_loss, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(grad_hidden, hidden.grad, rtol=3e-5, atol=3e-5)

        logsumexp, expected_weight = vocab_ce_forward_stats(
            raw_logits, weight, chunk_size=11
        )
        stats_loss, stats_grad_hidden = vocab_ce_hidden_backward_from_stats(
            logsumexp,
            expected_weight,
            hidden_ref,
            weight,
            labels,
            loss_mask=mask,
        )
        torch.testing.assert_close(stats_loss, reference_loss, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(stats_grad_hidden, hidden.grad, rtol=3e-5, atol=3e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Triton kernels")
    def test_compact_vocab_logsumexp_matches_fp32_reference(self):
        """Cover both launch geometries at the production vocabulary size."""

        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
            compact_vocab_logsumexp,
        )

        vocab_size = 248320
        for rows in (3, 48):
            logits = torch.randn(
                rows, vocab_size, device="cuda", dtype=torch.bfloat16
            )
            reference = torch.logsumexp(logits.float(), dim=-1)
            actual = compact_vocab_logsumexp(logits)
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Triton kernels")
    def test_compact_lse_argmax_fusion_matches_separate_operations(self):
        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
            compact_vocab_logsumexp,
            compact_vocab_logsumexp_argmax,
        )

        rows, vocab_size = 8, 248320
        logits = torch.randn(rows, vocab_size, device="cuda", dtype=torch.float32)
        # Exercise first-index tie breaking both within and across 8192-wide
        # chunks; the lower vocabulary id must win exactly as torch.argmax.
        logits[0, 101] = 100.0
        logits[0, 9001] = 100.0
        reference_lse = compact_vocab_logsumexp(logits)
        reference_argmax = torch.argmax(logits, dim=-1)
        fused_lse, fused_argmax = compact_vocab_logsumexp_argmax(logits)
        torch.testing.assert_close(fused_lse, reference_lse, rtol=0, atol=0)
        torch.testing.assert_close(fused_argmax, reference_argmax, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Triton kernels")
    def test_local_vocab_statistics_preserve_greedy_and_ce_boundary(self):
        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
            compact_global_vocab_stats,
            compact_local_vocab_stats,
            compact_vocab_logsumexp_argmax,
            compact_vocab_probability,
        )

        rows, tp_size, local_vocab = 8, 8, 31040
        logits = torch.randn(
            rows, tp_size * local_vocab, device="cuda", dtype=torch.bfloat16
        )
        # Exact cross-rank tie: the lower global token must win.
        logits[0, 101] = 100.0
        logits[0, 3 * local_vocab + 17] = 100.0
        reference_lse, reference_argmax = compact_vocab_logsumexp_argmax(logits)
        packets = [
            compact_local_vocab_stats(
                shard.contiguous(), global_vocab_start=rank * local_vocab
            )
            for rank, shard in enumerate(logits.split(local_vocab, dim=-1))
        ]
        gathered = torch.cat(packets, dim=-1)
        actual_lse, actual_argmax = compact_global_vocab_stats(
            gathered, tp_size=tp_size
        )
        torch.testing.assert_close(actual_argmax, reference_argmax, rtol=0, atol=0)
        torch.testing.assert_close(actual_lse, reference_lse, rtol=1e-6, atol=1e-6)

        rank = 3
        reference_probability = compact_vocab_probability(
            logits,
            reference_lse,
            local_start=rank * local_vocab,
            local_stop=(rank + 1) * local_vocab,
            output_dtype=torch.bfloat16,
        )
        actual_probability = compact_vocab_probability(
            logits[:, rank * local_vocab : (rank + 1) * local_vocab].contiguous(),
            actual_lse,
            local_start=0,
            local_stop=local_vocab,
            output_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(
            actual_probability, reference_probability, rtol=2e-3, atol=5e-7
        )

        # The manual backward consumes the vocabulary projection rather than
        # the probability matrix itself. Check the complete TP-summed CE
        # expected-hidden boundary with ordinary strict-BF16 GEMMs.
        projection_width = 256
        reference_expected = torch.zeros(
            rows, projection_width, device="cuda", dtype=torch.float32
        )
        actual_expected = torch.zeros_like(reference_expected)
        for shard_rank in range(tp_size):
            start = shard_rank * local_vocab
            stop = start + local_vocab
            weight = torch.randn(
                local_vocab,
                projection_width,
                device="cuda",
                dtype=torch.bfloat16,
            )
            reference_shard = compact_vocab_probability(
                logits,
                reference_lse,
                local_start=start,
                local_stop=stop,
                output_dtype=torch.bfloat16,
            )
            actual_shard = compact_vocab_probability(
                logits[:, start:stop].contiguous(),
                actual_lse,
                local_start=0,
                local_stop=local_vocab,
                output_dtype=torch.bfloat16,
            )
            reference_expected.add_((reference_shard @ weight).float())
            actual_expected.add_((actual_shard @ weight).float())
        torch.testing.assert_close(
            actual_expected, reference_expected, rtol=2e-3, atol=2e-4
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Triton kernels")
    def test_grouped3_ce_producer_is_bitwise_equivalent(self):
        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
            compact_vocab_logsumexp,
            compact_vocab_logsumexp_grouped3,
            compact_vocab_probability,
            compact_vocab_probability_grouped3,
        )

        rows, vocab_size = 8, 248320
        local_start, local_stop = 31040, 62080
        logits = tuple(
            torch.randn(rows, vocab_size, device="cuda", dtype=torch.float32)
            for _ in range(3)
        )
        reference_lse = torch.cat(
            [compact_vocab_logsumexp(value) for value in logits]
        )
        grouped_lse = compact_vocab_logsumexp_grouped3(logits)
        torch.testing.assert_close(grouped_lse, reference_lse, rtol=0, atol=0)

        reference_probability = torch.cat(
            [
                compact_vocab_probability(
                    value,
                    reference_lse[step * rows : (step + 1) * rows],
                    local_start=local_start,
                    local_stop=local_stop,
                    output_dtype=torch.bfloat16,
                )
                for step, value in enumerate(logits)
            ]
        )
        probability_out = torch.empty_like(reference_probability)
        grouped_probability = compact_vocab_probability_grouped3(
            logits,
            grouped_lse,
            local_start=local_start,
            local_stop=local_stop,
            output_dtype=torch.bfloat16,
            out=probability_out,
        )
        self.assertEqual(grouped_probability.data_ptr(), probability_out.data_ptr())
        torch.testing.assert_close(
            grouped_probability, reference_probability, rtol=0, atol=0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA Triton kernels")
    def test_compact_lse_preserves_ce_loss_and_hidden_gradient(self):
        from sglang.srt.speculative.triton_ops.online_mtp_lse import (
            compact_vocab_logsumexp,
            compact_vocab_probability,
        )

        rows, vocab_size, hidden_size = 48, 248320, 64
        local_start, local_stop = 31040, 62080
        logits = torch.randn(
            rows, vocab_size, device="cuda", dtype=torch.bfloat16
        )
        local_weight = torch.randn(
            local_stop - local_start,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        labels = torch.randint(0, vocab_size, (rows,), device="cuda")

        reference_lse = torch.logsumexp(logits.float(), dim=-1)
        compact_lse = compact_vocab_logsumexp(logits)
        reference_probability = torch.exp(
            logits[:, local_start:local_stop].float() - reference_lse[:, None]
        ).to(torch.bfloat16)
        compact_probability = torch.exp(
            logits[:, local_start:local_stop].float() - compact_lse[:, None]
        ).to(torch.bfloat16)
        probability_out = torch.empty_like(compact_probability)
        fused_probability = compact_vocab_probability(
            logits,
            compact_lse,
            local_start=local_start,
            local_stop=local_stop,
            output_dtype=torch.bfloat16,
            out=probability_out,
        )
        self.assertEqual(fused_probability.data_ptr(), probability_out.data_ptr())
        torch.testing.assert_close(
            fused_probability, compact_probability, rtol=0, atol=0
        )
        scaled_probability_out = torch.empty(
            compact_probability.shape,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        scaled_probability = compact_vocab_probability(
            logits,
            compact_lse,
            local_start=local_start,
            local_stop=local_stop,
            output_dtype=torch.float8_e4m3fn,
            out=scaled_probability_out,
            store_scale=448.0,
        )
        scaled_reference = (
            torch.exp(
                logits[:, local_start:local_stop].float() - compact_lse[:, None]
            )
            * 448.0
        ).to(torch.float8_e4m3fn)
        torch.testing.assert_close(
            scaled_probability.float(), scaled_reference.float(), rtol=0, atol=0
        )
        reference_expected = (reference_probability @ local_weight).float()
        compact_expected = (fused_probability @ local_weight).float()
        reference_loss = (
            reference_lse - logits.gather(1, labels[:, None]).squeeze(1).float()
        ).mean()
        compact_loss = (
            compact_lse - logits.gather(1, labels[:, None]).squeeze(1).float()
        ).mean()

        torch.testing.assert_close(compact_loss, reference_loss, rtol=1e-6, atol=1e-6)
        # The LSE difference is at most one FP32 ulp.  After the serving BF16
        # probability cast, only a few output entries can differ by one BF16
        # accumulation ulp; this is substantially tighter than BF16 autograd.
        torch.testing.assert_close(
            compact_expected, reference_expected, rtol=2e-3, atol=2e-5
        )

    def test_end_to_end_mlp_norm_ce_gradients_match_autograd(self):
        num_tokens, hidden_size, intermediate_size, vocab_size = 5, 12, 20, 29
        x = torch.randn(
            num_tokens, hidden_size, device=self.device, dtype=torch.float32
        )
        residual = torch.randn_like(x)
        gate_up_weight = torch.nn.Parameter(
            torch.randn(
                2 * intermediate_size,
                hidden_size,
                device=self.device,
                dtype=torch.float32,
            )
        )
        down_weight = torch.nn.Parameter(
            torch.randn(
                hidden_size,
                intermediate_size,
                device=self.device,
                dtype=torch.float32,
            )
        )
        norm_weight = torch.nn.Parameter(
            torch.randn(hidden_size, device=self.device, dtype=torch.float32)
        )
        lm_head_weight = torch.randn(
            vocab_size, hidden_size, device=self.device, dtype=torch.float32
        )
        labels = torch.randint(0, vocab_size, (num_tokens,), device=self.device)

        gate_up = x @ gate_up_weight.transpose(0, 1)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        mlp_output = activated @ down_weight.transpose(0, 1)
        norm_input = mlp_output + residual
        inv_rms = torch.rsqrt(norm_input.square().mean(dim=-1, keepdim=True) + 1e-6)
        final_hidden = norm_input * inv_rms * (norm_weight + 1.0)
        logits = final_hidden @ lm_head_weight.transpose(0, 1)
        reference_loss = torch.nn.functional.cross_entropy(logits, labels)
        reference_loss.backward()

        class TupleLinear:
            def __init__(self, weight):
                self.weight = weight

            def __call__(self, value):
                return value @ self.weight.transpose(0, 1), None

        mlp = SimpleNamespace(
            gate_up_proj=TupleLinear(gate_up_weight.detach()),
            down_proj=SimpleNamespace(weight=down_weight.detach()),
        )
        final_norm = SimpleNamespace(weight=norm_weight.detach(), variance_epsilon=1e-6)
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)], norm=final_norm),
            lm_head=SimpleNamespace(weight=lm_head_weight),
        )
        trainer = OnlineMTPMLPTrainer(model, accumulation_tokens=num_tokens)
        logsumexp, expected_weight = vocab_ce_forward_stats(logits, lm_head_weight)
        tensors = {
            "step_0.mlp_input": x,
            "step_0.mlp_gate_up": gate_up.detach(),
            "step_0.final_norm_input": norm_input.detach(),
            "step_0.ce_logsumexp": logsumexp.detach(),
            "step_0.ce_expected_weight": expected_weight.detach(),
        }
        ticket = OnlineMTPActivationTicket(
            ticket_id=0,
            weight_version=0,
            num_tokens=num_tokens,
            tensors=tensors,
            bytes_reserved=0,
            state=TicketState.VERIFIED,
            labels=labels,
        )
        manual_loss = trainer.backward_ticket(
            ticket, target_weight=lm_head_weight[labels]
        )

        recompute_trainer = OnlineMTPMLPTrainer(
            model,
            accumulation_tokens=num_tokens,
            recompute_gate_up=True,
        )
        recompute_tensors = {
            name: value for name, value in tensors.items() if "mlp_gate_up" not in name
        }
        recompute_ticket = OnlineMTPActivationTicket(
            ticket_id=1,
            weight_version=0,
            num_tokens=num_tokens,
            tensors=recompute_tensors,
            bytes_reserved=0,
            state=TicketState.VERIFIED,
            labels=labels,
        )
        recompute_loss = recompute_trainer.backward_ticket(
            recompute_ticket, target_weight=lm_head_weight[labels]
        )

        torch.testing.assert_close(manual_loss, reference_loss, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(recompute_loss, manual_loss, rtol=0, atol=0)
        scale = float(num_tokens)
        torch.testing.assert_close(
            trainer.accumulator.gradients["mlp.gate_up_proj.weight"] / scale,
            gate_up_weight.grad,
            rtol=5e-5,
            atol=5e-5,
        )
        torch.testing.assert_close(
            trainer.accumulator.gradients["mlp.down_proj.weight"] / scale,
            down_weight.grad,
            rtol=5e-5,
            atol=5e-5,
        )
        torch.testing.assert_close(
            trainer.accumulator.gradients["final_norm.weight"] / scale,
            norm_weight.grad,
            rtol=5e-5,
            atol=5e-5,
        )
        for name, gradient in trainer.accumulator.gradients.items():
            torch.testing.assert_close(
                recompute_trainer.accumulator.gradients[name],
                gradient,
                rtol=0,
                atol=0,
            )

        factorized_trainer = OnlineMTPMLPTrainer(
            model,
            accumulation_tokens=num_tokens,
            factorized_gradient_accumulation=True,
        )
        factorized_loss = factorized_trainer.backward_ticket(
            ticket, target_weight=lm_head_weight[labels]
        )
        torch.testing.assert_close(
            factorized_loss, reference_loss, rtol=2e-5, atol=2e-5
        )
        self.assertEqual(
            factorized_trainer.accumulator.gradients[
                "mlp.gate_up_proj.weight"
            ].count_nonzero(),
            0,
        )
        factorized_trainer.materialize_factorized_gradients()
        torch.testing.assert_close(
            factorized_trainer.accumulator.gradients["mlp.gate_up_proj.weight"] / scale,
            gate_up_weight.grad,
            rtol=5e-5,
            atol=5e-5,
        )
        torch.testing.assert_close(
            factorized_trainer.accumulator.gradients["mlp.down_proj.weight"] / scale,
            down_weight.grad,
            rtol=5e-5,
            atol=5e-5,
        )
        torch.testing.assert_close(
            factorized_trainer.accumulator.gradients["final_norm.weight"] / scale,
            norm_weight.grad,
            rtol=5e-5,
            atol=5e-5,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA BF16 GEMMs")
    def test_end_to_end_bf16_serving_gradients_match_autograd(self):
        """Exercise the dtypes and saved activations used by live serving."""

        num_tokens, hidden_size, intermediate_size, vocab_size = 7, 32, 48, 67
        dtype = torch.bfloat16
        x = (0.1 * torch.randn(num_tokens, hidden_size, device="cuda")).to(dtype)
        residual = (0.1 * torch.randn_like(x)).to(dtype)
        gate_up_weight = torch.nn.Parameter(
            (0.1 * torch.randn(2 * intermediate_size, hidden_size, device="cuda")).to(
                dtype
            )
        )
        down_weight = torch.nn.Parameter(
            (0.1 * torch.randn(hidden_size, intermediate_size, device="cuda")).to(dtype)
        )
        norm_weight = torch.nn.Parameter(
            (0.1 * torch.randn(hidden_size, device="cuda")).to(dtype)
        )
        lm_head_weight = (0.1 * torch.randn(vocab_size, hidden_size, device="cuda")).to(
            dtype
        )
        labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda")

        gate_up = x @ gate_up_weight.transpose(0, 1)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        mlp_output = activated @ down_weight.transpose(0, 1)
        norm_input = mlp_output + residual
        inv_rms = torch.rsqrt(
            norm_input.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        final_hidden = (norm_input.float() * inv_rms * (norm_weight.float() + 1.0)).to(
            dtype
        )
        logits = final_hidden @ lm_head_weight.transpose(0, 1)
        reference_loss = torch.nn.functional.cross_entropy(logits.float(), labels)
        reference_loss.backward()

        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=gate_up_weight.detach()),
            down_proj=SimpleNamespace(weight=down_weight.detach()),
        )
        final_norm = SimpleNamespace(weight=norm_weight.detach(), variance_epsilon=1e-6)
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)], norm=final_norm),
            lm_head=SimpleNamespace(weight=lm_head_weight),
        )
        trainer = OnlineMTPMLPTrainer(model, accumulation_tokens=num_tokens)
        probabilities = torch.softmax(logits.detach().float(), dim=-1).to(dtype)
        expected_weight = (probabilities @ lm_head_weight).float().detach()
        tensors = {
            "step_0.mlp_input": x,
            "step_0.mlp_gate_up": gate_up.detach(),
            "step_0.final_norm_input": norm_input.detach(),
            "step_0.ce_logsumexp": torch.logsumexp(logits.detach().float(), dim=-1),
            "step_0.ce_expected_weight": expected_weight,
        }
        ticket = OnlineMTPActivationTicket(
            ticket_id=0,
            weight_version=0,
            num_tokens=num_tokens,
            tensors=tensors,
            bytes_reserved=0,
            state=TicketState.VERIFIED,
            labels=labels,
        )
        manual_loss = trainer.backward_ticket(
            ticket, target_weight=lm_head_weight[labels].float()
        )

        fused_trainer = OnlineMTPMLPTrainer(
            model,
            accumulation_tokens=num_tokens,
            use_triton_swiglu=True,
            use_triton_ce_rmsnorm=True,
        )
        fused_ticket = OnlineMTPActivationTicket(
            ticket_id=1,
            weight_version=0,
            num_tokens=num_tokens,
            tensors={name: value.clone() for name, value in tensors.items()},
            bytes_reserved=0,
            state=TicketState.VERIFIED,
            labels=labels,
        )
        fused_loss = fused_trainer.backward_ticket(
            fused_ticket, target_weight=lm_head_weight[labels]
        )
        torch.testing.assert_close(fused_loss, manual_loss, rtol=3e-4, atol=2e-4)

        scale = float(num_tokens)
        torch.testing.assert_close(
            trainer.accumulator.gradients["mlp.gate_up_proj.weight"] / scale,
            gate_up_weight.grad,
            rtol=5e-2,
            atol=2e-3,
        )
        torch.testing.assert_close(
            trainer.accumulator.gradients["mlp.down_proj.weight"] / scale,
            down_weight.grad,
            rtol=5e-2,
            atol=2e-3,
        )
        torch.testing.assert_close(
            trainer.accumulator.gradients["final_norm.weight"] / scale,
            norm_weight.grad,
            rtol=5e-2,
            atol=2e-3,
        )
        for name, gradient in trainer.accumulator.gradients.items():
            torch.testing.assert_close(
                fused_trainer.accumulator.gradients[name],
                gradient,
                rtol=5e-2,
                atol=2e-3,
            )

    def test_activation_ring_reports_capacity_without_overcommit(self):
        tensor = torch.zeros(8, device=self.device)
        ring = ActivationTicketRing(
            max_bytes=tensor.numel() * tensor.element_size(), max_tickets=1
        )
        first = ring.try_stage(
            {"x": tensor}, weight_version=3, num_tokens=2, clone=True
        )
        self.assertIsNotNone(first)
        second = ring.try_stage(
            {"x": tensor}, weight_version=3, num_tokens=2, clone=True
        )
        self.assertIsNone(second)
        self.assertFalse(
            ring.can_stage(bytes_reserved=tensor.numel() * tensor.element_size())
        )
        labels = torch.tensor([1, 2], device=self.device)
        ring.mark_verified(first.ticket_id, labels)
        ready = ring.pop_verified()
        self.assertEqual(ready.state, TicketState.VERIFIED)
        ring.release(ready.ticket_id, backwarded=True)
        self.assertEqual(ring.live_bytes, 0)

    def test_fused_ce_transient_reservation_formula(self):
        rows, hidden_size = 6, 8
        # FP32 reduction: 6*(8+1)=54 elements, padded to 56 => 224 B.
        reduced_bytes = 224
        # Two worst-case cross-ticket BF16 hidden concatenations.
        hidden_concat_bytes = 2 * rows * hidden_size * 2
        loss_mask_bytes = rows
        mean_loss_bytes = 4
        self.assertEqual(
            fused_ce_transient_reservation_bytes(
                num_tokens=rows,
                hidden_size=hidden_size,
                activation_element_size=2,
            ),
            reduced_bytes
            + hidden_concat_bytes
            + loss_mask_bytes
            + mean_loss_bytes,
        )

    def test_fused_ce_stage_accounts_complete_async_working_set(self):
        rows, hidden_size, intermediate_size, vocab_size = 6, 8, 4, 13
        dtype = torch.bfloat16
        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                weight=torch.zeros(2 * intermediate_size, hidden_size, dtype=dtype)
            ),
            down_proj=SimpleNamespace(
                weight=torch.zeros(hidden_size, intermediate_size, dtype=dtype)
            ),
        )
        final_norm = SimpleNamespace(
            weight=torch.zeros(hidden_size, dtype=dtype), variance_epsilon=1e-6
        )
        lm_head = SimpleNamespace(
            weight=torch.zeros(vocab_size, hidden_size, dtype=dtype),
            shard_indices=SimpleNamespace(
                org_vocab_start_index=0, org_vocab_end_index=vocab_size
            ),
        )
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)], norm=final_norm),
            lm_head=lm_head,
        )
        runtime = OnlineMTPRuntime(
            model,
            activation_bytes=1 << 20,
            max_tickets=4,
            backward_batch_tokens=1,
            accumulation_tokens=32,
            group_ce_projection=True,
            fused_ce_reduction=True,
            use_triton_ce_rmsnorm=True,
            async_backward=False,
        )
        tap = OnlineMTPActivationTap(
            weight_version=0,
            clone_tensors=False,
            group_ce_projection=True,
            fused_ce_reduction=True,
        )
        ids = torch.arange(rows)
        tap.begin_step(0, input_ids=ids, positions=ids)
        tap.record("mlp_input", torch.zeros(rows, hidden_size, dtype=dtype))
        tap.record(
            "mlp_gate_up", torch.zeros(rows, 2 * intermediate_size, dtype=dtype)
        )
        tap.record("final_norm_input", torch.zeros(rows, hidden_size, dtype=dtype))
        tap.record("ce_logsumexp", torch.zeros(rows, dtype=torch.float32))
        tap.record(
            "ce_local_expected_weight",
            torch.zeros(rows, hidden_size, dtype=dtype),
        )
        tensor_bytes = sum(value.numel() * value.element_size() for value in tap.flatten().values())
        ticket_id = runtime.stage(tap)
        self.assertIsNotNone(ticket_id)
        transient_bytes = fused_ce_transient_reservation_bytes(
            num_tokens=rows,
            hidden_size=hidden_size,
            activation_element_size=torch.tensor([], dtype=dtype).element_size(),
        )
        expected_bytes = tensor_bytes + transient_bytes
        self.assertEqual(runtime.last_ticket_bytes, expected_bytes)
        self.assertEqual(runtime.ring.get(ticket_id).bytes_reserved, expected_bytes)
        runtime.ring.release(ticket_id)

    def test_fused_ce_working_set_capacity_applies_backpressure_before_prepare(self):
        rows, hidden_size = 6, 8
        model, _, fixture = self._fused_ce_fixture(
            device="cpu",
            dtype=torch.bfloat16,
            rows=rows,
            hidden_size=hidden_size,
            intermediate_size=4,
            vocab_size=13,
        )
        tap = self._tap_from_fused_ce_fixture(fixture)
        tensor_bytes = sum(
            value.numel() * value.element_size()
            for value in tap.flatten().values()
        )
        expected_bytes = tensor_bytes + fused_ce_transient_reservation_bytes(
            num_tokens=rows,
            hidden_size=hidden_size,
            activation_element_size=2,
        )

        def runtime(capacity):
            return OnlineMTPRuntime(
                model,
                activation_bytes=capacity,
                max_tickets=4,
                backward_batch_tokens=rows,
                accumulation_tokens=64,
                group_ce_projection=True,
                fused_ce_reduction=True,
                use_triton_ce_rmsnorm=True,
                async_backward=False,
            )

        too_small = runtime(expected_bytes - 1)
        with self.assertRaisesRegex(RuntimeError, "exceeding the configured"):
            too_small.stage(self._tap_from_fused_ce_fixture(fixture))

        exact = runtime(expected_bytes)
        first_id = exact.stage(self._tap_from_fused_ce_fixture(fixture))
        self.assertIsNotNone(first_id)
        exact.ring.mark_verified(first_id, fixture.labels)
        exact.ring.pop_all_verified()
        exact._pending.append(
            SimpleNamespace(
                ticket_ids=(first_id,),
                training_batch_id=0,
                done_event=SimpleNamespace(
                    synchronize=lambda: None,
                    query=lambda: True,
                ),
                mean_loss=torch.tensor(0.0),
            )
        )
        second_id = exact.stage(self._tap_from_fused_ce_fixture(fixture))
        self.assertIsNotNone(second_id)
        self.assertEqual(exact.backpressure_events, 1)
        self.assertEqual(exact.ring.live_tickets, 1)
        self.assertEqual(exact.ring.live_bytes, expected_bytes)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA streams")
    def test_cuda_graph_tap_snapshot_owns_unpadded_rows(self):
        template = OnlineMTPActivationTap(
            weight_version=2,
            clone_tensors=False,
        )
        source_ids = torch.arange(5, device="cuda")
        source_activation = torch.arange(20, device="cuda").reshape(5, 4)
        template.begin_step(0, input_ids=source_ids, positions=source_ids)
        template.record("mlp_input", source_activation)

        destination = OnlineMTPActivationTap(weight_version=2)
        copy_stream = torch.cuda.Stream()
        destination.copy_from_graph_template(
            template,
            num_rows=3,
            copy_stream=copy_stream,
        )
        self.assertIsNotNone(destination.ready_event)
        # Mutation is queued immediately on the serving stream.  The snapshot
        # method's device-side wait must order it after the asynchronous copy.
        source_ids.add_(100)
        source_activation.add_(100)
        torch.cuda.synchronize()
        self.assertEqual(destination.step_rows[0], 3)
        self.assertEqual(
            destination.steps[0]["mlp_input"].tolist(),
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA streams")
    def test_cuda_graph_snapshot_steps_are_zero_copy_concatenable(self):
        template = OnlineMTPActivationTap(weight_version=2, clone_tensors=False)
        for step in range(3):
            source = torch.arange(20, device="cuda").reshape(5, 4) + 100 * step
            ids = torch.arange(5, device="cuda") + 10 * step
            template.begin_step(step, input_ids=ids, positions=ids)
            template.record("mlp_input", source)

        destination = OnlineMTPActivationTap(weight_version=2)
        destination.copy_from_graph_template(
            template,
            num_rows=3,
            copy_stream=torch.cuda.Stream(),
        )
        rows = concatenate_adjacent_rows(
            destination.steps[step]["mlp_input"] for step in range(3)
        )
        self.assertEqual(rows.shape, (9, 4))
        self.assertEqual(
            rows.untyped_storage().data_ptr(),
            destination.steps[0]["mlp_input"].untyped_storage().data_ptr(),
        )
        torch.testing.assert_close(
            rows,
            torch.cat([destination.steps[step]["mlp_input"] for step in range(3)]),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA streams")
    def test_cuda_graph_snapshot_serving_path_uses_dtype_slabs(self):
        template = OnlineMTPActivationTap(weight_version=2, clone_tensors=False)
        expected = {}
        for step in range(3):
            ids = torch.arange(5, device="cuda") + 10 * step
            template.begin_step(step, input_ids=ids, positions=ids)
            for name, width, dtype in (
                ("mlp_input", 8, torch.bfloat16),
                ("final_norm_input", 8, torch.bfloat16),
                ("ce_logsumexp", None, torch.float32),
            ):
                shape = (5,) if width is None else (5, width)
                value = (
                    torch.arange(math.prod(shape), device="cuda")
                    .reshape(shape)
                    .to(dtype)
                    + 100 * step
                )
                template.record(name, value)
                expected[(step, name)] = value[:3].clone()

        destination = OnlineMTPActivationTap(weight_version=2)
        with patch("torch._foreach_copy_", wraps=torch._foreach_copy_) as foreach:
            destination.copy_from_graph_template(template, num_rows=3)
        self.assertIsNone(destination.ready_event)
        self.assertEqual(foreach.call_count, 2)

        # A later serving-stream mutation cannot overtake the snapshot.
        for step in template.steps.values():
            for value in step.values():
                value.add_(1000)
        torch.cuda.synchronize()
        for (step, name), value in expected.items():
            self.assertTrue(torch.equal(destination.steps[step][name], value))

        bf16_pointers = {
            destination.steps[step][name].untyped_storage().data_ptr()
            for step in range(3)
            for name in ("mlp_input", "final_norm_input")
        }
        fp32_pointers = {
            destination.steps[step]["ce_logsumexp"]
            .untyped_storage()
            .data_ptr()
            for step in range(3)
        }
        self.assertEqual(len(bf16_pointers), 1)
        self.assertEqual(len(fp32_pointers), 1)
        self.assertTrue(bf16_pointers.isdisjoint(fp32_pointers))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_swiglu_forward_out_reuses_dead_contiguous_storage(self):
        gate_up = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
        storage = torch.empty(7, 64, device="cuda", dtype=torch.bfloat16)
        out = storage.reshape(-1)[: 7 * 16].view(7, 16)
        from sglang.jit_kernel.activation import silu_and_mul

        expected = silu_and_mul(gate_up)
        actual = silu_and_mul_forward_out(gate_up, out)
        self.assertEqual(
            actual.untyped_storage().data_ptr(), storage.untyped_storage().data_ptr()
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_combining_verified_tickets_preserves_step_and_label_order(self):
        def ticket(ticket_id, step_rows, labels):
            tensors = {}
            for step_id, rows in enumerate(step_rows):
                tensors[f"step_{step_id}.mlp_input"] = torch.tensor(
                    rows, device=self.device
                )[:, None]
            return OnlineMTPActivationTicket(
                ticket_id=ticket_id,
                weight_version=4,
                num_tokens=sum(len(rows) for rows in step_rows),
                tensors=tensors,
                bytes_reserved=0,
                state=TicketState.VERIFIED,
                labels=torch.tensor(labels, device=self.device),
                metadata={"fused_ce_reduction": True},
            )

        first = ticket(1, [[1], [2]], [10, 11])
        second = ticket(2, [[3, 4], [5, 6]], [20, 21, 22, 23])
        combined = combine_verified_tickets([first, second])

        self.assertEqual(combined.num_tokens, 6)
        self.assertEqual(combined.metadata["combined_tickets"], 2)
        self.assertTrue(combined.metadata["fused_ce_reduction"])
        self.assertEqual(combined.labels.tolist(), [10, 11, 20, 21, 22, 23])
        ordered_inputs = torch.cat(
            [combined.tensors[f"step_{step}.mlp_input"] for step in range(4)]
        ).reshape(-1)
        self.assertEqual(ordered_inputs.tolist(), [1, 2, 3, 4, 5, 6])

    def test_backward_group_ticket_cap_flushes_below_token_threshold(self):
        hidden_size, intermediate_size, vocab_size = 4, 3, 7
        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                weight=torch.zeros(2 * intermediate_size, hidden_size)
            ),
            down_proj=SimpleNamespace(
                weight=torch.zeros(hidden_size, intermediate_size)
            ),
        )
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(mlp=mlp)],
                norm=SimpleNamespace(
                    weight=torch.zeros(hidden_size), variance_epsilon=1e-6
                ),
            ),
            lm_head=SimpleNamespace(weight=torch.zeros(vocab_size, hidden_size)),
        )
        runtime = OnlineMTPRuntime(
            model,
            activation_bytes=1 << 20,
            max_tickets=32,
            backward_batch_tokens=100,
            backward_max_group_tickets=2,
            accumulation_tokens=128,
            async_backward=False,
        )
        runtime.max_verified_ticket_tokens = 25

        def stage_verified(label, rows=1):
            ticket = runtime.ring.try_stage(
                {"step_0.mlp_input": torch.zeros(rows, hidden_size)},
                weight_version=0,
                num_tokens=rows,
                clone=False,
            )
            self.assertIsNotNone(ticket)
            runtime.ring.mark_verified(
                ticket.ticket_id,
                torch.full((rows,), label),
            )

        with patch.object(
            runtime.trainer,
            "backward_ticket",
            return_value=torch.tensor(0.0),
        ) as backward:
            stage_verified(1)
            self.assertIsNone(runtime._maybe_launch_backward_group())
            self.assertEqual(runtime.ring.ready_tickets, 1)
            stage_verified(2)
            self.assertEqual(runtime._maybe_launch_backward_group(), 0.0)

        backward.assert_called_once()
        self.assertEqual(runtime.backward_batches, 1)
        self.assertEqual(runtime.backward_tickets, 2)
        self.assertEqual(runtime.ring.live_tickets, 0)

        runtime.backward_max_group_tickets = 16
        runtime.max_verified_ticket_tokens = 0
        with patch.object(
            runtime.trainer,
            "backward_ticket",
            return_value=torch.tensor(0.0),
        ) as backward:
            for label in range(23):
                stage_verified(label % vocab_size)
                self.assertIsNone(runtime._maybe_launch_backward_group())
            stage_verified(1)
            self.assertEqual(runtime._maybe_launch_backward_group(), 0.0)
        backward.assert_called_once()

        # A 24-token scheduler ticket is already at the low-latency production
        # threshold, so it launches at 24 tokens instead of using the 16-ticket
        # grouping cap.
        runtime.max_verified_ticket_tokens = 24
        with patch.object(
            runtime.trainer,
            "backward_ticket",
            return_value=torch.tensor(0.0),
        ) as backward:
            for label in range(23):
                stage_verified(label % vocab_size)
                self.assertIsNone(runtime._maybe_launch_backward_group())
            stage_verified(1)
            self.assertEqual(runtime._maybe_launch_backward_group(), 0.0)
        backward.assert_called_once()
        self.assertEqual(runtime.backward_batches, 3)
        self.assertEqual(runtime.backward_tickets, 50)
        self.assertEqual(runtime.ring.live_tickets, 0)

        # A ticket between 24 and the configured token target unlocks the
        # larger grouping window, capped here at 16 tickets.
        runtime.max_verified_ticket_tokens = 25
        with patch.object(
            runtime.trainer,
            "backward_ticket",
            return_value=torch.tensor(0.0),
        ) as backward:
            for label in range(15):
                stage_verified(label % vocab_size)
                self.assertIsNone(runtime._maybe_launch_backward_group())
            stage_verified(1)
            self.assertEqual(runtime._maybe_launch_backward_group(), 0.0)
        backward.assert_called_once()
        self.assertEqual(runtime.backward_batches, 4)
        self.assertEqual(runtime.backward_tickets, 66)
        self.assertEqual(runtime.ring.live_tickets, 0)

        # Once one scheduler ticket itself reaches the configured token target,
        # only its tail remains. The tail uses the established 24-token launch
        # threshold instead of waiting for the larger ticket cap.
        runtime.backward_max_group_tickets = 32
        runtime.max_verified_ticket_tokens = 100
        with patch.object(
            runtime.trainer,
            "backward_ticket",
            return_value=torch.tensor(0.0),
        ) as backward:
            for label in range(23):
                stage_verified(label % vocab_size)
                self.assertIsNone(runtime._maybe_launch_backward_group())
            stage_verified(1)
            self.assertEqual(runtime._maybe_launch_backward_group(), 0.0)
        backward.assert_called_once()
        self.assertEqual(runtime.backward_batches, 5)
        self.assertEqual(runtime.backward_tickets, 90)
        self.assertEqual(runtime.ring.live_tickets, 0)

        # The opt-in inter-verify pipeline may aggregate sub-24-row tickets to
        # a larger complete-data group, reducing launch frequency without
        # dropping any ready ticket. Larger scheduler tickets still retain the
        # established adaptive thresholds above.
        runtime.pipelined_ce_group_tokens = 48
        runtime.max_verified_ticket_tokens = 3
        with patch.object(
            runtime.trainer,
            "backward_ticket",
            return_value=torch.tensor(0.0),
        ) as backward:
            for label in range(15):
                stage_verified(label % vocab_size, rows=3)
                self.assertIsNone(runtime._maybe_launch_backward_group())
            stage_verified(1, rows=3)
            self.assertEqual(runtime._maybe_launch_backward_group(), 0.0)
        backward.assert_called_once()
        self.assertEqual(runtime.backward_batches, 6)
        self.assertEqual(runtime.backward_tickets, 106)
        self.assertEqual(runtime.ring.live_tickets, 0)

    def test_delayed_adamw_apply_keeps_parameters_fixed_until_boundary(self):
        parameter = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], device=self.device, dtype=torch.float32)
        )
        parameter.requires_grad_(False)
        accumulator = FlatAdamWAccumulator([("p", parameter)], accumulation_tokens=4)
        self.assertIsNone(accumulator.first_moments)
        original = parameter.detach().clone()
        accumulator.accumulate(
            {"p": torch.tensor([2.0, -4.0], device=self.device)}, num_tokens=2
        )
        torch.testing.assert_close(parameter, original)
        self.assertFalse(accumulator.ready)
        accumulator.accumulate(
            {"p": torch.tensor([2.0, -4.0], device=self.device)}, num_tokens=2
        )
        self.assertTrue(accumulator.ready)
        grad_norm = accumulator.apply_(learning_rate=0.1, betas=(0.0, 0.0))
        self.assertGreater(grad_norm, 0.0)
        self.assertEqual(accumulator.tokens, 0)
        self.assertFalse(torch.equal(parameter, original))

    def test_adamw_prepare_keeps_active_weight_fixed_until_publish(self):
        initial = torch.tensor([1.0, -2.0], device=self.device)
        direct_parameter = torch.nn.Parameter(initial.clone(), requires_grad=False)
        staged_parameter = torch.nn.Parameter(initial.clone(), requires_grad=False)
        direct = FlatAdamWAccumulator(
            [("weight", direct_parameter)], accumulation_tokens=2
        )
        staged = FlatAdamWAccumulator(
            [("weight", staged_parameter)], accumulation_tokens=2
        )
        gradient = torch.tensor([0.25, -0.5], device=self.device)
        direct.accumulate({"weight": gradient}, num_tokens=2)
        staged.accumulate({"weight": gradient}, num_tokens=2)

        direct_norm = direct.apply_(learning_rate=1e-2)
        staged_norm = staged.prepare_(learning_rate=1e-2)
        torch.testing.assert_close(staged_parameter, initial)
        self.assertTrue(staged.has_prepared_update)

        staged.publish_()
        torch.testing.assert_close(staged_parameter, direct_parameter)
        self.assertEqual(staged_norm, direct_norm)
        self.assertFalse(staged.has_prepared_update)

    def test_adamw_uses_global_grad_norm_override_for_clipping(self):
        parameter = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], device=self.device, dtype=torch.float32),
            requires_grad=False,
        )
        accumulator = FlatAdamWAccumulator([("p", parameter)], accumulation_tokens=1)
        accumulator.accumulate(
            {"p": torch.tensor([3.0, 4.0], device=self.device)}, num_tokens=1
        )
        # Adam with beta1=beta2=0 normally erases the magnitude of a scalar
        # gradient.  A nonzero epsilon leaves this test sensitive to the
        # clipping scale, hence to the externally supplied TP-global norm.
        accumulator.apply_(
            learning_rate=0.1,
            betas=(0.0, 0.0),
            eps=1.0,
            max_grad_norm=1.0,
            grad_norm_override=10.0,
        )
        expected = torch.tensor(
            [1.0 - 0.1 * 0.3 / 1.3, -2.0 - 0.1 * 0.4 / 1.4],
            device=self.device,
        )
        torch.testing.assert_close(parameter, expected)

    def test_optimizer_warmup_is_state_and_weight_neutral(self):
        parameter = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], device=self.device, dtype=torch.float32),
            requires_grad=False,
        )
        accumulator = FlatAdamWAccumulator([("p", parameter)], accumulation_tokens=1)
        original = parameter.detach().clone()
        accumulator.warmup_optimizer_kernels_()
        torch.testing.assert_close(parameter, original)
        torch.testing.assert_close(accumulator.master_parameters["p"], original)
        self.assertEqual(accumulator.step, 0)
        self.assertEqual(accumulator.tokens, 0)
        self.assertEqual(accumulator.first_moments["p"].count_nonzero().item(), 0)
        self.assertEqual(accumulator.second_moments["p"].count_nonzero().item(), 0)

    def test_fused_optimizer_warmup_covers_norm_and_weight_decay(self):
        parameter = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], dtype=torch.float32), requires_grad=False
        )
        accumulator = FlatAdamWAccumulator(
            [("p", parameter)],
            accumulation_tokens=1,
            use_triton_optimizer=True,
        )

        def fake_apply(**kwargs):
            self.assertEqual(kwargs["weight_decay"], 0.125)
            accumulator.tokens = 0
            return 0.0

        with (
            patch.object(
                accumulator,
                "scaled_gradient_norm_sq",
                return_value=torch.zeros((), dtype=torch.float32),
            ) as norm,
            patch.object(accumulator, "apply_", side_effect=fake_apply) as apply,
        ):
            accumulator.warmup_optimizer_kernels_(weight_decay=0.125)
        norm.assert_called_once_with()
        apply.assert_called_once()

    def test_sync_update_refreshes_gemma_derived_weight(self):
        hidden_size, intermediate_size, vocab_size = 4, 3, 7
        mlp = SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                weight=torch.zeros(2 * intermediate_size, hidden_size)
            ),
            down_proj=SimpleNamespace(
                weight=torch.zeros(hidden_size, intermediate_size)
            ),
        )
        final_norm = SimpleNamespace(
            weight=torch.zeros(hidden_size),
            gemma_weight=torch.ones(hidden_size),
            variance_epsilon=1e-6,
        )
        lm_head = SimpleNamespace(weight=torch.zeros(vocab_size, hidden_size))
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)], norm=final_norm),
            lm_head=lm_head,
        )
        runtime = OnlineMTPRuntime(
            model,
            activation_bytes=1 << 20,
            max_tickets=2,
            backward_batch_tokens=1,
            accumulation_tokens=1,
            async_backward=False,
            apply_updates=False,
        )
        runtime.apply_updates = True
        for gradient in runtime.trainer.accumulator.gradients.values():
            gradient.fill_(0.25)
        runtime.trainer.accumulator.tokens = 1
        runtime._update_requested = True
        with patch.object(
            runtime.trainer, "tensor_parallel_grad_norm", return_value=0.5
        ):
            self.assertTrue(runtime._try_publish_update())
        torch.testing.assert_close(
            final_norm.gemma_weight, final_norm.weight + 1.0, rtol=0, atol=0
        )
        self.assertEqual(runtime.weight_version, 1)


if __name__ == "__main__":
    unittest.main()
