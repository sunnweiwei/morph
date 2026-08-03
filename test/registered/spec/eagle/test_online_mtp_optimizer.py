import unittest

import torch

from sglang.srt.speculative.online_mtp_training import FlatAdamWAccumulator
from sglang.srt.speculative.triton_ops.online_mtp_adamw import (
    grad_norm_num_partials,
    online_mtp_adamw_prepare_,
    online_mtp_scaled_grad_norm_sq,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")


class TestOnlineMTPFusedOptimizer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        torch.manual_seed(41)

    def _accumulator(self, *, fused: bool, gradient_dtype=None, direct=False):
        gradient_dtype = gradient_dtype or self.dtype
        first = torch.nn.Parameter(
            torch.randn(5, 37, device=self.device, dtype=self.dtype),
            requires_grad=False,
        )
        second = torch.nn.Parameter(
            torch.randn(1031, device=self.device, dtype=self.dtype),
            requires_grad=False,
        )
        return FlatAdamWAccumulator(
            (("first", first), ("second", second)),
            accumulation_tokens=17,
            gradient_dtype=gradient_dtype,
            use_triton_optimizer=fused,
            direct_parameter_update=direct,
        )

    def test_direct_publish_matches_shadow_publish_exactly(self):
        shadow = self._accumulator(fused=True)
        direct = self._accumulator(fused=True, direct=True)
        for name in shadow.parameters:
            direct.parameters[name].copy_(shadow.parameters[name])

        gradients = {
            name: torch.randn_like(buffer)
            for name, buffer in shadow.gradients.items()
        }
        for accumulator in (shadow, direct):
            accumulator.accumulate(gradients, num_tokens=17)
            accumulator.prepare_(
                learning_rate=3e-4,
                betas=(0.85, 0.97),
                eps=1e-6,
                weight_decay=0.1,
                max_grad_norm=0.7,
                grad_norm_override=0.5,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.assertTrue(direct.direct_parameter_update)
        self.assertFalse(direct.reuse_gradient_for_prepared)
        for name in shadow.parameters:
            self.assertEqual(
                direct.prepared_parameters[name].data_ptr(),
                direct.parameters[name].data_ptr(),
            )
            torch.testing.assert_close(
                direct.parameters[name], shadow.prepared_parameters[name],
                rtol=0.0, atol=0.0,
            )
            torch.testing.assert_close(
                direct.first_moments[name], shadow.first_moments[name],
                rtol=0.0, atol=0.0,
            )
            torch.testing.assert_close(
                direct.second_moments[name], shadow.second_moments[name],
                rtol=0.0, atol=0.0,
            )
            torch.testing.assert_close(
                direct.master_parameters[name], shadow.master_parameters[name],
                rtol=0.0, atol=0.0,
            )
            self.assertEqual(direct.gradients[name].count_nonzero().item(), 0)

        shadow.publish_()
        direct.publish_()
        for name in shadow.parameters:
            torch.testing.assert_close(
                direct.parameters[name], shadow.parameters[name], rtol=0.0, atol=0.0
            )

    def test_direct_publish_requires_triton_optimizer(self):
        with self.assertRaisesRegex(ValueError, "requires the Triton optimizer"):
            self._accumulator(fused=False, direct=True)

    def test_fused_prepare_matches_reference_across_steps_and_clipping(self):
        reference = self._accumulator(fused=False)
        fused = self._accumulator(fused=True)
        for name in reference.parameters:
            fused.parameters[name].copy_(reference.parameters[name])

        for step in range(1, 4):
            gradients = {
                name: torch.randn_like(buffer)
                for name, buffer in reference.gradients.items()
            }
            for accumulator in (reference, fused):
                accumulator.accumulate(gradients, num_tokens=11)
                accumulator.accumulate(
                    {name: gradient * 0.25 for name, gradient in gradients.items()},
                    num_tokens=6,
                )
            reference_before = {
                name: parameter.clone()
                for name, parameter in reference.parameters.items()
            }
            fused_before = {
                name: parameter.clone() for name, parameter in fused.parameters.items()
            }
            reference_norm = reference.prepare_(
                learning_rate=3e-4,
                betas=(0.85, 0.97),
                eps=1e-6,
                weight_decay=0.1,
                max_grad_norm=0.7,
            )
            fused_norm = fused.prepare_(
                learning_rate=3e-4,
                betas=(0.85, 0.97),
                eps=1e-6,
                weight_decay=0.1,
                max_grad_norm=0.7,
            )
            self.assertAlmostEqual(reference_norm, fused_norm, places=4)
            for name in reference.parameters:
                torch.testing.assert_close(
                    reference.parameters[name], reference_before[name]
                )
                torch.testing.assert_close(fused.parameters[name], fused_before[name])
                torch.testing.assert_close(
                    fused.first_moments[name],
                    reference.first_moments[name],
                    rtol=3e-6,
                    atol=3e-7,
                )
                torch.testing.assert_close(
                    fused.second_moments[name],
                    reference.second_moments[name],
                    rtol=3e-6,
                    atol=3e-7,
                )
                torch.testing.assert_close(
                    fused.master_parameters[name],
                    reference.master_parameters[name],
                    rtol=3e-6,
                    atol=3e-7,
                )
                torch.testing.assert_close(
                    fused.prepared_parameters[name],
                    reference.prepared_parameters[name],
                    rtol=0.0,
                    atol=0.0,
                )

            reference.publish_()
            fused.publish_()
            for name in reference.parameters:
                torch.testing.assert_close(
                    fused.parameters[name], reference.parameters[name]
                )
                self.assertEqual(fused.gradients[name].count_nonzero().item(), 0)
            self.assertEqual(fused.step, step)

    def test_prepared_shadow_aliases_consumed_matching_dtype_gradients(self):
        accumulator = self._accumulator(fused=True)
        accumulator.accumulate(
            {
                name: torch.ones_like(gradient)
                for name, gradient in accumulator.gradients.items()
            },
            num_tokens=17,
        )
        original_ptrs = {
            name: parameter.data_ptr()
            for name, parameter in accumulator.parameters.items()
        }
        accumulator.prepare_(learning_rate=1e-4, grad_norm_override=1.0)

        self.assertTrue(accumulator.reuse_gradient_for_prepared)
        for name in accumulator.parameters:
            self.assertEqual(
                accumulator.prepared_parameters[name].data_ptr(),
                accumulator.gradients[name].data_ptr(),
            )
            self.assertEqual(
                accumulator.parameters[name].data_ptr(), original_ptrs[name]
            )
        with self.assertRaisesRegex(RuntimeError, "prepared weights"):
            accumulator.zero()
        with self.assertRaisesRegex(RuntimeError, "prepared weights"):
            accumulator.accumulate(
                {"first": torch.ones_like(accumulator.gradients["first"])},
                num_tokens=1,
            )

        accumulator.publish_()
        for name in accumulator.parameters:
            self.assertEqual(
                accumulator.parameters[name].data_ptr(), original_ptrs[name]
            )
            self.assertEqual(accumulator.gradients[name].count_nonzero().item(), 0)

    def test_distinct_gradient_dtype_keeps_distinct_shadow_and_clears_prepare(self):
        if self.dtype == torch.float32:
            self.skipTest("requires a serving dtype distinct from FP32 gradients")
        accumulator = self._accumulator(fused=True, gradient_dtype=torch.float32)
        accumulator.accumulate(
            {
                name: torch.randn_like(gradient)
                for name, gradient in accumulator.gradients.items()
            },
            num_tokens=17,
        )
        accumulator.prepare_(learning_rate=1e-4, grad_norm_override=1.0)
        self.assertFalse(accumulator.reuse_gradient_for_prepared)
        for name in accumulator.parameters:
            self.assertNotEqual(
                accumulator.prepared_parameters[name].data_ptr(),
                accumulator.gradients[name].data_ptr(),
            )
            self.assertEqual(accumulator.gradients[name].count_nonzero().item(), 0)

    def test_compact_scaled_grad_norm_matches_reference(self):
        gradients = (
            torch.randn(2053, device=self.device, dtype=self.dtype),
            torch.randn(31, 17, device=self.device, dtype=self.dtype),
        )
        scales = (1.0, 0.125)
        required = grad_norm_num_partials(gradients)
        scratch = torch.full(
            (required + 3,), -123.0, dtype=torch.float32, device=self.device
        )
        actual = online_mtp_scaled_grad_norm_sq(
            gradients,
            denominator=37.0,
            contribution_scales=scales,
            partials_out=scratch,
        )
        expected = sum(
            scale * (gradient.float() / 37.0).square().sum()
            for gradient, scale in zip(gradients, scales)
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        if self.device.type == "cuda":
            self.assertEqual(required, 4)
            torch.testing.assert_close(
                scratch[required:],
                torch.full_like(scratch[required:], -123.0),
            )

    def test_grad_norm_rejects_overlapping_partial_scratch(self):
        gradient = torch.randn(2053, device=self.device, dtype=torch.float32)
        required = grad_norm_num_partials((gradient,))
        with self.assertRaisesRegex(ValueError, "must not overlap any gradient"):
            online_mtp_scaled_grad_norm_sq(
                (gradient,),
                denominator=1.0,
                partials_out=gradient[:required],
            )

    def test_adamw_rejects_shifted_gradient_prepared_overlap(self):
        storage = torch.randn(18, device=self.device, dtype=torch.float32)
        gradient = storage[:-1]
        prepared = storage[1:]
        state = [torch.zeros_like(gradient) for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "exactly aliased or disjoint"):
            online_mtp_adamw_prepare_(
                gradient,
                state[0],
                state[1],
                state[2],
                prepared,
                denominator=1.0,
                clip_scale=1.0,
                betas=(0.9, 0.999),
                corrections=(0.1, 0.001),
                epsilon=1e-8,
                learning_rate=1e-4,
                weight_decay=0.0,
            )

    def test_adamw_rejects_overlapping_state_tensors(self):
        gradient = torch.randn(17, device=self.device, dtype=torch.float32)
        shared_state = torch.zeros_like(gradient)
        with self.assertRaisesRegex(ValueError, "second_moment must not overlap"):
            online_mtp_adamw_prepare_(
                gradient,
                shared_state,
                shared_state,
                torch.zeros_like(gradient),
                torch.empty_like(gradient),
                denominator=1.0,
                clip_scale=1.0,
                betas=(0.9, 0.999),
                corrections=(0.1, 0.001),
                epsilon=1e-8,
                learning_rate=1e-4,
                weight_decay=0.0,
            )

    def test_default_optimizer_keeps_distinct_prepared_storage(self):
        accumulator = self._accumulator(fused=False)
        accumulator.materialize_optimizer_state()
        self.assertFalse(accumulator.reuse_gradient_for_prepared)
        for name in accumulator.parameters:
            self.assertNotEqual(
                accumulator.prepared_parameters[name].data_ptr(),
                accumulator.gradients[name].data_ptr(),
            )


if __name__ == "__main__":
    unittest.main()
