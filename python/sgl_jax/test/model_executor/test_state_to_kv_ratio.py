"""Memory budget split between recurrent state and KV cache for hybrid models."""

import unittest

import jax


class TestRecurrentPerReqBytes(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_kimi_linear_tp1_per_req_bytes(self):
        """Kimi-Linear-48B TP=1, default dtypes (f32 recurrent + bf16 conv).

        per_req_recurrent = L * (H/tp) * D * D * 4
                          = 20 * 32 * 128 * 128 * 4 = 41,943,040 (~40 MB)
        per_req_conv      = L * (K-1) * (proj_size/tp) * 2
                          = 20 * 3 * (12288/1) * 2 = 1,474,560 (~1.4 MB)
        """
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _compute_recurrent_per_req_bytes,
        )

        per_req = _compute_recurrent_per_req_bytes(
            num_layers=20,
            num_heads=32,
            head_dim=128,
            conv_kernel_size=4,
            tp_size=1,
            temporal_dtype_bytes=4,
            conv_dtype_bytes=2,
        )
        self.assertEqual(per_req, 20 * 32 * 128 * 128 * 4 + 20 * 3 * 12288 * 2)

    def test_kimi_linear_tp4_per_req_bytes(self):
        """TP=4: H/tp=8, proj_size/tp=3072. RFC line 419 cites ~100 MB recurrent / ~360 KB conv."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _compute_recurrent_per_req_bytes,
        )

        per_req = _compute_recurrent_per_req_bytes(
            num_layers=20,
            num_heads=32,
            head_dim=128,
            conv_kernel_size=4,
            tp_size=4,
            temporal_dtype_bytes=4,
            conv_dtype_bytes=2,
        )
        # per_req_recurrent = 20*8*128*128*4 = 10,485,760 (~10MB per device)
        # per_req_conv      = 20*3*(12288/4)*2 = 368,640 (~360KB)
        expected = 20 * 8 * 128 * 128 * 4 + 20 * 3 * (12288 // 4) * 2
        self.assertEqual(per_req, expected)


class TestStateToKvRatioSplit(unittest.TestCase):
    """state_to_kv_ratio splits available HBM into state + KV budget."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_default_ratio_zero_point_nine(self):
        """r=0.9 -> state_budget = available * 0.9/(1+0.9) ~ 47.4%."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        available = 100 * (1024**3)  # 100 GB
        state_budget, kv_budget = _split_state_kv_budget(available, ratio=0.9)
        # state = 100 * 0.9/1.9 = 47.368...
        self.assertAlmostEqual(state_budget, available * 0.9 / 1.9, delta=1)
        self.assertAlmostEqual(kv_budget, available - state_budget, delta=1)

    def test_ratio_zero_means_no_state_budget(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        state_budget, kv_budget = _split_state_kv_budget(1000, ratio=0.0)
        self.assertEqual(state_budget, 0)
        self.assertEqual(kv_budget, 1000)

    def test_ratio_extreme_high_starves_kv(self):
        """r=1000 -> state_budget approaches 100% of available."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        state_budget, kv_budget = _split_state_kv_budget(1000, ratio=1000.0)
        # state = 1000 * 1000/1001 ~ 999
        self.assertAlmostEqual(state_budget, 1000 * 1000 / 1001, delta=1)
        self.assertLess(kv_budget, 5)  # nearly nothing left

    def test_negative_ratio_raises(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        with self.assertRaises(AssertionError):
            _split_state_kv_budget(1000, ratio=-0.1)


class TestComputeMaxNumReqsFromStateBudget(unittest.TestCase):
    """Given state_budget + per_req bytes, derive max_num_reqs."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_basic_division(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _compute_max_num_reqs_from_state_budget,
        )

        # state_budget = 1000, per_req = 100 -> max_num_reqs = 10
        n = _compute_max_num_reqs_from_state_budget(state_budget=1000, per_req_bytes=100)
        self.assertEqual(n, 10)

    def test_floor_division_truncates(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _compute_max_num_reqs_from_state_budget,
        )

        # 999 / 100 = 9 (floor)
        n = _compute_max_num_reqs_from_state_budget(state_budget=999, per_req_bytes=100)
        self.assertEqual(n, 9)

    def test_zero_state_budget_returns_zero(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _compute_max_num_reqs_from_state_budget,
        )

        n = _compute_max_num_reqs_from_state_budget(state_budget=0, per_req_bytes=100)
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
