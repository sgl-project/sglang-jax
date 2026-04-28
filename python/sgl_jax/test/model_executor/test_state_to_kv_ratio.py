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


class TestSplitStateKvBudget(unittest.TestCase):
    """state_to_kv_ratio splits available HBM into state count + KV bytes.

    Mirrors sglang `handle_max_mamba_cache`: state count is floored by
    per_req first, then KV reclaims the leftover (state floor remainder
    goes back to KV instead of being wasted).
    """

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_state_floor_kv_reclaims_leftover(self):
        """Core contract: state count is floored, KV gets actual leftover."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        state_max_reqs, kv_budget = _split_state_kv_budget(
            available_bytes=10_000, ratio=1.0, per_req_state_bytes=300
        )
        self.assertEqual(state_max_reqs, 16)  # floor(5000 / 300)
        self.assertEqual(kv_budget, 10_000 - 16 * 300)  # 5200, NOT 5000

    def test_default_ratio_zero_point_nine(self):
        """r=0.9 -> state_budget_raw ~ 47.4% of available; KV reclaims floor remainder."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        available = 100 * (1024**3)  # 100 GiB
        per_req = 10 * (1024**2)  # 10 MiB
        state_max_reqs, kv_budget = _split_state_kv_budget(
            available, ratio=0.9, per_req_state_bytes=per_req
        )
        # state_budget_raw = available * 0.9/1.9 ~ 47.368 GiB
        # state_max_reqs ~ 47.368 GiB / 10 MiB ~ 4850
        # kv_budget = available - state_max_reqs * per_req
        self.assertEqual(kv_budget, available - state_max_reqs * per_req)
        self.assertGreater(state_max_reqs, 0)

    def test_ratio_zero_means_no_state(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        state_max_reqs, kv_budget = _split_state_kv_budget(
            available_bytes=10_000, ratio=0.0, per_req_state_bytes=300
        )
        self.assertEqual(state_max_reqs, 0)
        self.assertEqual(kv_budget, 10_000)

    def test_ratio_extreme_high_starves_kv(self):
        """r=1000 -> state_budget_raw ~ 99.9% of available."""
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        state_max_reqs, kv_budget = _split_state_kv_budget(
            available_bytes=1_000_000, ratio=1000.0, per_req_state_bytes=100
        )
        # state_budget_raw = 1_000_000 * 1000/1001 ~ 999_000
        # state_max_reqs = 999_000 // 100 = 9990
        # kv_budget = 1_000_000 - 9990 * 100 = 1000
        self.assertEqual(state_max_reqs, 9990)
        self.assertEqual(kv_budget, 1_000)

    def test_negative_ratio_raises(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        with self.assertRaises(AssertionError):
            _split_state_kv_budget(1000, ratio=-0.1, per_req_state_bytes=100)

    def test_per_req_state_bytes_zero_raises(self):
        from sgl_jax.srt.model_executor.hybrid_recurrent_utils import (
            _split_state_kv_budget,
        )

        with self.assertRaises(AssertionError):
            _split_state_kv_budget(1000, ratio=1.0, per_req_state_bytes=0)


if __name__ == "__main__":
    unittest.main()
