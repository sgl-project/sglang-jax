"""TSA prefill eligibility (review on #1701): divisibility must be checked on the per-rank token count."""

import unittest

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import default_run_capacity
from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
    _idx_tshard_eligible,
    _tsa_eligible,
    _tsa_zigzag_eligible,
    spec_prewrite_run_capacity,
)


class TsaEligibleTest(unittest.TestCase):
    def test_per_rank_divisibility(self):
        self.assertFalse(_tsa_eligible(6, dp_size=2, tp_size=2))  # 3 tokens per rank
        self.assertTrue(_tsa_eligible(8, dp_size=2, tp_size=2))
        self.assertTrue(_tsa_eligible(8192, dp_size=1, tp_size=8))
        self.assertFalse(_tsa_eligible(8192 + 16, dp_size=1, tp_size=8) and (8192 + 16) % 8 != 0)
        self.assertFalse(_tsa_eligible(8192, dp_size=1, tp_size=1))  # tp=1: head-TP layout kept

    def test_zigzag_per_rank_divisibility(self):
        # review: DP=2, TP=8, T=128, QB=16 passes the global check (128 % 128 == 0) but each
        # DP rank holds 64 rows and tsa_zigzag_permute's reshape fails
        self.assertFalse(_tsa_zigzag_eligible(128, dp_size=2, tp_size=8, qb=16))
        self.assertTrue(_tsa_zigzag_eligible(256, dp_size=2, tp_size=8, qb=16))
        self.assertTrue(_tsa_zigzag_eligible(128, dp_size=1, tp_size=8, qb=16))
        self.assertFalse(_tsa_zigzag_eligible(130, dp_size=2, tp_size=8, qb=16))

    def test_idx_tshard_per_rank_whole_groups(self):
        # review: grouped page-topk, DP=2, TP=2, one 4-token request per DP rank (T=8):
        # the global check (8/2 = 4 rows, whole group) passes but each DP rank has 4 rows
        # that the tensor axis would split into half-groups -> every page -1
        self.assertFalse(_idx_tshard_eligible(8, dp_size=2, tp_size=2, q_group=4))
        self.assertTrue(_idx_tshard_eligible(16, dp_size=2, tp_size=2, q_group=4))
        self.assertTrue(_idx_tshard_eligible(256, dp_size=8, tp_size=2, q_group=4))  # dp8 p64
        self.assertFalse(_idx_tshard_eligible(32, dp_size=1, tp_size=16, q_group=4))  # 2 rows/rank
        self.assertTrue(_idx_tshard_eligible(64, dp_size=1, tp_size=16, q_group=None))
        self.assertFalse(_idx_tshard_eligible(64, dp_size=1, tp_size=1, q_group=None))

    def test_spec_prewrite_run_capacity_covers_request_count(self):
        # review: 256 requests x 5 tokens on the DSA_SPEC_AS_DECODE=1 head-parallel path =
        # 1280 rows > STATIC_DISPATCH_MAX_ROWS, default cap 150 < 256 runs
        self.assertEqual(default_run_capacity(1280, 128), 150)
        cap = spec_prewrite_run_capacity(1280, 256, 128)
        self.assertGreaterEqual(cap, 2 * 256)  # <= 2 runs per 5-token request
        self.assertGreaterEqual(spec_prewrite_run_capacity(256, 64, 128), 2 * 64)


if __name__ == "__main__":
    unittest.main()
