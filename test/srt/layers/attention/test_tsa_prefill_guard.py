"""TSA prefill eligibility (review on #1701): divisibility must be checked on the per-rank token count."""

import unittest

from sgl_jax.srt.layers.attention.dsa_sparse_backend import _tsa_eligible


class TsaEligibleTest(unittest.TestCase):
    def test_per_rank_divisibility(self):
        self.assertFalse(_tsa_eligible(6, dp_size=2, tp_size=2))  # 3 tokens per rank
        self.assertTrue(_tsa_eligible(8, dp_size=2, tp_size=2))
        self.assertTrue(_tsa_eligible(8192, dp_size=1, tp_size=8))
        self.assertFalse(_tsa_eligible(8192 + 16, dp_size=1, tp_size=8) and (8192 + 16) % 8 != 0)
        self.assertFalse(_tsa_eligible(8192, dp_size=1, tp_size=1))  # tp=1: head-TP layout kept


if __name__ == "__main__":
    unittest.main()
