"""CPU unit tests for the bench_recurrent_reuse_sweep predict_knee math.

Covers only the pure analytic-knee formula (and the per-rank C_rank derivation),
no server. The HTTP K-sweep path is exercised by the controller against a live
recurrent server. This is a MANUAL benchmark helper, deliberately NOT registered
in run_suite.py. The benchmark lives under benchmark/, off the package path, so
it is loaded by file path.
"""

import importlib.util
import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Make sgl_jax importable when the package is not installed (CPU dev box).
_PKG_DIR = os.path.join(_REPO_ROOT, "python")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_BENCH_PATH = os.path.join(_REPO_ROOT, "benchmark", "hicache", "bench_recurrent_reuse_sweep.py")


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_recurrent_reuse_sweep", _BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPredictKnee(unittest.TestCase):
    def setUp(self):
        self.bench = _load_bench_module()

    def test_spec_example(self):
        """Doc example: size=192, dp=4, C_rank=4 -> S_rank=48, K* = 4*(48-12) = 144."""
        Kstar = self.bench.predict_knee(192, dp_size=4, C_rank=4)
        self.assertEqual(Kstar, 144.0)

    def test_dp1_no_reservation(self):
        """dp=1, C_rank=0 -> the whole pool is cacheable: K* = size."""
        self.assertEqual(self.bench.predict_knee(128, dp_size=1, C_rank=0), 128.0)

    def test_factor_scales_reservation(self):
        """factor multiplies the per-rank reservation (1 running + 2 ping-pong)."""
        # size=192, dp=4, C_rank=4, factor=2 -> 4*(48 - 2*4) = 160.
        self.assertEqual(self.bench.predict_knee(192, 4, C_rank=4, factor=2), 160.0)

    def test_snapshots_per_prefix_moves_knee_left(self):
        """More snapshots per prefix -> fewer distinct prefixes fit -> lower K*."""
        base = self.bench.predict_knee(192, 4, C_rank=4, snapshots_per_prefix=1)
        two = self.bench.predict_knee(192, 4, C_rank=4, snapshots_per_prefix=2)
        self.assertEqual(two, base / 2)

    def test_higher_concurrency_lowers_knee(self):
        """Larger C_rank eats more budget, lowering the knee."""
        low = self.bench.predict_knee(256, 4, C_rank=2)
        high = self.bench.predict_knee(256, 4, C_rank=8)
        self.assertLess(high, low)


class TestDeriveActualCRank(unittest.TestCase):
    def setUp(self):
        self.bench = _load_bench_module()

    def test_spread_over_dp(self):
        """parallel spread over DP ranks: 16 / 4 = 4 concurrent per rank."""
        self.assertEqual(self.bench.derive_actual_C_rank(parallel=16, dp_size=4), 4.0)

    def test_floor_of_one(self):
        """At least one in-flight request per rank is reserved."""
        self.assertEqual(self.bench.derive_actual_C_rank(parallel=2, dp_size=8), 1.0)

    def test_derived_not_constant(self):
        """C_rank tracks the actual --parallel, not a fixed constant."""
        self.assertNotEqual(
            self.bench.derive_actual_C_rank(8, 4),
            self.bench.derive_actual_C_rank(32, 4),
        )


if __name__ == "__main__":
    unittest.main()
