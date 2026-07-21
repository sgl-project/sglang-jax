"""CPU tests for the bench_recurrent_reuse_sweep knee math (predict_knee /
derive_actual_C_rank / detect_knee); loaded by file path (benchmark/ is off the
package path)."""

import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

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
        """Doc example: size=192, dp=4, C_rank=4, owned=3 -> S_rank=48, 4*(48-12)=144."""
        self.assertEqual(self.bench.predict_knee(192, dp_size=4, C_rank=4), 144.0)

    def test_owned_slots_scale_reservation(self):
        """request_owned_slots multiplies the per-rank consumption: overlap-off
        (owned=2) raises the knee: 4*(48 - 2*4) = 160."""
        self.assertEqual(self.bench.predict_knee(192, 4, C_rank=4, request_owned_slots=2), 160.0)

    def test_snapshots_per_prefix_divides_knee(self):
        """More snapshots per prefix -> fewer distinct prefixes fit -> lower K*."""
        base = self.bench.predict_knee(192, 4, C_rank=4, snapshots_per_prefix=1)
        self.assertEqual(
            self.bench.predict_knee(192, 4, C_rank=4, snapshots_per_prefix=2), base / 2
        )


class TestDeriveActualCRank(unittest.TestCase):
    def setUp(self):
        self.bench = _load_bench_module()

    def test_spread_over_dp(self):
        """parallel spread over DP ranks: 16 / 4 = 4 concurrent per rank."""
        self.assertEqual(self.bench.derive_actual_C_rank(parallel=16, dp_size=4), 4.0)

    def test_floor_of_one(self):
        """At least one in-flight request per rank is reserved (max(1, ...))."""
        self.assertEqual(self.bench.derive_actual_C_rank(parallel=2, dp_size=8), 1.0)


class TestDetectKnee(unittest.TestCase):
    def setUp(self):
        self.bench = _load_bench_module()

    @staticmethod
    def _curve(pairs):
        return [{"K": k, "reuse_frac": f} for k, f in pairs]

    def test_knee_at_plateau_edge(self):
        """Largest K still within plateau_frac * peak is the knee."""
        curve = self._curve([(8, 0.5), (16, 0.5), (32, 0.2), (64, 0.05)])
        self.assertEqual(self.bench.detect_knee(curve, plateau_frac=0.9), 16)

    def test_no_reuse_returns_none(self):
        """Peak reuse ~0 (no-cache / broken / all-miss) has no knee, not max K."""
        curve = self._curve([(8, 0.0), (16, 0.0), (32, 0.0)])
        self.assertIsNone(self.bench.detect_knee(curve, plateau_frac=0.9))


class TestRunKPoint(unittest.TestCase):
    def setUp(self):
        self.bench = _load_bench_module()
        self.args = SimpleNamespace(server_url="http://server", suffix_tokens=8, parallel=2)

    @staticmethod
    def _result(success=True, error=""):
        return SimpleNamespace(
            success=success,
            error=error,
            cached_tokens=64,
            prompt_len=96,
            ttft=0.1,
        )

    def _run(self, rounds):
        with (
            mock.patch.object(self.bench, "flush_cache"),
            mock.patch.object(self.bench, "make_prefix_ids", return_value=[1] * 80),
            mock.patch.object(self.bench, "make_suffix_ids", return_value=[2] * 8),
            mock.patch.object(self.bench, "_send_round", mock.AsyncMock(side_effect=rounds)),
        ):
            return self.bench.run_k_point(
                self.args,
                tokenizer=object(),
                generate_url="http://server/generate",
                K=2,
                interval=64,
            )

    def test_failed_warm_request_aborts_measurement(self):
        warm = [self._result(), self._result(False, "warm failed")]
        with self.assertRaisesRegex(RuntimeError, "warm requests failed"):
            self._run([warm])

    def test_failed_probe_request_aborts_measurement(self):
        warm = [self._result(), self._result()]
        probe = [self._result(), self._result(False, "probe failed")]
        with self.assertRaisesRegex(RuntimeError, "probe requests failed"):
            self._run([warm, probe])

    def test_complete_rounds_produce_reuse_metrics(self):
        results = [self._result(), self._result()]
        point = self._run([results, results])
        self.assertEqual(point["cached_tokens"], 128)
        self.assertEqual(point["prompt_tokens"], 192)


if __name__ == "__main__":
    unittest.main()
