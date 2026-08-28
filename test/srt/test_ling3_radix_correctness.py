"""CPU tests for the Ling-3 serving-level radix correctness workload."""

import importlib.util
import os
import sys
import unittest
from unittest import mock

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_PKG_DIR = os.path.join(_REPO_ROOT, "python")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_BENCH_PATH = os.path.join(
    _REPO_ROOT, "benchmark", "hicache", "bench_ling3_radix_correctness.py"
)


def _load_bench_module():
    name = "bench_ling3_radix_correctness"
    spec = importlib.util.spec_from_file_location(name, _BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestLing3RadixWorkload(unittest.TestCase):
    def setUp(self):
        self.bench = _load_bench_module()

    def test_workload_has_diverse_depths_and_sibling_prefixes(self):
        anchors, probes = self.bench.build_workload(
            families=8,
            branches=8,
            page_size=256,
            track_interval=512,
            seed=7,
        )

        self.assertEqual(len(anchors), 8)
        self.assertEqual(len(probes), 64)
        self.assertEqual(
            {anchor.shared_tokens for anchor in anchors},
            {768, 1280, 1792, 2304},
        )
        # Unrelated families start differently, while all branches in a family
        # share the exact anchor prefix and diverge immediately afterwards.
        self.assertEqual(len({anchor.input_ids[0] for anchor in anchors}), 8)
        for anchor in anchors:
            siblings = [probe for probe in probes if probe.family == anchor.family]
            roots = {
                tuple(probe.input_ids[: probe.shared_tokens]) for probe in siblings
            }
            suffixes = {
                tuple(probe.input_ids[probe.shared_tokens :]) for probe in siblings
            }
            self.assertEqual(len(roots), 1)
            self.assertEqual(len(suffixes), 8)
            self.assertEqual(
                next(iter(roots)),
                tuple(anchor.input_ids[: anchor.shared_tokens]),
            )

    def test_server_contract_requires_overlap_and_recurrent_radix(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "disable_radix_cache": False,
            "enable_unified_radix_tree": True,
            "enable_recurrent_extra_buffer": True,
            "disable_overlap_schedule": False,
            "dp_size": 8,
            "page_size": 256,
            "recurrent_track_interval": 512,
        }
        with mock.patch.object(self.bench.requests, "get", return_value=response):
            self.assertEqual(
                self.bench._server_contract("http://server", 8), (256, 512)
            )

        response.json.return_value["disable_overlap_schedule"] = True
        with (
            mock.patch.object(self.bench.requests, "get", return_value=response),
            self.assertRaisesRegex(AssertionError, "overlap scheduling enabled"),
        ):
            self.bench._server_contract("http://server", 8)


if __name__ == "__main__":
    unittest.main()
