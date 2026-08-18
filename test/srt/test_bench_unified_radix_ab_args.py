"""CPU smoke tests for the bench_unified_radix_ab arg layer (parsing, external-server
single-config assertion, dense default config) -- no server; loaded by file path.
Manual-only: not registered in run_suite.py."""

import importlib.util
import os
import sys
import unittest
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Make sgl_jax importable when the package is not installed (CPU dev box).
_PKG_DIR = os.path.join(_REPO_ROOT, "python")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

_BENCH_PATH = os.path.join(_REPO_ROOT, "benchmark", "hicache", "bench_unified_radix_ab.py")


def _load_bench_module():
    spec = importlib.util.spec_from_file_location("bench_unified_radix_ab", _BENCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(argv):
    bench = _load_bench_module()
    with mock.patch("sys.argv", ["bench_unified_radix_ab.py", *argv]):
        return bench, bench.parse_args()


class TestBenchUnifiedRadixABArgs(unittest.TestCase):
    def test_config_args_dense_byte_identical(self):
        """Dense CONFIG_ARGS entries are unchanged; recurrent is additive."""
        bench = _load_bench_module()
        self.assertEqual(bench.CONFIG_ARGS["no-cache"], ["--disable-radix-cache"])
        self.assertEqual(bench.CONFIG_ARGS["radix"], [])
        self.assertEqual(bench.CONFIG_ARGS["unified"], ["--enable-unified-radix-tree"])
        self.assertEqual(
            bench.CONFIG_ARGS["unified-recurrent"],
            ["--enable-unified-radix-tree", "--enable-recurrent-extra-buffer"],
        )

    def test_default_args_dense_unchanged(self):
        """No new flags -> dense defaults are exactly the original values."""
        _, args = _parse([])
        self.assertEqual(args.configs, ["no-cache", "radix", "unified"])
        self.assertIsNone(args.server_url)
        self.assertFalse(args.disable_overlap_schedule)
        # chunked_prefill_size has a parser default but is NOT emitted for dense
        # configs (see run_config), so the dense server command stays identical.
        self.assertEqual(args.chunked_prefill_size, 512)

    def test_new_flags_parse(self):
        bench, args = _parse(
            [
                "--configs",
                "unified-recurrent",
                "--disable-overlap-schedule",
                "--chunked-prefill-size",
                "256",
                "--server-url",
                "http://10.0.0.1:30000",
            ]
        )
        self.assertEqual(args.configs, ["unified-recurrent"])
        self.assertTrue(args.disable_overlap_schedule)
        self.assertEqual(args.chunked_prefill_size, 256)
        self.assertEqual(args.server_url, "http://10.0.0.1:30000")

    def test_server_url_requires_single_config(self):
        with self.assertRaises(AssertionError) as ctx:
            _parse(["--server-url", "http://10.0.0.1:30000", "--configs", "no-cache", "unified"])
        self.assertIn("exactly one --configs entry", str(ctx.exception))

    def test_server_url_single_config_ok(self):
        _, args = _parse(
            ["--server-url", "http://10.0.0.1:30000", "--configs", "unified-recurrent"]
        )
        self.assertEqual(args.configs, ["unified-recurrent"])

    def test_recurrent_chunked_prefill_emitted_only_for_recurrent(self):
        """run_config emits --chunked-prefill-size only for unified-recurrent."""
        bench = _load_bench_module()

        captured = {}

        def fake_popen(model, **kwargs):
            captured["other_args"] = kwargs["other_args"]
            raise RuntimeError("stop after building the server command")

        for config, expect_cps in (("radix", False), ("unified-recurrent", True)):
            _, args = _parse(["--configs", config, "--tp-size", "1", "--workloads", "random"])
            # Patch the lazily-imported launcher + run_benchmark inside run_config.
            with (
                mock.patch("sgl_jax.test.test_utils.popen_launch_server", side_effect=fake_popen),
                mock.patch("sgl_jax.bench_serving.run_benchmark"),
                mock.patch("sgl_jax.srt.utils.kill_process_tree"),
            ):
                with self.assertRaises(RuntimeError):
                    bench.run_config(args, config)
            has_cps = "--chunked-prefill-size" in captured["other_args"]
            self.assertEqual(has_cps, expect_cps, f"config={config}")


if __name__ == "__main__":
    unittest.main()
