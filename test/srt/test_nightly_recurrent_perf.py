import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NIGHTLY_DIR = _REPO_ROOT / "test" / "srt" / "nightly"
for _path in (_REPO_ROOT / "python", _NIGHTLY_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cases import AbsolutePerfBaseline, PerfCase  # noqa: E402
from drivers import run_benchmark_for_case  # noqa: E402
from results import build_perf_result, gate_perf_result, write_perf_json  # noqa: E402


def _case(**overrides):
    kwargs = {
        "name": "recurrent-perf",
        "input_len": 1024,
        "output_len": 128,
        "num_prompts": 8,
        "max_concurrency": 4,
        "use_trailing_baseline": False,
        "absolute_baselines": {
            "ttft_ms": AbsolutePerfBaseline(value=100.0, tolerance=0.1),
            "total_tps": AbsolutePerfBaseline(value=1000.0, tolerance=0.1),
        },
    }
    kwargs.update(overrides)
    return PerfCase(**kwargs)


def _metrics(**overrides):
    metrics = {
        "completed": 8,
        "median_ttft_ms": 110.0,
        "median_itl_ms": 5.0,
        "input_throughput": 800.0,
        "output_throughput": 100.0,
        "total_throughput": 900.0,
    }
    metrics.update(overrides)
    return metrics


class TestPerfBaseline(unittest.TestCase):
    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            AbsolutePerfBaseline(value=0, tolerance=0.1)
        with self.assertRaises(ValueError):
            AbsolutePerfBaseline(value=1, tolerance=1)

    def test_absolute_matrix_passes_at_bounds(self):
        case = _case()
        result = build_perf_result(case, "profile", "v6e-4", _metrics())

        self.assertIsNone(gate_perf_result(case, result))
        self.assertTrue(result["passed"])
        self.assertEqual(result["baseline_source"], "disabled")
        self.assertEqual(set(result["absolute_baselines"]), {"ttft_ms", "total_tps"})

    def test_absolute_matrix_reports_both_regressions(self):
        case = _case()
        result = build_perf_result(
            case,
            "profile",
            "v6e-4",
            _metrics(median_ttft_ms=111.0, total_throughput=899.0),
        )

        failure = gate_perf_result(case, result)

        self.assertEqual(failure[0], "threshold")
        self.assertIn("ttft_ms=111.0", failure[1])
        self.assertIn("total_tps=899.0", failure[1])
        self.assertFalse(result["passed"])

    def test_existing_floor_gate_still_works(self):
        case = _case(
            absolute_baselines=None,
            floors={"out_tps": 101.0},
            use_trailing_baseline=True,
        )
        result = build_perf_result(case, "profile", "v6e-4", _metrics())

        with patch("perf_baseline.fetch_trailing_baseline", return_value=None):
            failure = gate_perf_result(case, result)

        self.assertEqual(failure[0], "threshold")
        self.assertIn("out_tps=100.0 < floor 101.0", failure[1])

    def test_json_records_workload_and_gate(self):
        case = _case()
        metrics = _metrics()
        result = build_perf_result(case, "profile", "v6e-4", metrics)
        gate_perf_result(case, result)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"RESULTS_DIR": tmpdir}):
                path = write_perf_json(case, "profile", "v6e-4", metrics, result)
            payload = json.loads(path.read_text())

        self.assertEqual(payload["workload"], "random")
        self.assertTrue(payload["gate"]["passed"])
        self.assertEqual(payload["gate"]["absolute_baselines"]["ttft_ms"]["value"], 100.0)


class TestGeneratedSharedPrefixCase(unittest.TestCase):
    def test_default_warmup_preserves_existing_perf_cases(self):
        self.assertEqual(_case().warmup_requests, 0)

    def test_prompt_count_matches_group_shape(self):
        with self.assertRaises(ValueError):
            _case(
                workload="generated-shared-prefix",
                gsp_num_groups=3,
                gsp_prompts_per_group=2,
            )

    def test_driver_forwards_workload_shape(self):
        case = _case(
            workload="generated-shared-prefix",
            num_prompts=8,
            gsp_num_groups=4,
            gsp_prompts_per_group=2,
            gsp_system_prompt_len=2048,
            gsp_question_len=128,
            warmup_requests=1,
        )
        with (
            patch("sgl_jax.test.test_utils.get_benchmark_args") as get_args,
            patch("sgl_jax.bench_serving.run_benchmark", return_value={"completed": 8}),
        ):
            get_args.return_value = type("Args", (), {})()
            run_benchmark_for_case(case, "http://127.0.0.1:30063", "model")

        kwargs = get_args.call_args.kwargs
        self.assertEqual(kwargs["dataset_name"], "generated-shared-prefix")
        self.assertEqual(kwargs["gsp_num_groups"], 4)
        self.assertEqual(kwargs["gsp_prompts_per_group"], 2)
        self.assertEqual(kwargs["warmup_requests"], 1)

    def test_cases_are_exposed_without_xprof(self):
        runner = _NIGHTLY_DIR / "single_host" / "suite_runner.py"
        proc = subprocess.run(
            [sys.executable, str(runner), "--caselist"],
            check=True,
            capture_output=True,
            text=True,
        )
        entries = json.loads(proc.stdout)
        recurrent = {
            entry["case"]
            for entry in entries
            if entry["suite"] == "perf-text-models-v6e-4"
            and entry["case"].startswith("recurrent-qwen35-")
        }
        self.assertEqual(recurrent, {"recurrent-qwen35-gsp", "recurrent-qwen35-random"})


if __name__ == "__main__":
    unittest.main()
