import importlib.util
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

_TEST_SRT = os.path.dirname(os.path.abspath(__file__))
_NIGHTLY = os.path.join(_TEST_SRT, "nightly")
_SINGLE_HOST_RUNNER = os.path.join(_NIGHTLY, "single_host", "suite_runner.py")
for path in (_TEST_SRT, _NIGHTLY, os.path.dirname(_SINGLE_HOST_RUNNER)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cases import BenchCase, SuiteError  # noqa: E402
from drivers import run_bench_for_case  # noqa: E402


def _load_single_host_runner():
    spec = importlib.util.spec_from_file_location("nightly_single_host_runner", _SINGLE_HOST_RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestBenchDriver(unittest.TestCase):
    def _run_with_returncode(self, returncode):
        case = BenchCase(name="bench", script="fake.py", server="none", timeout=17)
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch(
                "drivers.subprocess.run", return_value=SimpleNamespace(returncode=returncode)
            ) as run,
            mock.patch("results.write_result"),
        ):
            result, fail = run_bench_for_case(case)
        self.assertEqual(run.call_args.kwargs["timeout"], 17)
        return result, fail

    def test_tagged_exit_codes(self):
        for returncode, expected in ((0, None), (10, "infra"), (20, "threshold"), (30, "case")):
            with self.subTest(returncode=returncode):
                result, fail = self._run_with_returncode(returncode)
                self.assertEqual(None if fail is None else fail[0], expected)
                self.assertEqual(result["passed"], returncode == 0)

    def test_unknown_exit_is_case_crash(self):
        _result, fail = self._run_with_returncode(1)
        self.assertEqual(fail[0], "case")

    def test_timeout_is_retryable_infra(self):
        case = BenchCase(name="bench", script="fake.py", server="none", timeout=7)
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch(
                "drivers.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["python", "fake.py"], timeout=7),
            ),
            mock.patch("results.write_result"),
        ):
            result, fail = run_bench_for_case(case)
        self.assertTrue(result["timed_out"])
        self.assertEqual(fail[0], "infra")


class TestBenchFailureAggregation(unittest.TestCase):
    def test_infra_failure_remains_infra(self):
        runner = _load_single_host_runner()
        case = BenchCase(name="bench", script="fake.py")
        with mock.patch("drivers.run_bench_for_case", return_value=({}, ("infra", "timeout"))):
            with self.assertRaises(SuiteError) as ctx:
                runner._eval_cases([case], spec=None, profile=None)
        self.assertEqual(ctx.exception.kind, "infra")


class TestBenchCaseSelection(unittest.TestCase):
    def setUp(self):
        self.runner = _load_single_host_runner()

    def test_recurrent_ab_key_selects_both_measurements_and_compare(self):
        suite = self.runner._select_cases(
            self.runner.SUITES["recurrent-ab-perf-v6e-4"], "recurrent-ab"
        )
        selected = [case for run in suite.runs for case in run.cases]
        self.assertEqual(len(selected), 3)

    def test_caselist_deduplicates_group_and_hides_manual_ablation(self):
        entries = self.runner._caselist()
        recurrent_ab = [entry for entry in entries if entry["case"] == "recurrent-ab"]
        self.assertEqual(len(recurrent_ab), 1)
        self.assertFalse(any(entry["suite"] == "recurrent-ablation-v6e-4" for entry in entries))


if __name__ == "__main__":
    unittest.main()
