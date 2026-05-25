"""Unit tests for scripts/ci/coordinator_decide.py and scripts/ci/finish_check.py."""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, "scripts/ci")
from coordinator_decide import detect_draft, detect_labels, summarize, write_outputs
from finish_check import check_jobs


class TestDetectDraft(unittest.TestCase):
    def test_pull_request_not_draft(self):
        self.assertEqual(detect_draft("pull_request", "false"), "false")

    def test_pull_request_draft(self):
        self.assertEqual(detect_draft("pull_request", "true"), "true")

    def test_pull_request_empty_draft(self):
        self.assertEqual(detect_draft("pull_request", ""), "false")

    def test_push_event(self):
        self.assertEqual(detect_draft("push", ""), "false")

    def test_push_event_ignores_draft_value(self):
        self.assertEqual(detect_draft("push", "true"), "false")


class TestDetectLabels(unittest.TestCase):
    def test_push_event_ignores_labels(self):
        flags = detect_labels("push", '["test:full"]')
        self.assertTrue(all(v is False for v in flags.values()))

    def test_no_labels(self):
        flags = detect_labels("pull_request", "[]")
        self.assertTrue(all(v is False for v in flags.values()))

    def test_full_label(self):
        flags = detect_labels("pull_request", '["test:full"]')
        self.assertTrue(flags["run_full"])
        self.assertFalse(flags["requires_4tpu"])

    def test_multi_chip_label(self):
        flags = detect_labels("pull_request", '["test:multi-chip"]')
        self.assertTrue(flags["requires_4tpu"])
        self.assertFalse(flags["run_full"])

    def test_perf_label(self):
        flags = detect_labels("pull_request", '["test:perf"]')
        self.assertTrue(flags["run_perf"])

    def test_perf_trace_label(self):
        flags = detect_labels("pull_request", '["test:perf-trace"]')
        self.assertTrue(flags["run_perf_trace"])

    def test_accuracy_extra_label(self):
        flags = detect_labels("pull_request", '["test:accuracy-extra"]')
        self.assertTrue(flags["run_accuracy_extra"])

    def test_multiple_labels(self):
        flags = detect_labels("pull_request", '["test:full", "test:perf"]')
        self.assertTrue(flags["run_full"])
        self.assertTrue(flags["run_perf"])

    def test_null_json(self):
        flags = detect_labels("pull_request", "null")
        self.assertTrue(all(v is False for v in flags.values()))

    def test_empty_string(self):
        flags = detect_labels("pull_request", "")
        self.assertTrue(all(v is False for v in flags.values()))

    def test_invalid_json(self):
        flags = detect_labels("pull_request", "{not valid json")
        self.assertTrue(all(v is False for v in flags.values()))

    def test_unrecognized_label_ignored(self):
        flags = detect_labels("pull_request", '["some-other-label"]')
        self.assertTrue(all(v is False for v in flags.values()))

    def test_dict_json_treated_as_no_labels(self):
        flags = detect_labels("pull_request", '{"key": "val"}')
        self.assertTrue(all(v is False for v in flags.values()))


class TestSummarize(unittest.TestCase):
    def test_main_package_only(self):
        result = summarize("true", "false")
        self.assertTrue(result["run_main_test"])
        self.assertFalse(result["run_pallas_bench"])

    def test_pallas_only(self):
        result = summarize("false", "true")
        self.assertFalse(result["run_main_test"])
        self.assertTrue(result["run_pallas_bench"])

    def test_both(self):
        result = summarize("true", "true")
        self.assertTrue(result["run_main_test"])
        self.assertTrue(result["run_pallas_bench"])

    def test_neither(self):
        result = summarize("false", "false")
        self.assertFalse(result["run_main_test"])
        self.assertFalse(result["run_pallas_bench"])


def _make_env(**overrides):
    """Build a full env dict for finish_check.check_jobs."""
    base = {
        "R_COORDINATOR": "success",
        "RUN_MAIN_TEST": "true",
        "RUN_PALLAS_BENCH": "false",
        "RUN_FULL": "false",
        "REQUIRES_4TPU": "false",
        "RUN_ACCURACY_EXTRA": "false",
        "RUN_PERF": "false",
        "RUN_PERF_TRACE": "false",
        "R_UNIT_1": "success",
        "R_UNIT_4": "success",
        "R_CPU": "success",
        "R_E2E_1": "success",
        "R_E2E_4": "success",
        "R_ACC_1": "success",
        "R_ACC_4": "success",
        "R_PERF_1": "success",
        "R_PERF_4": "success",
        "R_MULTI_CHIP": "skipped",
        "R_ACC_EXTRA": "skipped",
        "R_PERF_EXTRA": "skipped",
        "R_PERF_TRACE": "skipped",
        "R_PALLAS": "skipped",
    }
    base.update(overrides)
    return base


class TestCheckJobs(unittest.TestCase):
    def test_normal_pr_all_pass(self):
        failures = check_jobs(_make_env())
        self.assertEqual(failures, [])

    def test_coordinator_failure(self):
        failures = check_jobs(_make_env(R_COORDINATOR="failure"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], ("coordinator", "failure"))

    def test_coordinator_failure_short_circuits(self):
        failures = check_jobs(_make_env(R_COORDINATOR="failure", R_UNIT_1="failure"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "coordinator")

    def test_unit_test_failure(self):
        failures = check_jobs(_make_env(R_UNIT_1="failure"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], ("unit-test-1-tpu", "failure"))

    def test_cpu_test_failure(self):
        failures = check_jobs(_make_env(R_CPU="failure"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], ("unit-test-cpu", "failure"))

    def test_e2e_test_failure(self):
        failures = check_jobs(_make_env(R_E2E_4="failure"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], ("e2e-test-4-tpu", "failure"))

    def test_optional_skipped_when_not_triggered(self):
        failures = check_jobs(_make_env(R_MULTI_CHIP="skipped"))
        self.assertEqual(failures, [])

    def test_optional_failure_with_label(self):
        failures = check_jobs(_make_env(REQUIRES_4TPU="true", R_MULTI_CHIP="failure"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], ("multi-chip-extra-test-4-tpu", "failure"))

    def test_optional_failure_with_run_full(self):
        failures = check_jobs(
            _make_env(
                RUN_FULL="true",
                R_MULTI_CHIP="failure",
                R_ACC_EXTRA="success",
                R_PERF_EXTRA="success",
                R_PERF_TRACE="success",
            )
        )
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "multi-chip-extra-test-4-tpu")

    def test_run_full_checks_all_optional(self):
        failures = check_jobs(
            _make_env(
                RUN_FULL="true",
                R_MULTI_CHIP="failure",
                R_ACC_EXTRA="failure",
                R_PERF_EXTRA="failure",
                R_PERF_TRACE="failure",
            )
        )
        failed_names = [f[0] for f in failures]
        self.assertIn("multi-chip-extra-test-4-tpu", failed_names)
        self.assertIn("accuracy-extra-test-1-tpu", failed_names)
        self.assertIn("perf-extra-test-1-tpu", failed_names)
        self.assertIn("perf-trace-test-1-tpu", failed_names)

    def test_pallas_failure(self):
        failures = check_jobs(_make_env(RUN_PALLAS_BENCH="true", R_PALLAS="failure"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], ("pallas-kernel-benchmark", "failure"))

    def test_pallas_skipped_when_not_triggered(self):
        failures = check_jobs(_make_env(RUN_PALLAS_BENCH="false", R_PALLAS="skipped"))
        self.assertEqual(failures, [])

    def test_no_tests_all_skipped(self):
        failures = check_jobs(
            _make_env(
                RUN_MAIN_TEST="false",
                RUN_PALLAS_BENCH="false",
                R_UNIT_1="skipped",
                R_UNIT_4="skipped",
                R_CPU="skipped",
                R_E2E_1="skipped",
                R_E2E_4="skipped",
                R_ACC_1="skipped",
                R_ACC_4="skipped",
                R_PERF_1="skipped",
                R_PERF_4="skipped",
            )
        )
        self.assertEqual(failures, [])

    def test_mandatory_skipped_is_failure(self):
        failures = check_jobs(_make_env(R_PERF_1="skipped"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], ("performance-test-1-tpu", "skipped"))

    def test_multiple_mandatory_failures_all_reported(self):
        failures = check_jobs(
            _make_env(R_UNIT_1="failure", R_E2E_4="failure", R_PERF_1="cancelled")
        )
        failed_names = [f[0] for f in failures]
        self.assertIn("unit-test-1-tpu", failed_names)
        self.assertIn("e2e-test-4-tpu", failed_names)
        self.assertIn("performance-test-1-tpu", failed_names)
        self.assertEqual(len(failures), 3)

    def test_coordinator_cancelled(self):
        failures = check_jobs(_make_env(R_COORDINATOR="cancelled"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], ("coordinator", "cancelled"))

    def test_optional_gated_when_main_test_false(self):
        failures = check_jobs(
            _make_env(
                RUN_MAIN_TEST="false",
                RUN_FULL="true",
                R_UNIT_1="skipped",
                R_UNIT_4="skipped",
                R_CPU="skipped",
                R_E2E_1="skipped",
                R_E2E_4="skipped",
                R_ACC_1="skipped",
                R_ACC_4="skipped",
                R_PERF_1="skipped",
                R_PERF_4="skipped",
                R_MULTI_CHIP="skipped",
                R_ACC_EXTRA="skipped",
                R_PERF_EXTRA="skipped",
                R_PERF_TRACE="skipped",
            )
        )
        self.assertEqual(failures, [])


class TestWriteOutputs(unittest.TestCase):
    def test_writes_name_value_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            with patch.dict("os.environ", {"GITHUB_OUTPUT": path}):
                write_outputs({"run_full": "true", "run_main_test": "false"})
            with open(path) as f:
                content = f.read()
            self.assertEqual(content, "run_full=true\nrun_main_test=false\n")
        finally:
            os.remove(path)

    def test_no_github_output_no_crash(self):
        with patch.dict("os.environ", {}, clear=True):
            write_outputs({"run_full": "true"})


class TestYAMLConsistency(unittest.TestCase):
    """Verify coordinator.yml, pr-test.yml, and finish_check.py stay in sync."""

    REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")

    @classmethod
    def setUpClass(cls):
        import re

        import yaml

        coord_path = os.path.join(cls.REPO_ROOT, ".github/workflows/coordinator.yml")
        pr_path = os.path.join(cls.REPO_ROOT, ".github/workflows/pr-test.yml")

        with open(coord_path) as f:
            cls.coordinator = yaml.safe_load(f)
        with open(pr_path) as f:
            cls.pr_test = yaml.safe_load(f)
        with open(pr_path) as f:
            cls.pr_test_raw = f.read()

        cls.coord_workflow_outputs = set(cls.coordinator[True]["workflow_call"]["outputs"].keys())
        cls.coord_job_outputs = set(cls.coordinator["jobs"]["decide"]["outputs"].keys())
        cls.pr_test_output_refs = set(
            re.findall(r"needs\.coordinator\.outputs\.([a-z0-9_]+)", cls.pr_test_raw)
        )

    def test_coordinator_workflow_outputs_match_job_outputs(self):
        self.assertEqual(
            self.coord_workflow_outputs,
            self.coord_job_outputs,
            "coordinator.yml workflow_call outputs and job outputs must match",
        )

    def test_pr_test_references_valid_coordinator_outputs(self):
        extra = self.pr_test_output_refs - self.coord_workflow_outputs
        self.assertEqual(
            extra,
            set(),
            f"pr-test.yml references outputs not in coordinator.yml: {extra}",
        )

    def test_finish_check_env_vars_all_passed(self):
        from finish_check import MANDATORY_JOBS, OPTIONAL_JOBS, PALLAS_JOBS

        expected = {"R_COORDINATOR"}
        for jobs in MANDATORY_JOBS.values():
            for _, env_key in jobs:
                expected.add(env_key)
        for _, env_key, _ in OPTIONAL_JOBS:
            expected.add(env_key)
        for _, env_key in PALLAS_JOBS:
            expected.add(env_key)
        expected.update(
            [
                "RUN_MAIN_TEST",
                "RUN_PALLAS_BENCH",
                "REQUIRES_4TPU",
                "RUN_FULL",
                "RUN_ACCURACY_EXTRA",
                "RUN_PERF",
                "RUN_PERF_TRACE",
            ]
        )

        finish_job = self.pr_test["jobs"]["pr-test-finish"]
        actual = set()
        for step in finish_job["steps"]:
            if "env" in step:
                actual.update(step["env"].keys())

        missing = expected - actual
        self.assertEqual(
            missing,
            set(),
            f"finish_check.py expects env vars not passed in pr-test-finish: {missing}",
        )

    def test_finish_needs_includes_all_checked_jobs(self):
        from finish_check import MANDATORY_JOBS, OPTIONAL_JOBS, PALLAS_JOBS

        checked_jobs = set()
        for jobs in MANDATORY_JOBS.values():
            for job_name, _ in jobs:
                checked_jobs.add(job_name)
        for job_name, _, _ in OPTIONAL_JOBS:
            checked_jobs.add(job_name)
        for job_name, _ in PALLAS_JOBS:
            checked_jobs.add(job_name)

        finish_needs = set(self.pr_test["jobs"]["pr-test-finish"]["needs"])
        missing = checked_jobs - finish_needs
        self.assertEqual(
            missing,
            set(),
            f"Jobs in finish_check.py but not in pr-test-finish needs: {missing}",
        )

    def test_stage_dependencies(self):
        jobs = self.pr_test["jobs"]

        stage1_jobs = ["unit-test-1-tpu", "unit-test-cpu"]
        for name in stage1_jobs:
            self.assertIn("coordinator", jobs[name]["needs"])
            self.assertNotIn("unit-test-4-tpu", jobs[name].get("needs", []))

        stage2_jobs = [
            "unit-test-4-tpu",
            "e2e-test-1-tpu",
            "e2e-test-4-tpu",
            "accuracy-test-1-tpu",
            "accuracy-test-4-tpu",
        ]
        for name in stage2_jobs:
            self.assertIn(
                "unit-test-1-tpu",
                jobs[name]["needs"],
                f"{name} must depend on unit-test-1-tpu",
            )

        stage3_jobs = ["performance-test-1-tpu", "performance-test-4-tpu"]
        for name in stage3_jobs:
            for dep in stage2_jobs:
                self.assertIn(
                    dep,
                    jobs[name]["needs"],
                    f"{name} must depend on {dep}",
                )


class TestCoordinatorDecideIntegration(unittest.TestCase):
    def _run_script(self, env_overrides):
        env = {
            "PATH": os.environ.get("PATH", ""),
            "EVENT_NAME": "pull_request",
            "PR_DRAFT": "false",
            "PR_LABELS": "[]",
            "MAIN_PACKAGE": "true",
            "PALLAS_KERNEL": "false",
        }
        env.update(env_overrides)
        result = subprocess.run(
            [sys.executable, "scripts/ci/coordinator_decide.py"],
            capture_output=True,
            text=True,
            env=env,
            cwd=".",
        )
        return result

    def test_normal_pr(self):
        result = self._run_script({})
        self.assertEqual(result.returncode, 0)
        self.assertIn("run_main_test: true", result.stdout)

    def test_draft_pr_exits_1(self):
        result = self._run_script({"PR_DRAFT": "true"})
        self.assertEqual(result.returncode, 1)
        self.assertIn("draft", result.stdout.lower())

    def test_push_event(self):
        result = self._run_script({"EVENT_NAME": "push"})
        self.assertEqual(result.returncode, 0)
        self.assertIn("run_main_test: true", result.stdout)

    def test_no_relevant_changes(self):
        result = self._run_script({"MAIN_PACKAGE": "false", "PALLAS_KERNEL": "false"})
        self.assertEqual(result.returncode, 0)
        self.assertIn("run_main_test: false", result.stdout)
        self.assertIn("run_pallas_bench: false", result.stdout)


if __name__ == "__main__":
    unittest.main()
