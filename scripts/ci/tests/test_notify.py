"""Tests for failure_classifier, slack_notify, and regression_notify."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

from failure_classifier import (
    _FINISH_GATE_RE,
    _INFRASTRUCTURE_RE,
    classify_failure,
    is_finish_gate,
)
from regression_notify import _escape_mrkdwn, format_regression_summary
from slack_notify import format_slack_summary


class TestClassifyFailure(unittest.TestCase):
    """Test classify_failure regex-based classification."""

    def test_scheduling_failure_is_infrastructure(self):
        log = "FailedScheduling pod/gke-test-0 0/3 nodes available"
        self.assertEqual(classify_failure(log), "infrastructure")

    def test_not_trigger_scale_up_is_infrastructure(self):
        log = "NotTriggerScaleUp: max node group size reached"
        self.assertEqual(classify_failure(log), "infrastructure")

    def test_node_affinity_is_infrastructure(self):
        log = "0/3 nodes: didn't match Pod's node affinity/selector"
        self.assertEqual(classify_failure(log), "infrastructure")

    def test_scale_up_failed_is_infrastructure(self):
        self.assertEqual(classify_failure("in backoff after failed scale-up"), "infrastructure")
        self.assertEqual(classify_failure("scale-up failed for pool"), "infrastructure")

    def test_real_bug_not_misclassified(self):
        self.assertEqual(classify_failure("FAILED test_foo - AssertionError"), "bug")

    def test_real_bug_with_finish_text_not_misclassified(self):
        """Regression: 'did not succeed' in logs must NOT make a real bug into infra."""
        log = (
            "FAILED test_foo.py::test_bar - AssertionError: expected 1 got 2\n"
            "::error::job-x did not succeed: failure\n"
            "::error::One or more daily test jobs failed\n"
        )
        self.assertEqual(classify_failure(log), "bug")

    def test_oom_is_resource_exhaustion(self):
        self.assertEqual(
            classify_failure("OutOfMemoryError: RESOURCE_EXHAUSTED"), "resource_exhaustion"
        )

    def test_timeout_is_timeout(self):
        self.assertEqual(classify_failure("the operation was cancelled"), "timeout")

    def test_connection_refused_is_infrastructure(self):
        self.assertEqual(classify_failure("connection refused"), "infrastructure")

    def test_preempted_is_infrastructure(self):
        self.assertEqual(classify_failure("TPU was preempted"), "infrastructure")

    def test_empty_log_is_bug(self):
        self.assertEqual(classify_failure(""), "bug")

    def test_conclusion_startup_failure(self):
        self.assertEqual(
            classify_failure("any log", conclusion="startup_failure"), "infrastructure"
        )

    def test_conclusion_timed_out(self):
        self.assertEqual(classify_failure("any log", conclusion="timed_out"), "timeout")


class TestFinishGateRegex(unittest.TestCase):
    """Test _FINISH_GATE_RE matches finish-gate job names."""

    def test_nightly_finish(self):
        self.assertTrue(_FINISH_GATE_RE.search("nightly-test-daily-finish"))

    def test_pr_finish(self):
        self.assertTrue(_FINISH_GATE_RE.search("pr-test-finish"))

    def test_weekly_finish(self):
        self.assertTrue(_FINISH_GATE_RE.search("nightly-test-finish"))

    def test_regular_job_no_match(self):
        self.assertIsNone(_FINISH_GATE_RE.search("e2e-test-4-tpu (2)"))
        self.assertIsNone(_FINISH_GATE_RE.search("unit-test-1-tpu (0)"))

    def test_finish_mid_name_no_match(self):
        self.assertIsNone(_FINISH_GATE_RE.search("finish-setup-job"))


class TestIsFinishGate(unittest.TestCase):
    """Shared is_finish_gate predicate used by slack/issue/regression."""

    def test_matches_finish_gates(self):
        self.assertTrue(is_finish_gate("pr-test-finish"))
        self.assertTrue(is_finish_gate("nightly-test-daily-finish"))

    def test_rejects_real_jobs(self):
        self.assertFalse(is_finish_gate("e2e-test-4-tpu (2)"))
        self.assertFalse(is_finish_gate("finish-setup-job"))

    def test_handles_none(self):
        self.assertFalse(is_finish_gate(None))


class TestInfrastructureRegexNoFalsePositives(unittest.TestCase):
    """Ensure _INFRASTRUCTURE_RE does NOT match generic phrases."""

    def test_did_not_succeed_not_matched(self):
        self.assertIsNone(_INFRASTRUCTURE_RE.search("job did not succeed: failure"))

    def test_one_or_more_jobs_failed_not_matched(self):
        self.assertIsNone(_INFRASTRUCTURE_RE.search("One or more daily test jobs failed"))

    def test_assertion_error_not_matched(self):
        self.assertIsNone(_INFRASTRUCTURE_RE.search("AssertionError: expected 1 got 2"))


class TestFormatSlackSummary(unittest.TestCase):
    """Test format_slack_summary output."""

    def _make_jobs(self, names_and_types):
        return [
            {"name": n, "failure_type": t, "html_url": f"https://example.com/{i}"}
            for i, (n, t) in enumerate(names_and_types)
        ]

    def test_finish_gate_excluded_from_count(self):
        jobs = self._make_jobs(
            [
                ("e2e-test-4-tpu (0)", "bug"),
                ("nightly-test-daily-finish", "infrastructure"),
            ]
        )
        summary = format_slack_summary("https://run", "abc1234567", "author", jobs)
        self.assertIn("1 job failed", summary)
        self.assertNotIn("2 jobs", summary)
        self.assertNotIn("nightly-test-daily-finish", summary)

    def test_singular_plural(self):
        jobs_1 = self._make_jobs([("job-a", "bug")])
        jobs_2 = self._make_jobs([("job-a", "bug"), ("job-b", "timeout")])
        self.assertIn("1 job failed", format_slack_summary("url", "sha1234567", "a", jobs_1))
        self.assertIn("2 jobs failed", format_slack_summary("url", "sha1234567", "a", jobs_2))

    def test_author_escaped(self):
        jobs = self._make_jobs([("job", "bug")])
        summary = format_slack_summary("url", "sha1234567", "a<b&c", jobs)
        self.assertIn("a&lt;b&amp;c", summary)
        self.assertNotIn("a<b&c", summary)

    def test_truncation(self):
        jobs = self._make_jobs([(f"very-long-job-name-{i}", "bug") for i in range(100)])
        summary = format_slack_summary("url", "sha1234567", "author", jobs)
        self.assertLessEqual(len(summary), 2950)
        self.assertIn("truncated", summary)

    def test_includes_issue_link_when_provided(self):
        jobs = self._make_jobs([("job-a", "bug")])
        summary = format_slack_summary(
            "https://run",
            "abc1234567",
            "author",
            jobs,
            failure_issues={
                ("job-a", "bug"): {
                    "issue_number": 123,
                    "issue_url": "https://github.com/sgl-project/sglang-jax/issues/123",
                }
            },
        )
        self.assertIn("<https://example.com/0|job-a>", summary)
        self.assertIn(
            "Issue: <https://github.com/sgl-project/sglang-jax/issues/123|#123>",
            summary,
        )

    def test_works_without_failure_issues(self):
        jobs = self._make_jobs([("job-a", "bug")])
        summary = format_slack_summary("https://run", "abc1234567", "author", jobs)
        self.assertIn("job-a", summary)
        self.assertNotIn("Issue:", summary)

    def test_includes_ai_root_cause_per_job(self):
        jobs = self._make_jobs([("job-a", "bug")])
        summary = format_slack_summary(
            "https://run",
            "abc1234567",
            "author",
            jobs,
            analysis={"job-a": {"root_cause": "null deref in foo", "fix": "guard it"}},
        )
        self.assertIn("null deref in foo", summary)


class TestFormatRegressionSummary(unittest.TestCase):
    """Test format_regression_summary output."""

    def test_basic_format(self):
        summary = format_regression_summary(
            "abc1234",
            "https://run/1",
            "job-a, job-b",
            "bad commit",
            "revert it",
            "https://pr/1",
            42,
        )
        self.assertIn(":rotating_light:", summary)
        self.assertIn("`abc1234`", summary)
        self.assertIn("PR #42", summary)
        self.assertIn("job-a, job-b", summary)
        self.assertIn("bad commit", summary)
        self.assertIn("revert it", summary)
        self.assertNotIn("unverified", summary)

    def test_degraded_branch_unverified(self):
        summary = format_regression_summary(
            "",
            "https://run/1",
            "job-a",
            "cause",
            "fix",
            branch_unverified=True,
        )
        self.assertIn("branch unverified", summary)
        self.assertIn("unknown commit", summary)
        self.assertNotIn("`", summary.split("View run")[0].split("unknown commit")[0])

    def test_no_pr(self):
        summary = format_regression_summary(
            "abc1234",
            "https://run/1",
            "job-a",
            "cause",
            "fix",
        )
        self.assertNotIn("PR #", summary)
        self.assertIn("View run", summary)

    def test_mrkdwn_escaped(self):
        summary = format_regression_summary(
            "abc1234",
            "https://run/1",
            "job<a>",
            "a & b < c",
            "x > y",
        )
        self.assertIn("job&lt;a&gt;", summary)
        self.assertIn("a &amp; b &lt; c", summary)
        self.assertIn("x &gt; y", summary)

    def test_bulleted_sections_on_own_lines(self):
        summary = format_regression_summary(
            "abc1234",
            "https://run/1",
            "job-a",
            "• gate flipped below threshold\n• batch composition changed",
            "• loosen the gate\n• do not revert",
        )
        self.assertIn("*Root cause*\n• gate flipped below threshold", summary)
        self.assertIn("*Suggested fix*\n• loosen the gate", summary)

    def test_field_not_truncated(self):
        long_cause = "• " + "word " * 200  # long but under the Slack hard cap
        summary = format_regression_summary("abc1234", "https://run/1", "job-a", long_cause, "fix")
        self.assertIn(long_cause.strip(), summary)
        self.assertNotIn("…", summary)

    def test_empty_fields_omit_sections(self):
        summary = format_regression_summary("abc1234", "https://run/1", "job-a", "", "")
        self.assertNotIn("*Root cause*", summary)
        self.assertNotIn("*Suggested fix*", summary)


class TestEscapeMrkdwn(unittest.TestCase):
    def test_escapes_special_chars(self):
        self.assertEqual(_escape_mrkdwn("a & b < c > d"), "a &amp; b &lt; c &gt; d")

    def test_no_change_for_plain_text(self):
        self.assertEqual(_escape_mrkdwn("hello world"), "hello world")


if __name__ == "__main__":
    unittest.main()
