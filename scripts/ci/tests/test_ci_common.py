"""Tests for the shared nightly-pipeline helpers in ci_common."""

import json
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import ci_common


class TestUtcNow(unittest.TestCase):
    def test_iso_z_suffix_and_parseable(self):
        value = ci_common.utc_now()
        self.assertTrue(value.endswith("Z"))
        # Round-trips back to a tz-aware datetime.
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        self.assertIsNotNone(parsed.tzinfo)


class TestEscapeMrkdwn(unittest.TestCase):
    def test_escapes_special_chars(self):
        self.assertEqual(ci_common.escape_mrkdwn("a & b < c > d"), "a &amp; b &lt; c &gt; d")

    def test_coerces_non_string(self):
        self.assertEqual(ci_common.escape_mrkdwn(123), "123")


class TestGhJson(unittest.TestCase):
    @patch.object(ci_common.subprocess, "run")
    def test_parses_json_stdout(self, run):
        run.return_value = Mock(stdout='{"number": 7}\n')
        self.assertEqual(ci_common.gh_json(["api", "x"]), {"number": 7})

    @patch.object(ci_common.subprocess, "run")
    def test_empty_stdout_returns_empty_dict(self, run):
        run.return_value = Mock(stdout="   ")
        self.assertEqual(ci_common.gh_json(["api", "x"]), {})

    @patch.object(ci_common.subprocess, "run")
    def test_passes_input_text(self, run):
        run.return_value = Mock(stdout="{}")
        ci_common.gh_json(["api", "x"], input_text='{"a": 1}')
        self.assertEqual(run.call_args.kwargs["input"], '{"a": 1}')


class TestFailureIssueAssociation(unittest.TestCase):
    FIXTURE = {
        "failed_jobs": [
            {"job_name": "e2e", "failure_type": "bug", "issue_number": 5, "issue_url": "u5"},
            {"job_name": "perf", "failure_type": "timeout", "issue_number": 6, "issue_url": "u6"},
        ]
    }

    def test_index_keys_by_name_and_type(self):
        index = ci_common.index_failure_issues(self.FIXTURE)
        self.assertEqual(index[("e2e", "bug")]["issue_number"], 5)
        self.assertEqual(index[("perf", "timeout")]["issue_number"], 6)

    def test_index_handles_none(self):
        self.assertEqual(ci_common.index_failure_issues(None), {})

    def test_lookup_exact_match(self):
        index = ci_common.index_failure_issues(self.FIXTURE)
        self.assertEqual(ci_common.lookup_issue(index, "e2e", "bug")["issue_number"], 5)

    def test_lookup_miss_returns_none(self):
        index = ci_common.index_failure_issues(self.FIXTURE)
        self.assertIsNone(ci_common.lookup_issue(index, "e2e", "infrastructure"))

    def test_issue_for_job_uses_job_fields(self):
        index = ci_common.index_failure_issues(self.FIXTURE)
        job = {"name": "perf", "failure_type": "timeout"}
        self.assertEqual(ci_common.issue_for_job(job, index)["issue_url"], "u6")

    def test_load_failure_issues_from_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "failure_issues.json")
            with open(path, "w") as f:
                json.dump(self.FIXTURE, f)
            index = ci_common.load_failure_issues(path)
            self.assertEqual(index[("e2e", "bug")]["issue_number"], 5)

    def test_load_failure_issues_empty_path(self):
        self.assertEqual(ci_common.load_failure_issues(None), {})


class TestLoadAiAnalysis(unittest.TestCase):
    def test_indexes_by_job_name(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ai_analysis.json")
            with open(path, "w") as f:
                json.dump({"jobs": [{"name": "job-a", "root_cause": "rc", "fix": "fx"}]}, f)
            index = ci_common.load_ai_analysis(path)
            self.assertEqual(index["job-a"]["root_cause"], "rc")

    def test_missing_path_returns_empty(self):
        self.assertEqual(ci_common.load_ai_analysis(None), {})
        self.assertEqual(ci_common.load_ai_analysis("/no/such/file.json"), {})

    def test_malformed_returns_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ai_analysis.json")
            with open(path, "w") as f:
                f.write("not json{{")
            self.assertEqual(ci_common.load_ai_analysis(path), {})


if __name__ == "__main__":
    unittest.main()
