"""Tests for CI failure issue tracking and nightly release gateway helpers."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ci_failure_issues
import nightly_status


class TestCiFailureIssues(unittest.TestCase):
    def _job(self):
        return {
            "name": "unit-test-cpu",
            "html_url": "https://github.com/o/r/actions/jobs/1",
            "failure_type": "bug",
            "conclusion": "failure",
        }

    def test_marker_generation(self):
        marker = ci_failure_issues.build_marker("Nightly Test Daily", "job-a", "bug")
        self.assertEqual(
            marker,
            "<!-- ci-failure-monitor:workflow=Nightly Test Daily;job=job-a;failure_type=bug -->",
        )

    def test_issue_body_embeds_ai_analysis(self):
        payload = ci_failure_issues.issue_payload(
            workflow_name="Nightly Test Daily",
            run_id="1",
            run_url="u",
            job=self._job(),
            commit_sha="abc",
            commit_author="me",
            marker="<!--m-->",
            timestamp="t",
            assignees=[],
            analysis={"root_cause": "null deref in foo", "fix": "guard it"},
        )
        self.assertIn("## Root cause (AI)", payload["body"])
        self.assertIn("null deref in foo", payload["body"])
        self.assertIn("Fix: guard it", payload["body"])

    def test_issue_body_without_analysis_has_no_ai_section(self):
        payload = ci_failure_issues.issue_payload(
            workflow_name="Nightly Test Daily",
            run_id="1",
            run_url="u",
            job=self._job(),
            commit_sha="abc",
            commit_author="me",
            marker="<!--m-->",
            timestamp="t",
            assignees=[],
        )
        self.assertNotIn("Root cause (AI)", payload["body"])

    @patch.object(ci_failure_issues, "gh_json")
    def test_find_open_issue_uses_get_for_search_fields(self, gh_json):
        gh_json.return_value = {"items": []}
        ci_failure_issues.find_open_issue("o/r", "marker")
        args = gh_json.call_args.args[0]
        self.assertIn("--method", args)
        self.assertEqual(args[args.index("--method") + 1], "GET")

    @patch.object(ci_failure_issues, "gh_json")
    def test_find_open_issue_matches_marker_in_body(self, gh_json):
        gh_json.return_value = [
            {
                "number": 1,
                "html_url": "https://github.com/o/r/issues/1",
                "title": "wrong",
                "body": "not it",
            },
            {
                "number": 2,
                "html_url": "https://github.com/o/r/issues/2",
                "title": "right",
                "body": "prefix marker suffix",
            },
        ]
        issue = ci_failure_issues.find_open_issue("o/r", "marker")
        self.assertEqual(issue["number"], 2)

    @patch.object(ci_failure_issues, "add_issue_comment")
    @patch.object(ci_failure_issues, "create_issue")
    @patch.object(ci_failure_issues, "find_open_issue")
    def test_existing_issue_updates(self, find_open_issue, create_issue, add_comment):
        find_open_issue.return_value = {
            "number": 77,
            "url": "https://github.com/o/r/issues/77",
        }
        result = ci_failure_issues.process_failed_jobs(
            repo="o/r",
            run_id="100",
            workflow_name="Nightly Test Daily",
            run_url="https://run",
            commit_sha="abcdef",
            commit_author="author",
            failed_jobs=[self._job()],
        )
        self.assertFalse(create_issue.called)
        self.assertTrue(add_comment.called)
        self.assertEqual(result["failed_jobs"][0]["action"], "updated")
        self.assertEqual(result["failed_jobs"][0]["issue_number"], 77)

    @patch.object(ci_failure_issues, "add_issue_comment")
    @patch.object(ci_failure_issues, "create_issue")
    @patch.object(ci_failure_issues, "find_open_issue")
    def test_missing_issue_creates(self, find_open_issue, create_issue, add_comment):
        find_open_issue.return_value = None
        create_issue.return_value = {
            "number": 88,
            "url": "https://github.com/o/r/issues/88",
        }
        result = ci_failure_issues.process_failed_jobs(
            repo="o/r",
            run_id="100",
            workflow_name="Nightly Test Daily",
            run_url="https://run",
            commit_sha="abcdef",
            commit_author="author",
            failed_jobs=[self._job()],
        )
        self.assertTrue(create_issue.called)
        self.assertFalse(add_comment.called)
        self.assertEqual(result["failed_jobs"][0]["action"], "created")
        self.assertEqual(result["failed_jobs"][0]["issue_number"], 88)

    @patch.object(ci_failure_issues, "add_issue_comment")
    @patch.object(ci_failure_issues, "create_issue")
    @patch.object(ci_failure_issues, "find_open_issue")
    def test_finish_job_is_skipped(self, find_open_issue, create_issue, add_comment):
        job = self._job()
        job["name"] = "nightly-test-daily-finish"
        result = ci_failure_issues.process_failed_jobs(
            repo="o/r",
            run_id="100",
            workflow_name="Nightly Test Daily",
            run_url="https://run",
            commit_sha="abcdef",
            commit_author="author",
            failed_jobs=[job],
        )
        self.assertEqual(result["failed_jobs"], [])
        self.assertFalse(find_open_issue.called)
        self.assertFalse(create_issue.called)
        self.assertFalse(add_comment.called)

    @patch.object(ci_failure_issues, "find_open_issue")
    def test_per_job_error_records_partial_output(self, find_open_issue):
        find_open_issue.side_effect = RuntimeError("rate limited")
        result = ci_failure_issues.process_failed_jobs(
            repo="o/r",
            run_id="100",
            workflow_name="Nightly Test Daily",
            run_url="https://run",
            commit_sha="abcdef",
            commit_author="author",
            failed_jobs=[self._job()],
        )
        self.assertEqual(result["failed_jobs"][0]["action"], "error")
        self.assertIn("rate limited", result["failed_jobs"][0]["error"])

    @patch.object(ci_failure_issues, "gh_json")
    def test_create_issue_validates_response_shape(self, gh_json):
        gh_json.return_value = {}
        with self.assertRaises(RuntimeError):
            ci_failure_issues.create_issue("o/r", {"title": "t", "body": "b"})


class TestNightlyStatus(unittest.TestCase):
    def test_build_success_status(self):
        status = nightly_status.build_status(
            workflow_name="Nightly Test Daily",
            run_id="1",
            run_url="https://run",
            head_sha="abc",
            head_branch="main",
            conclusion="success",
        )
        self.assertEqual(status["status"], "healthy")
        self.assertEqual(status["latest_successful_nightly"]["run_id"], "1")
        self.assertEqual(status["failed_jobs"], [])

    def test_build_failure_status_with_issue_links(self):
        status = nightly_status.build_status(
            workflow_name="Nightly Test Daily",
            run_id="2",
            run_url="https://run",
            head_sha="abc",
            head_branch="main",
            conclusion="failure",
            classification={
                "failed_jobs": [
                    {
                        "name": "job-a",
                        "html_url": "https://job",
                        "conclusion": "failure",
                        "failure_type": "bug",
                    }
                ]
            },
            failure_issues={
                "failed_jobs": [
                    {
                        "job_name": "job-a",
                        "failure_type": "bug",
                        "issue_number": 123,
                        "issue_url": "https://issue",
                    }
                ]
            },
        )
        self.assertEqual(status["status"], "unhealthy")
        self.assertEqual(status["failed_jobs"][0]["issue_number"], 123)
        self.assertEqual(status["failed_jobs"][0]["issue_url"], "https://issue")

    def test_gateway_passes_with_recent_successful_nightly(self):
        now = datetime(2026, 6, 8, tzinfo=timezone.utc)
        status = {
            "status": "healthy",
            "updated_at": now.isoformat(),
            "latest_successful_nightly": {
                "run_id": "1",
                "completed_at": (now - timedelta(days=2)).isoformat(),
            },
        }
        ok, reasons = nightly_status.check_release_safe(status, now=now)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_gateway_fails_with_stale_successful_nightly(self):
        now = datetime(2026, 6, 8, tzinfo=timezone.utc)
        status = {
            "status": "healthy",
            "updated_at": now.isoformat(),
            "latest_successful_nightly": {
                "run_id": "1",
                "completed_at": (now - timedelta(days=8)).isoformat(),
            },
        }
        ok, reasons = nightly_status.check_release_safe(status, now=now)
        self.assertFalse(ok)
        self.assertIn("older than 7 days", reasons[0])

    def test_gateway_fails_with_missing_successful_nightly(self):
        ok, reasons = nightly_status.check_release_safe(
            {"status": "healthy", "updated_at": datetime.now(timezone.utc).isoformat()}
        )
        self.assertFalse(ok)
        self.assertIn("no successful nightly", reasons[0])

    def test_gateway_fails_closed_on_unknown_status(self):
        now = datetime(2026, 6, 8, tzinfo=timezone.utc)
        status = {
            "status": "unknown",
            "updated_at": now.isoformat(),
            "latest_successful_nightly": {
                "run_id": "1",
                "completed_at": (now - timedelta(days=2)).isoformat(),
            },
        }
        ok, reasons = nightly_status.check_release_safe(status, now=now)
        self.assertFalse(ok)
        self.assertIn("current nightly status is unknown", reasons[0])

    def test_gateway_fails_when_status_file_is_stale(self):
        now = datetime(2026, 6, 8, tzinfo=timezone.utc)
        status = {
            "status": "healthy",
            "updated_at": (now - timedelta(hours=49)).isoformat(),
            "latest_successful_nightly": {
                "run_id": "1",
                "completed_at": (now - timedelta(days=1)).isoformat(),
            },
        }
        ok, reasons = nightly_status.check_release_safe(status, now=now)
        self.assertFalse(ok)
        self.assertIn("nightly status is older than 48 hours", reasons[0])

    def test_gateway_fails_when_status_is_stale_vs_latest_completed_nightly(self):
        now = datetime(2026, 6, 8, tzinfo=timezone.utc)
        status = {
            "status": "healthy",
            "run_id": "old",
            "updated_at": now.isoformat(),
            "latest_successful_nightly": {
                "run_id": "old",
                "completed_at": (now - timedelta(days=1)).isoformat(),
            },
        }
        ok, reasons = nightly_status.check_release_safe(
            status,
            now=now,
            latest_completed_nightly={"run_id": "new"},
        )
        self.assertFalse(ok)
        self.assertIn("does not match latest completed nightly new", reasons[0])

    def test_gateway_passes_when_status_matches_latest_completed_nightly(self):
        now = datetime(2026, 6, 8, tzinfo=timezone.utc)
        status = {
            "status": "healthy",
            "run_id": "run-1",
            "conclusion": "success",
            "updated_at": now.isoformat(),
            "latest_successful_nightly": {
                "run_id": "run-1",
                "completed_at": (now - timedelta(days=1)).isoformat(),
            },
        }
        ok, reasons = nightly_status.check_release_safe(
            status,
            now=now,
            latest_completed_nightly={"run_id": "run-1", "conclusion": "success"},
        )
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_gateway_fails_when_latest_completed_nightly_is_not_success(self):
        now = datetime(2026, 6, 8, tzinfo=timezone.utc)
        status = {
            "status": "healthy",
            "run_id": "run-1",
            "conclusion": "success",
            "updated_at": now.isoformat(),
            "latest_successful_nightly": {
                "run_id": "run-1",
                "completed_at": (now - timedelta(days=1)).isoformat(),
            },
        }
        ok, reasons = nightly_status.check_release_safe(
            status,
            now=now,
            latest_completed_nightly={"run_id": "run-1", "conclusion": "failure"},
        )
        self.assertFalse(ok)
        self.assertIn("latest completed nightly conclusion is failure", reasons[0])

    def test_gateway_fails_when_latest_completed_nightly_missing(self):
        now = datetime(2026, 6, 8, tzinfo=timezone.utc)
        status = {
            "status": "healthy",
            "run_id": "run-1",
            "updated_at": now.isoformat(),
            "latest_successful_nightly": {
                "run_id": "run-1",
                "completed_at": (now - timedelta(days=1)).isoformat(),
            },
        }
        ok, reasons = nightly_status.check_release_safe(
            status,
            now=now,
            latest_completed_nightly={},
        )
        self.assertFalse(ok)
        self.assertIn("latest completed nightly cannot be verified", reasons[0])

    def _run_check_main(self, status_data, latest_completed=None):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as status_file:
            json.dump(status_data, status_file)
            status_path = status_file.name
        latest_path = None
        if latest_completed is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as latest_file:
                json.dump(latest_completed, latest_file)
                latest_path = latest_file.name
        argv = ["nightly_status.py", "check", "--status", status_path]
        if latest_path:
            argv.extend(["--latest-completed-nightly", latest_path])
        output = io.StringIO()
        try:
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
                try:
                    nightly_status.main()
                    code = 0
                except SystemExit as exc:
                    code = exc.code
            return code, output.getvalue()
        finally:
            os.remove(status_path)
            if latest_path:
                os.remove(latest_path)

    def test_check_main_prints_pass_token_and_exits_zero(self):
        now = datetime.now(timezone.utc)
        code, output = self._run_check_main(
            {
                "status": "healthy",
                "run_id": "run-1",
                "updated_at": now.isoformat(),
                "latest_successful_nightly": {
                    "run_id": "run-1",
                    "completed_at": now.isoformat(),
                },
            },
            {"run_id": "run-1"},
        )
        self.assertEqual(code, 0)
        self.assertIn("gateway-status=pass", output)

    def test_check_main_prints_fail_token_and_exits_one(self):
        now = datetime.now(timezone.utc)
        code, output = self._run_check_main(
            {
                "status": "unhealthy",
                "run_id": "run-1",
                "updated_at": now.isoformat(),
                "latest_successful_nightly": {
                    "run_id": "run-1",
                    "completed_at": now.isoformat(),
                },
            },
            {"run_id": "run-1"},
        )
        self.assertEqual(code, 1)
        self.assertIn("gateway-status=fail", output)
        self.assertIn("current nightly status is unhealthy", output)

    def test_check_main_malformed_status_fails_cleanly(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            path = f.name
        output = io.StringIO()
        try:
            with (
                patch.object(sys, "argv", ["nightly_status.py", "check", "--status", path]),
                contextlib.redirect_stdout(output),
            ):
                with self.assertRaises(SystemExit) as cm:
                    nightly_status.main()
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("gateway-status=fail", output.getvalue())
            self.assertIn("nightly status cannot be read", output.getvalue())
        finally:
            os.remove(path)

    def test_emergency_override_requires_justification(self):
        with patch.object(
            sys,
            "argv",
            [
                "nightly_status.py",
                "check",
                "--status",
                "nightly-status.json",
                "--skip-nightly-check",
                "--repo",
                "o/r",
            ],
        ):
            with self.assertRaises(SystemExit) as cm:
                nightly_status.main()
        self.assertEqual(cm.exception.code, 1)

    def test_emergency_override_requires_repo(self):
        with patch.object(
            sys,
            "argv",
            [
                "nightly_status.py",
                "check",
                "--status",
                "nightly-status.json",
                "--skip-nightly-check",
                "--emergency-justification",
                "release hotfix",
            ],
        ):
            with self.assertRaises(SystemExit) as cm:
                nightly_status.main()
        self.assertEqual(cm.exception.code, 1)

    @patch.object(nightly_status, "create_audit_issue")
    def test_emergency_override_main_prints_metadata(self, create_audit_issue):
        create_audit_issue.return_value = {"number": 321, "url": "https://issue"}
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "nightly_status.py",
                    "check",
                    "--status",
                    "nightly-status.json",
                    "--skip-nightly-check",
                    "--emergency-justification",
                    "release hotfix",
                    "--repo",
                    "o/r",
                ],
            ),
            contextlib.redirect_stdout(output),
        ):
            nightly_status.main()
        self.assertIn("gateway-status=emergency", output.getvalue())
        self.assertIn("audit-issue-number=321", output.getvalue())
        self.assertIn("audit-issue-url=https://issue", output.getvalue())

    @patch.object(nightly_status, "gh_json")
    def test_emergency_override_creates_audit_issue_metadata(self, gh_json):
        gh_json.return_value = {
            "number": 321,
            "html_url": "https://github.com/o/r/issues/321",
        }
        audit = nightly_status.create_audit_issue(
            "o/r",
            "release hotfix",
            "https://status",
            actor="alice",
            run_id="123",
            run_url="https://github.com/o/r/actions/runs/123",
            run_attempt="2",
            tag="v1.2.3",
            version="1.2.3",
        )
        self.assertEqual(audit["number"], 321)
        self.assertEqual(audit["url"], "https://github.com/o/r/issues/321")
        payload = json.loads(gh_json.call_args.kwargs["input_text"])
        self.assertIn("alice", payload["body"])
        self.assertIn("v1.2.3", payload["body"])
        self.assertIn("1.2.3", payload["body"])
        self.assertIn("https://github.com/o/r/actions/runs/123", payload["body"])
        self.assertIn("Run attempt | 2", payload["body"])


if __name__ == "__main__":
    unittest.main()
