"""Shared helpers for the nightly failure pipeline.

These primitives are reused across the Slack notifier, the GitHub issue
creator, and the nightly status builder/checker so that timestamp formatting,
``gh`` invocation, Slack mrkdwn escaping, and the failed-job-to-issue
association all behave identically wherever they appear.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 ``...Z`` string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def escape_mrkdwn(text: Any) -> str:
    """Escape Slack mrkdwn special characters."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def gh(
    args: list[str], *, input_text: str | None = None, timeout: int = 180, check: bool = True
) -> str:
    """Run a ``gh`` CLI command and return its stripped stdout."""
    result = subprocess.run(
        ["gh"] + args,
        input=input_text,
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
    )
    return result.stdout.strip()


def gh_json(args: list[str], *, input_text: str | None = None, timeout: int = 180) -> Any:
    """Run a ``gh`` command and parse its stdout as JSON (``{}`` when empty)."""
    output = gh(args, input_text=input_text, timeout=timeout)
    return json.loads(output) if output else {}


def index_failure_issues(
    failure_issues: dict[str, Any] | None
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index ``failure_issues.json`` records by ``(job_name, failure_type)``."""
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    if not failure_issues:
        return mapping
    for item in failure_issues.get("failed_jobs", []):
        mapping[(item.get("job_name", ""), item.get("failure_type", ""))] = item
    return mapping


def load_failure_issues(path: str | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Read ``failure_issues.json`` from ``path`` and return its issue index."""
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return index_failure_issues(data)


def lookup_issue(
    index: dict[tuple[str, str], dict[str, Any]], job_name: str, failure_type: str
) -> dict[str, Any] | None:
    """Resolve the issue record for a failed job from a pre-built index.

    Falls back to a ``(job_name, "")`` key so callers that omit a failure type
    still match. No current producer emits an empty failure type, so the
    fallback is inert today; it keeps the lookup tolerant of that shape.
    """
    return index.get((job_name, failure_type)) or index.get((job_name, ""))


def issue_for_job(
    job: dict[str, Any], index: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any] | None:
    """Resolve the issue record for a classification job dict."""
    return lookup_issue(index, job.get("name", ""), job.get("failure_type", ""))


def load_ai_analysis(path: str | None) -> dict[str, dict[str, str]]:
    """Read ai_analysis.json ({jobs:[{name,root_cause,fix}]}) indexed by job name.

    Returns {} when absent/empty/malformed so analysis stays optional and a
    broken AI step never blocks issues or Slack.
    """
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return {j.get("name", ""): j for j in data.get("jobs", []) if j.get("name")}
