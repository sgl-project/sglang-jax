"""Shared helpers for the nightly failure pipeline, so the Slack notifier,
issue creator, and status builder render the same failures identically."""

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


def clamp(text: str, limit: int = 600) -> str:
    """Truncate at a word boundary — backstop for when the model ignores the brief-output prompt."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + " …"


def gh_json(args: list[str], *, input_text: str | None = None, timeout: int = 180) -> Any:
    """Run a ``gh`` command and parse its stdout as JSON (``{}`` when empty)."""
    result = subprocess.run(
        ["gh"] + args,
        input=input_text,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    out = result.stdout.strip()
    return json.loads(out) if out else {}


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
    """Read ``failure_issues.json`` from ``path``; {} when absent/malformed."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return index_failure_issues(data)


def lookup_issue(
    index: dict[tuple[str, str], dict[str, Any]], job_name: str, failure_type: str
) -> dict[str, Any] | None:
    """Resolve a job's issue from the index (``(name, "")`` fallback is inert today but tolerant)."""
    return index.get((job_name, failure_type)) or index.get((job_name, ""))


def issue_for_job(
    job: dict[str, Any], index: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any] | None:
    """Resolve the issue record for a classification job dict."""
    return lookup_issue(index, job.get("name", ""), job.get("failure_type", ""))


def load_ai_analysis(path: str | None) -> dict[str, dict[str, str]]:
    """Index ai_analysis.json by job name; {} when absent/malformed so analysis stays optional."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return {j.get("name", ""): j for j in data.get("jobs", []) if j.get("name")}
