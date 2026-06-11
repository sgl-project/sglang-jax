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


def sanitize_marker_value(value: Any) -> str:
    """Flatten a value for a hidden ``<!-- ... -->`` dedup marker (drop newlines/``--``)."""
    return str(value).replace("\n", " ").replace("--", "-").strip()


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


def find_open_issue(repo: str, marker: str, labels: str | None = None) -> dict[str, Any] | None:
    """Find an open issue whose body carries ``marker`` (shared dedup model).

    ``labels`` narrows the scan to that label's open issues; ``None`` scans all.
    Returns ``{number, url, title}`` or ``None``.
    """
    page = 1
    while True:
        args = [
            "api",
            f"repos/{repo}/issues",
            "--method",
            "GET",
            "-f",
            "state=open",
            "-f",
            "per_page=100",
            "-f",
            f"page={page}",
        ]
        if labels:
            args += ["-f", f"labels={labels}"]
        issues = gh_json(args)
        if not isinstance(issues, list):
            return None
        for item in issues:
            if item.get("pull_request"):
                continue
            if marker in (item.get("body") or ""):
                return {
                    "number": item["number"],
                    "url": item["html_url"],
                    "title": item.get("title", ""),
                }
        if len(issues) < 100:
            return None
        page += 1


def create_issue(repo: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create an issue from ``payload`` (title/body/labels/...). Returns {number, url}."""
    data = gh_json(
        ["api", f"repos/{repo}/issues", "--method", "POST", "--input", "-"],
        input_text=json.dumps(payload),
    )
    if "number" not in data or "html_url" not in data:
        raise RuntimeError("GitHub issue create response missing number/html_url")
    return {"number": data["number"], "url": data["html_url"]}


def add_issue_comment(repo: str, issue_number: int, body: str) -> None:
    """Append a comment to an existing issue."""
    gh_json(
        ["api", f"repos/{repo}/issues/{issue_number}/comments", "--method", "POST", "--input", "-"],
        input_text=json.dumps({"body": body}),
    )


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
