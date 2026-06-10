"""Build and check nightly health status for release gating."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from ci_common import gh_json, index_failure_issues, lookup_issue, utc_now
from failure_classifier import is_finish_gate

DEFAULT_OUTPUT = "nightly-status.json"


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_json_file(path: str | None, default: Any) -> Any:
    if not path:
        return default
    with open(path) as f:
        return json.load(f)


def load_json_dict(path: str, description: str) -> dict[str, Any]:
    try:
        data = load_json_file(path, None)
    except Exception as exc:
        raise ValueError(f"{description} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{description} must be a JSON object")
    return data


def normalize_failed_jobs(
    classification: dict[str, Any] | None,
    failure_issues: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not classification:
        return []
    index = index_failure_issues(failure_issues)
    failed = []
    for job in classification.get("failed_jobs", []):
        if is_finish_gate(job.get("name", "")):
            continue
        failure_type = job.get("failure_type", "bug")
        issue = lookup_issue(index, job.get("name", ""), failure_type) or {}
        failed.append(
            {
                "name": job.get("name", ""),
                "url": job.get("html_url", ""),
                "conclusion": job.get("conclusion", "failure"),
                "failure_type": failure_type,
                "issue_number": issue.get("issue_number"),
                "issue_url": issue.get("issue_url"),
            }
        )
    return failed


def build_status(
    *,
    workflow_name: str,
    run_id: str,
    run_url: str,
    head_sha: str,
    head_branch: str,
    conclusion: str,
    classification: dict[str, Any] | None = None,
    failure_issues: dict[str, Any] | None = None,
    latest_successful_nightly: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failed_jobs = normalize_failed_jobs(classification, failure_issues)
    normalized_conclusion = conclusion or "unknown"
    status = "unknown"
    if normalized_conclusion == "success":
        status = "healthy"
    elif normalized_conclusion in {"failure", "cancelled", "timed_out", "startup_failure"}:
        status = "unhealthy"

    latest_success = latest_successful_nightly
    if status == "healthy":
        latest_success = {
            "run_id": str(run_id),
            "run_url": run_url,
            "completed_at": utc_now(),
        }

    return {
        "schema_version": 1,
        "updated_at": utc_now(),
        "workflow_name": workflow_name,
        "run_id": str(run_id),
        "run_url": run_url,
        "head_sha": head_sha,
        "head_branch": head_branch,
        "conclusion": normalized_conclusion,
        "status": status,
        "failed_jobs": failed_jobs,
        "latest_successful_nightly": latest_success,
    }


def check_release_safe(
    status: dict[str, Any],
    now: datetime | None = None,
    max_age_days: int = 7,
    max_status_age_hours: int = 48,
    latest_completed_nightly: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    now = now or datetime.now(timezone.utc)
    reasons = []
    current_status = status.get("status", "unknown")
    if current_status != "healthy":
        reasons.append(f"current nightly status is {current_status}")

    updated_at = parse_time(status.get("updated_at"))
    if updated_at is None:
        reasons.append("nightly status updated_at is missing or invalid")
    elif now - updated_at > timedelta(hours=max_status_age_hours):
        reasons.append(f"nightly status is older than {max_status_age_hours} hours")

    if latest_completed_nightly is not None:
        latest_run_id = str(latest_completed_nightly.get("run_id") or "")
        status_run_id = str(status.get("run_id") or "")
        if not latest_run_id:
            reasons.append("latest completed nightly cannot be verified")
        elif status_run_id != latest_run_id:
            reasons.append(
                f"nightly status run_id {status_run_id or 'unknown'} does not match latest completed nightly {latest_run_id}"
            )
        latest_conclusion = latest_completed_nightly.get("conclusion")
        if latest_conclusion and latest_conclusion != "success":
            reasons.append(f"latest completed nightly conclusion is {latest_conclusion}")

    latest = status.get("latest_successful_nightly") or {}
    completed_at = parse_time(latest.get("completed_at"))
    if not latest.get("run_id") or completed_at is None:
        reasons.append("no successful nightly is recorded")
    elif now - completed_at > timedelta(days=max_age_days):
        reasons.append(f"latest successful nightly is older than {max_age_days} days")

    return not reasons, reasons


def create_audit_issue(
    repo: str,
    justification: str,
    status_url: str = "",
    *,
    actor: str = "",
    run_id: str = "",
    run_url: str = "",
    run_attempt: str = "",
    tag: str = "",
    version: str = "",
) -> dict[str, Any]:
    title = "Nightly release gateway emergency override"
    body = (
        "## Emergency Release Override\n\n"
        f"| Field | Value |\n"
        f"| --- | --- |\n"
        f"| Actor | {actor or 'unknown'} |\n"
        f"| Release tag | `{tag or 'unknown'}` |\n"
        f"| Release version | `{version or 'unknown'}` |\n"
        f"| Workflow run | {f'[{run_id}]({run_url})' if run_id and run_url else run_id or 'unknown'} |\n"
        f"| Run attempt | {run_attempt or 'unknown'} |\n"
        f"| Status source | {status_url or 'not provided'} |\n"
        f"| Created at | {utc_now()} |\n\n"
        f"Justification:\n\n{justification}\n"
    )
    data = gh_json(
        ["api", f"repos/{repo}/issues", "--method", "POST", "--input", "-"],
        input_text=json.dumps({"title": title, "body": body}),
    )
    return {"number": data["number"], "url": data["html_url"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly status builder/checker")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--workflow-name", required=True)
    build.add_argument("--run-id", required=True)
    build.add_argument("--run-url", required=True)
    build.add_argument("--head-sha", default="")
    build.add_argument("--head-branch", default="")
    build.add_argument("--conclusion", default="unknown")
    build.add_argument("--classification")
    build.add_argument("--failure-issues")
    build.add_argument("--latest-successful-nightly")
    build.add_argument("--output", default=DEFAULT_OUTPUT)

    check = sub.add_parser("check")
    check.add_argument("--status", required=True)
    check.add_argument("--latest-completed-nightly")
    check.add_argument("--skip-nightly-check", action="store_true")
    check.add_argument("--emergency-justification", default="")
    check.add_argument("--repo", default="")
    check.add_argument("--status-url", default="")
    check.add_argument("--actor", default="")
    check.add_argument("--run-id", default="")
    check.add_argument("--run-url", default="")
    check.add_argument("--run-attempt", default="")
    check.add_argument("--tag", default="")
    check.add_argument("--version", default="")

    args = parser.parse_args()
    if args.command == "build":
        status = build_status(
            workflow_name=args.workflow_name,
            run_id=args.run_id,
            run_url=args.run_url,
            head_sha=args.head_sha,
            head_branch=args.head_branch,
            conclusion=args.conclusion,
            classification=load_json_file(args.classification, None),
            failure_issues=load_json_file(args.failure_issues, None),
            latest_successful_nightly=load_json_file(args.latest_successful_nightly, None),
        )
        with open(args.output, "w") as f:
            json.dump(status, f, indent=2)
            f.write("\n")
        print(f"Wrote {args.output}")
        return

    if args.skip_nightly_check:
        if not args.emergency_justification.strip():
            print("Emergency override requires a non-empty justification.")
            raise SystemExit(1)
        if not args.repo:
            print("Emergency override requires --repo to create an audit issue.")
            raise SystemExit(1)
        audit = create_audit_issue(
            args.repo,
            args.emergency_justification,
            args.status_url,
            actor=args.actor,
            run_id=args.run_id,
            run_url=args.run_url,
            run_attempt=args.run_attempt,
            tag=args.tag,
            version=args.version,
        )
        print("gateway-status=emergency")
        print(f"audit-issue-number={audit['number']}")
        print(f"audit-issue-url={audit['url']}")
        return

    try:
        status = load_json_dict(args.status, "nightly status")
        latest_completed = (
            load_json_dict(args.latest_completed_nightly, "latest completed nightly")
            if args.latest_completed_nightly
            else None
        )
    except ValueError as exc:
        print("gateway-status=fail")
        print(f"- {exc}")
        raise SystemExit(1)
    ok, reasons = check_release_safe(status, latest_completed_nightly=latest_completed)
    if ok:
        print("gateway-status=pass")
        return
    print("gateway-status=fail")
    for reason in reasons:
        print(f"- {reason}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
