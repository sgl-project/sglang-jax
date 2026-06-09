"""Create or update GitHub issues for failed CI jobs.

This extends the nightly failure notification pipeline by consuming the
``classification.json`` produced by ``slack_notify.py``. It intentionally does
not fetch or classify failures itself; the existing Slack notification pipeline
owns that stage.

CODEOWNERS is not present in this repository today. Until one exists, assignees
are intentionally limited to the optional CI_FAILURE_FALLBACK_ASSIGNEES env var
(comma-separated GitHub logins).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from ci_common import gh_json, load_ai_analysis, utc_now
from failure_classifier import is_finish_gate

ISSUE_MARKER_PREFIX = "ci-failure-monitor"
DEFAULT_OUTPUT = "failure_issues.json"


def sanitize_marker_value(value: str) -> str:
    return str(value).replace("\n", " ").replace("--", "-").strip()


def build_marker(workflow_name: str, job_name: str, failure_type: str) -> str:
    workflow = sanitize_marker_value(workflow_name)
    job = sanitize_marker_value(job_name)
    kind = sanitize_marker_value(failure_type)
    return f"<!-- {ISSUE_MARKER_PREFIX}:workflow={workflow};job={job};failure_type={kind} -->"


def load_classification(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    return data.get("failed_jobs", [])


def fallback_assignees() -> list[str]:
    raw = os.environ.get("CI_FAILURE_FALLBACK_ASSIGNEES", "")
    return [item.strip().lstrip("@") for item in raw.split(",") if item.strip()]


def find_open_issue(repo: str, marker: str) -> dict[str, Any] | None:
    page = 1
    while True:
        issues = gh_json(
            [
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
        )
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
            break
        page += 1
    return None


def failure_table(
    *,
    workflow_name: str,
    run_id: str,
    run_url: str,
    job: dict[str, Any],
    commit_sha: str,
    commit_author: str,
    timestamp: str,
) -> str:
    """Render the shared failure-detail table used by issues and comments."""
    return (
        "| Field | Value |\n"
        "| --- | --- |\n"
        f"| Workflow | {workflow_name} |\n"
        f"| Run | [{run_id}]({run_url}) |\n"
        f"| Job | [{job['name']}]({job.get('html_url', '')}) |\n"
        f"| Commit | `{commit_sha}` |\n"
        f"| Commit author | {commit_author} |\n"
        f"| Failure type | `{job['failure_type']}` |\n"
        f"| Updated at | {timestamp} |"
    )


def analysis_block(analysis: dict[str, str] | None) -> str:
    """AI root-cause section for the issue body, or '' when no analysis exists."""
    if not analysis or not analysis.get("root_cause"):
        return ""
    lines = ["## Root cause (AI)", f"- {analysis['root_cause']}"]
    if analysis.get("fix"):
        lines.append(f"- Fix: {analysis['fix']}")
    return "\n".join(lines)


def issue_payload(
    *,
    workflow_name: str,
    run_id: str,
    run_url: str,
    job: dict[str, Any],
    commit_sha: str,
    commit_author: str,
    marker: str,
    timestamp: str,
    assignees: list[str],
    analysis: dict[str, str] | None = None,
) -> dict[str, Any]:
    title = f"[CI Failure] {workflow_name} / {job['name']} [{job['failure_type']}]"
    owner_note = (
        "Fallback assignees: " + ", ".join(f"@{a}" for a in assignees)
        if assignees
        else "No CODEOWNERS file found; no fallback assignee configured."
    )
    table = failure_table(
        workflow_name=workflow_name,
        run_id=run_id,
        run_url=run_url,
        job=job,
        commit_sha=commit_sha,
        commit_author=commit_author,
        timestamp=timestamp,
    )
    parts = [marker, "## CI Failure", "", table]
    block = analysis_block(analysis)
    if block:
        parts += ["", block]
    parts += ["", owner_note, ""]
    body = "\n".join(parts)
    payload: dict[str, Any] = {"title": title, "body": body}
    if assignees:
        payload["assignees"] = assignees
    return payload


def comment_body(
    *,
    workflow_name: str,
    run_id: str,
    run_url: str,
    job: dict[str, Any],
    commit_sha: str,
    commit_author: str,
    marker: str,
    timestamp: str,
    analysis: dict[str, str] | None = None,
) -> str:
    table = failure_table(
        workflow_name=workflow_name,
        run_id=run_id,
        run_url=run_url,
        job=job,
        commit_sha=commit_sha,
        commit_author=commit_author,
        timestamp=timestamp,
    )
    parts = [marker, "### CI Failure Recurrence", "", table]
    block = analysis_block(analysis)
    if block:
        parts += ["", block]
    parts += [""]
    return "\n".join(parts)


def create_issue(repo: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = gh_json(
        ["api", f"repos/{repo}/issues", "--method", "POST", "--input", "-"],
        input_text=json.dumps(payload),
    )
    if "number" not in data or "html_url" not in data:
        raise RuntimeError("GitHub issue create response missing number/html_url")
    return {"number": data["number"], "url": data["html_url"]}


def add_issue_comment(repo: str, issue_number: int, body: str) -> None:
    gh_json(
        ["api", f"repos/{repo}/issues/{issue_number}/comments", "--method", "POST", "--input", "-"],
        input_text=json.dumps({"body": body}),
    )


def process_failed_jobs(
    *,
    repo: str,
    run_id: str,
    workflow_name: str,
    run_url: str,
    commit_sha: str,
    commit_author: str,
    failed_jobs: list[dict[str, Any]],
    analysis: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    timestamp = utc_now()
    assignees = fallback_assignees()
    analysis = analysis or {}
    records = []

    for job in failed_jobs:
        if is_finish_gate(job["name"]):
            continue
        marker = build_marker(workflow_name, job["name"], job["failure_type"])
        job_analysis = analysis.get(job["name"])
        try:
            existing = find_open_issue(repo, marker)
            if existing:
                add_issue_comment(
                    repo,
                    int(existing["number"]),
                    comment_body(
                        workflow_name=workflow_name,
                        run_id=run_id,
                        run_url=run_url,
                        job=job,
                        commit_sha=commit_sha,
                        commit_author=commit_author,
                        marker=marker,
                        timestamp=timestamp,
                        analysis=job_analysis,
                    ),
                )
                action = "updated"
                issue_number = int(existing["number"])
                issue_url = existing["url"]
            else:
                # Re-check immediately before create to reduce duplicate issues
                # when a manual rerun overlaps the automatic workflow_run.
                existing = find_open_issue(repo, marker)
                if existing:
                    add_issue_comment(
                        repo,
                        int(existing["number"]),
                        comment_body(
                            workflow_name=workflow_name,
                            run_id=run_id,
                            run_url=run_url,
                            job=job,
                            commit_sha=commit_sha,
                            commit_author=commit_author,
                            marker=marker,
                            timestamp=timestamp,
                            analysis=job_analysis,
                        ),
                    )
                    action = "updated"
                    issue_number = int(existing["number"])
                    issue_url = existing["url"]
                else:
                    created = create_issue(
                        repo,
                        issue_payload(
                            workflow_name=workflow_name,
                            run_id=run_id,
                            run_url=run_url,
                            job=job,
                            commit_sha=commit_sha,
                            commit_author=commit_author,
                            marker=marker,
                            timestamp=timestamp,
                            assignees=assignees,
                            analysis=job_analysis,
                        ),
                    )
                    action = "created"
                    issue_number = int(created["number"])
                    issue_url = created["url"]

            records.append(
                {
                    "job_name": job["name"],
                    "job_url": job.get("html_url", ""),
                    "failure_type": job["failure_type"],
                    "issue_number": issue_number,
                    "issue_url": issue_url,
                    "action": action,
                    "marker": marker,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "job_name": job["name"],
                    "job_url": job.get("html_url", ""),
                    "failure_type": job["failure_type"],
                    "issue_number": None,
                    "issue_url": "",
                    "action": "error",
                    "error": str(exc),
                    "marker": marker,
                }
            )

    return {
        "schema_version": 1,
        "updated_at": timestamp,
        "workflow_name": workflow_name,
        "run_id": str(run_id),
        "run_url": run_url,
        "failed_jobs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/update GitHub issues for CI failures")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workflow-name", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--commit-author", default="unknown")
    parser.add_argument(
        "--classification",
        required=True,
        help="classification.json from slack_notify.py",
    )
    parser.add_argument("--ai-analysis", help="ai_analysis.json from the AI analysis step")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    failed_jobs = load_classification(args.classification)
    if not failed_jobs:
        result = {
            "schema_version": 1,
            "updated_at": utc_now(),
            "workflow_name": args.workflow_name,
            "run_id": str(args.run_id),
            "run_url": args.run_url,
            "failed_jobs": [],
        }
    else:
        result = process_failed_jobs(
            repo=args.repo,
            run_id=args.run_id,
            workflow_name=args.workflow_name,
            run_url=args.run_url,
            commit_sha=args.commit_sha,
            commit_author=args.commit_author,
            failed_jobs=failed_jobs,
            analysis=load_ai_analysis(args.ai_analysis),
        )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
