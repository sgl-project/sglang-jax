"""CI Failure Monitor — detect failed jobs and create/update GitHub issues.

Triggered by workflow_run completion. Inspects job-level conclusions (not
run-level, since continue-on-error masks failures) and manages issues:
- Creates a new issue if no open issue exists for the failed job
- Appends a comment to the existing open issue if one already exists
- Multiple failed jobs from the same run are consolidated in one comment
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

EXCLUDED_JOBS = frozenset(
    {
        "check-changes",
        "pr-test-finish",
        "nightly-test-finish",
        "nightly-test-daily-finish",
    }
)

LABEL = "ci-failure"


def gh(*args: str, json_output: bool = False) -> str | dict | list:
    cmd = ["gh"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"gh command failed: {' '.join(cmd)}", file=sys.stderr)
        print(f"stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"gh exited {result.returncode}")
    if json_output:
        return json.loads(result.stdout)
    return result.stdout.strip()


def get_failed_jobs(run_id: str, repo: str) -> list[dict]:
    jobs = gh(
        "api",
        f"repos/{repo}/actions/runs/{run_id}/jobs",
        "--paginate",
        "--jq",
        ".jobs[]",
        json_output=True,
    )
    if isinstance(jobs, dict):
        jobs = [jobs]

    failed = []
    for job in jobs:
        name = job.get("name", "")
        conclusion = job.get("conclusion", "")
        if name in EXCLUDED_JOBS:
            continue
        if conclusion in ("failure", "cancelled", "timed_out"):
            failed.append(
                {
                    "name": name,
                    "conclusion": conclusion,
                    "html_url": job.get("html_url", ""),
                    "started_at": job.get("started_at", ""),
                    "completed_at": job.get("completed_at", ""),
                    "runner_name": job.get("runner_name", "unknown"),
                }
            )
    return failed


def find_open_issue(repo: str, job_name: str) -> dict | None:
    issues = gh(
        "issue",
        "list",
        "--repo",
        repo,
        "--label",
        LABEL,
        "--state",
        "open",
        "--search",
        f"in:title [CI Failure] {job_name}",
        "--json",
        "number,title",
        "--limit",
        "5",
        json_output=True,
    )
    if not issues:
        return None
    expected_title = f"[CI Failure] {job_name}"
    for issue in issues:
        if issue["title"] == expected_title:
            return issue
    return None


def create_issue(repo: str, job_name: str, body: str) -> int:
    title = f"[CI Failure] {job_name}"
    result = gh(
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        title,
        "--body",
        body,
        "--label",
        LABEL,
    )
    print(f"Created issue: {result}")
    for part in result.split("/"):
        if part.isdigit():
            return int(part)
    return 0


def add_comment(repo: str, issue_number: int, body: str) -> None:
    gh(
        "issue",
        "comment",
        "--repo",
        repo,
        str(issue_number),
        "--body",
        body,
    )
    print(f"Added comment to issue #{issue_number}")


def build_issue_body(
    job_name: str,
    workflow_name: str,
    run_id: str,
    run_url: str,
    conclusion: str,
    job_url: str,
    runner_name: str,
    commit_sha: str,
    commit_author: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""## CI Failure: {job_name}

| Field | Value |
|-------|-------|
| **Workflow** | {workflow_name} |
| **Job** | `{job_name}` |
| **Conclusion** | `{conclusion}` |
| **Runner** | `{runner_name}` |
| **Run** | [#{run_id}]({run_url}) |
| **Job logs** | [View logs]({job_url}) |
| **Commit** | `{commit_sha[:8]}` by {commit_author} |
| **Detected at** | {now} |

This issue was auto-created by the CI Failure Monitor.
"""


def build_comment_body(
    workflow_name: str,
    run_id: str,
    run_url: str,
    failed_jobs: list[dict],
    commit_sha: str,
    commit_author: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"### Failure recurrence — {now}\n"]
    lines.append(
        f"**Workflow:** {workflow_name} | **Run:** [#{run_id}]({run_url}) | **Commit:** `{commit_sha[:8]}` by {commit_author}\n"
    )

    if len(failed_jobs) == 1:
        j = failed_jobs[0]
        lines.append(
            f"- **{j['name']}** — `{j['conclusion']}` on `{j['runner_name']}` ([logs]({j['html_url']}))"
        )
    else:
        for j in failed_jobs:
            lines.append(
                f"- **{j['name']}** — `{j['conclusion']}` on `{j['runner_name']}` ([logs]({j['html_url']}))"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CI Failure Monitor")
    parser.add_argument("--run-id", required=True, help="GitHub Actions run ID")
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--workflow-name", required=True, help="Workflow name")
    parser.add_argument("--run-url", required=True, help="Workflow run URL")
    parser.add_argument("--commit-sha", required=True, help="Commit SHA")
    parser.add_argument("--commit-author", default="unknown", help="Commit author")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    print(f"Checking run {args.run_id} for failed jobs...")
    failed_jobs = get_failed_jobs(args.run_id, args.repo)

    if not failed_jobs:
        print("No failed jobs detected. All clear.")
        return

    print(f"Found {len(failed_jobs)} failed job(s):")
    for j in failed_jobs:
        print(f"  - {j['name']}: {j['conclusion']}")

    jobs_by_name: dict[str, list[dict]] = {}
    for j in failed_jobs:
        jobs_by_name.setdefault(j["name"], []).append(j)

    for job_name, jobs in jobs_by_name.items():
        existing = find_open_issue(args.repo, job_name)

        if existing:
            comment = build_comment_body(
                args.workflow_name,
                args.run_id,
                args.run_url,
                jobs,
                args.commit_sha,
                args.commit_author,
            )
            if args.dry_run:
                print(f"[DRY RUN] Would comment on issue #{existing['number']}:\n{comment}\n")
            else:
                add_comment(args.repo, existing["number"], comment)
        else:
            j = jobs[0]
            body = build_issue_body(
                job_name,
                args.workflow_name,
                args.run_id,
                args.run_url,
                j["conclusion"],
                j["html_url"],
                j["runner_name"],
                args.commit_sha,
                args.commit_author,
            )
            if args.dry_run:
                print(f"[DRY RUN] Would create issue for {job_name}:\n{body}\n")
            else:
                create_issue(args.repo, job_name, body)


if __name__ == "__main__":
    main()
