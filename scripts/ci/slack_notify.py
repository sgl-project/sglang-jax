"""Check nightly CI run for job failures and generate Slack notification."""

import argparse
import json
import os
import random
import subprocess
import sys
import time

_NON_RETRYABLE_PATTERNS = ("401", "403", "404", "422", "not found")


def run_gh_command(args, max_retries=5):
    """Run a gh CLI command with exponential backoff retry on transient failure."""
    for attempt in range(max_retries):
        try:
            return subprocess.run(
                ["gh"] + args, capture_output=True, text=True, check=True, timeout=120
            ).stdout
        except subprocess.TimeoutExpired:
            if attempt == max_retries - 1:
                raise
            wait = (2**attempt) + random.uniform(0, 1)
            print(f"Retry {attempt + 1}/{max_retries} in {wait:.1f}s: command timed out")
            time.sleep(wait)
        except subprocess.CalledProcessError as e:
            err_lower = str(e.stderr).lower()
            if "rate limit" not in err_lower and any(
                p in err_lower for p in _NON_RETRYABLE_PATTERNS
            ):
                raise
            if attempt == max_retries - 1:
                raise
            wait = (2**attempt) + random.uniform(0, 1)
            print(f"Retry {attempt + 1}/{max_retries} in {wait:.1f}s: {e.stderr.strip()}")
            time.sleep(wait)
    raise RuntimeError("run_gh_command: exhausted retries without raising")


def get_failed_jobs(repo, run_id):
    """Query GitHub API for jobs with conclusion=failure in the given run."""
    failed = []
    page = 1
    while True:
        output = run_gh_command(
            [
                "api",
                f"repos/{repo}/actions/runs/{run_id}/jobs",
                "--method",
                "GET",
                "-f",
                "per_page=100",
                "-f",
                f"page={page}",
            ]
        )
        data = json.loads(output)
        jobs = data.get("jobs", [])
        for job in jobs:
            if job.get("conclusion") == "failure":
                failed.append(job["name"])
        if len(jobs) < 100:
            break
        page += 1
    return failed


def format_slack_summary(workflow_name, run_url, commit_sha, author, failed_jobs):
    """Format a compact Slack mrkdwn summary for nightly failure notification."""
    short_sha = commit_sha[:7] if len(commit_sha) >= 7 else commit_sha
    lines = [":red_circle: *Nightly CI Failure*"]
    lines.append("")
    lines.append(f"*Workflow:* {workflow_name}")
    lines.append(f"*Commit:* `{short_sha}` by {author}")
    lines.append(f"*Run:* <{run_url}|View failed run>")
    lines.append("")
    lines.append(f"*Failed jobs ({len(failed_jobs)}):*")
    for job in failed_jobs:
        lines.append(f"• {job}")
    text = "\n".join(lines)
    if len(text) > 2900:
        text = text[:2900].rsplit("\n", 1)[0] + "\n… (truncated)"
    return text


def main():
    parser = argparse.ArgumentParser(description="Check nightly CI for failures")
    parser.add_argument("--repo", required=True, help="GitHub repository (owner/name)")
    parser.add_argument("--run-id", required=True, help="Workflow run ID to check")
    parser.add_argument("--workflow-name", default="Unknown", help="Name of the workflow")
    parser.add_argument("--run-url", default="", help="URL to the workflow run")
    parser.add_argument("--commit-sha", default="N/A", help="Head commit SHA")
    parser.add_argument("--commit-author", default="unknown", help="Commit author")
    parser.add_argument("--slack-output", type=str, help="Slack summary output file path")
    args = parser.parse_args()

    try:
        failed_jobs = get_failed_jobs(args.repo, args.run_id)
    except Exception as e:
        print(f"Error querying jobs for run {args.run_id}: {e}")
        sys.exit(1)

    if not failed_jobs:
        print("No failed jobs found — skipping notification")
        if args.slack_output:
            with open(args.slack_output, "w") as f:
                f.write("")
        return

    print(f"Failed jobs: {', '.join(failed_jobs)}")

    summary = format_slack_summary(
        args.workflow_name, args.run_url, args.commit_sha, args.commit_author, failed_jobs
    )

    if args.slack_output:
        with open(args.slack_output, "w") as f:
            f.write(summary)
        print(f"Slack summary written to {args.slack_output}")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(f"## Nightly CI Failure\n\n")
            f.write(f"**Workflow:** {args.workflow_name}\n\n")
            f.write(f"**Commit:** {args.commit_sha[:7]} by {args.commit_author}\n\n")
            f.write(f"**Failed jobs ({len(failed_jobs)}):**\n\n")
            for job in failed_jobs:
                f.write(f"- {job}\n")
        print("Report appended to GITHUB_STEP_SUMMARY.")


if __name__ == "__main__":
    main()
