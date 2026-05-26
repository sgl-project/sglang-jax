"""Check nightly CI run for job failures, classify failure types, and generate Slack notification."""

import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from failure_classifier import FAILURE_TYPES, REPORTABLE_CONCLUSIONS, classify_failure

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
    """Query GitHub API for jobs with reportable conclusions in the given run."""
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
            if job.get("conclusion") in REPORTABLE_CONCLUSIONS:
                failed.append(
                    {
                        "name": job["name"],
                        "id": job["id"],
                        "html_url": job["html_url"],
                        "conclusion": job["conclusion"],
                    }
                )
        if len(jobs) < 100:
            break
        page += 1
    return failed


def fetch_job_logs(repo, job_id, max_lines=500):
    """Fetch logs for a single job, truncated to last max_lines lines."""
    try:
        output = run_gh_command(
            ["api", f"repos/{repo}/actions/jobs/{job_id}/logs"],
            max_retries=3,
        )
        lines = output.splitlines()
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return "\n".join(lines)
    except Exception as e:
        print(f"Warning: failed to fetch logs for job {job_id}: {e}")
        return ""


def classify_jobs(repo, failed_jobs):
    """Fetch logs and classify each failed job. Mutates jobs in-place."""
    for job in failed_jobs:
        log_text = fetch_job_logs(repo, job["id"])
        job["failure_type"] = classify_failure(log_text, job.get("conclusion"))
        print(f"  {job['name']}: {job['failure_type']}")


def format_slack_summary(workflow_name, run_url, commit_sha, author, failed_jobs, ai_analysis=""):
    """Format Slack mrkdwn summary with failure types and per-job links."""
    short_sha = commit_sha[:7] if len(commit_sha) >= 7 else commit_sha
    lines = [":red_circle: *Nightly CI Failure*"]
    lines.append("")
    lines.append(f"*Workflow:* {workflow_name}")
    lines.append(f"*Commit:* `{short_sha}` by {author}")
    lines.append(f"*Run:* <{run_url}|View failed run>")
    lines.append("")
    lines.append(f"*Failed jobs ({len(failed_jobs)}):*")
    for job in failed_jobs:
        ft = FAILURE_TYPES.get(job["failure_type"], FAILURE_TYPES["bug"])
        safe_name = job["name"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_name = safe_name.replace("|", "/")
        lines.append(f"• {ft['emoji']} <{job['html_url']}|{safe_name}> [{ft['label']}]")

    if ai_analysis:
        lines.append("")
        lines.append("*AI Analysis:*")
        lines.append(ai_analysis)

    text = "\n".join(lines)
    if len(text) > 2900:
        text = text[:2900].rsplit("\n", 1)[0] + "\n… (truncated)"
    return text


def _write_github_output(name, value):
    """Write a name=value pair to GITHUB_OUTPUT."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")
    print(f"  output: {name}={value}")


def main():
    parser = argparse.ArgumentParser(description="Check nightly CI for failures")
    parser.add_argument("--repo", required=True, help="GitHub repository (owner/name)")
    parser.add_argument("--run-id", required=True, help="Workflow run ID to check")
    parser.add_argument("--workflow-name", default="Unknown", help="Name of the workflow")
    parser.add_argument("--run-url", default="", help="URL to the workflow run")
    parser.add_argument("--commit-sha", default="N/A", help="Head commit SHA")
    parser.add_argument("--commit-author", default="unknown", help="Commit author")
    parser.add_argument("--slack-output", type=str, help="Slack summary output file path")
    parser.add_argument(
        "--from-classification",
        type=str,
        help="Regenerate summary from saved classification JSON (skip API calls)",
    )
    parser.add_argument(
        "--ai-analysis",
        type=str,
        help="Path to file containing AI analysis text to include in summary",
    )
    args = parser.parse_args()

    ai_analysis = ""
    if args.ai_analysis and os.path.isfile(args.ai_analysis):
        with open(args.ai_analysis) as f:
            ai_analysis = f.read().strip()

    if args.from_classification:
        with open(args.from_classification) as f:
            data = json.load(f)
        failed_jobs = data["failed_jobs"]
        print(f"Loaded {len(failed_jobs)} jobs from {args.from_classification}")
    else:
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
            _write_github_output("has_bugs", "false")
            return

        print(f"Failed jobs: {', '.join(j['name'] for j in failed_jobs)}")
        print("Classifying failures...")
        classify_jobs(args.repo, failed_jobs)

        bug_jobs = [j for j in failed_jobs if j["failure_type"] == "bug"]
        _write_github_output("has_bugs", "true" if bug_jobs else "false")

        classification_data = {
            "failed_jobs": [
                {
                    "name": j["name"],
                    "id": j["id"],
                    "html_url": j["html_url"],
                    "failure_type": j["failure_type"],
                }
                for j in failed_jobs
            ],
            "has_bugs": bool(bug_jobs),
            "bug_job_names": [j["name"] for j in bug_jobs],
        }
        with open("classification.json", "w") as f:
            json.dump(classification_data, f, indent=2)
        print("Classification written to classification.json")

    summary = format_slack_summary(
        args.workflow_name,
        args.run_url,
        args.commit_sha,
        args.commit_author,
        failed_jobs,
        ai_analysis,
    )

    if args.slack_output:
        with open(args.slack_output, "w") as f:
            f.write(summary)
        print(f"Slack summary written to {args.slack_output}")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write("## Nightly CI Failure\n\n")
            f.write(f"**Workflow:** {args.workflow_name}\n\n")
            f.write(f"**Commit:** {args.commit_sha[:7]} by {args.commit_author}\n\n")
            f.write(f"**Failed jobs ({len(failed_jobs)}):**\n\n")
            for job in failed_jobs:
                f.write(f"- [{job['name']}]({job['html_url']}) — {job['failure_type']}\n")
            if ai_analysis:
                f.write(f"\n**AI Analysis:**\n\n{ai_analysis}\n")
        print("Report appended to GITHUB_STEP_SUMMARY.")


if __name__ == "__main__":
    main()
