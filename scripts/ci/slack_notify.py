"""Check nightly CI run for job failures, classify failure types, and generate Slack notification."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from ci_common import (
    escape_mrkdwn,
    issue_for_job,
    load_ai_analysis,
    load_failure_issues,
)
from failure_classifier import (
    FAILURE_TYPES,
    classify_jobs,
    get_failed_jobs,
    is_finish_gate,
)
from github_output import write_github_output


def format_slack_summary(
    run_url, commit_sha, author, failed_jobs, analysis=None, failure_issues=None
):
    """Compact Slack summary: one bullet per job with its issue link and AI root cause."""
    analysis = analysis or {}
    failure_issues = failure_issues or {}
    display_jobs = [j for j in failed_jobs if not is_finish_gate(j["name"])]
    short_sha = commit_sha[:7] if len(commit_sha) >= 7 else commit_sha
    job_word = "job" if len(display_jobs) == 1 else "jobs"
    lines = [
        f":red_circle: *Nightly CI Failure* — {len(display_jobs)} {job_word} failed  |  <{run_url}|View run>",
        f"`{short_sha}` by {escape_mrkdwn(author)}",
        "",
    ]
    for job in display_jobs:
        ft = FAILURE_TYPES.get(job["failure_type"], FAILURE_TYPES["bug"])
        safe_name = escape_mrkdwn(job["name"]).replace("|", "/")
        job_url = job.get("html_url")
        job_ref = f"<{job_url}|{safe_name}>" if job_url else safe_name
        issue = issue_for_job(job, failure_issues)
        issue_text = ""
        if issue and issue.get("issue_number") and issue.get("issue_url"):
            issue_text = f" — Issue: <{issue['issue_url']}|#{issue['issue_number']}>"
        lines.append(f"• {ft['emoji']} {job_ref} [{ft['label']}]{issue_text}")
        a = analysis.get(job["name"])
        if a and a.get("root_cause"):
            lines.append(f"      ↳ {escape_mrkdwn(a['root_cause'])}")

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
    parser.add_argument(
        "--from-classification",
        type=str,
        help="Regenerate summary from saved classification JSON (skip API calls)",
    )
    parser.add_argument(
        "--ai-analysis",
        type=str,
        help="ai_analysis.json from the AI analysis step",
    )
    parser.add_argument(
        "--failure-issues",
        type=str,
        help="failure_issues.json from ci_failure_issues.py",
    )
    args = parser.parse_args()

    analysis = load_ai_analysis(args.ai_analysis)
    failure_issue_map = load_failure_issues(args.failure_issues)

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
            write_github_output("has_bugs", "false")
            return

        print(f"Failed jobs: {', '.join(j['name'] for j in failed_jobs)}")
        print("Classifying failures...")
        classify_jobs(args.repo, failed_jobs)

        needs_ai_jobs = [
            j for j in failed_jobs if j["failure_type"] in ("bug", "resource_exhaustion")
        ]
        write_github_output("has_bugs", "true" if needs_ai_jobs else "false")

        classification_data = {
            "failed_jobs": [
                {
                    "name": j["name"],
                    "id": j["id"],
                    "html_url": j["html_url"],
                    "conclusion": j.get("conclusion", "failure"),
                    "failure_type": j["failure_type"],
                }
                for j in failed_jobs
            ],
            "has_bugs": bool(needs_ai_jobs),
            "bug_job_names": [j["name"] for j in needs_ai_jobs],
        }
        with open("classification.json", "w") as f:
            json.dump(classification_data, f, indent=2)
        print("Classification written to classification.json")

    summary = format_slack_summary(
        args.run_url,
        args.commit_sha,
        args.commit_author,
        failed_jobs,
        analysis,
        failure_issue_map,
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
                issue = issue_for_job(job, failure_issue_map)
                issue_text = ""
                if issue and issue.get("issue_number") and issue.get("issue_url"):
                    issue_text = f" — Issue: [#{issue['issue_number']}]({issue['issue_url']})"
                f.write(
                    f"- [{job['name']}]({job['html_url']}) — {job['failure_type']}{issue_text}\n"
                )
                a = analysis.get(job["name"])
                if a and a.get("root_cause"):
                    f.write(f"  - {a['root_cause']}\n")
        print("Report appended to GITHUB_STEP_SUMMARY.")


if __name__ == "__main__":
    main()
