"""Check nightly CI run for job failures and send Slack notification."""

import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def get_failed_jobs(repo, run_id, token):
    """Query GitHub API for jobs with conclusion=failure in the given run."""
    failed = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100&page={page}"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        req = Request(url, headers=headers)
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for job in data.get("jobs", []):
            if job.get("conclusion") == "failure":
                failed.append(job["name"])
        if len(data.get("jobs", [])) < 100:
            break
        page += 1
    return failed


def send_slack_notification(webhook_url, workflow_name, run_url, commit_sha, author, failed_jobs):
    """Send a Slack Block Kit notification for failed nightly jobs."""
    short_sha = commit_sha[:7] if len(commit_sha) >= 7 else commit_sha
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": ":red_circle: Nightly CI Failure"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Workflow:*\n{workflow_name}"},
                    {"type": "mrkdwn", "text": f"*Commit:*\n{short_sha}"},
                    {"type": "mrkdwn", "text": f"*Author:*\n{author}"},
                    {"type": "mrkdwn", "text": f"*Run:*\n<{run_url}|View failed run>"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Failed jobs:*\n{', '.join(failed_jobs)}"},
            },
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        webhook_url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(req) as resp:
            pass
    except HTTPError as e:
        print(f"Slack webhook failed: {e.code} {e.reason}")
        sys.exit(1)


def main():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GH_TOKEN or GITHUB_TOKEN must be set")
        sys.exit(1)

    repo = os.environ.get("REPO")
    run_id = os.environ.get("RUN_ID")
    if not repo or not run_id:
        print("Error: REPO and RUN_ID must be set")
        sys.exit(1)

    try:
        failed_jobs = get_failed_jobs(repo, run_id, token)
    except Exception as e:
        print(f"Error querying jobs for run {run_id}: {e}")
        sys.exit(1)

    if not failed_jobs:
        print("No failed jobs found — skipping notification")
        return

    print(f"Failed jobs: {', '.join(failed_jobs)}")

    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("::warning::SLACK_WEBHOOK_URL not configured — skipping Slack notification")
        return

    workflow_name = os.environ.get("WORKFLOW_NAME", "Unknown")
    run_url = os.environ.get("RUN_URL", "")
    commit_sha = os.environ.get("COMMIT_SHA", "N/A")
    commit_author = os.environ.get("COMMIT_AUTHOR", "unknown")

    send_slack_notification(
        webhook_url, workflow_name, run_url, commit_sha, commit_author, failed_jobs
    )
    print("Slack notification sent")


if __name__ == "__main__":
    main()
