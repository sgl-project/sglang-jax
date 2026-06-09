"""Check CI Auto Bisect result and generate Slack notification for code regressions."""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from ci_common import clamp
from ci_common import escape_mrkdwn as _escape_mrkdwn
from failure_classifier import is_finish_gate
from github_output import write_github_output


def _run_gh(args):
    """Run a gh CLI command, return stdout."""
    return subprocess.run(
        ["gh"] + args, capture_output=True, text=True, check=True, timeout=60
    ).stdout


def format_regression_summary(
    short_sha,
    run_url,
    failed_jobs,
    root_cause,
    suggested_fix,
    pr_url=None,
    pr_number=None,
    branch_unverified=False,
):
    """Slack summary for a post-merge regression.

    ``root_cause`` / ``suggested_fix`` are short bullet lines from the bisect
    agent; each renders under its own header. The full detail is in the PR comment.
    """
    pr_link = ""
    if pr_url and pr_number:
        pr_link = f"  |  <{pr_url}|PR #{pr_number}>"

    header = ":rotating_light: *Post-merge Regression Detected*"
    if branch_unverified:
        header += "  :warning: _branch unverified_"

    commit_part = f"`{short_sha}`" if short_sha else "_unknown commit_"
    lines = [
        header,
        f"{commit_part}{pr_link}  |  <{run_url}|View run>",
        "",
        f"*Failed jobs:* {_escape_mrkdwn(failed_jobs)}",
    ]
    if root_cause.strip():
        lines += ["", "*Root cause*", _escape_mrkdwn(clamp(root_cause))]
    if suggested_fix.strip():
        lines += ["", "*Suggested fix*", _escape_mrkdwn(clamp(suggested_fix))]

    text = "\n".join(lines)
    if len(text) > 2900:
        text = text[:2900].rsplit("\n", 1)[0] + "\n… (truncated)"
    return text


def main():
    parser = argparse.ArgumentParser(description="Gate and format regression Slack notification")
    parser.add_argument("--analysis", required=True, help="Path to analysis_result.json")
    parser.add_argument("--repo", required=True, help="GitHub repository (owner/name)")
    parser.add_argument("--slack-output", required=True, help="File to write Slack message text")
    args = parser.parse_args()

    with open(args.analysis) as f:
        result = json.load(f)

    classification = result.get("classification", "")
    if classification != "code_regression":
        print(f"Classification is '{classification}', not code_regression. Skipping.")
        write_github_output("send_slack", "false")
        return

    original_run_id = result.get("run_id", "")
    try:
        run_meta = json.loads(
            _run_gh(
                [
                    "api",
                    f"repos/{args.repo}/actions/runs/{original_run_id}",
                    "--jq",
                    "{branch: .head_branch, sha: .head_sha}",
                ]
            )
        )
    except Exception as e:
        print(f"Failed to query original run {original_run_id}: {e}")
        print("Sending degraded notification with branch-unverified warning.")
        run_meta = {"branch": "unknown", "sha": ""}

    head_branch = run_meta["branch"]
    branch_unverified = head_branch == "unknown"
    if not branch_unverified and head_branch != "main":
        print(f"Original run was on branch '{head_branch}', not main. Skipping.")
        write_github_output("send_slack", "false")
        return

    head_sha = run_meta["sha"]
    short_sha = head_sha[:7] if head_sha else ""
    run_url = result.get("run_url", "")
    all_failed = result.get("failed_jobs", [])
    # Drop matrix *-finish aggregation gates (noise); keep the raw list if that
    # would leave nothing to show.
    shown_jobs = [j for j in all_failed if not is_finish_gate(j)] or all_failed
    failed_jobs = ", ".join(shown_jobs)
    root_cause = result.get("root_cause", "")
    suggested_fix = result.get("suggested_fix", "")

    pr_url = None
    pr_number = None
    if head_sha:
        try:
            pr_json = json.loads(
                _run_gh(
                    [
                        "api",
                        f"repos/{args.repo}/commits/{head_sha}/pulls",
                        "--jq",
                        ".[0] | {number, html_url}",
                    ]
                )
            )
            pr_number = pr_json.get("number")
            pr_url = pr_json.get("html_url")
        except Exception:
            pass

    summary = format_regression_summary(
        short_sha,
        run_url,
        failed_jobs,
        root_cause,
        suggested_fix,
        pr_url,
        pr_number,
        branch_unverified=branch_unverified,
    )

    with open(args.slack_output, "w") as f:
        f.write(summary)
    print(f"Slack summary written to {args.slack_output}")
    write_github_output("send_slack", "true")


if __name__ == "__main__":
    main()
