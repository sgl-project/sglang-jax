"""Dispatch logic for the /rerun-failed-ci slash command.

Finds the latest pr-test workflow run for the PR branch and reruns its
failed jobs.  Reads GH_TOKEN, HEAD_REF, PR_NUMBER, REPO from env vars
(set by the YAML env: block) and calls the gh CLI.
"""

import json
import os
import subprocess
import sys


def _comment(pr_number, body):
    subprocess.run(
        ["gh", "pr", "comment", pr_number, "--body", body],
        check=True,
    )


def main():
    head_ref = os.environ.get("HEAD_REF", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    repo = os.environ.get("REPO", "")

    result = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--workflow=pr-test.yml",
            f"--branch={head_ref}",
            "--status=completed",
            "--limit=1",
            "--json",
            "databaseId,conclusion",
            "--jq",
            ".[0]",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(
            f"gh run list failed (exit {result.returncode}): {result.stderr.strip()}",
            file=sys.stderr,
        )
        _comment(
            pr_number,
            f"> **Slash command**: Failed to query runs for branch `{head_ref}` — please retry",
        )
        sys.exit(1)

    raw = result.stdout.strip()
    if not raw or raw == "null":
        _comment(
            pr_number,
            f"> **Slash command**: No completed pr-test run found for branch `{head_ref}`",
        )
        return

    run_info = json.loads(raw)
    run_id = str(run_info["databaseId"])
    conclusion = run_info.get("conclusion", "")

    if conclusion == "success":
        run_url = f"https://github.com/{repo}/actions/runs/{run_id}"
        _comment(
            pr_number,
            f"> **Slash command**: Latest [run #{run_id}]({run_url}) already succeeded — nothing to rerun",
        )
        return

    rerun = subprocess.run(
        ["gh", "run", "rerun", run_id, "--failed"],
        capture_output=True,
        text=True,
    )

    run_url = f"https://github.com/{repo}/actions/runs/{run_id}"

    if rerun.returncode != 0:
        print(f"gh run rerun failed: {rerun.stderr.strip()}", file=sys.stderr)
        _comment(
            pr_number,
            f"> **Slash command**: Failed to rerun [run #{run_id}]({run_url}) — {rerun.stderr.strip()}",
        )
        sys.exit(1)

    _comment(
        pr_number,
        f"> **Slash command**: Re-running failed jobs in [run #{run_id}]({run_url})",
    )

    print(f"Rerun triggered for run {run_id} on branch {head_ref}")


if __name__ == "__main__":
    main()
