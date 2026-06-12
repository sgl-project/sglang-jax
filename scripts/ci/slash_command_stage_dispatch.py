"""Dispatch logic for the /rerun-stage slash command.

Reads STAGE_SUITES_JSON (list of {suite, runner} dicts), HEAD_REF,
PR_NUMBER from env vars and dispatches each suite independently via
rerun-test.yml.
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
    stage = os.environ.get("STAGE", "")
    raw_json = os.environ.get("STAGE_SUITES_JSON", "")

    if not raw_json:
        print("STAGE_SUITES_JSON not set", file=sys.stderr)
        _comment(
            pr_number,
            f"> **Slash command**: Failed to dispatch stage `{stage}` — missing suite data",
        )
        sys.exit(1)

    suites = json.loads(raw_json)
    dispatched = []
    failed = []

    for entry in suites:
        suite_name = entry["suite"]
        runner = entry["runner"]
        result = subprocess.run(
            [
                "gh",
                "workflow",
                "run",
                "rerun-test.yml",
                "--ref",
                "main",
                "-f",
                f"suite={suite_name}",
                "-f",
                f"runner={runner}",
                "-f",
                f"ref={head_ref}",
                "-f",
                f"pr_number={pr_number}",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            dispatched.append(suite_name)
            print(f"Dispatched {suite_name} on {runner}")
        else:
            failed.append(suite_name)
            print(
                f"Failed to dispatch {suite_name}: {result.stderr.strip()}",
                file=sys.stderr,
            )

    lines = []
    if dispatched:
        suite_list = ", ".join(f"`{s}`" for s in dispatched)
        lines.append(f"Dispatched {suite_list} for branch `{head_ref}`")
    if failed:
        fail_list = ", ".join(f"`{s}`" for s in failed)
        lines.append(f"Failed to dispatch {fail_list}")

    body = "; ".join(lines)
    _comment(pr_number, f"> **Slash command** (`{stage}`): {body}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
