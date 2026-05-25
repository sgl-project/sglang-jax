"""Preflight checks for CI auto-bisect.

Resolves RUN_ID, validates eligibility (allowlist + failure status),
and short-circuits before the expensive agent call when there is
nothing to analyze.

Reads from environment variables (set by the YAML env: block):
  RUN_ID          — workflow run ID (may be empty)
  ISSUE_NUMBER    — PR / issue number (may be empty)
  REPO            — owner/repo string

Writes to $GITHUB_OUTPUT:
  eligible        — true | false
  run_id          — resolved run ID (or empty)
  run_url         — HTML URL of the run (or "none")
  issue_number    — passthrough or resolved
  classification  — only when eligible=false: "not_applicable"
  confidence      — only when eligible=false: "high"
  skip_reason     — only when eligible=false: human-readable explanation
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

ELIGIBLE_WORKFLOWS = frozenset(["PR Test", "Nightly Test", "Nightly Test Daily", "TPU Multi Test"])


def run_gh(args: List[str]) -> str:
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gh {' '.join(args)} failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def validate_run(repo: str, run_id: str) -> Dict:
    """Fetch run metadata and return {name, conclusion, html_url, head_sha}."""
    raw = run_gh(
        [
            "api",
            f"repos/{repo}/actions/runs/{run_id}",
            "--jq",
            "{name: .name, conclusion: .conclusion, html_url: .html_url, head_sha: .head_sha}",
        ]
    )
    return json.loads(raw)


def find_eligible_run(repo: str, sha: str) -> Tuple[Optional[Dict], List[str]]:
    """Find the most recent eligible failed run on *sha*.

    Returns (run_info_or_None, in_progress_workflow_names).
    run_info keys: id, name, conclusion, html_url, head_sha.
    """
    raw = run_gh(
        [
            "api",
            f"repos/{repo}/actions/runs?head_sha={sha}&per_page=50",
            "--jq",
            ".workflow_runs | map({id: (.id|tostring), name, conclusion, status, html_url, head_sha, created_at})",
        ]
    )
    all_runs = json.loads(raw) if raw else []

    eligible = [r for r in all_runs if r["name"] in ELIGIBLE_WORKFLOWS]

    failed = [r for r in eligible if r["conclusion"] == "failure"]
    if failed:
        failed.sort(key=lambda r: r["created_at"], reverse=True)
        best = failed[0]
        return (
            {
                "id": best["id"],
                "name": best["name"],
                "conclusion": best["conclusion"],
                "html_url": best["html_url"],
                "head_sha": best["head_sha"],
            },
            [],
        )

    in_progress = [
        r["name"]
        for r in eligible
        if r["status"] in ("in_progress", "queued", "waiting", "pending")
    ]
    return None, in_progress


def write_outputs(outputs: Dict) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as fh:
            for name, value in outputs.items():
                fh.write(f"{name}={value}\n")
    else:
        print("(GITHUB_OUTPUT not set — printing only)", file=sys.stderr)
    for name, value in outputs.items():
        print(f"  {name}={value}")


def write_skip_result(
    reason: str,
    *,
    run_id: str = "",
    run_url: str = "none",
    issue_number: str = "",
) -> None:
    """Write analysis_result.json, GITHUB_STEP_SUMMARY, and outputs for a skip."""
    result = {
        "run_id": run_id,
        "run_url": run_url,
        "classification": "not_applicable",
        "confidence": "high",
        "failed_jobs": [],
        "root_cause": reason,
        "evidence": "Determined by bisect_preflight.py (no agent invocation needed).",
        "suggested_fix": "none",
    }
    with open("analysis_result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"Wrote analysis_result.json: classification=not_applicable")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"## CI Auto Bisect — skipped\n\n")
            fh.write(f"**Classification:** not_applicable\n\n")
            fh.write(f"**Reason:** {reason}\n")

    write_outputs(
        {
            "eligible": "false",
            "run_id": run_id,
            "run_url": run_url,
            "issue_number": issue_number,
            "classification": "not_applicable",
            "confidence": "high",
            "skip_reason": reason,
        }
    )


def main() -> int:
    run_id = os.environ.get("RUN_ID", "").strip()
    issue_number = os.environ.get("ISSUE_NUMBER", "").strip()
    repo = os.environ.get("REPO", "").strip()

    if not repo:
        print("::error::REPO environment variable is required", file=sys.stderr)
        return 1

    # --- Path 1: explicit RUN_ID provided ---
    if run_id:
        print(f"Validating provided RUN_ID={run_id}")
        try:
            info = validate_run(repo, run_id)
        except RuntimeError as exc:
            write_skip_result(
                f"Run {run_id} not found or API error: {exc}",
                run_id=run_id,
                issue_number=issue_number,
            )
            return 0

        if info["name"] not in ELIGIBLE_WORKFLOWS:
            write_skip_result(
                f"Workflow \"{info['name']}\" is not in the auto-bisect allowlist "
                f"({', '.join(sorted(ELIGIBLE_WORKFLOWS))}). Only allowlisted workflow "
                f"failures are analyzed.",
                run_id=run_id,
                run_url=info.get("html_url", "none"),
                issue_number=issue_number,
            )
            return 0

        if info["conclusion"] != "failure":
            write_skip_result(
                f"Run {run_id} ({info['name']}) has conclusion=\"{info['conclusion']}\", "
                f'not "failure". Nothing to analyze.',
                run_id=run_id,
                run_url=info.get("html_url", "none"),
                issue_number=issue_number,
            )
            return 0

        print(f"Run {run_id} is eligible: {info['name']} / failure")
        write_outputs(
            {
                "eligible": "true",
                "run_id": run_id,
                "run_url": info.get("html_url", "none"),
                "issue_number": issue_number,
            }
        )
        return 0

    # --- Path 2: no RUN_ID — auto-detect from PR ---
    if not issue_number:
        write_skip_result("No RUN_ID or ISSUE_NUMBER provided.")
        return 0

    print(f"No RUN_ID provided. Auto-detecting from PR #{issue_number}")
    try:
        sha = run_gh(["api", f"repos/{repo}/pulls/{issue_number}", "--jq", ".head.sha"])
    except RuntimeError as exc:
        write_skip_result(
            f"Could not resolve HEAD SHA for PR #{issue_number}: {exc}",
            issue_number=issue_number,
        )
        return 0

    if not sha:
        write_skip_result(
            f"PR #{issue_number} returned an empty HEAD SHA.",
            issue_number=issue_number,
        )
        return 0

    print(f"PR #{issue_number} HEAD SHA: {sha}")
    try:
        found, in_progress = find_eligible_run(repo, sha)
    except RuntimeError as exc:
        write_skip_result(
            f"Failed to list runs for SHA {sha}: {exc}",
            issue_number=issue_number,
        )
        return 0

    if found:
        print(f"Found eligible run: {found['id']} ({found['name']})")
        write_outputs(
            {
                "eligible": "true",
                "run_id": found["id"],
                "run_url": found["html_url"],
                "issue_number": issue_number,
            }
        )
        return 0

    if in_progress:
        names = ", ".join(sorted(set(in_progress)))
        write_skip_result(
            f"No failed runs found for PR #{issue_number} (HEAD {sha[:12]}), "
            f"but these allowlisted workflows are still running: {names}. "
            f"Re-run auto-bisect after they complete.",
            issue_number=issue_number,
        )
        return 0

    write_skip_result(
        f"No eligible failed runs found for PR #{issue_number} (HEAD {sha[:12]}). "
        f"Only failures in {', '.join(sorted(ELIGIBLE_WORKFLOWS))} are analyzed.",
        issue_number=issue_number,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
