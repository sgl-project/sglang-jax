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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from failure_classifier import classify_run

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


def check_draft_pr(repo: str, head_sha: str, issue_number: str) -> Optional[str]:
    """Return a skip reason if the associated PR is a draft, else None."""
    if issue_number:
        try:
            draft = run_gh(["api", f"repos/{repo}/pulls/{issue_number}", "--jq", ".draft"])
            if draft == "true":
                return f"PR #{issue_number} is a draft. Auto-bisect skipped for draft PRs."
        except RuntimeError:
            pass
        return None

    if not head_sha:
        return None

    try:
        raw = run_gh(
            [
                "api",
                f"repos/{repo}/commits/{head_sha}/pulls",
                "--jq",
                '[.[] | select(.state=="open")] | first | {number, draft}',
            ]
        )
        if raw:
            pr_info = json.loads(raw)
            if pr_info.get("draft"):
                return f"PR #{pr_info['number']} is a draft. Auto-bisect skipped for draft PRs."
    except (RuntimeError, json.JSONDecodeError):
        pass
    return None


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
    skip_type: str = "no_eligible",
    run_id: str = "",
    run_url: str = "none",
    issue_number: str = "",
) -> None:
    """Write analysis_result.json, GITHUB_STEP_SUMMARY, and outputs for a skip.

    skip_type: "no_eligible" (nothing to analyze) or "error" (bad input / API failure).
    """
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
            "skip_type": skip_type,
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
                skip_type="error",
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
        draft_reason = check_draft_pr(repo, info.get("head_sha", ""), issue_number)
        if draft_reason:
            print(draft_reason)
            write_skip_result(
                draft_reason,
                run_id=run_id,
                run_url=info.get("html_url", "none"),
                issue_number=issue_number,
            )
            return 0
        if _try_preclassify_and_skip(repo, run_id, info.get("html_url", "none"), issue_number):
            return 0
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
        write_skip_result("No RUN_ID or ISSUE_NUMBER provided.", skip_type="error")
        return 0

    print(f"No RUN_ID provided. Auto-detecting from PR #{issue_number}")
    draft_reason = check_draft_pr(repo, "", issue_number)
    if draft_reason:
        print(draft_reason)
        write_skip_result(draft_reason, issue_number=issue_number)
        return 0
    try:
        sha = run_gh(["api", f"repos/{repo}/pulls/{issue_number}", "--jq", ".head.sha"])
    except RuntimeError as exc:
        write_skip_result(
            f"Could not resolve HEAD SHA for PR #{issue_number}: {exc}",
            skip_type="error",
            issue_number=issue_number,
        )
        return 0

    if not sha:
        write_skip_result(
            f"PR #{issue_number} returned an empty HEAD SHA.",
            skip_type="error",
            issue_number=issue_number,
        )
        return 0

    print(f"PR #{issue_number} HEAD SHA: {sha}")
    try:
        found, in_progress = find_eligible_run(repo, sha)
    except RuntimeError as exc:
        write_skip_result(
            f"Failed to list runs for SHA {sha}: {exc}",
            skip_type="error",
            issue_number=issue_number,
        )
        return 0

    if found:
        print(f"Found eligible run: {found['id']} ({found['name']})")
        if _try_preclassify_and_skip(repo, found["id"], found["html_url"], issue_number):
            return 0
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


def _post_classification_comment(
    repo: str, run_id: str, run_url: str, issue_number: str, failed_jobs: list
) -> None:
    """Post a PR comment summarizing pre-classified failures."""
    if not issue_number:
        print("No issue_number — skipping comment")
        return

    marker = f"<!-- ci-auto-bisect:run_id={run_id} -->"
    try:
        existing = run_gh(
            [
                "api",
                f"repos/{repo}/issues/{issue_number}/comments",
                "--paginate",
                "--jq",
                ".[].body",
            ]
        )
        if marker in existing:
            print(f"Comment for run {run_id} already exists — skipping")
            return
    except RuntimeError:
        return

    github_emoji = {
        "timeout": "⏳",
        "infrastructure": "☁️",
        "resource_exhaustion": "\U0001f4a5",
        "bug": "\U0001fab2",
    }
    lines = [marker]
    lines.append(f"## CI Auto Bisect — run {run_id}\n")
    lines.append(f"**[View run]({run_url})**\n")
    lines.append("All failures classified as non-code issues (AI analysis skipped):\n")
    for job in failed_jobs:
        emoji = github_emoji.get(job["failure_type"], "")
        lines.append(
            f"- {emoji} **{job['name']}** — {job['failure_type']}" f" ([view]({job['html_url']}))"
        )
    lines.append("\n_Re-run the workflow if the issue was transient._")

    body = "\n".join(lines)
    try:
        run_gh(["issue", "comment", issue_number, "--repo", repo, "--body", body])
        print(f"Posted classification comment on #{issue_number}")
    except RuntimeError as e:
        print(f"Warning: failed to post comment: {e}")


def _try_preclassify_and_skip(repo: str, run_id: str, run_url: str, issue_number: str) -> bool:
    """Attempt pre-classification. If all failures are non-bug, write skip and return True."""
    print("Pre-classifying failed jobs...")
    try:
        failed_jobs, needs_ai = classify_run(repo, run_id)
    except Exception as exc:
        print(f"Warning: pre-classification failed ({exc}), proceeding to AI analysis")
        return False
    if not failed_jobs or needs_ai:
        return False
    _post_classification_comment(repo, run_id, run_url, issue_number, failed_jobs)
    type_summary = ", ".join(f"{j['name']} ({j['failure_type']})" for j in failed_jobs)
    skip_reason = (
        f"All {len(failed_jobs)} failed job(s) are non-bug failures "
        f"(pre-classified by deterministic pattern matching): {type_summary}. "
        f"AI analysis skipped."
    )
    print(f"Skipping AI: {skip_reason}")
    write_skip_result(
        skip_reason,
        skip_type="no_eligible",
        run_id=run_id,
        run_url=run_url,
        issue_number=issue_number,
    )
    return True


if __name__ == "__main__":
    sys.exit(main())
