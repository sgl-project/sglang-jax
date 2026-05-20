#!/usr/bin/env python3
"""
CI Auto Bisect - AI-Assisted Root Cause Analysis

Analyzes failed CI runs using Claude API to identify root causes
and post analysis results.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass

import anthropic
import requests

GITHUB_API = "https://api.github.com"
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_LOG_CHARS = 50000  # per job, to stay within context


@dataclass
class AnalysisResult:
    run_id: int
    run_url: str
    classification: str  # code_regression, flaky_test, infrastructure, environment
    confidence: str  # high, medium, low
    failed_jobs: list
    root_cause: str
    suggested_fix: str
    raw_response: str = ""


def gh_get(url, token, params=None):
    """Make authenticated GitHub API GET request."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def gh_post(url, token, data):
    """Make authenticated GitHub API POST request."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    resp = requests.post(url, headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_run_info(repo, run_id, token):
    """Fetch workflow run information."""
    url = f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}"
    return gh_get(url, token)


def fetch_failed_jobs(repo, run_id, token):
    """Fetch all jobs for a run, return only failed ones."""
    jobs = []
    page = 1
    while True:
        url = f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}/jobs"
        data = gh_get(url, token, {"per_page": 100, "page": page, "filter": "latest"})
        jobs.extend(data.get("jobs", []))
        if len(data.get("jobs", [])) < 100:
            break
        page += 1
    return [j for j in jobs if j.get("conclusion") == "failure"]


def fetch_job_log_direct(repo, job_id, token, max_chars=MAX_LOG_CHARS):
    """Fetch logs for a specific job, truncated to last max_chars."""
    url = f"{GITHUB_API}/repos/{repo}/actions/jobs/{job_id}/logs"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
        if resp.status_code == 200:
            text = resp.text
            return text[-max_chars:] if len(text) > max_chars else text
        print(f"Warning: Log fetch for job {job_id} returned HTTP {resp.status_code}")
    except requests.RequestException as e:
        print(f"Warning: Failed to fetch logs for job {job_id}: {e}")
    return ""


def fetch_commit_diff(repo, sha, token):
    """Fetch the diff for a commit."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff",
    }
    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{repo}/commits/{sha}",
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 200:
            diff = resp.text
            if len(diff) > 30000:
                diff = diff[:30000] + "\n... (truncated)"
            return diff
    except requests.RequestException as e:
        print(f"Warning: Failed to fetch commit diff for {sha[:12]}: {e}")
    return ""


def analyze_with_claude(failed_jobs_info, commit_diff, run_info, api_key):
    """Call Claude API to analyze the failure."""
    client = anthropic.Anthropic(api_key=api_key)

    prompt_parts = [
        "You are a CI failure analyst for sglang-jax, a JAX-based LLM inference engine running on Google TPU.",
        "",
        "Analyze the following CI failure and classify it.",
        "",
        "## Workflow Run",
        f"- Run ID: {run_info.get('id')}",
        f"- Branch: {run_info.get('head_branch')}",
        f"- Commit: {run_info.get('head_sha', '')[:12]}",
        f"- Event: {run_info.get('event')}",
        f"- URL: {run_info.get('html_url')}",
        "",
    ]

    for job_info in failed_jobs_info:
        prompt_parts.extend(
            [
                f"## Failed Job: {job_info['name']}",
                f"### Logs (last {MAX_LOG_CHARS} chars):",
                "```",
                job_info.get("logs", "(no logs available)"),
                "```",
                "",
            ]
        )

    if commit_diff:
        prompt_parts.extend(
            [
                "## Commit Diff:",
                "```diff",
                commit_diff,
                "```",
                "",
            ]
        )

    prompt_parts.extend(
        [
            "## Instructions",
            "Classify this failure as exactly ONE of:",
            "- `code_regression`: A code change broke something",
            "- `flaky_test`: The test is intermittently failing",
            "- `infrastructure`: Hardware/runner/network issue",
            "- `environment`: Dependency/config/environment change",
            "",
            "Respond in this exact JSON format:",
            "```json",
            "{",
            '  "classification": "one of the above",',
            '  "confidence": "high/medium/low",',
            '  "root_cause": "Clear explanation of what went wrong",',
            '  "evidence": "What in the logs/diff supports this conclusion",',
            '  "suggested_fix": "Actionable recommendation"',
            "}",
            "```",
        ]
    )

    prompt = "\n".join(prompt_parts)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                raise
        except Exception as e:
            print(f"Claude API error (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                raise

    return ""


def parse_claude_response(raw):
    """Extract JSON from Claude's response."""
    json_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    # Try parsing the whole response as JSON
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "classification": "unknown",
            "confidence": "low",
            "root_cause": raw[:500],
            "evidence": "",
            "suggested_fix": "",
        }


def format_markdown_report(result, parsed):
    """Format analysis result as markdown."""
    classification_emoji = {
        "code_regression": "🔴",
        "flaky_test": "🟡",
        "infrastructure": "🟠",
        "environment": "🔵",
        "unknown": "⚪",
    }
    emoji = classification_emoji.get(parsed.get("classification", ""), "⚪")

    lines = [
        "# CI Failure Analysis",
        "",
        f"**Run:** [{result.run_id}]({result.run_url})",
        f"**Classification:** {emoji} `{parsed.get('classification', 'unknown')}`",
        f"**Confidence:** {parsed.get('confidence', 'unknown')}",
        "",
        "## Failed Jobs",
        "",
    ]
    for job in result.failed_jobs:
        lines.append(f"- {job}")
    lines.extend(
        [
            "",
            "## Root Cause",
            "",
            parsed.get("root_cause", "Unable to determine"),
            "",
            "## Evidence",
            "",
            parsed.get("evidence", "N/A"),
            "",
            "## Suggested Fix",
            "",
            parsed.get("suggested_fix", "N/A"),
            "",
            "---",
            "*Analysis generated by CI Auto Bisect using Claude API*",
        ]
    )
    return "\n".join(lines)


def post_issue_comment(repo, issue_number, body, token):
    """Post a comment on a GitHub issue."""
    url = f"{GITHUB_API}/repos/{repo}/issues/{issue_number}/comments"
    gh_post(url, token, {"body": body})
    print(f"Posted analysis comment on issue #{issue_number}")


def main():
    parser = argparse.ArgumentParser(description="AI-assisted CI failure analysis")
    parser.add_argument("--repo", default="sgl-project/sglang-jax", help="GitHub repo")
    parser.add_argument("--run-id", required=True, help="Failed workflow run ID")
    parser.add_argument("--issue-number", type=int, help="Issue number to comment on")
    parser.add_argument("--output", default="analysis_result.json", help="Output file")
    args = parser.parse_args()

    github_token = os.environ.get("GITHUB_TOKEN")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not github_token:
        print("Error: GITHUB_TOKEN not set")
        sys.exit(1)
    if not anthropic_key:
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # 1. Fetch run info
    print(f"Fetching run {args.run_id}...")
    run_info = fetch_run_info(args.repo, args.run_id, github_token)

    # 2. Get failed jobs
    print("Fetching failed jobs...")
    failed_jobs = fetch_failed_jobs(args.repo, args.run_id, github_token)
    if not failed_jobs:
        print("No failed jobs found in this run.")
        sys.exit(0)
    print(f"Found {len(failed_jobs)} failed jobs")

    # 3. Fetch logs for failed jobs
    print("Fetching job logs...")
    failed_jobs_info = []
    for job in failed_jobs:
        log = fetch_job_log_direct(args.repo, job["id"], github_token)
        failed_jobs_info.append(
            {
                "name": job["name"],
                "id": job["id"],
                "logs": log,
                "url": job.get("html_url", ""),
            }
        )

    # 4. Fetch commit diff
    sha = run_info.get("head_sha", "")
    print(f"Fetching diff for commit {sha[:12]}...")
    diff = fetch_commit_diff(args.repo, sha, github_token)

    # 5. Analyze with Claude
    print("Analyzing with Claude...")
    raw_response = analyze_with_claude(failed_jobs_info, diff, run_info, anthropic_key)
    parsed = parse_claude_response(raw_response)

    # 6. Build result
    result = AnalysisResult(
        run_id=int(args.run_id),
        run_url=run_info.get("html_url", ""),
        classification=parsed.get("classification", "unknown"),
        confidence=parsed.get("confidence", "low"),
        failed_jobs=[j["name"] for j in failed_jobs_info],
        root_cause=parsed.get("root_cause", ""),
        suggested_fix=parsed.get("suggested_fix", ""),
        raw_response=raw_response,
    )

    # 7. Save result
    with open(args.output, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"Results saved to {args.output}")

    # 8. Format markdown report
    report = format_markdown_report(result, parsed)

    # 9. Post to issue if requested
    if args.issue_number:
        post_issue_comment(args.repo, args.issue_number, report, github_token)

    # 10. Write to GITHUB_STEP_SUMMARY
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(report)

    print("Analysis complete!")
    # Exit with non-zero if it's a confirmed regression, to signal urgency to callers
    if parsed.get("classification") == "code_regression" and parsed.get("confidence") == "high":
        sys.exit(1)


if __name__ == "__main__":
    main()
