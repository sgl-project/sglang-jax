"""Failure classification for CI jobs.

Shared module used by slack_notify.py and bisect_preflight.py.
All failure-related logic lives here: fetching failed jobs, pulling
logs, classifying failure types.

To add a new failure type:
  1. Add a compiled regex to the _*_RE constants below.
  2. Add an entry in FAILURE_TYPES with label and emoji.
  3. Add a branch in classify_failure() at the correct priority level.
"""

import json
import random
import re
import subprocess
import time

FAILURE_TYPES = {
    "timeout": {"label": "timeout", "emoji": ":hourglass:"},
    "resource_exhaustion": {"label": "resource_exhaustion", "emoji": ":boom:"},
    "infrastructure": {"label": "infrastructure", "emoji": ":cloud:"},
    "bug": {"label": "bug", "emoji": ":beetle:"},
}

REPORTABLE_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}

_NON_RETRYABLE_PATTERNS = ("401", "403", "404", "422", "not found")

_RESOURCE_EXHAUSTION_RE = re.compile(
    r"resource_exhausted|outofmemoryerror|memoryerror"
    r"|\boom\b"
    r"|cannot allocate memory|resource temporarily unavailable"
    r"|killed by signal|signal 9",
    re.IGNORECASE,
)

_TIMEOUT_RE = re.compile(
    r"the operation was cancelled"
    r"|the runner has received a shutdown signal"
    r"|timeout after"
    r"|timeouterror"
    r"|deadline exceeded"
    r"|timed out",
    re.IGNORECASE,
)

_INFRASTRUCTURE_RE = re.compile(
    r"unable to connect|connection refused|connection reset|connectionerror"
    r"|503 service unavailable|502 bad gateway|500 internal server error"
    r"|serviceunavailable"
    r"|httperror|google\.api_core\.exceptions"
    r"|preempted|was preempted|tpu is not healthy"
    r"|could not find device|failed to create tpu",
    re.IGNORECASE,
)


def classify_failure(log_text, conclusion=None):
    """Classify failure type from job log text and/or GitHub conclusion.

    Returns one of: 'timeout', 'resource_exhaustion', 'infrastructure', 'bug'.
    Priority: conclusion-based → resource_exhaustion → timeout → infrastructure → bug.
    """
    if conclusion == "startup_failure":
        return "infrastructure"
    if conclusion == "timed_out":
        return "timeout"
    if not log_text:
        return "bug"

    if _RESOURCE_EXHAUSTION_RE.search(log_text):
        return "resource_exhaustion"

    if _TIMEOUT_RE.search(log_text):
        return "timeout"

    if _INFRASTRUCTURE_RE.search(log_text):
        return "infrastructure"

    return "bug"


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
    """Fetch logs and classify each failed job. Mutates jobs in-place, adding 'failure_type'."""
    for job in failed_jobs:
        log_text = fetch_job_logs(repo, job["id"])
        job["failure_type"] = classify_failure(log_text, job.get("conclusion"))
        print(f"  {job['name']}: {job['failure_type']}")


def classify_run(repo, run_id):
    """High-level: fetch failed jobs for a run and classify each one.

    Returns (failed_jobs, needs_ai) where failed_jobs is a list of dicts
    with 'failure_type' populated, and needs_ai is True if any job is
    'bug' or 'resource_exhaustion' (types that require AI analysis).
    """
    failed_jobs = get_failed_jobs(repo, run_id)
    if not failed_jobs:
        return [], False
    classify_jobs(repo, failed_jobs)
    needs_ai = any(j["failure_type"] in ("bug", "resource_exhaustion") for j in failed_jobs)
    return failed_jobs, needs_ai
