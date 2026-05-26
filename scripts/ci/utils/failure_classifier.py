"""Failure classification for CI jobs.

Shared module used by slack_notify.py (nightly notifications) and
potentially bisect_preflight.py (auto-bisect eligibility).

To add a new failure type:
  1. Add a compiled regex to the _*_RE constants below.
  2. Add an entry in FAILURE_TYPES with label and emoji.
  3. Add a branch in classify_failure() at the correct priority level.
"""

import re

FAILURE_TYPES = {
    "timeout": {"label": "timeout", "emoji": ":hourglass:"},
    "resource_exhaustion": {"label": "resource_exhaustion", "emoji": ":boom:"},
    "infrastructure": {"label": "infrastructure", "emoji": ":cloud:"},
    "bug": {"label": "bug", "emoji": ":beetle:"},
}

REPORTABLE_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}

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
