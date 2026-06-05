"""Slash command parser for PR comment-triggered CI operations.

Reads comment body and author information from environment variables,
parses the slash command, checks permissions, resolves job parameters,
and writes results to $GITHUB_OUTPUT.
"""

import json
import os
import re
import subprocess
import sys

ALLOWED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}

VALID_COMMANDS = {
    "rerun-failed-ci",
    "test",
    "rerun-group",
    "rerun-stage",
    "run-nightly",
}

_SAFE_ARG_RE = re.compile(r"[^a-zA-Z0-9_-]")

# case_key = a real case name (lowercase, e.g. "qwen3-8b-fa" or
# "qwen3-32b-c32-i4096-o1024"), so the charset includes hyphens.
_CASE_KEY_RE = re.compile(r"[^a-z0-9_.-]")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Single-host suite -> its nightly-test-daily.yml job (job_filter value). case_keys
# aren't listed here — they're derived from the catalog (single source of truth),
# so adding a case to a suite needs no change here.
_SINGLE_HOST_SUITE_JOBS = {
    "accuracy-text-models-v6e-4": "nightly-test-accuracy-text-models-4-tpu-daily",
    "perf-text-models-v6e-4": "nightly-test-perf-text-models-4-tpu-daily",
}
# Multi-host (the mimo-flash suite) is intentionally NOT wired into /run-nightly: its
# nightly job stays `if: false` until multi-host CI has 4-node v6e capacity. To enable
# later, add a suite->job map here and enumerate it in nightly_index().


class NightlyEnumerationError(RuntimeError):
    """Raised when a suite runner's --caselist can't be enumerated."""


def _run_caselist(runner_relpath: str) -> list[dict]:
    """Enumerate a runner's cases via its --caselist CLI (a subprocess).

    Mirrors pytest's `--collect-only`: the suite runner owns its case list and
    prints it as JSON; we don't import it (it's stdlib-only for --caselist, but a
    subprocess keeps the parser's import space clean and isolates failures). Runs
    on a plain CPU runner — --caselist needs no jax. The timeout guards the slash
    handler against a catalog that hangs at import. Raises on non-zero exit.
    """
    path = os.path.join(_REPO_ROOT, "test", "srt", "nightly", runner_relpath)
    try:
        proc = subprocess.run(
            [sys.executable, path, "--caselist"], capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired as exc:
        raise NightlyEnumerationError(
            f"{runner_relpath} --caselist timed out after {exc.timeout:.0f}s"
        ) from exc
    if proc.returncode != 0:
        raise NightlyEnumerationError(
            f"{runner_relpath} --caselist failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[:300]}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise NightlyEnumerationError(f"{runner_relpath} --caselist bad JSON: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(entry, dict) for entry in data):
        raise NightlyEnumerationError(f"{runner_relpath} --caselist must be a JSON list of objects")
    return data


def nightly_index() -> dict[str, tuple[str, str]]:
    """Map each runnable case_key -> (job, cases-arg for suite_runner --cases).

    The single-host runner self-enumerates via --caselist; this maps the suites it
    exposes to a job name, one key per case (cases=case). Multi-host is not wired in
    (see _SINGLE_HOST_SUITE_JOBS note above).
    """
    index: dict[str, tuple[str, str]] = {}
    for entry in _run_caselist("single_host/suite_runner.py"):
        suite, case = entry.get("suite"), entry.get("case")
        if suite is None or case is None:
            continue
        job = _SINGLE_HOST_SUITE_JOBS.get(suite)
        if job is None:
            continue
        index[case] = (job, case)
    return index


def format_nightly_list():
    """Render the runnable /run-nightly case_keys, grouped by job."""
    try:
        index = nightly_index()
    except NightlyEnumerationError as exc:
        return f"Could not list nightly cases: {exc}"
    by_job: dict[str, list[str]] = {}
    for key, (job, _cases) in index.items():
        by_job.setdefault(job, []).append(key)
    lines = ["**Available `/run-nightly` cases:**", ""]
    for job in sorted(by_job):
        names = ", ".join(f"`{k}`" for k in sorted(by_job[job]))
        lines.append(f"- `{job}`: {names}")
    lines.append("")
    lines.append("Usage: `/run-nightly <case_key>` — e.g. `/run-nightly qwen3-8b-fa`")
    return "\n".join(lines)


RUNNER_SUFFIXES = [
    ("-cpu", "arc-runner-cpu"),
    ("-v6e-4", "arc-runner-v6e-4"),
    ("-v6e-1", "arc-runner-v6e-1"),
]

STAGE_JOBS = {
    "stage1": ["unit-test-1-tpu", "unit-test-cpu"],
    "stage2": [
        "unit-test-4-tpu",
        "e2e-test-1-tpu",
        "e2e-test-4-tpu",
        "accuracy-test-1-tpu",
        "accuracy-test-4-tpu",
    ],
    "stage3": ["performance-test-1-tpu", "performance-test-4-tpu"],
}

STAGE_ALIASES = {
    "1": "stage1",
    "fast": "stage1",
    "stage1": "stage1",
    "2": "stage2",
    "medium": "stage2",
    "stage2": "stage2",
    "3": "stage3",
    "heavy": "stage3",
    "stage3": "stage3",
}

JOB_TO_SUITE = {
    "unit-test-1-tpu": ("unit-test-tpu-v6e-1", "arc-runner-v6e-1"),
    "unit-test-cpu": ("unit-test-cpu", "arc-runner-cpu"),
    "unit-test-4-tpu": ("unit-test-tpu-v6e-4", "arc-runner-v6e-4"),
    "e2e-test-1-tpu": ("e2e-test-tpu-v6e-1", "arc-runner-v6e-1"),
    "e2e-test-4-tpu": ("e2e-test-tpu-v6e-4", "arc-runner-v6e-4"),
    "accuracy-test-1-tpu": ("accuracy-test-tpu-v6e-1", "arc-runner-v6e-1"),
    "accuracy-test-4-tpu": ("accuracy-test-tpu-v6e-4", "arc-runner-v6e-4"),
    "performance-test-1-tpu": ("performance-test-tpu-v6e-1", "arc-runner-v6e-1"),
    "performance-test-4-tpu": ("performance-test-tpu-v6e-4", "arc-runner-v6e-4"),
}


def _sanitize_arg(arg, max_len=20):
    """Strip non-alphanumeric chars and truncate to prevent reflected content injection."""
    return _SAFE_ARG_RE.sub("", arg)[:max_len]


def resolve_runner(suite):
    """Derive the runner label from a test suite name. Returns None if unknown."""
    for suffix, runner in RUNNER_SUFFIXES:
        if suite.endswith(suffix):
            return runner
    return None


def parse_command(comment_body):
    """Extract (command, args) from comment body, or (None, []) if invalid."""
    if not comment_body:
        return (None, [])

    first_line = comment_body.strip().split("\n")[0].strip()
    if not first_line.startswith("/"):
        return (None, [])

    parts = first_line.split()
    command = parts[0][1:]
    args = parts[1:]

    if command not in VALID_COMMANDS:
        return (None, [])

    return (command, args)


def check_permission(actor, actor_association):
    """Return True if the actor is allowed to run slash commands."""
    return actor_association.upper() in ALLOWED_ASSOCIATIONS


def resolve_jobs(command, args):
    """Map command + args to an action descriptor dict.

    Keys:
        action     — "rerun_failed" | "add_label" | "rerun_group" | "rerun_stage"
                     | "run_nightly"
        suite      — test suite name (for rerun-group)
        runner     — runner label (for rerun-group)
        labels     — labels to add to the PR
        stage      — canonical stage name (for rerun-stage)
        jobs       — list of pr-test.yml job names (for rerun-stage)
        nightly_*  — job / cases / case_key (for run-nightly)
        error      — non-empty string on invalid input
    """
    result = {
        "action": "",
        "suite": "",
        "runner": "",
        "labels": [],
        "error": "",
        "stage": "",
        "jobs": [],
        "nightly_job": "",
        "nightly_cases": "",
        "nightly_case_key": "",
    }

    if command == "rerun-failed-ci":
        result["action"] = "rerun_failed"
        return result

    if command == "test":
        if not args:
            result["error"] = "Missing test type. Usage: /test perf"
            return result
        test_type = args[0].lower()
        if test_type == "perf":
            result["action"] = "add_label"
            result["labels"] = ["test:perf"]
            return result
        safe = _sanitize_arg(args[0])
        result["error"] = f"Unknown test type '{safe}'. Valid: perf"
        return result

    if command == "rerun-group":
        if not args:
            result["error"] = "Missing suite name. Usage: /rerun-group <suite-name>"
            return result
        suite = _sanitize_arg(args[0], max_len=64)
        runner = resolve_runner(suite)
        if runner is None:
            result["error"] = (
                f"Cannot determine runner for suite '{suite}'. "
                "Suite name must end with -v6e-1, -v6e-4, or -cpu"
            )
            return result
        result["action"] = "rerun_group"
        result["suite"] = suite
        result["runner"] = runner
        return result

    if command == "rerun-stage":
        if not args:
            result["error"] = "Missing stage. Usage: /rerun-stage <1|2|3|fast|medium|heavy>"
            return result
        raw_arg = _sanitize_arg(args[0])
        stage = STAGE_ALIASES.get(raw_arg.lower())
        if stage is None:
            result["error"] = f"Unknown stage '{raw_arg}'. Valid: 1/fast, 2/medium, 3/heavy"
            return result
        result["action"] = "rerun_stage"
        result["stage"] = stage
        result["jobs"] = STAGE_JOBS[stage]
        return result

    if command == "run-nightly":
        # No arg or "?" → list cases instead of erroring ("?" mirrors Prow's
        # /test ?, but a bare /run-nightly should be just as discoverable).
        if not args or args[0] == "?":
            result["action"] = "list_nightly"
            return result
        case_key = args[0].lower()
        # Reject malformed keys (don't silently strip) so the error names what was
        # typed. Echo with only illegal chars removed; the kept set is
        # injection-safe, "<empty>" if every char was illegal.
        if _CASE_KEY_RE.search(case_key):
            safe = _CASE_KEY_RE.sub("", case_key)[:60] or "<empty>"
            result["error"] = f"Invalid case_key '{safe}'. Allowed characters: [a-z0-9_.-]"
            return result
        try:
            index = nightly_index()
        except NightlyEnumerationError as exc:
            result["error"] = f"Could not list nightly cases: {exc}"
            return result
        entry = index.get(case_key)
        if entry is None:
            valid = ", ".join(sorted(index))
            result["error"] = f"Unknown case_key '{case_key}'. Valid: {valid}"
            return result
        job, cases = entry
        result["action"] = "run_nightly"
        result["nightly_job"] = job
        result["nightly_cases"] = cases
        result["nightly_case_key"] = case_key
        return result

    result["error"] = f"Unknown command '{command}'"
    return result


def write_outputs(outputs):
    """Write key=value pairs to $GITHUB_OUTPUT."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            for name, value in outputs.items():
                f.write(f"{name}={value}\n")
    else:
        print("(GITHUB_OUTPUT not set — printing only)", file=sys.stderr)


def main():
    comment_body = os.environ.get("COMMENT_BODY", "")
    actor = os.environ.get("ACTOR", "")
    actor_association = os.environ.get("ACTOR_ASSOCIATION", "")

    command, args = parse_command(comment_body)

    if command is None:
        write_outputs({"valid": "false", "error": "Not a valid slash command"})
        print("No valid slash command found")
        return

    permitted = check_permission(actor, actor_association)

    if not permitted:
        write_outputs(
            {
                "valid": "true",
                "permitted": "false",
                "command": command,
                "error": f"Permission denied for {actor} ({actor_association})",
            }
        )
        print(f"Permission denied: {actor} ({actor_association})")
        return

    job_info = resolve_jobs(command, args)

    if job_info["error"]:
        write_outputs(
            {
                "valid": "true",
                "permitted": "true",
                "command": command,
                "error": job_info["error"],
            }
        )
        print(f"Command error: {job_info['error']}")
        return

    outputs = {
        "valid": "true",
        "permitted": "true",
        "command": command,
        "action": job_info["action"],
        "error": "",
    }
    if job_info["suite"]:
        outputs["suite"] = job_info["suite"]
    if job_info["runner"]:
        outputs["runner"] = job_info["runner"]
    if job_info["labels"]:
        outputs["labels_csv"] = ",".join(job_info["labels"])
    if job_info["stage"]:
        outputs["stage"] = job_info["stage"]
        suites = []
        for job in job_info["jobs"]:
            suite_name, runner = JOB_TO_SUITE[job]
            suites.append({"suite": suite_name, "runner": runner})
        outputs["stage_suites_json"] = json.dumps(suites)
    if job_info["action"] == "run_nightly":
        outputs["nightly_job"] = job_info["nightly_job"]
        outputs["nightly_cases"] = job_info["nightly_cases"]
        outputs["nightly_case_key"] = job_info["nightly_case_key"]

    write_outputs(outputs)

    print("=== Slash Command ===")
    print(f"command: {command}")
    print(f"args: {args}")
    print(f"actor: {actor} ({actor_association})")
    print(f"action: {job_info['action']}")
    if job_info["suite"]:
        print(f"suite: {job_info['suite']}")
    if job_info["runner"]:
        print(f"runner: {job_info['runner']}")
    if job_info["labels"]:
        print(f"labels: {', '.join(job_info['labels'])}")
    if job_info["stage"]:
        print(f"stage: {job_info['stage']}")
        print(f"jobs: {', '.join(job_info['jobs'])}")
    if job_info["action"] == "run_nightly":
        print(f"case_key: {job_info['nightly_case_key']}")
        print(f"nightly_job: {job_info['nightly_job']}")
        if job_info["nightly_cases"]:
            print(f"nightly_cases: {job_info['nightly_cases']}")


if __name__ == "__main__":
    main()
