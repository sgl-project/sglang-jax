"""Accuracy case runner: drives run_eval against a given server URL.

The summary JSON written to ``${RESULTS_DIR}/<case>.json`` follows the schema
documented in ``test/srt/eval/schemas/accuracy_result.v1.yaml`` (shared across
single-host and multi-host accuracy nightlies). Bump
``ACCURACY_RESULT_SCHEMA_VERSION`` (and add a matching changelog entry to the
schema file) whenever the document shape changes.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_TEST_SRT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if _TEST_SRT not in sys.path:
    sys.path.insert(0, _TEST_SRT)

from multi_host_suite import AccuracyCase
from profile_loader import LaunchProfile

ACCURACY_RESULT_SCHEMA_VERSION = "1.0.0"


def _utc_iso(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _detect_commit_sha() -> str | None:
    explicit = os.environ.get("GITHUB_SHA") or os.environ.get("COMMIT_SHA")
    if explicit:
        return explicit
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _build_github_run_url() -> str | None:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def _build_summary(
    case: AccuracyCase,
    profile: LaunchProfile,
    metrics: dict | None,
    started_at: float,
    finished_at: float,
) -> dict:
    score = metrics.get("score") if isinstance(metrics, dict) else None
    passed: bool | None
    if case.score_threshold is None or score is None:
        passed = None
    else:
        passed = score >= case.score_threshold

    return {
        "schema_version": ACCURACY_RESULT_SCHEMA_VERSION,
        "type": "accuracy",
        "case": case.name,
        "dataset": case.dataset,
        "model_id": case.model_id,
        "profile": profile.name,
        "target": profile.target,
        "score": score,
        "score_threshold": case.score_threshold,
        "passed": passed,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "started_at": _utc_iso(started_at),
        "finished_at": _utc_iso(finished_at),
        "duration_seconds": round(finished_at - started_at, 3),
        "num_examples": case.limit,
        "eval_batch_size": case.eval_batch_size,
        "generation_config": dict(case.generation_config or {}),
        "repo_ref": os.environ.get("REPO_REF") or os.environ.get("GITHUB_REF_NAME"),
        "commit_sha": _detect_commit_sha(),
        "workload_name": os.environ.get("WORKLOAD_NAME"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_url": _build_github_run_url(),
    }


def run_accuracy_case(case: AccuracyCase, profile: LaunchProfile) -> None:
    from run_eval import run_eval

    gen = case.generation_config or {}
    args = SimpleNamespace(
        base_url=f"http://127.0.0.1:{profile.port}",
        host=None,
        port=None,
        model=case.model_id,
        eval_name=case.dataset,
        num_examples=case.limit,
        num_threads=case.eval_batch_size,
        temperature=gen.get("temperature", 0.0),
        max_tokens=gen.get("max_tokens", 2048),
        top_p=gen.get("top_p"),
        chat_template_kwargs=gen.get("chat_template_kwargs"),
    )

    print(
        f"[multi-host-suite] Running accuracy case "
        f"name={case.name}, dataset={case.dataset}, "
        f"num_threads={args.num_threads}, "
        f"temperature={args.temperature}, max_tokens={args.max_tokens}, "
        f"top_p={args.top_p}, chat_template_kwargs={args.chat_template_kwargs}, "
        f"limit={case.limit}",
        flush=True,
    )

    started_at = time.time()
    metrics = run_eval(args)
    finished_at = time.time()

    summary = _build_summary(case, profile, metrics, started_at, finished_at)

    results_dir = os.environ.get("RESULTS_DIR")
    if results_dir:
        out_path = Path(results_dir) / f"{case.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
        print(f"[multi-host-suite] Wrote accuracy summary to {out_path}", flush=True)
    else:
        print(
            f"[multi-host-suite] RESULTS_DIR unset; skipping accuracy summary write",
            flush=True,
        )

    score = summary["score"]
    if case.score_threshold is not None:
        if score is None:
            raise RuntimeError(
                f"Accuracy case {case.name} produced no score; cannot evaluate "
                f"against threshold={case.score_threshold}"
            )
        if score < case.score_threshold:
            raise RuntimeError(
                f"Accuracy case {case.name} score={score:.4f} below "
                f"threshold={case.score_threshold:.4f}"
            )
        print(
            f"[multi-host-suite] Accuracy case {case.name} passed: "
            f"score={score:.4f} >= threshold={case.score_threshold:.4f}",
            flush=True,
        )
    else:
        print(
            f"[multi-host-suite] Accuracy case {case.name} finished "
            f"(no threshold set, score={score})",
            flush=True,
        )
