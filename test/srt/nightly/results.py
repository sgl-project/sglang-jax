"""Single source of truth for the ``accuracy_result.v1.yaml`` document shape.

Both the single-host runner (``test/srt/nightly/single_host/accuracy_case_runner.py``)
and the multi-host runner (``test/srt/nightly/multi_host/accuracy_case_runner.py``)
build and write their result JSON here, so the schema in
``test/srt/nightly/schemas/accuracy_result.v1.yaml`` has exactly one
producer-side implementation and the two runners cannot drift.

Bump ``ACCURACY_RESULT_SCHEMA_VERSION`` (and add a matching changelog entry to
the schema file) whenever the document shape changes.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_NIGHTLY_DIR = os.path.dirname(os.path.abspath(__file__))
if _NIGHTLY_DIR not in sys.path:
    sys.path.insert(0, _NIGHTLY_DIR)

from cases import AccuracyCase  # noqa: E402

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


def build_accuracy_result(
    case: AccuracyCase,
    profile_name: str,
    target: str,
    metrics: dict | None,
    started_at: float,
    finished_at: float,
) -> dict:
    """Build one ``accuracy_result.v1.yaml`` document for a finished case.

    ``profile_name`` / ``target`` are plain strings so single-host callers can
    pass their own labels and multi-host callers can pass ``profile.name`` /
    ``profile.target`` from a LaunchProfile.
    """
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
        "profile": profile_name,
        "target": target,
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


def write_accuracy_result(result: dict, case_name: str) -> Path | None:
    """Write ``result`` to ``$RESULTS_DIR/<case_name>.json``.

    Returns the path written, or ``None`` when ``RESULTS_DIR`` is unset (so
    callers can log the skip in their own voice).
    """
    results_dir = os.environ.get("RESULTS_DIR")
    if not results_dir:
        return None
    out_path = Path(results_dir) / f"{case_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
    return out_path


def run_eval_for_case(case: AccuracyCase, base_url: str):
    """Drive ``run_eval`` for one case against a live server at ``base_url``.

    Shared by the single- and multi-host accuracy runners so the eval-args
    construction + ``run_eval`` call live in one place. Returns
    ``(metrics, started_at, finished_at)``; each host runner does its own
    logging / gating and calls ``build_accuracy_result`` + ``write_accuracy_result``.
    """
    import time
    from types import SimpleNamespace

    from run_eval import run_eval

    gen = case.generation_config or {}
    # Forward the full sampler config. run_eval routes the SGLang-only params
    # (SGLANG_EXTRA_SAMPLING_PARAMS) into extra_body; cherry-picking a subset
    # here would let a case set e.g. top_k in generation_config, record it in
    # the summary, yet silently drop it before it reaches the sampler.
    args = SimpleNamespace(
        base_url=base_url,
        host=None,
        port=None,
        model=case.model_id,
        eval_name=case.dataset,
        num_examples=case.limit,
        num_threads=case.eval_batch_size,
        temperature=gen.get("temperature", 0.0),
        max_tokens=gen.get("max_tokens", 2048),
        top_p=gen.get("top_p"),
        top_k=gen.get("top_k"),
        min_p=gen.get("min_p"),
        presence_penalty=gen.get("presence_penalty"),
        repetition_penalty=gen.get("repetition_penalty"),
        frequency_penalty=gen.get("frequency_penalty"),
        seed=gen.get("seed"),
        chat_template_kwargs=gen.get("chat_template_kwargs"),
    )
    started_at = time.time()
    metrics = run_eval(args)
    finished_at = time.time()
    return metrics, started_at, finished_at
