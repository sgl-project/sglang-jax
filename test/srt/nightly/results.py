"""Single source of truth for the nightly result shapes.

Accuracy emits ``accuracy_result.v1.yaml`` JSON (``build_accuracy_result`` +
``write_result``). Perf emits per-model CSV (``build_perf_result`` +
``write_perf_csv``) in the column layout ``scripts/ci/plot_perf.py`` and the
trailing-baseline gate consume.

Bump ``ACCURACY_RESULT_SCHEMA_VERSION`` (with a schema-file changelog entry)
whenever the accuracy document shape changes.
"""

import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_NIGHTLY_DIR = os.path.dirname(os.path.abspath(__file__))
if _NIGHTLY_DIR not in sys.path:
    sys.path.insert(0, _NIGHTLY_DIR)

from cases import AccuracyCase, PerfCase  # noqa: E402

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


def write_result(result: dict, case_name: str) -> Path | None:
    """Write ``result`` to ``$RESULTS_DIR/<case_name>.json``.

    Type-agnostic (accuracy or perf). Returns the path written, or ``None`` when
    ``RESULTS_DIR`` is unset (so callers can log the skip in their own voice).
    """
    results_dir = os.environ.get("RESULTS_DIR")
    if not results_dir:
        return None
    out_path = Path(results_dir) / f"{case_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
    return out_path


def build_perf_result(
    case: PerfCase,
    profile_name: str,
    target: str,
    metrics: dict,
    xprof_collected: bool = False,
) -> dict:
    """Build one perf-result record for a finished sweep point.

    Translates ``run_benchmark``'s raw metric names (``output_throughput``, ...)
    into the published CSV column names (``out_tps``, ...) — the single place
    this mapping happens; the gate, floors, baseline, and writer all use the CSV
    names. ``passed`` starts completion-only and is finalized by ``gate_perf_result``.
    """

    def _f(key: str) -> float:
        # Headline metrics default to 0.0 only as a guard; a completed run (the
        # gate) is trusted to return all of these from run_benchmark.
        return float(metrics.get(key, 0.0))

    completed = metrics.get("completed") if isinstance(metrics, dict) else None

    return {
        # CSV columns (plot_perf / ci-data layout); see PERF_CSV_COLUMNS. The
        # per-server profile name is the model_name label so the epmoe and fused
        # MoE servers (same HF checkpoint) get distinct files and plot legends.
        "model_name": profile_name,
        "tpu_size": _tpu_size(target),
        "workload": case.workload,
        "concurrency": case.max_concurrency,
        "input": case.input_len,
        "output": case.output_len,
        "ttft_ms": _f("median_ttft_ms"),
        "itl_ms": _f("median_itl_ms"),
        "in_tps": _f("input_throughput"),
        "out_tps": _f("output_throughput"),
        "total_tps": _f("total_throughput"),
        # Gate-helper fields (not written to CSV).
        "case": case.name,
        "completed": completed,
        "num_prompts": case.num_prompts,
        "xprof_collected": xprof_collected,
        # Dual-gate verdicts; populated by gate_perf_result() before the gate
        # returns. completion-only until then (None = gate not evaluated).
        "passed": completed == case.num_prompts,
        "absolute_floor_passed": None,
        "absolute_baselines": None,
        "trailing_passed": None,
        "baseline_source": None,
    }


# plot_perf.py (and the ci-data perf trend) consume these columns; keep the
# names and order in sync with scripts/ci/plot_perf.py and the ci-data history.
PERF_CSV_COLUMNS = (
    "concurrency",
    "input",
    "output",
    "ttft_ms",
    "itl_ms",
    "in_tps",
    "out_tps",
    "model_name",
    "tpu_size",
)


def _tpu_size(target: str) -> int:
    # "v6e-4" -> 4, "v6e-1" -> 1; default 1 if unparsable.
    try:
        return int(str(target).rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        return 1


def perf_csv_filename(model_name: str, tpu_size: int) -> str:
    """``daily_performance_results_<model>_tp_<n>.csv`` — the name plot_perf.py
    defaults to and the ci-data history uses. Shared by writer and baseline reader."""
    model_dir_name = str(model_name).replace("/", "_")
    return f"daily_performance_results_{model_dir_name}_tp_{tpu_size}.csv"


def write_perf_csv(result: dict) -> Path | None:
    """Append one sweep point to ``$RESULTS_DIR/<perf_csv_filename>`` (one CSV per
    model). Returns the path, or ``None`` when ``RESULTS_DIR`` is unset."""
    results_dir = os.environ.get("RESULTS_DIR")
    if not results_dir:
        return None
    out_path = Path(results_dir) / perf_csv_filename(result["model_name"], result["tpu_size"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {col: result[col] for col in PERF_CSV_COLUMNS}
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PERF_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out_path


def write_perf_json(
    case: PerfCase,
    profile_name: str,
    target: str,
    metrics: dict,
    result: dict | None = None,
) -> Path | None:
    """Write one perf point as ``$RESULTS_DIR/<case>.json`` for the GCS dashboard.

    Shared by the single- and multi-host runners — emits ``{type, case, profile,
    target}`` plus the flat ``run_benchmark`` metrics, the shape the dashboard's
    PerfSummary schema reads. Distinct from the per-model CSV (which feeds the
    ci-data trend plot); single-host writes both, multi-host writes only this.
    Returns the path, or ``None`` when ``RESULTS_DIR`` is unset.
    """
    results_dir = os.environ.get("RESULTS_DIR")
    if not results_dir:
        return None
    summary = {
        "type": "perf",
        "case": case.name,
        "workload": case.workload,
        "profile": profile_name,
        "target": target,
        **(metrics if isinstance(metrics, dict) else {}),
    }
    if result is not None:
        summary["gate"] = {
            "passed": result.get("passed"),
            "absolute_passed": result.get("absolute_floor_passed"),
            "absolute_baselines": result.get("absolute_baselines"),
            "trailing_passed": result.get("trailing_passed"),
            "baseline_source": result.get("baseline_source"),
        }
    # request_rate is inf for the concurrency sweep; json.dumps would write it as
    # the bare token `Infinity`, which JS JSON.parse rejects (the dashboard would
    # drop the whole case). Map non-finite top-level floats to None (valid JSON).
    summary = {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in summary.items()
    }
    out_path = Path(results_dir) / f"{case.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # default=float handles numpy scalars returned by bench_serving.
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float))
    return out_path


def gate_perf_result(case: PerfCase, result: dict) -> tuple[str, str] | None:
    """Dual-threshold perf gate (single-host runner).

    Evaluates three gates in order and records each verdict back into ``result``
    (so the emitted CSV row carries the pass/fail) before returning the first
    failure as ``(kind, message)`` or ``None`` if all pass:

    1. **completion** — ``completed == num_prompts`` (OOM / dropped requests).
       Tagged ``"case"`` (a real bug, CI does not retry).
    2. **absolute gate** — hard bounds from ``case.floors`` and fixed
       baseline/tolerance pairs from ``case.absolute_baselines``.
       Tagged ``"threshold"``.
    3. **trailing baseline** — each point's metrics (``gated_metrics``: a prefill
       point gates in_tps + ttft, a decode point adds out_tps + itl) within
       ``tolerance`` of their last-N-night mean — a drop for throughput, a rise
       for latency. Skipped (not failed) until ``TRAILING_MIN_NIGHTS`` exist.
       Tagged ``"threshold"``.

    Mutates ``result`` in place (verdict fields + final ``passed``).
    """
    from perf_baseline import (
        METRIC_DIRECTION,
        TRAILING_MIN_NIGHTS,
        TRAILING_TOLERANCE,
        fetch_trailing_baseline,
        gated_metrics,
    )

    # 1. completion
    completed = result.get("completed")
    if completed != case.num_prompts:
        result["absolute_floor_passed"] = None
        result["trailing_passed"] = None
        result["passed"] = False
        return ("case", f"{case.name}: completed={completed}/{case.num_prompts}")

    floors = case.floors or {}
    baselines = case.absolute_baselines or {}

    # 2. absolute gate — per-metric hard minimum/maximum, by metric direction
    # (higher-is-better → fail under floor; lower-is-better latency → fail over
    # floor). Today's registered floors are all out_tps (higher), so this is the
    # same `value < floor` as before; the direction split only matters if a
    # latency floor (e.g. ttft_ms) is ever registered.
    floor_failures: list[str] = []
    for metric, floor in floors.items():
        if floor is None:
            continue
        value = result.get(metric)
        if value is None:
            continue
        if METRIC_DIRECTION.get(metric, "higher") == "lower":
            if value > floor:
                floor_failures.append(f"{metric}={value:.1f} > floor {floor:.1f}")
        elif value < floor:
            floor_failures.append(f"{metric}={value:.1f} < floor {floor:.1f}")

    baseline_details: dict[str, dict] = {}
    for metric, baseline in baselines.items():
        direction = METRIC_DIRECTION.get(metric)
        if direction is None:
            result["passed"] = False
            return ("case", f"{case.name}: unknown baseline metric {metric!r}")
        value = result.get(metric)
        if value is None or value <= 0:
            result["passed"] = False
            return ("case", f"{case.name}: missing or invalid metric {metric}={value!r}")
        if direction == "lower":
            bound = baseline.value * (1.0 + baseline.tolerance)
            passed = value <= bound
            comparison = f"{metric}={value:.1f} > {bound:.1f}"
        else:
            bound = baseline.value * (1.0 - baseline.tolerance)
            passed = value >= bound
            comparison = f"{metric}={value:.1f} < {bound:.1f}"
        baseline_details[metric] = {
            "value": baseline.value,
            "tolerance": baseline.tolerance,
            "bound": bound,
            "direction": direction,
            "measured": value,
            "passed": passed,
        }
        if not passed:
            floor_failures.append(comparison)
    result["absolute_baselines"] = baseline_details
    result["absolute_floor_passed"] = not floor_failures

    # 3. trailing baseline (best-effort; skipped when history < min nights).
    # Per-point metrics (gated_metrics): a prefill point gates in_tps + ttft, a
    # decode point adds out_tps + itl. Independent of the absolute floor.
    n = TRAILING_MIN_NIGHTS
    tol = TRAILING_TOLERANCE
    csv_filename = perf_csv_filename(result["model_name"], result["tpu_size"])
    trailing_failures: list[str] = []
    trailing_evaluated = False
    if case.use_trailing_baseline:
        for metric, direction in gated_metrics(result["output"]).items():
            value = result.get(metric)
            if value is None:
                continue
            # A non-positive latency means the metric is missing/degenerate (build_perf_result
            # defaults missing latencies to 0.0), not a record-low latency — skip it rather
            # than letting `0.0 > bound` silently pass the lower-is-better gate.
            if direction == "lower" and value <= 0:
                continue
            mean = fetch_trailing_baseline(
                csv_filename,
                result["concurrency"],
                result["input"],
                result["output"],
                metric,
                n,
            )
            if mean is None:
                continue  # insufficient history -> skip this metric's trailing gate
            trailing_evaluated = True
            if direction == "higher":
                bound = mean * (1.0 - tol)
                if value < bound:
                    trailing_failures.append(
                        f"{metric}={value:.1f} < {mean:.1f}*(1-{tol})={bound:.1f}"
                    )
            else:  # lower-is-better (latency)
                bound = mean * (1.0 + tol)
                if value > bound:
                    trailing_failures.append(
                        f"{metric}={value:.1f} > {mean:.1f}*(1+{tol})={bound:.1f}"
                    )
    if not case.use_trailing_baseline:
        result["trailing_passed"] = None
        result["baseline_source"] = "disabled"
    elif trailing_evaluated:
        result["trailing_passed"] = not trailing_failures
        result["baseline_source"] = f"ci-data last-{n} mean"
    else:
        result["trailing_passed"] = None
        result["baseline_source"] = f"skipped (<{n} nights history)"

    result["passed"] = result["absolute_floor_passed"] and result["trailing_passed"] is not False

    if floor_failures:
        return ("threshold", f"{case.name} absolute: {'; '.join(floor_failures)}")
    if trailing_failures:
        return ("threshold", f"{case.name} trailing: {'; '.join(trailing_failures)}")
    return None
