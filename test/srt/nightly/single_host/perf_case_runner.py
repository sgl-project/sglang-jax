"""Single-host perf helpers, used by ``single_host/suite_runner.py``.

Runs one ``PerfCase`` sweep point against a live server and appends a row to the
per-model CSV. Driver/emitter (``run_benchmark_for_case`` / ``build_perf_result``
/ ``write_perf_csv``) live in ``results.py``; the sweep grid in ``cases.py``.
"""

import os
import sys

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_NIGHTLY_DIR = os.path.dirname(_SELF_DIR)
_TEST_SRT = os.path.dirname(_NIGHTLY_DIR)
for _p in (_TEST_SRT, _NIGHTLY_DIR, _SELF_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cases import PerfCase  # noqa: E402
from drivers import run_benchmark_for_case  # noqa: E402
from results import (  # noqa: E402
    build_perf_result,
    gate_perf_result,
    write_perf_csv,
    write_perf_json,
)


def run_perf_case(
    case: PerfCase,
    base_url: str,
    tokenizer: str,
    profile_name: str,
    target: str,
) -> tuple[dict, tuple[str, str] | None]:
    """Run one PerfCase sweep point against an already-running server.

    Appends one row to the per-model CSV in ``$RESULTS_DIR`` when set. Returns
    the full result dict (including ``completed`` and ``passed``); gating is left
    to the caller.
    """
    xprof = case.capture_trace
    print(
        f"[perf-runner] Running case name={case.name}, base_url={base_url}, "
        f"concurrency={case.max_concurrency}, input_len={case.input_len}, "
        f"output_len={case.output_len}, num_prompts={case.num_prompts}, profile={xprof}",
        flush=True,
    )

    metrics = run_benchmark_for_case(
        case,
        base_url,
        tokenizer,
        profile=xprof,
        profile_num_steps=case.profile_num_steps if xprof else None,
    )

    result = build_perf_result(case, profile_name, target, metrics, xprof_collected=xprof)
    fail = gate_perf_result(case, result)

    # Incomplete points (OOM / dropped requests) aren't persisted — a partial
    # metric must not pollute the trend or baseline. Threshold failures ran fully
    # and are recorded.
    incomplete = fail is not None and fail[0] == "case"
    if incomplete:
        print(
            f"[perf-runner] {case.name}: incomplete "
            f"(completed={result['completed']}/{case.num_prompts}) — skipping CSV/JSON",
            flush=True,
        )
    else:
        out_path = write_perf_csv(result)
        if out_path is not None:
            print(f"[perf-runner] Wrote result to {out_path}", flush=True)
        json_path = write_perf_json(case, profile_name, target, metrics)
        if json_path is not None:
            print(f"[perf-runner] Wrote dashboard JSON to {json_path}", flush=True)

    print(
        f"[perf-runner] {case.name}: completed={result['completed']}/{case.num_prompts}, "
        f"out_tps={result['out_tps']:.1f}, "
        f"ttft={result['ttft_ms']:.1f}ms, itl={result['itl_ms']:.1f}ms",
        flush=True,
    )

    return result, fail
