from __future__ import annotations

import gzip
import json
import os
import pathlib
import random
import re
import string
from typing import Any

import jax

MARKER = "SGLANG_JAX_BENCH"


def _extract_marker_durations_ms(trace: dict[str, Any], task: str | None = None) -> list[float]:
    marker_events: list[dict[str, Any]] = []
    for e in trace.get("traceEvents", []):
        args = e.get("args", {})
        tf_op = args.get("tf_op", "")
        if MARKER in tf_op:
            marker_events.append(e)

    marker_call_done_events = [e for e in marker_events if e.get("name", "").endswith("call-done")]
    if marker_call_done_events:
        marker_events = marker_call_done_events

    if not marker_events:
        if not task:
            return []
        event_matcher = re.compile(task)
        events = []
        for e in trace.get("traceEvents", []):
            if "name" in e and event_matcher.match(e["name"]):
                events.append(e)
        if not events:
            return []
        min_pid = min(e["pid"] for e in events)
        events_from_min_pid = [e for e in events if e["pid"] == min_pid]
        durations_ms: list[float] = []
        for e in events_from_min_pid:
            if e.get("args", {}).get("device_duration_ps"):
                durations_ms.append(float(e["args"]["device_duration_ps"]) / 1e9)
            elif "dur" in e:
                durations_ms.append(float(e["dur"]) / 1e3)
        return durations_ms

    min_pid = min(e["pid"] for e in marker_events)
    events_from_min_pid = [e for e in marker_events if e["pid"] == min_pid]
    durations_ms = []
    for e in events_from_min_pid:
        args = e.get("args", {})
        if "device_duration_ps" in args:
            durations_ms.append(float(args["device_duration_ps"]) / 1e9)
        elif "dur" in e:
            durations_ms.append(float(e["dur"]) / 1e3)
    return durations_ms


def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")
    with gzip.open(trace_files[0], "rb") as f:
        return json.load(f)


def multiple_iteration_timeit_from_trace(
    compute_func,
    data_generator,
    task: str,
    tries: int = 5,
    trace_root: str = "/tmp/sglang_jax_moe_trace",
) -> list[float]:
    """
    Profile multiple iterations and pull per-iteration kernel time from trace.
    """
    trace_name = f"{task}_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    trace_dir = os.path.join(trace_root, trace_name)
    os.makedirs(trace_dir, exist_ok=True)

    with jax.profiler.trace(trace_dir):
        for i in range(tries):
            data_args = data_generator()
            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):
                    out = compute_func(*data_args)
                    jax.block_until_ready(out)

    trace = _load_trace(trace_dir)
    return _extract_marker_durations_ms(trace, task=task)
