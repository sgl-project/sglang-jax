from __future__ import annotations

import gzip
import json
import os
import pathlib
import random
import re
import string
import time
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

    # If the trace contains markers for multiple tasks (e.g. a single trace
    # session timing multiple kernels), prefer filtering by the task-tagged
    # marker name produced by `multiple_iteration_timeit_from_trace(...)`.
    if task:
        task_marker_tag = f"{MARKER}:{task}:"
        task_marker_events = [
            e for e in marker_events if task_marker_tag in str(e.get("args", {}).get("tf_op", ""))
        ]
        if task_marker_events:
            marker_events = task_marker_events

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

    # When tracing TPU programs, the marker events for different iterations may
    # show up under different `pid`s. Group by marker name (`tf_op`) so we
    # reliably return one duration per iteration marker.
    events_by_tf_op: dict[str, list[dict[str, Any]]] = {}
    for e in marker_events:
        tf_op = str(e.get("args", {}).get("tf_op", ""))
        events_by_tf_op.setdefault(tf_op, []).append(e)

    selected_events: list[dict[str, Any]] = []
    for tf_op, events in events_by_tf_op.items():
        # Pick a stable representative event per marker (lowest pid).
        selected_events.append(min(events, key=lambda ev: ev.get("pid", 0)))

    def _event_sort_key(e: dict[str, Any]) -> tuple[int, float]:
        tf_op = str(e.get("args", {}).get("tf_op", ""))
        iter_idx = 0
        if task:
            m = re.search(re.escape(f"{MARKER}:{task}:") + r"(\\d+)", tf_op)
            if m:
                iter_idx = int(m.group(1))
        return (iter_idx, float(e.get("ts", 0.0)))

    selected_events.sort(key=_event_sort_key)

    durations_ms: list[float] = []
    for e in selected_events:
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
    warmup: int = 1,
    trace_root: str = "/tmp/sglang_jax_moe_trace",
) -> list[float]:
    """
    Profile multiple iterations and pull per-iteration kernel time from trace.
    """
    if warmup < 0:
        raise ValueError(f"Expected {warmup=} to be >= 0.")
    trace_name = f"{task}_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    trace_dir = os.path.join(trace_root, trace_name)
    os.makedirs(trace_dir, exist_ok=True)

    # Warm up outside the profiler trace so compilation / one-time setup does not
    # bloat the trace (and can otherwise make profiling flakier on TPU).
    for _ in range(warmup):
        out = compute_func(*data_generator())
        jax.block_until_ready(out)

    with jax.profiler.trace(trace_dir):
        for i in range(tries):
            data_args = data_generator()
            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                # Encode `task` into the marker name so a single trace session can
                # time multiple tasks without mixing results.
                with jax.named_scope(f"{MARKER}:{task}:{i}"):
                    out = compute_func(*data_args)
                    jax.block_until_ready(out)

    trace = _load_trace(trace_dir)
    return _extract_marker_durations_ms(trace, task=task)


def multiple_tasks_timeit_from_trace(
    task_to_compute: dict[str, object],
    *,
    tries: int = 5,
    warmup: int = 1,
    trace_root: str = "/tmp/sglang_jax_moe_trace",
    trace_group: str = "multi",
) -> dict[str, list[float]]:
    """Profile multiple tasks in a single trace session.

    This avoids repeatedly creating TPU profiler sessions (which can be flaky
    under heavy tuning loops), while still returning per-iteration kernel times
    extracted from the trace.
    """
    if tries <= 0:
        raise ValueError(f"Expected {tries=} to be > 0.")
    if warmup < 0:
        raise ValueError(f"Expected {warmup=} to be >= 0.")
    if not task_to_compute:
        return {}

    trace_name = f"{trace_group}_" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=6)
    )
    trace_dir = os.path.join(trace_root, trace_name)
    os.makedirs(trace_dir, exist_ok=True)

    for task, compute in task_to_compute.items():
        if not callable(compute):
            raise TypeError(f"Expected compute for task '{task}' to be callable.")
        for _ in range(warmup):
            out = compute()
            jax.block_until_ready(out)

    with jax.profiler.trace(trace_dir):
        global_step = 0
        for task, compute in task_to_compute.items():
            for i in range(tries):
                with jax.profiler.StepTraceAnnotation(task, step_num=global_step):
                    with jax.named_scope(f"{MARKER}:{task}:{i}"):
                        out = compute()
                        jax.block_until_ready(out)
                global_step += 1

    trace = _load_trace(trace_dir)
    return {task: _extract_marker_durations_ms(trace, task=task) for task in task_to_compute}


def multiple_iteration_timeit(
    compute_func,
    data_generator,
    *,
    tries: int = 5,
    warmup: int = 0,
) -> list[float]:
    """Wall-clock timing helper (no profiler trace).

    Returns a list of per-iteration durations in milliseconds. The timed region
    includes device dispatch and execution; outputs are synchronized via
    `jax.block_until_ready`.
    """
    if warmup < 0:
        raise ValueError(f"Expected {warmup=} to be >= 0.")
    if tries <= 0:
        raise ValueError(f"Expected {tries=} to be > 0.")

    for _ in range(warmup):
        out = compute_func(*data_generator())
        jax.block_until_ready(out)

    times_ms: list[float] = []
    for _ in range(tries):
        data_args = data_generator()
        start = time.perf_counter()
        out = compute_func(*data_args)
        jax.block_until_ready(out)
        times_ms.append((time.perf_counter() - start) * 1000)
    return times_ms
