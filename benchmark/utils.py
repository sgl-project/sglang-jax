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

_COMPILATION_CACHE_ENV_VARS = ("SGLANG_JAX_COMPILATION_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR")


def _maybe_enable_compilation_cache_from_env() -> None:
    cache_dir = None
    for key in _COMPILATION_CACHE_ENV_VARS:
        value = os.environ.get(key)
        if value:
            cache_dir = value
            break
    if not cache_dir:
        return

    try:
        from jax.experimental.compilation_cache import (
            compilation_cache as _compilation_cache,
        )
    except Exception:
        return
    _compilation_cache.set_cache_dir(cache_dir)


_maybe_enable_compilation_cache_from_env()


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

    def _durations_by_pid(events: list[dict[str, Any]]) -> dict[int, list[float]]:
        by_pid: dict[int, list[dict[str, Any]]] = {}
        for e in events:
            pid = e.get("pid")
            if isinstance(pid, int):
                by_pid.setdefault(pid, []).append(e)

        durations: dict[int, list[float]] = {}
        for pid, pid_events in by_pid.items():
            pid_events.sort(key=lambda ev: float(ev.get("ts", 0.0)))
            pid_durations: list[float] = []
            for e in pid_events:
                args = e.get("args", {})
                if args.get("device_duration_ps"):
                    pid_durations.append(float(args["device_duration_ps"]) / 1e9)
                elif "dur" in e:
                    pid_durations.append(float(e["dur"]) / 1e3)
            if pid_durations:
                durations[pid] = pid_durations
        return durations

    if not marker_events:
        if not task:
            return []
        event_matcher = re.compile(task)
        events = []
        for e in trace.get("traceEvents", []):
            if "name" in e and event_matcher.match(e["name"]):
                events.append(e)
        durations_by_pid = _durations_by_pid(events)
        if not durations_by_pid:
            return []
        return max(sorted(durations_by_pid.items()), key=lambda kv: len(kv[1]))[1]

    durations_by_pid = _durations_by_pid(marker_events)
    if not durations_by_pid:
        return []
    return max(sorted(durations_by_pid.items()), key=lambda kv: len(kv[1]))[1]


def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")

    combined: dict[str, Any] = {"traceEvents": []}
    for trace_file in sorted(trace_files):
        with gzip.open(trace_file, "rb") as f:
            shard = json.load(f)
        shard_events = shard.get("traceEvents", [])
        if isinstance(shard_events, list):
            combined["traceEvents"].extend(shard_events)
        if "displayTimeUnit" in shard and "displayTimeUnit" not in combined:
            combined["displayTimeUnit"] = shard["displayTimeUnit"]
        if "otherData" in shard and "otherData" not in combined:
            combined["otherData"] = shard["otherData"]
    return combined


def multiple_iteration_timeit_from_trace(
    compute_func,
    data_generator,
    task: str,
    tries: int = 5,
    warmup: int = 0,
    trace_root: str = "/tmp/sglang_jax_moe_trace",
) -> list[float]:
    """
    Profile multiple iterations and pull per-iteration kernel time from trace.
    """
    trace_name = f"{task}_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    trace_dir = os.path.join(trace_root, trace_name)
    os.makedirs(trace_dir, exist_ok=True)

    start = time.perf_counter()
    for _ in range(max(0, int(warmup))):
        data_args = data_generator()
        out = compute_func(*data_args)
        jax.block_until_ready(out)
    print(f"warmed up in {(time.perf_counter() - start) * 1000} ms")

    with jax.profiler.trace(trace_dir):
        for i in range(tries):
            data_args = data_generator()
            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):
                    out = compute_func(*data_args)
                    jax.block_until_ready(out)

    trace = _load_trace(trace_dir)
    return _extract_marker_durations_ms(trace, task=task)
