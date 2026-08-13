"""Benchmark GLM-5.2 per-channel FP8 matmul on one TPU device.

TP=1/TP=2 describe local matrix shapes only. No collective is executed. Primary
latency comes from XProf ``device_duration_ps``; host wall time is diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.kernels.quantized_matmul.glm52_per_channel_cases import (
    IMPLEMENTATIONS,
    MODES,
    BenchmarkCase,
    build_cases,
    expected_case_counts,
)

TPU_INFERENCE_REFERENCE_SHA = "a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336"
TRACE_MARKER = "GLM52_PER_CHANNEL_QMM"
# A v7x M=1 preflight observed relative_l2=2.47e-2 between a JIT-fused
# W8A8 XLA result and the separately executed, same-math oracle. Keep a 2x
# margin for the full shape matrix while reporting the raw error metrics.
CORRECTNESS_THRESHOLDS = {"w8a16": 1e-4, "w8a8": 5e-2}


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_dirty() -> bool | None:
    env_dirty = os.environ.get("SOURCE_GIT_DIRTY")
    if env_dirty is not None:
        return env_dirty.lower() in ("1", "true", "yes")
    status = _git_value("status", "--porcelain")
    return None if status is None else bool(status)


def _percentile(samples: Sequence[float], percent: float) -> float:
    values = sorted(float(value) for value in samples)
    if not values:
        raise ValueError("no samples")
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def summarize_samples(samples_us: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples_us]
    if not values:
        raise ValueError("no timing samples")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return {
        "count": len(values),
        "raw_samples_us": values,
        "min_us": min(values),
        "p50_us": _percentile(values, 50),
        "p90_us": _percentile(values, 90),
        "p95_us": _percentile(values, 95),
        "max_us": max(values),
        "mean_us": mean,
        "std_us": std,
        "cv": std / mean if mean else None,
    }


def _profile_aliases() -> set[tuple[str, int]]:
    return {
        ("kv_a_proj_with_mqa", 1),
        ("q_b_proj", 2),
        ("o_proj", 2),
        ("merged_gate_up_proj", 2),
    }


def _select_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    if args.suite == "anchor":
        suites = ("anchor",)
    elif args.suite == "full":
        suites = ("full",)
    else:
        suites = ("anchor", "full")

    if args.suite == "smoke":
        operations = args.operations or ["kv_a_proj_with_mqa"]
        tp_degrees = args.tp_degree or [1]
        m_values = args.m or [2]
    elif args.suite == "profiles":
        operations = args.operations or sorted({item[0] for item in _profile_aliases()})
        tp_degrees = args.tp_degree
        m_values = args.m or [2, 1024]
    else:
        operations = args.operations
        tp_degrees = args.tp_degree
        m_values = args.m

    cases = build_cases(
        suites=suites,
        operations=operations,
        tp_degrees=tp_degrees,
        m_values=m_values,
        modes=args.modes,
        implementations=args.implementations,
    )
    if args.suite == "profiles":
        cases = [
            case
            for case in cases
            if any(
                (alias.operation, alias.tp_degree) in _profile_aliases() for alias in case.aliases
            )
        ]
    return cases


def _print_cases(cases: Sequence[BenchmarkCase]) -> None:
    print("operation                 TP      M      N      K  mode    implementation  suites")
    for case in cases:
        alias = case.primary_alias
        suites = ",".join(case.suites)
        print(
            f"{alias.operation:<25} {alias.tp_degree:>2} {case.key.m:>6} "
            f"{case.key.n:>6} {case.key.k:>6} {case.key.mode:<7} "
            f"{case.key.implementation:<15} {suites}"
        )
    print(f"\nTotal physical cases: {len(cases)}")
    print(f"Expected complete counts: {expected_case_counts()}")


def _validate_device(args: argparse.Namespace) -> Any:
    import jax

    devices = jax.local_devices()
    if not 0 <= args.device_index < len(devices):
        raise ValueError(f"device-index must be in [0, {len(devices) - 1}]")
    device = devices[args.device_index]
    if device.platform != "tpu" and not args.allow_non_tpu:
        raise RuntimeError(
            f"TPU required, found {device.platform}/{device.device_kind}; "
            "use --allow-non-tpu for XLA-only smoke tests"
        )
    return device


def _make_activation(case: BenchmarkCase, device: Any, seed: int) -> tuple[Any, int | None]:
    import jax
    import jax.numpy as jnp

    key = jax.random.key(_stable_seed(seed, "activation", case.key.m, case.key.k))
    with jax.default_device(device):
        x = jax.random.normal(key, (case.key.m, case.key.k), dtype=jnp.bfloat16) * 0.125
        zero_row = None
        if case.key.m >= 2:
            zero_row = 0
            x = x.at[zero_row].set(jnp.zeros((case.key.k,), dtype=jnp.bfloat16))
        if case.key.m >= 3:
            x = x.at[1].multiply(jnp.asarray(1e-4, dtype=jnp.bfloat16))
        if case.key.m >= 4:
            x = x.at[-1, 0].set(jnp.asarray(32.0, dtype=jnp.bfloat16))
    jax.block_until_ready(x)
    return x, zero_row


def _make_weight(case: BenchmarkCase, device: Any, seed: int, ring_index: int) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    key = jax.random.key(_stable_seed(seed, "weight", case.key.n, case.key.k, ring_index))
    with jax.default_device(device):
        master = jax.random.uniform(
            key,
            (case.key.n, case.key.k),
            dtype=jnp.bfloat16,
            minval=-0.5,
            maxval=0.5,
        )
        channel_multiplier = jnp.exp(
            jnp.linspace(math.log(0.25), math.log(4.0), case.key.n, dtype=jnp.float32)
        ).reshape(case.key.n, 1)
        master = (master.astype(jnp.float32) * channel_multiplier).astype(jnp.bfloat16)
        dtype_info = jnp.finfo(jnp.float8_e4m3fn)
        abs_max = jnp.max(jnp.abs(master.astype(jnp.float32)), axis=1, keepdims=True)
        w_scale_2d = abs_max / float(dtype_info.max)
        safe_scale = w_scale_2d + (w_scale_2d == 0).astype(w_scale_2d.dtype)
        w_q = jnp.clip(
            master / safe_scale,
            float(dtype_info.min),
            float(dtype_info.max),
        ).astype(jnp.float8_e4m3fn)
        w_scale = jnp.squeeze(w_scale_2d, axis=1).astype(jnp.float32)
    jax.block_until_ready((w_q, w_scale))
    return w_q, w_scale


def _make_input_ring(
    case: BenchmarkCase, device: Any, seed: int, ring_count: int
) -> tuple[list[tuple[Any, Any, Any]], int | None]:
    import jax

    x, zero_row = _make_activation(case, device, seed)
    ring = []
    for ring_index in range(ring_count):
        w_q, w_scale = _make_weight(case, device, seed, ring_index)
        ring.append((x, w_q, w_scale))
    jax.block_until_ready(ring)
    return ring, zero_row


def _tpu_inference_reference(case: BenchmarkCase, x: Any, w_q: Any, w_scale: Any) -> Any:
    """Mirror the aligned Pallas kernel's quantize-array and matmul semantics."""

    import jax
    import jax.numpy as jnp

    if case.key.mode == "w8a8":
        dtype_info = jnp.finfo(w_q.dtype)
        abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
        x_scale = abs_max / float(dtype_info.max)
        safe_scale = jnp.where(x_scale == 0, 1.0, x_scale)
        scale_inv = jnp.nan_to_num(
            1 / safe_scale,
            nan=float(dtype_info.max),
            posinf=float(dtype_info.max),
            neginf=-float(dtype_info.max),
        )
        x_q = (x * scale_inv).astype(w_q.dtype)
        acc = jax.lax.dot_general(
            x_q,
            w_q,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.float32)
        acc *= x_scale.astype(jnp.float32)
    else:
        acc = jax.lax.dot_general(
            x,
            w_q,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.float32)
    return (acc * w_scale[None, :].astype(jnp.float32)).astype(x.dtype)


def _xla_per_channel_matmul(
    x: Any,
    w_q: Any,
    w_scale: Any,
    *,
    quantize_activation: bool,
) -> Any:
    """Dependency-light extraction of the production per-channel XLA branch."""

    import jax
    import jax.numpy as jnp

    if quantize_activation:
        dtype_info = jnp.finfo(w_q.dtype)
        x_abs_max = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
        x_scale = x_abs_max / float(dtype_info.max)
        safe_scale = x_scale + (x_scale == 0).astype(x_scale.dtype)
        x_q = jnp.clip(
            x / safe_scale,
            float(dtype_info.min),
            float(dtype_info.max),
        ).astype(w_q.dtype)
        out = jax.lax.dot_general(
            x_q,
            w_q,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out = (
            out.astype(jnp.float32)
            * x_scale.astype(jnp.float32)
            * w_scale[None, :].astype(jnp.float32)
        )
    else:
        out = jax.lax.dot_general(
            x,
            w_q,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out = out.astype(jnp.float32) * w_scale[None, :].astype(jnp.float32)
    return out.astype(x.dtype)


def _make_function(
    case: BenchmarkCase, tuned_value_override: tuple[int, int, int] | None = None
) -> Any:
    import jax

    if case.key.implementation == "xla":

        def operation(x, w_q, w_scale):
            return _xla_per_channel_matmul(
                x,
                w_q,
                w_scale,
                quantize_activation=case.key.mode == "w8a8",
            )

    elif case.key.implementation == "pallas_aligned":
        from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.kernel import (
            quantized_matmul_kernel,
        )
        from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.tuned_block_sizes import (
            TunedValue,
        )

        tuned_value = (
            TunedValue(*tuned_value_override) if tuned_value_override is not None else None
        )

        def operation(x, w_q, w_scale):
            x_q_dtype = w_q.dtype if case.key.mode == "w8a8" else x.dtype
            return quantized_matmul_kernel(
                x=x,
                w_q=w_q,
                w_scale=w_scale,
                x_q_dtype=x_q_dtype,
                tuned_value=tuned_value,
            )

    else:
        raise ValueError(f"unsupported implementation: {case.key.implementation}")

    def run(x, w_q, w_scale):
        with jax.named_scope(f"{TRACE_MARKER}_{case.case_id}"):
            return operation(x, w_q, w_scale)

    return jax.jit(run)


def _correctness_metrics(
    case: BenchmarkCase,
    output: Any,
    reference: Any,
    zero_row: int | None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    output_f32 = output.astype(jnp.float32)
    reference_f32 = reference.astype(jnp.float32)
    error = output_f32 - reference_f32
    denominator = jnp.maximum(jnp.linalg.norm(reference_f32), jnp.asarray(1e-12))
    relative_l2, max_abs, all_finite = jax.device_get(
        (
            jnp.linalg.norm(error) / denominator,
            jnp.max(jnp.abs(error)),
            jnp.all(jnp.isfinite(output_f32)),
        )
    )
    zero_row_exact = None
    if zero_row is not None:
        zero_row_exact = bool(jax.device_get(jnp.all(output[zero_row] == 0)))
    threshold = CORRECTNESS_THRESHOLDS[case.key.mode]
    passed = bool(all_finite) and float(relative_l2) <= threshold
    if zero_row_exact is not None:
        passed = passed and zero_row_exact
    return {
        "all_finite": bool(all_finite),
        "zero_row_index": zero_row,
        "zero_row_exact": zero_row_exact,
        "relative_l2": float(relative_l2),
        "max_abs": float(max_abs),
        "threshold": threshold,
        "passed": passed,
    }


def _next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _kernel_metadata(
    case: BenchmarkCase,
    x: Any,
    w_q: Any,
    tuned_value_override: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    if case.key.implementation != "pallas_aligned":
        return {
            "metadata_source": "logical_shape_only_pending_hlo_inspection",
            "BM": None,
            "BN": None,
            "BK": None,
            "padded_M": case.key.m,
            "padded_N": case.key.n,
            "padded_K": case.key.k,
            "grid": None,
            "n_in": None,
            "n_out": None,
            "save_x_q": False,
            "save_acc": False,
            "acc_scratch_bytes": 0,
            "x_scratch_bytes": 0,
            "vmem_limit_bytes": None,
        }

    import jax.numpy as jnp
    from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels import util
    from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.tuned_block_sizes import (
        TunedValue,
        get_device_vmem_limit,
        get_tuned_block_sizes,
    )

    x_q_dtype = w_q.dtype if case.key.mode == "w8a8" else x.dtype
    if tuned_value_override is None:
        tuned = get_tuned_block_sizes(
            n_batch=case.key.m,
            n_out=case.key.n,
            n_in=case.key.k,
            x_q_dtype=jnp.dtype(x_q_dtype).name,
            w_q_dtype=jnp.dtype(w_q.dtype).name,
        )
        metadata_source = "resolved_pallas_wrapper_configuration"
    else:
        tuned = TunedValue(*tuned_value_override)
        metadata_source = "benchmark_cli_tuned_value_override"
    bm, bn, bk, _ = tuned
    padded_m = _next_multiple(case.key.m, bm)
    padded_n = _next_multiple(case.key.n, bn)
    padded_k = _next_multiple(case.key.k, bk)
    n_batch = padded_m // bm
    n_out = padded_n // bn
    n_in = padded_k // bk
    save_acc = n_in > 1
    save_x_q = case.key.mode == "w8a8" and n_in == 1 and n_out > 1
    vmem_limit = util.get_vmem_limit(
        n_batch=n_batch,
        n_out=n_out,
        n_in=n_in,
        batch_block_size=bm,
        out_block_size=bn,
        in_block_size=bk,
        x_dtype=x.dtype,
        x_q_dtype=x_q_dtype,
        w_q_dtype=w_q.dtype,
        scale_dtype=jnp.float32,
        out_dtype=x.dtype,
        acc_dtype=jnp.float32,
        save_acc=save_acc,
        save_x_q=save_x_q,
        upper_limit_bytes=get_device_vmem_limit(),
        has_x_abs_max=case.key.mode == "w8a8",
    )
    acc_scratch_bytes = bm * bn * jnp.dtype(jnp.float32).itemsize if save_acc else 0
    x_scratch_bytes = 0
    if save_x_q:
        x_scratch_bytes = bm * bk * jnp.dtype(x_q_dtype).itemsize
        x_scratch_bytes += bm * jnp.dtype(jnp.float32).itemsize
    return {
        "metadata_source": metadata_source,
        "BM": bm,
        "BN": bn,
        "BK": bk,
        "padded_M": padded_m,
        "padded_N": padded_n,
        "padded_K": padded_k,
        "grid": [n_batch, n_out, n_in],
        "n_in": n_in,
        "n_out": n_out,
        "save_x_q": save_x_q,
        "save_acc": save_acc,
        "acc_scratch_bytes": acc_scratch_bytes,
        "x_scratch_bytes": x_scratch_bytes,
        "vmem_limit_bytes": vmem_limit,
    }


def _dump_compiler_ir(lowered: Any, case: BenchmarkCase, dump_dir: str | None) -> dict[str, Any]:
    if dump_dir is None:
        return {"stablehlo_path": None, "hlo_path": None}
    output_dir = Path(dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(case.case_id)
    paths = {
        "stablehlo_path": output_dir / f"{stem}.stablehlo.mlir",
        "hlo_path": output_dir / f"{stem}.hlo.txt",
    }
    stablehlo = lowered.compiler_ir(dialect="stablehlo")
    paths["stablehlo_path"].write_text(str(stablehlo), encoding="utf-8")
    hlo = lowered.compiler_ir(dialect="hlo")
    hlo_text = hlo.as_hlo_text() if hasattr(hlo, "as_hlo_text") else str(hlo)
    paths["hlo_path"].write_text(hlo_text, encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _load_trace_events(trace_dir: Path) -> list[dict[str, Any]]:
    profile_root = trace_dir / "plugins" / "profile"
    if not profile_root.exists():
        raise FileNotFoundError(f"no profiler output under {profile_root}")
    profiles = [path for path in profile_root.iterdir() if path.is_dir()]
    if not profiles:
        raise FileNotFoundError(f"no timestamped profile under {profile_root}")
    latest = max(profiles, key=os.path.getmtime)
    trace_files = sorted(latest.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"no trace.json.gz under {latest}")
    events = []
    for trace_file in trace_files:
        with gzip.open(trace_file, "rt", encoding="utf-8") as source:
            payload = json.load(source)
        shard_events = payload.get("traceEvents", [])
        if isinstance(shard_events, list):
            events.extend(shard_events)
    return events


def _event_duration_us(event: dict[str, Any]) -> float | None:
    device_duration_ps = event.get("args", {}).get("device_duration_ps")
    if device_duration_ps is not None:
        return float(device_duration_ps) / 1e6
    if "dur" in event:
        return float(event["dur"])
    return None


def _summarize_device_events(
    events: Sequence[dict[str, Any]],
    active_pid: Any,
    thread_names: dict[tuple[Any, Any], str],
    *,
    top_k: int = 20,
) -> dict[str, Any]:
    """Build a compact, non-additive summary of complete TPU trace events."""
    thread_counts: dict[str, int] = {}
    xla_ops: dict[str, dict[str, float | int | str]] = {}
    complete_event_count = 0
    for event in events:
        if event.get("ph") != "X" or event.get("pid") != active_pid:
            continue
        duration_us = _event_duration_us(event)
        if duration_us is None:
            continue
        complete_event_count += 1
        thread_name = thread_names.get((event.get("pid"), event.get("tid")), "unknown")
        thread_counts[thread_name] = thread_counts.get(thread_name, 0) + 1
        if thread_name != "XLA Ops":
            continue
        tf_op = str(event.get("args", {}).get("tf_op", ""))
        event_name = str(event.get("name", ""))
        key = tf_op or event_name or "unknown"
        aggregate = xla_ops.setdefault(
            key,
            {
                "name": event_name,
                "tf_op": tf_op,
                "count": 0,
                "inclusive_duration_us": 0.0,
                "max_duration_us": 0.0,
            },
        )
        aggregate["count"] = int(aggregate["count"]) + 1
        aggregate["inclusive_duration_us"] = float(aggregate["inclusive_duration_us"]) + float(
            duration_us
        )
        aggregate["max_duration_us"] = max(
            float(aggregate["max_duration_us"]), float(duration_us)
        )

    top_xla_ops = sorted(
        xla_ops.values(),
        key=lambda item: float(item["inclusive_duration_us"]),
        reverse=True,
    )[:top_k]
    return {
        "trace_event_count": len(events),
        "trace_at_known_event_cap": len(events) >= 1_000_000,
        "complete_device_event_count": complete_event_count,
        "thread_event_counts": dict(sorted(thread_counts.items())),
        "top_xla_ops_by_inclusive_duration": top_xla_ops,
        "duration_semantics": "inclusive; entries can overlap and must not be summed as utilization",
    }


def _extract_device_times_us(
    events: Sequence[dict[str, Any]], expected_samples: int
) -> tuple[list[float], dict[str, Any]]:
    process_names = {}
    thread_names = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            process_names[event.get("pid")] = str(event.get("args", {}).get("name", ""))
        if event.get("ph") == "M" and event.get("name") == "thread_name":
            thread_names[(event.get("pid"), event.get("tid"))] = str(
                event.get("args", {}).get("name", "")
            )

    devices = {pid: name for pid, name in process_names.items() if name.startswith("/device:TPU:")}
    if not devices:
        raise RuntimeError(f"no TPU device process in trace: {sorted(process_names.values())}")
    active_pid = min(devices, key=lambda pid: devices[pid])
    marker_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") == active_pid
        and TRACE_MARKER in str(event.get("args", {}).get("tf_op", ""))
        and _event_duration_us(event) is not None
    ]
    call_done = [
        event for event in marker_events if str(event.get("name", "")).endswith("call-done")
    ]
    if call_done:
        marker_events = call_done
    marker_events.sort(key=lambda event: float(event.get("ts", 0.0)))
    durations = [float(_event_duration_us(event)) for event in marker_events]
    method = "marker_call_done" if call_done else "marker"

    if len(durations) != expected_samples:
        modules = [
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("pid") == active_pid
            and thread_names.get((event.get("pid"), event.get("tid"))) == "XLA Modules"
            and _event_duration_us(event) is not None
        ]
        modules.sort(key=lambda event: float(event.get("ts", 0.0)))
        durations = [float(_event_duration_us(event)) for event in modules]
        method = "xla_modules"
    if len(durations) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} device events, found {len(durations)}; method={method}"
        )
    return durations, {
        "event_match_method": method,
        "event_count": len(durations),
        "expected_event_count": expected_samples,
        "active_device_process": devices[active_pid],
        "device_event_summary": _summarize_device_events(events, active_pid, thread_names),
        "valid": True,
    }


def _trace_samples(
    compiled: Any,
    input_ring: Sequence[tuple[Any, Any, Any]],
    case: BenchmarkCase,
    samples: int,
    trace_root: str,
    process_run_id: str,
) -> tuple[list[float], dict[str, Any]]:
    import jax

    unique = f"{case.case_id}_{process_run_id}_{os.getpid()}_{time.time_ns()}"
    trace_dir = Path(trace_root) / "latency" / _safe_filename(unique)
    trace_dir.mkdir(parents=True, exist_ok=False)
    with jax.profiler.trace(str(trace_dir)):
        for sample_id in range(samples):
            inputs = input_ring[sample_id % len(input_ring)]
            with jax.profiler.StepTraceAnnotation(case.case_id, step_num=sample_id):
                jax.block_until_ready(compiled(*inputs))
    try:
        durations, metadata = _extract_device_times_us(_load_trace_events(trace_dir), samples)
        metadata["trace_dir"] = str(trace_dir)
        return durations, metadata
    except RuntimeError as combined_error:
        fallback_root = (
            Path(trace_root) / "latency_per_sample" / _safe_filename(f"{unique}_per_sample")
        )
        fallback_root.mkdir(parents=True, exist_ok=False)
        durations = []
        methods = set()
        devices = set()
        for sample_id in range(samples):
            inputs = input_ring[sample_id % len(input_ring)]
            sample_dir = fallback_root / f"sample-{sample_id:04d}"
            with (
                jax.profiler.trace(str(sample_dir)),
                jax.profiler.StepTraceAnnotation(case.case_id, step_num=sample_id),
            ):
                jax.block_until_ready(compiled(*inputs))
            values, metadata = _extract_device_times_us(_load_trace_events(sample_dir), 1)
            durations.extend(values)
            methods.add(metadata["event_match_method"])
            devices.add(metadata["active_device_process"])
        return durations, {
            "event_match_method": "per_sample_xprof:" + ",".join(sorted(methods)),
            "event_count": len(durations),
            "expected_event_count": samples,
            "active_device_process": ",".join(sorted(devices)),
            "valid": len(durations) == samples,
            "trace_dir": str(fallback_root),
            "combined_trace_dir": str(trace_dir),
            "combined_trace_error": str(combined_error),
        }


def _wall_samples(
    compiled: Any, input_ring: Sequence[tuple[Any, Any, Any]], samples: int
) -> list[float]:
    import jax

    values = []
    for sample_id in range(samples):
        inputs = input_ring[sample_id % len(input_ring)]
        start = time.perf_counter_ns()
        jax.block_until_ready(compiled(*inputs))
        values.append((time.perf_counter_ns() - start) / 1_000.0)
    return values


def _source_metadata(device: Any) -> dict[str, Any]:
    import jax

    try:
        import libtpu

        libtpu_version = getattr(libtpu, "__version__", "unknown")
    except ImportError:
        libtpu_version = None
    return {
        "sglang_jax_commit": os.environ.get("GIT_COMMIT") or _git_value("rev-parse", "HEAD"),
        "sglang_jax_dirty": _git_dirty(),
        "tpu_inference_reference_sha": TPU_INFERENCE_REFERENCE_SHA,
        "jax_version": jax.__version__,
        "libtpu_version": libtpu_version,
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": str(device.device_kind),
        "local_device_count": jax.local_device_count(),
        "hostname": platform.node(),
        "process_pid": os.getpid(),
    }


def _skipped_result(
    case: BenchmarkCase, source: dict[str, Any], args: argparse.Namespace, reason: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "case": {
            **asdict(case.key),
            "case_id": case.case_id,
            "suites": list(case.suites),
            "aliases": [asdict(alias) for alias in case.aliases],
            "tp_mode": "single_device_local_shape_emulation",
            "collective_included": False,
            "weight_ring_count": args.weight_ring_count,
            "process_run_id": args.process_run_id,
        },
        "status": "skipped",
        "skip_reason": reason,
    }


def run_case(
    case: BenchmarkCase, args: argparse.Namespace, device: Any, source: dict[str, Any]
) -> dict[str, Any]:
    import jax

    if device.platform != "tpu" and case.key.implementation == "pallas_aligned":
        return _skipped_result(case, source, args, "Pallas TPU kernel cannot run on non-TPU")

    input_ring, zero_row = _make_input_ring(
        case,
        device,
        args.seed,
        args.weight_ring_count,
    )
    jitted = _make_function(case, args.tuned_value)
    lowering_start = time.perf_counter()
    lowered = jitted.lower(*input_ring[0])
    lowering_ms = (time.perf_counter() - lowering_start) * 1_000.0
    artifacts = _dump_compiler_ir(lowered, case, args.dump_hlo_dir)
    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_ms = (time.perf_counter() - compile_start) * 1_000.0

    first_output = compiled(*input_ring[0])
    tpu_reference = _tpu_inference_reference(case, *input_ring[0])
    xla_reference = _xla_per_channel_matmul(
        *input_ring[0],
        quantize_activation=case.key.mode == "w8a8",
    )
    jax.block_until_ready((first_output, tpu_reference, xla_reference))
    primary_oracle = (
        "tpu_inference_math" if case.key.implementation == "pallas_aligned" else "xla_runtime_math"
    )
    primary_reference = tpu_reference if primary_oracle == "tpu_inference_math" else xla_reference
    correctness = _correctness_metrics(case, first_output, primary_reference, zero_row)
    correctness["primary_oracle"] = primary_oracle
    correctness["against_tpu_inference_math"] = _correctness_metrics(
        case, first_output, tpu_reference, zero_row
    )
    correctness["against_xla_runtime_math"] = _correctness_metrics(
        case, first_output, xla_reference, zero_row
    )
    correctness["oracle_gap_xla_vs_tpu_inference"] = _correctness_metrics(
        case, xla_reference, tpu_reference, zero_row
    )
    if len(input_ring) > 1:
        last_output = compiled(*input_ring[-1])
        if primary_oracle == "tpu_inference_math":
            last_reference = _tpu_inference_reference(case, *input_ring[-1])
        else:
            last_reference = _xla_per_channel_matmul(
                *input_ring[-1],
                quantize_activation=case.key.mode == "w8a8",
            )
        jax.block_until_ready((last_output, last_reference))
        correctness["last_ring"] = _correctness_metrics(case, last_output, last_reference, zero_row)
        correctness["passed"] = correctness["passed"] and correctness["last_ring"]["passed"]
    if not correctness["passed"]:
        raise AssertionError(f"{case.case_id}: correctness failed: {correctness}")

    for warmup_id in range(args.warmup):
        jax.block_until_ready(compiled(*input_ring[warmup_id % len(input_ring)]))

    if device.platform == "tpu":
        device_samples, xprof = _trace_samples(
            compiled,
            input_ring,
            case,
            args.samples,
            args.trace_root,
            args.process_run_id,
        )
        primary_source = "xprof_device_duration_ps"
    else:
        device_samples = _wall_samples(compiled, input_ring, args.samples)
        xprof = {
            "valid": False,
            "event_match_method": None,
            "trace_dir": None,
            "diagnostic_only": True,
        }
        primary_source = "host_wall_non_tpu_smoke_only"
    wall = _wall_samples(compiled, input_ring, args.wall_samples)
    device_summary = summarize_samples(device_samples)
    kernel = _kernel_metadata(case, input_ring[0][0], input_ring[0][1], args.tuned_value)
    semantic_flops = 2 * case.key.m * case.key.n * case.key.k
    padded_flops = None
    padded_weight_bytes = None
    if case.key.implementation == "pallas_aligned":
        padded_flops = 2 * kernel["padded_M"] * kernel["padded_N"] * kernel["padded_K"]
        padded_weight_bytes = (
            kernel["padded_N"] * kernel["padded_K"] * input_ring[0][1].dtype.itemsize
            + kernel["padded_N"] * input_ring[0][2].dtype.itemsize
        )
    logical_weight_bytes = case.key.n * case.key.k * input_ring[0][1].dtype.itemsize
    logical_weight_bytes += case.key.n * input_ring[0][2].dtype.itemsize
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "case": {
            **asdict(case.key),
            "case_id": case.case_id,
            "suites": list(case.suites),
            "aliases": [asdict(alias) for alias in case.aliases],
            "tp_mode": "single_device_local_shape_emulation",
            "collective_included": False,
            "seed": args.seed,
            "weight_ring_count": args.weight_ring_count,
            "process_run_id": args.process_run_id,
        },
        "status": "ok",
        "kernel": kernel,
        "correctness": correctness,
        "timing": {
            "lowering_ms": lowering_ms,
            "compile_ms": compile_ms,
            "primary_source": primary_source,
            "device": device_summary,
            "host_wall_diagnostic": summarize_samples(wall),
            "xprof": xprof,
        },
        "derived": {
            "semantic_flops": semantic_flops,
            "padded_flops_estimate": padded_flops,
            "logical_weight_bytes": logical_weight_bytes,
            "padded_weight_bytes_estimate": padded_weight_bytes,
            "semantic_tflops_p50": semantic_flops / (device_summary["p50_us"] * 1e6),
        },
        "artifacts": {**artifacts, "xprof_trace_dir": xprof.get("trace_dir"), "llo_path": None},
    }
    result.update(
        {
            "operator_family": "quantized_matmul",
            "operator_name": "glm52_per_channel",
            "variant": f"{case.key.implementation}-{case.key.mode}",
            "m": case.key.m,
            "n": case.key.n,
            "k": case.key.k,
            "mode": case.key.mode,
            "implementation": case.key.implementation,
            "ring_size": args.weight_ring_count,
            "latency_us": device_summary["p50_us"],
            "p95_us": device_summary["p95_us"],
            "compile_time_s": compile_ms / 1_000.0,
            "relative_l2": correctness["relative_l2"],
        }
    )
    return result


def _append_jsonl(path: str, result: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(result, sort_keys=True) + "\n")


def _parse_tuned_value(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("tuned value must be BM,BN,BK integers") from error
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("tuned value must contain three positive integers: BM,BN,BK")
    bm, bn, bk = parts
    if bn % 128 or bk % 128:
        raise argparse.ArgumentTypeError("BN and BK must be multiples of TPU sublane size 128")
    return bm, bn, bk


def _apply_tuned_value_variant(
    cases: Sequence[BenchmarkCase], tuned_value: tuple[int, int, int] | None
) -> list[BenchmarkCase]:
    if tuned_value is None:
        return list(cases)
    bm, bn, bk = tuned_value
    variant = f"tuned_bm{bm}_bn{bn}_bk{bk}"
    return [replace(case, key=replace(case.key, variant=variant)) for case in cases]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "anchor", "full", "all", "profiles"),
        default="smoke",
    )
    parser.add_argument("--operations", nargs="+")
    parser.add_argument("--tp-degree", type=int, nargs="+")
    parser.add_argument("--m", type=int, nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=IMPLEMENTATIONS,
        default=list(IMPLEMENTATIONS),
    )
    parser.add_argument("--weight-ring-count", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--wall-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--process-run-id", default=f"pid-{os.getpid()}")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--trace-root", default="/tmp/glm52_per_channel_qmm_xprof")
    parser.add_argument("--dump-hlo-dir")
    parser.add_argument("--output-jsonl")
    parser.add_argument(
        "--tuned-value",
        type=_parse_tuned_value,
        metavar="BM,BN,BK",
        help="benchmark-only Pallas tile override; never mutates the production tuning table",
    )
    parser.add_argument("--allow-non-tpu", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.weight_ring_count <= 0:
        parser.error("weight-ring-count must be positive")
    if args.warmup < 0 or args.samples <= 0 or args.wall_samples <= 0:
        parser.error("warmup must be non-negative and sample counts must be positive")
    try:
        cases = _select_cases(args)
    except ValueError as error:
        parser.error(str(error))
    if not cases:
        parser.error("case filters selected no cases")
    if args.tuned_value is not None and any(
        case.key.implementation != "pallas_aligned" for case in cases
    ):
        parser.error("--tuned-value requires --implementations pallas_aligned")
    cases = _apply_tuned_value_variant(cases, args.tuned_value)
    if args.list_cases:
        _print_cases(cases)
        return 0

    device = _validate_device(args)
    source = _source_metadata(device)
    print(
        f"branch={_git_value('branch', '--show-current')} commit={source['sglang_jax_commit']} "
        f"device={source['device_kind']} cases={len(cases)} ring={args.weight_ring_count} "
        f"samples={args.samples} run_id={args.process_run_id}"
    )
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        result = run_case(case, args, device, source)
        results.append(result)
        if args.output_jsonl:
            _append_jsonl(args.output_jsonl, result)
        if result["status"] == "ok":
            timing = result["timing"]["device"]
            print(
                f"  p50={timing['p50_us']:.3f}us p95={timing['p95_us']:.3f}us "
                f"cv={timing['cv']:.3%} rel_l2={result['correctness']['relative_l2']:.3e}",
                flush=True,
            )
        else:
            print(f"  skipped: {result['skip_reason']}", flush=True)
        gc.collect()

    if not args.output_jsonl:
        print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
