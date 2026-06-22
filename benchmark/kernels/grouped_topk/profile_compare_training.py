"""Profile grouped_topk v1 against the training-safe ids+gather wrapper.

This benchmark writes one JSON line per (T, variant). Timings are extracted from JAX profiler
trace events, not host wall time, when running on TPU. CPU/interpret smoke tests may opt into host
fallback because CPU traces do not always carry device-duration events.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import statistics
import time
from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.grouped_topk.grouped_topk import grouped_topk_pallas
from sgl_jax.srt.kernels.grouped_topk.topk_v1_training import (
    grouped_topk_pallas_training,
)

SCOPE_V1_FUSED = "bench_grouped_topk_v1_fused"
SCOPE_V1_TRAINING_GATHER = "bench_grouped_topk_v1_training_gather"


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _logits(tokens: int, experts: int, seed: int) -> jax.Array:
    key = jax.random.PRNGKey(seed)
    return jax.nn.sigmoid(jax.random.normal(key, (tokens, experts), dtype=jnp.float32))


def _latest_trace_events(trace_root: pathlib.Path) -> list[dict[str, Any]]:
    profile_root = trace_root / "plugins" / "profile"
    if not profile_root.exists():
        return []

    profile_dirs = [path for path in profile_root.iterdir() if path.is_dir()]
    if not profile_dirs:
        return []
    latest = max(profile_dirs, key=os.path.getmtime)

    events: list[dict[str, Any]] = []
    for trace_file in sorted(latest.glob("*.trace.json.gz")):
        with gzip.open(trace_file, "rb") as f:
            trace = json.load(f)
        shard_events = trace.get("traceEvents", [])
        if isinstance(shard_events, list):
            events.extend(shard_events)
    return events


def _metadata_maps(
    events: Iterable[dict[str, Any]],
) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    process_names: dict[int, str] = {}
    thread_names: dict[tuple[int, int], str] = {}
    for event in events:
        if event.get("ph") != "M":
            continue
        args = event.get("args", {})
        name = event.get("name")
        if name == "process_name" and isinstance(event.get("pid"), int):
            process_names[event["pid"]] = args.get("name", "")
        elif (
            name == "thread_name"
            and isinstance(event.get("pid"), int)
            and isinstance(event.get("tid"), int)
        ):
            thread_names[(event["pid"], event["tid"])] = args.get("name", "")
    return process_names, thread_names


def _duration_ms(event: dict[str, Any]) -> float | None:
    args = event.get("args", {})
    if args.get("device_duration_ps"):
        return float(args["device_duration_ps"]) / 1e9
    if "dur" in event:
        return float(event["dur"]) / 1e3
    return None


def _xla_module_durations_ms(events: list[dict[str, Any]]) -> list[float]:
    process_names, thread_names = _metadata_maps(events)
    durations = []
    for event in events:
        if event.get("ph") != "X":
            continue
        pid = event.get("pid")
        tid = event.get("tid")
        if process_names.get(pid) != "/device:TPU:0":
            continue
        if thread_names.get((pid, tid)) != "XLA Modules":
            continue
        duration = _duration_ms(event)
        if duration is not None:
            durations.append(duration)
    return durations


def _scope_durations_ms(events: list[dict[str, Any]], scope: str) -> list[float]:
    durations = []
    for event in events:
        args = event.get("args", {})
        searchable = " ".join(
            str(value)
            for value in (
                event.get("name", ""),
                args.get("tf_op", ""),
                args.get("hlo_op", ""),
                args.get("long_name", ""),
            )
        )
        if scope not in searchable:
            continue
        duration = _duration_ms(event)
        if duration is not None:
            durations.append(duration)
    return durations


def _host_profile_ms(run_fn, scope: str, iters: int) -> list[float]:
    samples = []
    for i in range(iters):
        start = time.perf_counter()
        with jax.profiler.StepTraceAnnotation(scope, step_num=i), jax.named_scope(scope):
            jax.block_until_ready(run_fn())
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def _trace_profile_ms(
    run_fn,
    *,
    scope: str,
    trace_root: pathlib.Path,
    warmup: int,
    iters: int,
    allow_host_fallback: bool,
) -> tuple[list[float], str, pathlib.Path]:
    for _ in range(warmup):
        jax.block_until_ready(run_fn())

    run_trace_root = trace_root / f"{scope}_{os.getpid()}_{int(time.time() * 1000)}"
    run_trace_root.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(run_trace_root)):
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation(scope, step_num=i), jax.named_scope(scope):
                jax.block_until_ready(run_fn())

    events = _latest_trace_events(run_trace_root)
    samples = _xla_module_durations_ms(events) or _scope_durations_ms(events, scope)
    if samples:
        return samples, "trace", run_trace_root
    if allow_host_fallback:
        return _host_profile_ms(run_fn, scope, iters), "host_fallback", run_trace_root
    raise RuntimeError(f"No trace durations found for scope={scope} under {run_trace_root}")


def _summary_row(
    *,
    tokens: int,
    experts: int,
    n_group: int,
    topk_group: int,
    topk: int,
    variant: str,
    scope: str,
    samples_ms: list[float],
    timing_source: str,
    trace_path: pathlib.Path,
) -> dict[str, Any]:
    sorted_samples = sorted(samples_ms)
    p90_idx = min(len(sorted_samples) - 1, int(0.9 * (len(sorted_samples) - 1)))
    return {
        "T": tokens,
        "E": experts,
        "G": n_group,
        "Gtop": topk_group,
        "topk": topk,
        "variant": variant,
        "scope": scope,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p90_ms": sorted_samples[p90_idx],
        "num_samples": len(samples_ms),
        "samples_ms": samples_ms,
        "timing_source": timing_source,
        "trace_path": str(trace_path),
    }


def _make_jitted_variants(
    *,
    n_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str,
    interpret: bool,
):
    def v1_fused(logits, bias):
        with jax.named_scope(SCOPE_V1_FUSED):
            return grouped_topk_pallas(
                logits,
                bias,
                num_expert_group=n_group,
                topk_group=topk_group,
                topk=topk,
                block_tokens=block_tokens,
                interpret=interpret,
            )

    def v1_training_gather(logits, bias):
        with jax.named_scope(SCOPE_V1_TRAINING_GATHER):
            return grouped_topk_pallas_training(
                logits,
                bias,
                num_expert_group=n_group,
                topk_group=topk_group,
                topk=topk,
                block_tokens=block_tokens,
                interpret=interpret,
            )

    return {
        "v1_fused": (SCOPE_V1_FUSED, jax.jit(v1_fused)),
        "v1_training_gather": (SCOPE_V1_TRAINING_GATHER, jax.jit(v1_training_gather)),
    }


def run_comparison(
    *,
    token_sizes: list[int],
    e: int,
    n_group: int,
    topk_group: int,
    topk: int,
    output_path: str | os.PathLike,
    trace_root: str | os.PathLike,
    warmup: int,
    iters: int,
    block_tokens: int | str = "auto",
    interpret: bool = False,
    allow_host_fallback: bool = True,
) -> list[dict[str, Any]]:
    output_path = pathlib.Path(output_path)
    trace_root = pathlib.Path(trace_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with output_path.open("w") as out:
        for tokens in token_sizes:
            logits = jax.device_put(_logits(tokens, e, seed=tokens + e + topk))
            bias = jax.device_put(
                jax.random.normal(jax.random.PRNGKey(e), (e,), dtype=jnp.float32) * 0.1
            )
            variants = _make_jitted_variants(
                n_group=n_group,
                topk_group=topk_group,
                topk=topk,
                block_tokens=block_tokens,
                interpret=interpret,
            )

            fused_weights, fused_ids = variants["v1_fused"][1](logits, bias)
            training_weights, training_ids = variants["v1_training_gather"][1](logits, bias)
            jax.block_until_ready((fused_weights, fused_ids, training_weights, training_ids))
            if not bool(jnp.array_equal(fused_ids, training_ids)):
                raise AssertionError(f"top-k ids differ for T={tokens}")
            if not bool(jnp.allclose(fused_weights, training_weights, rtol=0, atol=1e-6)):
                raise AssertionError(f"top-k weights differ for T={tokens}")

            for variant, (scope, fn) in variants.items():
                samples_ms, timing_source, variant_trace = _trace_profile_ms(
                    lambda fn=fn: fn(logits, bias),
                    scope=scope,
                    trace_root=trace_root,
                    warmup=warmup,
                    iters=iters,
                    allow_host_fallback=allow_host_fallback,
                )
                row = _summary_row(
                    tokens=tokens,
                    experts=e,
                    n_group=n_group,
                    topk_group=topk_group,
                    topk=topk,
                    variant=variant,
                    scope=scope,
                    samples_ms=samples_ms,
                    timing_source=timing_source,
                    trace_path=variant_trace,
                )
                rows.append(row)
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
                print(
                    f"T={tokens:>5} variant={variant:<18} "
                    f"median={row['median_ms'] * 1e3:>9.2f}us "
                    f"mean={row['mean_ms'] * 1e3:>9.2f}us "
                    f"samples={row['num_samples']:>2} source={timing_source}"
                )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--E", type=int, default=256)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--Gtop", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--block-tokens", default="auto")
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--allow-host-fallback", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.environ.get("OUT", "/tmp/grouped_topk_profile"), "benchmark", "metrics.jsonl"
        ),
    )
    parser.add_argument(
        "--trace-root",
        default=os.path.join(
            os.environ.get("OUT", "/tmp/grouped_topk_profile"), "profiling", "xprof"
        ),
    )
    args = parser.parse_args()

    block_tokens: int | str
    if args.block_tokens == "auto":
        block_tokens = "auto"
    else:
        block_tokens = int(args.block_tokens)

    print(f"JAX {jax.__version__} | device={jax.devices()[0]} | n_dev={len(jax.devices())}")
    try:
        import libtpu

        print(f"libtpu {libtpu.__version__}")
    except Exception:
        pass

    run_comparison(
        token_sizes=_parse_csv_ints(args.T),
        e=args.E,
        n_group=args.G,
        topk_group=args.Gtop,
        topk=args.topk,
        output_path=args.output,
        trace_root=args.trace_root,
        warmup=args.warmup,
        iters=args.iters,
        block_tokens=block_tokens,
        interpret=args.interpret,
        allow_host_fallback=args.allow_host_fallback,
    )


if __name__ == "__main__":
    main()
