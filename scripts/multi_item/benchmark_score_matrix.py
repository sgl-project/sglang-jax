#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a reproducible multi-item scoring benchmark matrix with gate checks.

This script is designed for ranking workloads using score-from-cache v2 on
TPU/GPU/CPU backends and records:
  - Single-request throughput (items/s) and latency percentiles.
  - Concurrent request QPS plus request/item throughput.
  - P50/P95/P99 request latency.
  - Host-vs-device split totals from scheduler score timing counters.
  - Scoring cache query/hit/miss counters.
  - Ingress batch integrity counters and per-path messages/frame ratios.

It also includes targeted probes to verify tokenizer->ZMQ->scheduler batch
integrity across all scoring paths (packed, prefill+extend baseline, and
prefill+extend fastpath).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import jax

from sgl_jax.srt.entrypoints.engine import Engine


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError(f"Expected at least one integer in {raw!r}")
    return out


def clone_num_dict(data: dict | None, keys: list[str]) -> dict[str, float]:
    data = data or {}
    out: dict[str, float] = {}
    for key in keys:
        value = data.get(key, 0.0)
        out[key] = float(value) if value is not None else 0.0
    return out


def diff_num_dict(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, after_value in after.items():
        before_value = before.get(key, 0.0)
        out[key] = float(after_value) - float(before_value)
    return out


def make_tokens(query_len: int, num_items: int, item_len: int) -> tuple[list[int], list[list[int]]]:
    query = [1] * query_len
    items = [[2] * item_len for _ in range(num_items)]
    return query, items


@dataclass
class Snapshot:
    score_metrics: dict[str, float]
    score_timing_totals_s: dict[str, float]
    cache_metrics: dict[str, float]
    ingress_metrics: dict[str, float]
    score_path_messages: dict[str, float]
    score_path_frames: dict[str, float]


def snapshot_counters(engine: Engine) -> Snapshot:
    state = engine.get_server_info().get("internal_states", [{}])[0]
    score = state.get("score_from_cache_v2_metrics", {}) or {}
    score_timing = score.get("timing_totals_s", {}) or {}
    cache = state.get("scoring_cache_metrics", {}) or {}
    ingress = state.get("ingress_metrics", {}) or {}
    path_messages = ingress.get("score_path_messages", {}) or {}
    path_frames = ingress.get("score_path_frames", {}) or {}
    return Snapshot(
        score_metrics=clone_num_dict(score, ["attempted", "succeeded", "fallback"]),
        score_timing_totals_s=clone_num_dict(
            score_timing,
            ["queue_wait", "device_compute", "host_orchestration"],
        ),
        cache_metrics=clone_num_dict(cache, ["lookup_queries", "lookup_hits", "lookup_misses"]),
        ingress_metrics=clone_num_dict(
            ingress,
            ["tokenizer_frames", "tokenizer_messages", "rpc_frames", "rpc_messages"],
        ),
        score_path_messages={
            key: float(value) for key, value in path_messages.items() if value is not None
        },
        score_path_frames={
            key: float(value) for key, value in path_frames.items() if value is not None
        },
    )


def diff_snapshot(after: Snapshot, before: Snapshot) -> dict:
    score_delta = diff_num_dict(after.score_metrics, before.score_metrics)
    timing_delta = diff_num_dict(after.score_timing_totals_s, before.score_timing_totals_s)
    cache_delta = diff_num_dict(after.cache_metrics, before.cache_metrics)
    ingress_delta = diff_num_dict(after.ingress_metrics, before.ingress_metrics)
    all_path_keys = sorted(set(after.score_path_messages) | set(before.score_path_messages))
    path_msg_delta: dict[str, float] = {}
    path_frame_delta: dict[str, float] = {}
    path_msg_per_frame: dict[str, float] = {}
    for key in all_path_keys:
        msg = float(after.score_path_messages.get(key, 0.0)) - float(
            before.score_path_messages.get(key, 0.0)
        )
        frames = float(after.score_path_frames.get(key, 0.0)) - float(
            before.score_path_frames.get(key, 0.0)
        )
        path_msg_delta[key] = msg
        path_frame_delta[key] = frames
        path_msg_per_frame[key] = (msg / frames) if frames > 0 else 0.0

    attempted = score_delta.get("attempted", 0.0)
    split_per_request = {
        "queue_wait": (timing_delta["queue_wait"] / attempted) if attempted > 0 else 0.0,
        "device_compute": (timing_delta["device_compute"] / attempted) if attempted > 0 else 0.0,
        "host_orchestration": (
            timing_delta["host_orchestration"] / attempted if attempted > 0 else 0.0
        ),
    }

    return {
        "score_from_cache_v2_delta": score_delta,
        "score_timing_totals_s_delta": timing_delta,
        "score_timing_per_request_s": split_per_request,
        "scoring_cache_delta": cache_delta,
        "ingress_delta": ingress_delta,
        "ingress_score_path_messages_delta": path_msg_delta,
        "ingress_score_path_frames_delta": path_frame_delta,
        "ingress_score_path_messages_per_frame_delta": path_msg_per_frame,
        "ingress_tokenizer_messages_per_frame_delta": (
            ingress_delta["tokenizer_messages"] / ingress_delta["tokenizer_frames"]
            if ingress_delta["tokenizer_frames"] > 0
            else 0.0
        ),
        "ingress_rpc_messages_per_frame_delta": (
            ingress_delta["rpc_messages"] / ingress_delta["rpc_frames"]
            if ingress_delta["rpc_frames"] > 0
            else 0.0
        ),
    }


def build_engine(
    *,
    args: argparse.Namespace,
    mode: str,
    fastpath_v2: bool,
) -> Engine:
    model = args.model_path
    if model is None:
        model = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")

    download_dir = args.download_dir
    if not download_dir:
        download_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    os.makedirs(download_dir, exist_ok=True)

    common = dict(
        model_path=model,
        trust_remote_code=True,
        tp_size=args.tp_size,
        device=args.device,
        random_seed=args.seed,
        node_rank=0,
        mem_fraction_static=args.mem_fraction_static,
        max_prefill_tokens=32768,
        chunked_prefill_size=-1,
        download_dir=download_dir,
        dtype="bfloat16",
        precompile_bs_paddings=[args.max_running_requests],
        max_running_requests=args.max_running_requests,
        skip_server_warmup=True,
        attention_backend="fa",
        precompile_token_paddings=[1024, 4096, 16384],
        page_size=64,
        log_requests=False,
        log_level=args.log_level,
        disable_overlap_schedule=True,
        enable_tokenizer_batch_encode=bool(args.enable_tokenizer_batch_encode),
        enable_tokenizer_batch_send=bool(args.enable_tokenizer_batch_send),
    )

    if mode == "prefill_extend":
        common.update(
            dict(
                disable_radix_cache=False,
                enable_scoring_cache=True,
                multi_item_enable_prefill_extend=True,
                multi_item_extend_batch_size=args.extend_batch_size,
                multi_item_enable_score_from_cache_v2=bool(fastpath_v2),
                multi_item_score_from_cache_v2_items_per_step=args.items_per_step,
                multi_item_score_fastpath_log_metrics=bool(args.log_score_path_metrics),
                multi_item_score_label_only_logprob=(
                    bool(args.label_only_logprob) and bool(fastpath_v2)
                ),
            )
        )
    elif mode == "packed":
        common.update(
            dict(
                multi_item_scoring_delimiter=args.multi_item_delimiter,
                disable_radix_cache=True,
                enable_scoring_cache=False,
                max_multi_item_seq_len=32768,
                multi_item_mask_impl="dense",
                multi_item_segment_fallback_threshold=0,
                multi_item_enable_prefill_extend=False,
                multi_item_enable_score_from_cache_v2=False,
            )
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return Engine(**common)


def summarize_latency(latencies_s: list[float]) -> dict:
    mean_s = statistics.mean(latencies_s)
    stdev_s = statistics.pstdev(latencies_s) if len(latencies_s) > 1 else 0.0
    return {
        "count": len(latencies_s),
        "mean_s": mean_s,
        "stdev_s": stdev_s,
        "p50_s": percentile(latencies_s, 50),
        "p95_s": percentile(latencies_s, 95),
        "p99_s": percentile(latencies_s, 99),
        "min_s": min(latencies_s),
        "max_s": max(latencies_s),
        "all_s": latencies_s,
    }


def run_single_request_lane(
    *,
    engine: Engine,
    query_tokens: list[int],
    items: list[list[int]],
    label_token_ids: list[int],
    warmup_runs: int,
    timed_runs: int,
) -> dict:
    for _ in range(warmup_runs):
        out = engine.score(query=query_tokens, items=items, label_token_ids=label_token_ids)
        if len(out) != len(items):
            raise RuntimeError(
                f"Unexpected score row count during warmup: {len(out)} != {len(items)}"
            )

    before = snapshot_counters(engine)
    latencies_s: list[float] = []
    for _ in range(timed_runs):
        start = time.perf_counter()
        out = engine.score(query=query_tokens, items=items, label_token_ids=label_token_ids)
        elapsed = max(0.0, time.perf_counter() - start)
        if len(out) != len(items):
            raise RuntimeError(f"Unexpected score row count: {len(out)} != {len(items)}")
        latencies_s.append(elapsed)
    after = snapshot_counters(engine)

    latency = summarize_latency(latencies_s)
    per_run_items_s = [(len(items) / x) if x > 0 else 0.0 for x in latencies_s]
    return {
        "latency_s": latency,
        "throughput_items_s": {
            "mean": statistics.mean(per_run_items_s),
            "p50": percentile(per_run_items_s, 50),
            "p95_worst": len(items) / latency["p95_s"] if latency["p95_s"] > 0 else 0.0,
            "p99_worst": len(items) / latency["p99_s"] if latency["p99_s"] > 0 else 0.0,
        },
        "counter_delta": diff_snapshot(after, before),
    }


async def _run_concurrent_round(
    *,
    engine: Engine,
    query_tokens: list[int],
    items: list[list[int]],
    label_token_ids: list[int],
    concurrency: int,
    requests_per_worker: int,
) -> tuple[list[float], float]:
    latencies_s: list[float] = []

    async def worker() -> None:
        for _ in range(requests_per_worker):
            start = time.perf_counter()
            out = await engine.async_score(
                query=query_tokens, items=items, label_token_ids=label_token_ids
            )
            elapsed = max(0.0, time.perf_counter() - start)
            if len(out) != len(items):
                raise RuntimeError(
                    f"Unexpected concurrent score row count: {len(out)} != {len(items)}"
                )
            latencies_s.append(elapsed)

    start_wall = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    total_wall_s = max(0.0, time.perf_counter() - start_wall)
    return latencies_s, total_wall_s


def run_concurrent_lane(
    *,
    engine: Engine,
    query_tokens: list[int],
    items: list[list[int]],
    label_token_ids: list[int],
    concurrency: int,
    requests_per_worker: int,
) -> dict:
    before = snapshot_counters(engine)
    latencies_s, wall_s = engine.loop.run_until_complete(
        _run_concurrent_round(
            engine=engine,
            query_tokens=query_tokens,
            items=items,
            label_token_ids=label_token_ids,
            concurrency=concurrency,
            requests_per_worker=requests_per_worker,
        )
    )
    after = snapshot_counters(engine)
    total_requests = len(latencies_s)
    total_items = total_requests * len(items)
    qps = (total_requests / wall_s) if wall_s > 0 else 0.0
    items_per_sec = (total_items / wall_s) if wall_s > 0 else 0.0
    return {
        "concurrency": concurrency,
        "requests_per_worker": requests_per_worker,
        "total_requests": total_requests,
        "total_items": total_items,
        "wall_s": wall_s,
        "qps": qps,
        "throughput_items_s": items_per_sec,
        "latency_s": summarize_latency(latencies_s),
        "counter_delta": diff_snapshot(after, before),
    }


def run_score_path_probe(
    *,
    args: argparse.Namespace,
    probe_name: str,
    mode: str,
    fastpath_v2: bool,
    expected_paths: list[str],
) -> dict:
    query, items = make_tokens(
        args.path_probe_query_len, args.path_probe_items, args.path_probe_item_len
    )
    label_token_ids = [int(x) for x in args.label_token_ids]
    engine = None
    try:
        engine = build_engine(args=args, mode=mode, fastpath_v2=fastpath_v2)
        before = snapshot_counters(engine)
        out = engine.score(query=query, items=items, label_token_ids=label_token_ids)
        if len(out) != len(items):
            raise RuntimeError(
                f"Probe {probe_name} got wrong score count: {len(out)} != {len(items)}"
            )
        after = snapshot_counters(engine)
        delta = diff_snapshot(after, before)
        messages = delta["ingress_score_path_messages_delta"]
        frames = delta["ingress_score_path_frames_delta"]
        msg_per_frame = delta["ingress_score_path_messages_per_frame_delta"]

        missing_paths: list[str] = []
        for path in expected_paths:
            if messages.get(path, 0.0) <= 0.0 or frames.get(path, 0.0) <= 0.0:
                missing_paths.append(path)

        return {
            "probe_name": probe_name,
            "mode": mode,
            "fastpath_v2": fastpath_v2,
            "expected_paths": expected_paths,
            "missing_paths": missing_paths,
            "pass": len(missing_paths) == 0,
            "ingress_score_path_messages_delta": messages,
            "ingress_score_path_frames_delta": frames,
            "ingress_score_path_messages_per_frame_delta": msg_per_frame,
        }
    finally:
        if engine is not None:
            engine.shutdown()
        jax.clear_caches()


def evaluate_gates(
    *,
    results: dict,
    max_fallback_rate: float,
    min_path_messages_per_frame: float,
    max_p99_ms: float | None,
) -> dict:
    failures: list[str] = []

    single_delta = results["single_request"]["counter_delta"]["score_from_cache_v2_delta"]
    single_attempted = single_delta.get("attempted", 0.0)
    single_fallback = single_delta.get("fallback", 0.0)
    single_fallback_rate = (single_fallback / single_attempted) if single_attempted > 0 else 0.0
    if single_fallback_rate > max_fallback_rate:
        failures.append(
            f"single_request fallback_rate={single_fallback_rate:.6f} exceeds {max_fallback_rate:.6f}"
        )

    for row in results["concurrent"]:
        delta = row["counter_delta"]["score_from_cache_v2_delta"]
        attempted = delta.get("attempted", 0.0)
        fallback = delta.get("fallback", 0.0)
        fallback_rate = (fallback / attempted) if attempted > 0 else 0.0
        if fallback_rate > max_fallback_rate:
            failures.append(
                "concurrent "
                f"(c={row['concurrency']}) fallback_rate={fallback_rate:.6f} exceeds "
                f"{max_fallback_rate:.6f}"
            )

        if max_p99_ms is not None:
            p99_ms = row["latency_s"]["p99_s"] * 1000.0
            if p99_ms > max_p99_ms:
                failures.append(
                    f"concurrent (c={row['concurrency']}) p99_ms={p99_ms:.3f} exceeds {max_p99_ms:.3f}"
                )

    for probe in results["path_integrity_probes"]:
        if not probe["pass"]:
            failures.append(
                f"path_probe {probe['probe_name']} missing_paths={','.join(probe['missing_paths'])}"
            )
            continue
        for path_name in probe["expected_paths"]:
            ratio = probe["ingress_score_path_messages_per_frame_delta"].get(path_name, 0.0)
            if ratio < min_path_messages_per_frame:
                failures.append(
                    "path_probe "
                    f"{probe['probe_name']} path={path_name} messages_per_frame={ratio:.3f} "
                    f"below {min_path_messages_per_frame:.3f}"
                )

    return {
        "pass": len(failures) == 0,
        "failures": failures,
        "constraints": {
            "max_fallback_rate": max_fallback_rate,
            "min_path_messages_per_frame": min_path_messages_per_frame,
            "max_p99_ms": max_p99_ms,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="tpu")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--max-running-requests", type=int, default=24)
    parser.add_argument("--items-per-step", type=int, default=48)
    parser.add_argument("--extend-batch-size", type=int, default=64)
    parser.add_argument("--multi-item-delimiter", type=int, default=128001)
    parser.set_defaults(label_only_logprob=True)
    parser.add_argument(
        "--label-only-logprob",
        dest="label_only_logprob",
        action="store_true",
        help="Enable label-only logprob scoring in score-from-cache v2 (default: enabled).",
    )
    parser.add_argument(
        "--no-label-only-logprob",
        dest="label_only_logprob",
        action="store_false",
        help="Disable label-only logprob scoring in score-from-cache v2.",
    )
    parser.add_argument("--enable-tokenizer-batch-encode", action="store_true")
    parser.add_argument("--enable-tokenizer-batch-send", action="store_true")
    parser.add_argument("--log-score-path-metrics", action="store_true")

    parser.add_argument("--query-len", type=int, default=2000)
    parser.add_argument("--num-items", type=int, default=500)
    parser.add_argument("--item-len", type=int, default=20)
    parser.add_argument("--label-token-ids", type=parse_int_list, default=[198])
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=12)
    parser.add_argument("--concurrency-levels", type=parse_int_list, default=[1, 4, 8])
    parser.add_argument("--requests-per-worker", type=int, default=8)

    parser.add_argument("--path-probe-query-len", type=int, default=128)
    parser.add_argument("--path-probe-items", type=int, default=16)
    parser.add_argument("--path-probe-item-len", type=int, default=16)
    parser.add_argument("--skip-path-probes", action="store_true")

    parser.add_argument("--max-fallback-rate", type=float, default=0.0)
    parser.add_argument("--min-path-messages-per-frame", type=float, default=1.0)
    parser.add_argument("--max-p99-ms", type=float, default=None)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_tokens, items = make_tokens(args.query_len, args.num_items, args.item_len)
    label_token_ids = [int(x) for x in args.label_token_ids]

    engine = None
    try:
        engine = build_engine(args=args, mode="prefill_extend", fastpath_v2=True)
        single = run_single_request_lane(
            engine=engine,
            query_tokens=query_tokens,
            items=items,
            label_token_ids=label_token_ids,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )

        concurrent_rows: list[dict] = []
        for concurrency in args.concurrency_levels:
            row = run_concurrent_lane(
                engine=engine,
                query_tokens=query_tokens,
                items=items,
                label_token_ids=label_token_ids,
                concurrency=concurrency,
                requests_per_worker=args.requests_per_worker,
            )
            concurrent_rows.append(row)
    finally:
        if engine is not None:
            engine.shutdown()
        jax.clear_caches()

    probes: list[dict] = []
    if not args.skip_path_probes:
        probes.append(
            run_score_path_probe(
                args=args,
                probe_name="packed",
                mode="packed",
                fastpath_v2=False,
                expected_paths=["tokenizer_multi_item_packed"],
            )
        )
        probes.append(
            run_score_path_probe(
                args=args,
                probe_name="prefill_extend_baseline",
                mode="prefill_extend",
                fastpath_v2=False,
                expected_paths=[
                    "tokenizer_cache_for_scoring",
                    "tokenizer_extend_from_cache",
                    "rpc_release_scoring_cache",
                ],
            )
        )
        probes.append(
            run_score_path_probe(
                args=args,
                probe_name="prefill_extend_fastpath_v2",
                mode="prefill_extend",
                fastpath_v2=True,
                expected_paths=[
                    "tokenizer_cache_for_scoring",
                    "rpc_score_from_cache_v2",
                    "rpc_release_scoring_cache",
                ],
            )
        )

    results = {
        "config": {
            "model_path": args.model_path or os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B"),
            "device": args.device,
            "tp_size": args.tp_size,
            "seed": args.seed,
            "max_running_requests": args.max_running_requests,
            "items_per_step": args.items_per_step,
            "extend_batch_size": args.extend_batch_size,
            "query_len": args.query_len,
            "num_items": args.num_items,
            "item_len": args.item_len,
            "label_token_ids": label_token_ids,
            "warmup_runs": args.warmup_runs,
            "timed_runs": args.timed_runs,
            "concurrency_levels": args.concurrency_levels,
            "requests_per_worker": args.requests_per_worker,
            "enable_tokenizer_batch_encode": bool(args.enable_tokenizer_batch_encode),
            "enable_tokenizer_batch_send": bool(args.enable_tokenizer_batch_send),
            "label_only_logprob": bool(args.label_only_logprob),
        },
        "single_request": single,
        "concurrent": concurrent_rows,
        "path_integrity_probes": probes,
    }
    results["gates"] = evaluate_gates(
        results=results,
        max_fallback_rate=args.max_fallback_rate,
        min_path_messages_per_frame=args.min_path_messages_per_frame,
        max_p99_ms=args.max_p99_ms,
    )

    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote benchmark matrix JSON to {output_path}")

    if not results["gates"]["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
