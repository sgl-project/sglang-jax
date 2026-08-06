"""Grouped GLM-5.2 DSA cache-hit extend/decode benchmark.

The workload deliberately uses one native batch so DP admission sees all
requests together. It can build either independent prefixes or one shared
prefix per DP rank, then measures ``prefix hit + extend + decode`` and rejects
cache-miss or partial results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import requests


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def _make_inputs(
    concurrency: int,
    prefix_len: int,
    extend_len: int,
    *,
    prefix_mode: str,
    random_seed: int = 3,
    random_token_min: int = 1000,
    random_token_max: int = 32000,
) -> tuple[list, list]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if prefix_len < 2:
        raise ValueError("prefix_len must be at least 2")
    if extend_len < 1:
        raise ValueError("extend_len must be positive")
    if prefix_mode not in ("independent", "shared"):
        raise ValueError(f"unknown prefix_mode: {prefix_mode}")
    if random_token_min < 0 or random_token_max <= random_token_min:
        raise ValueError("random token range must be non-negative and non-empty")
    if random_token_max - random_token_min < 2 * concurrency:
        raise ValueError("random token range must contain at least 2 * concurrency IDs")

    rng = random.Random(random_seed)

    def random_tokens(length: int) -> list[int]:
        return [
            rng.randrange(random_token_min, random_token_max) for _ in range(length)
        ]

    prefixes = []
    extended = []
    shared_prefix = random_tokens(prefix_len)
    for request_id in range(concurrency):
        if prefix_mode == "shared":
            prefix = shared_prefix.copy()
        else:
            prefix = random_tokens(prefix_len)
            # Force distinct request heads so radix sharing cannot turn independent
            # prefix capacity into a shared-prefix case.
            prefix[0] = random_token_min + concurrency + request_id

        extension = random_tokens(extend_len)
        # Force a distinct branch immediately after the cached prefix. The remainder
        # stays random so router/top-k and fused MoE see varied token representations.
        extension[0] = random_token_min + request_id
        prefixes.append(prefix)
        extended.append(prefix + extension)
    return prefixes, extended


def _run_native_batch(
    base_url: str,
    input_ids: list[list[int]],
    output_len: int,
    *,
    label: str,
) -> dict:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json={
            "rid": [f"{label}-{i}" for i in range(len(input_ids))],
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": output_len,
                "min_new_tokens": output_len,
                "ignore_eos": True,
                "stream_interval": 1,
            },
            "stream": True,
        },
        stream=True,
        timeout=(30, None),
    )
    response.raise_for_status()

    first_token_at: dict[int, float] = {}
    finished_at: dict[int, float] = {}
    final_meta: dict[int, dict] = {}
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data:"):
            continue
        payload = raw_line[5:].strip()
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        if "error" in event:
            raise RuntimeError(f"{label} request failed: {event['error']}")
        index = int(event["index"])
        meta = event["meta_info"]
        now = time.perf_counter()
        if meta.get("completion_tokens", 0) >= 1 and index not in first_token_at:
            first_token_at[index] = now
        if meta.get("finish_reason") is not None:
            finished_at[index] = now
            final_meta[index] = meta

    ended = time.perf_counter()
    expected = set(range(len(input_ids)))
    if set(final_meta) != expected or set(first_token_at) != expected:
        raise RuntimeError(
            f"{label} incomplete: final={sorted(final_meta)}, first={sorted(first_token_at)}"
        )
    return {
        "wall_s": ended - started,
        "ttft_s": [first_token_at[i] - started for i in range(len(input_ids))],
        "decode_s": [finished_at[i] - first_token_at[i] for i in range(len(input_ids))],
        "cached_tokens": [
            int(final_meta[i].get("cached_tokens", 0)) for i in range(len(input_ids))
        ],
        "completion_tokens": [
            int(final_meta[i].get("completion_tokens", 0))
            for i in range(len(input_ids))
        ],
    }


def _start_profile(
    base_url: str,
    output_dir: Path,
    *,
    host_tracer_level: int,
    python_tracer_level: int,
    num_steps: int | None = None,
    profile_by_stage: bool = False,
    profile_stages: list[str] | None = None,
) -> None:
    payload = {
        "output_dir": str(output_dir),
        "host_tracer_level": host_tracer_level,
        "python_tracer_level": python_tracer_level,
    }
    if num_steps is not None:
        payload["num_steps"] = num_steps
    if profile_by_stage:
        payload["profile_by_stage"] = True
    if profile_stages is not None:
        payload["profile_stages"] = profile_stages

    response = requests.post(
        f"{base_url}/start_profile",
        json=payload,
        timeout=(30, None),
    )
    response.raise_for_status()


def _stop_profile(base_url: str) -> None:
    status = requests.get(f"{base_url}/profile_status", timeout=60)
    status.raise_for_status()
    if status.json().get("status") == "idle":
        return

    response = requests.post(f"{base_url}/stop_profile", timeout=(30, None))
    response.raise_for_status()
    status = requests.get(f"{base_url}/profile_status", timeout=60)
    status.raise_for_status()
    if status.json().get("status") != "idle":
        raise RuntimeError(f"profile did not stop cleanly: {status.text}")


def _run_native_batch_with_admission_barrier(
    base_url: str,
    input_ids: list[list[int]],
    output_len: int,
    *,
    label: str,
    on_admitted: Callable[[], None],
    timeout_s: float = 60,
) -> dict:
    """Queue a native batch atomically before releasing the scheduler.

    Profiling startup can otherwise race the tokenizer manager's per-request
    enqueue loop and split a native C32 request into, for example, C29 + C3.
    Pausing an idle scheduler lets every request reach the waiting queue before
    the stage profiler is armed and inference resumes.
    """
    pause = requests.post(
        f"{base_url}/pause_generation",
        json={"mode": "in_place"},
        timeout=(30, None),
    )
    pause.raise_for_status()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _run_native_batch,
            base_url,
            input_ids,
            output_len,
            label=label,
        )
        try:
            deadline = time.monotonic() + timeout_s
            while True:
                if future.done():
                    future.result()
                    raise RuntimeError("native batch finished while generation was paused")
                server_info = requests.get(
                    f"{base_url}/get_server_info", timeout=60
                )
                server_info.raise_for_status()
                internal_states = server_info.json().get("internal_states", [])
                waiting_sizes = [
                    int(state.get("waiting_queue_size", -1))
                    for state in internal_states
                ]
                if waiting_sizes and min(waiting_sizes) >= len(input_ids):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "native batch did not reach every scheduler waiting queue: "
                        f"expected={len(input_ids)}, observed={waiting_sizes}"
                    )
                time.sleep(0.1)

            on_admitted()
        finally:
            resume = requests.post(
                f"{base_url}/continue_generation",
                json={},
                timeout=(30, None),
            )
            resume.raise_for_status()

        return future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--dp-size", type=int, default=16)
    parser.add_argument("--prefix-len", type=int, default=16 * 1024)
    parser.add_argument("--extend-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--random-seed", type=int, default=3)
    parser.add_argument("--random-token-min", type=int, default=1000)
    parser.add_argument("--random-token-max", type=int, default=32000)
    parser.add_argument(
        "--variant",
        default="exact_dsa_exact_lax_topk",
        help="Label recorded in the output metrics for the serving variant.",
    )
    parser.add_argument(
        "--profile-output-dir",
        type=Path,
        help="Profile only the measured cache-hit extend/decode request.",
    )
    parser.add_argument("--profile-host-tracer-level", type=int, default=0)
    parser.add_argument("--profile-python-tracer-level", type=int, default=0)
    parser.add_argument(
        "--profile-num-steps",
        type=int,
        help="Number of forward steps to trace per selected stage.",
    )
    parser.add_argument(
        "--profile-by-stage",
        action="store_true",
        help="Write separate traces for the selected prefill/decode stages.",
    )
    parser.add_argument(
        "--profile-admission-barrier",
        action="store_true",
        help=(
            "Pause an idle scheduler until the measured native batch is fully "
            "queued, then arm profiling and resume inference."
        ),
    )
    parser.add_argument(
        "--profile-stages",
        nargs="+",
        choices=("prefill", "decode"),
        help="Stages to trace when --profile-by-stage is set.",
    )
    parser.add_argument(
        "--prefix-mode",
        choices=("independent", "shared"),
        default="independent",
    )
    parser.add_argument("--cache-hit-tolerance", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    prefixes, extended = _make_inputs(
        args.concurrency,
        args.prefix_len,
        args.extend_len,
        prefix_mode=args.prefix_mode,
        random_seed=args.random_seed,
        random_token_min=args.random_token_min,
        random_token_max=args.random_token_max,
    )
    if args.prefix_mode == "shared":
        if args.dp_size > args.concurrency:
            raise ValueError(
                "dp_size cannot exceed concurrency for shared-prefix warmup"
            )
        # With round-robin DP scheduling, one identical request per rank installs
        # the shared prefix on every rank. The measured C=2/DP batch can then hit
        # the same rank-local prefix without recomputing 32 independent prefixes.
        warm_inputs = prefixes[: args.dp_size]
    else:
        warm_inputs = prefixes

    flush = requests.post(f"{base_url}/flush_cache", timeout=60)
    flush.raise_for_status()
    warm = _run_native_batch(base_url, warm_inputs, 1, label="warm-prefix")
    profile_started = False

    def start_profile() -> None:
        nonlocal profile_started
        assert args.profile_output_dir is not None
        _start_profile(
            base_url,
            args.profile_output_dir,
            host_tracer_level=args.profile_host_tracer_level,
            python_tracer_level=args.profile_python_tracer_level,
            num_steps=args.profile_num_steps,
            profile_by_stage=args.profile_by_stage,
            profile_stages=args.profile_stages,
        )
        profile_started = True

    if args.profile_admission_barrier and args.profile_output_dir is None:
        raise ValueError("--profile-admission-barrier requires --profile-output-dir")

    try:
        if args.profile_admission_barrier:
            measured = _run_native_batch_with_admission_barrier(
                base_url,
                extended,
                args.output_len,
                label="cache-hit-extend-decode",
                on_admitted=start_profile,
            )
        else:
            if args.profile_output_dir is not None:
                start_profile()
            measured = _run_native_batch(
                base_url, extended, args.output_len, label="cache-hit-extend-decode"
            )
    finally:
        if profile_started:
            _stop_profile(base_url)

    minimum_expected_hit = args.prefix_len - args.cache_hit_tolerance
    if min(measured["cached_tokens"]) < minimum_expected_hit:
        raise RuntimeError(
            f"cache-hit invariant failed: min={min(measured['cached_tokens'])}, "
            f"expected>={minimum_expected_hit}"
        )
    if measured["completion_tokens"] != [args.output_len] * args.concurrency:
        raise RuntimeError(
            f"completion invariant failed: {measured['completion_tokens']}"
        )

    ttft = measured["ttft_s"]
    decode = measured["decode_s"]
    tpots_ms = [value * 1000 / max(args.output_len - 1, 1) for value in decode]
    result = {
        "variant": args.variant,
        "concurrency": args.concurrency,
        "dp_size": args.dp_size,
        "prefix_mode": args.prefix_mode,
        "warm_concurrency": len(warm_inputs),
        "prefix_len": args.prefix_len,
        "extend_len": args.extend_len,
        "output_len": args.output_len,
        "random_seed": args.random_seed,
        "random_token_min": args.random_token_min,
        "random_token_max": args.random_token_max,
        "profile_output_dir": (
            str(args.profile_output_dir) if args.profile_output_dir is not None else None
        ),
        "profile_admission_barrier": args.profile_admission_barrier,
        "minimum_expected_cache_hit": minimum_expected_hit,
        "warm_wall_s": warm["wall_s"],
        "wall_s": measured["wall_s"],
        "ttft_mean_s": statistics.fmean(ttft),
        "ttft_p50_s": statistics.median(ttft),
        "ttft_p95_s": _percentile(ttft, 0.95),
        "decode_mean_s": statistics.fmean(decode),
        "decode_p50_s": statistics.median(decode),
        "tpot_p50_ms": statistics.median(tpots_ms),
        "tpot_p95_ms": _percentile(tpots_ms, 0.95),
        "output_throughput_tok_s": args.concurrency
        * args.output_len
        / measured["wall_s"],
        "cached_tokens_min": min(measured["cached_tokens"]),
        "cached_tokens_max": max(measured["cached_tokens"]),
        "cached_tokens": measured["cached_tokens"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
