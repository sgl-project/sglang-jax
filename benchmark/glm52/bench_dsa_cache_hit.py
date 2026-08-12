"""Correctness-first GLM-5.2 DSA cache-hit extend/decode benchmark.

The measured requests are admitted through an explicit scheduler barrier.  A
native HTTP batch alone is not atomic at the scheduler boundary: the tokenizer
manager forwards its members one by one, so a large C64 request can otherwise
be scheduled as partial waves.  The barrier keeps the ordered native batch in
the waiting queue until every expected request is visible, then releases one
prefill/decode workload.

The workload supports shared, unique (legacy name: independent), and grouped
prefixes.  It validates the C32/C64-style concurrency shape, relevant serving
budgets, cache hits, and full output completion before reporting metrics.
"""

from __future__ import annotations

import argparse
import array
import ast
import concurrent.futures
import hashlib
import json
import random
import re
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import requests


# A complete TPU XPlane can take several minutes to flush after the measured
# request finishes. Profile-control RPCs wait for that flush on the server, so
# their read timeout must be longer than ordinary health/status requests.
PROFILE_CONTROL_TIMEOUT_S = 600

PREFIX_MODE_ALIASES = {
    "grouped": "grouped",
    "independent": "unique",
    "shared": "shared",
    "shared-prefix": "shared",
    "unique": "unique",
    "unique-prefix": "unique",
}


def _normalize_prefix_mode(prefix_mode: str) -> str:
    try:
        return PREFIX_MODE_ALIASES[prefix_mode]
    except KeyError as error:
        choices = ", ".join(sorted(PREFIX_MODE_ALIASES))
        raise ValueError(
            f"unknown prefix_mode: {prefix_mode!r}; expected one of: {choices}"
        ) from error


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
    prefix_group_count: int = 2,
    random_seed: int = 3,
    random_token_min: int = 1000,
    random_token_max: int = 32000,
) -> tuple[list, list]:
    prefix_mode = _normalize_prefix_mode(prefix_mode)
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if prefix_len < 2:
        raise ValueError("prefix_len must be at least 2")
    if extend_len < 1:
        raise ValueError("extend_len must be positive")
    if prefix_mode == "grouped":
        if prefix_group_count < 2:
            raise ValueError("grouped mode requires at least two prefix groups")
        if concurrency % prefix_group_count != 0:
            raise ValueError(
                "concurrency must be divisible by prefix_group_count in grouped mode"
            )
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
    grouped_prefixes = (
        [random_tokens(prefix_len) for _ in range(prefix_group_count)]
        if prefix_mode == "grouped"
        else []
    )
    requests_per_group = (
        concurrency // prefix_group_count if prefix_mode == "grouped" else 0
    )
    for request_id in range(concurrency):
        if prefix_mode == "shared":
            prefix = shared_prefix.copy()
        elif prefix_mode == "grouped":
            group_id = request_id // requests_per_group
            prefix = grouped_prefixes[group_id].copy()
        else:  # unique
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


def _prefix_layout(prefixes: list[list[int]]) -> dict:
    """Return compact, deterministic evidence for the generated prefix groups."""
    digest_to_count: dict[str, int] = {}
    for prefix in prefixes:
        digest = hashlib.sha256(array.array("I", prefix).tobytes()).hexdigest()
        digest_to_count[digest] = digest_to_count.get(digest, 0) + 1
    return {
        "unique_prefixes": len(digest_to_count),
        "prefix_groups": [
            {"sha256": digest, "requests": count}
            for digest, count in digest_to_count.items()
        ],
    }


def _make_warm_inputs(
    prefixes: list[list[int]],
    extended: list[list[int]],
    *,
    warm_branch_token: int,
) -> list[list[int]]:
    """Add a known branch token so warmup cannot cache a measured extension token."""
    if len(prefixes) != len(extended):
        raise ValueError("prefix and extended input counts must match")

    warm_inputs = []
    for index, (prefix, measured_input) in enumerate(zip(prefixes, extended)):
        if measured_input[: len(prefix)] != prefix or len(measured_input) <= len(
            prefix
        ):
            raise ValueError(f"invalid measured extension at request {index}")
        measured_branch_token = measured_input[len(prefix)]
        if measured_branch_token == warm_branch_token:
            raise ValueError(
                "warm branch token collides with measured extension: "
                f"request={index}, token={warm_branch_token}"
            )
        warm_inputs.append(prefix + [warm_branch_token])
    return warm_inputs


def _build_workload_layout(
    prefixes: list[list[int]],
    *,
    warm_inputs: list[list[int]] | None = None,
    prefix_mode: str,
    prefix_group_count: int,
    dp_size: int,
    expected_requests_per_dp: int,
) -> dict:
    """Validate concurrency semantics and construct deterministic warm batches."""
    prefix_mode = _normalize_prefix_mode(prefix_mode)
    concurrency = len(prefixes)
    warm_inputs = prefixes if warm_inputs is None else warm_inputs
    if len(warm_inputs) != concurrency:
        raise ValueError(
            f"warm input count ({len(warm_inputs)}) must equal concurrency ({concurrency})"
        )
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    if expected_requests_per_dp <= 0:
        raise ValueError("expected_requests_per_dp must be positive")
    if concurrency % dp_size != 0:
        raise ValueError(
            f"concurrency ({concurrency}) must be divisible by dp_size ({dp_size})"
        )

    requests_per_dp = concurrency // dp_size
    if requests_per_dp != expected_requests_per_dp:
        raise ValueError(
            "concurrency invariant failed: "
            f"concurrency={concurrency}, dp_size={dp_size}, "
            f"observed_requests_per_dp={requests_per_dp}, "
            f"expected_requests_per_dp={expected_requests_per_dp}"
        )

    prefix_layout = _prefix_layout(prefixes)
    if prefix_mode == "grouped":
        # Contiguous groups plus round-robin routing put one member of every
        # prefix group on every DP rank.  Requiring one group per request slot
        # makes the cache placement and the measured routing unambiguous.
        if prefix_group_count != requests_per_dp:
            raise ValueError(
                "grouped mode requires prefix_group_count == requests_per_dp: "
                f"groups={prefix_group_count}, requests_per_dp={requests_per_dp}"
            )
        if concurrency != dp_size * prefix_group_count:
            raise ValueError(
                "grouped mode requires concurrency == dp_size * prefix_group_count"
            )
        warm_batches = [
            warm_inputs[group_id * dp_size : (group_id + 1) * dp_size]
            for group_id in range(prefix_group_count)
        ]
        cached_prefixes_per_dp = prefix_group_count
    elif prefix_mode == "shared":
        # One identical request per rank installs the shared prefix everywhere.
        warm_batches = [warm_inputs[:dp_size]]
        cached_prefixes_per_dp = 1
    else:  # unique
        # Preserve the measured request order.  Since both warm and measured
        # request counts are multiples of dp_size, round-robin assigns every
        # unique prefix to the same DP rank in both phases.
        warm_batches = [warm_inputs]
        cached_prefixes_per_dp = requests_per_dp

    if any(len(batch) % dp_size != 0 for batch in warm_batches):
        raise AssertionError(
            f"warm batches must preserve round-robin alignment: "
            f"sizes={[len(batch) for batch in warm_batches]}, dp_size={dp_size}"
        )

    return {
        "prefix_mode": prefix_mode,
        "prefix_layout": prefix_layout,
        "requests_per_dp": requests_per_dp,
        "cached_prefixes_per_dp": cached_prefixes_per_dp,
        "warm_batches": warm_batches,
    }


def _run_native_batch(
    base_url: str,
    input_ids: list[list[int]],
    output_len: int,
    *,
    label: str,
    timing_state: dict | None = None,
) -> dict:
    started = time.perf_counter()
    started_unix_ns = time.time_ns()
    if timing_state is not None:
        timing_state["request_start_perf_s"] = started
        timing_state["request_start_unix_ns"] = started_unix_ns
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
    response_headers_at = time.perf_counter()
    response_headers_unix_ns = time.time_ns()

    first_token_at: dict[int, float] = {}
    first_token_unix_ns: dict[int, int] = {}
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
            first_token_unix_ns[index] = time.time_ns()
        if meta.get("finish_reason") is not None:
            finished_at[index] = now
            final_meta[index] = meta

    ended = time.perf_counter()
    expected = set(range(len(input_ids)))
    if set(final_meta) != expected or set(first_token_at) != expected:
        raise RuntimeError(
            f"{label} incomplete: final={sorted(final_meta)}, first={sorted(first_token_at)}"
        )
    result = {
        "wall_s": ended - started,
        "ttft_s": [first_token_at[i] - started for i in range(len(input_ids))],
        "request_to_headers_s": [
            response_headers_at - started for _ in range(len(input_ids))
        ],
        "headers_to_first_token_s": [
            first_token_at[i] - response_headers_at for i in range(len(input_ids))
        ],
        "request_start_unix_ns": [started_unix_ns for _ in range(len(input_ids))],
        "response_headers_unix_ns": [
            response_headers_unix_ns for _ in range(len(input_ids))
        ],
        "first_token_unix_ns": [first_token_unix_ns[i] for i in range(len(input_ids))],
        "decode_s": [finished_at[i] - first_token_at[i] for i in range(len(input_ids))],
        "decode_batch_wall_s": max(finished_at.values()) - min(first_token_at.values()),
        "cached_tokens": [
            int(final_meta[i].get("cached_tokens", 0)) for i in range(len(input_ids))
        ],
        "completion_tokens": [
            int(final_meta[i].get("completion_tokens", 0))
            for i in range(len(input_ids))
        ],
    }
    if timing_state is not None and "measurement_start_perf_s" in timing_state:
        measurement_start = float(timing_state["measurement_start_perf_s"])
        admission_wait_s = measurement_start - started
        if admission_wait_s < 0:
            raise RuntimeError(
                f"measurement start precedes HTTP submission: {admission_wait_s=}"
            )
        result["http_wall_s"] = result["wall_s"]
        result["http_ttft_s"] = result["ttft_s"]
        result["wall_s"] = ended - measurement_start
        result["ttft_s"] = [value - admission_wait_s for value in result["ttft_s"]]
        if min(result["ttft_s"]) < 0:
            raise RuntimeError(
                "a measured token arrived before the scheduler release: "
                f"ttft={result['ttft_s']}"
            )
        result["admission_wait_s"] = admission_wait_s
        result["measurement_start_unix_ns"] = int(
            timing_state["measurement_start_unix_ns"]
        )
        result["measurement_origin"] = "scheduler_release"
    else:
        result["http_wall_s"] = result["wall_s"]
        result["http_ttft_s"] = result["ttft_s"]
        result["admission_wait_s"] = 0.0
        result["measurement_start_unix_ns"] = started_unix_ns
        result["measurement_origin"] = "http_submission"
    return result


def _throughput_metrics(measured: dict) -> dict:
    """Name end-to-end and decode-window throughput without mixing scopes."""
    completion_tokens = [int(value) for value in measured["completion_tokens"]]
    e2e_wall_s = float(measured["wall_s"])
    decode_batch_wall_s = float(measured["decode_batch_wall_s"])
    if e2e_wall_s <= 0:
        raise RuntimeError(
            f"end-to-end throughput window must be positive: {e2e_wall_s=}"
        )

    total_output_tokens = sum(completion_tokens)
    # Each request's first token defines TTFT and is excluded from the decode
    # window numerator. The window starts at the earliest first token and ends
    # at the last completion, which is conservative if first-token times differ.
    decode_output_tokens = sum(max(tokens - 1, 0) for tokens in completion_tokens)
    if decode_output_tokens > 0 and decode_batch_wall_s <= 0:
        raise RuntimeError(
            "decode throughput window must be positive when decode tokens exist: "
            f"decode_output_tokens={decode_output_tokens}, "
            f"decode_batch_wall_s={decode_batch_wall_s}"
        )
    e2e_output_throughput = total_output_tokens / e2e_wall_s
    decode_output_throughput = (
        decode_output_tokens / decode_batch_wall_s if decode_output_tokens > 0 else 0.0
    )
    measurement_origin = measured.get("measurement_origin", "scheduler_release")
    return {
        # Backward-compatible historical key. Its scope is now explicit below.
        "output_throughput_tok_s": e2e_output_throughput,
        "output_throughput_scope": f"{measurement_origin}_to_last_completion",
        "e2e_output_throughput_tok_s": e2e_output_throughput,
        "e2e_output_tokens": total_output_tokens,
        "decode_throughput_tok_s": decode_output_throughput,
        "decode_throughput_scope": (
            "earliest_first_token_to_last_completion_excluding_first_token_per_request"
        ),
        "decode_output_tokens": decode_output_tokens,
        "decode_batch_wall_s": decode_batch_wall_s,
    }


def _ceil_to_page(tokens: int, page_size: int) -> int:
    return -(-tokens // page_size) * page_size


def _get_server_info(base_url: str) -> dict:
    response = requests.get(f"{base_url}/get_server_info", timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected get_server_info response: {payload!r}")
    return payload


def _require_idle_scheduler(server_info: dict, *, phase: str) -> None:
    states = server_info.get("internal_states", [])
    if not states:
        raise RuntimeError(f"{phase}: get_server_info returned no internal_states")

    violations = []
    for index, state in enumerate(states):
        waiting = int(state.get("waiting_queue_size", -1))
        pending = int(state.get("pending_dp_reqs_size", -1))
        running = int(state.get("running_batch_size", -1))
        chunked = state.get("chunked_req_rids", [])
        if (
            waiting != 0
            or pending != 0
            or running != 0
            or any(rid is not None for rid in chunked)
        ):
            violations.append(
                {
                    "state": index,
                    "waiting": waiting,
                    "pending_dp": pending,
                    "running": running,
                    "chunked_req_rids": chunked,
                }
            )
    if violations:
        raise RuntimeError(f"{phase}: scheduler is not idle: {violations}")


def _per_dp_token_capacities(server_info: dict, dp_size: int) -> tuple[str, list[int]]:
    states = server_info.get("internal_states", [])
    raw = [
        int(state.get("memory_usage", {}).get("token_capacity", -1)) for state in states
    ]
    if len(raw) == dp_size and min(raw, default=-1) > 0:
        return "per_dp_scheduler_states", raw
    if len(raw) == 1 and raw[0] > 0:
        if raw[0] % dp_size != 0:
            raise RuntimeError(
                "global token capacity is not divisible by dp_size: "
                f"capacity={raw[0]}, dp_size={dp_size}"
            )
        return "single_global_scheduler_state", [raw[0] // dp_size] * dp_size
    raise RuntimeError(
        "cannot determine per-DP token capacity: "
        f"states={len(states)}, dp_size={dp_size}, raw_capacities={raw}"
    )


def _validate_server_configuration(
    server_info: dict,
    *,
    concurrency: int,
    dp_size: int,
    requests_per_dp: int,
    cached_prefixes_per_dp: int,
    prefix_len: int,
    extend_len: int,
    output_len: int,
    warm_suffix_len: int = 2,
) -> dict:
    """Fail before warmup when the server cannot admit one synchronized batch."""
    errors = []
    observed_dp_size = int(server_info.get("dp_size", -1))
    if observed_dp_size != dp_size:
        errors.append(f"dp_size={observed_dp_size}, expected {dp_size}")

    dp_policy = server_info.get("dp_schedule_policy")
    if dp_policy != "round_robin":
        errors.append(f"dp_schedule_policy={dp_policy!r}, expected 'round_robin'")

    if bool(server_info.get("disable_radix_cache", False)):
        errors.append(
            "disable_radix_cache=true is incompatible with cache-hit benchmark"
        )

    page_size = int(server_info.get("page_size", -1))
    if page_size <= 0:
        errors.append(f"invalid page_size={page_size}")
        page_size = 1

    rounded_cached_prefix = _ceil_to_page(prefix_len + warm_suffix_len, page_size)
    rounded_extend = _ceil_to_page(extend_len, page_size)
    rounded_output = _ceil_to_page(output_len, page_size)
    required_global_prefill_tokens = concurrency * rounded_extend
    required_per_dp_chunk_tokens = requests_per_dp * rounded_extend
    required_context_tokens = prefix_len + extend_len + output_len

    max_prefill_tokens = int(server_info.get("max_prefill_tokens", -1))
    if max_prefill_tokens < required_global_prefill_tokens:
        errors.append(
            f"max_prefill_tokens={max_prefill_tokens} < "
            f"required_global_prefill_tokens={required_global_prefill_tokens}"
        )

    chunked_prefill_size = int(server_info.get("chunked_prefill_size", -1))
    if chunked_prefill_size < required_per_dp_chunk_tokens:
        errors.append(
            f"chunked_prefill_size={chunked_prefill_size} < "
            f"required_per_dp_chunk_tokens={required_per_dp_chunk_tokens}"
        )

    max_running_requests = server_info.get("max_running_requests")
    if max_running_requests is None or int(max_running_requests) < concurrency:
        errors.append(
            f"max_running_requests={max_running_requests} < concurrency={concurrency}"
        )

    context_length = int(server_info.get("context_length", -1))
    if context_length <= required_context_tokens:
        errors.append(
            f"context_length={context_length} must be greater than "
            f"requested tokens={required_context_tokens}"
        )

    capacity_layout, per_dp_capacities = _per_dp_token_capacities(server_info, dp_size)
    # Admission uses strict capacity comparisons.  Add one page of headroom so
    # an exact-fit pool is rejected before it can serialize an otherwise valid
    # synchronized batch.
    required_per_dp_token_capacity = (
        cached_prefixes_per_dp * rounded_cached_prefix
        + requests_per_dp * (rounded_extend + rounded_output)
        + page_size
    )
    if min(per_dp_capacities) < required_per_dp_token_capacity:
        errors.append(
            f"per-DP token capacity min={min(per_dp_capacities)} < "
            f"required={required_per_dp_token_capacity} "
            f"(cached_prefixes_per_dp={cached_prefixes_per_dp})"
        )

    evidence = {
        "concurrency": concurrency,
        "dp_size": dp_size,
        "requests_per_dp": requests_per_dp,
        "cached_prefixes_per_dp": cached_prefixes_per_dp,
        "warm_suffix_len": warm_suffix_len,
        "rounded_cached_prefix_tokens": rounded_cached_prefix,
        "required_global_prefill_tokens": required_global_prefill_tokens,
        "observed_max_prefill_tokens": max_prefill_tokens,
        "required_per_dp_chunk_tokens": required_per_dp_chunk_tokens,
        "observed_chunked_prefill_size": chunked_prefill_size,
        "required_context_tokens": required_context_tokens,
        "observed_context_length": context_length,
        "observed_max_running_requests": max_running_requests,
        "capacity_layout": capacity_layout,
        "per_dp_token_capacity_min": min(per_dp_capacities),
        "per_dp_token_capacity_max": max(per_dp_capacities),
        "required_per_dp_token_capacity": required_per_dp_token_capacity,
        "page_size": page_size,
        "dp_schedule_policy": dp_policy,
    }
    if errors:
        raise RuntimeError(
            "server configuration cannot satisfy synchronized benchmark: "
            + "; ".join(errors)
        )
    return evidence


def _extract_log_int(line: str, label: str) -> int:
    match = re.search(re.escape(label) + r"\s*(\d+)", line)
    if not match:
        raise RuntimeError(f"missing {label!r} in server log line: {line}")
    return int(match.group(1))


def _extract_log_list(line: str, label: str) -> list[int]:
    match = re.search(re.escape(label) + r"\s*(\[[^\]]+\])", line)
    if not match:
        raise RuntimeError(f"missing {label!r} in server log line: {line}")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise RuntimeError(f"invalid {label!r} in server log line: {line}")
    return value


def _validate_measured_server_log(
    server_log: Path,
    *,
    start_offset: int,
    concurrency: int,
    dp_size: int,
    requests_per_dp: int,
    extend_len: int,
    expected_cached_tokens: int,
) -> dict:
    """Require one full measured prefill and one full-concurrency decode sample."""
    with server_log.open("rb") as source:
        source.seek(start_offset)
        lines = source.read().decode(errors="replace").splitlines()

    measured_prefills = [
        line
        for line in lines
        if "Prefill batch." in line and _extract_log_int(line, "#cached-token:") > 0
    ]
    if len(measured_prefills) != 1:
        raise RuntimeError(
            "measured prefill must be exactly one scheduler batch: "
            f"observed_count={len(measured_prefills)}, lines={measured_prefills}"
        )

    prefill = measured_prefills[0]
    expected_prefill_layout = [requests_per_dp] * dp_size
    observed_prefill_layout = _extract_log_list(prefill, "#prefill per DP:")
    expected_prefill_tokens = concurrency * extend_len
    prefill_checks = {
        "new_seq": (_extract_log_int(prefill, "#new-seq:"), concurrency),
        "new_token": (
            _extract_log_int(prefill, "#new-token:"),
            expected_prefill_tokens,
        ),
        "cached_token": (
            _extract_log_int(prefill, "#cached-token:"),
            expected_cached_tokens,
        ),
        "running_req": (_extract_log_int(prefill, "#running-req:"), 0),
        "queue_req": (_extract_log_int(prefill, "#queue-req:"), 0),
    }
    mismatches = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in prefill_checks.items()
        if observed != expected
    }
    if observed_prefill_layout != expected_prefill_layout:
        mismatches["prefill_per_dp"] = {
            "observed": observed_prefill_layout,
            "expected": expected_prefill_layout,
        }
    if mismatches:
        raise RuntimeError(
            f"measured prefill concurrency invariant failed: {mismatches}"
        )

    full_decode_lines = [
        line
        for line in lines
        if "Decode batch." in line
        and _extract_log_int(line, "#running-req:") == concurrency
    ]
    if not full_decode_lines:
        raise RuntimeError(
            f"missing full-concurrency decode evidence for concurrency={concurrency}"
        )
    decode = full_decode_lines[-1]
    observed_decode_layout = _extract_log_list(decode, "#running-req per DP:")
    expected_decode_layout = [requests_per_dp] * dp_size
    if observed_decode_layout != expected_decode_layout:
        raise RuntimeError(
            "decode concurrency invariant failed: "
            f"observed={observed_decode_layout}, expected={expected_decode_layout}"
        )

    return {
        "prefill_batch_count": 1,
        "prefill_new_seq": concurrency,
        "prefill_new_token": expected_prefill_tokens,
        "prefill_cached_token": expected_cached_tokens,
        "prefill_per_dp": observed_prefill_layout,
        "decode_per_dp": observed_decode_layout,
        "prefill_log_line": prefill,
        "decode_log_line": decode,
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
    status = requests.get(
        f"{base_url}/profile_status", timeout=PROFILE_CONTROL_TIMEOUT_S
    )
    status.raise_for_status()
    if status.json().get("status") == "idle":
        return

    response = requests.post(f"{base_url}/stop_profile", timeout=(30, None))
    response.raise_for_status()
    status = requests.get(
        f"{base_url}/profile_status", timeout=PROFILE_CONTROL_TIMEOUT_S
    )
    status.raise_for_status()
    if status.json().get("status") != "idle":
        raise RuntimeError(f"profile did not stop cleanly: {status.text}")


def _set_scheduler_paused(base_url: str, paused: bool) -> None:
    """Pause only model scheduling while leaving HTTP admission active."""
    response = requests.post(
        f"{base_url}/set_internal_state",
        json={
            "request_id": "benchmark-admission-barrier",
            "state_data": {"engine_paused": paused},
        },
        timeout=(30, None),
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", False):
        raise RuntimeError(f"could not update scheduler pause state: {payload}")


def _admission_snapshot(server_info: dict, expected_rids: set[str]) -> dict:
    """Summarize queued request identities across replicated or sharded states."""
    states = server_info.get("internal_states", [])
    waiting_union: set[str] = set()
    running_union: set[str] = set()
    waiting_sizes = []
    pending_sizes = []
    paused = []
    chunked_rids = []
    state_rid_counts = []
    malformed_states = []

    for index, state in enumerate(states):
        waiting_rids = [str(rid) for rid in state.get("waiting_queue_rids", [])]
        running_rids = [str(rid) for rid in state.get("running_batch_rids", [])]
        waiting_size = int(state.get("waiting_queue_size", -1))
        if waiting_size != len(waiting_rids):
            malformed_states.append(
                {
                    "state": index,
                    "waiting_queue_size": waiting_size,
                    "waiting_queue_rids": len(waiting_rids),
                }
            )
        waiting_union.update(waiting_rids)
        running_union.update(running_rids)
        waiting_sizes.append(waiting_size)
        pending_sizes.append(int(state.get("pending_dp_reqs_size", -1)))
        paused.append(bool(state.get("engine_paused", False)))
        chunked_rids.extend(
            rid for rid in state.get("chunked_req_rids", []) if rid is not None
        )
        state_rid_counts.append(len(set(waiting_rids)))

    missing = sorted(expected_rids - waiting_union)
    unexpected = sorted(waiting_union - expected_rids)
    return {
        "complete": bool(states)
        and not malformed_states
        and not missing
        and not unexpected
        and not running_union
        and not chunked_rids
        and all(size == 0 for size in pending_sizes)
        and all(paused),
        "state_count": len(states),
        "state_waiting_sizes": waiting_sizes,
        "state_unique_rid_counts": state_rid_counts,
        "waiting_unique_count": len(waiting_union),
        "missing_rids": missing,
        "unexpected_rids": unexpected,
        "running_rids": sorted(running_union),
        "pending_dp_reqs_sizes": pending_sizes,
        "chunked_rids": sorted(str(rid) for rid in chunked_rids),
        "engine_paused": paused,
        "malformed_states": malformed_states,
    }


def _run_native_batch_with_admission_barrier(
    base_url: str,
    input_ids: list[list[int]],
    output_len: int,
    *,
    label: str,
    on_admitted: Callable[[], None] | None = None,
    profile_settle_s: float = 5.0,
    timeout_s: float = 180,
) -> dict:
    """Queue one ordered native batch completely before releasing the scheduler.

    Request order is load-bearing for round-robin cache placement, especially
    for unique prefixes.  The scheduler state is therefore checked by exact RID
    identity instead of summing queue sizes, which may be replicated across
    multi-node scheduler states.
    """
    initial_info = _get_server_info(base_url)
    _require_idle_scheduler(initial_info, phase="before admission barrier")
    expected_rid_list = [f"{label}-{i}" for i in range(len(input_ids))]
    expected_rids = set(expected_rid_list)
    timing_state: dict = {}
    barrier_started = time.monotonic()
    poll_count = 0
    ready_snapshot = None
    queue_ready_s = None
    paused = False

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    try:
        _set_scheduler_paused(base_url, True)
        paused = True
        future = executor.submit(
            _run_native_batch,
            base_url,
            input_ids,
            output_len,
            label=label,
            timing_state=timing_state,
        )
        try:
            deadline = time.monotonic() + timeout_s
            while True:
                if future.done():
                    future.result()
                    raise RuntimeError(
                        "native batch finished while generation was paused"
                    )
                poll_count += 1
                snapshot = _admission_snapshot(
                    _get_server_info(base_url), expected_rids
                )
                if snapshot["unexpected_rids"] or snapshot["running_rids"]:
                    raise RuntimeError(
                        "admission barrier observed unrelated or prematurely running requests: "
                        f"{snapshot}"
                    )
                if snapshot["complete"]:
                    ready_snapshot = snapshot
                    queue_ready_s = time.monotonic() - barrier_started
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "native batch did not reach the scheduler waiting queue intact: "
                        f"expected_total={len(input_ids)}, observed={snapshot}"
                    )
                time.sleep(0.25)

            if on_admitted is not None:
                on_admitted()
                # TPU device tracing is initialized asynchronously after the
                # control request returns. Give it a bounded head start before
                # the first measured forward is released.
                if profile_settle_s > 0:
                    time.sleep(profile_settle_s)

            timing_state["measurement_start_perf_s"] = time.perf_counter()
            timing_state["measurement_start_unix_ns"] = time.time_ns()
        finally:
            # Always release before waiting for the request thread. Otherwise
            # an admission assertion failure would deadlock on a paused server.
            if paused:
                _set_scheduler_paused(base_url, False)
                paused = False

        result = future.result()
    finally:
        if paused:
            _set_scheduler_paused(base_url, False)
            paused = False
        executor.shutdown(wait=True, cancel_futures=True)

    assert ready_snapshot is not None
    assert queue_ready_s is not None
    result["admission_evidence"] = {
        "expected_request_count": len(input_ids),
        "expected_rids_sha256": hashlib.sha256(
            "\n".join(expected_rid_list).encode()
        ).hexdigest(),
        "queue_ready_s": queue_ready_s,
        "poll_count": poll_count,
        "snapshot": ready_snapshot,
        "ordered_native_batch": True,
    }
    return result


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
            "Deprecated compatibility flag. Admission barriers are enabled by "
            "default for both benchmark and profile runs."
        ),
    )
    parser.add_argument(
        "--disable-admission-barrier",
        action="store_true",
        help=(
            "Disable atomic measured-batch admission. Intended only for scheduler "
            "arrival experiments; correctness benchmark runs should not use it."
        ),
    )
    parser.add_argument(
        "--admission-timeout-s",
        type=float,
        default=180,
        help="Maximum time to wait for every measured RID to enter the waiting queue.",
    )
    parser.add_argument(
        "--profile-settle-s",
        type=float,
        default=5,
        help="Delay between arming a profile and releasing the queued native batch.",
    )
    parser.add_argument(
        "--profile-stages",
        nargs="+",
        choices=("prefill", "decode"),
        help="Stages to trace when --profile-by-stage is set.",
    )
    parser.add_argument(
        "--prefix-mode",
        choices=tuple(sorted(PREFIX_MODE_ALIASES)),
        default="unique",
    )
    parser.add_argument(
        "--prefix-group-count",
        type=int,
        default=2,
        help=(
            "Number of distinct cached prefixes in grouped mode. Requests are "
            "laid out in contiguous, equally sized groups so round-robin DP "
            "scheduling installs one copy of every prefix on every DP rank."
        ),
    )
    parser.add_argument(
        "--expected-requests-per-dp",
        type=int,
        default=2,
        help=(
            "Hard concurrency invariant. C32/DP16 and C64/DP32 both expect two "
            "measured requests per DP rank."
        ),
    )
    parser.add_argument("--cache-hit-tolerance", type=int, default=64)
    parser.add_argument(
        "--server-log",
        type=Path,
        help=(
            "Server log used to assert one full measured prefill and full-concurrency "
            "decode. Target Falcon runners should always provide this."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    prefixes, extended = _make_inputs(
        args.concurrency,
        args.prefix_len,
        args.extend_len,
        prefix_mode=args.prefix_mode,
        prefix_group_count=args.prefix_group_count,
        random_seed=args.random_seed,
        random_token_min=args.random_token_min,
        random_token_max=args.random_token_max,
    )
    warm_branch_token = args.random_token_max - 1
    warm_inputs = _make_warm_inputs(
        prefixes,
        extended,
        warm_branch_token=warm_branch_token,
    )
    workload = _build_workload_layout(
        prefixes,
        warm_inputs=warm_inputs,
        prefix_mode=args.prefix_mode,
        prefix_group_count=args.prefix_group_count,
        dp_size=args.dp_size,
        expected_requests_per_dp=args.expected_requests_per_dp,
    )
    prefix_mode = workload["prefix_mode"]
    prefix_layout = workload["prefix_layout"]
    warm_batches = workload["warm_batches"]

    print(
        "GLM52_CACHE_BENCH_PREFIX_LAYOUT "
        + json.dumps(
            {
                "concurrency": args.concurrency,
                "dp_size": args.dp_size,
                "requests_per_dp": workload["requests_per_dp"],
                "prefix_mode": prefix_mode,
                "prefix_group_count": prefix_layout["unique_prefixes"],
                "prefix_groups": prefix_layout["prefix_groups"],
                "warm_branch_token": warm_branch_token,
                "warm_batch_sizes": [len(batch) for batch in warm_batches],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    flush = requests.post(f"{base_url}/flush_cache", timeout=60)
    flush.raise_for_status()
    server_info = _get_server_info(base_url)
    _require_idle_scheduler(server_info, phase="after cache flush")
    server_config_evidence = _validate_server_configuration(
        server_info,
        concurrency=args.concurrency,
        dp_size=args.dp_size,
        requests_per_dp=workload["requests_per_dp"],
        cached_prefixes_per_dp=workload["cached_prefixes_per_dp"],
        prefix_len=args.prefix_len,
        extend_len=args.extend_len,
        output_len=args.output_len,
    )
    print(
        "GLM52_CACHE_BENCH_SERVER_CONFIG "
        + json.dumps(server_config_evidence, sort_keys=True),
        flush=True,
    )
    warm_results = []
    for group_id, warm_inputs in enumerate(warm_batches):
        print(
            "GLM52_CACHE_BENCH_WARM_START "
            f"group={group_id} concurrency={len(warm_inputs)}",
            flush=True,
        )
        warm_results.append(
            _run_native_batch(
                base_url, warm_inputs, 1, label=f"warm-prefix-group-{group_id}"
            )
        )
        print(
            "GLM52_CACHE_BENCH_WARM_DONE "
            f"group={group_id} concurrency={len(warm_inputs)}",
            flush=True,
        )
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

    if args.profile_admission_barrier and args.disable_admission_barrier:
        raise ValueError(
            "--profile-admission-barrier conflicts with --disable-admission-barrier"
        )
    admission_barrier_enabled = not args.disable_admission_barrier
    if args.server_log is not None and not args.server_log.is_file():
        raise FileNotFoundError(f"server log does not exist: {args.server_log}")
    server_log_offset = (
        args.server_log.stat().st_size if args.server_log is not None else None
    )

    try:
        print(
            "GLM52_CACHE_BENCH_MEASURED_SUBMIT "
            f"concurrency={args.concurrency} dp_size={args.dp_size} "
            f"requests_per_dp={workload['requests_per_dp']} "
            f"prefix_mode={prefix_mode} "
            f"prefix_group_count={prefix_layout['unique_prefixes']} "
            f"admission_barrier={admission_barrier_enabled}",
            flush=True,
        )
        if admission_barrier_enabled:
            measured = _run_native_batch_with_admission_barrier(
                base_url,
                extended,
                args.output_len,
                label="cache-hit-extend-decode",
                on_admitted=(
                    start_profile if args.profile_output_dir is not None else None
                ),
                profile_settle_s=args.profile_settle_s,
                timeout_s=args.admission_timeout_s,
            )
            print(
                "GLM52_CACHE_BENCH_ADMISSION_READY "
                + json.dumps(measured["admission_evidence"], sort_keys=True),
                flush=True,
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

    server_log_evidence = None
    if args.server_log is not None:
        assert server_log_offset is not None
        server_log_evidence = _validate_measured_server_log(
            args.server_log,
            start_offset=server_log_offset,
            concurrency=args.concurrency,
            dp_size=args.dp_size,
            requests_per_dp=workload["requests_per_dp"],
            extend_len=args.extend_len,
            expected_cached_tokens=sum(measured["cached_tokens"]),
        )
        print(
            "GLM52_CACHE_BENCH_SERVER_LOG_VALIDATED "
            + json.dumps(server_log_evidence, sort_keys=True),
            flush=True,
        )

    ttft = measured["ttft_s"]
    request_to_headers = measured["request_to_headers_s"]
    headers_to_first_token = measured["headers_to_first_token_s"]
    decode = measured["decode_s"]
    tpots_ms = [value * 1000 / max(args.output_len - 1, 1) for value in decode]
    throughput_metrics = _throughput_metrics(measured)
    result = {
        "variant": args.variant,
        "concurrency": args.concurrency,
        "dp_size": args.dp_size,
        "requests_per_dp": workload["requests_per_dp"],
        "expected_requests_per_dp": args.expected_requests_per_dp,
        "prefix_mode": prefix_mode,
        "requested_prefix_mode": args.prefix_mode,
        "prefix_group_count": prefix_layout["unique_prefixes"],
        "prefix_groups": prefix_layout["prefix_groups"],
        "warm_concurrency": sum(len(batch) for batch in warm_batches),
        "warm_batch_sizes": [len(batch) for batch in warm_batches],
        "warm_branch_token": warm_branch_token,
        "prefix_len": args.prefix_len,
        "extend_len": args.extend_len,
        "output_len": args.output_len,
        "random_seed": args.random_seed,
        "random_token_min": args.random_token_min,
        "random_token_max": args.random_token_max,
        "profile_output_dir": (
            str(args.profile_output_dir)
            if args.profile_output_dir is not None
            else None
        ),
        "admission_barrier_enabled": admission_barrier_enabled,
        "profile_admission_barrier": bool(
            args.profile_output_dir is not None and admission_barrier_enabled
        ),
        "admission_evidence": measured.get("admission_evidence"),
        "server_config_evidence": server_config_evidence,
        "server_log": str(args.server_log) if args.server_log is not None else None,
        "server_log_evidence": server_log_evidence,
        "minimum_expected_cache_hit": minimum_expected_hit,
        "warm_wall_s": sum(warm["wall_s"] for warm in warm_results),
        "warm_batch_wall_s": [warm["wall_s"] for warm in warm_results],
        "wall_s": measured["wall_s"],
        "http_wall_s": measured["http_wall_s"],
        "admission_wait_s": measured["admission_wait_s"],
        "measurement_start_unix_ns": measured["measurement_start_unix_ns"],
        "measurement_origin": measured["measurement_origin"],
        "completed_requests": sum(
            value == args.output_len for value in measured["completion_tokens"]
        ),
        "ttft_mean_s": statistics.fmean(ttft),
        "ttft_p50_s": statistics.median(ttft),
        "ttft_p90_s": _percentile(ttft, 0.90),
        "ttft_p95_s": _percentile(ttft, 0.95),
        "ttft_p99_s": _percentile(ttft, 0.99),
        "ttft_max_s": max(ttft),
        "request_to_headers_p50_ms": statistics.median(request_to_headers) * 1000,
        "request_to_headers_p95_ms": _percentile(request_to_headers, 0.95) * 1000,
        "headers_to_first_token_p50_ms": statistics.median(headers_to_first_token)
        * 1000,
        "headers_to_first_token_p95_ms": (
            _percentile(headers_to_first_token, 0.95) * 1000
        ),
        "request_start_unix_ns": measured["request_start_unix_ns"],
        "response_headers_unix_ns": measured["response_headers_unix_ns"],
        "first_token_unix_ns": measured["first_token_unix_ns"],
        "decode_mean_s": statistics.fmean(decode),
        "decode_p50_s": statistics.median(decode),
        "tpot_mean_ms": statistics.fmean(tpots_ms),
        "tpot_p50_ms": statistics.median(tpots_ms),
        "tpot_p90_ms": _percentile(tpots_ms, 0.90),
        "tpot_p95_ms": _percentile(tpots_ms, 0.95),
        "tpot_p99_ms": _percentile(tpots_ms, 0.99),
        "tpot_max_ms": max(tpots_ms),
        **throughput_metrics,
        "cached_tokens_min": min(measured["cached_tokens"]),
        "cached_tokens_max": max(measured["cached_tokens"]),
        "cached_tokens": measured["cached_tokens"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
