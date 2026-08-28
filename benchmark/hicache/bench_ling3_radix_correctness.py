"""Concurrent radix-cache correctness probe for Ling-3 hybrid KDA + MLA.

The probe points at an already-running server. It stresses the recurrent
extra-buffer path with unrelated prefix families, sibling branches, exact
replays, several prefix depths, and multiple concurrency levels. For each
level it verifies:

1. unrelated anchors are cold after ``/flush_cache``;
2. divergent sibling prompts reuse the anchor's recurrent prefix;
3. same-order concurrent replays preserve output IDs and keep the prefix hit;
4. shuffled replays keep prefix hits under a different batch composition;
5. a post-flush same-order cold round is byte-identical to the hit result.

Example::

    PYTHONPATH=python python benchmark/hicache/bench_ling3_radix_correctness.py \
      --server-url http://127.0.0.1:30000 \
      --parallel-list 8 32 64 \
      --output-json /tmp/ling3_radix_correctness.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass

import requests

from sgl_jax.test.kits.cache_hit_kit import (
    async_request_sglang_generate,
    flush_cache,
    gen_payload,
)


@dataclass(frozen=True)
class ProbePrompt:
    family: int
    branch: int
    shared_tokens: int
    input_ids: list[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ling-3 concurrent recurrent-radix correctness probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--parallel-list", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--output-length", type=int, default=8)
    parser.add_argument("--expected-dp-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _server_contract(base_url: str, expected_dp_size: int) -> tuple[int, int]:
    response = requests.get(f"{base_url}/get_server_info", timeout=30)
    response.raise_for_status()
    info = response.json()

    assert info.get("disable_radix_cache") is False, "radix cache is disabled"
    assert info.get("enable_unified_radix_tree") is True, (
        "unified radix tree is not enabled"
    )
    assert info.get("enable_recurrent_extra_buffer") is True, (
        "recurrent extra buffer is required for Ling-3 with page_size > 1"
    )
    assert info.get("disable_overlap_schedule") is False, (
        "Ling-3 radix validation must run with overlap scheduling enabled"
    )
    assert info.get("dp_size") == expected_dp_size, (
        f"server dp_size={info.get('dp_size')}, expected {expected_dp_size}"
    )

    page_size = int(info["page_size"])
    track_interval = int(info["recurrent_track_interval"])
    assert page_size > 1
    assert track_interval > 0 and track_interval % page_size == 0
    return page_size, track_interval


def _token_sequence(seed: int, length: int) -> list[int]:
    """Produce valid, deterministic non-special token IDs with no common head."""
    rng = random.Random(seed)
    return [1000 + rng.randrange(30000) for _ in range(length)]


def build_workload(
    *,
    families: int,
    branches: int,
    page_size: int,
    track_interval: int,
    seed: int,
) -> tuple[list[ProbePrompt], list[ProbePrompt]]:
    """Return one anchor and several divergent siblings per prefix family."""
    anchors = []
    probes = []
    for family in range(families):
        # Exercise one through four recurrent checkpoints, with an extra page
        # beyond the last checkpoint so every branch can diverge before the next.
        depth = 1 + family % 4
        shared_tokens = depth * track_interval + page_size
        shared = _token_sequence(seed + family * 1009, shared_tokens)

        anchor_suffix = _token_sequence(
            seed + family * 1009 + 1, max(8, page_size // 8)
        )
        anchors.append(
            ProbePrompt(
                family=family,
                branch=-1,
                shared_tokens=shared_tokens,
                input_ids=shared + anchor_suffix,
            )
        )

        for branch in range(branches):
            # Keep the suffix below the remaining distance to the next track
            # boundary: siblings share checkpoints but not their terminal page.
            suffix_len = min(page_size - 1, 17 + 7 * branch)
            suffix = _token_sequence(seed + family * 1009 + 100 + branch, suffix_len)
            probes.append(
                ProbePrompt(
                    family=family,
                    branch=branch,
                    shared_tokens=shared_tokens,
                    input_ids=shared + suffix,
                )
            )
    return anchors, probes


async def _send(payloads, url: str, parallel: int):
    semaphore = asyncio.Semaphore(parallel)

    async def one(payload):
        async with semaphore:
            return await async_request_sglang_generate(payload, url)

    return await asyncio.gather(
        *[asyncio.create_task(one(payload)) for payload in payloads]
    )


def _run_requests(prompts, generate_url, parallel, output_length):
    payloads = [gen_payload(prompt.input_ids, output_length) for prompt in prompts]
    start = time.perf_counter()
    responses = asyncio.run(_send(payloads, generate_url, parallel))
    wall_time = time.perf_counter() - start
    for prompt, response in zip(prompts, responses):
        assert response.success, (
            f"family={prompt.family} branch={prompt.branch} failed: {response.error}"
        )
    return responses, wall_time


def run_level(args, page_size: int, track_interval: int, parallel: int) -> dict:
    anchors, probes = build_workload(
        families=args.families,
        branches=args.branches,
        page_size=page_size,
        track_interval=track_interval,
        seed=args.seed,
    )
    generate_url = f"{args.server_url}/generate"

    flush_cache(args.server_url)
    anchor_responses, _ = _run_requests(
        anchors,
        generate_url,
        min(parallel, len(anchors)),
        args.output_length,
    )
    assert all(response.cached_tokens == 0 for response in anchor_responses), (
        "unrelated anchors must be cold immediately after flush"
    )

    first_responses, first_wall = _run_requests(
        probes, generate_url, parallel, args.output_length
    )
    for prompt, response in zip(probes, first_responses):
        expected_floor = (prompt.shared_tokens // track_interval) * track_interval
        assert response.cached_tokens >= expected_floor, (
            f"family={prompt.family} branch={prompt.branch}: cached_tokens="
            f"{response.cached_tokens}, expected at least {expected_floor}"
        )
        assert response.cached_tokens <= response.prompt_len

    expected_output = {
        (prompt.family, prompt.branch): tuple(response.output_ids)
        for prompt, response in zip(probes, first_responses)
    }
    first_response = {
        (prompt.family, prompt.branch): response
        for prompt, response in zip(probes, first_responses)
    }
    # Keep request order and concurrency identical for the byte-exact replay.
    # TPU matmul schedules can change at batch-composition boundaries; ordering
    # differences are stressed separately below without conflating those
    # numerical differences with recurrent-state corruption.
    replay_responses, replay_wall = _run_requests(
        probes, generate_url, parallel, args.output_length
    )
    for prompt, response in zip(probes, replay_responses):
        key = (prompt.family, prompt.branch)
        expected_ids = expected_output[key]
        actual_ids = tuple(response.output_ids)
        assert actual_ids == expected_ids, (
            f"family={prompt.family} branch={prompt.branch}: hit output differs "
            f"from first output; expected_output_ids={expected_ids}, "
            f"actual_output_ids={actual_ids}, "
            f"first_cached_tokens={first_response[key].cached_tokens}, "
            f"replay_cached_tokens={response.cached_tokens}, "
            f"prompt_len={response.prompt_len}"
        )
        expected_floor = (prompt.shared_tokens // track_interval) * track_interval
        assert response.cached_tokens >= expected_floor

    shuffled = list(probes)
    random.Random(args.seed + parallel).shuffle(shuffled)
    shuffled_responses, shuffled_wall = _run_requests(
        shuffled, generate_url, parallel, args.output_length
    )
    for prompt, response in zip(shuffled, shuffled_responses):
        expected_floor = (prompt.shared_tokens // track_interval) * track_interval
        assert response.cached_tokens >= expected_floor, (
            f"family={prompt.family} branch={prompt.branch}: shuffled replay "
            f"cached_tokens={response.cached_tokens}, expected at least {expected_floor}"
        )

    # Compare the complete workload against a truly cold post-flush execution,
    # preserving request order and concurrency. This catches recurrent snapshot
    # corruption that an exact replay alone could reproduce twice.
    flush_cache(args.server_url)
    cold_responses, _ = _run_requests(
        probes,
        generate_url,
        parallel,
        args.output_length,
    )
    for prompt, response in zip(probes, cold_responses):
        assert response.cached_tokens == 0
        key = (prompt.family, prompt.branch)
        expected_ids = expected_output[key]
        actual_ids = tuple(response.output_ids)
        assert actual_ids == expected_ids, (
            f"family={prompt.family}: cold output differs from radix-hit output; "
            f"expected_output_ids={expected_ids}, actual_output_ids={actual_ids}, "
            f"first_cached_tokens={first_response[key].cached_tokens}, "
            f"cold_cached_tokens={response.cached_tokens}, prompt_len={response.prompt_len}"
        )

    first_ttft = [response.ttft for response in first_responses]
    replay_ttft = [response.ttft for response in replay_responses]
    return {
        "parallel": parallel,
        "requests_per_probe_round": len(probes),
        "prefix_families": args.families,
        "branches_per_family": args.branches,
        "first_cached_tokens": sum(
            response.cached_tokens for response in first_responses
        ),
        "replay_cached_tokens": sum(
            response.cached_tokens for response in replay_responses
        ),
        "shuffled_cached_tokens": sum(
            response.cached_tokens for response in shuffled_responses
        ),
        "first_ttft_p50_ms": statistics.median(first_ttft) * 1000,
        "replay_ttft_p50_ms": statistics.median(replay_ttft) * 1000,
        "first_throughput_req_s": len(first_responses) / first_wall,
        "replay_throughput_req_s": len(replay_responses) / replay_wall,
        "shuffled_throughput_req_s": len(shuffled_responses) / shuffled_wall,
        "correctness": "pass",
    }


def main():
    args = parse_args()
    page_size, track_interval = _server_contract(args.server_url, args.expected_dp_size)
    results = {
        "server_url": args.server_url,
        "page_size": page_size,
        "recurrent_track_interval": track_interval,
        "levels": [],
    }
    for parallel in args.parallel_list:
        print(f"[Ling-3 radix] parallel={parallel}")
        level = run_level(args, page_size, track_interval, parallel)
        results["levels"].append(level)
        print(json.dumps(level, indent=2, sort_keys=True))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2, sort_keys=True)
    print("[Ling-3 radix] PASS")


if __name__ == "__main__":
    main()
