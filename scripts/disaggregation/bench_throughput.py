#!/usr/bin/env python3
"""PD benchmark: throughput measurement for disaggregated prefill-decode.

Sends paired requests to both P and D servers with matching transfer IDs,
measuring decode-side throughput across different input/output sizes and
concurrency levels.

Usage:
    python scripts/disaggregation/bench_throughput.py \
        --prefill-url http://<P>:10000 \
        --decode-url  http://<D>:10001 \
        --configs "512:1024:16,1024:1024:32,2048:1024:32,4096:1024:64"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid

import aiohttp
import numpy as np


async def _send_pd_pair(
    session: aiohttp.ClientSession,
    p_url: str,
    d_url: str,
    prompt_text: str,
    max_new_tokens: int,
    transfer_id: str,
    bootstrap_room: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Send paired P+D request, return D response with timing."""
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "bootstrap_room": bootstrap_room,
        "disagg_transfer_id": transfer_id,
    }

    async with semaphore:
        t0 = time.perf_counter()

        async def _post(url):
            timeout = aiohttp.ClientTimeout(total=600)
            async with session.post(f"{url}/generate", json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                return await resp.json()

        # Send P first, then D after brief delay
        p_task = asyncio.create_task(_post(p_url))
        await asyncio.sleep(0.5)
        d_task = asyncio.create_task(_post(d_url))

        _, d_resp = await asyncio.gather(p_task, d_task)
        latency = time.perf_counter() - t0

        meta = d_resp.get("meta_info", {})
        return {
            "latency": latency,
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "text": d_resp.get("text", ""),
        }


def _generate_random_prompt(tokenizer, target_len: int) -> str:
    """Generate a prompt of approximately target_len tokens."""
    vocab_size = tokenizer.vocab_size
    token_ids = [random.randint(10, vocab_size - 1) for _ in range(target_len)]
    return tokenizer.decode(token_ids)


async def run_benchmark(
    p_url: str,
    d_url: str,
    input_len: int,
    output_len: int,
    concurrency: int,
    num_requests: int,
    tokenizer,
) -> dict:
    """Run one benchmark config and return stats."""
    semaphore = asyncio.Semaphore(concurrency)
    prompts = [_generate_random_prompt(tokenizer, input_len) for _ in range(num_requests)]

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(prompts):
            tid = f"bench-{uuid.uuid4().hex[:12]}"
            tasks.append(
                _send_pd_pair(
                    session,
                    p_url,
                    d_url,
                    prompt,
                    output_len,
                    tid,
                    bootstrap_room=i,
                    semaphore=semaphore,
                )
            )

        t_start = time.perf_counter()
        results = []
        completed = 0
        errors = 0
        for coro in asyncio.as_completed(tasks):
            try:
                r = await coro
                results.append(r)
                completed += 1
            except Exception as e:
                errors += 1
                print(f"  request error: {e}")
        t_total = time.perf_counter() - t_start

    if not results:
        return {
            "input_len": input_len,
            "output_len": output_len,
            "concurrency": concurrency,
            "error": "all requests failed",
        }

    latencies = [r["latency"] for r in results]
    total_output_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)

    return {
        "input_len": input_len,
        "output_len": output_len,
        "concurrency": concurrency,
        "num_requests": num_requests,
        "completed": completed,
        "errors": errors,
        "total_time_s": round(t_total, 2),
        "output_throughput_tok_s": round(total_output_tokens / t_total, 1),
        "total_throughput_tok_s": round((total_prompt_tokens + total_output_tokens) / t_total, 1),
        "avg_latency_s": round(np.mean(latencies), 2),
        "p50_latency_s": round(np.percentile(latencies, 50), 2),
        "p99_latency_s": round(np.percentile(latencies, 99), 2),
        "avg_output_tokens": round(np.mean([r["completion_tokens"] for r in results]), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="PD throughput benchmark")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument(
        "--configs",
        type=str,
        default="512:1024:16,1024:1024:32,2048:1024:32,4096:1024:64",
        help="Comma-separated configs as input_len:output_len:concurrency",
    )
    parser.add_argument(
        "--num-requests", type=int, default=32, help="Number of requests per config"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="Tokenizer path (defaults to model from server)"
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    # Load tokenizer
    model_path = args.model_path or args.tokenizer
    if model_path is None:
        # Try to get from server
        import requests as req

        try:
            info = req.get(f"{args.decode_url}/get_model_info", timeout=10).json()
            model_path = info.get("model_path", "Qwen/Qwen3-8B-Base")
        except Exception:
            model_path = "Qwen/Qwen3-8B-Base"

    from transformers import AutoTokenizer

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    configs = []
    for c in args.configs.split(","):
        parts = c.strip().split(":")
        configs.append((int(parts[0]), int(parts[1]), int(parts[2])))

    p_url = args.prefill_url.rstrip("/")
    d_url = args.decode_url.rstrip("/")

    print(f"P: {p_url}")
    print(f"D: {d_url}")
    print(f"Configs: {configs}")
    print(f"Requests per config: {args.num_requests}")
    print()

    all_results = []
    for input_len, output_len, conc in configs:
        print(f"--- input={input_len}, output={output_len}, concurrency={conc} ---")
        result = asyncio.run(
            run_benchmark(p_url, d_url, input_len, output_len, conc, args.num_requests, tokenizer)
        )
        all_results.append(result)
        print(f"  throughput: {result.get('output_throughput_tok_s', 'N/A')} output tok/s")
        print(f"  total throughput: {result.get('total_throughput_tok_s', 'N/A')} tok/s")
        print(f"  avg latency: {result.get('avg_latency_s', 'N/A')}s")
        print(f"  p99 latency: {result.get('p99_latency_s', 'N/A')}s")
        print(f"  completed: {result.get('completed', 0)}/{args.num_requests}")
        print()

    # Summary table
    print("=" * 80)
    print(
        f"{'Input':>6} {'Output':>7} {'Conc':>5} {'OutTok/s':>10} "
        f"{'TotalTok/s':>11} {'AvgLat':>8} {'P99Lat':>8} {'OK':>4}"
    )
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['input_len']:>6} {r['output_len']:>7} {r['concurrency']:>5} "
            f"{r.get('output_throughput_tok_s', 'ERR'):>10} "
            f"{r.get('total_throughput_tok_s', 'ERR'):>11} "
            f"{r.get('avg_latency_s', 'ERR'):>8} "
            f"{r.get('p99_latency_s', 'ERR'):>8} "
            f"{r.get('completed', 0):>4}"
        )
    print("=" * 80)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    failed = any(r.get("error") for r in all_results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
