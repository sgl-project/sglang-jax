#!/usr/bin/env python3
"""GSM8K evaluation via PD disaggregation.

Sends paired P+D requests for each question. Accuracy and throughput
measured from the D-side responses.

Usage:
    python scripts/pd_bench_gsm8k.py \
        --prefill-url http://<P>:10000 \
        --decode-url  http://<D>:10001 \
        --num-questions 200 --parallel 16
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
import uuid

import aiohttp
import numpy as np
from tqdm import tqdm

INVALID = -9999999


def _download_and_cache(url: str) -> str:
    cache_dir = os.path.join(tempfile.gettempdir(), "sgl_jax_bench_cache")
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split("/")[-1]
    path = os.path.join(cache_dir, filename)
    if not os.path.isfile(path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)
    return path


def _read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _get_answer_value(s: str):
    s = s.replace(",", "")
    numbers = re.findall(r"\d+", s)
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def _build_prompt(lines, idx, few_shot_lines, num_shots):
    shots = ""
    for i in range(num_shots):
        shots += f"Question: {few_shot_lines[i]['question']}\nAnswer: {few_shot_lines[i]['answer']}\n\n"
    return shots + f"Question: {lines[idx]['question']}\nAnswer:"


async def _send_pd(
    session: aiohttp.ClientSession,
    p_url: str,
    d_url: str,
    text: str,
    max_new_tokens: int,
    transfer_id: str,
    room: int,
    sem: asyncio.Semaphore,
    pbar: tqdm,
):
    payload = {
        "text": text,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "stop": ["Question", "Assistant:", "<|separator|>"],
        },
        "bootstrap_room": room,
        "disagg_transfer_id": transfer_id,
    }

    async with sem:
        timeout = aiohttp.ClientTimeout(total=300)

        async def _post(url):
            async with session.post(
                f"{url}/generate", json=payload, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                return await resp.json()

        p_task = asyncio.create_task(_post(p_url))
        await asyncio.sleep(0.5)
        d_task = asyncio.create_task(_post(d_url))
        _, d_resp = await asyncio.gather(p_task, d_task)
        pbar.update(1)
        return d_resp


async def run_eval(args):
    # Load data
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = args.data_path
    if not os.path.isfile(data_path):
        data_path = _download_and_cache(url)
    lines = list(_read_jsonl(data_path))

    num_q = min(args.num_questions, len(lines))
    prompts = [_build_prompt(lines, i, lines, args.num_shots) for i in range(num_q)]
    labels = [_get_answer_value(lines[i]["answer"]) for i in range(num_q)]

    p_url = args.prefill_url.rstrip("/")
    d_url = args.decode_url.rstrip("/")

    print(f"Running {num_q} GSM8K questions via PD (parallel={args.parallel})")
    sem = asyncio.Semaphore(args.parallel)
    pbar = tqdm(total=num_q, desc="GSM8K-PD")

    async with aiohttp.ClientSession() as session:
        tasks = [
            _send_pd(
                session, p_url, d_url,
                prompts[i], args.max_new_tokens,
                f"gsm-{uuid.uuid4().hex[:8]}", i,
                sem, pbar,
            )
            for i in range(num_q)
        ]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        latency = time.perf_counter() - t0

    pbar.close()

    preds = []
    errors = 0
    total_output_tokens = 0
    for r in results:
        if isinstance(r, Exception):
            preds.append(INVALID)
            errors += 1
        else:
            preds.append(_get_answer_value(r.get("text", "")))
            total_output_tokens += r.get("meta_info", {}).get(
                "completion_tokens", 0
            )

    acc = np.mean(np.array(preds) == np.array(labels))
    invalid_rate = np.mean(np.array(preds) == INVALID)
    throughput = total_output_tokens / latency if latency > 0 else 0

    print(f"\nAccuracy: {acc:.3f}")
    print(f"Invalid: {invalid_rate:.3f}")
    print(f"Errors: {errors}")
    print(f"Latency: {latency:.1f}s")
    print(f"Output throughput: {throughput:.1f} tok/s")

    if args.output_file:
        with open(args.output_file, "w") as f:
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    f.write(f"=== Q{i} === ERROR: {r}\n\n")
                else:
                    f.write(f"=== Q{i} === pred={preds[i]} label={labels[i]}\n")
                    f.write(r.get("text", "") + "\n\n")
        print(f"Outputs saved to {args.output_file}")

    summary = {
        "task": "gsm8k-pd",
        "accuracy": round(acc, 3),
        "invalid_rate": round(invalid_rate, 3),
        "errors": errors,
        "num_questions": num_q,
        "latency_s": round(latency, 1),
        "output_throughput_tok_s": round(throughput, 1),
        "parallel": args.parallel,
    }
    if args.result_file:
        with open(args.result_file, "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"Summary appended to {args.result_file}")

    return 0 if acc > 0 else 1


def main():
    parser = argparse.ArgumentParser(description="GSM8K PD evaluation")
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--result-file", type=str, default="gsm8k_pd_results.jsonl")
    args = parser.parse_args()
    return asyncio.run(run_eval(args))


if __name__ == "__main__":
    sys.exit(main())
