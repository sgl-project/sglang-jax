#!/usr/bin/env python3
"""Baseline cache miss sweep for cache-miss-v3 (option C base).

Run on pod inside /tmp/sglang-jax-baseline/.venv :
    python3 baseline_sweep.py --dp 1 --out /tmp/baseline_dp1.jsonl
    python3 baseline_sweep.py --dp 2 --out /tmp/baseline_dp2.jsonl

Assumes server already running on localhost:8000.
"""

import argparse
import concurrent.futures
import json
import pathlib
import sys
import time

import requests

# Filler text for prompt construction (English, ~4 chars/token rule of thumb)
FILLER = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen "
    "liquor jugs. How vexingly quick daft zebras jump. Sphinx of black quartz "
    "judge my vow. Two driven jocks help fax my big quiz. Five quacking zephyrs "
    "jolt my wax bed. The five boxing wizards jump quickly. Jackdaws love my "
    "big sphinx of quartz. We promptly judged antique ivory buckles for the "
    "next prize. A wizard's job is to vex chumps quickly in fog. Watch Jeopardy "
    "Alex Trebek's fun TV quiz game. Heavy boxes perform quick waltzes and "
    "jigs. Crazy Fredrick bought many very exquisite opal jewels. "
) * 50  # ~6500 chars, enough for 4096 tokens worth


def make_prompt(target_tokens: int) -> str:
    """Approximate target_tokens via char count (1 token ~ 4 chars)."""
    target_chars = target_tokens * 4
    return FILLER[:target_chars]


DP1_CELLS = (
    [(128, b) for b in range(1, 17)]
    + [(256, b) for b in range(1, 17)]
    + [(500, b) for b in range(1, 9)]
    + [(1024, b) for b in range(1, 5)]
    + [(1500, b) for b in [1, 2]]
    + [(2048, b) for b in [1, 2]]
    + [(3000, b) for b in [1]]
    + [(4096, b) for b in [1]]
)

DP2_CELLS = (
    [(128, b) for b in [2, 4, 6, 8, 10, 12, 14, 16]]
    + [(256, b) for b in [2, 4, 6, 8, 10, 12, 14, 16]]
    + [(500, b) for b in [2, 4, 6, 8]]
    + [(1024, b) for b in [2, 4]]
    + [(1500, b) for b in [2]]
    + [(2048, b) for b in [2]]
)


def send_one(prompt: str, max_new_tokens: int) -> dict:
    try:
        r = requests.post(
            "http://127.0.0.1:8000/generate",
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            },
            timeout=1800,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def run_cell(seqlen: int, bsz: int, max_new_tokens: int) -> dict:
    prompt = make_prompt(seqlen)
    t0 = time.perf_counter()
    if bsz == 1:
        results = [send_one(prompt, max_new_tokens)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=bsz) as ex:
            results = list(ex.map(lambda _: send_one(prompt, max_new_tokens), range(bsz)))
    elapsed = time.perf_counter() - t0

    first_tokens = []
    output_id_lists = []
    cache_miss_counts = []
    prompt_tokens = []
    completion_tokens = []
    errors = []
    for r in results:
        if "error" in r:
            errors.append(r["error"])
            continue
        ids = r.get("output_ids") or []
        first_tokens.append(ids[0] if ids else None)
        output_id_lists.append(ids)
        meta = r.get("meta_info") or {}
        cache_miss_counts.append(meta.get("cache_miss_count"))
        prompt_tokens.append(meta.get("prompt_tokens"))
        completion_tokens.append(meta.get("completion_tokens"))
    return {
        "seqlen_target": seqlen,
        "bsz": bsz,
        "max_new_tokens": max_new_tokens,
        "elapsed_s": round(elapsed, 3),
        "first_tokens": first_tokens,
        "output_ids": output_id_lists,
        "cache_miss_counts": cache_miss_counts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "errors": errors,
        "all_same_first_token": len(set(first_tokens)) == 1 if first_tokens else None,
        "all_same_output": (
            len({tuple(o) for o in output_id_lists}) == 1 if output_id_lists else None
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=int, required=True, choices=[1, 2])
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1)
    args = ap.parse_args()

    cells = DP1_CELLS if args.dp == 1 else DP2_CELLS
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for i, (seqlen, bsz) in enumerate(cells):
            print(
                f"[{i+1}/{len(cells)}] dp={args.dp} seqlen={seqlen} bsz={bsz} mnt={args.max_new_tokens} ...",
                flush=True,
            )
            row = run_cell(seqlen, bsz, args.max_new_tokens)
            row["dp_size"] = args.dp
            f.write(json.dumps(row) + "\n")
            f.flush()
            time.sleep(1)


if __name__ == "__main__":
    main()
