#!/usr/bin/env python3
"""Collect multi-item scoring evaluation metrics for one endpoint."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import requests

DEFAULT_QUERY_IDS = [1957, 1437, 25975, 25]
DEFAULT_ITEM_IDS = [
    [358, 2948, 419, 1985, 13],
    [1096, 374, 17478, 323, 38123, 13],
    [1084, 4278, 438, 3601, 13],
    [56938, 4271, 323, 4937, 9691, 13],
    [2806, 5802, 279, 3349, 13],
]
DEFAULT_LABEL_TOKEN_IDS = [9834, 902]


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


def max_abs_diff(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} != {len(vec_b)}")
    return max(abs(a - b) for a, b in zip(vec_a, vec_b))


def call_score(
    *,
    url: str,
    model: str,
    query_ids: list[int],
    item_ids: list[list[int]],
    label_token_ids: list[int],
    timeout: float,
) -> list[list[float]]:
    payload = {
        "model": model,
        "query": query_ids,
        "items": item_ids,
        "label_token_ids": label_token_ids,
        "apply_softmax": True,
        "item_first": False,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Scoring endpoint failed ({resp.status_code}) for {url}: {resp.text[:400]}"
        )
    body = resp.json()
    scores = body.get("scores")
    if not isinstance(scores, list):
        raise RuntimeError(f"Malformed response from {url}: {body}")
    return scores


def score_items(
    *,
    mode: str,
    url: str,
    model: str,
    query_ids: list[int],
    item_ids: list[list[int]],
    label_token_ids: list[int],
    timeout: float,
) -> list[list[float]]:
    if mode == "multi":
        return call_score(
            url=url,
            model=model,
            query_ids=query_ids,
            item_ids=item_ids,
            label_token_ids=label_token_ids,
            timeout=timeout,
        )
    if mode == "serial":
        scores: list[list[float]] = []
        for item in item_ids:
            one = call_score(
                url=url,
                model=model,
                query_ids=query_ids,
                item_ids=[item],
                label_token_ids=label_token_ids,
                timeout=timeout,
            )
            if len(one) != 1:
                raise RuntimeError(f"Expected one score row, got {len(one)}")
            scores.append(one[0])
        return scores
    raise ValueError(f"Unsupported mode: {mode}")


def build_perf_items(n: int) -> list[list[int]]:
    return [
        copy.deepcopy(DEFAULT_ITEM_IDS[i % len(DEFAULT_ITEM_IDS)]) for i in range(n)
    ]


def evaluate_performance(
    *,
    mode: str,
    url: str,
    model: str,
    query_ids: list[int],
    label_token_ids: list[int],
    counts: list[int],
    rounds: int,
    warmup: int,
    timeout: float,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for n in counts:
        items = build_perf_items(n)

        for _ in range(warmup):
            score_items(
                mode=mode,
                url=url,
                model=model,
                query_ids=query_ids,
                item_ids=items,
                label_token_ids=label_token_ids,
                timeout=timeout,
            )

        latencies: list[float] = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            score_items(
                mode=mode,
                url=url,
                model=model,
                query_ids=query_ids,
                item_ids=items,
                label_token_ids=label_token_ids,
                timeout=timeout,
            )
            latencies.append(time.perf_counter() - t0)

        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        out[str(n)] = {
            "latencies_sec": latencies,
            "p50_sec": p50,
            "p95_sec": p95,
            "items_per_sec_p50": (n / p50) if p50 > 0 else 0.0,
            "items_per_sec_p95": (n / p95) if p95 > 0 else 0.0,
        }
    return out


def evaluate_mutations(
    *,
    mode: str,
    url: str,
    model: str,
    query_ids: list[int],
    label_token_ids: list[int],
    timeout: float,
) -> dict:
    base_items = copy.deepcopy(DEFAULT_ITEM_IDS)
    base_scores = score_items(
        mode=mode,
        url=url,
        model=model,
        query_ids=query_ids,
        item_ids=base_items,
        label_token_ids=label_token_ids,
        timeout=timeout,
    )
    if len(base_scores) != len(base_items):
        raise RuntimeError(f"Expected {len(base_items)} scores, got {len(base_scores)}")

    mutated_idx = 1

    same_len_items = copy.deepcopy(base_items)
    same_len_items[mutated_idx][0] = same_len_items[mutated_idx][0] + 1
    same_len_scores = score_items(
        mode=mode,
        url=url,
        model=model,
        query_ids=query_ids,
        item_ids=same_len_items,
        label_token_ids=label_token_ids,
        timeout=timeout,
    )

    changed_len_items = copy.deepcopy(base_items)
    changed_len_items[mutated_idx] = changed_len_items[mutated_idx] + [13]
    changed_len_scores = score_items(
        mode=mode,
        url=url,
        model=model,
        query_ids=query_ids,
        item_ids=changed_len_items,
        label_token_ids=label_token_ids,
        timeout=timeout,
    )

    same_unchanged_diffs: list[float] = []
    changed_unchanged_diffs: list[float] = []
    for i in range(len(base_items)):
        if i == mutated_idx:
            continue
        same_unchanged_diffs.append(max_abs_diff(base_scores[i], same_len_scores[i]))
        changed_unchanged_diffs.append(
            max_abs_diff(base_scores[i], changed_len_scores[i])
        )

    return {
        "query_ids": query_ids,
        "item_ids": base_items,
        "base_scores": base_scores,
        "same_length_mutation": {
            "mutated_index": mutated_idx,
            "mutated_item_ids": same_len_items[mutated_idx],
            "scores": same_len_scores,
            "unchanged_item_abs_diffs": same_unchanged_diffs,
            "max_abs_diff_unchanged": (
                max(same_unchanged_diffs) if same_unchanged_diffs else 0.0
            ),
            "mean_abs_diff_unchanged": (
                sum(same_unchanged_diffs) / len(same_unchanged_diffs)
                if same_unchanged_diffs
                else 0.0
            ),
        },
        "changed_length_mutation": {
            "mutated_index": mutated_idx,
            "mutated_item_ids": changed_len_items[mutated_idx],
            "scores": changed_len_scores,
            "unchanged_item_abs_diffs": changed_unchanged_diffs,
            "max_abs_diff_unchanged": (
                max(changed_unchanged_diffs) if changed_unchanged_diffs else 0.0
            ),
            "mean_abs_diff_unchanged": (
                sum(changed_unchanged_diffs) / len(changed_unchanged_diffs)
                if changed_unchanged_diffs
                else 0.0
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["multi", "serial"], required=True)
    parser.add_argument(
        "--url",
        required=True,
        help="Score endpoint URL, e.g. http://127.0.0.1:30010/v1/score",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--item-counts", default="1,8,32,64,128")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write endpoint-specific evaluation JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = [int(x) for x in args.item_counts.split(",") if x.strip()]
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mutation_data = evaluate_mutations(
        mode=args.mode,
        url=args.url,
        model=args.model,
        query_ids=DEFAULT_QUERY_IDS,
        label_token_ids=DEFAULT_LABEL_TOKEN_IDS,
        timeout=args.timeout,
    )
    perf_data = evaluate_performance(
        mode=args.mode,
        url=args.url,
        model=args.model,
        query_ids=DEFAULT_QUERY_IDS,
        label_token_ids=DEFAULT_LABEL_TOKEN_IDS,
        counts=counts,
        rounds=args.rounds,
        warmup=args.warmup,
        timeout=args.timeout,
    )

    semantic_alignment = None
    if args.mode == "serial":
        query_plus_delimiter_ids = DEFAULT_QUERY_IDS + [151643]
        base_scores_query_plus_delimiter = score_items(
            mode="serial",
            url=args.url,
            model=args.model,
            query_ids=query_plus_delimiter_ids,
            item_ids=mutation_data["item_ids"],
            label_token_ids=DEFAULT_LABEL_TOKEN_IDS,
            timeout=args.timeout,
        )
        semantic_alignment = {
            "query_plus_delimiter_ids": query_plus_delimiter_ids,
            "base_scores_query_plus_delimiter": base_scores_query_plus_delimiter,
        }

    out = {
        "mode": args.mode,
        "url": args.url,
        "model": args.model,
        "labels": DEFAULT_LABEL_TOKEN_IDS,
        "delimiter": 151643,
        "isolation": mutation_data,
        "performance": perf_data,
    }
    if semantic_alignment is not None:
        out["semantic_alignment"] = semantic_alignment
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({"output_json": str(output_path), "mode": args.mode}))


if __name__ == "__main__":
    main()
