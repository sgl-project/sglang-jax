#!/usr/bin/env python3
"""Compare a running SGLang-JAX Ling3 Tiny server with Torch CPU goldens."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_dump_manifest(dump_dir: Path) -> list[dict]:
    records = []
    for manifest in sorted(dump_dir.glob("manifest-p*.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return sorted(records, key=lambda record: (record["process_id"], record["index"]))


def _wait_for_prefill_logits(
    dump_dir: Path,
    minimum_index: int,
    timeout: float = 30.0,
) -> np.ndarray:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        records = _read_dump_manifest(dump_dir)
        matches = [
            record
            for record in records
            if record["index"] > minimum_index
            and record["component"] == "ling3_io"
            and record["name"] == "next_token_logits"
            and "extend" in (record.get("forward_mode") or "")
        ]
        if matches:
            array = np.load(dump_dir / matches[0]["filename"])
            if array.ndim == 2:
                array = array[0]
            return np.asarray(array, dtype=np.float32)
        time.sleep(0.25)
    raise TimeoutError("Timed out waiting for the JAX prefill logits dump")


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    return (shifted - math.log(float(np.exp(shifted).sum()))).astype(np.float32)


def _raw_logit_metrics(golden: np.ndarray, actual: np.ndarray, top_k: int) -> dict:
    if golden.shape != actual.shape:
        raise ValueError(f"Logits shape mismatch: golden={golden.shape}, JAX={actual.shape}")
    difference = actual.astype(np.float64) - golden.astype(np.float64)
    denominator = float(np.linalg.norm(golden) * np.linalg.norm(actual))
    cosine = float(np.dot(golden, actual) / denominator) if denominator else 1.0

    golden_logprobs = _log_softmax(golden)
    actual_logprobs = _log_softmax(actual)
    golden_ids = np.argsort(golden_logprobs)[-top_k:]
    actual_ids = np.argsort(actual_logprobs)[-top_k:]
    overlap = len(set(golden_ids.tolist()) & set(actual_ids.tolist())) / top_k
    common_ids = np.asarray(sorted(set(golden_ids.tolist()) & set(actual_ids.tolist())))
    common_delta = (
        float(np.max(np.abs(golden_logprobs[common_ids] - actual_logprobs[common_ids])))
        if common_ids.size
        else math.inf
    )
    return {
        "cosine_similarity": cosine,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference))),
        "topk_overlap": overlap,
        "topk_common_max_logprob_delta": common_delta,
        "golden_argmax": int(np.argmax(golden)),
        "jax_argmax": int(np.argmax(actual)),
    }


def _server_step_metrics(golden: dict[str, np.ndarray], response: dict, top_k: int) -> dict:
    output_ids = np.asarray(response["output_ids"], dtype=np.int32)
    expected_ids = golden["greedy_token_ids"]
    exact_tokens = bool(np.array_equal(output_ids, expected_ids))

    output_top = response["meta_info"]["output_top_logprobs"]
    step_overlaps = []
    step_common_deltas = []
    for step, row in enumerate(output_top[: expected_ids.shape[0]]):
        actual = {int(item[1]): float(item[0]) for item in row}
        expected = {
            int(token_id): float(logprob)
            for token_id, logprob in zip(
                golden["step_topk_ids"][step], golden["step_topk_logprobs"][step]
            )
        }
        common = set(actual) & set(expected)
        step_overlaps.append(len(common) / top_k)
        step_common_deltas.append(
            max(abs(actual[token_id] - expected[token_id]) for token_id in common)
            if common
            else math.inf
        )
    return {
        "greedy_token_ids_exact": exact_tokens,
        "expected_token_ids": expected_ids.tolist(),
        "jax_token_ids": output_ids.tolist(),
        "min_step_topk_overlap": min(step_overlaps) if step_overlaps else 0.0,
        "max_step_common_logprob_delta": (
            max(step_common_deltas) if step_common_deltas else math.inf
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--jax-dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--min-logit-cosine", type=float, default=0.999)
    parser.add_argument("--max-logit-rmse", type=float, default=0.5)
    parser.add_argument("--min-topk-overlap", type=float, default=0.80)
    parser.add_argument("--max-topk-logprob-delta", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.golden_dir / "manifest.json").read_text(encoding="utf-8"))
    results = {
        "schema_version": 1,
        "golden_manifest": manifest,
        "thresholds": {
            "min_logit_cosine": args.min_logit_cosine,
            "max_logit_rmse": args.max_logit_rmse,
            "min_topk_overlap": args.min_topk_overlap,
            "max_topk_logprob_delta": args.max_topk_logprob_delta,
            "greedy_token_ids_exact": True,
        },
        "cases": [],
    }

    top_k = int(manifest["top_k"])
    for artifact in manifest["artifacts"]:
        name = artifact["name"]
        with np.load(args.golden_dir / artifact["artifact"]) as archive:
            golden = {key: archive[key] for key in archive.files}

        existing = _read_dump_manifest(args.jax_dump_dir)
        minimum_index = max((record["index"] for record in existing), default=-1)
        payload = {
            "input_ids": golden["input_ids"].tolist(),
            "sampling_params": {
                "temperature": 0.0,
                "top_k": 1,
                "max_new_tokens": int(golden["greedy_token_ids"].shape[0]),
            },
            "return_logprob": True,
            "top_logprobs_num": top_k,
        }
        response = _post_json(
            f"{args.base_url.rstrip('/')}/generate", payload, args.request_timeout
        )
        jax_logits = _wait_for_prefill_logits(args.jax_dump_dir, minimum_index)
        raw_metrics = _raw_logit_metrics(golden["first_token_logits"], jax_logits, top_k)
        step_metrics = _server_step_metrics(golden, response, top_k)
        passed = (
            raw_metrics["cosine_similarity"] >= args.min_logit_cosine
            and raw_metrics["rmse"] <= args.max_logit_rmse
            and raw_metrics["topk_overlap"] >= args.min_topk_overlap
            and raw_metrics["topk_common_max_logprob_delta"] <= args.max_topk_logprob_delta
            and step_metrics["min_step_topk_overlap"] >= args.min_topk_overlap
            and step_metrics["max_step_common_logprob_delta"] <= args.max_topk_logprob_delta
            and step_metrics["greedy_token_ids_exact"]
        )
        case_result = {
            "name": name,
            "passed": passed,
            "raw_first_token": raw_metrics,
            "generation": step_metrics,
        }
        results["cases"].append(case_result)
        print(json.dumps(case_result, sort_keys=True), flush=True)

    results["passed"] = all(case["passed"] for case in results["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; passed={results['passed']}", flush=True)
    if not results["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
