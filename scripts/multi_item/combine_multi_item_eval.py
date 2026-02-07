#!/usr/bin/env python3
"""Combine multi/serial endpoint eval artifacts and compute Torch parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def max_abs_diff(vec_a: list[float], vec_b: list[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector length mismatch: {len(vec_a)} != {len(vec_b)}")
    return max(abs(a - b) for a, b in zip(vec_a, vec_b))


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def compute_torch_scores(
    *,
    model_name: str,
    query_prefix_ids: list[int],
    item_ids: list[list[int]],
    label_token_ids: list[int],
) -> list[list[float]]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    label_ids = torch.tensor(label_token_ids, dtype=torch.long)
    out: list[list[float]] = []
    with torch.no_grad():
        for item in item_ids:
            input_ids = torch.tensor([query_prefix_ids + item], dtype=torch.long)
            logits = model(input_ids=input_ids).logits[0, -1, :]
            label_logits = logits[label_ids]
            probs = torch.softmax(label_logits, dim=-1).tolist()
            out.append([float(x) for x in probs])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multi-json", required=True)
    parser.add_argument("--serial-json", required=True)
    parser.add_argument("--output-multi-vs-serial-json", required=True)
    parser.add_argument("--output-jax-torch-parity-json", required=True)
    parser.add_argument(
        "--torch-model",
        default=None,
        help="Override model name for torch reference. Default uses model from multi json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    multi = load_json(args.multi_json)
    serial = load_json(args.serial_json)

    multi_scores = multi["isolation"]["base_scores"]
    serial_scores = serial["isolation"]["base_scores"]
    if len(multi_scores) != len(serial_scores):
        raise ValueError(
            f"Score count mismatch: multi={len(multi_scores)} serial={len(serial_scores)}"
        )

    equiv_abs_diffs = [max_abs_diff(m, s) for m, s in zip(multi_scores, serial_scores)]

    perf: dict[str, dict] = {}
    for key in sorted(set(multi["performance"].keys()) & set(serial["performance"].keys()), key=int):
        m = multi["performance"][key]
        s = serial["performance"][key]
        speedup = (s["p50_sec"] / m["p50_sec"]) if m["p50_sec"] > 0 else 0.0
        perf[key] = {
            "multi_p50_sec": m["p50_sec"],
            "multi_p95_sec": m["p95_sec"],
            "serial_p50_sec": s["p50_sec"],
            "serial_p95_sec": s["p95_sec"],
            "speedup_vs_serial_p50": speedup,
            "multi_items_per_sec_p50": m["items_per_sec_p50"],
            "serial_items_per_sec_p50": s["items_per_sec_p50"],
            "multi_latencies_sec": m["latencies_sec"],
            "serial_latencies_sec": s["latencies_sec"],
        }

    multi_vs_serial = {
        "model": multi["model"],
        "multi_url": multi["url"],
        "serial_url": serial["url"],
        "labels": multi["labels"],
        "equivalence": {
            "abs_diffs_multi_vs_serial": equiv_abs_diffs,
            "max_abs_diff": max(equiv_abs_diffs) if equiv_abs_diffs else 0.0,
            "mean_abs_diff": (
                sum(equiv_abs_diffs) / len(equiv_abs_diffs) if equiv_abs_diffs else 0.0
            ),
            "multi_scores": multi_scores,
            "serial_scores": serial_scores,
        },
        "isolation": {
            "same_length_mutation_unchanged_diffs": multi["isolation"]["same_length_mutation"][
                "unchanged_item_abs_diffs"
            ],
            "same_length_mutation_max_abs_diff": multi["isolation"]["same_length_mutation"][
                "max_abs_diff_unchanged"
            ],
            "changed_length_mutation_unchanged_diffs": multi["isolation"][
                "changed_length_mutation"
            ]["unchanged_item_abs_diffs"],
            "changed_length_mutation_max_abs_diff": multi["isolation"][
                "changed_length_mutation"
            ]["max_abs_diff_unchanged"],
        },
        "performance": perf,
    }
    serial_plus_delim_scores = (
        serial.get("semantic_alignment", {}).get("base_scores_query_plus_delimiter")
    )
    if serial_plus_delim_scores is not None and len(serial_plus_delim_scores) == len(multi_scores):
        aligned_diffs = [
            max_abs_diff(m, s) for m, s in zip(multi_scores, serial_plus_delim_scores, strict=True)
        ]
        multi_vs_serial["equivalence_query_plus_delimiter"] = {
            "abs_diffs_multi_vs_serial_query_plus_delimiter": aligned_diffs,
            "max_abs_diff": max(aligned_diffs) if aligned_diffs else 0.0,
            "mean_abs_diff": (
                sum(aligned_diffs) / len(aligned_diffs) if aligned_diffs else 0.0
            ),
            "serial_query_plus_delimiter_scores": serial_plus_delim_scores,
        }
    save_json(args.output_multi_vs_serial_json, multi_vs_serial)

    torch_model = args.torch_model or multi["model"]
    query_ids = multi["isolation"]["query_ids"]
    item_ids = multi["isolation"]["item_ids"]
    labels = multi["labels"]
    delimiter = int(multi["delimiter"])
    torch_serial_scores = compute_torch_scores(
        model_name=torch_model,
        query_prefix_ids=query_ids,
        item_ids=item_ids,
        label_token_ids=labels,
    )
    torch_multi_semantic_scores = compute_torch_scores(
        model_name=torch_model,
        query_prefix_ids=query_ids + [delimiter],
        item_ids=item_ids,
        label_token_ids=labels,
    )
    abs_multi_vs_torch_multi_semantic = [
        max_abs_diff(a, b) for a, b in zip(multi_scores, torch_multi_semantic_scores, strict=True)
    ]
    abs_multi_vs_torch_serial = [
        max_abs_diff(a, b) for a, b in zip(multi_scores, torch_serial_scores, strict=True)
    ]
    abs_serial_vs_torch_serial = [
        max_abs_diff(a, b) for a, b in zip(serial_scores, torch_serial_scores, strict=True)
    ]
    parity = {
        "model": multi["model"],
        "delimiter": delimiter,
        "labels": labels,
        "query_ids": query_ids,
        "item_ids": item_ids,
        "jax_multi_scores": multi_scores,
        "jax_serial_scores": serial_scores,
        "torch_serial_reference_scores": torch_serial_scores,
        "torch_multi_semantic_reference_scores": torch_multi_semantic_scores,
        "abs_diff_jax_multi_vs_torch_serial": abs_multi_vs_torch_serial,
        "abs_diff_jax_multi_vs_torch_multi_semantic": abs_multi_vs_torch_multi_semantic,
        "abs_diff_jax_serial_vs_torch_serial": abs_serial_vs_torch_serial,
        "max_abs_diff_jax_multi_vs_torch_serial": (
            max(abs_multi_vs_torch_serial) if abs_multi_vs_torch_serial else 0.0
        ),
        "max_abs_diff_jax_multi_vs_torch_multi_semantic": (
            max(abs_multi_vs_torch_multi_semantic) if abs_multi_vs_torch_multi_semantic else 0.0
        ),
        "max_abs_diff_jax_serial_vs_torch_serial": (
            max(abs_serial_vs_torch_serial) if abs_serial_vs_torch_serial else 0.0
        ),
    }
    save_json(args.output_jax_torch_parity_json, parity)

    print(
        json.dumps(
            {
                "output_multi_vs_serial_json": args.output_multi_vs_serial_json,
                "output_jax_torch_parity_json": args.output_jax_torch_parity_json,
            }
        )
    )


if __name__ == "__main__":
    main()
