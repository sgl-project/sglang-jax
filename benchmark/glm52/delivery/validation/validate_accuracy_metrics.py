"""Validate a completed sgl-eval metrics artifact against an accuracy gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate_metrics(root: Path, min_score: float, expected_examples: int) -> dict:
    metrics_paths = list(root.rglob("metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"no metrics.json found under {root}")
    metrics_path = max(metrics_paths, key=lambda path: path.stat().st_mtime_ns)
    metrics = json.loads(metrics_path.read_text())
    score = float((metrics.get("aggregate") or {}).get("score", float("nan")))
    num_examples = int(metrics.get("num_examples") or 0)
    partial = bool(metrics.get("partial", False))
    passed = (
        math.isfinite(score)
        and score >= min_score
        and num_examples == expected_examples
        and not partial
    )
    return {
        "dataset": metrics.get("name"),
        "expected_examples": expected_examples,
        "metrics_path": str(metrics_path),
        "min_score": min_score,
        "num_examples": num_examples,
        "partial": partial,
        "passed": passed,
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-score", type=float, required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    args = parser.parse_args()
    report = validate_metrics(args.root, args.min_score, args.expected_examples)
    gate_path = args.root / "accuracy-gate.json"
    gate_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("GLM52_DELIVERY_ACCURACY_GATE", json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("GLM-5.2 accuracy gate failed")


if __name__ == "__main__":
    main()
