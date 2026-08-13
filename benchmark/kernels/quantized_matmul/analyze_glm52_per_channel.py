"""Aggregate independent GLM-5.2 per-channel benchmark JSONL runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.kernels.quantized_matmul.bench_glm52_per_channel import (
    TPU_INFERENCE_REFERENCE_SHA,
    summarize_samples,
)


def _read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                row["_input_path"] = str(path)
                row["_input_line"] = line_number
                rows.append(row)
    return rows


def _measurement_key(row: dict[str, Any]) -> tuple[Any, ...]:
    case = row["case"]
    return (
        case["m"],
        case["n"],
        case["k"],
        case["mode"],
        case["implementation"],
        case.get("variant", "baseline"),
        case["weight_ring_count"],
    )


def _comparison_key(measurement: dict[str, Any]) -> tuple[Any, ...]:
    case = measurement["case"]
    return (
        case["m"],
        case["n"],
        case["k"],
        case["mode"],
        case.get("variant", "baseline"),
        case["weight_ring_count"],
    )


def aggregate_rows(
    rows: Sequence[dict[str, Any]],
    expected_runs: int,
    cv_limit: float,
    expected_samples_per_run: int | None = None,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    skipped = []
    issues = []
    source_commits = set()
    reference_shas = set()
    dirty_sources = []
    for row in rows:
        source = row.get("source", {})
        source_commits.add(source.get("sglang_jax_commit"))
        reference_shas.add(source.get("tpu_inference_reference_sha"))
        if source.get("sglang_jax_dirty"):
            dirty_sources.append(row)
        if row.get("status") != "ok":
            skipped.append(row)
            continue
        groups[_measurement_key(row)].append(row)

    measurements = []
    for key, group in sorted(groups.items()):
        run_ids = {row["case"]["process_run_id"] for row in group}
        if len(run_ids) != expected_runs:
            issues.append(
                {
                    "kind": "independent_run_count",
                    "key": list(key),
                    "expected": expected_runs,
                    "actual": len(run_ids),
                }
            )
        rows_by_run: dict[str, int] = defaultdict(int)
        for row in group:
            rows_by_run[row["case"]["process_run_id"]] += 1
            sample_count = len(row["timing"]["device"]["raw_samples_us"])
            if expected_samples_per_run is not None and sample_count != expected_samples_per_run:
                issues.append(
                    {
                        "kind": "sample_count",
                        "key": list(key),
                        "run_id": row["case"]["process_run_id"],
                        "expected": expected_samples_per_run,
                        "actual": sample_count,
                    }
                )
        duplicates = {run_id: count for run_id, count in rows_by_run.items() if count != 1}
        if duplicates:
            issues.append(
                {
                    "kind": "duplicate_rows_per_run",
                    "key": list(key),
                    "counts": duplicates,
                }
            )
        samples = [sample for row in group for sample in row["timing"]["device"]["raw_samples_us"]]
        run_cvs = [row["timing"]["device"]["cv"] for row in group]
        high_cv = [value for value in run_cvs if value is not None and value > cv_limit]
        if high_cv:
            issues.append(
                {
                    "kind": "high_cv",
                    "key": list(key),
                    "limit": cv_limit,
                    "values": high_cv,
                }
            )
        correctness_passed = all(row["correctness"]["passed"] for row in group)
        if not correctness_passed:
            issues.append({"kind": "correctness", "key": list(key)})
        case = dict(group[0]["case"])
        measurement = {
            "case": case,
            "run_ids": sorted(run_ids),
            "run_count": len(run_ids),
            "sample_count": len(samples),
            "timing": summarize_samples(samples),
            "run_cvs": run_cvs,
            "correctness_passed": correctness_passed,
            "kernel": group[0].get("kernel"),
        }
        measurements.append(measurement)

    comparisons_by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for measurement in measurements:
        comparisons_by_key[_comparison_key(measurement)][
            measurement["case"]["implementation"]
        ] = measurement

    implementation_comparisons = []
    for key, implementations in sorted(comparisons_by_key.items()):
        if not {"xla", "pallas_aligned"}.issubset(implementations):
            continue
        xla = implementations["xla"]["timing"]
        pallas = implementations["pallas_aligned"]["timing"]
        delta_us = xla["p50_us"] - pallas["p50_us"]
        improvement = delta_us / xla["p50_us"] if xla["p50_us"] else 0.0
        p95_ratio = pallas["p95_us"] / xla["p95_us"] if xla["p95_us"] else None
        implementation_comparisons.append(
            {
                "key": {
                    "m": key[0],
                    "n": key[1],
                    "k": key[2],
                    "mode": key[3],
                    "variant": key[4],
                    "weight_ring_count": key[5],
                },
                "xla_p50_us": xla["p50_us"],
                "pallas_p50_us": pallas["p50_us"],
                "pallas_speedup": xla["p50_us"] / pallas["p50_us"],
                "absolute_improvement_us": delta_us,
                "relative_improvement": improvement,
                "p95_ratio": p95_ratio,
                "performance_gate_passed": (
                    improvement >= 0.05
                    and delta_us >= 0.5
                    and p95_ratio is not None
                    and p95_ratio <= 1.02
                ),
            }
        )

    mode_groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for measurement in measurements:
        case = measurement["case"]
        key = (
            case["m"],
            case["n"],
            case["k"],
            case["implementation"],
            case.get("variant", "baseline"),
            case["weight_ring_count"],
        )
        mode_groups[key][case["mode"]] = measurement
    mode_comparisons = []
    for key, modes in sorted(mode_groups.items()):
        if not {"w8a8", "w8a16"}.issubset(modes):
            continue
        w8a8 = modes["w8a8"]["timing"]["p50_us"]
        w8a16 = modes["w8a16"]["timing"]["p50_us"]
        mode_comparisons.append(
            {
                "key": {
                    "m": key[0],
                    "n": key[1],
                    "k": key[2],
                    "implementation": key[3],
                    "variant": key[4],
                    "weight_ring_count": key[5],
                },
                "w8a8_p50_us": w8a8,
                "w8a16_p50_us": w8a16,
                "w8a8_over_w8a16": w8a8 / w8a16,
                "interpretation": "control_ratio_not_pure_activation_quantization_cost",
            }
        )

    ring_groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for measurement in measurements:
        case = measurement["case"]
        key = (
            case["m"],
            case["n"],
            case["k"],
            case["mode"],
            case["implementation"],
            case.get("variant", "baseline"),
        )
        ring_groups[key][case["weight_ring_count"]] = measurement
    ring_calibration = []
    for key, rings in sorted(ring_groups.items()):
        if not {4, 16}.issubset(rings):
            continue
        ring4 = rings[4]["timing"]["p50_us"]
        ring16 = rings[16]["timing"]["p50_us"]
        relative_delta = abs(ring4 - ring16) / ring16 if ring16 else None
        ring_calibration.append(
            {
                "key": list(key),
                "ring4_p50_us": ring4,
                "ring16_p50_us": ring16,
                "relative_delta": relative_delta,
                "cache_plateau_reached": relative_delta is not None and relative_delta <= 0.02,
            }
        )

    clean_source_commits = {value for value in source_commits if value}
    clean_reference_shas = {value for value in reference_shas if value}
    if len(clean_source_commits) != 1:
        issues.append(
            {
                "kind": "source_commit_mismatch",
                "values": sorted(clean_source_commits),
            }
        )
    if clean_reference_shas != {TPU_INFERENCE_REFERENCE_SHA}:
        issues.append(
            {
                "kind": "reference_sha_mismatch",
                "expected": TPU_INFERENCE_REFERENCE_SHA,
                "values": sorted(clean_reference_shas),
            }
        )
    if dirty_sources:
        issues.append({"kind": "dirty_source", "row_count": len(dirty_sources)})

    return {
        "schema_version": 1,
        "source_commits": sorted(clean_source_commits),
        "reference_shas": sorted(clean_reference_shas),
        "input_rows": len(rows),
        "skipped_rows": len(skipped),
        "measurement_groups": len(measurements),
        "expected_independent_runs": expected_runs,
        "expected_samples_per_run": expected_samples_per_run,
        "cv_limit": cv_limit,
        "issues": issues,
        "measurements": measurements,
        "implementation_comparisons": implementation_comparisons,
        "mode_comparisons": mode_comparisons,
        "ring_calibration": ring_calibration,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--expected-runs", type=int, default=3)
    parser.add_argument("--expected-samples-per-run", type=int, default=30)
    parser.add_argument("--cv-limit", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_runs <= 0 or args.expected_samples_per_run <= 0 or args.cv_limit < 0:
        raise SystemExit(
            "expected-runs/expected-samples-per-run must be positive and cv-limit non-negative"
        )
    report = aggregate_rows(
        _read_jsonl(args.inputs),
        args.expected_runs,
        args.cv_limit,
        args.expected_samples_per_run,
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    print(
        f"groups={report['measurement_groups']} comparisons="
        f"{len(report['implementation_comparisons'])} issues={len(report['issues'])}",
    )
    return 2 if args.strict and report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
