"""Search TPU7 tiles for GLM-5.2 per-channel W8A16 local M=8 matmuls.

This orchestrates the reviewed ``bench_glm52_per_channel`` harness.  Every
candidate runs in a fresh process, must pass the same-boundary numerical
oracle, and is ranked by XProf device duration rather than host wall time.
The two search leaders per shape are then repeated three times before a
production registry entry is proposed.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Shape:
    operation: str
    tp_degree: int
    n: int
    k: int
    bn_candidates: tuple[int, ...]
    include_half_k: bool = False


SHAPES = (
    Shape("q_a_proj", 1, 2048, 6144, (512, 1024, 2048)),
    Shape("q_b_proj", 1, 16384, 2048, (1024, 2048, 4096, 8192)),
    Shape("kv_a_proj_with_mqa", 1, 576, 6144, (128, 256, 576)),
    Shape("o_proj", 1, 6144, 16384, (512, 1024, 1536, 2048)),
    Shape("merged_gate_up_proj", 2, 12288, 6144, (1024, 2048, 3072, 4096), True),
    Shape("down_proj", 1, 6144, 12288, (512, 1024, 1536, 2048, 3072)),
    Shape("indexer_wq_b", 1, 4096, 2048, (1024, 2048, 4096)),
    Shape("indexer_wk", 1, 128, 6144, (128,)),
)


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _candidate_tiles(shape: Shape) -> list[tuple[int, int, int]]:
    bk_values = (shape.k, shape.k // 2) if shape.include_half_k else (shape.k,)
    return sorted(
        {
            (8, bn, bk)
            for bn in shape.bn_candidates
            for bk in bk_values
            if bn <= shape.n and bk > 0 and shape.k % bk == 0
        }
    )


def _run_one(
    *,
    shape: Shape,
    tile: tuple[int, int, int],
    phase: str,
    run_id: str,
    output: Path,
    trace_root: Path,
    warmup: int,
    samples: int,
    failures: Path,
) -> bool:
    bm, bn, bk = tile
    cmd = [
        sys.executable,
        "-m",
        "benchmark.kernels.quantized_matmul.bench_glm52_per_channel",
        "--suite",
        "anchor",
        "--operations",
        shape.operation,
        "--tp-degree",
        str(shape.tp_degree),
        "--m",
        "8",
        "--modes",
        "w8a16",
        "--implementations",
        "pallas_aligned",
        "--tuned-value",
        f"{bm},{bn},{bk}",
        "--weight-ring-count",
        "1",
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--wall-samples",
        "2",
        "--process-run-id",
        run_id,
        "--trace-root",
        str(trace_root / phase / shape.operation / run_id),
        "--output-jsonl",
        str(output),
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    print(completed.stdout, end="", flush=True)
    if completed.returncode == 0:
        return True
    _append_jsonl(
        failures,
        {
            "phase": phase,
            "operation": shape.operation,
            "m": 8,
            "n": shape.n,
            "k": shape.k,
            "tile": tile,
            "run_id": run_id,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-8000:],
            "stdout_tail": completed.stdout[-4000:],
        },
    )
    print(completed.stderr[-4000:], file=sys.stderr, flush=True)
    return False


def _tile(row: dict) -> tuple[int, int, int]:
    kernel = row["kernel"]
    return (
        int(kernel["BM"]),
        int(kernel["BN"]),
        int(kernel["BK"]),
    )


def _device_p50(row: dict) -> float:
    return float(row["timing"]["device"]["p50_us"])


def _median(values: list[float]) -> float:
    if not values:
        return math.inf
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search-warmup", type=int, default=3)
    parser.add_argument("--search-samples", type=int, default=5)
    parser.add_argument("--formal-warmup", type=int, default=10)
    parser.add_argument("--formal-samples", type=int, default=20)
    parser.add_argument("--formal-runs", type=int, default=3)
    parser.add_argument("--leaders", type=int, default=2)
    args = parser.parse_args()
    if min(
        args.search_samples,
        args.formal_samples,
        args.formal_runs,
        args.leaders,
    ) <= 0:
        parser.error("sample, run, and leader counts must be positive")

    root = Path(args.output_dir)
    benchmark = root / "benchmark"
    traces = root / "profiling" / "xprof"
    search_path = benchmark / "search.jsonl"
    formal_path = benchmark / "formal.jsonl"
    failure_path = benchmark / "failures.jsonl"
    metrics_path = benchmark / "metrics.jsonl"
    for path in (search_path, formal_path, failure_path, metrics_path):
        if path.exists():
            path.unlink()

    for shape in SHAPES:
        for index, tile in enumerate(_candidate_tiles(shape), start=1):
            _run_one(
                shape=shape,
                tile=tile,
                phase="search",
                run_id=f"search-{index:03d}-bm{tile[0]}-bn{tile[1]}-bk{tile[2]}",
                output=search_path,
                trace_root=traces,
                warmup=args.search_warmup,
                samples=args.search_samples,
                failures=failure_path,
            )

    search_rows = _read_jsonl(search_path)
    leaders: dict[str, list[tuple[int, int, int]]] = {}
    for shape in SHAPES:
        valid = [
            row
            for row in search_rows
            if row.get("status") == "ok"
            and row["case"]["aliases"][0]["operation"] == shape.operation
            and row["correctness"]["passed"]
        ]
        valid.sort(key=_device_p50)
        if not valid:
            raise RuntimeError(f"no valid M=8 candidate for {shape.operation}")
        leaders[shape.operation] = [_tile(row) for row in valid[: args.leaders]]

    for shape in SHAPES:
        for tile in leaders[shape.operation]:
            for repeat in range(1, args.formal_runs + 1):
                _run_one(
                    shape=shape,
                    tile=tile,
                    phase="formal",
                    run_id=(
                        f"formal-r{repeat}-bm{tile[0]}-bn{tile[1]}-bk{tile[2]}"
                    ),
                    output=formal_path,
                    trace_root=traces,
                    warmup=args.formal_warmup,
                    samples=args.formal_samples,
                    failures=failure_path,
                )

    formal_rows = _read_jsonl(formal_path)
    summary = {
        "schema_version": 1,
        "operator": "glm52_per_channel_w8a16_m8",
        "selection_metric": "median_of_independent_run_device_p50_us",
        "max_accepted_run_p50_cv": 0.02,
        "shapes": {},
    }
    for shape in SHAPES:
        candidates = []
        for tile in leaders[shape.operation]:
            rows = [
                row
                for row in formal_rows
                if row.get("status") == "ok"
                and row["case"]["aliases"][0]["operation"] == shape.operation
                and _tile(row) == tile
                and row["correctness"]["passed"]
            ]
            p50s = [_device_p50(row) for row in rows]
            mean = statistics.mean(p50s) if p50s else math.inf
            cv = statistics.pstdev(p50s) / mean if len(p50s) > 1 and mean else math.inf
            candidates.append(
                {
                    "tile": list(tile),
                    "independent_runs": len(rows),
                    "run_p50_us": p50s,
                    "median_run_p50_us": _median(p50s),
                    "run_p50_cv": cv,
                    "max_relative_l2": max(
                        (float(row["correctness"]["relative_l2"]) for row in rows),
                        default=None,
                    ),
                    "credible": len(rows) == args.formal_runs and cv <= 0.02,
                }
            )
        candidates.sort(key=lambda item: item["median_run_p50_us"])
        credible = [item for item in candidates if item["credible"]]
        if not credible:
            raise RuntimeError(f"no statistically credible winner for {shape.operation}")
        winner = credible[0]
        summary["shapes"][shape.operation] = {
            "m": 8,
            "n": shape.n,
            "k": shape.k,
            "winner": winner,
            "formal_candidates": candidates,
        }
        _append_jsonl(
            metrics_path,
            {
                "variant": shape.operation,
                "m": 8,
                "n": shape.n,
                "k": shape.k,
                "latency_ms": winner["median_run_p50_us"] / 1000.0,
                "run_p50_cv": winner["run_p50_cv"],
                "bm": winner["tile"][0],
                "bn": winner["tile"][1],
                "bk": winner["tile"][2],
            },
        )

    (benchmark / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("GLM52_PER_CHANNEL_M8_TUNE_RESULT", json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
