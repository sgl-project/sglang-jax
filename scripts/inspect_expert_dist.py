#!/usr/bin/env python3
"""
Inspect expert distribution .npy files.

Supports the formats used by init_expert_location and expert distribution recorder:
1) np.save(dict) -> object scalar array whose item() is a dict
2) np.save(array) -> plain ndarray
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


def _load_npy_as_dict(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.ndim == 0:
        data = loaded.item()
        if isinstance(data, dict):
            return data
        return {"value": data}
    return {"physical_to_logical_map": loaded}


def _describe_value(key: str, value: Any) -> str:
    if isinstance(value, np.ndarray):
        desc = f"{key}: ndarray shape={value.shape}, dtype={value.dtype}"
        if value.size > 0 and np.issubdtype(value.dtype, np.number):
            desc += f", min={value.min()}, max={value.max()}, sum={value.sum()}"
        return desc
    return f"{key}: {type(value).__name__} = {value}"


def _pick_count_matrix(data: dict[str, Any]) -> tuple[str | None, np.ndarray | None]:
    candidates = (
        "logical_count",
        "physical_count",
        "experts_count",
        "physical_to_logical_map",
    )
    for key in candidates:
        value = data.get(key)
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and np.issubdtype(value.dtype, np.number)
        ):
            return key, value

    for key, value in data.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim == 2
            and np.issubdtype(value.dtype, np.number)
        ):
            return key, value
    return None, None


def _topk_pairs(row: np.ndarray, k: int) -> list[tuple[int, int]]:
    if row.size == 0 or k <= 0:
        return []
    k = min(k, row.size)
    idx = np.argpartition(row, -k)[-k:]
    idx = idx[np.argsort(row[idx])[::-1]]
    return [(int(i), int(row[i])) for i in idx]


def _export_csv(matrix: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "expert", "count"])
        for layer in range(matrix.shape[0]):
            for expert in range(matrix.shape[1]):
                writer.writerow([layer, expert, int(matrix[layer, expert])])


def main():
    parser = argparse.ArgumentParser(description="Inspect expert distribution .npy content.")
    parser.add_argument(
        "npy_path", help="Path to .npy file, e.g. debug_outputs/expert_dist_xxx.npy"
    )
    parser.add_argument("--topk", type=int, default=8, help="Top-k experts to print per layer.")
    parser.add_argument(
        "--layer",
        type=int,
        action="append",
        default=[],
        help="Specify layer(s) to print; can be repeated.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=16,
        help="When --layer is not set, print at most this many layers.",
    )
    parser.add_argument(
        "--dump-layer",
        action="store_true",
        help="Print the full count vector for selected layers.",
    )
    parser.add_argument(
        "--export-csv", default="", help="Optional output CSV path for full matrix."
    )
    args = parser.parse_args()

    if np is None:
        raise SystemExit("numpy is required. Install it in the runtime environment first.")

    path = Path(args.npy_path)
    if not path.exists():
        alt = Path("debug_outputs") / path.name
        if alt.exists():
            path = alt
        else:
            raise SystemExit(f"File not found: {args.npy_path}")

    data = _load_npy_as_dict(path)
    print(f"file: {path}")
    print("keys:")
    for key in data:
        print(f"  - {_describe_value(key, data[key])}")

    matrix_key, matrix = _pick_count_matrix(data)
    if matrix is None:
        print("No 2D numeric matrix found; only metadata printed.")
        return

    matrix_i64 = matrix.astype(np.int64, copy=False)
    print(
        f"\nselected matrix: {matrix_key}, shape={matrix_i64.shape}, "
        f"total={int(matrix_i64.sum())}, nonzero={int(np.count_nonzero(matrix_i64))}"
    )

    if args.export_csv:
        out_path = Path(args.export_csv)
        _export_csv(matrix_i64, out_path)
        print(f"exported csv: {out_path}")

    if args.layer:
        layers = []
        for layer in args.layer:
            if 0 <= layer < matrix_i64.shape[0]:
                layers.append(layer)
            else:
                print(f"skip invalid layer index: {layer}")
    else:
        n = min(matrix_i64.shape[0], max(args.max_layers, 0))
        layers = list(range(n))
        if matrix_i64.shape[0] > n:
            print(f"(truncated: showing first {n}/{matrix_i64.shape[0]} layers)")

    print("\nlayer summary:")
    for layer in layers:
        row = matrix_i64[layer]
        total = int(row.sum())
        nonzero = int(np.count_nonzero(row))
        top = _topk_pairs(row, args.topk)
        top_str = ", ".join([f"{idx}:{cnt}" for idx, cnt in top]) if top else "(empty)"
        print(
            f"  layer={layer} total={total} nonzero={nonzero}/{row.size} top{args.topk}=[{top_str}]"
        )
        if args.dump_layer:
            print(np.array2string(row, separator=",", max_line_width=200))


if __name__ == "__main__":
    main()
