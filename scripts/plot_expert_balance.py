#!/usr/bin/env python3
"""
Plot per-layer expert balance ratios over inference segments.

Input CSV is produced by expert balance debug (expert_balance_*.csv).
Outputs one PNG per layer with multiple lines: max/mean, min/mean, hot_topk, cold_topk, std/mean.
Also shades segments that look like Hotspot or Sparse Hotspot (heuristic).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Iterable


def _parse_layers(spec: str, all_layers: Iterable[int]) -> list[int]:
    spec = spec.strip()
    if spec.lower() in ("all", "*", ""):
        return sorted(set(all_layers))
    layers: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        m = re.fullmatch(r"(\\d+)-(\\d+)", part)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            if end < start:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def _read_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-layer expert balance metrics over segments."
    )
    parser.add_argument("csv_path", help="Path to expert_balance_*.csv")
    parser.add_argument(
        "--out-dir",
        default="expert_balance_plots",
        help="Output directory for PNGs.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection: 'all', '0,1,2', or '0-47'.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["segment_idx", "tokens"],
        default="segment_idx",
        help="X axis uses segment index or cumulative tokens.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip segments with has_data=0.",
    )
    parser.add_argument(
        "--hotspot-hot-multiple-threshold",
        type=float,
        default=3.0,
        help="hot_topk/mean >= threshold => Hotspot.",
    )
    parser.add_argument(
        "--sparse-cold-multiple-threshold",
        type=float,
        default=0.1,
        help="cold_topk/mean <= threshold => Sparse Hotspot (with active ratio check).",
    )
    parser.add_argument(
        "--sparse-active-ratio-threshold",
        type=float,
        default=0.9,
        help="active_experts/num_experts <= threshold => Sparse Hotspot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG DPI.",
    )
    parser.add_argument(
        "--show-mean-line",
        action="store_true",
        default=True,
        help="Show mean baseline (y=1) on ratio plots.",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime import guard
        raise SystemExit(
            "matplotlib is required. Install it or run in an env that has it."
        ) from exc

    rows = _read_rows(args.csv_path)
    if not rows:
        raise SystemExit("No rows found in CSV.")

    all_layers = {int(r["layer"]) for r in rows}
    layers = _parse_layers(args.layers, all_layers)

    os.makedirs(args.out_dir, exist_ok=True)

    # Group rows by layer
    by_layer: dict[int, list[dict]] = {layer: [] for layer in layers}
    for r in rows:
        layer = int(r["layer"])
        if layer not in by_layer:
            continue
        if args.skip_empty and r.get("has_data", "1") in ("0", 0, "false", "False"):
            continue
        by_layer[layer].append(r)

    device_cols = {
        "device_max_over_mean",
        "device_min_over_mean",
        "device_cv",
    }
    has_device_metrics = all(col in rows[0] for col in device_cols)
    has_device_hotcold = (
        "device_hot_topk_mean_multiple" in rows[0] and "device_cold_topk_mean_multiple" in rows[0]
    )
    has_device_mean = "device_mean_count" in rows[0]

    for layer, layer_rows in by_layer.items():
        if not layer_rows:
            continue
        layer_rows.sort(key=lambda x: int(x["segment_idx"]))

        seg_tokens = int(layer_rows[0]["segment_tokens"])
        seg_by = layer_rows[0].get("segment_by", "tokens")
        seg_idx = [int(r["segment_idx"]) for r in layer_rows]
        if args.x_axis == "tokens" and seg_by != "tokens":
            x = seg_idx
            x_label = "segment_idx"
        else:
            x = [(idx * seg_tokens) if args.x_axis == "tokens" else idx for idx in seg_idx]
            x_label = "tokens" if args.x_axis == "tokens" else "segment_idx"
        mean_count = [float(r["mean_count"]) for r in layer_rows]
        min_count = [float(r["min_count"]) for r in layer_rows]
        max_count = [float(r["max_count"]) for r in layer_rows]
        hot_mult = [float(r["hot_topk_mean_multiple"]) for r in layer_rows]
        cold_mult = [float(r["cold_topk_mean_multiple"]) for r in layer_rows]
        active_experts = [int(r["active_experts"]) for r in layer_rows]
        num_experts = int(layer_rows[0]["num_experts"])

        layer_has_device = has_device_metrics and any(
            r.get("device_has_data", "0") not in ("0", 0, "false", "False") for r in layer_rows
        )
        if layer_has_device:
            fig = plt.figure(figsize=(13, 7))
            gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[4, 1], height_ratios=[1, 1])
            ax_exp = fig.add_subplot(gs[0, 0])
            ax_dev = fig.add_subplot(gs[1, 0], sharex=ax_exp)
            ax_note = fig.add_subplot(gs[:, 1])
        else:
            fig = plt.figure(figsize=(13, 4))
            gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1])
            ax_exp = fig.add_subplot(gs[0, 0])
            ax_dev = None
            ax_note = fig.add_subplot(gs[0, 1])
        std_count = [float(r["std_count"]) for r in layer_rows]
        hot_counts = [hm * mu for hm, mu in zip(hot_mult, mean_count)]
        cold_counts = [cm * mu for cm, mu in zip(cold_mult, mean_count)]
        ax_exp.plot(x, max_count, label="max_count")
        ax_exp.plot(x, min_count, label="min_count")
        ax_exp.plot(x, hot_counts, label="hot_topk_mean_count")
        ax_exp.plot(x, cold_counts, label="cold_topk_mean_count")
        ax_exp.plot(x, mean_count, label="mean_count")
        ax_exp.plot(x, std_count, label="std_count")

        active_ratio = [ae / num_experts for ae in active_experts]
        sparse_flags = [
            (ar <= args.sparse_active_ratio_threshold)
            and (cm <= args.sparse_cold_multiple_threshold)
            for ar, cm in zip(active_ratio, cold_mult)
        ]
        hotspot_flags = [(hm >= args.hotspot_hot_multiple_threshold) for hm in hot_mult]

        def _shade(flags, color, label):
            if not any(flags):
                return
            start = None
            for i, flag in enumerate(flags + [False]):
                if flag and start is None:
                    start = i
                if start is not None and not flag:
                    if args.x_axis == "tokens":
                        x0 = seg_idx[start] * seg_tokens
                        x1 = (seg_idx[i - 1] + 1) * seg_tokens
                    else:
                        x0 = seg_idx[start] - 0.5
                        x1 = seg_idx[i - 1] + 0.5
                    ax_exp.axvspan(x0, x1, color=color, alpha=0.12, label=label)
                    start = None

        enable_shading = False
        if enable_shading:
            _shade(hotspot_flags, "#f6a600", "Hotspot")
            _shade(sparse_flags, "#d9534f", "Sparse Hotspot")

        ax_exp.set_title(f"Layer {layer} - expert level")
        ax_exp.set_ylabel("count")
        ax_exp.grid(True, alpha=0.3)
        handles, labels = ax_exp.get_legend_handles_labels()
        # de-duplicate legend labels
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)
        ax_exp.legend(uniq_handles, uniq_labels, loc="upper right")

        if ax_dev is not None:
            dev_mean = [float(r["device_mean_count"]) for r in layer_rows]
            dev_std = [float(r["device_std_count"]) for r in layer_rows]
            dev_max = [float(r["device_max_count"]) for r in layer_rows]
            dev_min = [float(r["device_min_count"]) for r in layer_rows]
            ax_dev.plot(x, dev_max, label="device max_count")
            ax_dev.plot(x, dev_min, label="device min_count")
            ax_dev.plot(x, dev_mean, label="device mean_count")
            ax_dev.plot(x, dev_std, label="device std_count")
            if has_device_hotcold:
                dev_hot_mult = [float(r["device_hot_topk_mean_multiple"]) for r in layer_rows]
                dev_cold_mult = [float(r["device_cold_topk_mean_multiple"]) for r in layer_rows]
                dev_hot_cnt = [m * mu for m, mu in zip(dev_hot_mult, dev_mean)]
                dev_cold_cnt = [m * mu for m, mu in zip(dev_cold_mult, dev_mean)]
                ax_dev.plot(x, dev_hot_cnt, label="device hot_topk_mean_count")
                ax_dev.plot(x, dev_cold_cnt, label="device cold_topk_mean_count")
            ax_dev.set_title(f"Layer {layer} - device level")
            ax_dev.set_ylabel("count")
            ax_dev.grid(True, alpha=0.3)
            ax_dev.legend(loc="upper right")

        ax_exp.set_xlabel(x_label)
        ax_note.axis("off")
        note = (
            "Counts:\n"
            "max/min/mean/std counts\n"
            "hot_topk_mean_count = avg top‑k counts\n"
            "cold_topk_mean_count = avg coldest top‑k\n"
            "Device subplot (if present):\n"
            "device max/min/mean/std counts"
        )
        if has_device_hotcold:
            note += "\n+ device hot/cold top‑k mean counts"
        ax_note.text(
            0.0,
            1.0,
            note,
            ha="left",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()

        out_path = os.path.join(args.out_dir, f"layer_{layer:03d}.png")
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
