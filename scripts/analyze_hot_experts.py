#!/usr/bin/env python3
"""
Analyze hottest experts/devices per layer and track their imbalance over segments.

Reads expert_balance_*.csv (experts_count column) and outputs:
- CSV per layer for hot experts: hot_experts_layer_XXX.csv
- CSV per layer for hot devices (if grouping available): hot_devices_layer_XXX.csv
- Optional plots of ratio-to-mean over segment index.
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


def _decode_counts(row: dict, num_experts: int) -> "np.ndarray":
    import numpy as np

    counts_str = row.get("experts_count", "")
    if counts_str:
        parts = [p for p in counts_str.replace(" ", "").split(",") if p != ""]
        arr = np.array([int(p) for p in parts], dtype=np.int64)
    else:
        return np.zeros(num_experts, dtype=np.int64)

    if num_experts and arr.size != num_experts:
        if arr.size > num_experts:
            return arr[:num_experts]
        out = np.zeros(num_experts, dtype=np.int64)
        out[: arr.size] = arr
        return out
    return arr


def _write_hot_csv(path: str, header: list[str], rows: list[list]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hottest experts/devices per layer over segments."
    )
    parser.add_argument("csv_path", help="Path to expert_balance_*.csv")
    parser.add_argument(
        "--out-dir",
        default="expert_balance_analysis",
        help="Output directory for CSVs/plots.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection: 'all', '0,1,2', or '0-47'.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="Number of hottest experts/devices to track.",
    )
    parser.add_argument(
        "--device-group-size",
        type=int,
        default=0,
        help="Group experts by this size to form devices (e.g. 4).",
    )
    parser.add_argument(
        "--out-prefix",
        default="",
        help="Filename prefix for outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--skip-initial",
        type=int,
        default=0,
        help="Ignore first N segments in plots.",
    )
    parser.add_argument(
        "--delta-mode",
        choices=["none", "ratio", "pct", "diff"],
        default="none",
        help="Plot change vs baseline: ratio, pct, or diff.",
    )
    parser.add_argument(
        "--delta-baseline",
        choices=["first", "mean", "median"],
        default="first",
        help="Baseline for delta-mode (computed after skip-initial).",
    )
    args = parser.parse_args()

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise SystemExit("numpy is required.") from exc

    rows = []
    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise SystemExit("No rows found in CSV.")

    all_layers = {int(r["layer"]) for r in rows}
    layers = _parse_layers(args.layers, all_layers)
    out_prefix = args.out_prefix or ""

    # Group rows by layer
    by_layer: dict[int, list[dict]] = {layer: [] for layer in layers}
    for r in rows:
        layer = int(r["layer"])
        if layer not in by_layer:
            continue
        by_layer[layer].append(r)

    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover
            raise SystemExit("matplotlib is required for plots.") from exc
    else:
        plt = None

    os.makedirs(args.out_dir, exist_ok=True)

    for layer, layer_rows in by_layer.items():
        if not layer_rows:
            continue
        layer_rows.sort(key=lambda x: int(x["segment_idx"]))

        seg_idx = [int(r["segment_idx"]) for r in layer_rows]
        segment_counter = int(layer_rows[0].get("segment_counter", 0) or 0)
        num_experts = int(layer_rows[0]["num_experts"])
        ep_size = int(layer_rows[0].get("ep_size", "0") or 0)

        counts_list = [_decode_counts(r, num_experts) for r in layer_rows]
        totals = [int(r.get("total_assignments", 0) or 0) for r in layer_rows]
        if all(t == 0 for t in totals):
            totals = [int(c.sum()) for c in counts_list]

        counts_mat = np.stack(counts_list, axis=0)  # segments × experts
        total_per_expert = counts_mat.sum(axis=0)
        k_exp = max(1, min(args.topk, num_experts))
        hot_exp_idx = np.argsort(total_per_expert)[-k_exp:][::-1]

        mean_per_seg = [
            (totals[i] / num_experts) if num_experts > 0 else 0.0 for i in range(len(totals))
        ]

        # Hot expert CSV
        exp_header = [
            "segment_idx",
            "segment_counter",
            "mean_count",
        ]
        for idx in hot_exp_idx:
            exp_header += [f"expert_{idx}_count", f"expert_{idx}_ratio"]

        exp_rows = []
        for i, counts in enumerate(counts_mat):
            mu = mean_per_seg[i]
            row = [seg_idx[i], segment_counter, mu]
            for idx in hot_exp_idx:
                cnt = int(counts[idx])
                ratio = (cnt / mu) if mu > 0 else 0.0
                row += [cnt, ratio]
            exp_rows.append(row)

        exp_path = os.path.join(args.out_dir, f"{out_prefix}hot_experts_layer_{layer:03d}.csv")
        _write_hot_csv(exp_path, exp_header, exp_rows)

        # Plot hot experts
        if plt is not None:
            fig = plt.figure(figsize=(12, 4))
            ax = fig.add_subplot(1, 1, 1)
            for idx in hot_exp_idx:
                ratios = [
                    (counts_mat[i, idx] / mean_per_seg[i]) if mean_per_seg[i] > 0 else 0.0
                    for i in range(len(seg_idx))
                ]
                plot_x = seg_idx[args.skip_initial :] if args.skip_initial > 0 else seg_idx
                plot_y = ratios[args.skip_initial :] if args.skip_initial > 0 else ratios
                if args.delta_mode != "none" and plot_y:
                    base_vals = plot_y
                    if args.delta_baseline == "mean":
                        baseline = float(np.mean(base_vals))
                    elif args.delta_baseline == "median":
                        baseline = float(np.median(base_vals))
                    else:
                        baseline = float(base_vals[0])
                    if baseline == 0:
                        baseline = 1.0
                    if args.delta_mode == "ratio":
                        plot_y = [v / baseline for v in plot_y]
                    elif args.delta_mode == "pct":
                        plot_y = [(v - baseline) / baseline * 100.0 for v in plot_y]
                    elif args.delta_mode == "diff":
                        plot_y = [v - baseline for v in plot_y]
                ax.plot(plot_x, plot_y, label=f"expert {idx}")
            title_suffix = "ratio to mean"
            y_label = "x mean"
            if args.delta_mode == "ratio":
                title_suffix = "ratio to baseline"
                y_label = "x baseline"
            elif args.delta_mode == "pct":
                title_suffix = "pct change"
                y_label = "% vs baseline"
            elif args.delta_mode == "diff":
                title_suffix = "diff vs baseline"
                y_label = "delta"
            ax.set_title(f"Layer {layer} - hot experts {title_suffix}")
            ax.set_xlabel("segment_idx")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            fig.tight_layout()
            out_path = os.path.join(args.out_dir, f"{out_prefix}hot_experts_layer_{layer:03d}.png")
            fig.savefig(out_path, dpi=160)
            plt.close(fig)

        # Hot devices (optional)
        group_size = args.device_group_size
        if group_size <= 0 and ep_size > 0 and num_experts % ep_size == 0:
            group_size = num_experts // ep_size
        if group_size > 0:
            num_devices = num_experts // group_size
            if num_devices <= 0:
                continue
            usable = num_devices * group_size
            counts_mat_dev = (
                counts_mat[:, :usable]
                .reshape(counts_mat.shape[0], num_devices, group_size)
                .sum(axis=2)
            )
            total_per_device = counts_mat_dev.sum(axis=0)
            k_dev = max(1, min(args.topk, num_devices))
            hot_dev_idx = np.argsort(total_per_device)[-k_dev:][::-1]

            mean_dev_per_seg = [
                (totals[i] / num_devices) if num_devices > 0 else 0.0 for i in range(len(totals))
            ]

            dev_header = [
                "segment_idx",
                "segment_counter",
                "mean_device_count",
            ]
            for idx in hot_dev_idx:
                dev_header += [f"device_{idx}_count", f"device_{idx}_ratio"]

            dev_rows = []
            for i in range(counts_mat_dev.shape[0]):
                mu = mean_dev_per_seg[i]
                row = [seg_idx[i], segment_counter, mu]
                for idx in hot_dev_idx:
                    cnt = int(counts_mat_dev[i, idx])
                    ratio = (cnt / mu) if mu > 0 else 0.0
                    row += [cnt, ratio]
                dev_rows.append(row)

            dev_path = os.path.join(args.out_dir, f"{out_prefix}hot_devices_layer_{layer:03d}.csv")
            _write_hot_csv(dev_path, dev_header, dev_rows)

            if plt is not None:
                fig = plt.figure(figsize=(12, 4))
                ax = fig.add_subplot(1, 1, 1)
                for idx in hot_dev_idx:
                    ratios = [
                        (
                            (counts_mat_dev[i, idx] / mean_dev_per_seg[i])
                            if mean_dev_per_seg[i] > 0
                            else 0.0
                        )
                        for i in range(len(seg_idx))
                    ]
                    plot_x = seg_idx[args.skip_initial :] if args.skip_initial > 0 else seg_idx
                    plot_y = ratios[args.skip_initial :] if args.skip_initial > 0 else ratios
                    if args.delta_mode != "none" and plot_y:
                        base_vals = plot_y
                        if args.delta_baseline == "mean":
                            baseline = float(np.mean(base_vals))
                        elif args.delta_baseline == "median":
                            baseline = float(np.median(base_vals))
                        else:
                            baseline = float(base_vals[0])
                        if baseline == 0:
                            baseline = 1.0
                        if args.delta_mode == "ratio":
                            plot_y = [v / baseline for v in plot_y]
                        elif args.delta_mode == "pct":
                            plot_y = [(v - baseline) / baseline * 100.0 for v in plot_y]
                        elif args.delta_mode == "diff":
                            plot_y = [v - baseline for v in plot_y]
                    ax.plot(plot_x, plot_y, label=f"device {idx}")
                title_suffix = "ratio to mean"
                y_label = "x mean"
                if args.delta_mode == "ratio":
                    title_suffix = "ratio to baseline"
                    y_label = "x baseline"
                elif args.delta_mode == "pct":
                    title_suffix = "pct change"
                    y_label = "% vs baseline"
                elif args.delta_mode == "diff":
                    title_suffix = "diff vs baseline"
                    y_label = "delta"
                ax.set_title(f"Layer {layer} - hot devices {title_suffix}")
                ax.set_xlabel("segment_idx")
                ax.set_ylabel(y_label)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right")
                fig.tight_layout()
                out_path = os.path.join(
                    args.out_dir, f"{out_prefix}hot_devices_layer_{layer:03d}.png"
                )
                fig.savefig(out_path, dpi=160)
                plt.close(fig)

    print(f"Saved analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()
