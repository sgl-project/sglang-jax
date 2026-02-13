#!/usr/bin/env python3
"""
Plot per-layer expert balance counts over inference segments.

Input CSV is produced by expert balance debug (expert_balance_*.csv).
Outputs one PNG per layer with multiple lines derived from raw expert counts:
max/min/mean/std plus hot/cold top-k means. Device subplot is derived by
grouping experts per device.
Optionally render heatmaps (experts × segments) to visualize balance.
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


def _decode_counts(row: dict, num_experts: int) -> "np.ndarray":
    import numpy as np

    counts_str = row.get("experts_count", "")
    if counts_str:
        parts = [p for p in counts_str.replace(" ", "").split(",") if p != ""]
        arr = np.array([int(p) for p in parts], dtype=np.int64)
    else:
        b64 = row.get("experts_count_b64", "")
        if not b64:
            return np.zeros(num_experts, dtype=np.int64)
        import base64

        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.int64)

    if num_experts and arr.size != num_experts:
        if arr.size > num_experts:
            return arr[:num_experts]
        out = np.zeros(num_experts, dtype=np.int64)
        out[: arr.size] = arr
        return out
    return arr


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
        "--out-prefix",
        default="",
        help="Filename prefix for outputs (e.g. 'expert_' or 'device_').",
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
        "--skip-initial",
        type=int,
        default=0,
        help="Ignore first N segments in plots.",
    )
    parser.add_argument(
        "--balance-topk",
        type=int,
        default=4,
        help="Top-k size for hot/cold mean counts.",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate per-layer heatmaps (experts × segments).",
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Only generate heatmaps (skip line plots).",
    )
    parser.add_argument(
        "--heatmap-max-segments",
        type=int,
        default=300,
        help="Max columns in heatmap; segments will be binned if exceeded.",
    )
    parser.add_argument(
        "--heatmap-bin-size",
        type=int,
        default=0,
        help="Fixed bin size for heatmap (overrides --heatmap-max-segments if >0).",
    )
    parser.add_argument(
        "--heatmap-agg",
        choices=["mean", "sum"],
        default="mean",
        help="Aggregation for binned segments in heatmap.",
    )
    parser.add_argument(
        "--heatmap-normalize",
        choices=["none", "mean"],
        default="none",
        help="Normalize counts by per-segment mean (ratio).",
    )
    parser.add_argument(
        "--heatmap-experts-per-device",
        type=int,
        default=0,
        help="Group experts by this size before heatmap (e.g. 4).",
    )
    parser.add_argument(
        "--heatmap-log",
        action="store_true",
        help="Use log1p scale in heatmap.",
    )
    parser.add_argument(
        "--heatmap-vmax-percentile",
        type=float,
        default=99.5,
        help="Clip heatmap vmax to this percentile for better contrast.",
    )
    parser.add_argument(
        "--heatmap-vmin",
        type=float,
        default=0.0,
        help="Heatmap vmin.",
    )
    parser.add_argument(
        "--heatmap-cmap",
        default="magma",
        help="Colormap for heatmap.",
    )
    parser.add_argument(
        "--heatmap-discrete-step",
        type=float,
        default=0.0,
        help="If >0, bucket values >=1 by this step; values <1 share one color.",
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
    out_prefix = args.out_prefix or ""

    # Group rows by layer
    by_layer: dict[int, list[dict]] = {layer: [] for layer in layers}
    for r in rows:
        layer = int(r["layer"])
        if layer not in by_layer:
            continue
        if args.skip_empty:
            total = r.get("total_assignments")
            if total is not None and int(total) == 0:
                continue
        by_layer[layer].append(r)

    for layer, layer_rows in by_layer.items():
        if not layer_rows:
            continue
        layer_rows.sort(key=lambda x: int(x["segment_idx"]))

        if args.skip_initial > 0:
            layer_rows = layer_rows[args.skip_initial :]

        if not layer_rows:
            continue

        seg_idx = [int(r["segment_idx"]) for r in layer_rows]
        x = seg_idx
        x_label = "segment_idx"
        num_experts = int(layer_rows[0]["num_experts"])
        ep_size = int(layer_rows[0].get("ep_size", "0") or 0)
        experts_per_device = (
            (num_experts // ep_size) if ep_size > 0 and num_experts % ep_size == 0 else None
        )

        layer_has_device = experts_per_device is not None and experts_per_device > 0

        if args.heatmap:
            import numpy as np

            seg_count = len(layer_rows)
            bin_size = args.heatmap_bin_size if args.heatmap_bin_size > 0 else 0
            if bin_size <= 0 and seg_count > args.heatmap_max_segments:
                bin_size = int(np.ceil(seg_count / args.heatmap_max_segments))
            if bin_size <= 0:
                bin_size = 1

            bins = []
            group_size = args.heatmap_experts_per_device
            grouped = False
            num_groups = num_experts
            if group_size and group_size > 0:
                num_groups = num_experts // group_size
                if num_groups == 0:
                    num_groups = 1
                if num_experts % group_size != 0:
                    print(
                        f"[heatmap] layer {layer}: num_experts={num_experts} "
                        f"not divisible by group_size={group_size}, truncating."
                    )
                grouped = True

            for i in range(0, seg_count, bin_size):
                chunk = layer_rows[i : i + bin_size]
                counts_list = [_decode_counts(r, num_experts) for r in chunk]
                if not counts_list:
                    continue
                mat = np.stack(counts_list, axis=1)  # experts × steps
                if grouped and group_size > 0:
                    usable = num_groups * group_size
                    mat = mat[:usable, :]
                    mat = mat.reshape(num_groups, group_size, mat.shape[1]).sum(axis=1)
                if args.heatmap_normalize == "mean":
                    means = []
                    for r in chunk:
                        total = int(r.get("total_assignments", 0) or 0)
                        denom = num_groups if grouped else num_experts
                        means.append((total / denom) if denom > 0 else 0.0)
                    mean_arr = np.array(means, dtype=np.float64)
                    mean_arr[mean_arr <= 0] = 1.0
                    mat = mat / mean_arr[None, :]
                if args.heatmap_agg == "sum":
                    agg = mat.sum(axis=1)
                else:
                    agg = mat.mean(axis=1)
                bins.append(agg)

            if bins:
                heat = np.stack(bins, axis=1)  # experts × binned segments
                use_log = args.heatmap_log
                if args.heatmap_discrete_step > 0 and args.heatmap_normalize == "mean":
                    use_log = False
                if use_log:
                    heat = np.log1p(heat)
                vmax = None
                if heat.size:
                    vmax = float(np.percentile(heat, args.heatmap_vmax_percentile))
                    if vmax <= args.heatmap_vmin:
                        vmax = None

                fig = plt.figure(figsize=(13, 6))
                gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1])
                ax_hm = fig.add_subplot(gs[0, 0])
                ax_note = fig.add_subplot(gs[0, 1])
                if args.heatmap_discrete_step > 0 and args.heatmap_normalize == "mean":
                    import numpy as np
                    from matplotlib import colors

                    step = args.heatmap_discrete_step
                    upper = float(np.max(heat)) if heat.size else 1.0
                    upper = max(upper, 1.0 + step)
                    boundaries = [0.0, 1.0]
                    val = 1.0 + step
                    while val < upper + step * 0.5:
                        boundaries.append(val)
                        val += step
                    cmap = plt.get_cmap(args.heatmap_cmap, len(boundaries) - 1)
                    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
                    im = ax_hm.imshow(
                        heat,
                        aspect="auto",
                        origin="lower",
                        interpolation="nearest",
                        cmap=cmap,
                        norm=norm,
                        vmin=None,
                        vmax=None,
                    )
                else:
                    im = ax_hm.imshow(
                        heat,
                        aspect="auto",
                        origin="lower",
                        interpolation="nearest",
                        vmin=args.heatmap_vmin,
                        vmax=vmax,
                        cmap=args.heatmap_cmap,
                    )
                ax_hm.set_title(f"Layer {layer} - expert heatmap")
                ax_hm.set_xlabel("segment_bin")
                ax_hm.set_ylabel("device_idx" if grouped else "expert_idx")
                cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
                if args.heatmap_normalize == "mean":
                    cbar.set_label("x mean", rotation=90)
                else:
                    cbar.set_label("count", rotation=90)
                ax_note.axis("off")
                note = (
                    f"segments={seg_count}\n"
                    f"bin_size={bin_size}\n"
                    f"agg={args.heatmap_agg}\n"
                    f"normalize={args.heatmap_normalize}\n"
                    f"log1p={use_log}\n"
                    f"vmax_p={args.heatmap_vmax_percentile}\n"
                    f"cmap={args.heatmap_cmap}\n"
                    f"group_size={group_size if grouped else 1}\n"
                    f"step={args.heatmap_discrete_step}"
                )
                ax_note.text(0.0, 1.0, note, ha="left", va="top", fontsize=8)
                fig.tight_layout()
                out_path = os.path.join(args.out_dir, f"{out_prefix}layer_{layer:03d}_heatmap.png")
                fig.savefig(out_path, dpi=args.dpi)
                plt.close(fig)

            if args.heatmap_only:
                continue
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
        mean_count = []
        min_count = []
        max_count = []
        std_count = []
        hot_counts = []
        cold_counts = []
        dev_mean = []
        dev_min = []
        dev_max = []
        dev_std = []
        dev_hot_counts = []
        dev_cold_counts = []

        k_exp = max(1, min(args.balance_topk, num_experts))
        k_dev = (
            max(1, min(args.balance_topk, ep_size)) if layer_has_device and ep_size > 0 else None
        )

        for r in layer_rows:
            counts = _decode_counts(r, num_experts)
            total = int(r.get("total_assignments", counts.sum()) or 0)
            mu = (total / num_experts) if num_experts > 0 else 0.0
            mean_count.append(mu)
            min_count.append(float(counts.min()) if counts.size else 0.0)
            max_count.append(float(counts.max()) if counts.size else 0.0)
            std_count.append(float(counts.std()) if counts.size else 0.0)
            if counts.size:
                sorted_counts = counts if counts.size == 1 else counts.copy()
                if counts.size > 1:
                    sorted_counts.sort()
                hot_counts.append(float(sorted_counts[-k_exp:].mean()))
                cold_counts.append(float(sorted_counts[:k_exp].mean()))
            else:
                hot_counts.append(0.0)
                cold_counts.append(0.0)

            if layer_has_device and experts_per_device is not None:
                dev_counts = counts.reshape(ep_size, experts_per_device).sum(axis=1)
                dev_total = int(dev_counts.sum())
                dev_mu = (dev_total / ep_size) if ep_size > 0 else 0.0
                dev_mean.append(dev_mu)
                dev_min.append(float(dev_counts.min()) if dev_counts.size else 0.0)
                dev_max.append(float(dev_counts.max()) if dev_counts.size else 0.0)
                dev_std.append(float(dev_counts.std()) if dev_counts.size else 0.0)
                if dev_counts.size:
                    dev_sorted = dev_counts if dev_counts.size == 1 else dev_counts.copy()
                    if dev_counts.size > 1:
                        dev_sorted.sort()
                    dev_hot_counts.append(float(dev_sorted[-k_dev:].mean()))
                    dev_cold_counts.append(float(dev_sorted[:k_dev].mean()))
                else:
                    dev_hot_counts.append(0.0)
                    dev_cold_counts.append(0.0)

        ax_exp.plot(x, max_count, label="max_count")
        ax_exp.plot(x, min_count, label="min_count")
        ax_exp.plot(x, hot_counts, label="hot_topk_mean_count")
        ax_exp.plot(x, cold_counts, label="cold_topk_mean_count")
        ax_exp.plot(x, mean_count, label="mean_count")
        ax_exp.plot(x, std_count, label="std_count")

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
            ax_dev.plot(x, dev_max, label="device max_count")
            ax_dev.plot(x, dev_min, label="device min_count")
            ax_dev.plot(x, dev_mean, label="device mean_count")
            ax_dev.plot(x, dev_std, label="device std_count")
            ax_dev.plot(x, dev_hot_counts, label="device hot_topk_mean_count")
            ax_dev.plot(x, dev_cold_counts, label="device cold_topk_mean_count")
            ax_dev.set_title(f"Layer {layer} - device level")
            ax_dev.set_ylabel("count")
            ax_dev.grid(True, alpha=0.3)
            ax_dev.legend(loc="upper right")

        ax_exp.set_xlabel(x_label)
        ax_note.axis("off")
        note = (
            "Counts (derived from experts_count):\n"
            "max/min/mean/std counts\n"
            "hot_topk_mean_count = avg top‑k counts\n"
            "cold_topk_mean_count = avg coldest top‑k\n"
            f"topk = {k_exp}\n"
        )
        if layer_has_device:
            note += (
                "\nDevice subplot (derived by grouping experts):\n"
                "device max/min/mean/std counts\n"
                "device hot/cold top‑k mean counts"
            )
        ax_note.text(
            0.0,
            1.0,
            note,
            ha="left",
            va="top",
            fontsize=8,
        )
        fig.tight_layout()

        out_path = os.path.join(args.out_dir, f"{out_prefix}layer_{layer:03d}.png")
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
