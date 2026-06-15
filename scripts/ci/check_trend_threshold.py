#!/usr/bin/env python3
"""Detect nightly trend drift from dashboard observability JSON.

Reads result JSON under --data-dir (the /observability-storage mount), compares
a recent window vs a baseline window per (config, metric), writes adverse drifts
to --output. Pure (no network/gh) so it runs on the TPU runner that has the
mount; alerting from the report is a separate step (trend_alert_issues.py).
"""
import argparse
import json
import math
import os
import statistics
import sys

# Field names and metric keys are shared by convention with the producer
# (test/srt/nightly/results.py); no enforced contract, so keep them in sync.
# Adverse direction: "higher" metrics (throughput/score) drift on a DROP,
# "lower" metrics (latency) on a RISE.
DIRECTION = {
    "input_throughput": "higher",
    "output_throughput": "higher",
    "median_ttft_ms": "lower",
    "median_itl_ms": "lower",
    "score": "higher",
}


def _perf_metric_names(random_output_len):
    # Prefill (output_len==1) has no meaningful decode metrics (out_tps / ITL).
    base = ["input_throughput", "median_ttft_ms"]
    if random_output_len > 1:
        base += ["output_throughput", "median_itl_ms"]
    return base


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _positive_finite(x):
    # Valid metric/score: a positive finite number (excludes bool). 0/negative is
    # a broken point (0 score = failed eval); a low positive score is real drift.
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def _run_id_from_workload(workload):
    return workload.rsplit("-", 1)[-1] if "-" in workload else workload


def _load_one(obj, date, workload):
    if not isinstance(obj, dict):
        return None  # a stray non-dict JSON on the shared mount — skip, don't crash
    kind = obj.get("type")
    if kind == "perf":
        rol = obj.get("random_output_len")
        names = _perf_metric_names(rol) if isinstance(rol, int) else []
        metrics = {n: obj[n] for n in names if _positive_finite(obj.get(n))}
        valid = obj.get("completed") not in (None, 0) and bool(metrics)
        return {
            "kind": "perf",
            "config": obj.get("profile"),
            "date": date,
            "run_id": _run_id_from_workload(workload),
            "point": (obj.get("max_concurrency"), obj.get("random_input_len"), rol),
            "metrics": metrics,
            "valid": valid,
        }
    if kind == "accuracy":
        score = obj.get("score")
        # A low positive score (below gate) is real drift and kept; only
        # null / non-finite / <=0 (failed eval) is dropped.
        metrics = {"score": score} if _positive_finite(score) else {}
        valid = _positive_finite(score)
        return {
            "kind": "accuracy",
            "config": obj.get("case"),
            "date": date,
            "run_id": _run_id_from_workload(workload),
            "point": None,
            "metrics": metrics,
            "valid": valid,
        }
    return None


def _hashable_point(point):
    """``point`` is None (accuracy) or a tuple of scalar perf dims. A non-scalar
    element would make the (config, point, name) series key unhashable, so skip it."""
    return point is None or all(isinstance(p, (int, float, str)) or p is None for p in point)


def load_records(data_dir, max_dates=None):
    """Load every ``<data_dir>/<date>/<workload>/<case>.json`` into flat records.

    Only the most recent ``max_dates`` date dirs are read (YYYY-MM-DD names sort
    chronologically), bounding the per-night gcsfuse read cost as history grows.
    """
    records = []
    dates = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if max_dates is not None and max_dates > 0:
        dates = dates[-max_dates:]
    for date in dates:
        date_path = os.path.join(data_dir, date)
        for workload in sorted(os.listdir(date_path)):
            wpath = os.path.join(date_path, workload)
            if not os.path.isdir(wpath):
                continue
            for fn in sorted(os.listdir(wpath)):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(wpath, fn)) as f:
                        obj = json.load(f)
                    rec = _load_one(obj, date, workload)
                except (json.JSONDecodeError, OSError, AttributeError, TypeError, ValueError):
                    continue  # skip a corrupt/odd-shaped file, don't crash the whole run
                # config + point form the series key, so both must be hashable.
                if (
                    rec
                    and isinstance(rec["config"], str)
                    and rec["config"]
                    and _hashable_point(rec["point"])
                ):
                    records.append(rec)
    return records


def build_series(records):
    """dict[key]->list[{date,run_id,value}] sorted by (date,run_id).
    key=(config,point,metric_name). Drops invalid; same (key,date) keeps largest run_id."""
    by_key_date = {}
    for r in records:
        if not r["valid"]:
            continue
        if not str(r["run_id"]).isdigit():
            continue
        for name, value in r["metrics"].items():
            if not _finite(value):
                continue
            key = (r["config"], r["point"], name)
            # NOTE: key has no source dimension. Before re-enabling multi-host
            # (if:false), add source or same-named single/multi-host configs merge.
            kd = (key, r["date"])
            prev = by_key_date.get(kd)
            if prev is None or int(r["run_id"]) > int(prev[0]):
                by_key_date[kd] = (r["run_id"], value)
    series = {}
    for (key, date), (run_id, value) in by_key_date.items():
        series.setdefault(key, []).append({"date": date, "run_id": run_id, "value": value})
    for key in series:
        series[key].sort(key=lambda p: (p["date"], int(p["run_id"])))
    return series


def detect_drift(points, metric, recent, baseline, threshold_pct, min_zscore):
    """Recent vs baseline median comparison; a drift detail or None.

    Median + MAD so one bad night neither moves the window nor trips a false
    alert. Drift requires BOTH an adverse change past threshold_pct AND
    >= min_zscore scaled-MAD sigmas past the baseline scatter. None until both
    windows are full (needs recent + baseline points) or the baseline is ~0.
    Windows by point count, not days.

    Detects a *change*, not a floor: a step regression alerts only while it sits
    in recent vs an older baseline, then auto-clears once it ages into the
    baseline (~15 nights) though it never recovered — floors are the suite gate's.
    """
    if len(points) < recent + baseline:
        return None
    recent_vals = [p["value"] for p in points[-recent:]]
    baseline_vals = [p["value"] for p in points[-(recent + baseline) : -recent]]
    r_med = statistics.median(recent_vals)
    b_med = statistics.median(baseline_vals)
    if abs(b_med) < 1e-9:
        return None
    pct = (r_med - b_med) / abs(b_med) * 100
    direction = DIRECTION[metric]
    over_threshold = pct < -threshold_pct if direction == "higher" else pct > threshold_pct
    if not over_threshold:
        return None
    # MAD collapses to 0 when >half the points equal the median, so only a truly
    # flat baseline (min == max) skips the z gate; else fall back to stdev.
    scaled_mad = 1.4826 * statistics.median([abs(v - b_med) for v in baseline_vals])
    dispersion = scaled_mad
    if dispersion < 1e-9 and min(baseline_vals) != max(baseline_vals):
        dispersion = statistics.stdev(baseline_vals)
    delta = abs(r_med - b_med)
    if dispersion >= 1e-9 and delta < min_zscore * dispersion:
        return None
    zscore = None if dispersion < 1e-9 else round(delta / dispersion, 2)
    return {
        "metric": metric,
        "direction": direction,
        "baseline_median": round(b_med, 4),
        "recent_median": round(r_med, 4),
        "pct_change": round(pct, 2),
        "threshold_pct": threshold_pct,
        "baseline_mad": round(scaled_mad, 4),
        "zscore": zscore,
        "min_zscore": min_zscore,
        "n_recent": len(recent_vals),
        "n_baseline": len(baseline_vals),
    }


def aggregate_by_config(drifts):
    """Group ``[(key, detail), ...]`` into ``{config: [detail + point, ...]}``."""
    agg = {}
    for (config, point, _metric), detail in drifts:
        agg.setdefault(config, []).append({**detail, "point": list(point) if point else None})
    return agg


def find_drifts(series, *, recent, baseline, accuracy_pct, perf_pct, min_zscore):
    """Run detect_drift over every series, choosing the threshold by metric kind."""
    drifts = []
    for key, points in series.items():
        _config, _point, metric = key
        threshold = accuracy_pct if metric == "score" else perf_pct
        detail = detect_drift(
            points,
            metric,
            recent=recent,
            baseline=baseline,
            threshold_pct=threshold,
            min_zscore=min_zscore,
        )
        if detail is not None:
            drifts.append((key, detail))
    return drifts


def _point_str(point):
    if not point:
        return "-"
    c, i, o = point
    return f"c{c}-i{i}-o{o}"


def write_step_summary(agg):
    """Append a human-readable drift summary to GITHUB_STEP_SUMMARY when present."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    with open(summary_file, "a") as f:
        if not agg:
            f.write("\n### Trend monitor: no drift detected\n")
            return
        f.write(f"\n### Trend monitor: {len(agg)} config(s) drifting\n\n")
        for config in sorted(agg):
            f.write(f"- **{config}**\n")
            for it in agg[config]:
                f.write(
                    "  - {metric} ({point}): {pct:+.2f}% ({direction})\n".format(
                        metric=it["metric"],
                        point=_point_str(it.get("point")),
                        pct=it["pct_change"],
                        direction=it["direction"],
                    )
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect nightly metric trend drift.")
    parser.add_argument("--data-dir", required=True, help="Root of <date>/<workload>/<case>.json")
    parser.add_argument("--output", default="drift.json", help="Where to write the drift report")
    parser.add_argument("--recent-points", type=int, default=5)
    parser.add_argument("--baseline-points", type=int, default=10)
    parser.add_argument(
        "--max-dates",
        type=int,
        default=60,
        help="Only read the most recent N date dirs (bounds per-night read cost)",
    )
    parser.add_argument("--accuracy-threshold-pct", type=float, default=2.0)
    parser.add_argument("--perf-threshold-pct", type=float, default=10.0)
    parser.add_argument(
        "--min-zscore",
        type=float,
        default=2.0,
        help="Min robust sigmas (scaled MAD) past baseline scatter to count as drift",
    )
    args = parser.parse_args(argv)
    if args.max_dates <= 0:
        parser.error("--max-dates must be a positive integer")

    if not os.path.isdir(args.data_dir):
        print(f"::warning::data-dir {args.data_dir} missing; no trend check", file=sys.stderr)
        agg = {}
    else:
        records = load_records(args.data_dir, max_dates=args.max_dates)
        if not records:
            # 0 records is indistinguishable from a healthy "no drift" in the
            # report (both {"configs": {}}), so warn — a field rename/moved mount.
            print(
                f"::warning::trend monitor loaded 0 records from {args.data_dir}; "
                "check the producer fields and mount path",
                file=sys.stderr,
            )
        series = build_series(records)
        drifts = find_drifts(
            series,
            recent=args.recent_points,
            baseline=args.baseline_points,
            accuracy_pct=args.accuracy_threshold_pct,
            perf_pct=args.perf_threshold_pct,
            min_zscore=args.min_zscore,
        )
        agg = aggregate_by_config(drifts)

    # The report is the contract with scripts/ci/trend_alert_issues.py:
    # {"schema_version": 1, "configs": {config: [drift detail, ...]}}.
    with open(args.output, "w") as f:
        json.dump({"schema_version": 1, "configs": agg}, f, indent=2, sort_keys=True)
    print(f"Wrote drift report ({len(agg)} drifting config(s)) to {args.output}")

    write_step_summary(agg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
