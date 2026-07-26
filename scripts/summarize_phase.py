#!/usr/bin/env python3
"""Summarize the scheduler per-step phase CSV (SGLANG_SCHED_PHASE_CSV).

Reads step,bs,recv_ms,sched_ms,build_ms,result_ms and reports the distribution
(mean + percentiles) of the host work, the forward step, and the masking ratio
host/forward -- so we can see not just the average but the tail, over a varied
workload. Pure stdlib, no numpy.

  python scripts/summarize_phase.py /tmp/phase.csv [--margin 0.85]
"""
import argparse
import csv


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def stats_line(name, xs, unit="ms"):
    if not xs:
        return f"{name:<14} (no samples)"
    mean = sum(xs) / len(xs)
    return (
        f"{name:<14} mean={mean:7.3f}  p50={pct(xs,50):7.3f}  p90={pct(xs,90):7.3f}  "
        f"p99={pct(xs,99):7.3f}  max={max(xs):7.3f}  min={min(xs):7.3f} {unit}"
    )


def histogram(xs, lo, hi, nbins=20, width=50):
    if not xs:
        return
    step = (hi - lo) / nbins
    counts = [0] * nbins
    for x in xs:
        b = int((x - lo) / step) if step > 0 else 0
        b = max(0, min(nbins - 1, b))
        counts[b] += 1
    peak = max(counts) or 1
    for i, c in enumerate(counts):
        edge = lo + i * step
        bar = "#" * int(width * c / peak)
        print(f"  [{edge:5.2f},{edge+step:5.2f})  {c:6d} |{bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--margin", type=float, default=0.85,
                    help="ratio below this = masked-with-headroom")
    args = ap.parse_args()

    host, fwd, ratio, res, build = [], [], [], [], []
    bs = []
    cols = ["step", "bs", "recv_ms", "sched_ms", "build_ms", "result_ms"]
    with open(args.csv) as f:
        first = f.readline()
        f.seek(0)
        # Tolerate a headerless CSV (e.g. if the header row was rotated away):
        # if the first line isn't the header, supply the fixed field names.
        has_header = first.startswith("step,")
        reader = csv.DictReader(f) if has_header else csv.DictReader(f, fieldnames=cols)
        for row in reader:
            h = float(row["recv_ms"]) + float(row["sched_ms"]) + float(row["build_ms"])
            r = float(row["result_ms"])
            step = h + r
            if step <= 0:
                continue
            host.append(h)
            res.append(r)
            build.append(float(row["build_ms"]))
            fwd.append(step)
            ratio.append(h / step)
            bs.append(int(row["bs"]))

    n = len(host)
    print(f"\n===== scheduler phase distribution ({n} forward steps) =====")
    print(stats_line("HOST", host))
    print(stats_line("  build", build))
    print(stats_line("result(fwd)", res))
    print(stats_line("step=H+R", fwd))
    print(stats_line("batch size", bs, unit=""))
    print()
    print(stats_line("host/forward", ratio, unit=""))

    masked = sum(1 for x in ratio if x < 1.0)
    headroom = sum(1 for x in ratio if x < args.margin)
    print(
        f"\nmasked (ratio<1.00): {masked}/{n} = {100*masked/max(1,n):.1f}%"
        f"   |   with headroom (ratio<{args.margin:.2f}): {headroom}/{n} = {100*headroom/max(1,n):.1f}%"
    )
    worst = max(ratio) if ratio else float("nan")
    print(f"worst-case ratio (p100/max): {worst:.3f}  -> "
          f"{'PASS (all masked)' if worst < 1.0 else 'FAIL (some steps host-bound)'}")

    print("\nhost/forward histogram:")
    histogram(ratio, 0.0, 1.0)

    # Verdict (also usable as a CI guard): gate on p99, not max -- a lone drain
    # step (result~=0 => ratio 1.0) is a batch-boundary artifact, not un-masking.
    p99 = pct(ratio, 99)
    ok = p99 < args.margin
    print(f"\nGUARD: p99 host/forward = {p99:.3f} {'<' if ok else '>='} margin {args.margin:.2f}"
          f"  -> {'PASS (masked)' if ok else 'FAIL (#293 regression)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
