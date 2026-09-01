#!/usr/bin/env python3
"""Reconstruct one EPD request's critical path (including wait segments) from
the CPU-sim traces and render a self-contained HTML waterfall.

Unlike the aggregated flame graph (CPU self-time), this shows a single request
in time order — where it *waits* (encoder round-trip / RTT / transfer) vs where
it *computes* (prefill / decode) — which is where EPD latency usually hides.

    python scripts/disaggregation/trace_to_timeline_html.py \
        --profiler-dir /tmp/epd-sim-profile --rtt-ms 30

Opens with: open /tmp/epd-sim-profile/epd_timeline.html
Only the standard library is required.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import html
import json
import os
import statistics


def _load_X(pattern: str, names: set[str] | None = None):
    files = glob.glob(pattern)
    if not files:
        return []
    ev = json.load(gzip.open(files[0]))["traceEvents"]
    out = []
    for e in ev:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if names is None or e["name"] in names or any(e["name"].startswith(n) for n in names):
            out.append((e["ts"], e["dur"], e["name"]))
    return sorted(out)


def _cluster_requests(spans, gap_threshold_us: float):
    """Group sequential sim_device_wait spans into per-request clusters.
    Returns list of (preceding_gap_us, [ (ts,dur), ... ])."""
    clusters = []
    cur = []
    prev_end = None
    gap_before = 0.0
    for ts, dur, _ in spans:
        if prev_end is None:
            gap_before = 0.0
        else:
            g = ts - prev_end
            if g > gap_threshold_us and cur:
                clusters.append((gap_before, cur))
                cur = []
                gap_before = g
        cur.append((ts, dur))
        prev_end = ts + dur
    if cur:
        clusters.append((gap_before, cur))
    return clusters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiler-dir", default="/tmp/epd-sim-profile")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--rtt-ms", type=float, default=None, help="one-way RTT used, to annotate the wait"
    )
    ap.add_argument("--gap-threshold-ms", type=float, default=25.0)
    args = ap.parse_args()

    lang = _load_X(
        os.path.join(args.profiler_dir, "language", "plugins", "profile", "*", "*.trace.json.gz"),
        {"sim_device_wait"},
    )
    enc = _load_X(
        os.path.join(args.profiler_dir, "encoder_0", "plugins", "profile", "*", "*.trace.json.gz"),
        {"mm_encode"},
    )
    if not lang:
        print("no sim_device_wait spans found; run the driver first")
        return 1

    clusters = _cluster_requests(lang, args.gap_threshold_ms * 1000)
    # Drop the first cluster (warmup / no clean preceding gap), pick the median.
    usable = [c for c in clusters[1:] if c[0] > 0 and len(c[1]) >= 2]
    if not usable:
        usable = clusters
    usable.sort(key=lambda c: c[1][-1][0] + c[1][-1][1] - c[1][0][0])
    rep = usable[len(usable) // 2]
    gap_us, spans = rep

    prefill_us = spans[0][1]
    # A ~0-duration "prefill" (or overlapping spans) means the cluster is not a
    # clean single request: the capture was concurrent / chunked, so per-request
    # segmentation is invalid. Flag it instead of showing misleading zeros.
    degenerate = prefill_us < 500  # < 0.5 ms
    decode = spans[1:]
    decode_total_us = sum(d for _, d in decode)
    # within-cluster orchestration (gaps between decode steps) = sampler + scheduling
    orch_us = (spans[-1][0] + spans[-1][1] - spans[0][0]) - prefill_us - decode_total_us
    enc_mean_us = statistics.mean([d for _, d, _ in enc]) if enc else 0.0

    total_us = gap_us + prefill_us + decode_total_us + max(0.0, orch_us)

    # Break down the embedding-wait bar.
    rtt_us = (args.rtt_ms * 2 * 1000) if args.rtt_ms is not None else 0.0  # two hops
    wait_parts = []
    if rtt_us:
        wait_parts.append(("network RTT (2 hops)", min(rtt_us, gap_us), "#d98c00"))
    if enc_mean_us:
        wait_parts.append(("encoder ViT (modeled)", enc_mean_us, "#c0392b"))
    accounted = sum(v for _, v, _ in wait_parts)
    wait_parts.append(("dispatch + transfer + coord", max(0.0, gap_us - accounted), "#e0a458"))

    def ms(us):
        return us / 1000.0

    def pct(us):
        return us / total_us * 100 if total_us else 0

    # ---- render HTML (self-contained: inline CSS, no JS deps) ----
    PXPMS = 900.0 / ms(total_us)  # fit request to ~900px

    def bar(x_us, w_us, label, color, sub=""):
        left = ms(x_us) * PXPMS
        width = max(1.0, ms(w_us) * PXPMS)
        tip = f"{label}: {ms(w_us):.1f} ms ({pct(w_us):.1f}%)"
        txt = label if width > 60 else ""
        return (
            f'<div class="bar" style="left:{left:.1f}px;width:{width:.1f}px;background:{color}" '
            f'title="{html.escape(tip)}"><span>{html.escape(txt)}</span></div>'
        )

    rows = []
    x = 0.0
    # wait (broken down)
    for lbl, w, col in wait_parts:
        rows.append(bar(x, w, lbl, col))
        x += w
    # prefill
    rows.append(bar(x, prefill_us, "prefill (modeled)", "#2e86c1"))
    x += prefill_us
    # decode strip (each step a thin bar) + orchestration folded
    decode_start = x
    # scale decode block to decode_total+orch so it lines up with total
    dblock = decode_total_us + max(0.0, orch_us)
    step_w = dblock / max(1, len(decode))
    for i in range(len(decode)):
        rows.append(bar(decode_start + i * step_w, step_w, "", "#28b463"))
    x += dblock

    legend = [
        ("network RTT (2 hops)", "#d98c00"),
        ("encoder ViT", "#c0392b"),
        ("dispatch + transfer + coord", "#e0a458"),
        ("prefill", "#2e86c1"),
        (f"decode x{len(decode)} + sampler", "#28b463"),
    ]
    legend_html = " ".join(
        f'<span class="lg"><i style="background:{c}"></i>{html.escape(n)}</span>' for n, c in legend
    )

    # summary table
    def trow(name, us, note=""):
        return f"<tr><td>{html.escape(name)}</td><td>{ms(us):.1f} ms</td><td>{pct(us):.1f}%</td><td>{html.escape(note)}</td></tr>"

    table = [
        trow(
            "embedding round-trip (wait)",
            gap_us,
            "EPD-specific: encoder dispatch + RTT + ViT + transfer",
        ),
        (
            trow("  ├ network RTT (2 hops)", rtt_us, "from --simulate-network-rtt-ms")
            if rtt_us
            else ""
        ),
        trow("  ├ encoder ViT (modeled)", enc_mean_us, "from encoder trace") if enc_mean_us else "",
        trow(
            "  └ dispatch+transfer+coord", max(0.0, gap_us - rtt_us - enc_mean_us), "HTTP/ZMQ/queue"
        ),
        trow("prefill (modeled)", prefill_us, "device compute"),
        trow(f"decode x{len(decode)} (modeled)", decode_total_us, "device compute, per step"),
        trow("sampler + orchestration", max(0.0, orch_us), "runs on host between steps"),
    ]

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>EPD CPU-sim — single request critical path</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,monospace;margin:24px;color:#222}}
 h1{{font-size:18px}} h2{{font-size:14px;color:#555;margin-top:24px}}
 .track{{position:relative;height:40px;margin:8px 0 4px;background:#f4f4f4;border:1px solid #ddd}}
 .bar{{position:absolute;top:2px;height:36px;border-right:1px solid rgba(255,255,255,.6);
       color:#fff;font-size:11px;overflow:hidden;white-space:nowrap}}
 .bar span{{padding:2px 4px;display:inline-block;line-height:34px}}
 .ruler{{position:relative;height:16px;font-size:10px;color:#888}}
 .tick{{position:absolute;border-left:1px solid #ccc;height:6px;padding-left:2px}}
 .lg{{margin-right:14px;font-size:12px;white-space:nowrap}}
 .lg i{{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:-1px}}
 table{{border-collapse:collapse;margin-top:8px;font-size:12px}}
 td,th{{border:1px solid #ddd;padding:4px 8px;text-align:left}}
 .note{{color:#666;font-size:12px;margin-top:6px;max-width:900px}}
</style></head><body>
<h1>EPD CPU-sim — single request critical path (representative request)</h1>
{'<div style="background:#c0392b;color:#fff;padding:8px;font-size:13px;border-radius:4px">⚠ This capture looks CONCURRENT / chunked (representative prefill ≈ 0 ms), so per-request reconstruction below is UNRELIABLE. The single-request timeline is only valid for a sequential drive (CONCURRENCY=1). Under concurrency use the flame graph / Perfetto.</div>' if degenerate else ''}
<div class="note">Time flows left → right. <b>Amber/red = waiting</b> (encoder round-trip, network RTT),
<b>blue/green = compute</b> (prefill / decode). Total request ≈ {ms(total_us):.0f} ms.
This is the view for the "wait-type" latency the aggregated flame graph cannot show.</div>
<h2>Waterfall</h2>
<div>{legend_html}</div>
<div class="track">{''.join(rows)}</div>
<div class="ruler">{''.join(f'<span class="tick" style="left:{i*100*PXPMS:.0f}px">{i*100}ms</span>' for i in range(int(ms(total_us)//100)+1))}</div>
<h2>Per-stage breakdown</h2>
<table><tr><th>stage</th><th>time</th><th>% of request</th><th>note</th></tr>
{''.join(t for t in table if t)}
</table>
<div class="note">Absolute ms depend on the --simulate-* coefficients you set; plug in TPU-measured
values to make them real. The <b>shape</b> (how much is wait vs compute, and where the
embedding round-trip sits) is what to read. Encoder ViT is placed inside the wait window
(cross-process clocks aren't aligned, so its position within the wait is illustrative).</div>
</body></html>"""

    out = args.out or os.path.join(args.profiler_dir, "epd_timeline.html")
    open(out, "w").write(doc)
    print(f"wrote {out}")
    if degenerate:
        print(
            "  WARNING: representative prefill ~0 ms -> capture looks concurrent/chunked; "
            "per-request timeline is unreliable. Re-capture with CONCURRENCY=1 or use the flame graph."
        )
    print(
        f"representative request ~{ms(total_us):.0f} ms: "
        f"wait {ms(gap_us):.0f} / prefill {ms(prefill_us):.0f} / "
        f"decode {ms(decode_total_us):.0f} ({len(decode)} steps) / orch {ms(max(0,orch_us)):.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
