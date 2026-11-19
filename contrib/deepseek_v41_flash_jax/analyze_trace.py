"""Summarize a JAX/XLA profiler trace (*.trace.json.gz) for one TPU chip.

Usage: python3 analyze_trace.py <profile_dir> [top_n]
Prints: per-step module times, device busy fraction, top ops by total time, and a
compute / communication / memory-movement / other split.
Standard library only; run on the host that wrote the trace.
"""

import collections
import glob
import gzip
import json
import os
import re
import sys

COMM = (
    "all-reduce",
    "all-gather",
    "reduce-scatter",
    "all-to-all",
    "collective-permute",
    "ragged-all-to-all",
    "psum",
    "send",
    "recv",
    "allreduce",
    "allgather",
)
MEM = (
    "copy",
    "transpose",
    "dynamic-slice",
    "dynamic-update-slice",
    "gather",
    "scatter",
    "concatenate",
    "pad",
    "reshape",
    "bitcast",
    "broadcast",
    "slice",
    "convert",
)
COMP = (
    "fusion",
    "convolution",
    "dot",
    "matmul",
    "ragged",
    "custom-call",
    "pallas",
    "mosaic",
    "attention",
    "softmax",
    "moe",
    "reduce",
    "sort",
    "topk",
)


def norm(name: str) -> str:
    """Group numbered instances: kernel.537 -> kernel, fusion_12.3 -> fusion."""
    n = re.sub(r"(\.\d+)+$", "", name)
    return re.sub(r"[._]\d+$", "", n)


def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in COMM):
        return "communication"
    if any(k in n for k in COMP):
        return "compute"
    if any(k in n for k in MEM):
        return "memory-movement"
    return "other"


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def main():
    root = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    files = sorted(glob.glob(os.path.join(root, "**", "*.trace.json.gz"), recursive=True))
    if not files:
        print("no *.trace.json.gz under", root)
        return
    path = files[-1]
    print("trace:", path)
    ev = load(path).get("traceEvents", [])
    pname, tname = {}, {}
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pname[e["pid"]] = e["args"]["name"]
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tname[(e["pid"], e["tid"])] = e["args"]["name"]
    tpu_pids = sorted(p for p, n in pname.items() if "TPU:0" in n or "/device:TPU:0" in n)
    if not tpu_pids:
        tpu_pids = sorted(p for p, n in pname.items() if "TPU" in n)[:1]
    pid = tpu_pids[0] if tpu_pids else None
    print("device process:", pname.get(pid))
    threads = {t: n for (p, t), n in tname.items() if p == pid}
    print("threads:", sorted(set(threads.values())))
    ops_tids = [t for t, n in threads.items() if n.lower() == "xla ops"]
    mod_tids = [t for t, n in threads.items() if n.lower() == "xla modules"]
    xs = [e for e in ev if e.get("ph") == "X" and e.get("pid") == pid and "dur" in e]

    mods = [e for e in xs if e["tid"] in mod_tids]
    if mods:
        by = collections.defaultdict(list)
        for e in mods:
            by[e["name"].split("(")[0]].append(e["dur"] / 1000.0)
        print("\n== XLA modules (per call, ms)")
        for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1]))[:8]:
            print(f"  {k[:60]:<60} n={len(v):<4} mean={sum(v)/len(v):8.2f} max={max(v):8.2f}")

    if mods:
        step = sorted(
            (e for e in mods if e["name"].startswith("jit_jitted_run")), key=lambda e: e["ts"]
        )
        if len(step) > 1:
            gaps = [(b["ts"] - (a["ts"] + a["dur"])) / 1000.0 for a, b in zip(step, step[1:])]
            durs = [e["dur"] / 1000.0 for e in step]
            print(
                f"\n== decode/prefill steps: n={len(step)} device {sum(durs)/len(durs):.2f} ms + gap {sum(gaps)/len(gaps):.2f} ms (mean) -> {sum(durs)/len(durs)+sum(gaps)/len(gaps):.2f} ms per step"
            )
            nsteps = len(step)
    ops = [e for e in xs if e["tid"] in ops_tids]
    if not ops:
        ops = xs
    tot = sum(e["dur"] for e in ops)
    span = (max(e["ts"] + e["dur"] for e in ops) - min(e["ts"] for e in ops)) if ops else 0
    print(f"\n== ops: {len(ops)} events, summed {tot/1000:.1f} ms over a {span/1000:.1f} ms window")
    agg = collections.defaultdict(lambda: [0, 0.0])
    cat = collections.defaultdict(float)
    for e in ops:
        k = norm(e["name"])
        agg[k][0] += 1
        agg[k][1] += e["dur"]
        cat[classify(k)] += e["dur"]
    print("\n== time split (summed op time)")
    for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<18} {v/1000:10.1f} ms  {100*v/max(tot,1):5.1f}%")
    print(f"\n== top {top_n} op types by total time (n = calls over the whole capture)")
    for k, (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:top_n]:
        print(f"  {100*d/max(tot,1):5.1f}%  {d/1000:9.2f} ms  n={n:<5} {k[:90]}")


if __name__ == "__main__":
    main()
