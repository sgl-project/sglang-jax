"""Device time by source line (python file:line from the trace's `source` field) for the main step module. Usage: python3 src_report.py <profile_dir> [n] [module-name-substring]  (the substring restricts the report to ops inside matching modules, e.g. chunk_forward = decode)"""

import collections
import glob
import gzip
import json
import os
import re
import sys

root, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 25
f = sorted(glob.glob(os.path.join(root, "**", "*.trace.json.gz"), recursive=True))[-1]
ev = json.load(gzip.open(f, "rt")).get("traceEvents", [])
pn = {
    e["pid"]: e["args"]["name"]
    for e in ev
    if e.get("ph") == "M" and e.get("name") == "process_name"
}
tn = {
    (e["pid"], e["tid"]): e["args"]["name"]
    for e in ev
    if e.get("ph") == "M" and e.get("name") == "thread_name"
}
pid = sorted(p for p, v in pn.items() if "TPU:0" in v)[0]
ops_t = {t for (p, t), v in tn.items() if p == pid and v.lower() == "xla ops"}
mod_t = {t for (p, t), v in tn.items() if p == pid and v.lower() == "xla modules"}
mods = [e for e in ev if e.get("ph") == "X" and e.get("pid") == pid and e["tid"] in mod_t]
sub = sys.argv[3] if len(sys.argv) > 3 else None
if sub:
    mods = [e for e in mods if sub in e["name"]]
    ivs = sorted((e["ts"], e["ts"] + e["dur"]) for e in mods)
    import bisect

    starts = [a for a, _ in ivs]

    def inside(ts):
        j = bisect.bisect_right(starts, ts) - 1
        return j >= 0 and ts <= ivs[j][1]

else:
    inside = lambda ts: True
mx = max((e["dur"] for e in mods), default=1)
ncalls = max(1, sum(1 for e in mods if e["dur"] >= 0.5 * mx))
opn = collections.defaultdict(float)
agg = collections.defaultdict(lambda: [0.0, collections.Counter()])
tot = 0.0
nosrc = 0.0
for e in ev:
    if e.get("ph") == "X" and e.get("pid") == pid and e["tid"] in ops_t and inside(e["ts"]):
        a = e.get("args", {}) or {}
        d = e["dur"] / 1000.0
        tot += d
        src = a.get("source", "")
        m = re.findall(r"([\w./]+\.py):(\d+)", src)
        key = f"{os.path.basename(m[-1][0])}:{m[-1][1]}" if m else None
        if key is None:
            nosrc += d
            key = "(no source: " + (a.get("hlo_category") or "?") + ")"
        opn[re.sub(r"[._]?\d+$", "", re.sub(r"(\.\d+)+$", "", e["name"]))[:44]] += d
        agg[key][0] += d
        agg[key][1][a.get("hlo_category", "?")] += d
print(
    f"main step calls: {ncalls}; module mean {sum(e['dur'] for e in mods if e['dur'] >= 0.5*mx)/ncalls/1000:.1f} ms; summed op time/call {tot/ncalls:.1f} ms; without source {nosrc/ncalls:.1f} ms"
)
for k, (t, cats) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]:
    top = ", ".join(f"{c} {v/ncalls:.0f}" for c, v in cats.most_common(3))
    print(f"{t/ncalls:8.1f} ms  {100*t/tot:5.1f}%  {k:<28} [{top}]")
print("-- by op name")
for k, t in sorted(opn.items(), key=lambda kv: -kv[1])[:10]:
    print(f"{t/ncalls:8.1f} ms  {k}")
