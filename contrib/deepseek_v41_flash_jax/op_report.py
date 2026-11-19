"""Top individual XLA ops (by name) from a profiler trace dir, with their long_name/source: python3 op_report.py <dir> [n]"""

import collections
import glob
import gzip
import json
import os
import re
import sys

root, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15
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
xs = [e for e in ev if e.get("ph") == "X" and e.get("pid") == pid and "dur" in e]
mods = [e["dur"] / 1000 for e in xs if e["tid"] in mod_t]
_md = [e["dur"] for e in xs if e["tid"] in mod_t]
nm = (
    max(1, sum(1 for d in _md if d >= 0.5 * max(_md))) if _md else 1
)  # calls of the main step module (the longest modules)
print(f"module calls: {len(mods)}  mean {sum(mods)/max(1,len(mods)):.2f} ms")
agg = collections.defaultdict(lambda: [0.0, 0, ""])
for e in xs:
    if e["tid"] in ops_t:
        a = agg[e["name"]]
        a[0] += e["dur"] / 1000
        a[1] += 1
        if not a[2]:
            ar = e.get("args", {})
            a[2] = (ar.get("long_name") or ar.get("tf_op") or ar.get("hlo_op") or "")[:170]
tot = sum(v[0] for v in agg.values())
print(f"summed op time {tot:.1f} ms over {nm} module call(s)")
for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]:
    print(f"{100*v[0]/tot:5.1f}% {v[0]/nm:8.3f} ms/call n={v[1]//nm:<5} {k[:44]:<44} {v[2]}")
