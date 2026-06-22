"""Benchmark `grouped_topk_pallas` (argmax-selection) vs the 3-`sort` `_biased_grouped_topk`.

Measures the **routing** device time of each path on a TPU host (e.g. falcon v7x): the whole routing
is wrapped in a `jax.named_scope`, the gate matmul that produces `router_logits` is kept OUTSIDE the
scope, and we sum the device time of every XLA op under the scope from a profiler trace. This reads
the end-to-end routing time directly (no subtraction) and — crucially — feeds `router_logits` from a
real gate matmul so the `sort` path gets the realistic GOOD layout (an isolated jit of the routing
picks a non-representative layout that is ~15x slower; do not benchmark it standalone).

The fused kernel runs at `block_tokens="auto"` (the tuned table). Run on a TPU host:

    python -m benchmark.kernels.grouped_topk.bench_grouped_topk --T 64,128,256,512,1024,2048,4096,8192,16384,32768
"""

import argparse
import functools
import glob
import gzip
import json
import os
import re
import time

import jax
import jax.numpy as jnp

try:
    # Real path on a TPU host with sgl_jax installed.
    from sgl_jax.srt.kernels.grouped_topk.grouped_topk import grouped_topk_pallas
except Exception:  # noqa: BLE001
    # The falcon-embedded variant prepends the kernel source; `grouped_topk_pallas` is then already
    # defined at module scope, nothing to import.
    pass

TRACE_ROOT = os.environ.get("TOPK_TRACE_ROOT", "/tmp/tpu_logs/grouped_topk_bench")
H = 7168  # hidden size of the gate matmul that feeds router_logits
SCOPE_SORT = "SORTTOPK"
SCOPE_FUSED = "FUSEDTOPK"


def ref_biased_grouped_topk(router_logits, correction_bias, *, num_expert_group, topk_group, topk):
    """Verbatim `gate.py:TopK._biased_grouped_topk` (the 3-`sort` reference)."""
    router_logits = router_logits.astype(jnp.float32)
    nt = router_logits.shape[0]
    s = router_logits.reshape(nt, -1) + jnp.expand_dims(correction_bias, 0)
    sg = s.reshape(nt, num_expert_group, -1)
    gscore = jnp.sum(jax.lax.top_k(sg, k=2)[0], axis=-1)
    gi = jax.lax.top_k(gscore, k=topk_group)[1]
    gm = jnp.clip(jax.nn.one_hot(gi, num_expert_group).sum(axis=1), 0, 1)
    epg = router_logits.shape[-1] // num_expert_group
    sm = jnp.broadcast_to(jnp.expand_dims(gm, -1), (nt, num_expert_group, epg)).reshape(nt, -1)
    tmp = jnp.where(sm, s, float("-inf"))
    ids = jax.lax.top_k(tmp, k=topk)[1]
    w = jnp.take_along_axis(router_logits, ids, axis=1)
    return w, ids


def _gate(hidden, w_gate):
    return jax.nn.sigmoid(jnp.dot(hidden, w_gate, precision=jax.lax.Precision.HIGHEST))


def make_sort(w_gate, bias, G, Gtop, k):
    def fn(hidden):
        logits = _gate(hidden, w_gate)  # gate OUTSIDE the scope
        with jax.named_scope(SCOPE_SORT):
            return ref_biased_grouped_topk(
                logits, bias, num_expert_group=G, topk_group=Gtop, topk=k
            )

    return fn


def make_fused(w_gate, bias, G, Gtop, k):
    def fn(hidden):
        logits = _gate(hidden, w_gate)  # gate OUTSIDE the scope
        with jax.named_scope(SCOPE_FUSED):
            return grouped_topk_pallas(
                logits,
                bias,
                num_expert_group=G,
                topk_group=Gtop,
                topk=k,
                block_tokens="auto",
                interpret=False,
            )

    return fn


def _trace_scope_us(run_fn, scope, tag, warmup=3, iters=20):
    """Per-iter device time (us) summed over all XLA ops whose name/args contain `scope`, plus the
    count of distinct matched op names (a sanity check that the scope was found)."""
    tag = re.sub(r"[^A-Za-z0-9]", "_", tag)
    for _ in range(warmup):
        jax.block_until_ready(run_fn())
    troot = os.path.join(TRACE_ROOT, f"{tag}_{os.getpid()}_{int(time.time() * 1000)}")
    os.makedirs(troot, exist_ok=True)
    with jax.profiler.trace(troot):
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation("s", step_num=i):
                jax.block_until_ready(run_fn())
    dirs = glob.glob(os.path.join(troot, "plugins", "profile", "*"))
    if not dirs:
        return float("nan"), 0
    latest = max(dirs, key=os.path.getmtime)
    evs = []
    for tf in sorted(glob.glob(os.path.join(latest, "*.trace.json.gz"))):
        evs += json.load(gzip.open(tf)).get("traceEvents", [])
    pn, tn = {}, {}
    for e in evs:
        if e.get("ph") == "M":
            a = e.get("args", {})
            if e["name"] == "process_name":
                pn[e["pid"]] = a.get("name", "")
            if e["name"] == "thread_name":
                tn[(e["pid"], e["tid"])] = a.get("name", "")
    nmod = 0
    scope_tot = 0.0
    names = set()
    for e in evs:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        if pn.get(e["pid"], "") != "/device:TPU:0":
            continue
        t = tn.get((e["pid"], e["tid"]), "")
        dur = e["dur"]
        ddp = e.get("args", {}).get("device_duration_ps")
        if ddp:
            dur = float(ddp) / 1e6
        if t == "XLA Modules":
            nmod += 1
            continue
        if t != "XLA Ops":
            continue
        blob = e["name"] + " " + json.dumps(e.get("args", {}))
        if scope in blob:
            scope_tot += dur
            names.add(re.sub(r"\(\d+\)$", "", e["name"]))
    return scope_tot / max(nmod, 1), len(names)


def bench_config(name, E, G, Gtop, k, Ts):
    w_gate = jax.device_put(jax.random.normal(jax.random.PRNGKey(3), (H, E), dtype=jnp.float32))
    bias = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1)
    sfn = jax.jit(make_sort(w_gate, bias, G, Gtop, k))
    ffn = jax.jit(make_fused(w_gate, bias, G, Gtop, k))
    print(f"\n=== {name} (E={E}, G={G}, Gtop={Gtop}, k={k}) ===")
    print(f"{'T':>7} {'sort_us':>9} {'fused_us':>9} {'speedup':>8} | {'nS':>3} {'nF':>3}")
    for T in Ts:
        h = jax.device_put(jax.random.normal(jax.random.PRNGKey(5), (T, H), dtype=jnp.float32))
        jax.block_until_ready(sfn(h))
        jax.block_until_ready(ffn(h))
        sort_us, nS = _trace_scope_us(functools.partial(sfn, h), SCOPE_SORT, f"sort_{name}_{T}")
        fused_us, nF = _trace_scope_us(functools.partial(ffn, h), SCOPE_FUSED, f"fused_{name}_{T}")
        sp = sort_us / fused_us if (fused_us == fused_us and fused_us > 0) else float("nan")
        print(f"{T:>7} {sort_us:>9.2f} {fused_us:>9.2f} {sp:>7.2f}x | {nS:>3} {nF:>3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="64,128,256,512,1024,2048,4096,8192,16384,32768")
    ap.add_argument(
        "--configs", default="A_E256:256/8/4/8,B_E512:512/8/4/8", help="name:E/G/Gtop/k comma list"
    )
    a = ap.parse_args()
    print(f"JAX {jax.__version__} | {jax.devices()[0].platform} | n_dev {len(jax.devices())}")
    try:
        import libtpu

        print("libtpu", libtpu.__version__)
    except Exception:  # noqa: BLE001
        pass
    print(
        "routing device time = sum of ops under named_scope (gate matmul outside scope; "
        "router_logits gate-fed for the realistic good layout). No subtraction."
    )
    Ts = [int(x) for x in a.T.split(",")]
    for spec in a.configs.split(","):
        name, cfg = spec.split(":")
        E, G, Gtop, k = (int(v) for v in cfg.split("/"))
        bench_config(name, E, G, Gtop, k, Ts)


if __name__ == "__main__":
    main()
