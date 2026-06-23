"""Autotune `block_tokens` (BT) for `grouped_topk_pallas` on TPU.

Given a *local* token count T (under sequence-parallelism the topk runs on the per-device token
shard, so T is small) and a grouped-topk config (E, G, Gtop, k), this sweeps BT candidates, skips
any that overflow VMEM, times the survivors via a profiler trace, and reports the fastest BT.

Mirrors the sweep/skip/emit shape of `benchmark/kernels/mla/get_block_spec_config_mla.py`.

Run on a TPU host (e.g. falcon v7x). CPU has no Pallas-TPU device; for local logic checks pass
`--interpret` (uses interpret mode, timing is meaningless).

The paste-ready block this prints goes into `tuned_block_sizes.py:TUNED_BT[<device>]`, which
`grouped_topk_pallas(block_tokens="auto")` then looks up.

Example:
    python -m benchmark.kernels.grouped_topk.tune_grouped_topk_bt --T 256,512,1024,2048,4096,8192
"""

import argparse
import functools
import glob
import gzip
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp

try:
    # Real path on a TPU host with sgl_jax installed.
    from sgl_jax.srt.kernels.grouped_topk.v1.kernel import grouped_topk_pallas
except Exception:  # noqa: BLE001
    # The falcon-embedded variant prepends the kernel source, so `grouped_topk_pallas`
    # is already defined at module scope; nothing to import.
    pass

TRACE_ROOT = os.environ.get("TOPK_TRACE_ROOT", "/tmp/tpu_logs/topk_tune")
VMEM_DEFAULT = 64 * 2**20  # v7x VMEM if get_tpu_info() is unavailable
N_LIVE = 6  # ~live [BT,E] f32 buffers at the ③ peak (see analysis-zh.md §VMEM)


# ---- profiler-trace device-time (XLA Modules median), mirrors sort_study.py:trace_run ----
def _latest_trace(troot):
    dirs = glob.glob(os.path.join(troot, "plugins", "profile", "*"))
    if not dirs:
        return []
    latest = max(dirs, key=os.path.getmtime)
    evs = []
    for tf in sorted(glob.glob(os.path.join(latest, "*.trace.json.gz"))):
        evs += json.load(gzip.open(tf)).get("traceEvents", [])
    return evs


def _trace_ms(run_fn, tag, warmup=3, iters=20):
    for _ in range(warmup):
        jax.block_until_ready(run_fn())
    troot = os.path.join(TRACE_ROOT, f"{tag}_{os.getpid()}_{int(time.time() * 1000)}")
    os.makedirs(troot, exist_ok=True)
    with jax.profiler.trace(troot):
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation("s", step_num=i):
                jax.block_until_ready(run_fn())
    evs = _latest_trace(troot)
    pn, tn = {}, {}
    for e in evs:
        if e.get("ph") == "M":
            a = e.get("args", {})
            if e["name"] == "process_name":
                pn[e["pid"]] = a.get("name", "")
            if e["name"] == "thread_name":
                tn[(e["pid"], e["tid"])] = a.get("name", "")
    mod = [
        e["dur"]
        for e in evs
        if e.get("ph") == "X"
        and "dur" in e
        and pn.get(e["pid"]) == "/device:TPU:0"
        and tn.get((e["pid"], e["tid"])) == "XLA Modules"
    ]
    return statistics.median(mod) / 1e3 if mod else float("nan")  # us -> ms


def _candidates(T):
    """128-multiple divisors of T, plus T itself; all <= T."""
    cand = {d for d in range(128, T + 1, 128) if T % d == 0}
    cand.add(T)
    return sorted(d for d in cand if d <= T)


def _vmem_cap_bytes():
    try:
        from jax.experimental.pallas import tpu as pltpu

        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # noqa: BLE001
        return VMEM_DEFAULT


def _logits(T, E, seed=2):
    return jax.nn.sigmoid(jax.random.normal(jax.random.PRNGKey(seed), (T, E), dtype=jnp.float32))


def tune_one(T, E, G, Gtop, k, *, tries, vmem_frac, interpret, cap_bytes):
    logits = jax.device_put(_logits(T, E))
    bias = jax.device_put(jax.random.normal(jax.random.PRNGKey(1), (E,), dtype=jnp.float32) * 0.1)
    budget = vmem_frac * cap_bytes
    print(
        f"\n## T={T} E={E} G={G} Gtop={Gtop} k={k}  (VMEM cap {cap_bytes/2**20:.0f}MiB, budget {budget/2**20:.0f}MiB)"
    )
    print(f"{'BT':>6} {'estVMEM':>9} {'status':>9} {'ms':>10}")
    rows = []
    for bt in _candidates(T):
        est = N_LIVE * bt * E * 4
        if est > budget:
            print(f"{bt:>6} {est/2**20:8.1f}M {'VMEM-pre':>9} {'-':>10}")
            continue
        fn = jax.jit(
            functools.partial(
                grouped_topk_pallas,
                num_expert_group=G,
                topk_group=Gtop,
                topk=k,
                block_tokens=bt,
                interpret=interpret,
            )
        )
        try:
            jax.block_until_ready(fn(logits, bias))  # warmup / compile
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            oom = ("RESOURCE_EXHAUSTED" in msg) or ("vmem" in msg.lower())
            print(f"{bt:>6} {est/2**20:8.1f}M {('OOM-skip' if oom else 'FAIL'):>9} {'-':>10}")
            continue
        ms = min(_trace_ms(lambda: fn(logits, bias), tag=f"t{T}_e{E}_bt{bt}") for _ in range(tries))
        print(f"{bt:>6} {est/2**20:8.1f}M {'ok':>9} {ms*1e3:9.2f}u")
        rows.append((bt, ms))
    if not rows:
        print("  (no BT fit VMEM)")
        return None
    best_bt, best_ms = min(rows, key=lambda r: r[1])
    base = dict(rows).get(min(512, T))
    win = (base / best_ms - 1) * 100 if base else float("nan")
    print(
        f"  best BT={best_bt}  {best_ms*1e3:.2f}us"
        + (f"   ({win:+.1f}% vs BT={min(512,T)} baseline)" if base else "")
    )
    return best_bt, best_ms, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--configs", default="256/8/4/8,512/8/4/8", help="E/G/Gtop/k comma list")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--vmem-frac", type=float, default=0.9)
    ap.add_argument("--interpret", action="store_true")
    a = ap.parse_args()
    print(f"JAX {jax.__version__} | {jax.devices()[0].platform} | n_dev {len(jax.devices())}")
    try:
        import libtpu

        print("libtpu", libtpu.__version__)
    except Exception:  # noqa: BLE001
        pass
    cap = _vmem_cap_bytes()
    dev = jax.devices()[0].device_kind
    Ts = [int(x) for x in a.T.split(",")]
    cfgs = [tuple(int(v) for v in c.split("/")) for c in a.configs.split(",")]
    best = {}
    for E, G, Gtop, k in cfgs:
        for T in Ts:
            r = tune_one(
                T,
                E,
                G,
                Gtop,
                k,
                tries=a.tries,
                vmem_frac=a.vmem_frac,
                interpret=a.interpret,
                cap_bytes=cap,
            )
            if r:
                best[(T, E, G, Gtop, k)] = r[0]
    # paste-ready block for tuned_block_sizes.py: (device, pow2(T), E, G, Gtop, k) -> BT
    print(f"\n# --- paste-ready for TUNED_BT[{dev!r}] ---")
    for (T, E, G, Gtop, k), bt in sorted(best.items()):
        p2 = 1 << (T - 1).bit_length()
        print(f"#   ({p2}, {E}, {G}, {Gtop}, {k}): {bt},")


if __name__ == "__main__":
    main()
