"""Standalone micro-bench for fused_ep_moe_v2.

Trace-based timing (device_duration_ps) matches v1 bench methodology.
Supports config sweeping via comma-separated env vars.

Env vars:
  BENCH_TOKENS  — comma-separated token counts (default: 4096)
  BENCH_BT      — comma-separated bt candidates (default: 128)
  BENCH_BF      — comma-separated bf candidates (default: 256)
  BENCH_BTC     — comma-separated btc candidates (default: 128)
  BENCH_BTS     — comma-separated bts candidates (default: auto)
  BENCH_BSE     — bse value (default: 256)
  BENCH_FP8     — 1 to enable fp8 weights
  BENCH_QBK     — quant_block_k for fp8 (default: 128)
  BENCH_TUNE    — 1 to auto-generate bt/bf candidates
  BENCH_WARMUP  — warmup iterations (default: 2)
  BENCH_ITERS   — timed iterations (default: 5)
  BENCH_CHECK   — 1 to run correctness check (single-host only)
  BENCH_D/F/E/TOPK — model dims (default: MiMo V2 Pro)
"""
from __future__ import annotations

import gzip
import itertools
import json
import os
import pathlib
import re
import sys
import time
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

t0 = time.time()
KERNEL_NAME_RE = re.compile(r"fused-moe-v2-k_.*")
TRACE_ROOT = "/tmp/tpu_logs/v2_trace"


def log(msg):
    print(f"[{time.time()-t0:.1f}s][p{jax.process_index()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Trace timing (mirrors benchmark/utils.py)
# ---------------------------------------------------------------------------

def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")
    combined: dict[str, Any] = {"traceEvents": []}
    for tf in sorted(trace_files):
        with gzip.open(tf, "rb") as fh:
            shard = json.load(fh)
        events = shard.get("traceEvents", [])
        if isinstance(events, list):
            combined["traceEvents"].extend(events)
    return combined


def _extract_durations_ms(trace: dict[str, Any]) -> list[float]:
    """Extract per-iteration device durations for the v2 kernel from trace.

    Matches events by kernel name regex (same approach as v1 bench),
    extracts device_duration_ps for accurate on-device timing.
    """
    matched = [e for e in trace.get("traceEvents", [])
               if "name" in e and KERNEL_NAME_RE.match(e["name"])]
    if not matched:
        return []
    by_pid: dict[int, list[dict[str, Any]]] = {}
    for e in matched:
        pid = e.get("pid")
        if isinstance(pid, int):
            by_pid.setdefault(pid, []).append(e)
    durations: dict[int, list[float]] = {}
    for pid, evts in by_pid.items():
        evts.sort(key=lambda x: float(x.get("ts", 0)))
        d: list[float] = []
        for e in evts:
            args = e.get("args", {})
            if args.get("device_duration_ps"):
                d.append(float(args["device_duration_ps"]) / 1e9)
            elif "dur" in e:
                d.append(float(e["dur"]) / 1e3)
        if d:
            durations[pid] = d
    if not durations:
        return []
    return max(sorted(durations.items()), key=lambda kv: len(kv[1]))[1]


def trace_timeit(run_fn, warmup: int, iters: int) -> list[float]:
    """Warmup then profile *iters* calls, return per-iter device durations (ms)."""
    for _ in range(warmup):
        out = run_fn()
        jax.block_until_ready(out)

    tag = f"{os.getpid()}_{int(time.time())}"
    trace_dir = os.path.join(TRACE_ROOT, f"run_{tag}")
    os.makedirs(trace_dir, exist_ok=True)

    with jax.profiler.trace(trace_dir):
        for i in range(iters):
            out = run_fn()
            jax.block_until_ready(out)

    if jax.process_index() != 0:
        return []
    try:
        trace = _load_trace(trace_dir)
        return _extract_durations_ms(trace)
    except FileNotFoundError:
        return []


def wall_timeit(run_fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        out = run_fn()
        jax.block_until_ready(out)
    times = []
    for _ in range(iters):
        t_start = time.monotonic()
        out = run_fn()
        jax.block_until_ready(out)
        times.append((time.monotonic() - t_start) * 1e3)
    return times


def split_timeit(run_fn, warmup: int, iters: int) -> tuple[list[float], list[float]]:
    """Measure dispatch and block_until_ready separately."""
    for _ in range(warmup):
        out = run_fn()
        jax.block_until_ready(out)
    dispatch_times = []
    wait_times = []
    for _ in range(iters):
        t0 = time.monotonic()
        out = run_fn()
        t1 = time.monotonic()
        jax.block_until_ready(out)
        t2 = time.monotonic()
        dispatch_times.append((t1 - t0) * 1e3)
        wait_times.append((t2 - t1) * 1e3)
    return dispatch_times, wait_times


# ---------------------------------------------------------------------------
# Env parsing helpers
# ---------------------------------------------------------------------------

def parse_csv_int(env_key: str, default: list[int]) -> list[int]:
    v = os.environ.get(env_key)
    if v is None:
        return default
    return [int(x.strip()) for x in v.split(",")]


def parse_csv_int_or_none(env_key: str) -> list[int | None]:
    v = os.environ.get(env_key)
    if v is None:
        return [None]
    return [int(x.strip()) for x in v.split(",")]


def align_local_tokens_for_v2(local_num_tokens: int) -> int:
    if local_num_tokens <= 8:
        return 8
    return ((local_num_tokens + 7) // 8) * 8


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

jax.distributed.initialize()
log(f"initialized: {jax.device_count()} devices, {jax.process_count()} procs")

from kernel import fused_ep_moe_v2, ref_moe, FusedMoEBlockConfig

P = jax.sharding.PartitionSpec
num_devices = jax.device_count()
devices = np.array(jax.devices()).reshape(1, num_devices)
mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
ep_size = num_devices

d = int(os.environ.get("BENCH_D", "6144"))
f = int(os.environ.get("BENCH_F", "2048"))
E = int(os.environ.get("BENCH_E", "384"))
top_k = int(os.environ.get("BENCH_TOPK", "8"))
bse = int(os.environ.get("BENCH_BSE", "256"))
warmup = int(os.environ.get("BENCH_WARMUP", "2"))
iters = int(os.environ.get("BENCH_ITERS", "5"))
check_correctness = os.environ.get("BENCH_CHECK", "0") == "1"
use_fp8 = os.environ.get("BENCH_FP8", "0") == "1"
quant_block_k = int(os.environ.get("BENCH_QBK", "128"))
tune_mode = os.environ.get("BENCH_TUNE", "0") == "1"
use_wall = os.environ.get("BENCH_WALL", "0") == "1"
use_split = os.environ.get("BENCH_SPLIT", "0") == "1"
decode_mode = os.environ.get("BENCH_DECODE_MODE", "0") == "1"
direct_scaled_dot = os.environ.get("BENCH_DIRECT_SCALED_DOT", "0") == "1"
skip_decode_sync = os.environ.get("BENCH_SKIP_DECODE_SYNC", "0") == "1"
inkernel_metadata = os.environ.get("BENCH_INKERNEL_MD", "0") == "1"
if use_split:
    timeit_fn = None
    timing_label = "split"
else:
    timeit_fn = wall_timeit if use_wall else trace_timeit
    timing_label = "wall" if use_wall else "trace"

# Ablation flags
all_disable = os.environ.get("FUSED_MOE_BENCHMARK_ALL_DISABLE", "0") == "1"
disable_a2a = all_disable or os.environ.get("DISABLE_A2A", "0") == "1"
disable_sync_barrier = all_disable or os.environ.get("DISABLE_SYNC_BARRIER", "0") == "1"
disable_weight_load = all_disable or os.environ.get("DISABLE_WEIGHT_LOAD", "0") == "1"
disable_dynamic_ffn1 = all_disable or os.environ.get("DISABLE_DYNAMIC_FFN1", "0") == "1"
disable_dynamic_ffn2 = all_disable or os.environ.get("DISABLE_DYNAMIC_FFN2", "0") == "1"
disable_acc_and_store = all_disable or os.environ.get("DISABLE_ACC_AND_STORE", "0") == "1"
ablation_flags = {
    "disable_a2a": disable_a2a,
    "disable_sync_barrier": disable_sync_barrier,
    "disable_weight_load": disable_weight_load,
    "disable_dynamic_ffn1": disable_dynamic_ffn1,
    "disable_dynamic_ffn2": disable_dynamic_ffn2,
    "disable_acc_and_store": disable_acc_and_store,
}
active_ablation = [k for k, v in ablation_flags.items() if v]
if active_ablation:
    log(f"ablation flags: {active_ablation}")
if decode_mode:
    log("decode_mode=True (single-buffer weights, serial bf loop)")
if direct_scaled_dot:
    log("direct_scaled_dot=True (fp8 dot per quant group, scale after dot)")
if skip_decode_sync:
    log("skip_decode_sync=True (skip kernel barriers only when num_bt=1)")
if inkernel_metadata:
    log("inkernel_metadata=True (in-kernel ICI allgather, no JAX lax.all_gather)")

bt_candidates = parse_csv_int("BENCH_BT", [128])
bf_candidates = parse_csv_int("BENCH_BF", [256])
btc_candidates = parse_csv_int("BENCH_BTC", [128])
bts_candidates = parse_csv_int_or_none("BENCH_BTS")
token_candidates = parse_csv_int("BENCH_TOKENS", [4096])


def generate_tune_candidates(intermediate_size, local_num_tokens, ep_size):
    bfs = sorted(set(
        v for v in [128, 256, 512, 1024, 2048]
        if v <= intermediate_size and intermediate_size % v == 0
    ))
    bt_list = []
    p = 8
    while p <= local_num_tokens:
        if local_num_tokens % p == 0:
            bt_list.append(p)
        p *= 2
    if not bt_list:
        bt_list = [local_num_tokens]
    max_bt = max(bt_list)
    if max_bt < 8:
        bts_list = [8, 16, 32]
        bts_list = [b for b in bts_list if b <= max_bt * ep_size]
        if not bts_list:
            bts_list = [None]
    else:
        bts_list = [None]
    btc_list = [128]
    return bt_list, bfs, btc_list, bts_list



if tune_mode:
    log(f"tune mode: auto-generating candidates for f={f}, ep={ep_size}")

ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

log(f"model: E={E} d={d} f={f} k={top_k} ep={ep_size} fp8={use_fp8}")
log(f"sweep: tokens={token_candidates} bt={bt_candidates} bf={bf_candidates} btc={btc_candidates} bts={bts_candidates}")

# --- Create weight arrays (shared across token counts) ---
key = jax.random.key(42)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)


def make_sharded(rng_key, shape, dtype, scale=1.0):
    local_shape = (shape[0] // num_devices, *shape[1:])
    per_device_arrays = []
    for i, dev in enumerate(jax.local_devices()):
        shard_key = jax.random.fold_in(rng_key, jax.process_index() * len(jax.local_devices()) + i)
        shard = jax.device_put(
            jax.random.normal(shard_key, local_shape, dtype=dtype) * scale, dev,
        )
        per_device_arrays.append(shard)
    return jax.make_array_from_single_device_arrays(shape, ep_sharding, per_device_arrays)


log("creating weight arrays...")
w1 = make_sharded(k2, (E, d, f), jnp.bfloat16, 0.01)
w2 = make_sharded(k3, (E, f, d), jnp.bfloat16, 0.01)
w3 = make_sharded(k4, (E, d, f), jnp.bfloat16, 0.01)

w1_scale_s = w2_scale_s = w3_scale_s = None
qbk_arg = None
if use_fp8:
    log(f"quantizing weights to fp8 (quant_block_k={quant_block_k})...")

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(("data", "tensor")),),
        out_specs=(P(("data", "tensor")), P(("data", "tensor"))),
        check_vma=False,
    )
    def quantize_shard_map(w):
        local_w = w
        E_loc, K_dim, N_dim = local_w.shape
        w_f32 = local_w.astype(jnp.float32).reshape(E_loc, K_dim // quant_block_k, quant_block_k, N_dim)
        amax = jnp.max(jnp.abs(w_f32), axis=2, keepdims=True)
        scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
        w_q = (w_f32 / scale).astype(jnp.float8_e4m3fn)
        w_q = w_q.reshape(E_loc, K_dim, N_dim)
        return w_q, scale.astype(jnp.float32)

    w1, w1_scale_s = quantize_shard_map(w1)
    w2, w2_scale_s = quantize_shard_map(w2)
    w3, w3_scale_s = quantize_shard_map(w3)
    qbk_arg = quant_block_k
    log("fp8 quantization done")

log("weights ready")

# --- Sweep ---
results: list[tuple[int, str, float, list[float]]] = []

for num_tokens in token_candidates:
    log(f"--- tokens={num_tokens} ---")

    if tune_mode:
        local_nt = num_tokens // ep_size
        padded_local_nt = align_local_tokens_for_v2(local_nt)
        bt_candidates, bf_candidates, btc_candidates, bts_candidates = generate_tune_candidates(f, padded_local_nt, ep_size)
        log(f"  tune: bt={bt_candidates} bf={bf_candidates} (local_nt={local_nt}→{padded_local_nt})")

    tokens = make_sharded(k1, (num_tokens, d), jnp.bfloat16)
    gating_local_shape = (num_tokens // num_devices, E)
    gating_per_dev = []
    for i, dev in enumerate(jax.local_devices()):
        shard_key = jax.random.fold_in(k5, jax.process_index() * len(jax.local_devices()) + i)
        gating_per_dev.append(jax.device_put(
            jax.random.normal(shard_key, gating_local_shape, dtype=jnp.float32), dev,
        ))
    gating = jax.make_array_from_single_device_arrays(
        (num_tokens, E), ep_sharding, gating_per_dev,
    )
    _, topk_idx = lax.top_k(gating, top_k)
    topk_logits = jnp.take_along_axis(gating, topk_idx, axis=-1)
    topk_wts = jax.nn.softmax(topk_logits, axis=-1)

    configs_to_try = list(itertools.product(bt_candidates, bf_candidates, btc_candidates, bts_candidates))
    seen_resolved_configs = set()

    for bt, bf, btc, bts in configs_to_try:
        bc = FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=bse, bts=bts)
        tag = f"bt={bt},bf={bf},btc={btc},bts={bts}"

        padded_nt = num_tokens
        local_nt_raw = num_tokens // ep_size
        aligned_local_nt = align_local_tokens_for_v2(local_nt_raw)
        pad_local = aligned_local_nt - local_nt_raw
        if pad_local > 0:
            padded_nt = (local_nt_raw + pad_local) * ep_size

        try:
            bc_resolved = bc.effective_for(num_tokens=padded_nt, ep_size=ep_size)
        except ValueError as e:
            log(f"  SKIP {tag}: {e}")
            continue

        tag_resolved = f"bt={bc_resolved.bt},bf={bc_resolved.bf},btc={bc_resolved.btc},bts={bc_resolved.bts}"
        resolved_key = (bc_resolved.bt, bc_resolved.bf, bc_resolved.btc, bc_resolved.bts)
        if resolved_key in seen_resolved_configs:
            log(f"  SKIP duplicate resolved config {tag} -> {tag_resolved}")
            continue
        seen_resolved_configs.add(resolved_key)

        def run_fn():
            return fused_ep_moe_v2(
                mesh, tokens, w1, w2, w3,
                topk_wts, topk_idx, top_k,
                block_config=bc,
                quant_block_k=qbk_arg,
                w1_scale=w1_scale_s, w2_scale=w2_scale_s, w3_scale=w3_scale_s,
                disable_a2a=disable_a2a,
                disable_sync_barrier=disable_sync_barrier,
                disable_weight_load=disable_weight_load,
                disable_dynamic_ffn1=disable_dynamic_ffn1,
                disable_dynamic_ffn2=disable_dynamic_ffn2,
                disable_acc_and_store=disable_acc_and_store,
                decode_mode=decode_mode,
                direct_scaled_dot=direct_scaled_dot,
                skip_decode_sync_barrier=skip_decode_sync,
                use_jax_allreduce_metadata=not inkernel_metadata,
            )

        try:
            if use_split:
                dispatch_times, wait_times = split_timeit(run_fn, warmup=warmup, iters=iters)
                if jax.process_index() == 0 and dispatch_times:
                    d_avg = np.mean(dispatch_times)
                    w_avg = np.mean(wait_times)
                    wall_avg = d_avg + w_avg
                    log(f"  {tag_resolved}: wall={wall_avg:.3f}ms = dispatch={d_avg:.3f}ms + wait={w_avg:.3f}ms")
                    log(f"    dispatch: {[round(t, 3) for t in dispatch_times]}")
                    log(f"    wait:     {[round(t, 3) for t in wait_times]}")
                    results.append((num_tokens, tag_resolved, wall_avg, [d + w for d, w in zip(dispatch_times, wait_times)]))
            else:
                times = timeit_fn(run_fn, warmup=warmup, iters=iters)
                if jax.process_index() == 0:
                    if times:
                        avg = np.mean(times)
                        log(f"  {tag_resolved}: {avg:.3f} ms ({timing_label}) | samples={[round(t, 3) for t in times]}")
                        results.append((num_tokens, tag_resolved, avg, times))
                    else:
                        log(f"  {tag_resolved}: no timing data")
        except Exception as e:
            log(f"  FAIL {tag}: {e}")
            continue

# --- Summary ---
if jax.process_index() == 0 and results:
    log("")
    log("=== Summary ===")
    by_tokens: dict[int, list[tuple[str, float, list[float]]]] = {}
    for nt, tag, avg, times in results:
        by_tokens.setdefault(nt, []).append((tag, avg, times))
    for nt in sorted(by_tokens.keys()):
        entries = sorted(by_tokens[nt], key=lambda x: x[1])
        best_tag, best_avg, best_times = entries[0]
        log(f"  tokens={nt}: best={best_avg:.3f}ms [{best_tag}]")
        if len(entries) > 1:
            for tag, avg, _ in entries[1:]:
                log(f"    {avg:.3f}ms [{tag}]")

# --- Correctness check (optional, single-host only) ---
if check_correctness:
    if jax.process_count() > 1:
        log("SKIP correctness check in multi-host mode (use single-host ep=8)")
    else:
        log("computing reference...")
        bc0 = FusedMoEBlockConfig(bt=bt_candidates[0], bf=bf_candidates[0], btc=btc_candidates[0], bse=bse, bts=bts_candidates[0])
        nt0 = token_candidates[0]
        tokens_c = make_sharded(k1, (nt0, d), jnp.bfloat16)
        gating_c_local = (nt0 // num_devices, E)
        gating_c_dev = []
        for i, dev in enumerate(jax.local_devices()):
            sk = jax.random.fold_in(k5, jax.process_index() * len(jax.local_devices()) + i)
            gating_c_dev.append(jax.device_put(
                jax.random.normal(sk, gating_c_local, dtype=jnp.float32), dev,
            ))
        gating_c = jax.make_array_from_single_device_arrays(
            (nt0, E), ep_sharding, gating_c_dev,
        )
        _, tidx = lax.top_k(gating_c, top_k)
        twts = jax.nn.softmax(jnp.take_along_axis(gating_c, tidx, axis=-1), axis=-1)

        result = fused_ep_moe_v2(
            mesh, tokens_c, w1, w2, w3,
            twts, tidx, top_k,
            block_config=bc0,
            quant_block_k=qbk_arg,
            w1_scale=w1_scale_s, w2_scale=w2_scale_s, w3_scale=w3_scale_s,
            direct_scaled_dot=direct_scaled_dot,
            skip_decode_sync_barrier=skip_decode_sync,
            use_jax_allreduce_metadata=not inkernel_metadata,
        )
        ref_kwargs = {}
        if use_fp8:
            ref_kwargs["quant_block_k"] = quant_block_k
            ref_kwargs["w1_scale"] = jax.device_get(w1_scale_s)
            ref_kwargs["w2_scale"] = jax.device_get(w2_scale_s)
            ref_kwargs["w3_scale"] = jax.device_get(w3_scale_s)
        ref = ref_moe(
            jax.device_get(tokens_c), jax.device_get(w1), jax.device_get(w2), jax.device_get(w3),
            jax.device_get(twts), jax.device_get(tidx), top_k,
            **ref_kwargs,
        )
        result_f32 = jax.device_get(
            jax.device_put(result, jax.sharding.NamedSharding(mesh, P()))
        ).astype(np.float32)
        ref_f32 = np.asarray(ref).astype(np.float32)
        max_err = np.max(np.abs(result_f32 - ref_f32))
        rel_err = float(max_err / (np.max(np.abs(ref_f32)) + 1e-6))
        log(f"max_abs_err={max_err:.4f}, rel_err={rel_err:.6f}")
        if rel_err > 0.05:
            log("FAIL: relative error too high")
            sys.exit(1)
        log("PASS")

log("done")
