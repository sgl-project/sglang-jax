"""Wall-time comparison: v1 vs v2 fused_ep_moe with tuned configs.

Both kernels are timed end-to-end (dispatch + metadata allreduce + pallas kernel).
v1's allreduce metadata runs outside the pallas_call but inside the jitted function,
so wall timing captures it fairly for both.

Env vars:
  BENCH_FP8     — 1 for fp8 weights (default: 1)
  BENCH_QBK     — quant_block_k (default: 128)
  BENCH_WARMUP  — warmup iterations (default: 3)
  BENCH_ITERS   — timed iterations (default: 10)
  BENCH_D/F/E/TOPK — model dims (default: MiMo V2 Pro)
"""
from __future__ import annotations

import os
import gzip
import json
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
TRACE_ROOT = "/tmp/tpu_logs/v2_compare_trace"


def log(msg):
    print(f"[{time.time()-t0:.1f}s][p{jax.process_index()}] {msg}", flush=True)


jax.distributed.initialize()
log(f"initialized: {jax.device_count()} devices, {jax.process_index()}")

from sgl_jax.srt.kernels.fused_moe.v2.kernel import fused_ep_moe_v2, FusedMoEBlockConfig
from sgl_jax.srt.kernels.fused_moe.v1 import kernel as v1_kernel
v1_fused_ep_moe = v1_kernel.fused_ep_moe
V1BlockConfig = v1_kernel.FusedMoEBlockConfig

P = jax.sharding.PartitionSpec
num_devices = jax.device_count()
devices = np.array(jax.devices()).reshape(1, num_devices)
mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
ep_size = num_devices

d = int(os.environ.get("BENCH_D", "6144"))
f = int(os.environ.get("BENCH_F", "2048"))
E = int(os.environ.get("BENCH_E", "384"))
top_k = int(os.environ.get("BENCH_TOPK", "8"))
warmup = int(os.environ.get("BENCH_WARMUP", "3"))
iters = int(os.environ.get("BENCH_ITERS", "10"))
use_fp8 = os.environ.get("BENCH_FP8", "1") == "1"
quant_block_k = int(os.environ.get("BENCH_QBK", "128"))
use_trace = os.environ.get("BENCH_TRACE", "0") == "1"
direct_scaled_dot = os.environ.get("BENCH_DIRECT_SCALED_DOT", "0") == "1"
v2_decode_mode = os.environ.get("BENCH_V2_DECODE_MODE", "0") == "1"
v2_bf_override = os.environ.get("BENCH_V2_BF")
v2_bt_override = os.environ.get("BENCH_V2_BT")
v2_btc_override = os.environ.get("BENCH_V2_BTC")
v2_bts_override = os.environ.get("BENCH_V2_BTS")


def parse_csv_int(env_key: str, default: list[int]) -> list[int]:
    v = os.environ.get(env_key)
    if v is None:
        return default
    return [int(x.strip()) for x in v.split(",")]

# Tuned configs per token count
# V1: from tuned_block_configs.py (bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse)
V1_TUNED = {
    64:   (2, 2048, 2048, 2048, 4, 4, 2048, 2048, 2048, 2048),
    128:  (4, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 2048),
    256:  (8, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 2048),
    512:  (16, 2048, 2048, 2048, 16, 16, 2048, 2048, 2048, 2048),
    8192: (128, 1024, 2048, 2048, 64, 64, 1024, 2048, 2048, 1024),
    16384: (128, 1024, 512, 512, 128, 128, 1024, 512, 512, 1024),
}
# V2: from v2 tune sweep at ep=32 fp8.
# Tuple layout accepts either (bt, bf, btc, bse) or (bt, bf, btc, bse, bts).
V2_TUNED = {
    64:   (8, 512, 8, 256, 8),
    128:  (8, 256, 8, 256, 8),
    256:  (8, 512, 8, 256, 8),
    512:  (16, 256, 16, 256, 16),
    8192: (128, 1024, 128, 256),
    16384: (128, 1024, 128, 256),
}

V2_DIRECT_SCALED_DOT_TUNED = {
    64:   (8, 512, 8, 256, 8),
    128:  (8, 512, 8, 256, 8),
    256:  (8, 512, 16, 256, 16),
    512:  (16, 512, 16, 256, 16),
    8192: (128, 1024, 128, 256),
    16384: (128, 512, 80, 256, 160),
}

token_candidates = parse_csv_int("BENCH_TOKENS", list(V1_TUNED.keys()))

ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

log(f"model: E={E} d={d} f={f} k={top_k} ep={ep_size} fp8={use_fp8}")
log(f"tokens={token_candidates} warmup={warmup} iters={iters} timing={'trace' if use_trace else 'wall'}")
log("metadata modes: V1=jax_out_of_kernel V2=in_kernel")
if direct_scaled_dot:
    log("direct_scaled_dot=True")
if v2_decode_mode:
    log("v2_decode_mode=True")


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


def align_local_tokens_for_v2(local_num_tokens: int) -> int:
    if local_num_tokens <= 8:
        return 8
    return ((local_num_tokens + 7) // 8) * 8


def unpack_v2_config(cfg: tuple[int, ...]) -> tuple[int, int, int, int, int | None]:
    if len(cfg) == 4:
        bt, bf, btc, bse = cfg
        bts = None
    elif len(cfg) == 5:
        bt, bf, btc, bse, bts = cfg
    else:
        raise ValueError(f"Unexpected V2 config length: {len(cfg)}")
    return bt, bf, btc, bse, bts


key = jax.random.key(42)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)

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


def wall_timeit(run_fn, warmup_n, iters_n):
    for _ in range(warmup_n):
        out = run_fn()
        jax.block_until_ready(out)
    times = []
    for _ in range(iters_n):
        t_start = time.monotonic()
        out = run_fn()
        jax.block_until_ready(out)
        times.append((time.monotonic() - t_start) * 1e3)
    return times


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


def _extract_pallas_durations_ms(trace: dict[str, Any], kernel_name_re: re.Pattern[str]) -> list[float]:
    matched = [
        e for e in trace.get("traceEvents", [])
        if "name" in e and kernel_name_re.match(e["name"])
    ]
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


def _as_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _device_duration_us(e: dict[str, Any]) -> float | None:
    args = e.get("args", {})
    dur_ps = _as_float(args.get("device_duration_ps"))
    if dur_ps is not None and dur_ps > 0:
        return dur_ps / 1e6
    return None


def _event_device_interval_us(e: dict[str, Any]) -> tuple[float, float] | None:
    ts = _as_float(e.get("ts"))
    dur = _device_duration_us(e)
    if ts is None or dur is None:
        return None
    return ts, ts + dur


def _slowest_per_iteration(by_pid: dict[int, list[float]]) -> list[float]:
    if not by_pid:
        return []
    n = min(len(v) for v in by_pid.values())
    if n == 0:
        return []
    return [max(v[i] for v in by_pid.values()) for i in range(n)]


def _extract_fair_device_spans_ms(
    trace: dict[str, Any],
    kernel_name_re: re.Pattern[str],
    *,
    include_preceding_hlo_categories: tuple[str, ...] = (),
    max_preceding_gap_us: float = 5000.0,
) -> list[float]:
    events = trace.get("traceEvents", [])
    pallas_by_pid: dict[int, list[tuple[float, float]]] = {}
    preceding_by_pid: dict[int, list[tuple[float, float]]] = {}

    for e in events:
        pid = e.get("pid")
        if not isinstance(pid, int):
            continue
        interval = _event_device_interval_us(e)
        if interval is None:
            continue
        name = e.get("name", "")
        if isinstance(name, str) and kernel_name_re.match(name):
            pallas_by_pid.setdefault(pid, []).append(interval)
            continue
        hlo_category = e.get("args", {}).get("hlo_category")
        if hlo_category in include_preceding_hlo_categories:
            preceding_by_pid.setdefault(pid, []).append(interval)

    if not pallas_by_pid:
        return []
    for evts in pallas_by_pid.values():
        evts.sort()
    for evts in preceding_by_pid.values():
        evts.sort()

    spans_by_pid: dict[int, list[float]] = {}
    for pid, pallas_events in pallas_by_pid.items():
        preceding_events = preceding_by_pid.get(pid, [])
        pid_spans: list[float] = []
        for p_start, p_end in pallas_events:
            span_start = p_start
            if preceding_events:
                candidates = [
                    (m_start, m_end)
                    for m_start, m_end in preceding_events
                    if m_end <= p_start and (p_start - m_end) <= max_preceding_gap_us
                ]
                if candidates:
                    span_start = max(candidates, key=lambda x: x[1])[0]
            pid_spans.append((p_end - span_start) / 1e3)
        spans_by_pid[pid] = pid_spans

    return _slowest_per_iteration(spans_by_pid)


def trace_timeit(
    run_fn,
    warmup_n,
    iters_n,
    kernel_name_re,
    step_name,
    *,
    include_preceding_hlo_categories: tuple[str, ...] = (),
):
    for _ in range(warmup_n):
        out = run_fn()
        jax.block_until_ready(out)

    tag = f"{os.getpid()}_{int(time.time())}"
    trace_dir = os.path.join(TRACE_ROOT, f"run_{tag}")
    os.makedirs(trace_dir, exist_ok=True)

    with jax.profiler.trace(trace_dir):
        for i in range(iters_n):
            with jax.profiler.StepTraceAnnotation(step_name, step_num=i):
                out = run_fn()
                jax.block_until_ready(out)

    if jax.process_index() != 0:
        return {"pallas": [], "fair": []}
    trace = _load_trace(trace_dir)
    return {
        "pallas": _extract_pallas_durations_ms(trace, kernel_name_re),
        "fair": _extract_fair_device_spans_ms(
            trace,
            kernel_name_re,
            include_preceding_hlo_categories=include_preceding_hlo_categories,
        ),
    }


results = {}

for num_tokens in token_candidates:
    log(f"=== tokens={num_tokens} ===")

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

    # --- V1 (tuned config) ---
    bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse = V1_TUNED[num_tokens]
    v1_bc = V1BlockConfig(bt=bt, bf=bf, bd1=bd1, bd2=bd2, btc=btc, bfc=bfc,
                          bd1c=bd1c, bd2c=bd2c, bse=bse, bts=bts)
    v1_bc_eff = v1_bc.effective_for(num_tokens=num_tokens, ep_size=ep_size,
                                     dtype=jnp.bfloat16, quant_block_k=qbk_arg)
    log(f"  V1 tuned: bt={v1_bc_eff.bt},bf={v1_bc_eff.bf},bd1={v1_bc_eff.bd1},"
        f"bts={v1_bc_eff.bts},btc={v1_bc_eff.btc},bse={v1_bc_eff.bse}")

    def run_v1(bc=v1_bc):
        return v1_fused_ep_moe(
            mesh, tokens, w1, w2, w3,
            topk_wts, topk_idx, top_k,
            block_config=bc,
            quant_block_k=qbk_arg,
            w1_scale=w1_scale_s, w2_scale=w2_scale_s, w3_scale=w3_scale_s,
            disable_all_reduce_metadata=False,
            use_jax_allreduce_metadata=True,
        )

    log("  V1: compiling + running...")
    if use_trace:
        v1_trace = trace_timeit(
            run_v1, warmup, iters, re.compile(r"fused-moe-k_.*"),
            f"v1_call_tokens_{num_tokens}",
            include_preceding_hlo_categories=("all-gather",),
        )
        v1_times = v1_trace["fair"]
        v1_pallas_times = v1_trace["pallas"]
    else:
        v1_times = wall_timeit(run_v1, warmup, iters)
        v1_pallas_times = []

    # --- V2 (tuned config) ---
    v2_tuned_table = V2_DIRECT_SCALED_DOT_TUNED if direct_scaled_dot else V2_TUNED
    bt2, bf2, btc2, bse2, bts2 = unpack_v2_config(v2_tuned_table[num_tokens])
    if v2_bt_override is not None:
        bt2 = int(v2_bt_override)
    if v2_bf_override is not None:
        bf2 = int(v2_bf_override)
    if v2_btc_override is not None:
        btc2 = int(v2_btc_override)
    if v2_bts_override is not None:
        bts2 = int(v2_bts_override)
    v2_bc = FusedMoEBlockConfig(bt=bt2, bf=bf2, btc=btc2, bse=bse2, bts=bts2)
    local_nt_raw = num_tokens // ep_size
    aligned_local_nt = align_local_tokens_for_v2(local_nt_raw)
    pad_local = aligned_local_nt - local_nt_raw
    padded_nt = (local_nt_raw + pad_local) * ep_size if pad_local > 0 else num_tokens
    v2_bc_eff = v2_bc.effective_for(num_tokens=padded_nt, ep_size=ep_size)
    log(f"  V2 tuned: bt={v2_bc_eff.bt},bf={v2_bc_eff.bf},"
        f"btc={v2_bc_eff.btc},bts={v2_bc_eff.bts},bse={v2_bc_eff.bse}"
        f"{' (padded ' + str(num_tokens) + '->' + str(padded_nt) + ')' if pad_local > 0 else ''}")

    def run_v2(bc=v2_bc):
        return fused_ep_moe_v2(
            mesh, tokens, w1, w2, w3,
            topk_wts, topk_idx, top_k,
            block_config=bc,
            quant_block_k=qbk_arg,
            w1_scale=w1_scale_s, w2_scale=w2_scale_s, w3_scale=w3_scale_s,
            use_jax_allreduce_metadata=False,
            direct_scaled_dot=direct_scaled_dot,
        )

    log("  V2: compiling + running...")
    if use_trace:
        v2_trace = trace_timeit(
            run_v2, warmup, iters, re.compile(r"fused-moe-v2-k_.*"),
            f"v2_call_tokens_{num_tokens}",
        )
        v2_times = v2_trace["fair"]
        v2_pallas_times = v2_trace["pallas"]
    else:
        v2_times = wall_timeit(run_v2, warmup, iters)
        v2_pallas_times = []

    if jax.process_index() == 0:
        if not v1_times or not v2_times:
            log(f"  missing timing data: v1={v1_times}, v2={v2_times}")
            continue
        v1_avg = np.mean(v1_times)
        v2_avg = np.mean(v2_times)
        label = "device-fair" if use_trace else "wall"
        log(f"  V1 {label}: {v1_avg:.3f} ms | samples={[round(t, 3) for t in v1_times]}")
        log(f"  V2 {label}: {v2_avg:.3f} ms | samples={[round(t, 3) for t in v2_times]}")
        if use_trace and v1_pallas_times and v2_pallas_times:
            v1_pallas_avg = np.mean(v1_pallas_times)
            v2_pallas_avg = np.mean(v2_pallas_times)
            log(
                f"  pallas-only diagnostic: V1={v1_pallas_avg:.3f} ms "
                f"V2={v2_pallas_avg:.3f} ms"
            )
        diff = v2_avg - v1_avg
        pct = (v2_avg / v1_avg - 1) * 100
        log(f"  V2 vs V1: {diff:+.3f} ms ({pct:+.1f}%)")
        results[num_tokens] = (v1_avg, v2_avg)

if jax.process_index() == 0:
    log("")
    log("=== Summary ===")
    for nt in sorted(results.keys()):
        v1_ms, v2_ms = results[nt]
        diff = v2_ms - v1_ms
        pct = (v2_ms / v1_ms - 1) * 100
        log(f"  tokens={nt}: V1={v1_ms:.3f}ms  V2={v2_ms:.3f}ms  delta={diff:+.3f}ms ({pct:+.1f}%)")

log("done")
