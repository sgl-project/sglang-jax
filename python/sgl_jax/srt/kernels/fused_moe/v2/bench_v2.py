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
  BENCH_DIRECT_SCALED_DOT — 1 to use direct-scaled-dot for both FFN1/FFN2
  BENCH_DIRECT_SCALED_DOT_FFN1/FFN2 — optional comma-separated 0/1 hybrid sweep
  BENCH_FFN1_DEQUANT_MODE — full or fchunk when FFN1 direct-scaled-dot is off
  BENCH_FFN1_DEQUANT_CHUNK — comma-separated FFN1 dequant chunk sizes for fchunk
  BENCH_W2_FETCH_ORDER — after_w13 or before_w13 for current-expert W2 DMA
  BENCH_W2_FETCH_PRIORITY — comma-separated 0/1 priority for current-expert W2 DMA
  BENCH_SKIP_INTER_BT_SYNC — comma-separated 0/1 skip inter-BT sync barrier
  BENCH_INTERLEAVE_BT — comma-separated 0/1 interleave BT gather banking
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
import math
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


def parse_csv_str(env_key: str, default: list[str]) -> list[str]:
    v = os.environ.get(env_key)
    if v is None:
        return default
    return [x.strip() for x in v.split(",")]


def parse_csv_int_or_none(env_key: str) -> list[int | None]:
    v = os.environ.get(env_key)
    if v is None:
        return [None]
    return [int(x.strip()) for x in v.split(",")]


def parse_csv_bool(env_key: str, default: list[bool]) -> list[bool]:
    v = os.environ.get(env_key)
    if v is None:
        return default
    out = []
    for raw in v.split(","):
        item = raw.strip().lower()
        if item in ("1", "true", "t", "yes", "y"):
            out.append(True)
        elif item in ("0", "false", "f", "no", "n"):
            out.append(False)
        else:
            raise ValueError(f"Unsupported boolean value {raw!r} for {env_key}")
    return out


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
routing_mode = os.environ.get("BENCH_ROUTING_MODE", "random")
bse = int(os.environ.get("BENCH_BSE", "256"))
warmup = int(os.environ.get("BENCH_WARMUP", "2"))
iters = int(os.environ.get("BENCH_ITERS", "5"))
check_correctness = os.environ.get("BENCH_CHECK", "0") == "1"
use_fp8 = os.environ.get("BENCH_FP8", "0") == "1"
_qbk_str = os.environ.get("BENCH_QBK", "128")
quant_block_k = None if _qbk_str.lower() == "none" else int(_qbk_str)
tune_mode = os.environ.get("BENCH_TUNE", "0") == "1"
use_wall = os.environ.get("BENCH_WALL", "0") == "1"
use_split = os.environ.get("BENCH_SPLIT", "0") == "1"
direct_scaled_dot = os.environ.get("BENCH_DIRECT_SCALED_DOT", "0") == "1"
direct_scaled_dot_ffn1_modes = parse_csv_bool(
    "BENCH_DIRECT_SCALED_DOT_FFN1", [direct_scaled_dot],
)
direct_scaled_dot_ffn2_modes = parse_csv_bool(
    "BENCH_DIRECT_SCALED_DOT_FFN2", [direct_scaled_dot],
)
cast_ffn1_input_fp8 = os.environ.get("BENCH_CAST_FFN1_INPUT_FP8", "0") == "1"
cast_ffn2_input_fp8 = os.environ.get("BENCH_CAST_FFN2_INPUT_FP8", "0") == "1"
enable_act_quant = os.environ.get("BENCH_ACT_QUANT", "0") == "1"
ffn1_dequant_modes = parse_csv_str("BENCH_FFN1_DEQUANT_MODE", ["full"])
ffn1_dequant_chunks = parse_csv_int_or_none("BENCH_FFN1_DEQUANT_CHUNK")
inkernel_metadata = os.environ.get("BENCH_INKERNEL_MD", "1") == "1"
enable_bt_scatter_overlap = os.environ.get("BENCH_BT_SCATTER_OVERLAP", "1") == "1"
cross_expert_prefetch_modes = parse_csv_str("BENCH_CROSS_EXPERT_PREFETCH", ["full"])
next_w2_prologue_priorities = parse_csv_int("BENCH_NEXT_W2_PRIORITY", [1])
w2_fetch_orders = parse_csv_str("BENCH_W2_FETCH_ORDER", ["after_w13"])
w2_fetch_priorities = parse_csv_int("BENCH_W2_FETCH_PRIORITY", [1])
skip_inter_bt_sync_modes = parse_csv_bool(
    "BENCH_SKIP_INTER_BT_SYNC", [True],
)
interleave_bt_modes = parse_csv_bool(
    "BENCH_INTERLEAVE_BT", [True],
)
valid_ffn1_dequant_modes = {"full", "fchunk"}
invalid_ffn1_dequant_modes = [
    mode for mode in ffn1_dequant_modes if mode not in valid_ffn1_dequant_modes
]
if invalid_ffn1_dequant_modes:
    raise ValueError(
        f"Unsupported BENCH_FFN1_DEQUANT_MODE values {invalid_ffn1_dequant_modes}; "
        "expected one of full or fchunk."
    )
valid_cross_expert_prefetch_modes = {"none", "full", "w13"}
invalid_modes = [
    mode for mode in cross_expert_prefetch_modes
    if mode not in valid_cross_expert_prefetch_modes
]
if invalid_modes:
    raise ValueError(
        f"Unsupported BENCH_CROSS_EXPERT_PREFETCH values {invalid_modes}; "
        "expected one of none, full, or w13."
    )
invalid_priorities = [
    priority
    for priority in (
        list(next_w2_prologue_priorities)
        + list(w2_fetch_priorities)
    )
    if priority not in (0, 1)
]
if invalid_priorities:
    raise ValueError(
        f"Unsupported DMA priority values {invalid_priorities}; "
        "TPU DMA priority supports only 0 or 1."
    )
valid_w2_fetch_orders = {"after_w13", "before_w13"}
invalid_w2_fetch_orders = [
    mode for mode in w2_fetch_orders if mode not in valid_w2_fetch_orders
]
if invalid_w2_fetch_orders:
    raise ValueError(
        f"Unsupported BENCH_W2_FETCH_ORDER values {invalid_w2_fetch_orders}; "
        "expected one of after_w13 or before_w13."
    )
valid_routing_modes = {"random", "deterministic", "hot_expert"}
if routing_mode not in valid_routing_modes:
    raise ValueError(
        f"Unsupported BENCH_ROUTING_MODE={routing_mode!r}; "
        "expected one of random or deterministic."
    )
if use_split:
    timeit_fn = None
    timing_label = "split"
else:
    timeit_fn = wall_timeit if use_wall else trace_timeit
    timing_label = "wall" if use_wall else "trace"

# Ablation flags
all_disable = os.environ.get("FUSED_MOE_BENCHMARK_ALL_DISABLE", "0") == "1"
disable_a2a = all_disable or os.environ.get("DISABLE_A2A", "0") == "1"
disable_a2a_scatter = all_disable or os.environ.get("DISABLE_A2A_SCATTER", "0") == "1"
disable_a2a_scatter_local_copy = (
    all_disable or os.environ.get("DISABLE_A2A_SCATTER_LOCAL_COPY", "0") == "1"
)
disable_a2a_scatter_remote_copy = (
    all_disable or os.environ.get("DISABLE_A2A_SCATTER_REMOTE_COPY", "0") == "1"
)
disable_a2a_scatter_recv_wait = (
    all_disable or os.environ.get("DISABLE_A2A_SCATTER_RECV_WAIT", "0") == "1"
)
disable_a2a_scatter_send_wait = (
    all_disable or os.environ.get("DISABLE_A2A_SCATTER_SEND_WAIT", "0") == "1"
)
disable_a2a_gather = all_disable or os.environ.get("DISABLE_A2A_GATHER", "0") == "1"
disable_a2a_gather_local_copy = (
    all_disable or os.environ.get("DISABLE_A2A_GATHER_LOCAL_COPY", "0") == "1"
)
disable_a2a_gather_remote_copy = (
    all_disable or os.environ.get("DISABLE_A2A_GATHER_REMOTE_COPY", "0") == "1"
)
disable_sync_barrier = all_disable or os.environ.get("DISABLE_SYNC_BARRIER", "0") == "1"
disable_weight_load = all_disable or os.environ.get("DISABLE_WEIGHT_LOAD", "0") == "1"
disable_w1_load = all_disable or os.environ.get("DISABLE_W1_LOAD", "0") == "1"
disable_w3_load = all_disable or os.environ.get("DISABLE_W3_LOAD", "0") == "1"
disable_w2_load = all_disable or os.environ.get("DISABLE_W2_LOAD", "0") == "1"
disable_expert_x_load = all_disable or os.environ.get("DISABLE_EXPERT_X_LOAD", "0") == "1"
disable_expert_ffn = all_disable or os.environ.get("DISABLE_EXPERT_FFN", "0") == "1"
disable_dynamic_ffn1 = all_disable or os.environ.get("DISABLE_DYNAMIC_FFN1", "0") == "1"
disable_dynamic_ffn2 = all_disable or os.environ.get("DISABLE_DYNAMIC_FFN2", "0") == "1"
disable_expert_store = all_disable or os.environ.get("DISABLE_EXPERT_STORE", "0") == "1"
disable_expert_stage_writeback = (
    all_disable or os.environ.get("DISABLE_EXPERT_STAGE_WRITEBACK", "0") == "1"
)
disable_expert_store_dma = (
    all_disable or os.environ.get("DISABLE_EXPERT_STORE_DMA", "0") == "1"
)
disable_expert_store_wait = (
    all_disable or os.environ.get("DISABLE_EXPERT_STORE_WAIT", "0") == "1"
)
disable_acc_and_store = all_disable or os.environ.get("DISABLE_ACC_AND_STORE", "0") == "1"
disable_acc_load = all_disable or os.environ.get("DISABLE_ACC_LOAD", "0") == "1"
disable_acc_compute = all_disable or os.environ.get("DISABLE_ACC_COMPUTE", "0") == "1"
disable_acc_store_vmem = all_disable or os.environ.get("DISABLE_ACC_STORE_VMEM", "0") == "1"
disable_output_store = all_disable or os.environ.get("DISABLE_OUTPUT_STORE", "0") == "1"
ablation_flags = {
    "disable_a2a": disable_a2a,
    "disable_a2a_scatter": disable_a2a_scatter,
    "disable_a2a_scatter_local_copy": disable_a2a_scatter_local_copy,
    "disable_a2a_scatter_remote_copy": disable_a2a_scatter_remote_copy,
    "disable_a2a_scatter_recv_wait": disable_a2a_scatter_recv_wait,
    "disable_a2a_scatter_send_wait": disable_a2a_scatter_send_wait,
    "disable_a2a_gather": disable_a2a_gather,
    "disable_a2a_gather_local_copy": disable_a2a_gather_local_copy,
    "disable_a2a_gather_remote_copy": disable_a2a_gather_remote_copy,
    "disable_sync_barrier": disable_sync_barrier,
    "disable_weight_load": disable_weight_load,
    "disable_w1_load": disable_w1_load,
    "disable_w3_load": disable_w3_load,
    "disable_w2_load": disable_w2_load,
    "disable_expert_x_load": disable_expert_x_load,
    "disable_expert_ffn": disable_expert_ffn,
    "disable_dynamic_ffn1": disable_dynamic_ffn1,
    "disable_dynamic_ffn2": disable_dynamic_ffn2,
    "disable_expert_store": disable_expert_store,
    "disable_expert_stage_writeback": disable_expert_stage_writeback,
    "disable_expert_store_dma": disable_expert_store_dma,
    "disable_expert_store_wait": disable_expert_store_wait,
    "disable_acc_and_store": disable_acc_and_store,
    "disable_acc_load": disable_acc_load,
    "disable_acc_compute": disable_acc_compute,
    "disable_acc_store_vmem": disable_acc_store_vmem,
    "disable_output_store": disable_output_store,
}
active_ablation = [k for k, v in ablation_flags.items() if v]
if active_ablation:
    log(f"ablation flags: {active_ablation}")
if direct_scaled_dot:
    log("direct_scaled_dot=True (fp8 dot per quant group, scale after dot)")
if cast_ffn1_input_fp8 or cast_ffn2_input_fp8:
    log(
        "input cast controls: "
        f"ffn1_fp8={cast_ffn1_input_fp8} ffn2_fp8={cast_ffn2_input_fp8}"
    )
if (
    direct_scaled_dot_ffn1_modes != [direct_scaled_dot]
    or direct_scaled_dot_ffn2_modes != [direct_scaled_dot]
):
    log(
        "direct_scaled_dot hybrid sweep: "
        f"ffn1={direct_scaled_dot_ffn1_modes} ffn2={direct_scaled_dot_ffn2_modes}"
    )
if ffn1_dequant_modes != ["full"] or ffn1_dequant_chunks != [None]:
    log(
        "ffn1_dequant sweep: "
        f"mode={ffn1_dequant_modes} chunk={ffn1_dequant_chunks}"
    )
if inkernel_metadata:
    log("inkernel_metadata=True (in-kernel ICI allgather, no JAX lax.all_gather)")
if enable_bt_scatter_overlap:
    log("bt_scatter_overlap=True (next-BT scatter HBM bank overlap)")
log(
    "cross_expert_prefetch="
    f"{cross_expert_prefetch_modes} next_w2_priority={next_w2_prologue_priorities}"
)
if w2_fetch_orders != ["after_w13"] or w2_fetch_priorities != [1]:
    log(
        "w2_fetch sweep: "
        f"order={w2_fetch_orders} priority={w2_fetch_priorities}"
    )
if skip_inter_bt_sync_modes != [True]:
    log(f"skip_inter_bt_sync sweep: {skip_inter_bt_sync_modes}")
if interleave_bt_modes != [True]:
    log(f"interleave_bt sweep: {interleave_bt_modes}")

bt_candidates = parse_csv_int("BENCH_BT", [128])
bf_candidates = parse_csv_int("BENCH_BF", [256])
btc_candidates = parse_csv_int("BENCH_BTC", [128])
bts_candidates = parse_csv_int_or_none("BENCH_BTS")
token_candidates = parse_csv_int("BENCH_TOKENS", [4096])


def _align_to(x, a):
    return ((x + a - 1) // a) * a


def _pow2_floor(x):
    if x <= 1:
        return 1
    return 1 << int(math.floor(math.log2(x)))


def _pow2_ceil(x):
    if x <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(x)))


def _ladder_div2(start):
    out = []
    v = int(start)
    while v > 0:
        out.append(v)
        if v == 1:
            break
        v //= 2
    return sorted(set(out), reverse=True)


def _aligned_divisors(n, alignment=8):
    """All divisors of n that are multiples of alignment, descending."""
    if n <= 0:
        return []
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            if i % alignment == 0:
                divs.add(i)
            j = n // i
            if j % alignment == 0:
                divs.add(j)
    return sorted(divs, reverse=True)


def _estimate_vmem_bytes_v2(
    *,
    bt,
    bf,
    btc,
    bse,
    bts,
    hidden_size,
    intermediate_size,
    num_experts,
    top_k,
    ep_size,
    num_tokens,
    use_fp8=False,
    quant_block_k=128,
    direct_scaled_dot=True,
    interleave_bt=True,
    enable_bt_scatter_overlap=True,
    verbose=False,
):
    local_num_tokens = num_tokens // ep_size
    t_packing = 1  # bf16 activations
    w_bytes = 1 if use_fp8 else 2
    token_bytes = 2  # bf16
    h_per_t = hidden_size // t_packing
    padded_num_experts = _align_to(num_experts, 128)
    padded_top_k = _align_to(top_k, 128)
    acc_bt = math.gcd(bt, 16)
    num_bt = local_num_tokens // bt if bt > 0 else 1
    use_bt_scatter_bank = enable_bt_scatter_overlap and num_bt > 1
    use_gather_bank = interleave_bt and num_bt > 1
    smem_banks = num_bt if use_gather_bank else 2

    # Gather accumulation: (2, top_k, acc_bt, t_packing, h_per_t)
    b_a2a_g_acc = 2 * top_k * acc_bt * hidden_size * token_bytes
    # TopK weights: (smem_banks, bt, padded_top_k) f32
    b_topk_w = smem_banks * bt * padded_top_k * 4
    # TopK ids: (smem_banks, bt, padded_top_k) i32
    b_topk_id = smem_banks * bt * padded_top_k * 4
    # Output: (smem_banks, bt, hidden_size) t_dtype
    b_output = smem_banks * bt * hidden_size * token_bytes

    # Weight double buffers: (2, t_packing, h_per_t, bf) or (2, t_packing, bf, h_per_t)
    b_w1 = 2 * hidden_size * bf * w_bytes
    b_w3 = 2 * hidden_size * bf * w_bytes
    b_w2 = 2 * bf * hidden_size * w_bytes

    # Scale buffers (fp8 only)
    b_w1_scale = 0
    b_w3_scale = 0
    b_w2_scale = 0
    if use_fp8:
        _n_sg = 1 if quant_block_k is None else h_per_t // quant_block_k
        _n_sg2 = 1 if quant_block_k is None else bf // quant_block_k
        b_w1_scale = 2 * t_packing * _n_sg * bf * 4
        b_w3_scale = b_w1_scale
        b_w2_scale = 2 * t_packing * _n_sg2 * h_per_t * 4

    # Dequant scratch (fp8 + not direct_scaled_dot)
    b_w1_dq = 0
    b_w3_dq = 0
    b_w2_dq = 0
    if use_fp8 and not direct_scaled_dot:
        b_w1_dq = t_packing * h_per_t * bf * 2  # bf16
        b_w3_dq = b_w1_dq
        b_w2_dq = t_packing * bf * h_per_t * 2  # bf16

    # Gate/up accumulators: (bts, bf) f32 each
    b_gate_acc = bts * bf * 4
    b_up_acc = bts * bf * 4
    # Token staging: (bts, t_packing, h_per_t) t_dtype
    b_x = bts * hidden_size * token_bytes
    # Output accumulator: (bts, t_packing, h_per_t) f32
    b_y_acc = bts * hidden_size * 4
    # Output staging: (bts, t_packing, h_per_t) t_dtype
    b_y_stage = bts * hidden_size * token_bytes

    # Scoped metadata temporaries (run_scoped, only one path active)
    local_num_experts = num_experts // ep_size
    b_scoped = (
        bt * padded_top_k * 4          # t2e_routing
        + ep_size * padded_num_experts * 4  # d2e_count
        + 2 * padded_num_experts * 4    # expert_offsets
        + padded_num_experts * 4        # expert_starts
        + padded_num_experts * 4        # expert_sizes
    )

    # Semaphore overhead (conservative flat estimate)
    num_bt_banks = num_bt if use_gather_bank else (2 if use_bt_scatter_bank else 1)
    b_sems = (
        2 * 4           # x_stage + y_store (1 each)
        + smem_banks * 10 * 4  # local_sems
        + 3 * (num_bt_banks * local_num_experts * 4 if (use_bt_scatter_bank or use_gather_bank) else local_num_experts * 4)
        + (num_bt_banks * 4 if use_gather_bank else 4)  # a2a_gather
        + 3 * 4         # a2a_acc + md_send + md_recv + barrier
    )

    # SMEM (not VMEM, but allocated alongside — counts toward compiler budget)
    b_smem = (
        smem_banks * bt * padded_top_k * 4     # t2e_routing smem
        + smem_banks * ep_size * padded_num_experts * 4  # d2e_count smem
        + smem_banks * 2 * padded_num_experts * 4  # expert_offsets smem
        + smem_banks * padded_num_experts * 4  # expert_starts smem
        + smem_banks * padded_num_experts * 4  # expert_sizes smem
    )

    total = (
        b_a2a_g_acc + b_topk_w + b_topk_id + b_output
        + b_w1 + b_w3 + b_w2
        + b_w1_scale + b_w3_scale + b_w2_scale
        + b_w1_dq + b_w3_dq + b_w2_dq
        + b_gate_acc + b_up_acc + b_x + b_y_acc + b_y_stage
        + b_scoped + b_sems
    )

    if verbose:
        mb = lambda b: f"{b / (1024*1024):.2f}"
        log(f"    VMEM Breakdown (bt={bt} bf={bf} btc={btc} bts={bts}):")
        log(f"      a2a_g_acc:      {mb(b_a2a_g_acc)} MB  (2,{top_k},{acc_bt},{hidden_size})")
        log(f"      topk_weights:   {mb(b_topk_w)} MB  ({smem_banks},{bt},{padded_top_k}) f32")
        log(f"      topk_ids:       {mb(b_topk_id)} MB")
        log(f"      output:         {mb(b_output)} MB  ({smem_banks},{bt},{hidden_size})")
        log(f"      W1 x2:          {mb(b_w1)} MB  (2,{hidden_size},{bf})")
        log(f"      W3 x2:          {mb(b_w3)} MB")
        log(f"      W2 x2:          {mb(b_w2)} MB")
        if use_fp8:
            log(f"      W1/W3 scale:    {mb(b_w1_scale)} MB each")
            log(f"      W2 scale:       {mb(b_w2_scale)} MB")
        if b_w1_dq:
            log(f"      W1/W3 dequant:  {mb(b_w1_dq)} MB each")
            log(f"      W2 dequant:     {mb(b_w2_dq)} MB")
        log(f"      gate+up acc:    {mb(b_gate_acc + b_up_acc)} MB  ({bts},{bf}) f32")
        log(f"      x+y_acc+y_stg:  {mb(b_x + b_y_acc + b_y_stage)} MB")
        log(f"      scoped+sems:    {mb(b_scoped + b_sems)} MB")
        log(f"      Total:          {mb(total)} MB")

    return total


def generate_tune_candidates(
    intermediate_size,
    hidden_size,
    local_num_tokens,
    ep_size,
    num_experts,
    top_k,
    *,
    use_fp8=False,
    quant_block_k=128,
    direct_scaled_dot=True,
    interleave_bt=True,
    enable_bt_scatter_overlap=True,
    vmem_budget=64 * 1024 * 1024,
    vmem_headroom=0.95,
    max_configs=48,
    bse=256,
    verbose=False,
):
    effective_budget = int(vmem_budget * vmem_headroom)

    bf_list = sorted(set(
        v for v in [128, 256, 512, 1024, 2048]
        if v <= intermediate_size and intermediate_size % v == 0
    ))

    bt_list = []
    for p_val in [2, 4]:
        if local_num_tokens == p_val:
            bt_list.append(p_val)
    p = 8
    while p <= local_num_tokens:
        if local_num_tokens % p == 0:
            bt_list.append(p)
        p *= 2
    if not bt_list:
        bt_list = [local_num_tokens]
    bt_list = sorted(set(bt_list))

    configs = []
    seen = set()
    first_verbose = True

    for bt in bt_list:
        max_bts = bt * ep_size
        expected = bt * ep_size * top_k / num_experts
        lo = _pow2_floor(expected)
        hi = _pow2_ceil(expected)
        # Non-power-of-2 candidates near expected, aligned to 8
        exp_floor8 = (int(expected) // 8) * 8
        exp_ceil8 = _align_to(int(math.ceil(expected)), 8)
        # 1.25x expected covers routing imbalance
        exp_hi8 = _align_to(int(math.ceil(expected * 1.25)), 8)
        bts_cands = sorted({
            v for v in [bt, lo, hi, hi * 2, exp_floor8, exp_ceil8, exp_hi8]
            if 0 < v <= max_bts and v % 8 == 0
        })
        if not bts_cands:
            bts_cands = [bt]

        for bts_val in bts_cands:
            btc_cands = _aligned_divisors(bts_val, 8)
            if not btc_cands:
                continue

            for bf in bf_list:
                for btc in btc_cands:
                    bc = FusedMoEBlockConfig(
                        bt=bt, bf=bf, btc=btc, bse=bse, bts=bts_val,
                    )
                    num_tokens_total = local_num_tokens * ep_size
                    try:
                        bc_eff = bc.effective_for(
                            num_tokens=num_tokens_total, ep_size=ep_size,
                        )
                    except ValueError:
                        continue

                    key = (bc_eff.bt, bc_eff.bf, bc_eff.btc, bc_eff.bts)
                    if key in seen:
                        continue
                    seen.add(key)

                    est = _estimate_vmem_bytes_v2(
                        bt=bc_eff.bt, bf=bc_eff.bf, btc=bc_eff.btc,
                        bse=bc_eff.bse, bts=bc_eff.bts,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        num_experts=num_experts,
                        top_k=top_k, ep_size=ep_size,
                        num_tokens=num_tokens_total,
                        use_fp8=use_fp8,
                        quant_block_k=quant_block_k,
                        direct_scaled_dot=direct_scaled_dot,
                        interleave_bt=interleave_bt,
                        enable_bt_scatter_overlap=enable_bt_scatter_overlap,
                        verbose=verbose and first_verbose,
                    )
                    first_verbose = False
                    if est > effective_budget:
                        log(
                            f"  VMEM skip bt={bc_eff.bt},bf={bc_eff.bf},"
                            f"btc={bc_eff.btc},bts={bc_eff.bts}: "
                            f"{est/(1024*1024):.1f}MB > "
                            f"{effective_budget/(1024*1024):.1f}MB"
                        )
                        continue
                    configs.append(bc)

    if len(configs) <= max_configs:
        log(f"  tune: {len(configs)} configs (all pass VMEM filter)")
        return configs

    buckets = {}
    for cfg in configs:
        bk = (cfg.bt, cfg.bts or cfg.bt)
        buckets.setdefault(bk, []).append(cfg)
    for bk in buckets:
        buckets[bk].sort(key=lambda c: (c.bf, c.btc), reverse=True)

    selected = []
    selected_keys = set()
    bucket_keys = sorted(buckets.keys(), reverse=True)
    while len(selected) < max_configs:
        made_progress = False
        for bk in bucket_keys:
            bucket = buckets[bk]
            if not bucket:
                continue
            cfg = bucket.pop(0)
            key = (cfg.bt, cfg.bf, cfg.btc, cfg.bts)
            if key not in selected_keys:
                selected_keys.add(key)
                selected.append(cfg)
                made_progress = True
            if len(selected) >= max_configs:
                break
        if not made_progress:
            break

    log(
        f"  tune: {len(configs)} valid -> {len(selected)} selected "
        f"(max={max_configs}, {len(bucket_keys)} bt/bts buckets)"
    )
    return selected


if tune_mode:
    log(f"tune mode: auto-generating candidates for f={f}, ep={ep_size}")

ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

log(f"model: E={E} d={d} f={f} k={top_k} ep={ep_size} fp8={use_fp8}")
if routing_mode != "random":
    log(f"routing_mode={routing_mode}")
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


def make_deterministic_topk(num_tokens, top_k, num_experts):
    local_tokens = num_tokens // num_devices
    per_device_ids = []
    per_device_weights = []
    for i, dev in enumerate(jax.local_devices()):
        global_device_id = jax.process_index() * len(jax.local_devices()) + i
        token_start = global_device_id * local_tokens
        token_ids = jnp.arange(token_start, token_start + local_tokens, dtype=jnp.int32)
        k_offsets = jnp.arange(top_k, dtype=jnp.int32)
        ids = (token_ids[:, None] + k_offsets[None, :]) % jnp.int32(num_experts)
        weights = jnp.full((local_tokens, top_k), 1.0 / top_k, dtype=jnp.float32)
        per_device_ids.append(jax.device_put(ids, dev))
        per_device_weights.append(jax.device_put(weights, dev))
    topk_ids = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k), ep_sharding, per_device_ids,
    )
    topk_weights = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k), ep_sharding, per_device_weights,
    )
    return topk_weights, topk_ids


def make_hot_expert_topk(num_tokens, top_k, num_experts, hot_frac=0.1, hot_load=0.7):
    """Hot-expert routing: hot_load fraction of (token, k) slots route to
    hot_frac fraction of experts; rest uniformly to the cold tail."""
    local_tokens = num_tokens // num_devices
    num_hot = max(1, int(num_experts * hot_frac))
    per_device_ids = []
    per_device_weights = []
    rng = np.random.default_rng(42)
    for i, dev in enumerate(jax.local_devices()):
        is_hot = rng.random((local_tokens, top_k)) < hot_load
        hot_ids = rng.integers(0, num_hot, size=(local_tokens, top_k), dtype=np.int32)
        cold_ids = rng.integers(num_hot, num_experts, size=(local_tokens, top_k), dtype=np.int32)
        ids_np = np.where(is_hot, hot_ids, cold_ids).astype(np.int32)
        ids = jnp.array(ids_np)
        weights = jnp.full((local_tokens, top_k), 1.0 / top_k, dtype=jnp.float32)
        per_device_ids.append(jax.device_put(ids, dev))
        per_device_weights.append(jax.device_put(weights, dev))
    topk_ids = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k), ep_sharding, per_device_ids,
    )
    topk_weights = jax.make_array_from_single_device_arrays(
        (num_tokens, top_k), ep_sharding, per_device_weights,
    )
    return topk_weights, topk_ids


log("creating weight arrays...")
w1 = make_sharded(k2, (E, d, f), jnp.bfloat16, 0.01)
w2 = make_sharded(k3, (E, f, d), jnp.bfloat16, 0.01)
w3 = make_sharded(k4, (E, d, f), jnp.bfloat16, 0.01)

w1_scale_s = w2_scale_s = w3_scale_s = None
qbk_arg = None
if use_fp8:
    log(f"quantizing weights to fp8 (quant_block_k={quant_block_k})...")

    if quant_block_k is None:
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
            w_f32 = local_w.astype(jnp.float32)
            amax = jnp.max(jnp.abs(w_f32), axis=1, keepdims=True)
            scale = jnp.maximum(amax / 448.0, jnp.float32(1e-12))
            w_q = (w_f32 / scale).astype(jnp.float8_e4m3fn)
            scale = scale[:, :, None, :]  # (E, 1, N) -> (E, 1, 1, N)
            return w_q, scale.astype(jnp.float32)
    else:
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
        tune_configs = generate_tune_candidates(
            f, d, padded_local_nt, ep_size, E, top_k,
            use_fp8=use_fp8,
            quant_block_k=quant_block_k,
            direct_scaled_dot=direct_scaled_dot,
            interleave_bt=interleave_bt_modes[0],
            enable_bt_scatter_overlap=enable_bt_scatter_overlap,
            bse=bse,
            verbose=(num_tokens == token_candidates[0]),
        )
        block_configs_to_try = tune_configs
    else:
        block_configs_to_try = [
            FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=bse, bts=bts)
            for bt, bf, btc, bts in itertools.product(
                bt_candidates, bf_candidates, btc_candidates, bts_candidates,
            )
        ]

    tokens = make_sharded(k1, (num_tokens, d), jnp.bfloat16)
    if routing_mode == "deterministic":
        topk_wts, topk_idx = make_deterministic_topk(num_tokens, top_k, E)
    elif routing_mode == "hot_expert":
        topk_wts, topk_idx = make_hot_expert_topk(num_tokens, top_k, E)
    else:
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

    configs_to_try = [
        (bc_raw, *flags)
        for bc_raw in block_configs_to_try
        for flags in itertools.product(
            cross_expert_prefetch_modes,
            next_w2_prologue_priorities,
            direct_scaled_dot_ffn1_modes,
            direct_scaled_dot_ffn2_modes,
            ffn1_dequant_modes,
            ffn1_dequant_chunks,
            w2_fetch_orders,
            w2_fetch_priorities,
            skip_inter_bt_sync_modes,
            interleave_bt_modes,
        )
    ]
    seen_resolved_configs = set()

    for (
        bc,
        xprefetch_mode,
        next_w2_priority,
        direct_ffn1,
        direct_ffn2,
        ffn1_dequant_mode,
        ffn1_dequant_chunk,
        w2_fetch_order,
        w2_fetch_priority,
        skip_inter_bt_sync,
        interleave_bt,
    ) in configs_to_try:
        if xprefetch_mode != "w13" and next_w2_priority != next_w2_prologue_priorities[0]:
            continue
        if direct_ffn1 and (
            ffn1_dequant_mode != ffn1_dequant_modes[0]
            or ffn1_dequant_chunk != ffn1_dequant_chunks[0]
        ):
            continue
        if ffn1_dequant_mode != "fchunk" and ffn1_dequant_chunk is not None:
            continue
        bt, bf, btc, bts = bc.bt, bc.bf, bc.btc, bc.bts
        ffn1_mode_tag = "direct" if direct_ffn1 else ffn1_dequant_mode
        tag = (
            f"bt={bt},bf={bf},btc={btc},bts={bts},"
            f"xprefetch={xprefetch_mode},w2p={next_w2_priority},"
            f"w2order={w2_fetch_order},w2fp={w2_fetch_priority},"
            f"direct_f1={int(direct_ffn1)},direct_f2={int(direct_ffn2)},"
            f"cast_f1={int(cast_ffn1_input_fp8)},cast_f2={int(cast_ffn2_input_fp8)},"
            f"ffn1dq={ffn1_mode_tag},ffn1chunk={ffn1_dequant_chunk},"
            f"skip_ibt={int(skip_inter_bt_sync)},"
            f"ilv_bt={int(interleave_bt)}"
        )

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

        tag_resolved = (
            f"bt={bc_resolved.bt},bf={bc_resolved.bf},"
            f"btc={bc_resolved.btc},bts={bc_resolved.bts},"
            f"xprefetch={xprefetch_mode},w2p={next_w2_priority},"
            f"w2order={w2_fetch_order},w2fp={w2_fetch_priority},"
            f"direct_f1={int(direct_ffn1)},direct_f2={int(direct_ffn2)},"
            f"cast_f1={int(cast_ffn1_input_fp8)},cast_f2={int(cast_ffn2_input_fp8)},"
            f"ffn1dq={ffn1_mode_tag},ffn1chunk={ffn1_dequant_chunk},"
            f"skip_ibt={int(skip_inter_bt_sync)},"
            f"ilv_bt={int(interleave_bt)}"
        )
        resolved_key = (
            bc_resolved.bt,
            bc_resolved.bf,
            bc_resolved.btc,
            bc_resolved.bts,
            xprefetch_mode,
            next_w2_priority,
            direct_ffn1,
            direct_ffn2,
            ffn1_dequant_mode,
            ffn1_dequant_chunk,
            w2_fetch_order,
            w2_fetch_priority,
            skip_inter_bt_sync,
            interleave_bt,
        )
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
                disable_a2a_scatter=disable_a2a_scatter,
                disable_a2a_scatter_local_copy=disable_a2a_scatter_local_copy,
                disable_a2a_scatter_remote_copy=disable_a2a_scatter_remote_copy,
                disable_a2a_scatter_recv_wait=disable_a2a_scatter_recv_wait,
                disable_a2a_scatter_send_wait=disable_a2a_scatter_send_wait,
                disable_a2a_gather=disable_a2a_gather,
                disable_a2a_gather_local_copy=disable_a2a_gather_local_copy,
                disable_a2a_gather_remote_copy=disable_a2a_gather_remote_copy,
                disable_sync_barrier=disable_sync_barrier,
                disable_weight_load=disable_weight_load,
                disable_w1_load=disable_w1_load,
                disable_w3_load=disable_w3_load,
                disable_w2_load=disable_w2_load,
                disable_expert_x_load=disable_expert_x_load,
                disable_expert_ffn=disable_expert_ffn,
                disable_dynamic_ffn1=disable_dynamic_ffn1,
                disable_dynamic_ffn2=disable_dynamic_ffn2,
                disable_expert_store=disable_expert_store,
                disable_expert_stage_writeback=disable_expert_stage_writeback,
                disable_expert_store_dma=disable_expert_store_dma,
                disable_expert_store_wait=disable_expert_store_wait,
                disable_acc_and_store=disable_acc_and_store,
                disable_acc_load=disable_acc_load,
                disable_acc_compute=disable_acc_compute,
                disable_acc_store_vmem=disable_acc_store_vmem,
                disable_output_store=disable_output_store,
                direct_scaled_dot=direct_scaled_dot,
                direct_scaled_dot_ffn1=direct_ffn1,
                direct_scaled_dot_ffn2=direct_ffn2,
                ffn1_dequant_mode=ffn1_dequant_mode,
                ffn1_dequant_chunk=ffn1_dequant_chunk,
                cast_ffn1_input_fp8=cast_ffn1_input_fp8,
                cast_ffn2_input_fp8=cast_ffn2_input_fp8,
                enable_act_quant=enable_act_quant,
                cross_expert_prefetch_mode=xprefetch_mode,
                next_w2_prologue_priority=next_w2_priority,
                w2_fetch_order=w2_fetch_order,
                w2_fetch_priority=w2_fetch_priority,
                skip_inter_bt_sync=skip_inter_bt_sync,
                interleave_bt=interleave_bt,
                enable_bt_scatter_overlap=enable_bt_scatter_overlap,
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
            direct_scaled_dot_ffn1=direct_scaled_dot_ffn1_modes[0],
            direct_scaled_dot_ffn2=direct_scaled_dot_ffn2_modes[0],
            ffn1_dequant_mode=ffn1_dequant_modes[0],
            ffn1_dequant_chunk=ffn1_dequant_chunks[0],
            cast_ffn1_input_fp8=cast_ffn1_input_fp8,
            cast_ffn2_input_fp8=cast_ffn2_input_fp8,
            enable_act_quant=enable_act_quant,
            w2_fetch_order=w2_fetch_orders[0],
            w2_fetch_priority=w2_fetch_priorities[0],
            skip_inter_bt_sync=skip_inter_bt_sync_modes[0],
            interleave_bt=interleave_bt_modes[0],
            enable_bt_scatter_overlap=enable_bt_scatter_overlap,
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
