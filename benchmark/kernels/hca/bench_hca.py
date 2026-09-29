"""Benchmark native stateful HCA over a (batch size x sequence length) grid.

One measured call is the complete stateful HCA operator: compressor projection
and recurrent-state update, sliding-window/compressed-cache writes, and
cache-aware attention. Compilation and host metadata construction are excluded.

Three request shapes, each swept over the same grid:
  - prefill: B fresh requests of S tokens each (prefix_len = 0)
  - decode : B requests contributing one token each against an S-token context
  - ragged : B requests each extending an S-token prefix by an S-token chunk,
             which drives the ragged path (history gather, combined KV,
             per-request page offsets). Mixed q_len coverage lives in
             test/srt/kernels/hca, not here.

Page size is HCABackend's shipped default (128), so these numbers are
comparable with the packaged tpu-inference comparison.

Usage:
  python -m benchmark.kernels.hca.bench_hca
  python -m benchmark.kernels.hca.bench_hca --batch-sizes 1,4,8,32 --seq-lens 128,512,2048
  python -m benchmark.kernels.hca.bench_hca --modes prefill --profile
"""

from __future__ import annotations

import argparse
import gc
import math
import re
import tempfile
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hca_backend import HCABackend
from sgl_jax.srt.mem_cache.hca_allocator import HCAKVPoolAllocator
from sgl_jax.srt.mem_cache.hca_pool import HCAKVPool, HCARecurrentStatePool
from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

MODES = ("prefill", "decode", "ragged")
HIDDEN, HEADS, HEAD_DIM, WINDOW, RATIO = 4096, 64, 512, 128, 128


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def make_inputs(q_lens, prefix_lens, *, seed: int) -> dict:
    """Deterministic request-major batch for arbitrary q_len/prefix_len vectors.

    Cache tiers are populated only where a prefix actually exists, so a fresh
    request sees zeros and a continuing one sees live history.
    """
    q_lens = np.asarray(q_lens, np.int32)
    prefix_lens = np.asarray(prefix_lens, np.int32)
    batch = q_lens.size
    seq_lens = prefix_lens + q_lens
    positions = np.concatenate(
        [np.arange(start, end) for start, end in zip(prefix_lens, seq_lens)]
    ).astype(np.int32)
    tokens = int(q_lens.sum())
    keys = jax.random.split(jax.random.key(seed), 13)

    def normal(index, shape, dtype=jnp.bfloat16, scale=0.02):
        return jax.random.normal(keys[index], shape, dtype) * scale

    # Rows past a request's live prefix must stay inert: zero KV, -inf scores.
    slots = np.arange(WINDOW, dtype=np.int32)[None, :]
    live = slots < np.minimum(prefix_lens[:, None], WINDOW)
    entries = max(1, int(np.floor_divide(seq_lens, RATIO).max()))
    old = np.floor_divide(prefix_lens, RATIO)[:, None]
    fresh = (np.arange(entries, dtype=np.int32)[None, :] < old)[..., None]
    angle = normal(7, (int(seq_lens.max()), 32), jnp.float32)

    data = {
        "hidden": normal(0, (tokens, HIDDEN)),
        "q": normal(1, (tokens, HEADS, HEAD_DIM)),
        "new_kv": normal(2, (tokens, HEAD_DIM)),
        "wkv": normal(3, (HEAD_DIM, HIDDEN)),
        "wgate": normal(4, (HEAD_DIM, HIDDEN)),
        "ape": normal(5, (RATIO, HEAD_DIM), jnp.float32),
        "norm_weight": normal(6, (HEAD_DIM,)),
        "cos": jnp.cos(angle),
        "sin": jnp.sin(angle),
        "window_cache": jnp.where(live[..., None], normal(8, (batch, WINDOW, HEAD_DIM)), 0),
        "compressed_cache": jnp.where(fresh, normal(9, (batch, entries, HEAD_DIM)), 0),
        "kv_state": jnp.where(
            live[..., None], normal(10, (batch, WINDOW, HEAD_DIM), jnp.float32), 0.0
        ),
        "score_state": jnp.where(
            live[..., None],
            normal(11, (batch, WINDOW, HEAD_DIM), jnp.float32),
            -jnp.inf,
        ),
        "attention_sink": normal(12, (HEADS,), jnp.float32),
        "positions": jnp.asarray(positions),
        "seq_lens": seq_lens,
        "q_lens": q_lens,
        "prefix_lens": prefix_lens,
    }
    jax.block_until_ready([v for v in data.values() if isinstance(v, jax.Array)])
    return data


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------
def prepare(data: dict, *, decode: bool, page_size: int = 128):
    """Build pools, page tables and backend metadata; return a jitted callable.

    Large tensors are bound as JIT operands rather than captured as XLA
    constants, so the timed region is the kernel and not constant folding.
    The three pool buffers are donated, matching how the model runner calls
    this in serving: without donation XLA must preserve the inputs, and the
    in-place cache update costs a full copy of both KV tiers every step
    (measured 20-26% on the large shapes).
    """
    seq_lens = np.asarray(data["seq_lens"], np.int32)
    batch = seq_lens.size
    max_context_len = int(max(seq_lens.max(), WINDOW))
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1], object).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )

    def put(value, spec):
        return jax.device_put(value, NamedSharding(mesh, spec))

    with jax.set_mesh(mesh):
        state_pool = HCARecurrentStatePool([0], batch, mesh)
        kv_pool = HCAKVPool(
            max(batch * max_context_len, page_size),
            page_size,
            jnp.bfloat16,
            1,
            mesh,
            max_num_requests=batch,
            max_context_len=max_context_len,
            layer_ids=[0],
        )
        request_pool = HybridReqToTokenPool(batch, max_context_len, np.int32, state_pool, dp_size=1)
        allocator = HCAKVPoolAllocator(kv_pool, request_pool)
        req_indices = np.asarray(
            allocator.alloc(
                [
                    SimpleNamespace(
                        req_pool_idx=None,
                        recurrent_pool_idx=None,
                        is_chunked=0,
                        kv_committed_len=0,
                        dp_rank=0,
                    )
                    for _ in range(batch)
                ]
            ),
            np.int32,
        )
        state_slots = request_pool.get_linear_recurrent_indices(req_indices)
        entries = data["compressed_cache"].shape[1]
        allocator.ensure_compressed_capacity(
            req_indices, np.full((batch,), entries * RATIO, np.int32)
        )
        allocator.ensure_compressed_capacity(req_indices, seq_lens)

        state_pool.state_buffers[0] = (
            state_pool.state_buffers[0]
            .at[state_slots]
            .set(
                jnp.stack((data["kv_state"], data["score_state"]), axis=2),
                mode="promise_in_bounds",
                unique_indices=True,
                out_sharding=P("data", None, None, None),
            )
        )
        kv_pool.window_buffer[0] = kv_pool._scatter_slots(
            kv_pool.window_buffer[0],
            allocator.window_slots[req_indices].reshape(-1),
            data["window_cache"].reshape(-1, HEAD_DIM),
        )
        kv_pool.compressed_buffer[0] = kv_pool._scatter_slots(
            kv_pool.compressed_buffer[0],
            allocator.compressed_slots[req_indices, :entries].reshape(-1),
            data["compressed_cache"].reshape(-1, HEAD_DIM),
        )

        mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
        backend = HCABackend(mesh=mesh, page_size=page_size)
        backend.allocator = allocator
        backend.forward_metadata = backend.get_forward_metadata(
            SimpleNamespace(
                forward_mode=mode,
                req_pool_indices=req_indices,
                seq_lens=seq_lens,
                positions=np.asarray(data["positions"], np.int32),
                extend_seq_lens=None if decode else data["q_lens"],
                extend_prefix_lens=None if decode else data["prefix_lens"],
                recurrent_indices=state_slots,
            )
        )
        layer = SimpleNamespace(layer_id=0, scaling=HEAD_DIM**-0.5)
        forward_batch = SimpleNamespace(
            forward_mode=mode, positions=put(data["positions"], P("data"))
        )

        def execute(
            hidden,
            q,
            new_kv,
            wkv,
            wgate,
            ape,
            norm,
            cos,
            sin,
            sink,
            fused,
            state,
            window,
            compressed,
        ):
            state_pool.state_buffers[0] = state
            kv_pool.window_buffer[0] = window
            kv_pool.compressed_buffer[0] = compressed
            output, updated = backend(
                q,
                new_kv,
                new_kv,
                layer,
                forward_batch,
                kv_pool,
                recurrent_state_pool=state_pool,
                compressor_input=hidden,
                wkv=wkv,
                wgate=wgate,
                ape=ape,
                norm_weight=norm,
                cos=cos,
                sin=sin,
                attention_sink=sink,
                fused_weight=fused,
            )
            # Measure the kernel-native [T, H, D] output. SGLang's backend
            # contract flattens to [T, H*D] on the way out; converting back
            # lets XLA elide the pair instead of materialising a full-size
            # relayout that the operator itself never asked for. Without this
            # the flatten alone is 5.9 ms of a 22 ms step at 65536 tokens.
            return output.reshape(-1, HEADS, HEAD_DIM), updated

        fixed = (
            put(data["hidden"], P("data", None)),
            put(data["q"], P("data", "tensor", None)),
            put(data["new_kv"], P("data", None)),
            put(data["wkv"], P(None, None)),
            put(data["wgate"], P(None, None)),
            put(data["ape"], P(None, None)),
            put(data["norm_weight"], P(None)),
            put(data["cos"], P(None, None)),
            put(data["sin"], P(None, None)),
            put(data["attention_sink"], P("tensor")),
            put(jnp.concatenate((data["wkv"], data["wgate"]), axis=0).T, P(None, None)),
        )
        # Arguments 11..13 are the pool buffers and are donated, so each call
        # has to hand the updated handles to the next one -- exactly what the
        # model runner does between steps.
        jitted = jax.jit(execute, donate_argnums=(11, 12, 13))
        pools = [
            state_pool.state_buffers[0],
            kv_pool.window_buffer[0],
            kv_pool.compressed_buffer[0],
        ]

        def step():
            output, updated = jitted(*fixed, *pools)
            pools[:] = list(updated)
            return output

        return step


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
TC_PREFIX = "VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_"
HBM_READ = re.compile(
    r"^VF_CHIP_HBM_[01]_HBMC_\d+_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_RD_RESP_PS[01]$"
)
HBM_WRITE = re.compile(
    r"^VF_CHIP_HBM_[01]_HBMC_\d+_CMN_HI_FREQ_STATS_COUNTERS_"
    r"UNPRIVILEGED_(?:WR_REQ|PARTIAL_WRITE_REQ)_PS[01]$"
)
V6E_HZ = 1.75e9


def measure(call, *, warmup: int, iters: int) -> float:
    """Mean latency in ms, after discarding the compiling call."""
    jax.block_until_ready(call())
    for _ in range(warmup):
        jax.block_until_ready(call())
    samples = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        jax.block_until_ready(call())
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.mean(samples))


def resources(call, *, iterations: int) -> tuple[float, float, float]:
    """MXU busy %, HBM bandwidth %, HBM GB/s, all over XLA-module-active time.

    Neither percentage is a wall-time utilization: the denominator is the time
    the device actually had a module resident.
    """
    for _ in range(5):
        jax.block_until_ready(call())
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    with tempfile.TemporaryDirectory() as tmp:  # counters only; keep nothing
        with jax.profiler.trace(tmp, profiler_options=options):
            for _ in range(iterations):
                result = call()
            jax.block_until_ready(result)
        traces = list(Path(tmp).glob("plugins/profile/**/*.xplane.pb"))
        if len(traces) != 1:
            raise RuntimeError(f"expected one xplane.pb, found {traces}")
        profile = jax.profiler.ProfileData.from_file(str(traces[0]))

    device = profile.find_plane_with_name("/device:TPU:0")
    with warnings.catch_warnings():  # profiler stats objects lack __module__
        warnings.simplefilter("ignore", DeprecationWarning)
        stats = dict(device.stats)
    counters: dict[str, int] = {}
    modules = []
    for line in device.lines:
        if line.name == "XLA Modules":
            modules = [(e.start_ns, e.end_ns) for e in line.events]
        elif line.name.startswith("counters_"):
            for event in line.events:
                value = dict(event.stats).get("counter_value")
                if value is not None:
                    counters[event.name] = counters.get(event.name, 0) + int(value)

    merged = []
    for start, end in sorted(modules):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    active_ns = sum(end - start for start, end in merged)
    active_s, cycles = active_ns / 1e9, active_ns * V6E_HZ / 1e9

    mxu = 0.5 * counters.get(TC_PREFIX + "MXU_BUSY_1", 0) + counters.get(
        TC_PREFIX + "MXU_BUSY_2", 0
    )
    hbm = 32 * sum(v for k, v in counters.items() if HBM_READ.match(k) or HBM_WRITE.match(k))
    peak = float(stats.get("peak_hbm_bw_gigabytes_per_second", 0.0)) * 1e9
    return (
        100.0 * mxu / cycles if cycles else math.nan,
        100.0 * hbm / (peak * active_s) if peak and active_s else math.nan,
        hbm / active_s / 1e9 if active_s else math.nan,
    )


def build(mode: str, batch: int, sequence: int, seed: int) -> tuple[object, int]:
    """Return (callable, total query tokens) for one grid point."""
    if mode == "prefill":
        q_lens, prefix_lens, tokens = [sequence] * batch, [0] * batch, batch * sequence
    elif mode == "decode":
        q_lens, prefix_lens, tokens = [1] * batch, [sequence - 1] * batch, batch
    else:
        q_lens, prefix_lens, tokens = (
            [sequence] * batch,
            [sequence] * batch,
            batch * sequence,
        )
    data = make_inputs(q_lens, prefix_lens, seed=seed)
    return prepare(data, decode=mode == "decode"), tokens


def run_benchmark_grid(
    modes: tuple[str, ...],
    batch_sizes: list[int],
    seq_lens: list[int],
    warmup: int,
    iters: int,
    profile: bool,
    profile_iterations: int,
) -> None:
    width = 94 if profile else 58
    print("=" * width)
    print(
        f"HCA Kernel Benchmark (hidden={HIDDEN}, H={HEADS}, D={HEAD_DIM}, "
        f"window={WINDOW}, compression={RATIO}x) on {jax.devices()[0]}"
    )
    print("=" * width)
    header = (
        f"{'mode':>8s} | {'B':>3s} | {'S':>6s} | {'total Q':>8s} | "
        f"{'Lat (ms)':>10s} | {'Q tok/s':>11s}"
    )
    if profile:
        header += f" | {'MXU':>7s} | {'HBM util':>9s} | {'HBM GB/s':>9s}"
    print(header)
    print("-" * width)

    seed = 20270000
    for mode in modes:
        for batch in batch_sizes:
            for sequence in seq_lens:
                seed += 1
                # A shape that exhausts device memory must not discard the rest
                # of the sweep: report it and continue.
                try:
                    call, tokens = build(mode, batch, sequence, seed)
                    latency = measure(call, warmup=warmup, iters=iters)
                    row = (
                        f"{mode:>8s} | {batch:3d} | {sequence:6d} | {tokens:8d} | "
                        f"{latency:10.4f} | {tokens * 1e3 / latency:11.0f}"
                    )
                    if profile:
                        mxu, hbm_pct, hbm_gbs = resources(call, iterations=profile_iterations)
                        row += f" | {mxu:6.1f}% | {hbm_pct:8.1f}% | {hbm_gbs:9.1f}"
                    print(row, flush=True)
                    del call
                except Exception as error:  # noqa: BLE001 - reported, not hidden
                    tokens = batch if mode == "decode" else batch * sequence
                    print(
                        f"{mode:>8s} | {batch:3d} | {sequence:6d} | {tokens:8d} | "
                        f"{'FAILED':>10s} | {type(error).__name__}",
                        flush=True,
                    )
                gc.collect()
    print("=" * width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native stateful HCA")
    parser.add_argument("--modes", default=",".join(MODES), help=f"subset of {','.join(MODES)}")
    parser.add_argument("--batch-sizes", default="1,4,8,32", help="Comma-separated batch sizes")
    parser.add_argument(
        "--seq-lens",
        default="128,512,2048,8192",
        help="Comma-separated per-request lengths",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Timing iterations")
    parser.add_argument(
        "--profile", action="store_true", help="Also report MXU/HBM from XProf counters"
    )
    parser.add_argument("--profile-iterations", type=int, default=10)
    args = parser.parse_args()

    if jax.default_backend() != "tpu":
        raise RuntimeError("HCA benchmark requires a physical TPU")
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    if set(modes) - set(MODES):
        parser.error(f"unknown modes: {sorted(set(modes) - set(MODES))}")
    if min(args.warmup, args.iters) < 1:
        parser.error("warmup and iters must be positive")

    run_benchmark_grid(
        modes=modes,
        batch_sizes=[int(x) for x in args.batch_sizes.split(",") if x.strip()],
        seq_lens=[int(x) for x in args.seq_lens.split(",") if x.strip()],
        warmup=args.warmup,
        iters=args.iters,
        profile=args.profile,
        profile_iterations=args.profile_iterations,
    )


if __name__ == "__main__":
    main()
