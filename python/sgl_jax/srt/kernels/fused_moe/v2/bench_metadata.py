"""Standalone benchmark for jax_allreduce_metadata_by_bt.

Isolates the metadata all_gather cost from the pallas kernel.
Uses split timing (dispatch vs block_until_ready) to match bench_v2 methodology.

Env vars:
  BENCH_TOKENS  — comma-separated token counts (default: 512)
  BENCH_BT      — bt value (default: 16)
  BENCH_E       — num experts (default: 384)
  BENCH_TOPK    — top_k (default: 8)
  BENCH_WARMUP  — warmup iterations (default: 3)
  BENCH_ITERS   — timed iterations (default: 20)
"""
from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:.1f}s][p{jax.process_index()}] {msg}", flush=True)


jax.distributed.initialize()
log(f"initialized: {jax.device_count()} devices, {jax.process_count()} procs")

from kernel import jax_allreduce_metadata_by_bt

P = jax.sharding.PartitionSpec
num_devices = jax.device_count()
devices = np.array(jax.devices()).reshape(1, num_devices)
mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

E = int(os.environ.get("BENCH_E", "384"))
top_k = int(os.environ.get("BENCH_TOPK", "8"))
bt = int(os.environ.get("BENCH_BT", "16"))
warmup = int(os.environ.get("BENCH_WARMUP", "3"))
iters = int(os.environ.get("BENCH_ITERS", "20"))
token_candidates = [int(x) for x in os.environ.get("BENCH_TOKENS", "512").split(",")]

padded_num_experts = ((E + 127) // 128) * 128

log(f"E={E} top_k={top_k} bt={bt} ep={num_devices} padded_E={padded_num_experts}")


def make_topk_ids(num_tokens):
    key = jax.random.key(42)
    local_shape = (num_tokens // num_devices, top_k)
    per_device = []
    for i, dev in enumerate(jax.local_devices()):
        sk = jax.random.fold_in(key, jax.process_index() * len(jax.local_devices()) + i)
        per_device.append(jax.device_put(
            jax.random.randint(sk, local_shape, 0, E, dtype=jnp.int32), dev,
        ))
    return jax.make_array_from_single_device_arrays(
        (num_tokens, top_k), ep_sharding, per_device,
    )


for num_tokens in token_candidates:
    log(f"--- tokens={num_tokens} ---")
    topk_ids = make_topk_ids(num_tokens)

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(("data", "tensor")),),
        out_specs=(P(("data", "tensor")), P(), P()),
        check_vma=False,
    )
    def metadata_fn(topk_ids):
        starts, sizes, d2e_counts = jax_allreduce_metadata_by_bt(
            topk_ids, padded_num_experts, bt, num_devices, "data", "tensor",
        )
        return starts, sizes, d2e_counts

    # Compile + warmup
    t_compile = time.monotonic()
    out = metadata_fn(topk_ids)
    jax.block_until_ready(out)
    compile_ms = (time.monotonic() - t_compile) * 1e3
    log(f"  compile: {compile_ms:.0f}ms")

    for _ in range(warmup - 1):
        out = metadata_fn(topk_ids)
        jax.block_until_ready(out)

    # Split timing
    dispatch_times = []
    wait_times = []
    for _ in range(iters):
        t_start = time.monotonic()
        out = metadata_fn(topk_ids)
        t_mid = time.monotonic()
        jax.block_until_ready(out)
        t_end = time.monotonic()
        dispatch_times.append((t_mid - t_start) * 1e3)
        wait_times.append((t_end - t_mid) * 1e3)

    if jax.process_index() == 0:
        d_arr = np.array(dispatch_times)
        w_arr = np.array(wait_times)
        wall_arr = d_arr + w_arr
        log(f"  metadata: wall={np.mean(wall_arr):.3f}ms = dispatch={np.mean(d_arr):.3f}ms + wait={np.mean(w_arr):.3f}ms")
        log(f"    dispatch: {[round(t, 3) for t in dispatch_times]}")
        log(f"    wait:     {[round(t, 3) for t in wait_times]}")
        log(f"    wall:     {[round(t, 3) for t in wall_arr.tolist()]}")

log("done")
