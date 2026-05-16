"""Standalone micro-bench for fused_ep_moe_v2."""
from __future__ import annotations

import os
import sys
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

from kernel import fused_ep_moe_v2, ref_moe, FusedMoEBlockConfig

P = jax.sharding.PartitionSpec
num_devices = jax.device_count()

devices = np.array(jax.devices()).reshape(1, num_devices)
mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
ep_size = num_devices

# MiMo V2 Pro config
d, f, E, top_k = 6144, 2048, 384, 8
bt = int(os.environ.get("BENCH_BT", "128"))
bf = int(os.environ.get("BENCH_BF", "256"))
btc = int(os.environ.get("BENCH_BTC", "128"))
bse = int(os.environ.get("BENCH_BSE", "256"))
num_tokens = int(os.environ.get("BENCH_TOKENS", str(bt * ep_size)))
warmup = int(os.environ.get("BENCH_WARMUP", "2"))
iters = int(os.environ.get("BENCH_ITERS", "5"))
check_correctness = os.environ.get("BENCH_CHECK", "0") == "1"

log(f"config: E={E} d={d} f={f} k={top_k} bt={bt} bf={bf} btc={btc} ep={ep_size} tokens={num_tokens}")

ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

key = jax.random.key(42)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)

log("creating arrays...")

def make_sharded(key, shape, dtype, scale=1.0):
    """Create a sharded array — each process generates only its local shard."""
    local_shape = (shape[0] // num_devices, *shape[1:])
    per_device_arrays = []
    for i, dev in enumerate(jax.local_devices()):
        shard_key = jax.random.fold_in(key, jax.process_index() * len(jax.local_devices()) + i)
        shard = jax.device_put(
            jax.random.normal(shard_key, local_shape, dtype=dtype) * scale,
            dev,
        )
        per_device_arrays.append(shard)
    return jax.make_array_from_single_device_arrays(
        shape, ep_sharding, per_device_arrays,
    )

tokens = make_sharded(k1, (num_tokens, d), jnp.bfloat16)
w1 = make_sharded(k2, (E, d, f), jnp.bfloat16, 0.01)
w2 = make_sharded(k3, (E, f, d), jnp.bfloat16, 0.01)
w3 = make_sharded(k4, (E, d, f), jnp.bfloat16, 0.01)

gating_local_shape = (num_tokens // num_devices,  E)
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

log("arrays ready (pre-sharded)")
tokens_s = tokens
w1_s = w1
w2_s = w2
w3_s = w3
topk_wts_s = topk_wts
topk_idx_s = topk_idx

bc = FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=bse)

log("warmup (compile + run)...")
for i in range(warmup):
    result = fused_ep_moe_v2(
        mesh, tokens_s, w1_s, w2_s, w3_s,
        topk_wts_s, topk_idx_s, top_k,
        block_config=bc,
    )
    result.block_until_ready()
    log(f"  warmup {i+1}/{warmup} done")

log(f"timing {iters} iterations...")
times = []
for i in range(iters):
    t_start = time.time()
    result = fused_ep_moe_v2(
        mesh, tokens_s, w1_s, w2_s, w3_s,
        topk_wts_s, topk_idx_s, top_k,
        block_config=bc,
    )
    result.block_until_ready()
    elapsed = time.time() - t_start
    times.append(elapsed)
    log(f"  iter {i+1}: {elapsed*1000:.2f} ms")

if jax.process_index() == 0:
    times_ms = [t * 1000 for t in times]
    log(f"wall time: min={min(times_ms):.2f} ms, avg={np.mean(times_ms):.2f} ms, max={max(times_ms):.2f} ms")

if check_correctness:
    if jax.process_count() > 1:
        log("SKIP correctness check in multi-host mode (use single-host ep=8)")
    else:
        log("computing reference...")
        ref = ref_moe(
            jax.device_get(tokens), jax.device_get(w1),
            jax.device_get(w2), jax.device_get(w3),
            jax.device_get(topk_wts), jax.device_get(topk_idx), top_k,
        )
        result_gathered = jax.device_get(
            jax.device_put(result, jax.sharding.NamedSharding(mesh, P()))
        )
        result_f32 = result_gathered.astype(np.float32)
        ref_f32 = np.asarray(ref).astype(np.float32)
        max_err = np.max(np.abs(result_f32 - ref_f32))
        rel_err = float(max_err / (np.max(np.abs(ref_f32)) + 1e-6))
        log(f"max_abs_err={max_err:.4f}, rel_err={rel_err:.6f}")
        if rel_err > 0.05:
            log("FAIL: relative error too high")
            sys.exit(1)
        log("PASS")

log("done")
