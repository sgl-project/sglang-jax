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

key = jax.random.key(42)
k1, k2, k3, k4, k5 = jax.random.split(key, 5)

log("creating arrays...")
tokens = jax.random.normal(k1, (num_tokens, d), dtype=jnp.bfloat16)
w1 = jax.random.normal(k2, (E, d, f), dtype=jnp.bfloat16) * 0.01
w2 = jax.random.normal(k3, (E, f, d), dtype=jnp.bfloat16) * 0.01
w3 = jax.random.normal(k4, (E, d, f), dtype=jnp.bfloat16) * 0.01

gating = jax.random.normal(k5, (num_tokens, E), dtype=jnp.float32)
_, topk_idx = lax.top_k(gating, top_k)
topk_logits = jnp.take_along_axis(gating, topk_idx, axis=-1)
topk_wts = jax.nn.softmax(topk_logits, axis=-1)

ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

log("sharding arrays...")
tokens_s = jax.device_put(tokens, ep_sharding)
w1_s = jax.device_put(w1, ep_sharding)
w2_s = jax.device_put(w2, ep_sharding)
w3_s = jax.device_put(w3, ep_sharding)
topk_wts_s = jax.device_put(topk_wts, ep_sharding)
topk_idx_s = jax.device_put(topk_idx, ep_sharding)

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
    log("computing reference...")
    ref = ref_moe(tokens, w1, w2, w3, topk_wts, topk_idx, top_k)
    result_gathered = jax.device_get(
        jax.device_put(result, jax.sharding.NamedSharding(mesh, P()))
    )
    if jax.process_index() == 0:
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
