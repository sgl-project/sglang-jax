"""Minimal multi-device correctness test for fused_ep_moe_v2."""
from __future__ import annotations

import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

t0 = time.time()
_pid = [None]


def log(msg):
    if _pid[0] is None:
        _pid[0] = 0
    print(f"[{time.time()-t0:.1f}s][p{_pid[0]}] {msg}", flush=True)


log("before distributed.initialize")
jax.distributed.initialize()
_pid[0] = jax.process_index()
log(f"initialized: {jax.device_count()} devices, {jax.process_count()} processes")

sys.path.insert(0, "/tmp/tpu_logs/sglang-jax/python/sgl_jax/srt/kernels/fused_moe/v2")
from kernel import fused_ep_moe_v2, ref_moe, FusedMoEBlockConfig

P = jax.sharding.PartitionSpec
num_devices = jax.device_count()

devices = np.array(jax.devices()).reshape(1, num_devices)
mesh = jax.sharding.Mesh(devices, ("data", "tensor"))
ep_size = num_devices

# bt must be >= 128 for btc=128 alignment
d, f, E, top_k = 768, 256, 64, 2
bt, bf, btc = 128, 256, 128
num_tokens = bt * ep_size

log(f"config: E={E} d={d} f={f} k={top_k} bt={bt} ep={ep_size} tokens={num_tokens}")

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

bc = FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=256)

log("calling fused_ep_moe_v2 (compile + run) with disable_a2a=True...")
result = fused_ep_moe_v2(
    mesh, tokens_s, w1_s, w2_s, w3_s,
    topk_wts_s, topk_idx_s, top_k,
    block_config=bc,
    disable_a2a=True,
    disable_sync_barrier=True,
)
log("blocking on result...")
result.block_until_ready()
log("done!")

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
