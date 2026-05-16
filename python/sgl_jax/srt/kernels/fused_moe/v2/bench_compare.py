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
log(f"initialized: {jax.device_count()} devices, {jax.process_index()}")

sys.path.insert(0, "/tmp/tpu_logs/v2")
from kernel import fused_ep_moe_v2, FusedMoEBlockConfig

sys.path.insert(0, "/tmp/tpu_logs/v1")
import v1_kernel
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

# Tuned configs per token count
# V1: from tuned_block_configs.py (bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse)
V1_TUNED = {
    64:   (2, 2048, 2048, 2048, 4, 4, 2048, 2048, 2048, 2048),
    128:  (4, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 2048),
    256:  (8, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 2048),
    512:  (16, 2048, 2048, 2048, 16, 16, 2048, 2048, 2048, 2048),
    8192: (128, 1024, 2048, 2048, 64, 64, 1024, 2048, 2048, 1024),
    16384: (128, 1024, 2048, 2048, 64, 64, 1024, 2048, 2048, 1024),
}
# V2: from v2 tune sweep at ep=32 fp8 (bt, bf, btc, bse)
# For 64/128: V2 pads local_nt to 8 internally, these configs resolve to bt=8
V2_TUNED = {
    64:   (8, 512, 128, 256),
    128:  (8, 256, 128, 256),
    256:  (8, 512, 128, 256),
    512:  (16, 256, 128, 256),
    8192: (128, 1024, 128, 256),
    16384: (128, 1024, 128, 256),
}

token_candidates = list(V1_TUNED.keys())

ep_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor")))

log(f"model: E={E} d={d} f={f} k={top_k} ep={ep_size} fp8={use_fp8}")
log(f"tokens={token_candidates} warmup={warmup} iters={iters}")


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
        )

    log("  V1: compiling + running...")
    v1_times = wall_timeit(run_v1, warmup, iters)

    # --- V2 (tuned config) ---
    bt2, bf2, btc2, bse2 = V2_TUNED[num_tokens]
    v2_bc = FusedMoEBlockConfig(bt=bt2, bf=bf2, btc=btc2, bse=bse2)
    local_nt_raw = num_tokens // ep_size
    pad_align = 8
    pad_local = (pad_align - local_nt_raw % pad_align) % pad_align
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
        )

    log("  V2: compiling + running...")
    v2_times = wall_timeit(run_v2, warmup, iters)

    if jax.process_index() == 0:
        v1_avg = np.mean(v1_times)
        v2_avg = np.mean(v2_times)
        log(f"  V1: {v1_avg:.3f} ms | samples={[round(t, 3) for t in v1_times]}")
        log(f"  V2: {v2_avg:.3f} ms | samples={[round(t, 3) for t in v2_times]}")
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
