"""int8+scale expert path == float path on the same FP4-grid weights (tiny model, sharded on 8 virtual devices)."""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
R = sys.argv[1]
sys.path.insert(0, R)
sys.path.insert(0, ".")
import engine as EN
import jax
import jax.numpy as jnp
import loader as L
import model as M
import model_ref as J
import numpy as np
import sharding as SH
import torch

base = dict(
    dtype="bf16",
    expert_dtype=None,
    n_mtp_layers=0,
    compress_ratios=(0, 2, 2, 1, 1, 1),
    kv_source_layers=(1, 3),
    index_source_layers=(1, 3, 5),
    n_layers=6,
    max_seq_len=256,
    max_batch_size=2,
    window_size=32,
    candidate_source_layer=3,
    candidate_topk_blocks=4,
    candidate_block_size=4,
    index_topk=8,
)
a = J.Args.from_torch(M.ModelArgs(**base))
sd = {k: v.float().numpy() for k, v in torch.load(f"{R}/ref_weights.pt").items()}
rng = np.random.default_rng(0)


def quant_fp4(
    w,
):  # [N,K] float -> (int8 x2 values, uint8 exponent) on the e2m1 grid with pow2 block scales
    n, k = w.shape
    wb = w.reshape(n, k // 32, 32)
    amax = np.abs(wb).max(-1, keepdims=True).clip(1e-8)
    e = np.ceil(np.log2(amax / 6.0)).astype(np.int32)
    sc = np.ldexp(np.float32(1), e)
    t = wb / sc
    grid = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    idx = np.abs(np.abs(t)[..., None] - grid).argmin(-1)
    q = np.sign(t) * grid[idx]
    return (
        (q * 2).astype(np.int8).reshape(n, k),
        (e[..., 0] + 127).astype(np.uint8),
        (q * sc).reshape(n, k),
    )


p_int8, p_float, p_f8 = {}, {}, {}
for k, v in sd.items():
    if ".ffn.experts." in k:
        continue
    p_int8[k] = p_float[k] = p_f8[k] = jnp.asarray(v)
for l in range(a.n_layers):
    for w in ("w1", "w2", "w3"):
        qs = [
            quant_fp4(sd[f"layers.{l}.ffn.experts.{e}.{w}.weight"])
            for e in range(a.n_routed_experts)
        ]
        p_int8[f"layers.{l}.ffn.experts.{w}"] = jnp.asarray(np.stack([q[0] for q in qs]))
        p_int8[f"layers.{l}.ffn.experts.{w}_scale"] = jnp.asarray(np.stack([q[1] for q in qs]))
        p_float[f"layers.{l}.ffn.experts.{w}"] = jnp.asarray(np.stack([q[2] for q in qs]))
        cv = [L.fp4_to_fp8_rowscale(q[0], q[1]) for q in qs]
        assert all(c[2] == 0 for c in cv), "fp8 row-scale conversion not exact in the tiny model"
        p_f8[f"layers.{l}.ffn.experts.{w}"] = jnp.asarray(np.stack([c[0] for c in cv]))
        p_f8[f"layers.{l}.ffn.experts.{w}_rowscale"] = jnp.asarray(np.stack([c[1] for c in cv]))
mesh = SH.make_mesh(jax.devices(), 2, 4)
ids = jnp.asarray(np.load(f"{R}/ref_outputs_noquant.npz")["input_ids"])
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn"))
outs = []
for P_ in (p_int8, p_float, p_f8):
    P_ = SH.shard_params(P_, mesh)
    st = SH.shard_states(EN.init_states(a, 2), mesh)
    with mesh:
        lg, st = step(P_, a, ids[:, :32], jnp.int32(0), st, ropes)
        lg, st = step(P_, a, ids[:, 32:33], jnp.int32(32), st, ropes)
    outs.append(np.asarray(lg))
d = np.abs(outs[0] - outs[1])
print(
    f"int8+scale path vs float path (same FP4-grid weights): max|d|={d.max():.2e}  rel={d.max()/np.abs(outs[1]).max():.2e}"
)
d = np.abs(outs[2] - outs[1])
print(
    f"fp8+rowscale path vs float path: max|d|={d.max():.2e}  rel={d.max()/np.abs(outs[1]).max():.2e}"
)
# sparse (shard_map + loop over used local experts) vs dense fp8 path, prefill-sized and decode-sized calls
J.MOE_MESH = mesh
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn"))  # fresh trace
print(
    "sparse lowering contains while:",
    "while"
    in step.lower(
        SH.shard_params(p_f8, mesh),
        a,
        ids[:, 32:33],
        jnp.int32(32),
        SH.shard_states(EN.init_states(a, 2), mesh),
        ropes,
    ).as_text(),
)
P_ = SH.shard_params(p_f8, mesh)
st = SH.shard_states(EN.init_states(a, 2), mesh)
with mesh:
    lg, st = step(P_, a, ids[:, :32], jnp.int32(0), st, ropes)
    lg, st = step(P_, a, ids[:, 32:33], jnp.int32(32), st, ropes)
d = np.abs(np.asarray(lg) - outs[2])
import os

print(f"(V41_MOE_CHUNK={os.environ.get('V41_MOE_CHUNK', '128')})", end=" ")
print(
    f"grouped fp8 path vs dense fp8 path: max|d|={d.max():.2e}  rel={d.max()/np.abs(outs[2]).max():.2e}"
)
