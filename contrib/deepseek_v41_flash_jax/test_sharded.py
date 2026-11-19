"""Sharded jit (GSPMD) on virtual CPU devices vs the reference. Usage: [V41_NOQUANT=1] python test_sharded.py <ref_cpu_dir>"""

import os
import sys
import time

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
SUFFIX = "_noquant" if os.environ.get("V41_NOQUANT") == "1" else ""
R = sys.argv[1]
sys.path.insert(0, R)
sys.path.insert(0, ".")
import engine as EN
import jax
import jax.numpy as jnp
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
mesh = SH.make_mesh(jax.devices(), dp=2, tp=4)
params = {k: jnp.asarray(v.float().numpy()) for k, v in torch.load(f"{R}/ref_weights.pt").items()}
params = SH.shard_params(SH.stack_experts(params, a.n_layers, a.n_routed_experts), mesh)
ex = params["layers.0.ffn.experts.w1"]
print(
    "devices:",
    len(jax.devices()),
    "| expert tensor",
    ex.shape,
    "-> per-device shard",
    ex.addressable_shards[0].data.shape,
)
print(
    "wq_b",
    params["layers.0.attn.wq_b.weight"].shape,
    "-> shard",
    params["layers.0.attn.wq_b.weight"].addressable_shards[0].data.shape,
)
r0 = np.load(f"{R}/ref_outputs{SUFFIX}.npz")
ids = jnp.asarray(r0["input_ids"])
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
states = SH.shard_states(EN.init_states(a, 2), mesh)
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn"))


def cmp(tag, got, want):
    d = np.abs(np.asarray(got) - want)
    print(
        f"  {tag:<12} max|d|={d.max():.2e} rel={d.max()/np.abs(want).max():.2e} argmax_match={bool((np.asarray(got).argmax(-1)==want.argmax(-1)).all())}"
    )


t = time.time()
with mesh:
    pos = 0
    for S in (32, 32):
        lg, states = step(params, a, ids[:, pos : pos + S], jnp.int32(pos), states, ropes)
        pos += S
    cmp("prefill 64", lg, r0["prefill_logits"])
    for i in range(64, 72):
        lg, states = step(params, a, ids[:, i : i + 1], jnp.int32(i), states, ropes)
        cmp(f"decode {i}", lg, r0[f"decode_logits_{i}"])
print(f"done {time.time()-t:.0f}s")
