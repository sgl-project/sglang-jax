"""Loader end to end on the synthetic checkpoint: build_params -> sharded engine -> compare with the reference (dequantized weights)."""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["V41_NOQUANT"] = "1"
R, CK = sys.argv[1], sys.argv[2]
sys.path.insert(0, R)
sys.path.insert(0, ".")
import config_real as C
import engine as EN
import jax
import jax.numpy as jnp
import loader as L
import model as M
import model_ref as J
import numpy as np
import sharding as SH

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
mesh = SH.make_mesh(jax.devices(), 2, 4)
table = C.param_table(a, act=jnp.float32)
ck = L.SafetensorsDir(CK)
params = L.build_params(ck, table, mesh, SH.param_spec, a.n_routed_experts, threads=4)
ref = np.load(f"{CK}/ref_outputs_deq_noquant.npz")
ids = jnp.asarray(ref["input_ids"])
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
st = SH.shard_states(EN.init_states(a, 2), mesh)
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn"))
with mesh:
    lg, st = step(params, a, ids[:, :32], jnp.int32(0), st, ropes)
    lg, st = step(params, a, ids[:, 32:64], jnp.int32(32), st, ropes)
    d = np.abs(np.asarray(lg) - ref["prefill_logits"])
    print(
        f"prefill rel={d.max()/np.abs(ref['prefill_logits']).max():.2e} argmax_match={(np.asarray(lg).argmax(-1)==ref['prefill_logits'].argmax(-1)).all()}"
    )
    for i in range(64, 68):
        lg, st = step(params, a, ids[:, i : i + 1], jnp.int32(i), st, ropes)
        d = np.abs(np.asarray(lg) - ref[f"decode_logits_{i}"])
        print(
            f"decode {i} rel={d.max()/np.abs(ref[f'decode_logits_{i}']).max():.2e} argmax_match={(np.asarray(lg).argmax(-1)==ref[f'decode_logits_{i}'].argmax(-1)).all()}"
        )
