"""Tiled indexer + capacity view vs reference. Runs test_engine-style prefill/decode with (a) forced small key tiles and (b) a
capacity `cap` smaller than the allocated cache; logits must match the reference like the untiled path.
Usage: V41_NOQUANT=1 V41_INDEX_TILE=16 python test_longctx.py <ref_cpu_dir>"""

import os
import sys

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
params = {k: jnp.asarray(v.float().numpy()) for k, v in torch.load(f"{R}/ref_weights.pt").items()}
r0 = np.load(f"{R}/ref_outputs{SUFFIX}.npz")
ids = jnp.asarray(r0["input_ids"])
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn", "cap"))
worst = 0
for cap in (None, 128):
    st = EN.init_states(a, 2)
    pos = 0
    for S in (32, 32):
        lg, st = step(params, a, ids[:, pos : pos + S], jnp.int32(pos), st, ropes, cap=cap)
        pos += S
    d = np.abs(np.asarray(lg) - r0["prefill_logits"]).max() / np.abs(r0["prefill_logits"]).max()
    worst = max(worst, d)
    print(f"cap={cap} tile={os.environ.get('V41_INDEX_TILE')} prefill rel={d:.2e}")
    for i in range(64, 72):
        lg, st = step(params, a, ids[:, i : i + 1], jnp.int32(i), st, ropes, cap=cap)
        w = r0[f"decode_logits_{i}"]
        d = np.abs(np.asarray(lg) - w).max() / np.abs(w).max()
        worst = max(worst, d)
        ok = (np.asarray(lg).argmax(-1) == w.argmax(-1)).all()
        if not ok or i in (64, 71):
            print(f"  decode {i} rel={d:.2e} argmax_match={bool(ok)}")
print(f"worst relative diff = {worst:.2e}")
