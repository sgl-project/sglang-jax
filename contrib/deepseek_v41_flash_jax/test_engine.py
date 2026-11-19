"""Unified chunk step (jit, traced position) vs the reference. Usage: [V41_NOQUANT=1] python test_engine.py <ref_cpu_dir>"""

import os
import sys
import time

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
rl = np.load(f"{R}/ref_outputs_len{SUFFIX}.npz")
ids = jnp.asarray(r0["input_ids"])
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn"))


def run(chunks, dec_from, n_dec):
    st = EN.init_states(a, 2)
    pos = 0
    lg = None
    for S in chunks:
        lg, st = step(params, a, ids[:, pos : pos + S], jnp.int32(pos), st, ropes)
        pos += S
    out = [lg]
    for i in range(dec_from, dec_from + n_dec):
        lg, st = step(params, a, ids[:, i : i + 1], jnp.int32(i), st, ropes)
        out.append(lg)
    return out


def cmp(tag, got, want):
    d = np.abs(np.asarray(got) - want)
    same = (np.asarray(got).argmax(-1) == want.argmax(-1)).all()
    print(
        f"  {tag:<28} max|d|={d.max():.2e} rel={d.max()/np.abs(want).max():.2e} argmax_match={bool(same)}"
    )
    return d.max() / np.abs(want).max()


worst = 0
t = time.time()
for name, chunks in [
    ("64 as [64]", [64]),
    ("64 as [32,32]", [32, 32]),
    ("64 as [16]*4", [16] * 4),
    ("64 as [8]*8", [8] * 8),
]:
    print(name)
    o = run(chunks, 64, 8)
    worst = max(worst, cmp("prefill", o[0], r0["prefill_logits"]))
    for k, i in enumerate(range(64, 72)):
        worst = max(worst, cmp(f"decode {i}", o[k + 1], r0[f"decode_logits_{i}"]))
for L, chunks in [(77, [64, 8, 4, 1]), (50, [32, 16, 2]), (77, [32, 32, 8, 4, 1])]:
    print(f"L={L} as {chunks}")
    o = run(chunks, L, 3)
    worst = max(worst, cmp(f"prefill_{L}", o[0], rl[f"prefill_{L}"]))
    for k, i in enumerate(range(L, L + 3)):
        worst = max(worst, cmp(f"decode {i}", o[k + 1], rl[f"decode_{L}_{i}"]))
print(f"\nworst relative diff = {worst:.2e}   ({time.time()-t:.0f}s)")
