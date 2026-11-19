"""Engram: hash ids (exact) and full-model logits vs the reference. Usage: [V41_NOQUANT=1] python test_engram.py <ref_cpu_dir>"""

import os
import sys
import time

SUFFIX = "_noquant" if os.environ.get("V41_NOQUANT") == "1" else ""
R = sys.argv[1]
sys.path.insert(0, R)
sys.path.insert(0, ".")
import engram_jax as E
import jax.numpy as jnp
import model as M
import model_ref as J
import numpy as np
import torch
from engram import EngramLayout as RefLayout
from engram import NgramHashState
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(f"{R}/../tok")
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
    engram_layer_ids=(1,),
    engram_max_ngram_size=3,
    engram_n_heads=2,
    engram_head_dim=64,
    engram_vocab_size=1000,
    engram_compressed_vocab_size=99092,
    engram_pad_id=2,
    engram_num_embeddings=(4062,),
)
margs = M.ModelArgs(**base)
a = J.Args.from_torch(margs)
ref = np.load(f"{R}/ref_outputs_engram{SUFFIX}.npz")
ids = ref["input_ids"]

# --- 1) hash ids vs the reference hasher
rl = RefLayout.from_args(margs)
rh = NgramHashState(margs, rl, tok)
token_map, vsize = E.build_compressed_token_map(tok)
layout = E.EngramLayout.make(margs.engram_layer_ids, margs.engram_num_embeddings, 3, 2, 64, 1000)
assert layout.primes == rl.primes, (layout.primes, rl.primes)
hasher = E.NgramHasher(layout, token_map, vsize, margs.engram_pad_id, 2, 256)
ok = True
for sp, chunk in [(0, ids[:, :64])] + [(i, ids[:, i : i + 1]) for i in range(64, 72)]:
    want = rh(torch.from_numpy(chunk), sp).numpy()
    got = hasher(chunk, sp)
    ok &= bool((want == got).all())
print("hash ids identical to reference on prefill + 8 decode steps:", ok)

# --- 2) full model
params = {
    k: jnp.asarray(v.float().numpy()) for k, v in torch.load(f"{R}/ref_weights_engram.pt").items()
}
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
states = J.init_state(a, 2)
hasher2 = E.NgramHasher(layout, token_map, vsize, margs.engram_pad_id, 2, 256)
eng = {"hasher": hasher2}
x = jnp.asarray(ids)


def cmp(name, got, want):
    d = np.abs(np.asarray(got) - want)
    print(
        f"{name:<12} max|d|={d.max():.3e} mean|d|={d.mean():.3e} rel={d.max()/np.abs(want).max():.2e} argmax_match={(np.asarray(got).argmax(-1)==want.argmax(-1)).all()}"
    )


lg, states = J.forward(params, a, x[:, :64], 0, states, ropes, eng)
cmp("prefill", lg, ref["prefill_logits"])
for i in range(64, 72):
    lg, states = J.forward(params, a, x[:, i : i + 1], i, states, ropes, eng)
    cmp(f"decode {i}", lg, ref[f"decode_logits_{i}"])
