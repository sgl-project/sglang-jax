"""Unified chunk step WITH host Engram rows vs the reference (no-quant). Usage: V41_NOQUANT=1 python test_engine_engram.py <ref_cpu_dir>"""

import os
import sys

R = sys.argv[1]
sys.path.insert(0, R)
sys.path.insert(0, ".")
import engine as EN
import engram_jax as E
import jax
import jax.numpy as jnp
import model as M
import model_ref as J
import numpy as np
import torch
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
a = J.Args.from_torch(M.ModelArgs(**base))
sd = torch.load(f"{R}/ref_weights_engram.pt")
params = {k: jnp.asarray(v.float().numpy()) for k, v in sd.items()}
ref = np.load(f"{R}/ref_outputs_engram_noquant.npz")
ids = ref["input_ids"]
tmap, vs = E.build_compressed_token_map(tok)
layout = E.EngramLayout.make(a.engram_layer_ids, (4062,), 3, 2, 64, 1000)
hasher = E.NgramHasher(layout, tmap, vs, 2, 2, 256)
table_w, table_s = np.asarray(params["layers.1.engram.embed.weight"]), np.asarray(
    params["layers.1.engram.embed.scale"]
)
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn"))


def rows_for(chunk, pos):
    hid = hasher(chunk, pos)  # [B,S,1,ncols]
    return {1: jnp.asarray(E.gather_rows(table_w, table_s, hid[:, :, 0, :]))}


st = EN.init_states(a, 2)
pos = 0
for S in (32, 32):
    lg, st = step(
        params,
        a,
        jnp.asarray(ids[:, pos : pos + S]),
        jnp.int32(pos),
        st,
        ropes,
        rows_for(ids[:, pos : pos + S], pos),
    )
    pos += S
d = np.abs(np.asarray(lg) - ref["prefill_logits"])
print(f"prefill (2 chunks)  rel={d.max()/np.abs(ref['prefill_logits']).max():.2e}")
for i in range(64, 72):
    lg, st = step(
        params,
        a,
        jnp.asarray(ids[:, i : i + 1]),
        jnp.int32(i),
        st,
        ropes,
        rows_for(ids[:, i : i + 1], i),
    )
    d = np.abs(np.asarray(lg) - ref[f"decode_logits_{i}"])
    print(
        f"decode {i}  rel={d.max()/np.abs(ref[f'decode_logits_{i}']).max():.2e} argmax_match={(np.asarray(lg).argmax(-1)==ref[f'decode_logits_{i}'].argmax(-1)).all()}"
    )
