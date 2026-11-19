"""Per-row positions + active mask vs the reference: rows run staggered (one prefills while the other decodes alone, then
both decode at DIFFERENT positions, then the second continues alone). Every row's logits must match the reference run
of that row. Usage: V41_NOQUANT=1 python test_perrow.py <ref_cpu_dir>"""

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
ids = np.asarray(r0["input_ids"])
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn"))
st = EN.init_states(a, 2)
worst = [0.0]


def cmp(tag, got, want):
    d = np.abs(np.asarray(got) - want).max() / np.abs(want).max()
    ok = int(np.asarray(got).argmax()) == int(want.argmax())
    worst[0] = max(worst[0], d)
    print(f"  {tag:<34} rel={d:.2e} argmax_match={ok}")


def call(tok_rows, pos, active):
    global st
    lg, st = step(
        params,
        a,
        jnp.asarray(np.stack(tok_rows).astype(np.int32)),
        jnp.asarray(pos, jnp.int32),
        st,
        ropes,
        active=jnp.asarray(active),
    )
    return np.asarray(lg)


junk = lambda S: np.full(S, 7, np.int64)
# row 0 prefill (row 1 idle, junk tokens, must not be written)
for p in (0, 32):
    lg = call([ids[0, p : p + 32], junk(32)], [p, 0], [True, False])
cmp("row0 prefill", lg[0], r0["prefill_logits"][0])
# row 0 decodes alone 64..67
for i in range(64, 68):
    lg = call([ids[0, i : i + 1], junk(1)], [i, 0], [True, False])
    cmp(f"row0 decode {i} (alone)", lg[0], r0[f"decode_logits_{i}"][0])
# row 1 prefill as [32,16,8,8] while row 0 holds
for p, S in ((0, 32), (32, 16), (48, 8), (56, 8)):
    lg = call([junk(S), ids[1, p : p + S]], [68, p], [False, True])
cmp("row1 prefill", lg[1], r0["prefill_logits"][1])
# both decode at different positions: row0 68..71, row1 64..67
for k in range(4):
    i0, i1 = 68 + k, 64 + k
    lg = call([ids[0, i0 : i0 + 1], ids[1, i1 : i1 + 1]], [i0, i1], [True, True])
    cmp(f"row0 decode {i0} (joint)", lg[0], r0[f"decode_logits_{i0}"][0])
    cmp(f"row1 decode {i1} (joint)", lg[1], r0[f"decode_logits_{i1}"][1])
# row 1 continues alone 68..71
for i in range(68, 72):
    lg = call([junk(1), ids[1, i : i + 1]], [72, i], [False, True])
    cmp(f"row1 decode {i} (alone)", lg[1], r0[f"decode_logits_{i}"][1])
print(f"\nworst relative diff = {worst[0]:.2e}")
