"""Padded chunk (per-row nvalid) vs the reference: one prefill call with S=128 carrying a 76-token row (a 77-token prompt whose last token goes
through the decode path) and a 50-token row, then lockstep decode at different positions. Usage: V41_NOQUANT=1 python test_padded.py <ref_cpu_dir>"""

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
rl = np.load(f"{R}/ref_outputs_len{SUFFIX}.npz")
ids = np.asarray(np.load(f"{R}/ref_outputs{SUFFIX}.npz")["input_ids"])
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
step = jax.jit(EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn", "cap"))
st = EN.init_states(a, 2)
worst = 0.0


def cmp(tag, got, want):
    global worst
    d = np.abs(np.asarray(got) - want).max() / np.abs(want).max()
    ok = int(np.asarray(got).argmax()) == int(want.argmax())
    worst = max(worst, d)
    print(f"  {tag:<38} rel={d:.2e} argmax_match={ok}")


S = 128
toks = np.zeros((2, S), np.int32)
toks[0, :76] = ids[0, :76]
toks[1, :50] = ids[1, :50]
lg, st = step(
    params,
    a,
    jnp.asarray(toks),
    jnp.asarray([0, 0], jnp.int32),
    st,
    ropes,
    active=jnp.asarray([True, True]),
    nvalid=jnp.asarray([76, 50], jnp.int32),
)
cmp("row1 padded prefill (50 of 128)", lg[1], rl["prefill_50"][1])
steps = [((76, 50), "A"), ((77, 51), "B"), ((78, 52), "C"), ((79, 53), "D")]
for (p0, p1), tag in steps:
    lg, st = step(
        params,
        a,
        jnp.asarray([[ids[0, p0]], [ids[1, p1]]], jnp.int32),
        jnp.asarray([p0, p1], jnp.int32),
        st,
        ropes,
    )
    lg = np.asarray(lg)
    w0 = rl["prefill_77"][0] if p0 == 76 else rl.get(f"decode_77_{p0}")
    w1 = rl.get(f"decode_50_{p1}")
    if w0 is not None:
        cmp(f"row0 pos {p0} (odd-tail / decode)", lg[0], w0[0] if w0.ndim > 1 else w0)
    if w1 is not None:
        cmp(f"row1 pos {p1} (decode)", lg[1], w1[1] if w1.ndim > 1 else w1)
print(f"worst relative diff = {worst:.2e}")
