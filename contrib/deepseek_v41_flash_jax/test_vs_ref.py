"""Compare the JAX port with DeepSeek's reference on CPU. Usage: python test_vs_ref.py <ref_cpu_dir>"""

import os
import sys
import time

SUFFIX = "_noquant" if os.environ.get("V41_NOQUANT") == "1" else ""
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
import model as M  # the reference (ref_cpu)
import model_ref as J  # our port

torch.manual_seed(0)
ref = np.load(f"{sys.argv[1]}/ref_outputs{SUFFIX}.npz")
sd = torch.load(f"{sys.argv[1]}/ref_weights.pt")
params = {k: jnp.asarray(v.float().numpy()) for k, v in sd.items()}
args = M.ModelArgs(
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
a = J.Args.from_torch(args)
x = jnp.asarray(ref["input_ids"])
ropes = [J.make_rope(a, l) for l in range(a.n_layers)]
states = J.init_state(a, x.shape[0])


def cmp(name, got, want):
    got, want = np.asarray(got), np.asarray(want)
    diff = np.abs(got - want)
    same_top = (got.argmax(-1) == want.argmax(-1)).all()
    print(
        f"{name:<14} max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} rel={diff.max()/np.abs(want).max():.2e} argmax_match={same_top}"
    )
    return diff.max()


t = time.time()
logits, states = J.forward(params, a, x[:, :64], 0, states, ropes)
worst = cmp("prefill", logits, ref["prefill_logits"])
for i in range(64, 72):
    logits, states = J.forward(params, a, x[:, i : i + 1], i, states, ropes)
    worst = max(worst, cmp(f"decode {i}", logits, ref[f"decode_logits_{i}"]))
print(f"done in {time.time()-t:.1f}s; worst max|d| = {worst:.3e}")
