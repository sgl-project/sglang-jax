"""Generator (chunked prefill + decode + sampling) vs reference greedy decoding, tiny model, synthetic checkpoint, 8 virtual CPU devices."""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["V41_NOQUANT"] = "1"
R, CK = sys.argv[1], sys.argv[2]
sys.path.insert(0, R)
sys.path.insert(0, ".")
import config_real as C
import generate as G
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
margs = M.ModelArgs(**base)
a = J.Args.from_torch(margs)
mesh = SH.make_mesh(jax.devices(), 2, 4)
params = L.build_params(
    L.SafetensorsDir(CK),
    C.param_table(a, act=jnp.float32),
    mesh,
    SH.param_spec,
    a.n_routed_experts,
    threads=4,
    log=lambda *x: None,
)
gen = G.Generator(
    params, a, mesh, batch=2, sizes=(32, 8, 2), eos_id=-5
)  # no EOS stop: compare a fixed number of tokens
print("decompose(37) =", G.decompose(37, (32, 8, 2)))
prompt = np.load(f"{CK}/ref_outputs_deq_noquant.npz")["input_ids"][0, :37].tolist()
got = gen.generate(prompt, 12, temperature=0.0)
# reference greedy
deq = torch.load(f"{CK}/ref_weights_deq.pt")
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
ref = M.Transformer(margs)
ref.float()
ref.load_state_dict(deq)
ref.eval()
x = torch.tensor([prompt, prompt])
want = []
with torch.no_grad():
    _, lg, _ = ref(x)
    for i in range(12):
        t = int(lg[0].argmax())
        want.append(t)
        _, lg, _ = ref(torch.tensor([[t], [t]]), len(prompt) + i)
print("generator :", got)
print("reference :", want)
print(
    "IDENTICAL:",
    got == want,
    "| stats:",
    {k: round(v, 3) if isinstance(v, float) else v for k, v in gen.last_stats.items()},
)
# sampling path runs and differs from greedy for temperature > 0
s1 = gen.generate(prompt, 6, temperature=1.0, seed=1)
print("temperature=1.0 sample:", s1)
