"""Prefix cache: requests sharing a prompt prefix (and a multi-turn style extension) must give EXACTLY the single-request greedy tokens with
the cache on, while skipping the cached part of the prefill. Usage: python test_prefix.py <ref_cpu_dir> <synth_ckpt>"""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["V41_NOQUANT"] = "1"
os.environ.update(
    V41_PREFIX_CACHE="1",
    V41_PREFIX_GRANULE="32",
    V41_PREFIX_MAX="128",
    V41_PREFIX_SLOTS="8",
    V41_INDEX_TILE="16",
)
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
params = L.build_params(
    L.SafetensorsDir(CK),
    C.param_table(a, act=jnp.float32),
    mesh,
    SH.param_spec,
    a.n_routed_experts,
    threads=4,
    log=lambda *x: None,
)
gen = G.Generator(params, a, mesh, batch=4, sizes=(32, 8, 2), eos_id=-5)
ids = np.load(f"{CK}/ref_outputs_deq_noquant.npz")["input_ids"]
X = ids[0, :70].tolist()
prompts = {
    "A": X + ids[1, :10].tolist(),
    "B": X + ids[1, 10:20].tolist(),
    "C": X + ids[1, :10].tolist() + ids[0, 70:75].tolist(),
    "D": ids[1, 30:60].tolist(),
}
want = {
    k: gen.generate(v, 8) for k, v in prompts.items()
}  # no cache: Generator.generate does not use the scheduler's cache
sch = G.BatchScheduler(gen)
gen.reset()
assert sch.pcache is not None
got = {}
for k in ("A", "B", "C", "D", "B"):
    sch.add(k + str(len(got)), prompts[k], 8)
    while sch.busy():
        for rid, out, st in sch.step():
            got[rid] = out
ok = True
for rid, out in got.items():
    same = out == want[rid[0]]
    ok &= same
    print(f"request {rid}: tokens identical to no-cache run: {same}")
pc = sch.pcache
print(
    f"cache: {len(pc.map)} snapshots, lengths {sorted(pc.lens)}, hits {pc.hits}, misses {pc.misses}, prompt tokens saved {pc.saved_tokens}"
)
ok &= pc.hits >= 3 and pc.saved_tokens > 0
print("ALL PASS" if ok else "SOME FAILED")
