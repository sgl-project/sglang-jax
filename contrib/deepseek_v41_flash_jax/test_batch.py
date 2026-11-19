"""BatchScheduler: several requests of different lengths through B=2 rows (incl. row reuse) must give exactly the tokens
of single-request greedy generation. Also checks the per-row Engram hasher against per-row scalar calls.
Usage: python test_batch.py <ref_cpu_dir> <synth_ckpt>"""

import os
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["V41_NOQUANT"] = "1"
R, CK = sys.argv[1], sys.argv[2]
sys.path.insert(0, R)
sys.path.insert(0, ".")
import config_real as C
import engram_jax as E
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
gen = G.Generator(
    params, a, mesh, batch=int(os.environ.get("ROWS", "2")), sizes=(32, 8, 2), eos_id=-5
)
ids = np.load(f"{CK}/ref_outputs_deq_noquant.npz")["input_ids"]
reqs = [
    (0, ids[0, :37].tolist(), 10),
    (1, ids[1, :20].tolist(), 14),
    (2, ids[0, 5:56].tolist(), 6),
    (3, ids[1, 3:12].tolist(), 9),
    (4, ids[0, 9:30].tolist(), 7),
    (5, ids[1, 1:45].tolist(), 11),
    (6, ids[0, 2:5].tolist(), 5),
]
single = {rid: gen.generate(p, n) for rid, p, n in reqs}
sch = G.BatchScheduler(gen, ladder=[64, 128])
gen.reset()
print("ladder", sch.ladder)
for rid, p, n in reqs[:3] + reqs[4:]:
    sch.add(rid, p, n)
got, rounds = {}, 0
while sch.busy():
    for rid, out, st in sch.step():
        got[rid] = out
    rounds += 1
    if rounds == 3:
        sch.add(*reqs[3][:1], reqs[3][1], reqs[3][2])  # arrives mid-flight
ok = all(got[r] == single[r] for r in single)
for r in single:
    print(
        f"req {r}: batch {got[r]}\n       single {single[r]}  {'OK' if got[r] == single[r] else 'MISMATCH'}"
    )
print(
    "rows",
    sch.B,
    "buckets(per-dp)",
    sch.buckets,
    "single-row prefill:",
    sch.single,
    "| ALL IDENTICAL:",
    ok,
    "| rounds",
    rounds,
    "| device steps",
    sch.steps,
)

# per-row hasher vs per-row scalar calls
lay = E.EngramLayout.make((1,), (4062,), 3, 2, 8, 5000)
tmap = np.arange(5000) % 3000
h1 = E.NgramHasher(lay, tmap, 3000, 2, 2, 64)
hs = [E.NgramHasher(lay, tmap, 3000, 2, 1, 64) for _ in range(2)]
rng = np.random.default_rng(0)
toks = rng.integers(0, 5000, (2, 40))
h1(np.stack([toks[0, :8], toks[1, :8]]), [0, 0], [True, False])
hs[0](toks[0:1, :8], 0)  # row 0 prefill alone
h1(np.stack([toks[0, :8], toks[1, :8]]), [0, 0], [False, True])
hs[1](toks[1:2, :8], 0)  # row 1 prefill alone
good = True
for k in range(6):
    a_ = h1(
        np.stack([toks[0, 8 + k : 9 + k], toks[1, 8 + k : 9 + k]]), [8 + k, 8 + k], [True, True]
    )
    good &= np.array_equal(a_[0], hs[0](toks[0:1, 8 + k : 9 + k], 8 + k)[0]) and np.array_equal(
        a_[1], hs[1](toks[1:2, 8 + k : 9 + k], 8 + k)[0]
    )
print("hasher per-row == scalar per-row:", bool(good))
