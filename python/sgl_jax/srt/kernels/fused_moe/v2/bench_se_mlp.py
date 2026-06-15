"""External shared-expert dense MLP cost (per-device, EP=32 layout).

Answers: is in-kernel SE (+81us) faster than SE as a separate dense MLP?
Times the fp8 per-channel SE FFN at the same per-device workload (ntok/nd local
tokens, hidden d, inter se_inter). Runs REPS copies inside one jit to amortize
dispatch, wall / REPS ~= per-call device time.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:.1f}s][p{jax.process_index()}] {m}", flush=True)


if os.environ.get("BENCH_SINGLE_HOST", "0") != "1":
    jax.distributed.initialize()
P = jax.sharding.PartitionSpec
nd = jax.device_count()
mesh = jax.sharding.Mesh(jax.devices(), ("x",))
repl = jax.sharding.NamedSharding(mesh, P())
shard = jax.sharding.NamedSharding(mesh, P("x"))

d = int(os.environ.get("BENCH_D", "8192"))
se_inter = int(os.environ.get("BENCH_SE_INTER", "2048"))
ntok = int(os.environ.get("BENCH_TOKENS", "16384"))
use_fp8 = os.environ.get("BENCH_FP8", "1") == "1"
REPS = int(os.environ.get("BENCH_REPS", "50"))
warmup, iters = 3, 10


def repl_put(x):
    return jax.make_array_from_single_device_arrays(
        x.shape, repl, [jax.device_put(np.asarray(x), dv) for dv in jax.local_devices()]
    )


key = jax.random.key(0)
k1, k2, k3, kt = jax.random.split(key, 4)


def q_pc(k, shp):
    w = jax.random.normal(k, shp, jnp.float32) * 0.01
    amax = jnp.max(jnp.abs(w), 0, keepdims=True)
    sc = jnp.maximum(amax / 448.0, 1e-12)
    return np.asarray((w / sc).astype(jnp.float8_e4m3fn)), np.asarray(sc)


w1q, s1 = q_pc(k1, (d, se_inter))
w3q, s3 = q_pc(k2, (d, se_inter))
w2q, s2 = q_pc(k3, (se_inter, d))
w1 = repl_put(w1q)
w3 = repl_put(w3q)
w2 = repl_put(w2q)
s1 = repl_put(s1)
s3 = repl_put(s3)
s2 = repl_put(s2)

tok_local = ntok // nd
toks = jax.make_array_from_single_device_arrays(
    (ntok, d),
    shard,
    [
        jax.device_put(
            (jax.random.normal(jax.random.fold_in(kt, i), (tok_local, d), jnp.bfloat16)), dv
        )
        for i, dv in enumerate(jax.local_devices())
    ],
)


@jax.jit
@jax.shard_map(
    mesh=mesh, in_specs=(P("x"), P(), P(), P(), P(), P(), P()), out_specs=P("x"), check_vma=False
)
def body(x, w1, w3, w2, s1, s3, s2):
    def one(xf):
        if os.environ.get("BENCH_TRUE_BF16", "0") == "1":
            xb = xf.astype(jnp.bfloat16)
            w1b = (w1.astype(jnp.float32) * s1).astype(jnp.bfloat16)
            w3b = (w3.astype(jnp.float32) * s3).astype(jnp.bfloat16)
            w2b = (w2.astype(jnp.float32) * s2).astype(jnp.bfloat16)
            g = jnp.dot(xb, w1b, preferred_element_type=jnp.float32)
            u = jnp.dot(xb, w3b, preferred_element_type=jnp.float32)
            a = (jax.nn.silu(g) * u).astype(jnp.bfloat16)
            return jnp.dot(a, w2b, preferred_element_type=jnp.float32)
        g = (xf @ w1.astype(jnp.float32)) * s1
        u = (xf @ w3.astype(jnp.float32)) * s3
        a = jax.nn.silu(g) * u
        return (a @ w2.astype(jnp.float32)) * s2

    xf = x.astype(jnp.float32)
    # Chain so each FFN depends on the previous output -> XLA can't CSE them.
    for _ in range(REPS):
        xf = one(xf)
    return xf.astype(jnp.bfloat16)


def run():
    return body(toks, w1, w3, w2, s1, s3, s2)


for _ in range(warmup):
    jax.block_until_ready(run())
ts = []
for _ in range(iters):
    a = time.monotonic()
    jax.block_until_ready(run())
    ts.append((time.monotonic() - a) * 1e3)
if jax.process_index() == 0:
    per = np.mean(ts) / REPS
    log(
        f"external SE MLP fp8: {per*1000:.1f} us/call  (tok_local={tok_local}, d={d}, se_inter={se_inter}, REPS={REPS}, wall_total={np.mean(ts):.2f}ms)"
    )
log("done")
