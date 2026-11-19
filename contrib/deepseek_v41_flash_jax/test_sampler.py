"""Batched sampler: greedy rows, top-k / top-p restriction, exactness of the full-vocabulary path, and the all-greedy fast path."""

import sys

sys.path.insert(0, ".")
import generate as G
import jax
import jax.numpy as jnp
import numpy as np

f = jax.jit(G._sample_rows)
rng = np.random.default_rng(0)
V = 256
logits = jnp.asarray(rng.normal(size=(8, V)).astype(np.float32) * 3.0)
ones, zeros_i = jnp.ones(8, jnp.float32), jnp.zeros(8, jnp.int32)
ok = True


def check(name, c):
    global ok
    ok &= bool(c)
    print(f"  {'PASS' if c else 'FAIL'}  {name}")


am = np.asarray(jnp.argmax(logits, -1))
t = np.asarray(f(logits, zeros_i, zeros_i, jnp.zeros(8), ones, zeros_i))
check("temperature 0 == argmax", (t == am).all())
t = np.asarray(f(logits, zeros_i + 3, zeros_i, jnp.full(8, 0.7), ones, jnp.full(8, 1, jnp.int32)))
check("top_k=1 == argmax", (t == am).all())
t = np.asarray(
    f(logits, zeros_i + 3, zeros_i, jnp.full(8, 1.0), jnp.full(8, 1e-4, jnp.float32), zeros_i)
)
check("tiny top_p == argmax", (t == am).all())
top5 = np.argsort(-np.asarray(logits), -1)[:, :5]
inside = True
for seed in range(60):
    t = np.asarray(
        f(logits, zeros_i + seed, zeros_i, jnp.full(8, 1.5), ones, jnp.full(8, 5, jnp.int32))
    )
    inside &= all(t[i] in top5[i] for i in range(8))
check("top_k=5 samples stay in the top-5 (60 seeds x 8 rows)", inside)
# full path: empirical frequencies vs softmax on a 4-way peaked row
lg = jnp.tile(
    jnp.asarray(np.r_[2.0, 1.0, 0.0, -1.0, np.full(V - 4, -30.0)].astype(np.float32))[None],
    (4096, 1),
)
t = np.asarray(
    f(
        lg,
        jnp.arange(4096, dtype=jnp.int32),
        zeros_i.at[:].set(0)[:1].repeat(4096),
        jnp.ones(4096),
        jnp.ones(4096),
        jnp.zeros(4096, jnp.int32),
    )
)
want = np.exp([2, 1, 0, -1]) / np.exp([2, 1, 0, -1]).sum()
got = np.bincount(t, minlength=V)[:4] / 4096
check(
    f"full path matches softmax within 3 sigma ({np.round(got,3)} vs {np.round(want,3)})",
    (np.abs(got - want) < 3 * np.sqrt(want * (1 - want) / 4096) + 1e-3).all(),
)
# restricted path: top_p=0.8 on the same distribution keeps tokens until cumulative mass >= 0.8 -> {0,1} (0.644, +0.237 = 0.881)
t = np.asarray(
    f(
        lg,
        jnp.arange(4096, dtype=jnp.int32),
        zeros_i[:1].repeat(4096),
        jnp.ones(4096),
        jnp.full(4096, 0.8, jnp.float32),
        jnp.zeros(4096, jnp.int32),
    )
)
p0 = (t == 0).mean()
check(
    f"top_p=0.8 only emits tokens 0/1 and p(token0)~{0.644/0.881:.3f} (got {p0:.3f})",
    set(np.unique(t)) <= {0, 1} and abs(p0 - 0.644 / 0.881) < 0.03,
)
print("ALL PASS" if ok else "SOME FAILED")
