"""Cost of the n-gram reduce on TPU with 32-bit lanes only.

A 64-bit value must be carried as limbs. `x mod p` (x < 2^63, p ~ 2e7 < 2^25)
then needs progressive reduction: acc = (acc << k | chunk) % p with k small
enough that acc*2^k + chunk stays inside int32. acc < p < 2^25 forces k <= 4,
so a 64-bit value takes 16 nibble steps, each with an int32 modulo.

This times that shape against the numpy baseline it would have to beat.
"""

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

T, HEADS = 8192, 16
NIBBLES = 16


def kernel(x_ref, p_ref, o_ref):
    # x_ref: [T, HEADS] int32 nibble-packed stand-in; p_ref: [1, HEADS] primes
    acc = jnp.zeros_like(x_ref[...])
    p = p_ref[...]
    for i in range(NIBBLES):
        nib = (x_ref[...] >> (4 * (i % 8))) & 0xF
        acc = (acc * 16 + nib) % p
    o_ref[...] = acc


def bench(fn, *a, reps=20, warm=3):
    for _ in range(warm):
        jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*a))
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


rng = np.random.default_rng(0)
x = jnp.asarray(rng.integers(0, 2**31 - 1, (T, HEADS), np.int32))
p = jnp.asarray(np.array([[20000003 + 2 * i for i in range(HEADS)]], np.int32))

f = jax.jit(pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((T, HEADS), jnp.int32)))
try:
    r = jax.block_until_ready(f(x, p))
    print(f"pallas {NIBBLES}-nibble reduce  [{T}, {HEADS}] : {bench(f, x, p):.3f} ms")
except Exception as e:
    print("pallas FAIL:", str(e).replace("\n", " ")[:200])


# XLA doing the same thing, as a sanity check on the op count
def xla_reduce(x, p):
    acc = jnp.zeros_like(x)
    for i in range(NIBBLES):
        acc = (acc * 16 + ((x >> (4 * (i % 8))) & 0xF)) % p
    return acc


g = jax.jit(xla_reduce)
print(f"xla    {NIBBLES}-nibble reduce  [{T}, {HEADS}] : {bench(g, x, p):.3f} ms")

# what it has to beat: the whole host hash at T=8192
print(f"host numpy compute_ngram_ids (whole hash, T={T})  : 0.910 ms  [measured earlier]")

# and a single native int32 mod, for scale
h = jax.jit(lambda x, p: x % p)
print(f"one native i32 mod        [{T}, {HEADS}] : {bench(h, x, p):.3f} ms")
