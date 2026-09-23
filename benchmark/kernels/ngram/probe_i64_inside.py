"""int64 INSIDE a Pallas TPU kernel, with int32 crossing the boundary.

The n-gram hash only needs the wide type for intermediates: token ids in are
int32, row ids out are int32 (< 2^31), and the 64-bit value lives entirely
inside. That is a different question from passing int64 through pallas_call.
"""

import os
import sys

if "--x64" in sys.argv:
    os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

print(f"jax {jax.__version__}  x64={jax.config.jax_enable_x64}")
N = 256
I64 = jnp.int64
MULT_HI, MULT_LO = 0x0003_5A4E, 0x1B2C_9D71  # multiplier as two 32-bit limbs
PRIME = 20000003


def probe(name, body):
    x = jnp.arange(N, dtype=jnp.int32)
    try:
        f = pl.pallas_call(body, out_shape=jax.ShapeDtypeStruct((N,), jnp.int32))
        r = np.asarray(jax.block_until_ready(jax.jit(f)(x)))
        print(f"  {name:<34} OK    {r[:3]}")
        return r
    except Exception as e:
        print(f"  {name:<34} FAIL  {type(e).__name__}: {str(e)[:130]}")
        return None


print("\nint32 in -> int32 out, int64 intermediates:")


def k_widen(x_ref, o_ref):
    o_ref[...] = x_ref[...].astype(I64).astype(jnp.int32)


probe("widen to i64 and back", k_widen)


def k_mul(x_ref, o_ref):
    m = (jnp.int64(MULT_HI) << jnp.int64(32)) | jnp.int64(MULT_LO)
    o_ref[...] = ((x_ref[...].astype(I64) * m) & jnp.int64(0x7FFFFFFF)).astype(jnp.int32)


probe("i32 * i64 multiplier", k_mul)


def k_xor(x_ref, o_ref):
    m = (jnp.int64(MULT_HI) << jnp.int64(32)) | jnp.int64(MULT_LO)
    a = x_ref[...].astype(I64) * m
    b = (x_ref[...].astype(I64) + 1) * m
    o_ref[...] = ((a ^ b) & jnp.int64(0x7FFFFFFF)).astype(jnp.int32)


probe("xor of two i64 products", k_xor)


def k_mod(x_ref, o_ref):
    m = (jnp.int64(MULT_HI) << jnp.int64(32)) | jnp.int64(MULT_LO)
    o_ref[...] = ((x_ref[...].astype(I64) * m) % jnp.int64(PRIME)).astype(jnp.int32)


r_mod = probe("i64 product % prime  (the real op)", k_mod)


def k_full(x_ref, o_ref):
    m0 = (jnp.int64(MULT_HI) << jnp.int64(32)) | jnp.int64(MULT_LO)
    m1 = (jnp.int64(0x0001_2345) << jnp.int64(32)) | jnp.int64(0x6789_ABCD)
    t = x_ref[...].astype(I64)
    rolling = t * m0
    rolling = rolling ^ ((t + 7) * m1)
    o_ref[...] = (rolling % jnp.int64(PRIME)).astype(jnp.int32)


r_full = probe("mul, xor, mod  (one hash term)", k_full)

if r_full is not None:
    t = np.arange(N, dtype=np.int64)
    m0 = (MULT_HI << 32) | MULT_LO
    m1 = (0x00012345 << 32) | 0x6789ABCD
    want = ((t * m0) ^ ((t + 7) * m1)) % PRIME
    ok = np.array_equal(r_full, want.astype(np.int32))
    print(f"\n  numeric match vs numpy int64: {'YES' if ok else 'NO'}")
    if not ok:
        print(f"    got  {r_full[:5]}\n    want {want[:5]}")
