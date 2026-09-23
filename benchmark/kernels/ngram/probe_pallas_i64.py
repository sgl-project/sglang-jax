"""Does Pallas/Mosaic TPU support the int64 ops the n-gram hash needs?

The hash is: token * multiplier (multiplier ~6e13, product < 2^63), XOR the
terms, then % a ~2e7 prime. Every step is 64-bit. TPU lanes are 32-bit.
"""

import os

os.environ["JAX_ENABLE_X64"] = "1"
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

print("jax", jax.__version__, "x64", jax.config.jax_enable_x64)
N = 256
mult = np.int64(0x0003_5A4E_1B2C_9D71)  # ~9.5e14, a realistic multiplier
prime = np.int64(20000003)


def probe(name, kernel, dtype=jnp.int64, out_dtype=None):
    out_dtype = out_dtype or dtype
    x = jnp.arange(N, dtype=dtype)
    try:
        f = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((N,), out_dtype))
        r = jax.block_until_ready(jax.jit(f)(x))
        print(f"  {name:<28} OK    first={np.asarray(r)[:3]}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:160]
        print(f"  {name:<28} FAIL  {type(e).__name__}: {msg}")


print("\nint64 in a Pallas TPU kernel:")
probe("copy", lambda x_ref, o_ref: o_ref.__setitem__(..., x_ref[...]))
probe("mul by i64 scalar", lambda x_ref, o_ref: o_ref.__setitem__(..., x_ref[...] * mult))
probe("xor", lambda x_ref, o_ref: o_ref.__setitem__(..., x_ref[...] ^ jnp.int64(12345)))
probe("mod by i64 prime", lambda x_ref, o_ref: o_ref.__setitem__(..., x_ref[...] % prime))
probe("mul then mod", lambda x_ref, o_ref: o_ref.__setitem__(..., (x_ref[...] * mult) % prime))

print("\nint32 baseline (for contrast):")
probe(
    "i32 mul",
    lambda x_ref, o_ref: o_ref.__setitem__(..., x_ref[...] * jnp.int32(7)),
    dtype=jnp.int32,
)
probe(
    "i32 mod",
    lambda x_ref, o_ref: o_ref.__setitem__(..., x_ref[...] % jnp.int32(97)),
    dtype=jnp.int32,
)

print("\nuint32 emulation primitives (what a 64-bit fallback would need):")
probe(
    "u32 mul_hi via jnp",
    lambda x_ref, o_ref: o_ref.__setitem__(
        ...,
        ((x_ref[...].astype(jnp.uint64) * jnp.uint64(2654435761)) >> jnp.uint64(32)).astype(
            jnp.uint32
        ),
    ),
    dtype=jnp.uint32,
)
