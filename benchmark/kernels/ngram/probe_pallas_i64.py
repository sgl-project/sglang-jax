"""Probe native int64 at the Pallas boundary and inside an int32-only call.

Run both with and without --x64. Compilation alone is NOT success: every
compiled result is compared with NumPy's wide-integer calculation.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

N = 256
MULT_HI, MULT_LO = 0x0003_5A4E, 0x1B2C_9D71
MULT = np.int64((MULT_HI << 32) | MULT_LO)
MULT1 = np.int64(0x0001_2345_6789_ABCD)
PRIME = 20_000_003


def probe(name, operation, oracle, *, dtype=jnp.int64, high_bits=False):
    host = np.arange(N, dtype=np.int64) + ((1 << 32) if high_bits else 0)
    expected = oracle(host)

    def kernel(x_ref, out_ref):
        out_ref[...] = operation(x_ref[...])

    try:
        x = jnp.asarray(host, dtype=dtype)
        call = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((N,), dtype))
        got = np.asarray(jax.block_until_ready(jax.jit(call)(x)))
    except Exception as exc:
        message = str(exc).replace("\n", " ")[:160]
        print(f"  {name:<32} FAIL {type(exc).__name__}: {message}")
        return
    matches = np.array_equal(got, expected)
    print(f"  {name:<32} compiled ({got.dtype}); numeric match: {'YES' if matches else 'NO'}")
    if not matches:
        print(f"    got  {got[:5]}\n    want {expected[:5]}")


def wide_multiplier():
    # Limb construction also traces with x64 disabled, exposing silent narrowing.
    return (jnp.int64(MULT_HI) << jnp.int64(32)) | jnp.int64(MULT_LO)


def product(t):
    return t.astype(jnp.int64) * wide_multiplier()


def xor_products(t):
    return product(t) ^ product(t + 1)


def full_hash(t):
    m1 = (jnp.int64(0x0001_2345) << jnp.int64(32)) | jnp.int64(0x6789_ABCD)
    return (product(t) ^ ((t.astype(jnp.int64) + 7) * m1)) % jnp.int64(PRIME)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x64", action="store_true", help="enable JAX's native 64-bit types")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    print(f"jax {jax.__version__}  x64={jax.config.jax_enable_x64}")

    print("\nint64 crossing the Pallas boundary:")
    probe("copy high bits", lambda t: t, lambda t: t, high_bits=True)
    probe("mul by i64 scalar", product, lambda t: t * MULT)
    probe("xor high bits", lambda t: t ^ jnp.int64(12345), lambda t: t ^ 12345, high_bits=True)
    probe("mod by i64 prime", lambda t: t % jnp.int64(PRIME), lambda t: t % PRIME, high_bits=True)
    probe("mul then mod", lambda t: product(t) % jnp.int64(PRIME), lambda t: (t * MULT) % PRIME)

    print("\nint32 boundary, int64 intermediates:")
    cases = (
        ("widen and back", lambda t: t.astype(jnp.int64), lambda t: t),
        (
            "i32 * i64 multiplier",
            lambda t: product(t) & jnp.int64(0x7FFFFFFF),
            lambda t: (t * MULT) & 0x7FFFFFFF,
        ),
        (
            "xor of i64 products",
            lambda t: xor_products(t) & jnp.int64(0x7FFFFFFF),
            lambda t: ((t * MULT) ^ ((t + 1) * MULT)) & 0x7FFFFFFF,
        ),
        ("product % prime", lambda t: product(t) % jnp.int64(PRIME), lambda t: (t * MULT) % PRIME),
        (
            "mul, xor, mod (hash term)",
            full_hash,
            lambda t: ((t * MULT) ^ ((t + 7) * MULT1)) % PRIME,
        ),
    )
    for name, operation, oracle in cases:
        probe(name, lambda t, op=operation: op(t).astype(jnp.int32), oracle, dtype=jnp.int32)

    print("\n32-bit baselines and uint64 intermediate:")
    probe("i32 mul", lambda t: t * 7, lambda t: t * 7, dtype=jnp.int32)
    probe("i32 mod", lambda t: t % 97, lambda t: t % 97, dtype=jnp.int32)
    probe(
        "u32 mul_hi via uint64",
        lambda t: ((t.astype(jnp.uint64) * jnp.uint64(2654435761)) >> jnp.uint64(32)).astype(
            jnp.uint32
        ),
        lambda t: ((t.astype(np.uint64) * np.uint64(2654435761)) >> np.uint64(32)).astype(
            np.uint32
        ),
        dtype=jnp.uint32,
    )


if __name__ == "__main__":
    main()
