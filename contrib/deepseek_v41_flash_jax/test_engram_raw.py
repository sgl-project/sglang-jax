"""Device-side Engram dequantization (uint8 bytes + e8m0 exponents) must be bit-identical to the host float32 path, for every possible byte."""

import sys

sys.path.insert(0, ".")
import engram_jax as E
import jax
import jax.numpy as jnp
import loader as L
import ml_dtypes
import numpy as np

rng = np.random.default_rng(0)
R, D = 4096, 256
w8 = np.tile(np.arange(256, dtype=np.uint8), (R * D // 256 + 1,))[: R * D].reshape(
    R, D
)  # every fp8 byte value appears (incl. NaN codes)
w8 = w8[rng.permutation(R)]
s8 = rng.integers(30, 231, (R, D // 32), dtype=np.uint8)
s8[:4] = [
    [30] * 8,
    [31] * 8,
    [230] * 8,
    [127] * 8,
]  # exponent edge cases (real tables use bytes ~117-122; bytes < ~20 would give float32 denormal products that the device flushes)


class T(L.EngramTable):
    def __init__(self):
        self.w = w8.view(ml_dtypes.float8_e4m3fn)
        self.s = s8.view(ml_dtypes.float8_e8m0fnu)
        self.block = 32
        self.w8 = w8
        self.s8 = s8
        self.lut = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        self.pow2 = np.ldexp(np.float32(1.0), np.arange(256) - 127).astype(np.float32)


t = T()
ids = (
    np.concatenate([np.arange(8), rng.integers(0, R, (2, 3, 16)).ravel()])
    .reshape(-1, 1)[:8]
    .reshape(8)
)
ids = rng.integers(0, R, (4, 5, 16))
ids[0, 0, :4] = [0, 1, 2, 3]
host = E.gather_rows(t.w, t.s, ids)  # reference path (astype + ldexp semantics)
a, b = t.raw(ids)
dev = np.asarray(jax.jit(E.dequant_rows)(jnp.asarray(a), jnp.asarray(b)))
same = np.array_equal(host, dev, equal_nan=True)
print(
    "device dequant == host dequant (bit-identical incl. NaN codes and exponent edge cases):",
    bool(same),
    "| shapes",
    a.shape,
    b.shape,
    dev.shape,
)
bad = np.argwhere(~((host == dev) | (np.isnan(host) & np.isnan(dev))))
print("mismatches:", len(bad), bad[:3].tolist())
print("ALL PASS" if same else "FAILED")
