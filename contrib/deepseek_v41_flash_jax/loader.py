"""Checkpoint loader for DeepSeek-V4.1 (HF safetensors, reference tensor names). NumPy + ml_dtypes only.

Formats (from the checkpoint / DeepSeek convert.py):
  dense linears : float8_e4m3fn [O, I] + scale float8_e8m0 [O/32, I/32]  (32x32 blocks)  -> dequantized to bf16
  routed experts: int8 [N, K/2] holding two e2m1 values per byte (LOW nibble first, sign = bit 3) + scale e8m0 [N, K/32]
                  -> kept as exact int8 values (2*e2m1 are integers) + uint8 exponent bytes; dequantized on device
  engram tables : float8_e4m3fn [R, D] + e8m0 [R, D/32]  -> memory-mapped on the host, rows gathered per step
"""

from __future__ import annotations

import json
import mmap
import os
import struct

import ml_dtypes
import numpy as np

_DT = {
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
    "F8_E8M0": ml_dtypes.float8_e8m0fnu,
    "I8": np.int8,
    "U8": np.uint8,
    "BF16": ml_dtypes.bfloat16,
    "F32": np.float32,
    "F16": np.float16,
    "I32": np.int32,
    "I64": np.int64,
}

# E2M1 magnitudes * 2 as integers, code order (bit 3 = sign): 0,.5,1,1.5,2,3,4,6 -> 0,1,2,3,4,6,8,12
FP4_X2 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int8)


class SafetensorsDir:
    """Lazy reader over a sharded checkpoint directory (mmap; tensors are views, no copies until you convert)."""

    def __init__(self, root: str):
        self.root = root
        self.weight_map = json.load(open(os.path.join(root, "model.safetensors.index.json")))[
            "weight_map"
        ]
        self._files = {}

    def _open(self, fname):
        if fname not in self._files:
            f = open(os.path.join(self.root, fname), "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            (hlen,) = struct.unpack("<Q", mm[:8])
            header = json.loads(mm[8 : 8 + hlen])
            self._files[fname] = (f, mm, header, 8 + hlen)
        return self._files[fname]

    def has(self, name):
        return name in self.weight_map

    def get(self, name) -> np.ndarray:
        _, mm, header, base = self._open(self.weight_map[name])
        meta = header[name]
        s, e = meta["data_offsets"]
        return np.frombuffer(
            mm,
            dtype=_DT[meta["dtype"]],
            count=(e - s) // np.dtype(_DT[meta["dtype"]]).itemsize,
            offset=base + s,
        ).reshape(meta["shape"])

    def shape(self, name):
        return tuple(self._open(self.weight_map[name])[2][name]["shape"])


# ---------------------------------------------------------------------------------------------- conversions


def e8m0_to_f32(s: np.ndarray) -> np.ndarray:
    """float8_e8m0 -> float32 (2**(e-127)); accepts the ml_dtypes view or raw uint8 exponents."""
    e = np.asarray(s).view(np.uint8).astype(np.int32)
    return np.ldexp(np.float32(1.0), e - 127).astype(np.float32)


def dequant_fp8_block(w, scale, block=32) -> np.ndarray:
    """[O, I] fp8 + [O/block, I/block] e8m0 -> float32 [O, I]."""
    o, i = w.shape
    wf = np.asarray(w).astype(np.float32).reshape(o // block, block, i // block, block)
    return (wf * e8m0_to_f32(scale)[:, None, :, None]).reshape(o, i)


def unpack_fp4_x2(packed: np.ndarray) -> np.ndarray:
    """int8 [N, K/2] -> int8 [N, K] holding 2*e2m1 (low nibble = even column)."""
    b = np.asarray(packed).view(np.uint8)
    out = np.empty((b.shape[0], b.shape[1] * 2), np.int8)
    out[:, 0::2] = FP4_X2[b & 0x0F]
    out[:, 1::2] = FP4_X2[b >> 4]
    return out


def dequant_expert(w_int8_x2: np.ndarray, scale_u8: np.ndarray, block=32) -> np.ndarray:
    """int8 (2*e2m1) [N,K] + uint8 e8m0 [N,K/32] -> float32 [N,K] (reference value of the FP4 weight)."""
    n, k = w_int8_x2.shape
    s = e8m0_to_f32(scale_u8) * 0.5
    return (w_int8_x2.astype(np.float32).reshape(n, k // block, block) * s[..., None]).reshape(n, k)


def fp4_to_fp8_rowscale(w_x2: np.ndarray, scale_u8: np.ndarray, block=32):
    """Experts for the TPU: [N,K] int8 (2*e2m1) + [N,K/32] e8m0 exponents -> float8_e4m3fn [N,K] + float32 row scale [N].

    row scale r = max over the row's block scales (a power of two); block j is stored as value * (s_j / r) in e4m3, which
    is exact while s_j / r >= 2**-8 (e4m3 keeps 0.5 * 2**-8 = 2**-9, its smallest subnormal). Returns (fp8, r, n_inexact)
    where n_inexact counts elements whose reconstruction differs from the FP4 checkpoint value."""
    n, k = w_x2.shape
    s = e8m0_to_f32(scale_u8)  # [N, K/32] powers of two
    r = s.max(axis=1)  # [N]
    off = s / r[:, None]  # power of two <= 1
    v = w_x2.astype(np.float32).reshape(n, k // block, block) * 0.5
    q = (v * off[..., None]).astype(ml_dtypes.float8_e4m3fn)
    rec = q.astype(np.float32) * r[:, None, None]
    want = v * s[..., None]
    return q.reshape(n, k), r.astype(np.float32), int((rec != want).sum())


# ---------------------------------------------------------------------------------------------- engram tables


class EngramTable:
    """Memory-mapped fp8 table + e8m0 scales; `rows(ids)` returns dequantized float32 rows [..., D].

    Dequantization uses 256-entry lookup tables on the raw bytes (the per-element ml_dtypes fp8 -> float32 conversion was ~40% of the host
    time of a decode step); results are bit-identical to the astype/ldexp path."""

    def __init__(self, ckpt: SafetensorsDir, layer: int, block=32):
        self.w = ckpt.get(f"layers.{layer}.engram.embed.weight")
        self.s = ckpt.get(f"layers.{layer}.engram.embed.scale")
        self.block = block
        self.w8 = self.w.view(np.uint8)
        self.s8 = np.asarray(self.s).view(np.uint8)
        self.lut = (
            np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        )  # fp8 e4m3fn byte -> float32
        self.pow2 = np.ldexp(np.float32(1.0), np.arange(256) - 127).astype(
            np.float32
        )  # e8m0 byte -> 2**(e-127)

    def raw_threaded(self, ids: np.ndarray, pool, parts: int = 8):
        """Same result as raw(ids), gathered by `parts` threads (numpy releases the GIL in take for plain dtypes): the single-threaded
        gather of 16x512 tokens x 16 heads x 2 layers took ~110 ms with the chips idle."""
        flat = ids.reshape(-1)
        n = flat.shape[0]
        w = np.empty((n, self.w8.shape[1]), np.uint8)
        s = np.empty((n, self.s8.shape[1]), np.uint8)
        bounds = np.linspace(0, n, parts + 1).astype(np.int64)

        def one(k):
            a, b = bounds[k], bounds[k + 1]
            np.take(self.w8, flat[a:b], axis=0, out=w[a:b], mode="clip")
            np.take(self.s8, flat[a:b], axis=0, out=s[a:b], mode="clip")

        list(pool.map(one, range(parts)))
        return w.reshape(*ids.shape, -1), s.reshape(*ids.shape, -1)

    def raw(self, ids: np.ndarray):
        """Raw table bytes for `ids`: (fp8 bytes [..., D] uint8, e8m0 exponent bytes [..., D/block] uint8). Dequantization is then done on
        the device (engram_jax.dequant_rows): the host-side float32 dequantization cost ~75% of a prefill call, this is a plain gather.
        """
        return self.w8[ids], self.s8[ids]

    def rows(self, ids: np.ndarray) -> np.ndarray:
        w = self.lut[self.w8[ids]]
        s = self.pow2[self.s8[ids]]
        w = w.reshape(*w.shape[:-1], -1, self.block) * s[..., None]
        return w.reshape(*ids.shape, -1)


# ---------------------------------------------------------------------------------------------- sharded parameter build


def build_params(
    ckpt: SafetensorsDir, table: dict, mesh, spec_fn, n_experts: int, threads: int = 16, log=print
):
    """name -> sharded jax.Array, read from the checkpoint with per-device callbacks.

    `table` is config_real.param_table(a) (name -> (shape, dtype)). Dense tensors are dequantized to the table dtype on the host;
    routed experts are converted FP4 -> float8_e4m3fn + float32 row scales, only for the expert block each addressable device owns.
    """
    import re
    from concurrent.futures import ThreadPoolExecutor

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding

    npdt = {
        jnp.bfloat16: ml_dtypes.bfloat16,
        jnp.float32: np.float32,
        jnp.float8_e4m3fn: ml_dtypes.float8_e4m3fn,
    }
    pool = ThreadPoolExecutor(threads)
    inexact = [0]
    block_cache = {}

    def dense(name, dtype):
        w = ckpt.get(name)
        sname = name[: -len(".weight")] + ".scale" if name.endswith(".weight") else None
        if sname and ckpt.has(sname) and w.dtype == ml_dtypes.float8_e4m3fn:
            return dequant_fp8_block(w, ckpt.get(sname)).astype(npdt[dtype])
        return np.asarray(w).astype(npdt[dtype])

    def expert_block(layer, w, e0, e1):
        key = (layer, w, e0, e1)
        if key not in block_cache:

            def one(e):
                n = f"layers.{layer}.ffn.experts.{e}.{w}"
                q, r, bad = fp4_to_fp8_rowscale(
                    unpack_fp4_x2(ckpt.get(n + ".weight")), np.asarray(ckpt.get(n + ".scale"))
                )
                return q, r, bad

            res = list(pool.map(one, range(e0, e1)))
            inexact[0] += sum(r[2] for r in res)
            block_cache[key] = [np.stack([r[0] for r in res]), np.stack([r[1] for r in res]), 2]
        return block_cache[key]

    out = {}
    for name, (shape, dtype) in table.items():
        sh = NamedSharding(mesh, spec_fn(name))
        m = re.match(r"layers\.(\d+)\.ffn\.experts\.(w[123])(_rowscale)?$", name)
        if m:
            layer, w, is_scale = int(m.group(1)), m.group(2), bool(m.group(3))

            def cb(idx, layer=layer, w=w, is_scale=is_scale):
                e0 = idx[0].start or 0
                e1 = idx[0].stop if idx[0].stop is not None else n_experts
                blk = expert_block(layer, w, e0, e1)
                res = blk[1] if is_scale else blk[0]
                blk[2] -= 1
                if blk[2] == 0:
                    block_cache.pop((layer, w, e0, e1), None)
                return res

            out[name] = jax.make_array_from_callback(shape, sh, cb)
        else:
            full = dense(name, dtype)
            assert full.shape == tuple(shape), (name, full.shape, shape)
            out[name] = jax.make_array_from_callback(shape, sh, lambda idx, full=full: full[idx])
        if len(out) % 100 == 0:
            log(
                f"  loaded {len(out)}/{len(table)} tensors; fp8-row-scale inexact elements so far: {inexact[0]}"
            )
    log(f"loaded {len(out)} tensors; fp8 row-scale conversion inexact elements: {inexact[0]}")
    return out
