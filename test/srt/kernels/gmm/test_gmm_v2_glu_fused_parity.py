"""TPU parity of the SwiGLU-epilogue gmm paths against the plain two-matmul path.

Three contracts the EPMoE layer relies on (layers/moe.py, W4A8 int4 experts, per-channel scale):

1. ``gmm_v2(x, interleave(w0, w1), glu_chunk=c)`` == ``silu(gmm_v2(x, w0)) * gmm_v2(x, w1)``:
   the epilogue reads the bf16-rounded pre-activation (same rounding as the two separate
   outputs) and does the SwiGLU math in f32 with one final rounding, which is what XLA's fused
   ``silu(w0) * w1`` does on the two outputs. Compared bit-for-bit against that f32 contraction;
   the XLA-fused form on bf16 operands rounds its sigmoid to bf16 first, so it is only held to a
   scale-relative bound (measured 15-17 bf16 ulps at near-zero outputs on v7x).
2. ``gmm_v2_fused_glu(x, w01, w2)`` == ``gmm_v2(gmm_v2(x, w01, glu_chunk=c), w2)``: documented
   bit-identical (same tile matmul arithmetic, the intermediate just stays in VMEM).
3. Both hold with the layer's row padding / tile_m override, zero-size groups, groups that
   straddle the lhs sublane tiling, and an EP shard offset (group_offset) into the 256 experts.

Shapes mirror GLM-5.2 int4 experts on v7x tp8/ep8: K=6144, I=2048, 32 local experts out of 256.
Run on a TPU host (the kernels have no interpret path)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import gmm_v2
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2_fused_glu import (
    gmm_v2_fused_glu,
)
from sgl_jax.srt.layers.moe import (
    _GLU_EPILOGUE_CHUNK,
    _MOE_FUSED_GLU_DOWN_MAX_ROWS,
    _MOE_FUSED_TILE_M,
    interleave_gate_up,
)

E, LOCAL, K, I = 256, 32, 6144, 2048
CHUNK = _GLU_EPILOGUE_CHUNK
OUT = jnp.bfloat16


def _is_tpu() -> bool:
    try:
        return jax.devices()[0].platform == "tpu"
    except RuntimeError:
        return False


def _group_sizes(rng, m_rows, rank):
    """Global per-expert counts: the rank's LOCAL experts share m_rows // 2 rows (one of them
    forced empty, several odd-sized so groups straddle the 16-row bf16 sublane tiling), the other
    experts share the rest (never touched by this shard)."""
    local = rng.multinomial(m_rows // 2, np.ones(LOCAL) / LOCAL).astype(np.int32)
    local[3] += local[5]
    local[5] = 0  # an empty local expert
    others = rng.multinomial(m_rows - m_rows // 2, np.ones(E - LOCAL) / (E - LOCAL)).astype(
        np.int32
    )
    counts = np.concatenate([others[: rank * LOCAL], local, others[rank * LOCAL :]]).astype(
        np.int32
    )
    assert counts.sum() == m_rows and counts.shape == (E,)
    return counts


def _int4(rng, shape):
    return jnp.asarray(rng.integers(-8, 8, shape), jnp.int4)


def _scale(rng, n):
    # keeps gate/up around unit scale for K=6144 int4 columns so the SwiGLU is not saturated
    return jnp.asarray(rng.uniform(0.5, 1.5, (LOCAL, 1, 1, n)) * 0.01, jnp.float32)


def _bf16_ulps(a, b):
    """Max distance in bf16 ulps between two bf16 arrays (via their bit patterns)."""
    ia = np.asarray(a).view(np.uint16).astype(np.int32)
    ib = np.asarray(b).view(np.uint16).astype(np.int32)
    return int(np.max(np.abs(ia - ib))) if ia.size else 0


@unittest.skipUnless(_is_tpu(), "TPU kernel")
class GmmV2GluFusedParityTest(unittest.TestCase):
    def _operands(self, m_rows, rank, seed):
        """Operands for ``m_rows`` routed rows; the lhs is padded to a multiple of 32 rows exactly
        as the megablox backend / the layer do before calling gmm_v2 (pad rows are booked on the
        last global expert, like ``group_sizes.at[-1].add(pad)``)."""
        rng = np.random.default_rng(seed)
        pad = (-m_rows) % 32
        x = rng.standard_normal((m_rows, K))
        if pad:
            x = np.concatenate([x, np.zeros((pad, K))], axis=0)
        x = jnp.asarray(x, jnp.bfloat16)
        w0 = _int4(rng, (LOCAL, K, I))
        w1 = _int4(rng, (LOCAL, K, I))
        s0, s1 = _scale(rng, I), _scale(rng, I)
        w2 = _int4(rng, (LOCAL, I, K))
        s2 = _scale(rng, K)
        counts = _group_sizes(rng, m_rows, rank)
        counts[-1] += pad
        gs = jnp.asarray(counts)
        off = jnp.asarray([rank * LOCAL], jnp.int32)
        w01 = interleave_gate_up(w0, w1, CHUNK)
        s01 = interleave_gate_up(s0, s1, CHUNK)
        # rows owned by this shard's experts: the only rows gmm_v2 writes with zero_initialize=False
        lo = int(counts[: rank * LOCAL].sum())
        hi = lo + int(counts[rank * LOCAL : (rank + 1) * LOCAL].sum())
        return x, w0, w1, s0, s1, w01, s01, w2, s2, gs, off, (lo, hi)

    @staticmethod
    def _gmm(x, w, s, gs, off, **kw):
        return gmm_v2(
            x,
            w,
            gs,
            rhs_scale=s,
            group_offset=off,
            preferred_element_type=OUT,
            acc_dtype=jnp.float32,
            **kw,
        )

    def _check_glu_epilogue(self, x, w0, w1, s0, s1, w01, s01, gs, off, rows, tag):
        """Contract 1: the glu_chunk epilogue vs two plain calls + SwiGLU, on the rows this shard
        owns (with zero_initialize=False the other rows are unwritten memory in every path)."""
        lo, hi = rows
        h_glu = jax.jit(lambda *a: self._gmm(*a, zero_initialize=False, glu_chunk=CHUNK))(
            x, w01, s01, gs, off
        )
        g = jax.jit(lambda *a: self._gmm(*a, zero_initialize=False))(x, w0, s0, gs, off)
        u = jax.jit(lambda *a: self._gmm(*a, zero_initialize=False))(x, w1, s1, gs, off)
        self.assertEqual(h_glu.shape, (x.shape[0], I))
        self.assertGreater(hi, lo)
        h_glu, g, u = h_glu[lo:hi], g[lo:hi], u[lo:hi]
        # the kernel's math: f32 SwiGLU over the bf16-rounded gate/up, one rounding
        gf, uf = g.astype(jnp.float32), u.astype(jnp.float32)
        ref_f32 = jax.jit(lambda gf, uf: (gf * jax.nn.sigmoid(gf) * uf).astype(OUT))(gf, uf)
        # the layer's non-GLU form: XLA fuses silu(bf16) * bf16
        ref_xla = jax.jit(lambda g, u: jax.nn.silu(g) * u)(g, u)
        ulps_f32 = _bf16_ulps(h_glu, ref_f32)
        # XLA evaluates jax.nn.silu on bf16 operands with bf16 intermediates (sigmoid rounded
        # before the products), so this form differs by bf16 rounding; bit-pattern ulps blow up
        # near zero, hence a scale-relative bound on the max deviation.
        scale = float(jnp.max(jnp.abs(ref_f32.astype(jnp.float32))))
        dev_xla = float(
            jnp.max(jnp.abs(h_glu.astype(jnp.float32) - ref_xla.astype(jnp.float32)))
        ) / max(scale, 1e-30)
        print(
            f"GLU-EPILOGUE {tag}: rows [{lo},{hi}) vs f32-contraction {ulps_f32} ulp, "
            f"vs XLA bf16 silu*up max|d|/max|ref| = {dev_xla:.3e}"
        )
        np.testing.assert_array_equal(
            np.asarray(h_glu),
            np.asarray(ref_f32),
            err_msg=f"{tag}: glu_chunk epilogue != f32 silu*up",
        )
        self.assertLessEqual(dev_xla, 1.0 / 64, f"{tag}: glu_chunk epilogue far from XLA silu(g)*u")

    def _check_fused(self, x, w01, s01, w2, s2, gs, off, rows, tag):
        """Contract 2: one fused pallas_call vs the two-call (glu_chunk then down) path, with the
        layer's tile_m override; the down call zero-initializes, so every row is defined."""
        m = x.shape[0]
        tile_m = _MOE_FUSED_TILE_M if m > _MOE_FUSED_TILE_M else None

        def fused(x, w01, s01, w2, s2, gs, off):
            return gmm_v2_fused_glu(
                x,
                w01,
                s01,
                w2,
                s2,
                gs,
                off,
                glu_chunk=CHUNK,
                preferred_element_type=OUT,
                acc_dtype=jnp.float32,
                zero_initialize=True,
                tile_m=tile_m,
            )

        out_fused = jax.jit(fused)(x, w01, s01, w2, s2, gs, off)
        self.assertIsNotNone(out_fused, f"{tag}: fused kernel declined the production shapes")

        def two_call(x, w01, s01, w2, s2, gs, off):
            h = self._gmm(x, w01, s01, gs, off, zero_initialize=False, glu_chunk=CHUNK)
            return self._gmm(h, w2, s2, gs, off, zero_initialize=True)

        out_two = jax.jit(two_call)(x, w01, s01, w2, s2, gs, off)
        self.assertEqual(out_fused.shape, (m, K))
        lo, hi = rows
        nz = int(jnp.sum(jnp.any(out_fused != 0, axis=1)))
        print(
            f"FUSED-GLU {tag}: rows={m} tile_m={tile_m} shard rows [{lo},{hi}) nonzero_rows={nz} "
            f"ulp(two-call)={_bf16_ulps(out_fused, out_two)}"
        )
        np.testing.assert_array_equal(
            np.asarray(out_fused), np.asarray(out_two), err_msg=f"{tag}: fused != two-call"
        )
        # rows outside this shard's experts are zero (down zero_initialize) in both paths
        self.assertLessEqual(nz, hi - lo)
        outside = np.concatenate([np.asarray(out_fused[:lo]), np.asarray(out_fused[hi:])])
        self.assertFalse(outside.size and np.any(outside != 0), f"{tag}: non-shard rows not zero")

    def _case(self, m_rows, rank, seed=0):
        tag = f"m={m_rows} rank={rank}"
        x, w0, w1, s0, s1, w01, s01, w2, s2, gs, off, rows = self._operands(m_rows, rank, seed)
        self._check_glu_epilogue(x, w0, w1, s0, s1, w01, s01, gs, off, rows, tag)
        self._check_fused(x, w01, s01, w2, s2, gs, off, rows, tag)

    def test_bs1_decode_rows(self):
        # 8 tokens x topk routed rows land in one 32-row tile (bs1 tp8 decode: m = 32 after padding)
        for rank in (0, 7):
            self._case(32, rank, seed=rank)

    def test_unaligned_rows_padding(self):
        # m not a multiple of 32: pad rows booked on the last expert (rank 7's shard owns it)
        self._case(40, 3, seed=11)
        self._case(200, 5, seed=12)

    def test_cc8_rows(self):
        for rank in (0, 3):
            self._case(256, rank, seed=20 + rank)

    def test_max_fused_rows(self):
        # the largest batch the layer still routes to the fused kernel
        self._case(_MOE_FUSED_GLU_DOWN_MAX_ROWS, 7, seed=30)


if __name__ == "__main__":
    unittest.main()
