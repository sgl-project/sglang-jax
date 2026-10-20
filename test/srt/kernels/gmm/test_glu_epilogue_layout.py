"""SwiGLU epilogue layout contract: interleave_gate_up + glu_from_interleaved == silu(x@w0) * (x@w1)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import (
    glu_from_interleaved,
    validate_glu_chunk,
)
from sgl_jax.srt.layers.moe import interleave_gate_up


class GluEpilogueLayoutTest(unittest.TestCase):
    def test_interleaved_matmul_matches_separate_gate_up(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        m, k, n, chunk = 64, 256, 1024, 256
        x = jax.random.normal(k1, (m, k), jnp.float32)
        w0 = jax.random.normal(k2, (k, n), jnp.float32) * 0.05
        w1 = jax.random.normal(k3, (k, n), jnp.float32) * 0.05
        fused = interleave_gate_up(w0, w1, chunk)
        self.assertEqual(fused.shape, (k, 2 * n))
        # layout: chunk c of w0 then chunk c of w1
        np.testing.assert_array_equal(np.asarray(fused[:, 0:chunk]), np.asarray(w0[:, 0:chunk]))
        np.testing.assert_array_equal(
            np.asarray(fused[:, chunk : 2 * chunk]), np.asarray(w1[:, 0:chunk])
        )
        np.testing.assert_array_equal(
            np.asarray(fused[:, 2 * chunk : 3 * chunk]), np.asarray(w0[:, chunk : 2 * chunk])
        )
        # exact: apply the epilogue to the interleaved per-column products
        g = x @ w0
        u = x @ w1
        pre = interleave_gate_up(g, u, chunk)
        got = glu_from_interleaved(pre, chunk)
        ref = jax.nn.silu(g) * u
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
        # end to end through one matmul on the interleaved weight (dot blocking may differ)
        got2 = glu_from_interleaved(x @ fused, chunk)
        np.testing.assert_allclose(np.asarray(got2), np.asarray(ref), rtol=1e-5, atol=1e-5)

    def test_interleave_handles_scale_layouts(self):
        e, n, chunk = 3, 512, 128
        s0 = jnp.arange(e * n, dtype=jnp.float32).reshape(e, 1, 1, n)
        s1 = -s0
        fs = interleave_gate_up(s0, s1, chunk)
        self.assertEqual(fs.shape, (e, 1, 1, 2 * n))
        np.testing.assert_array_equal(
            np.asarray(fs[1, 0, 0, chunk : 2 * chunk]), np.asarray(s1[1, 0, 0, :chunk])
        )
        bs0 = jnp.ones((e, 4, 1, n))
        self.assertEqual(interleave_gate_up(bs0, bs0, chunk).shape, (e, 4, 1, 2 * n))

    def test_validate_glu_chunk(self):
        validate_glu_chunk(tile_n=4096, size_n=4096, chunk=256)
        with self.assertRaises(ValueError):
            validate_glu_chunk(tile_n=4096, size_n=4096, chunk=100)
        with self.assertRaises(ValueError):
            validate_glu_chunk(tile_n=2048 + 256, size_n=4096, chunk=256)
        with self.assertRaises(ValueError):
            validate_glu_chunk(tile_n=4096, size_n=4096 + 256, chunk=256)

    def test_epilogue_chunk_constant(self):
        from sgl_jax.srt.layers import moe

        self.assertEqual(moe._GLU_EPILOGUE_CHUNK, 256)


if __name__ == "__main__":
    unittest.main()
