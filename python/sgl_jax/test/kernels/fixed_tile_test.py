"""Issue #1795: fixed-tile matmul / RMSNorm must give a row the same bits regardless of the
padded batch size (bucket) and of the row's position in the batch. On TPU, XLA's own dot
does not (down_proj [M,2688]@[2688,5376] differed unless the row offset was a multiple
of 256). Runs in Pallas interpret mode on CPU; the TPU run checks the real lowering.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.fixed_tile import fixed_tile_matmul, fixed_tile_rmsnorm

K, N = 2688, 640  # down_proj local K (5376*4/8); N kept small for CPU time


def _data(m, seed=0):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((m, K)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((K, N)) * 0.02, jnp.bfloat16)
    return x, w


class FixedTileMatmulTest(unittest.TestCase):
    def test_matches_reference(self):
        x, w = _data(200)
        got = np.asarray(fixed_tile_matmul(x, w))
        ref = np.asarray(jnp.dot(x.astype(jnp.float32), w.astype(jnp.float32)))
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-3)

    def test_row_offset_invariant(self):
        # The same 64 rows placed at offsets that broke XLA's dot on TPU (100, 384, 510).
        x, w = _data(1024)
        rows = x[:64]
        base = np.asarray(fixed_tile_matmul(x, w))[:64]
        for off in (100, 384, 510, 900):
            moved = x.at[off : off + 64].set(rows)
            got = np.asarray(fixed_tile_matmul(moved, w))[off : off + 64]
            self.assertEqual(got.tobytes(), base.tobytes(), f"offset {off}")

    def test_bucket_invariant(self):
        # Same rows, different padded M (256 / 512 / 1024 prefill buckets).
        x, w = _data(1024)
        ref = np.asarray(fixed_tile_matmul(x[:160], w))
        for m in (256, 512, 1024):
            got = np.asarray(fixed_tile_matmul(x[:m], w))[:160]
            self.assertEqual(got.tobytes(), ref.tobytes(), f"M={m}")

    def test_k_not_multiple_of_128_is_zero_padded(self):
        rng = np.random.default_rng(1)
        x = jnp.asarray(rng.standard_normal((130, 200)), jnp.float32)
        w = jnp.asarray(rng.standard_normal((200, 256)), jnp.float32)
        np.testing.assert_allclose(
            np.asarray(fixed_tile_matmul(x, w)), np.asarray(x @ w), rtol=1e-4, atol=1e-3
        )


class FixedTileRMSNormTest(unittest.TestCase):
    def test_matches_gemma_rmsnorm(self):
        rng = np.random.default_rng(2)
        x = jnp.asarray(rng.standard_normal((300, 5376)), jnp.bfloat16)
        wt = jnp.asarray(rng.standard_normal((5376,)), jnp.float32)
        xf = x.astype(jnp.float32)
        ref = (xf * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6) * wt).astype(
            jnp.bfloat16
        )
        got = fixed_tile_rmsnorm(x, wt, 1e-6)
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(ref, np.float32), rtol=2e-2, atol=2e-2
        )

    def test_bucket_and_row_invariant(self):
        rng = np.random.default_rng(3)
        x = jnp.asarray(rng.standard_normal((1024, 5376)), jnp.bfloat16)
        wt = jnp.asarray(rng.standard_normal((5376,)), jnp.float32)
        ref = np.asarray(fixed_tile_rmsnorm(x[:160], wt, 1e-6))
        for m in (256, 512, 1024):
            got = np.asarray(fixed_tile_rmsnorm(x[:m], wt, 1e-6))[:160]
            self.assertEqual(got.tobytes(), ref.tobytes(), f"M={m}")
        moved = x.at[510:670].set(x[:160])
        got = np.asarray(fixed_tile_rmsnorm(moved, wt, 1e-6))[510:670]
        self.assertEqual(got.tobytes(), ref.tobytes())

    def test_3d_per_head(self):
        rng = np.random.default_rng(4)
        x = jnp.asarray(rng.standard_normal((70, 4, 256)), jnp.float32)
        got = fixed_tile_rmsnorm(x, None, 1e-6)
        ref = x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-5, atol=1e-5)


class FixedTileMatmulKGridTest(unittest.TestCase):
    """Calls with the same (bk, bn) but a different number of K blocks must each be right, in
    either order. On TPU the kernel once reused the first call's K-grid length (Gemma-4-31B
    tp=8: gate/up [5376,2688] then down_proj [2688,5376], both bk=bn=384)."""

    def _check(self, shapes):
        rng = np.random.default_rng(5)
        for k, n in shapes:
            x = jnp.asarray(rng.standard_normal((16, k)), jnp.bfloat16)
            w = jnp.asarray(rng.standard_normal((k, n)) * 0.02, jnp.bfloat16)
            got = np.asarray(fixed_tile_matmul(x, w), np.float64)
            ref = np.asarray(x, np.float64) @ np.asarray(w, np.float64)
            err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
            self.assertLess(err, 1e-3, f"K={k} N={n}")

    def test_fewer_then_more_k_blocks(self):
        self._check([(1024, 768), (2048, 768)])  # bk=512: 2 then 4 K blocks

    def test_more_then_fewer_k_blocks(self):
        self._check([(2048, 768), (1024, 768)])


if __name__ == "__main__":
    unittest.main()
