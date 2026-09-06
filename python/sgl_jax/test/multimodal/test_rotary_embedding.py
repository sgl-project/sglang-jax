import unittest

import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.multimodal.layers.rotary_embedding import NDRotaryEmbedding


class TestNDRotaryEmbedding(unittest.TestCase):
    def test_wan_construction_and_eval_shape(self):
        # Wan's 128-dimensional attention heads split over time, height, width.
        kwargs = dict(rope_dim_list=[44, 42, 42], rope_theta=10000)
        for construct in (NDRotaryEmbedding, nnx.eval_shape):
            with self.subTest(construct=construct.__name__):
                if construct is nnx.eval_shape:
                    rope = construct(lambda: NDRotaryEmbedding(**kwargs))
                else:
                    rope = construct(**kwargs)
                cos, sin = rope.forward_from_grid((2, 3, 4))
                self.assertEqual(cos.shape, (24, 64))
                self.assertEqual(sin.shape, (24, 64))

    def test_wan_grid_matches_rotary_formula_with_frame_offset(self):
        dims = [44, 42, 42]
        rope = NDRotaryEmbedding(rope_dim_list=dims, rope_theta=10000)
        cos, sin = rope.forward_from_grid((2, 3, 4), start_frame=5)
        positions = np.stack(
            np.meshgrid(np.arange(5, 7), np.arange(3), np.arange(4), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        angles = np.concatenate(
            [
                positions[:, axis, None] * 10000.0 ** (-np.arange(0, dim, 2) / dim)
                for axis, dim in enumerate(dims)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(cos, np.cos(angles), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(sin, np.sin(angles), atol=1e-6, rtol=1e-6)

    def test_explicit_positions_preserve_axis_factors_and_real_repetition(self):
        dims = [4, 6, 8]
        rescale = [1.0, 2.0, 3.0]
        interpolation = [0.5, 1.0, 2.0]
        positions = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.float32)
        # FLUX requests real embeddings with each frequency repeated twice.
        rope = NDRotaryEmbedding(
            rope_dim_list=dims,
            rope_theta=10000,
            theta_rescale_factor=rescale,
            interpolation_factor=interpolation,
            use_real=True,
            repeat_interleave_real=True,
        )
        cos, sin = rope(jnp.asarray(positions))
        angles = np.concatenate(
            [
                positions[:, axis, None]
                * interpolation[axis]
                * (10000 * rescale[axis] ** (dim / (dim - 2))) ** (-np.arange(0, dim, 2) / dim)
                for axis, dim in enumerate(dims)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(cos, np.repeat(np.cos(angles), 2, axis=-1), atol=1e-6)
        np.testing.assert_allclose(sin, np.repeat(np.sin(angles), 2, axis=-1), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
