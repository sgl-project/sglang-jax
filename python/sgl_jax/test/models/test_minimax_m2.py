"""Unit tests for MiniMax-M2 model components.

Run on CPU with simulated devices:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      python -m pytest python/sgl_jax/test/models/test_minimax_m2.py -v
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.models.minimax_m2 import MiniMaxM2QKNorm
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])


def _ref_rmsnorm(x: np.ndarray, scale: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    var = np.mean(x32**2, axis=-1, keepdims=True)
    return (scale.astype(np.float32) * x32 * (var + eps) ** -0.5).astype(x.dtype)


class TestMiniMaxM2QKNorm(unittest.TestCase):
    def test_matches_reference(self):
        tokens, dim, eps = 7, 64, 1e-6
        rng = np.random.default_rng(0)
        x_host = rng.normal(size=(tokens, dim)).astype(np.float32)
        scale_host = rng.normal(size=(dim,)).astype(np.float32)

        with jax.set_mesh(mesh):
            norm = MiniMaxM2QKNorm(dim, epsilon=eps, param_dtype=jnp.float32, kernel_axes=(None,))
            norm.scale[...] = jax.device_put(scale_host, NamedSharding(mesh, P(None)))
            out = jax.jit(norm)(jax.device_put(x_host, NamedSharding(mesh, P(None, None))))

        ref = _ref_rmsnorm(x_host, scale_host, eps)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    @unittest.skipIf(
        mesh.shape.get("tensor", 1) < 2,
        "TP consistency test requires >=2 tensor-parallel devices.",
    )
    def test_tp_variance_allreduce(self):
        """With the feature axis tensor-sharded, GSPMD must all-reduce the
        variance so the result equals the unsharded reference. This is the
        ``qk_norm_type=per_layer`` semantics of MiniMax-M2."""
        tokens, dim, eps = 5, 128, 1e-6
        rng = np.random.default_rng(42)
        x_host = rng.normal(size=(tokens, dim)).astype(np.float32)
        scale_host = rng.normal(size=(dim,)).astype(np.float32)

        with jax.set_mesh(mesh):
            norm = MiniMaxM2QKNorm(dim, epsilon=eps, param_dtype=jnp.float32)
            norm.scale[...] = jax.device_put(scale_host, NamedSharding(mesh, P("tensor")))
            x = jax.device_put(x_host, NamedSharding(mesh, P(None, "tensor")))
            out = jax.jit(norm)(x)

        ref = _ref_rmsnorm(x_host, scale_host, eps)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)


class TestMiniMaxM2AttentionShapes(unittest.TestCase):
    """Dry-run the FP8-static-quant + kv_head_padding shape paths on a mini
    config with tp > num_kv_heads, so shape errors surface in <10s instead of
    after a 10-min 230GB load on TPU."""

    @unittest.skipIf(
        mesh.shape.get("tensor", 1) < 4,
        "Requires tp>=4 to exercise kv_head_padding (num_kv_heads=2).",
    )
    def test_quantized_attention_shapes(self):
        from types import SimpleNamespace

        from flax import nnx

        from sgl_jax.srt.layers.linear import QuantizedLinear
        from sgl_jax.srt.models.minimax_m2 import MiniMaxM2Attention

        tp = mesh.shape["tensor"]
        cfg = SimpleNamespace(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=128,
            use_qk_norm=True,
            rms_norm_eps=1e-6,
            rotary_dim=64,
            rope_theta=10000,
            max_position_embeddings=2048,
        )
        with jax.set_mesh(mesh):
            attn = nnx.eval_shape(
                lambda: MiniMaxM2Attention(cfg, mesh=mesh, layer_id=0, dtype=jnp.bfloat16)
            )
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                ql = QuantizedLinear.from_linear(
                    getattr(attn, name),
                    weight_dtype=jnp.float8_e4m3fn,
                    activation_dtype=None,
                    is_static_input=True,
                    weight_block_size=(128, 128),
                    allow_narrow_n_blockwise=True,
                )
                setattr(attn, name, ql)

        padded_kv = max(cfg.num_key_value_heads, tp)
        self.assertEqual(attn.kv_head_num, padded_kv)
        self.assertEqual(attn.k_proj.weight_q.shape, (padded_kv * 128, 512))
        self.assertEqual(attn.k_proj.weight_scale.shape[2], padded_kv * 128)
        self.assertEqual(attn.v_proj.weight_scale.shape[2], padded_kv * 128)
        self.assertEqual(attn.k_norm.scale.shape, (padded_kv * 128,))
        self.assertEqual(attn.q_proj.weight_q.shape, (8 * 128, 512))
        self.assertEqual(attn.o_proj.weight_q.shape, (512, 8 * 128))
        # All placeholder shapes match what kv_head_padding will produce at
        # load time, so no post-load reshape mismatch.


if __name__ == "__main__":
    unittest.main()
