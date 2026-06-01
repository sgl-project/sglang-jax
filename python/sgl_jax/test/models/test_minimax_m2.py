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

from sgl_jax.srt.models.minimax_m2 import MiniMaxM2ForCausalLM, MiniMaxM2QKNorm
from sgl_jax.srt.models.registry import ModelRegistry
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


class TestMiniMaxM2Registry(unittest.TestCase):
    def test_registered(self):
        self.assertIn("MiniMaxM2ForCausalLM", ModelRegistry.models)
        self.assertIs(ModelRegistry.models["MiniMaxM2ForCausalLM"], MiniMaxM2ForCausalLM)


# Ground truth: per-layer keys from MiniMaxAI/MiniMax-M2 model.safetensors.index.json
# (layer 0, experts 0..255 collapsed to a single representative).
_HF_LAYER0_NON_EXPERT_KEYS = {
    "model.layers.0.block_sparse_moe.e_score_correction_bias",
    "model.layers.0.block_sparse_moe.gate.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.k_norm.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.k_proj.weight_scale_inv",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.o_proj.weight_scale_inv",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.weight_scale_inv",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.v_proj.weight_scale_inv",
}
_HF_EXPERT_KEYS_PER_EXPERT = {
    "w1.weight",
    "w1.weight_scale_inv",
    "w2.weight",
    "w2.weight_scale_inv",
    "w3.weight",
    "w3.weight_scale_inv",
}


class TestMiniMaxM2WeightMappings(unittest.TestCase):
    """Static coverage check: every HF checkpoint key is consumed by the loader
    mappings (so a 230GB load on v6e doesn't fail late on a missing key)."""

    @staticmethod
    def _make_config(num_experts: int = 4):
        from types import SimpleNamespace

        return SimpleNamespace(
            num_hidden_layers=1,
            num_local_experts=num_experts,
            use_qk_norm=True,
            use_routing_bias=True,
        )

    def _layer0_hf_keys_from_mappings(self, mappings: dict, num_experts: int) -> set[str]:
        keys: set[str] = set()
        for k, wm in mappings.items():
            if k.startswith("__MOE_EXPERTS__"):
                keys.update(wm.target_path[1:])
            else:
                keys.add(k)
        return {k for k in keys if k.startswith("model.layers.0.")}

    @staticmethod
    def _stub_model(cfg):
        # Bypass full model construction (62 layers x 256 experts is too costly
        # for a static-mapping test); only `.config` is read by the mapping fn.
        stub = MiniMaxM2ForCausalLM.__new__(MiniMaxM2ForCausalLM)
        object.__setattr__(stub, "config", cfg)
        return stub

    def test_static_fp8_layer_coverage(self):
        cfg = self._make_config(num_experts=4)
        stub = self._stub_model(cfg)
        mappings = stub._create_layer_mappings(
            0, is_static_quant=True, moe_backend="epmoe", use_fused=False
        )
        got = self._layer0_hf_keys_from_mappings(mappings, num_experts=4)

        expected = set(_HF_LAYER0_NON_EXPERT_KEYS)
        for i in range(4):
            for suffix in _HF_EXPERT_KEYS_PER_EXPERT:
                expected.add(f"model.layers.0.block_sparse_moe.experts.{i}.{suffix}")

        self.assertFalse(expected - got, f"missing mappings for: {sorted(expected - got)}")
        self.assertFalse(got - expected, f"unexpected mappings: {sorted(got - expected)}")

    def test_bf16_layer_coverage(self):
        cfg = self._make_config(num_experts=2)
        stub = self._stub_model(cfg)
        mappings = stub._create_layer_mappings(
            0, is_static_quant=False, moe_backend="epmoe", use_fused=False
        )
        got = self._layer0_hf_keys_from_mappings(mappings, num_experts=2)

        expected = {k for k in _HF_LAYER0_NON_EXPERT_KEYS if not k.endswith("weight_scale_inv")}
        for i in range(2):
            for w in ("w1", "w2", "w3"):
                expected.add(f"model.layers.0.block_sparse_moe.experts.{i}.{w}.weight")

        self.assertFalse(expected - got, f"missing mappings for: {sorted(expected - got)}")
        self.assertFalse(got - expected, f"unexpected mappings: {sorted(got - expected)}")


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
