"""CPU: the GLM DSA NextN draft declares its checkpoint with the shared ``WeightSpec``
loader entry point (post #1716) and every declared target exists on the module.

``loader.load(mappings, dummy=True)`` runs the same preparation hooks as a real
load (absorbed MLA, fused MLP packing, fused shared experts) and then fills the
final parameter schema, so a stale target path or a mapping object from the old
``WeightMapping`` API fails here instead of at draft-worker ``load_model`` time.
"""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import save_file

from sgl_jax.srt.model_loader.loader import validate_model_parameters
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.glm5_moe import GlmMoeDsaForCausalLMNextN


def _tiny_config(**over):
    cfg = dict(
        hidden_size=64,
        vocab_size=128,
        rms_norm_eps=1e-5,
        moe_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        norm_topk_prob=True,
        rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        pad_token_id=0,
        is_static_checkpoint=False,
        ignored_layers=[],
        max_position_embeddings=1024,
        use_qk_norm=True,
        attention_bias=False,
        use_dsa_sparse=False,
        first_k_dense_replace=0,
        n_group=1,
        topk_group=1,
        n_shared_experts=1,
        routed_scaling_factor=1.0,
        quantization_config=None,
        tie_word_embeddings=False,
    )
    cfg.update(over)
    return SimpleNamespace(**cfg)


def _mesh():
    return jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(1, 4),
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


class TestGlm5NextNWeightMappings(unittest.TestCase):
    def test_mappings_are_weight_specs_targeting_the_draft_module(self):
        cfg = _tiny_config()
        model_config = SimpleNamespace(hf_config=cfg, quantization_config=None)
        mappings = GlmMoeDsaForCausalLMNextN._create_weight_mappings(model_config)
        self.assertTrue(mappings)
        prefix = f"model.layers.{cfg.num_hidden_layers}."
        for source, spec in mappings.items():
            self.assertIsInstance(spec, WeightSpec, source)
            self.assertTrue(source.startswith(prefix), source)
            targets = (spec.target_path,) if isinstance(spec.target_path, str) else spec.target_path
            for target in targets:
                self.assertFalse(target.startswith("model.layers."), (source, target))
        self.assertIn(f"{prefix}eh_proj.weight", mappings)
        self.assertEqual(mappings[f"{prefix}eh_proj.weight"].target_path, "eh_proj.weight")
        self.assertTrue(
            any(
                t.startswith("mtp_block.")
                for t in (
                    m.target_path for m in mappings.values() if isinstance(m.target_path, str)
                )
            )
        )

    def test_dummy_load_fills_every_declared_parameter(self):
        cfg = _tiny_config()
        mesh = _mesh()
        with tempfile.TemporaryDirectory() as tmp:
            save_file(
                {"placeholder": np.zeros((1,), np.float32)}, os.path.join(tmp, "model.safetensors")
            )
            model_config = SimpleNamespace(
                model_path=tmp, hf_config=cfg, quantization_config=None, _dummy_mode=True
            )
            with jax.set_mesh(mesh):
                model = nnx.eval_shape(
                    lambda: GlmMoeDsaForCausalLMNextN(cfg, mesh=mesh, dtype=jnp.bfloat16)
                )
                loader = WeightLoader(model, model_config, mesh, dtype=jnp.bfloat16)
                mappings = GlmMoeDsaForCausalLMNextN._create_weight_mappings(model_config)
                # Every str target must resolve on the draft module before any hook runs.
                for source, spec in mappings.items():
                    targets = (
                        (spec.target_path,)
                        if isinstance(spec.target_path, str)
                        else spec.target_path
                    )
                    for target in targets:
                        loader._get_param(model, target)
                # Full entry point the draft worker calls (dummy mode: same hooks, no I/O); a stale
                # post-load call or loader API drift fails here instead of at server boot.
                model.load_weights(model_config)
                validate_model_parameters(model, allow_shared=True)
                self.assertEqual(
                    model.get_shared_weight_paths(), ("embed_tokens.embedding", "lm_head.embedding")
                )


if __name__ == "__main__":
    unittest.main()
