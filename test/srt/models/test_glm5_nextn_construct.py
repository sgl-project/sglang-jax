"""CPU: the GLM DSA NextN draft model constructs against the current layer APIs.

Regression for the merge with main where ``ParallelLMHead`` dropped
``kernel_axes`` in favour of ``mesh=`` / ``enable_dp_lm_head=``: the draft
worker builds this class at ``load_model`` time and a stale kwarg is a hard
TypeError that no other CPU test reaches.
"""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np

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


class TestGlm5NextNConstruct(unittest.TestCase):
    def test_constructs_lm_head_and_logits_processor(self):
        mesh = _mesh()
        with jax.set_mesh(mesh):
            m = GlmMoeDsaForCausalLMNextN(_tiny_config(), mesh=mesh, dtype=jnp.bfloat16)
        self.assertEqual(tuple(m.lm_head.embedding.value.shape), (128, 64))
        self.assertIs(m.logits_processor.mesh, mesh)

    def test_dp_lm_head_flag_reaches_lm_head_and_logits_processor(self):
        mesh = _mesh()
        with jax.set_mesh(mesh):
            m = GlmMoeDsaForCausalLMNextN(
                _tiny_config(enable_dp_lm_head=True), mesh=mesh, dtype=jnp.bfloat16
            )
        self.assertTrue(m.lm_head.enable_dp_lm_head)
        self.assertTrue(m.logits_processor.enable_dp_lm_head)


if __name__ == "__main__":
    unittest.main()
