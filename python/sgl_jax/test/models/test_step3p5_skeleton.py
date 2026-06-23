"""Skeleton smoke-tests for Step3p5ForCausalLM (Task 1, Step 3.5 Flash).

Verifies:
1. Registry resolves ``"Step3p5ForCausalLM"`` to the correct class.
2. The skeleton constructs without error on a tiny CPU config.
3. ``patch_model_config`` sets ``mc.head_dim = 128``.
4. The worktree source (not the editable-install main repo) is under test.

Run from ``python/`` with::

    JAX_PLATFORMS=cpu python -m pytest sgl_jax/test/models/test_step3p5_skeleton.py -x -v -o "pythonpath=."
"""

from __future__ import annotations

import inspect
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Single-device CPU mesh: ici [data=1, tensor=1].
_mesh = create_device_mesh(
    ici_parallelism=[1, 1],
    dcn_parallelism=[1, 1],
    devices=[jax.devices()[0]],
)


def _tiny_config():
    """Minimal Step3p5Config for unit tests (2 layers, hidden=256)."""
    from sgl_jax.srt.configs.step3p5 import Step3p5Config

    return Step3p5Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=128,
        vocab_size=128,
        max_position_embeddings=512,
        rope_theta=[5000000.0, 10000.0],
        rope_scaling=None,
        layer_types=["full_attention", "sliding_attention"],
        partial_rotary_factors=[0.5, 1.0],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 128,
        },
        swiglu_limits=[0.0, 0.0],
        swiglu_limits_shared=[0.0, 0.0],
        moe_layers_enum="1",
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=64,
        share_expert_dim=64,
        sliding_window=16,
    )


class TestStep3p5Registry(unittest.TestCase):
    """Registry resolution: Step3p5ForCausalLM must be discoverable."""

    def test_registry_resolves_arch(self):
        from sgl_jax.srt.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls(["Step3p5ForCausalLM"])
        self.assertEqual(arch, "Step3p5ForCausalLM")
        self.assertEqual(cls.__name__, "Step3p5ForCausalLM")

    def test_entry_class_is_list_containing_model(self):
        """EntryClass is a list (matching peers like deepseek_v3, minimax_m2)."""
        from sgl_jax.srt.models.step3p5 import EntryClass, Step3p5ForCausalLM

        self.assertIsInstance(EntryClass, list)
        self.assertIn(Step3p5ForCausalLM, EntryClass)


class TestStep3p5Skeleton(unittest.TestCase):
    """Construction and patch_model_config smoke tests."""

    def test_construct_no_exception(self):
        """Skeleton instantiates on a tiny config without raising."""
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _tiny_config()
        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16)

        self.assertIsNotNone(model.model.embed_tokens)
        self.assertIsNotNone(model.lm_head)
        self.assertIsNotNone(model.logits_processor)
        self.assertIsNotNone(model.model)

    def test_embed_tokens_shape(self):
        """embed_tokens.embedding has shape (vocab, hidden)."""
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _tiny_config()
        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16)

        emb = model.model.embed_tokens.embedding[...]
        self.assertEqual(emb.shape, (cfg.vocab_size, cfg.hidden_size))

    def test_lm_head_shape(self):
        """lm_head.embedding has shape (vocab, hidden) — untied from embed_tokens."""
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _tiny_config()
        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16)

        lm = model.lm_head.embedding[...]
        self.assertEqual(lm.shape, (cfg.vocab_size, cfg.hidden_size))

    def test_patch_model_config_sets_head_dim(self):
        """patch_model_config sets mc.head_dim = 128."""
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        mc = SimpleNamespace(head_dim=0)
        Step3p5ForCausalLM.patch_model_config(mc)
        self.assertEqual(mc.head_dim, 128)


class TestStep3p5WorktreeSource(unittest.TestCase):
    """Ensure the worktree's source is loaded, not the editable-install main repo."""

    def test_source_is_from_worktree(self):
        from sgl_jax.srt.models import step3p5

        src = inspect.getsourcefile(step3p5)
        self.assertIsNotNone(src, "Could not determine source file for step3p5 module")
        # The worktree path contains 'step35-flash'
        self.assertIn(
            "step35-flash",
            src,
            f"step3p5 module loaded from unexpected path: {src!r}. "
            "Run with '-o pythonpath=.' from the worktree's python/ dir.",
        )


class TestStep3p5Config(unittest.TestCase):
    """Config plumbing: num_key_value_heads alias, field preservation."""

    def test_num_key_value_heads_alias(self):
        """num_key_value_heads equals num_attention_groups (GQA alias)."""
        cfg = _tiny_config()
        self.assertEqual(cfg.num_key_value_heads, cfg.num_attention_groups)
        self.assertEqual(cfg.num_key_value_heads, 2)

    def test_model_type(self):
        from sgl_jax.srt.configs.step3p5 import Step3p5Config

        self.assertEqual(Step3p5Config.model_type, "step3p5")

    def test_layer_types_preserved(self):
        cfg = _tiny_config()
        self.assertEqual(cfg.layer_types, ["full_attention", "sliding_attention"])

    def test_rope_theta_list_preserved(self):
        cfg = _tiny_config()
        self.assertIsInstance(cfg.rope_theta, list)
        self.assertEqual(len(cfg.rope_theta), 2)

    def test_registered_in_auto_config(self):
        """AutoConfig.for_model('step3p5') returns Step3p5Config."""
        from transformers import AutoConfig

        import sgl_jax.srt.hf_transformers_utils  # noqa: F401 — triggers registration
        from sgl_jax.srt.configs.step3p5 import Step3p5Config

        cfg = AutoConfig.for_model(
            "step3p5",
            architectures=["Step3p5ForCausalLM"],
        )
        self.assertIsInstance(cfg, Step3p5Config)


if __name__ == "__main__":
    unittest.main()
