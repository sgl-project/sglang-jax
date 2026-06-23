"""Tests for WeightLoader.assert_all_assigned guard (opt-in).

Run from python/ directory::

    JAX_PLATFORMS=cpu python -m pytest sgl_jax/test/models/test_weight_loader_assert.py -x -v -o "pythonpath=."
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_mesh = create_device_mesh(
    ici_parallelism=[1, 1],
    dcn_parallelism=[1, 1],
    devices=[jax.devices()[0]],
)
jax.sharding.set_mesh(_mesh)

_VOCAB = 32
_HIDDEN = 32
_INTER = 64
_MOE_INTER = 16
_SHARE_DIM = 16
_NUM_EXPERTS = 4
_TOPK = 2
_NUM_HEADS_FULL = 4
_NUM_HEADS_SLIDE = 4
_NUM_KV_HEADS = 2
_HEAD_DIM = 128
_NUM_LAYERS = 3  # layers 0=dense-full, 1=dense-slide, 2=MoE-slide


def _make_config():
    from sgl_jax.srt.configs.step3p5 import Step3p5Config

    return Step3p5Config(
        hidden_size=_HIDDEN,
        intermediate_size=_INTER,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS_FULL,
        num_attention_groups=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        vocab_size=_VOCAB,
        rms_norm_eps=1e-5,
        max_position_embeddings=64,
        rope_theta=[5000000.0, 10000.0, 10000.0],
        rope_scaling=None,
        layer_types=["full_attention", "sliding_attention", "sliding_attention"],
        partial_rotary_factors=[0.5, 1.0, 1.0],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": _NUM_HEADS_SLIDE,
            "num_attention_groups": _NUM_KV_HEADS,
            "head_dim": _HEAD_DIM,
        },
        swiglu_limits=[0.0, 0.0, 0.0],
        swiglu_limits_shared=[0.0, 0.0, 0.0],
        moe_layers_enum="2",
        moe_num_experts=_NUM_EXPERTS,
        moe_top_k=_TOPK,
        moe_intermediate_size=_MOE_INTER,
        share_expert_dim=_SHARE_DIM,
        moe_router_scaling_factor=3.0,
        norm_expert_weight=True,
        use_moe_router_bias=True,
        sliding_window=16,
        yarn_only_types=["full_attention"],
    )


_RNG = np.random.default_rng(77)


def _rand(*shape, dtype=np.float32):
    return _RNG.standard_normal(shape).astype(dtype)


def _build_checkpoint(cfg) -> dict[str, np.ndarray]:
    from sgl_jax.srt.models.step3p5 import _moe_layer_ids

    weights: dict[str, np.ndarray] = {}
    H = cfg.hidden_size
    moe_ids = set(_moe_layer_ids(cfg))

    weights["model.embed_tokens.weight"] = _rand(_VOCAB, H)
    weights["model.norm.weight"] = _rand(H)
    weights["lm_head.weight"] = _rand(_VOCAB, H)

    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        layer_types = cfg.layer_types or []
        is_sliding = (layer_types[i] == "sliding_attention") if i < len(layer_types) else False
        num_q = (
            cfg.attention_other_setting["num_attention_heads"]
            if is_sliding
            else cfg.num_attention_heads
        )
        num_kv = (
            cfg.attention_other_setting.get("num_attention_groups", cfg.num_attention_groups)
            if is_sliding
            else cfg.num_attention_groups
        )
        q_dim = num_q * _HEAD_DIM
        kv_dim = num_kv * _HEAD_DIM

        weights[f"{p}.input_layernorm.weight"] = _rand(H)
        weights[f"{p}.post_attention_layernorm.weight"] = _rand(H)
        weights[f"{p}.self_attn.q_proj.weight"] = _rand(q_dim, H)
        weights[f"{p}.self_attn.k_proj.weight"] = _rand(kv_dim, H)
        weights[f"{p}.self_attn.v_proj.weight"] = _rand(kv_dim, H)
        weights[f"{p}.self_attn.o_proj.weight"] = _rand(H, q_dim)
        weights[f"{p}.self_attn.g_proj.weight"] = _rand(num_q, H)
        weights[f"{p}.self_attn.q_norm.weight"] = _rand(_HEAD_DIM)
        weights[f"{p}.self_attn.k_norm.weight"] = _rand(_HEAD_DIM)

        if i not in moe_ids:
            weights[f"{p}.mlp.gate_proj.weight"] = _rand(_INTER, H)
            weights[f"{p}.mlp.up_proj.weight"] = _rand(_INTER, H)
            weights[f"{p}.mlp.down_proj.weight"] = _rand(H, _INTER)
        else:
            E, M, S = _NUM_EXPERTS, _MOE_INTER, _SHARE_DIM
            weights[f"{p}.moe.gate.weight"] = _rand(E, H)
            weights[f"{p}.moe.router_bias"] = _rand(E)
            weights[f"{p}.moe.gate_proj.weight"] = _rand(E, M, H)
            weights[f"{p}.moe.up_proj.weight"] = _rand(E, M, H)
            weights[f"{p}.moe.down_proj.weight"] = _rand(E, H, M)
            weights[f"{p}.share_expert.gate_proj.weight"] = _rand(S, H)
            weights[f"{p}.share_expert.up_proj.weight"] = _rand(S, H)
            weights[f"{p}.share_expert.down_proj.weight"] = _rand(H, S)

    return weights


class _DummyModelConfig:
    def __init__(self, model_path: str, cfg):
        self.model_path = model_path
        self._dummy_mode = False
        self.quantization_config = None
        self.num_attention_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_attention_groups
        self.num_hidden_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        self.hf_text_config = cfg
        self.hf_config = cfg

    def get_total_num_kv_heads(self):
        return self.num_kv_heads

    def get_num_kv_head_replicas(self, tensor_parallel_size: int = 1) -> int:
        return 1

    def needs_kv_head_replication(self, tensor_parallel_size: int) -> bool:
        return False

    def get_kv_padding_strategy(self) -> str:
        return "replicate"

    @property
    def ep_size(self):
        return 1


class TestAssertAllAssigned(unittest.TestCase):
    """Tests for the assert_all_assigned guard in WeightLoader."""

    def _build_and_load(self, ckpt: dict, assert_all_assigned: bool = True):
        """Build a Step3p5 model and load from ckpt dict via WeightLoader."""
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM
        from sgl_jax.srt.utils.weight_utils import WeightLoader

        cfg = _make_config()
        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(ckpt, os.path.join(tmpdir, "model.safetensors"))
            mc = _DummyModelConfig(tmpdir, cfg)
            loader = WeightLoader(model=model, model_config=mc, mesh=_mesh, dtype=jnp.bfloat16)
            mappings = model._create_weight_mappings(mc)
            with jax.set_mesh(_mesh):
                loader.load_weights_from_safetensors(
                    mappings, assert_all_assigned=assert_all_assigned
                )
        return model

    def test_positive_correct_load_passes_assertion(self):
        """A correct full-checkpoint load must pass assert_all_assigned=True without error."""
        cfg = _make_config()
        ckpt = _build_checkpoint(cfg)
        # Should not raise.
        self._build_and_load(ckpt, assert_all_assigned=True)

    def test_negative_miskeyed_mapping_raises(self):
        """Omitting shared-expert weights triggers the assertion (reproduces the original bug).

        The original bug: share_expert.* mapped under wrong key 'moe.share_expert.*'
        → shared-expert params silently stay at random init. Here we simulate the
        effect by omitting the shared-expert checkpoint entries entirely, which means
        the loader never assigns those params.
        """
        cfg = _make_config()
        ckpt = _build_checkpoint(cfg)

        # Remove the shared-expert keys for MoE layer 2 — simulates a mis-keyed mapping
        # where the mapping references a key the checkpoint doesn't have.
        for proj in ("gate_proj", "up_proj", "down_proj"):
            del ckpt[f"model.layers.2.share_expert.{proj}.weight"]

        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM
        from sgl_jax.srt.utils.weight_utils import WeightLoader

        mc_cfg = _make_config()
        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(mc_cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(ckpt, os.path.join(tmpdir, "model.safetensors"))
            mc = _DummyModelConfig(tmpdir, mc_cfg)
            loader = WeightLoader(model=model, model_config=mc, mesh=_mesh, dtype=jnp.bfloat16)
            mappings = model._create_weight_mappings(mc)
            with jax.set_mesh(_mesh), self.assertRaises(ValueError) as ctx:
                loader.load_weights_from_safetensors(mappings, assert_all_assigned=True)

        err = str(ctx.exception)
        self.assertIn("assert_all_assigned", err)
        # All three shared-expert projections must be listed as unassigned.
        self.assertIn("shared_experts.gate_proj.weight", err)
        self.assertIn("shared_experts.up_proj.weight", err)
        self.assertIn("shared_experts.down_proj.weight", err)

    def test_default_off_no_regression(self):
        """Default assert_all_assigned=False: even a partial load does not raise."""
        cfg = _make_config()
        ckpt = _build_checkpoint(cfg)
        # Omit shared experts — would fail if assert_all_assigned were True.
        for proj in ("gate_proj", "up_proj", "down_proj"):
            del ckpt[f"model.layers.2.share_expert.{proj}.weight"]

        # Must not raise with default (assert_all_assigned=False).
        self._build_and_load(ckpt, assert_all_assigned=False)

    def test_whitelist_suppresses_unassigned_error(self):
        """Whitelisted param paths are excluded from the unassigned check."""
        cfg = _make_config()
        ckpt = _build_checkpoint(cfg)
        # Omit shared-expert gate_proj only.
        del ckpt["model.layers.2.share_expert.gate_proj.weight"]

        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM
        from sgl_jax.srt.utils.weight_utils import WeightLoader

        mc_cfg = _make_config()
        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(mc_cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(ckpt, os.path.join(tmpdir, "model.safetensors"))
            mc = _DummyModelConfig(tmpdir, mc_cfg)
            loader = WeightLoader(model=model, model_config=mc, mesh=_mesh, dtype=jnp.bfloat16)
            mappings = model._create_weight_mappings(mc)
            with jax.set_mesh(_mesh):
                # Whitelist the omitted param — should not raise.
                loader.load_weights_from_safetensors(
                    mappings,
                    assert_all_assigned=True,
                    unassigned_whitelist=["model.layers.2.mlp.shared_experts.gate_proj.weight"],
                )

    def test_step3p5_load_weights_uses_assertion(self):
        """model.load_weights() on Step3p5 calls the assert by default.

        A correct full load must complete without error, confirming the opt-in
        is wired into Step3p5ForCausalLM.load_weights().
        """
        from sgl_jax.srt.models.step3p5 import Step3p5ForCausalLM

        cfg = _make_config()
        ckpt = _build_checkpoint(cfg)

        with jax.set_mesh(_mesh):
            model = Step3p5ForCausalLM(cfg, mesh=_mesh, dtype=jnp.bfloat16, attn_impl="naive")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(ckpt, os.path.join(tmpdir, "model.safetensors"))
            mc = _DummyModelConfig(tmpdir, cfg)
            with jax.set_mesh(_mesh):
                model.load_weights(mc)  # must not raise


if __name__ == "__main__":
    unittest.main()
