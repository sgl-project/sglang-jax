"""Qwen2.5-VL vision-config normalization (review code-review §2/§13).

Confirms the in-model ViT config is read directly from the parsed HF vision_config with NO
per-model defaults: variant-varying dims come from the checkpoint (a missing one RAISES, instead of
the deleted QwenVLModelVitConfig's hidden_size=3584 trap), out_hidden_size falls back to the
top-level LLM hidden_size, and rms_norm_eps is the one HF-absent architecture constant. A tiny real
ViT construction proves the normalized namespace covers every field the ViT reads.
"""

import types
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.qwen_vl.config_helpers import (
    _QWEN_VL_VISION_RMS_NORM_EPS,
    normalize_qwen_vl_vision_config,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _hf(vision_fields, top_hidden_size=2048):
    """A stand-in parsed HF config: top-level hidden_size + a vision_config namespace."""
    return types.SimpleNamespace(
        hidden_size=top_hidden_size,
        vision_config=types.SimpleNamespace(**vision_fields),
    )


# A realistic 7B-shaped vision_config (real vision width 1280, NOT the 3584 LLM dim).
_VISION_7B = dict(
    depth=32,
    hidden_size=1280,
    intermediate_size=3420,
    num_heads=16,
    in_channels=3,
    patch_size=14,
    spatial_merge_size=2,
    temporal_patch_size=2,
    window_size=112,
    hidden_act="silu",
    out_hidden_size=3584,
    fullatt_block_indexes=[7, 15, 23, 31],
)

# Every attribute the in-model ViT (qwen2_5_vit.py) reads off its config.
_VIT_REQUIRED_ATTRS = (
    "depth",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "window_size",
    "hidden_act",
    "out_hidden_size",
    "fullatt_block_indexes",
    "rms_norm_eps",
)


class TestNormalizeQwenVLVisionConfig(unittest.TestCase):
    def test_reads_real_values_not_defaults(self):
        cfg = normalize_qwen_vl_vision_config(_hf(_VISION_7B))
        self.assertEqual(cfg.hidden_size, 1280)  # real vision width, not 3584
        self.assertEqual(cfg.depth, 32)
        self.assertEqual(cfg.num_heads, 16)
        self.assertEqual(cfg.out_hidden_size, 3584)
        self.assertEqual(cfg.fullatt_block_indexes, [7, 15, 23, 31])

    def test_all_vit_fields_present(self):
        cfg = normalize_qwen_vl_vision_config(_hf(_VISION_7B))
        for attr in _VIT_REQUIRED_ATTRS:
            self.assertTrue(hasattr(cfg, attr), f"normalized config missing {attr!r}")

    def test_missing_vision_config_raises(self):
        with self.assertRaises(ValueError):
            normalize_qwen_vl_vision_config(types.SimpleNamespace(hidden_size=2048))

    def test_missing_required_dim_raises(self):
        bad = {k: v for k, v in _VISION_7B.items() if k != "hidden_size"}
        with self.assertRaises(ValueError):
            normalize_qwen_vl_vision_config(_hf(bad))

    def test_out_hidden_size_falls_back_to_top_level(self):
        no_out = {k: v for k, v in _VISION_7B.items() if k != "out_hidden_size"}
        cfg = normalize_qwen_vl_vision_config(_hf(no_out, top_hidden_size=3584))
        self.assertEqual(cfg.out_hidden_size, 3584)  # merger out == LLM in

    def test_rms_norm_eps_constant_when_absent_and_read_when_present(self):
        cfg = normalize_qwen_vl_vision_config(_hf(_VISION_7B))
        self.assertEqual(cfg.rms_norm_eps, _QWEN_VL_VISION_RMS_NORM_EPS)
        with_eps = dict(_VISION_7B, rms_norm_eps=1e-6)
        cfg2 = normalize_qwen_vl_vision_config(_hf(with_eps))
        self.assertEqual(cfg2.rms_norm_eps, 1e-6)

    def test_alias_field_names_accepted(self):
        # sibling Qwen-VL configs may use num_attention_heads / num_hidden_layers / etc.
        aliased = dict(_VISION_7B)
        aliased["num_attention_heads"] = aliased.pop("num_heads")
        aliased["num_hidden_layers"] = aliased.pop("depth")
        cfg = normalize_qwen_vl_vision_config(_hf(aliased))
        self.assertEqual(cfg.num_heads, 16)
        self.assertEqual(cfg.depth, 32)


class TestTinyViTConstruction(unittest.TestCase):
    """End-to-end: the normalized namespace builds the real ViT (proves field coverage)."""

    def test_constructs_from_normalized_config(self):
        from sgl_jax.srt.models.qwen2_5VL.qwen2_5_vit import (
            Qwen2_5_VL_VisionTransformer,
        )

        tiny = dict(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            in_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            window_size=112,
            hidden_act="silu",
            out_hidden_size=128,
            fullatt_block_indexes=[1],
        )
        cfg = normalize_qwen_vl_vision_config(_hf(tiny, top_hidden_size=128))
        cpu_devices = jax.devices("cpu")
        mesh = create_device_mesh(
            ici_parallelism=[-1, len(cpu_devices)],
            dcn_parallelism=[1, 1],
            devices=cpu_devices,
        )
        with jax.set_mesh(mesh):
            vit = Qwen2_5_VL_VisionTransformer(
                config=cfg,
                dtype=jnp.bfloat16,
                rngs=nnx.Rngs(0),
                mesh=mesh,
                norm_eps=cfg.rms_norm_eps,
            )
        # patch_embed projects to the (tiny) vision hidden_size; merger outputs out_hidden_size.
        self.assertEqual(vit.config.hidden_size, 64)
        self.assertEqual(len(vit.blocks), 2)
        self.assertEqual(vit.fullatt_block_indexes, [1])


if __name__ == "__main__":
    unittest.main()
