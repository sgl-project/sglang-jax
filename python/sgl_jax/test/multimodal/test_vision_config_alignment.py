"""Vision-config alignment to upstream (review code-review §2 / §13 / §15).

Qwen2.5-VL: the ViT consumes the native HF ``Qwen2_5_VLVisionConfig`` directly (no per-model
dataclass, no renaming, no overridden defaults); the RMSNorm eps comes from the ``norm_eps`` param
(HF's vision_config carries none).

MiMo-V2.5: the vision config is rebuilt via ``MiMoVLVisionConfig.from_dict`` (upstream pattern) --
field names + defaults match the checkpoint, so the checkpoint's own values flow through and defaults
only fill genuinely-absent fields (``qk_channels``; ``in_channels`` for the ``in_chans`` checkpoints).
"""

import types
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.utils.mesh_utils import create_device_mesh


class TestMiMoVLVisionConfigFromDict(unittest.TestCase):
    @staticmethod
    def _checkpoint_like():
        # Shaped like the real MiMo-V2.5 checkpoint vision_config: uses in_chans (not in_channels),
        # omits qk_channels, carries fullatt_block_indexes / vit_window_attn_types / use_sink.
        return dict(
            depth=28,
            hidden_size=1280,
            num_heads=32,
            intermediate_size=4608,
            out_hidden_size=4096,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            window_size=128,
            hidden_act="silu",
            in_chans=3,
            num_key_value_heads=8,
            fullatt_block_indexes=[0, 9, 18, 27],
            visual_token_window_size=64,
            use_sink=True,
            vit_window_attn_types=[-1] + [0] * 27,
        )

    def test_checkpoint_values_flow_through(self):
        from sgl_jax.srt.models.mimo_v2_5.config_utils import MiMoVLVisionConfig

        c = MiMoVLVisionConfig.from_dict(self._checkpoint_like())
        self.assertEqual(c.hidden_size, 1280)
        self.assertEqual(c.out_hidden_size, 4096)  # checkpoint value, NOT the 2048 default
        self.assertEqual(c.fullatt_block_indexes, [0, 9, 18, 27])  # checkpoint, NOT [7,15,23,31]
        self.assertEqual(c.visual_token_window_size, 64)
        self.assertEqual(c.num_key_value_heads, 8)
        self.assertEqual(c.use_sink, True)  # carried via PretrainedConfig **kwargs

    def test_absent_fields_get_defaults(self):
        from sgl_jax.srt.models.mimo_v2_5.config_utils import MiMoVLVisionConfig

        c = MiMoVLVisionConfig.from_dict(self._checkpoint_like())
        self.assertEqual(c.qk_channels, 64)  # checkpoint omits -> default 64 (= old normalize)
        self.assertEqual(c.in_channels, 3)  # checkpoint uses in_chans -> in_channels default 3
        self.assertEqual(c.in_chans, 3)  # in_chans carried through via kwargs


class TestQwenViTFromHFNativeConfig(unittest.TestCase):
    """The Qwen2.5-VL ViT builds straight from an HF-native-style vision_config (no rms_norm_eps)."""

    def test_constructs_from_hf_native_style_config(self):
        from sgl_jax.srt.models.qwen2_5VL.qwen2_5_vit import (
            Qwen2_5_VL_VisionTransformer,
        )

        # Canonical HF field names, NO rms_norm_eps (eps comes from the norm_eps param).
        cfg = types.SimpleNamespace(
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
        cpu = jax.devices("cpu")
        mesh = create_device_mesh(
            ici_parallelism=[-1, len(cpu)], dcn_parallelism=[1, 1], devices=cpu
        )
        with jax.set_mesh(mesh):
            vit = Qwen2_5_VL_VisionTransformer(
                config=cfg, dtype=jnp.bfloat16, rngs=nnx.Rngs(0), mesh=mesh, norm_eps=1e-6
            )
        self.assertEqual(vit.config.hidden_size, 64)
        self.assertEqual(len(vit.blocks), 2)
        self.assertEqual(vit.fullatt_block_indexes, [1])


if __name__ == "__main__":
    unittest.main()
