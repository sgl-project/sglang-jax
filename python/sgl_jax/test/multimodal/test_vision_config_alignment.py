"""Vision-config alignment to upstream (review code-review §2 / §13 / §15).

MiMo-V2.5: the vision config is rebuilt via ``MiMoVLVisionConfig.from_dict`` (upstream pattern) --
field names + defaults match the checkpoint, so the checkpoint's own values flow through and defaults
only fill genuinely-absent fields (``qk_channels``; ``in_channels`` for the ``in_chans`` checkpoints).

(Qwen2.5-VL runs on the staged path on this branch, so its in-model ViT test lives with the staged
components, not here.)
"""

import unittest


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


if __name__ == "__main__":
    unittest.main()
