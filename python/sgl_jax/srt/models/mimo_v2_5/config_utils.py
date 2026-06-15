"""MiMo-V2.5 config helpers, used by the in-model wrapper.

Originally extracted from the staged ``MiMoV2_5Embedding`` static methods (M6-S2) so the in-model
wrapper (``mimo_v2_5_inmodel.py``) didn't import the staged embed class for them; that staged embed
model has since been removed (M6-S5). Pure python.
"""

from __future__ import annotations

from transformers import PretrainedConfig


def get_config_value(config: PretrainedConfig, key: str, default=None):
    """Read ``key`` from the HF config, falling back to its ``processor_config``.

    MiMo-V2.5 keeps several token ids (e.g. ``audio_token_id``) only inside
    ``processor_config``, so fall back to it when the top-level attr is missing.
    """
    value = getattr(config, key, None)
    if value is not None:
        return value
    processor_config = getattr(config, "processor_config", None)
    if isinstance(processor_config, dict):
        return processor_config.get(key, default)
    if processor_config is not None:
        return getattr(processor_config, key, default)
    return default


class MiMoVLVisionConfig(PretrainedConfig):
    """Vision config for the MiMo-VL ViT, mirroring upstream sglang ``models/mimo_vl.py``
    (review code-review §13). Field names + defaults match upstream / the HF checkpoint, so
    ``MiMoVLVisionConfig.from_dict(checkpoint_vision_dict)`` reconstructs the vision config with the
    checkpoint's own values; defaults only fill genuinely-absent fields (the MiMo-V2.5 checkpoint
    omits ``qk_channels`` -> default 64, and uses ``in_chans`` so ``in_channels`` -> default 3).
    No field renaming and no overriding of checkpoint values (the inherited HF ``from_dict`` also
    sets through-fields the checkpoint carries but this signature omits, e.g. ``use_sink`` /
    ``in_chans`` / ``num_query_groups``, via ``PretrainedConfig`` ``**kwargs``).
    """

    model_type = "mimovl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=28,
        hidden_size=1280,
        hidden_act="silu",
        intermediate_size=4608,
        num_heads=32,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=2,
        window_size=128,
        out_hidden_size=2048,
        fullatt_block_indexes=None,
        initializer_range=0.02,
        kv_channels=64,
        qk_channels=64,
        num_query_groups=4,
        num_key_value_heads=8,
        vit_window_attn_types=None,
        visual_token_window_size=64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.out_hidden_size = out_hidden_size
        self.fullatt_block_indexes = (
            fullatt_block_indexes if fullatt_block_indexes is not None else [7, 15, 23, 31]
        )
        self.initializer_range = initializer_range
        self.kv_channels = kv_channels
        self.qk_channels = qk_channels
        self.num_query_groups = num_query_groups
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_heads
        )
        self.visual_token_window_size = visual_token_window_size
        self.vit_window_attn_types = vit_window_attn_types or [-1] * depth
