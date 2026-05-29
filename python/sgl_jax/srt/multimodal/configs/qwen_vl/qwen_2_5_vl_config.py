import dataclasses

from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs


@dataclasses.dataclass
class QwenVLModelVitConfig(MultiModalModelConfigs):
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"

    depth = 32
    hidden_size = 3584
    hidden_act = "silu"
    intermediate_size = 3420
    num_heads = 16
    in_channels = 3
    patch_size = 14
    spatial_merge_size = 2
    temporal_patch_size = 2
    tokens_per_second = 4
    window_size = 112
    out_hidden_size = 3584
    fullatt_block_indexes = [7, 15, 23, 31]
    initializer_range = 0.02
    rms_norm_eps = 1e-05
    vocab_size = 151936
    text_hidden_size = 2048
