import dataclasses

from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs


@dataclasses.dataclass
class KimiK25ModelVitConfig(MultiModalModelConfigs):
    model_type = "kimi_k25"
    base_config_key = "vision_config"

    init_pos_emb_height = 64
    init_pos_emb_time = 4
    init_pos_emb_width = 64
    merge_kernel_size = [2, 2]
    merge_type = "sd2_tpool"
    mm_hidden_size = 1152
    mm_projector_type = "patchmerger"
    patch_size = 14
    pos_emb_type = "divided_fixed"
    projector_hidden_act = "gelu"
    projector_ln_eps = 1e-05
    text_hidden_size = 7168
    vt_hidden_size = 1152
    vt_intermediate_size = 4304
    vt_num_attention_heads = 16
    vt_num_hidden_layers = 27
