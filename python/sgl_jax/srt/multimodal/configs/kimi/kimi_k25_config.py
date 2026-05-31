# python/sgl_jax/srt/multimodal/configs/kimi/kimi_k25_config.py
from dataclasses import dataclass
import jax.numpy as jnp
from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs

@dataclass
class KimiK25ModelVitConfig(MultiModalModelConfigs):
    _attn_implementation: str = "flash_attention_2"
    init_pos_emb_height: int = 64
    init_pos_emb_time: int = 4
    init_pos_emb_width: int = 64
    merge_kernel_size: tuple[int, int] = (2, 2)
    merge_type: str = "sd2_tpool"
    mm_hidden_size: int = 1152
    mm_projector_type: str = "patchmerger"
    patch_size: int = 14
    pos_emb_type: str = "divided_fixed"
    projector_hidden_act: str = "gelu"
    projector_ln_eps: float = 1e-05
    text_hidden_size: int = 7168
    video_attn_type: str = "spatial_temporal"
    vt_hidden_size: int = 1152
    vt_intermediate_size: int = 4304
    vt_num_attention_heads: int = 16
    vt_num_hidden_layers: int = 27
    
    model_class: type | None = None
