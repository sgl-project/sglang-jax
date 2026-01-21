import dataclasses

import jax.numpy as jnp
from jax.lax import Precision

from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs


@dataclasses.dataclass
class WanModelConfig(MultiModalModelConfigs):

    num_layers: int = 30
    num_attention_heads: int = 12
    attention_head_dim: int = 128
    hidden_size: int = 1536
    in_channels: int = 16
    out_channels: int = 16
    ffn_dim: int = 8960
    freq_dim: int = 256
    text_dim: int = 4096
    flow_shift: float | None = 3.0
    image_dim: int | None = None  # None for T2V, set for I2V
    patch_size: tuple[int, int, int] = (1, 2, 2)
    cross_attn_norm: bool = True
    qk_norm: str | None = "rms_norm_across_heads"
    epsilon: float = 1e-6
    added_kv_proj_dim: int | None = None  # None for T2V, set for I2V
    rope_max_seq_len: int = 1024
    boundary_ratio: float | None = None  # For Wan2.2 MoE expert switching

    # Runtime/inference params (not in HF config)
    weights_dtype: jnp.dtype = jnp.bfloat16
    dtype: jnp.dtype = jnp.bfloat16
    precision: Precision = Precision.HIGHEST
    max_text_len: int = 512
    num_frames: int = 5
    # num_frames - 1 must be divisible by scale_factor_temporal
    latent_size: tuple[int, int] = (64, 64)
    # latent_size must be divisible by scale_factor_spatial
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    latent_input_dim: int = 16
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    model_class: None = None  # To be set to the model class

    def get_total_num_kv_heads(self) -> int:
        return self.num_attention_heads
