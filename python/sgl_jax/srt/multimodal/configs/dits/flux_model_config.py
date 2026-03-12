from dataclasses import dataclass

import jax.numpy as jnp
from jax.lax import Precision

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs


@dataclass
class FluxModelConfig(MultiModalModelConfigs):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    hidden_size: int = 3072
    epsilon: float = 1e-6
    attention_impl: str = "usp"

    weights_dtype: jnp.dtype = jnp.bfloat16
    dtype: jnp.dtype = jnp.bfloat16
    precision: Precision = Precision.HIGHEST
    model_class: None = None
    quantization_config: QuantizationConfig | None = None

    def __post_init__(self) -> None:
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim

    def get_total_num_kv_heads(self) -> int:
        return self.num_attention_heads
