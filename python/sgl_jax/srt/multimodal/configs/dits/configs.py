import dataclasses

import jax.numpy as jnp
from jax.lax import Precision


@dataclasses.dataclass(frozen=True)
class WanModelConfig:
    """Configuration for Wan2.1-T2V-1.3B Diffusion Transformer."""

    weights_dtype: jnp.dtype = jnp.bfloat16
    dtype: jnp.dtype = jnp.bfloat16
    precision = Precision.HIGHEST
    num_layers: int = 30
    hidden_dim: int = 1536
    latent_input_dim: int = 16
    latent_output_dim: int = 16
    ffn_dim: int = 8960
    freq_dim: int = 256
    num_heads: int = 12
    head_dim: int = 128
    text_embed_dim: int = 4096
    max_text_len: int = 512
    num_frames: int = 11
    latent_size: tuple[int, int] = (60, 90)
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    patch_size: tuple[int, int, int] = (1, 2, 2)
    cross_attn_norm: bool = True
    qk_norm: str | None = "rms_norm_across_heads"
    eps: float = 1e-6
    added_kv_proj_dim: int | None = None  # None for T2V, set for I2V
    rope_max_seq_len: int = 1024

    def __post_init__(self):
        assert (
            self.hidden_dim == self.num_heads * self.head_dim
        ), "hidden_dim must equal num_heads * head_dim"
