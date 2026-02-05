import dataclasses

import jax.numpy as jnp

from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs

@dataclasses.dataclass
class MiMoAudioConfig(MultiModalModelConfigs):
    sampling_rate: int = 24000
    n_mels: int = 128
    nfft: int = 960
    hop_length: int = 240
    window_size: int = 960
    fmin: int = 0
    fmax: int | None = None
    kernel_size: int = 3
    stride_size: int = 2
    d_model: int = 1280
    scale_embedding: bool = False
    activation_function: str = "gelu"
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    encoder_causal: bool = False
    encoder_attn_window_size: tuple[int, int] = (-1, -1)
    encoder_skip_layer_id: int | None = 3
    decoder_layers: int = 32
    decoder_attention_heads: int = 20
    decoder_ffn_dim: int = 5120
    decoder_causal: bool = True
    decoder_attn_window_size: tuple[int, int] = (-1, -1)
    decoder_kernel_size: int = 3
    decoder_stride_size: int = 2
    avg_pooler: int = 2
    num_quantizers: int = 20
    codebook_size: list[int] = dataclasses.field(
        default_factory=lambda: [1024, 1024, 128, 128, 128, 128, 128, 128, 128, 128,
                                  128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
    )
    threshold_ema_dead_code: int = 2
    position_embedding_type: str = "rope"
    rope_theta: float = 10000.0
    rope_type: str = "default"
    ln_type: str = "LayerNorm"
    max_audio_seconds: int = 1800
    vocoder_dim: int = 256
    vocoder_attention_heads: int = 16
    vocoder_intermediate_dim: int = 1024
    vocoder_num_layers: int = 16
    vocoder_attn_window_size: tuple[int, int] = (40, 10)
    vocoder_padding: str = "same"
    dtype: jnp.dtype = jnp.float32
