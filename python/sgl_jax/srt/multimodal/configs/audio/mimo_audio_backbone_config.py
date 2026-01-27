"""MiMo Audio Backbone configuration for sglang-jax."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from transformers import PretrainedConfig


@dataclass
class MiMoAudioBackboneConfig(PretrainedConfig):
    """Configuration class for MiMo Audio Backbone model.

    This is a Qwen2-based model with additional components for audio token generation:
    - Main transformer (36 layers): processes interleaved text and speech embeddings
    - Local transformer (16 layers): generates audio tokens for each group
    - Input local transformer (6 layers): processes speech embeddings with bidirectional attention
    """

    model_type: str = "mimo_audio"

    # Main model config (Qwen2-7B based)
    vocab_size: int = 151680
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 11008
    max_position_embeddings: int = 8192
    rope_theta: float = 640000.0
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True

    # Audio-specific config
    group_size: int = 4
    audio_channels: int = 8

    # Local transformer config (for audio token generation)
    local_dim: int = 1024
    local_layers: int = 16
    local_attn_heads: int = 64
    local_ffn_dim: int = 4096
    local_head_dim: int = 16  # local_dim // local_attn_heads
    local_attn_dropout: float = 0.1

    # Input local transformer config (for speech embedding processing)
    input_local_layers: int = 6
    input_local_dim: int = 1024
    input_full_attention: bool = True  # bidirectional attention

    # Speech vocab sizes per channel (8 channels total)
    speech_vocab_sizes: Tuple[int, ...] = (1025, 1025, 129, 129, 129, 129, 129, 129)
    speech_empty_ids: Tuple[int, ...] = (1024, 1024, 128, 128, 128, 128, 128, 128)
    delay_pattern: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

    # Model path for weight loading
    model_path: Optional[str] = None
    model_class: Optional[type] = None
    revision: Optional[str] = None
    dtype: str = "bfloat16"

    def __post_init__(self):
        # Ensure tuples are properly converted from lists (JSON doesn't support tuples)
        if isinstance(self.speech_vocab_sizes, list):
            self.speech_vocab_sizes = tuple(self.speech_vocab_sizes)
        if isinstance(self.speech_empty_ids, list):
            self.speech_empty_ids = tuple(self.speech_empty_ids)
        if isinstance(self.delay_pattern, list):
            self.delay_pattern = tuple(self.delay_pattern)

    def get_total_num_kv_heads(self) -> int:
        """Return total number of KV heads (required by WeightLoader)."""
        return self.num_key_value_heads

    @property
    def hf_config(self):
        """WeightLoader expects model_config.hf_config."""
        return self

    def create_main_transformer_config(self) -> "MiMoAudioBackboneConfig":
        """Create config for main transformer (Qwen2 layers)."""
        return self

    def create_patch_decoder_config(self) -> dict:
        """Create config dict for patch decoder."""
        return {
            "hidden_size": self.local_dim,
            "num_hidden_layers": self.local_layers,
            "num_attention_heads": self.local_attn_heads,
            "num_key_value_heads": self.local_attn_heads,
            "intermediate_size": self.local_ffn_dim,
            "head_dim": self.local_head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "attention_bias": self.attention_bias,
        }

    def create_patch_encoder_config(self) -> dict:
        """Create config dict for patch encoder."""
        return {
            "hidden_size": self.input_local_dim,
            "num_hidden_layers": self.input_local_layers,
            "num_attention_heads": self.local_attn_heads,
            "num_key_value_heads": self.local_attn_heads,
            "intermediate_size": self.local_ffn_dim,
            "head_dim": self.local_head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "attention_bias": self.attention_bias,
            "use_causal_mask": False,  # bidirectional attention
        }


@dataclass
class MiMoAudioArguments:
    """Arguments for MiMo Audio model inference.

    Token indices should be loaded from the model's config at runtime.
    """

    model_name_or_path: Optional[str] = None
    sosp_idx: int = 0  # start of speech token index
    eosp_idx: int = 0  # end of speech token index
    sostm_idx: int = 0  # start of streaming token index
    eostm_idx: int = 0  # end of streaming token index
    eot_idx: int = 0  # end of turn token index
    empty_idx: int = 0  # empty token index


@dataclass
class MiMoSamplerConfig:
    """Configuration for MiMo audio token sampling."""

    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
