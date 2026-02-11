from dataclasses import dataclass
from typing import Optional

from transformers import PretrainedConfig


@dataclass
class MiMoAudioBackboneConfig(PretrainedConfig):

    model_type: str = "mimo_audio"

    # Main transformer (Qwen2-7B)
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

    # Audio processing
    group_size: int = 4
    audio_channels: int = 8

    # Patch decoder (audio token generation, 16 layers)
    local_dim: int = 1024
    local_layers: int = 16
    local_attn_heads: int = 64
    local_ffn_dim: int = 4096
    local_head_dim: int = 16
    local_attn_dropout: float = 0.1

    # Patch encoder (speech embedding processing, 6 layers, bidirectional)
    input_local_layers: int = 6
    input_local_dim: int = 1024
    input_full_attention: bool = True

    # Speech vocabulary per channel
    speech_vocab_sizes: list[int] = None
    speech_empty_ids: list[int] = None
    delay_pattern: list[int] = None

    # Model loading
    model_path: Optional[str] = None
    model_class: Optional[type] = None
    revision: Optional[str] = None
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.speech_vocab_sizes is None:
            self.speech_vocab_sizes = [1025, 1025, 129, 129, 129, 129, 129, 129]
        if self.speech_empty_ids is None:
            self.speech_empty_ids = [1024, 1024, 128, 128, 128, 128, 128, 128]
        if self.delay_pattern is None:
            self.delay_pattern = [0, 1, 2, 3, 4, 5, 6, 7]

    def get_total_num_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def hf_config(self):
        return self


@dataclass
class MiMoAudioArguments:
    model_name_or_path: Optional[str] = None
    sosp_idx: int = 151665
    eosp_idx: int = 151666
    sostm_idx: int = 151670
    eostm_idx: int = 151671
    eot_idx: int = 151672
    empty_idx: int = 151667


@dataclass
class MiMoSamplerConfig:
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
