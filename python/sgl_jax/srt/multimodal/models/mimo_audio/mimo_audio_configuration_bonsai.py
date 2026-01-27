from dataclasses import dataclass
from typing import TYPE_CHECKING
from bonsai.models.qwen3.modeling import ShardingCfg

if TYPE_CHECKING:
    from bonsai.models.qwen2.modeling import ModelConfig as Qwen2Config


@dataclass
class MiMoAudioConfig:
    vocab_size: int = 151680
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 11008
    max_position_embeddings: int = 8192
    rope_theta: int = 640000
    head_dim: int = 128

    group_size: int = 4
    audio_channels: int = 8

    local_dim: int = 1024
    local_layers: int = 16
    local_attn_heads: int = 64
    local_ffn_dim: int = 4096
    local_attn_dropout: float = 0.1

    input_local_layers: int = 6
    input_local_dim: int = 1024
    input_full_attention: bool = True

    shd_cfg: ShardingCfg = ShardingCfg.no_sharding()

    @classmethod
    def with_sharding(cls, **kwargs):
        kwargs["shd_cfg"] = ShardingCfg.default()
        return cls(**kwargs)

    def create_qwen2_config(self) -> "Qwen2Config":
        from bonsai.models.qwen2.modeling import ModelConfig as Qwen2Config

        return Qwen2Config(
            num_layers=36,
            vocab_size=151680,
            emb_dim=4096,
            mlp_dim=11008,
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            rope_theta=640000,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            shd_cfg=self.shd_cfg,
        )

    def create_local_qwen2_config(self) -> "Qwen2Config":
        from bonsai.models.qwen2.modeling import ModelConfig as Qwen2Config

        return Qwen2Config(
            num_layers=16,
            vocab_size=151680,
            emb_dim=1024,
            mlp_dim=4096,
            num_heads=64,
            head_dim=16,  # 1024 // 64 = 16
            num_kv_heads=64,
            rope_theta=640000,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            shd_cfg=self.shd_cfg,
        )

    def create_input_local_qwen2_config(self) -> "Qwen2Config":
        from bonsai.models.qwen2.modeling import ModelConfig as Qwen2Config

        return Qwen2Config(
            num_layers=6,
            vocab_size=151680,
            emb_dim=1024,
            mlp_dim=4096,
            num_heads=64,
            head_dim=16,
            num_kv_heads=64,
            rope_theta=640000,
            norm_eps=1e-6,
            tie_word_embeddings=False,
            use_causal_mask=False,
            shd_cfg=self.shd_cfg,
        )


@dataclass
class MiMoAudioArguments:
    model_name_or_path: str
    sosp_idx: int
    eosp_idx: int
    sostm_idx: int
    eostm_idx: int
    eot_idx: int
    empty_idx: int


@dataclass
class MiMoSamplerConfig:
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
