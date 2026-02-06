from dataclasses import dataclass, field

from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs

@dataclass
class Qwen3VLVisionConfig:
    """Vision encoder configuration for Qwen3-VL."""

    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple = (5, 11, 17)
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @classmethod
    def qwen3vl_2b(cls):
        return cls(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            out_hidden_size=2048,
            deepstack_visual_indexes=(5, 11, 17),
        )

    @classmethod
    def qwen3vl_4b(cls):
        return cls(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            out_hidden_size=2560,
            deepstack_visual_indexes=(5, 11, 17),
        )

    @classmethod
    def qwen3vl_8b(cls):
        return cls(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            out_hidden_size=4096,
            deepstack_visual_indexes=(8, 16, 24),
        )

    @classmethod
    def qwen3vl_32b(cls):
        return cls(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            out_hidden_size=5120,
            deepstack_visual_indexes=(8, 16, 24),
        )


@dataclass
class Qwen3VLTextConfig:
    """Text decoder configuration for Qwen3-VL."""

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000
    mrope_section: tuple = (24, 20, 20)  # T, H, W partitions of head_dim
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    @classmethod
    def qwen3vl_2b(cls):
        return cls(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            tie_word_embeddings=True,
        )

    @classmethod
    def qwen3vl_4b(cls):
        return cls(
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            tie_word_embeddings=True,
        )

    @classmethod
    def qwen3vl_8b(cls):
        return cls(
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            tie_word_embeddings=False,
        )

    @classmethod
    def qwen3vl_32b(cls):
        return cls(
            hidden_size=5120,
            intermediate_size=25600,
            num_hidden_layers=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            tie_word_embeddings=False,
        )


@dataclass
class Qwen3VLConfig(MultiModalModelConfigs):
    """Combined configuration for Qwen3-VL model."""
    
    vision_config: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    text_config: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

    @classmethod
    def qwen3vl_2b(cls):
        """Qwen3-VL 2B configuration."""
        return cls(
            vision_config=Qwen3VLVisionConfig.qwen3vl_2b(),
            text_config=Qwen3VLTextConfig.qwen3vl_2b(),
        )

    @classmethod
    def qwen3vl_4b(cls):
        """Qwen3-VL 4B configuration."""
        return cls(
            vision_config=Qwen3VLVisionConfig.qwen3vl_4b(),
            text_config=Qwen3VLTextConfig.qwen3vl_4b(),
        )

    @classmethod
    def qwen3vl_8b(cls):
        """Qwen3-VL 8B configuration."""
        return cls(
            vision_config=Qwen3VLVisionConfig.qwen3vl_8b(),
            text_config=Qwen3VLTextConfig.qwen3vl_8b(),
        )

    @classmethod
    def qwen3vl_32b(cls):
        return cls(
            vision_config=Qwen3VLVisionConfig.qwen3vl_32b(),
            text_config=Qwen3VLTextConfig.qwen3vl_32b(),
        )
