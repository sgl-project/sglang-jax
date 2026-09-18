"""Local Hugging Face configuration for Muse Glimmer checkpoints."""

from __future__ import annotations

from transformers import PretrainedConfig


class MuseGlimmerTextConfig(PretrainedConfig):
    model_type = "muse_glimmer_text"

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 6656,
        intermediate_size: int = 19968,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        hidden_activation: str = "silu",
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-5,
        post_norm_eps: float = 1e-8,
        sliding_window: int = 2048,
        layer_types: list[str] | None = None,
        rope_parameters: dict | None = None,
        qk_scale_factor: float = 3.87,
        output_multiplier: float = 0.19611613513818404,
        final_logit_softcapping: float = 20.0,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.post_norm_eps = post_norm_eps
        self.sliding_window = sliding_window
        self.layer_types = layer_types or [
            "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
            for i in range(num_hidden_layers)
        ]
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": 500000.0,
        }
        self.rope_theta = self.rope_parameters.get("rope_theta", 500000.0)
        self.qk_scale_factor = qk_scale_factor
        self.output_multiplier = output_multiplier
        self.final_logit_softcapping = final_logit_softcapping
        self.attention_bias = attention_bias
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MuseGlimmerConfig(PretrainedConfig):
    model_type = "muse_glimmer"
    sub_configs = {"text_config": MuseGlimmerTextConfig}

    def __init__(
        self,
        text_config: dict | MuseGlimmerTextConfig | None = None,
        image_token_id: int = 200092,
        video_token_id: int = 200091,
        **kwargs,
    ) -> None:
        if text_config is None:
            text_config = MuseGlimmerTextConfig()
        elif isinstance(text_config, dict):
            text_config = MuseGlimmerTextConfig(**text_config)
        self.text_config = text_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        # ModelConfig has a few root-level hybrid-attention checks.
        self.layer_types = list(text_config.layer_types)
        self.sliding_window = text_config.sliding_window
        self.num_hidden_layers = text_config.num_hidden_layers
        self.num_attention_heads = text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        super().__init__(**kwargs)


class MuseGlimmerAssistantConfig(PretrainedConfig):
    model_type = "muse_glimmer_assistant"

    def __init__(
        self,
        vocab_size: int = 202048,
        use_sliding_window: bool = True,
        dflash_config: dict | None = None,
        rope_parameters: dict | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.use_sliding_window = use_sliding_window
        self.dflash_config = {"causal": False} | (dflash_config or {})
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": 500000.0,
        }
        self.rope_theta = self.rope_parameters.get("rope_theta", 500000.0)
        super().__init__(**kwargs)


__all__ = [
    "MuseGlimmerAssistantConfig",
    "MuseGlimmerConfig",
    "MuseGlimmerTextConfig",
]
