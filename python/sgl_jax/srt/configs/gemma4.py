from transformers import PretrainedConfig


class Gemma4VisionConfig(PretrainedConfig):
    """Local Gemma 4 vision config for Transformers versions predating Gemma 4."""

    model_type = "gemma4_vision"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=72,
        patch_size=16,
        pooling_kernel_size=3,
        position_embedding_size=10240,
        default_output_length=280,
        rms_norm_eps=1e-6,
        rope_parameters=None,
        standardize=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size
        self.position_embedding_size = position_embedding_size
        self.default_output_length = default_output_length
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = rope_parameters or {
            "rope_theta": 100.0,
            "rope_type": "default",
        }
        self.standardize = standardize


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = PretrainedConfig(**text_config)
        self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_config = Gemma4VisionConfig(**vision_config)
        self.vision_config = vision_config

        super().__init__(**kwargs)

        tc = self.text_config if self.text_config is not None else self
        if not getattr(tc, "_gemma4_remapped", False):
            tc.swa_head_dim = getattr(tc, "head_dim", None)
            tc.head_dim = getattr(tc, "global_head_dim", tc.swa_head_dim)
            tc.swa_num_key_value_heads = getattr(tc, "num_key_value_heads", None)
            tc.num_key_value_heads = getattr(
                tc, "num_global_key_value_heads", tc.swa_num_key_value_heads
            )
            tc._gemma4_remapped = True

        if self.text_config is not None:
            self.swa_head_dim = tc.swa_head_dim
            self.head_dim = tc.head_dim
            self.swa_num_key_value_heads = tc.swa_num_key_value_heads
            self.num_key_value_heads = tc.num_key_value_heads
            self.layer_types = getattr(tc, "layer_types", None)
            self.sliding_window = getattr(tc, "sliding_window", None)
            self.attention_k_eq_v = getattr(tc, "attention_k_eq_v", None)
            self.hybrid_layer_pattern = getattr(tc, "hybrid_layer_pattern", None)
