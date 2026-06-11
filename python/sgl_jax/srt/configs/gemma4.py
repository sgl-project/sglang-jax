from transformers import PretrainedConfig


class Gemma4Config(PretrainedConfig):
    model_type = "gemma4"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = PretrainedConfig(**text_config)
        self.text_config = text_config

        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig(**vision_config)
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
