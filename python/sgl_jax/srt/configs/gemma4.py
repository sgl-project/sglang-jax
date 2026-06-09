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
