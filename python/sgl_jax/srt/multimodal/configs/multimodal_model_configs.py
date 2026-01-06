from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.utils import load_stage_configs_from_yaml


class MultiModalModelConfigs(ModelConfig):
    def __init__(self, model_arch: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_arch = model_arch

    @staticmethod
    def from_server_args(server_args: MultimodalServerArgs):
        model_config = ModelConfig.from_server_args(server_args)
        model_config.default_yaml_path = (
            "python/sgl_jax/srt/multimodal/models/static_configs/wan2_1_stage_config.yaml"
        )
        if model_config.hf_config.architectures[0] == "WanTransformer3DModel":
            model_config.default_yaml_path = (
                "python/sgl_jax/srt/multimodal/models/static_configs/wan2_1_stage_config.yaml"
            )
        if hasattr(server_args, "stages_yaml_path") and server_args.stages_yaml_path is not None:
            model_config.default_yaml_path = server_args.stages_yaml_path
        model_config.stages_config = load_stage_configs_from_yaml(model_config.default_yaml_path)

        return model_config
