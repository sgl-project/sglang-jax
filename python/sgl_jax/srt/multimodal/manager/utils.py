from omegaconf import OmegaConf


def load_stage_configs_from_yaml(config_path: str) -> list:
    """Load stage configurations from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        List of stage configuration dictionaries from the file's stage_args
    """
    config_data = OmegaConf.load(config_path)

    return config_data.stage_args


# if __name__ == '__main__':
# config = load_stage_configs_from_yaml("/Users/icdi/Desktop/inf/sglang-jax/python/sgl_jax/srt/multimodal/models/static_configs/wan2_1_stage_config.yaml")
# print(config[0].runtime.num_tpus)
