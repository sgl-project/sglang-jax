import jax
from jax.sharding import Mesh

from torch import nn

from sgl_jax.srt.configs.device_config import DeviceConfig
from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader.loader import BaseModelLoader, get_model_loader
from sgl_jax.srt.model_loader.arch import (
    get_architecture_class_name,
    get_model_architecture,
)


def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
    rng: jax.Array,
    mesh: Mesh,
) -> nn.Module:
    loader = get_model_loader(load_config, rng, mesh)
    return loader.load_model(
        model_config=model_config,
        device_config=device_config,
    )


__all__ = [
    "get_model",
    "get_model_loader",
    "BaseModelLoader",
    "get_architecture_class_name",
    "get_model_architecture",
]
