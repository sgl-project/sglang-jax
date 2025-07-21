# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/model_executor/model_loader/loader.py

# ruff: noqa: SIM117
import collections
import concurrent
import dataclasses
import fnmatch
import glob
import json
import logging
import math
import os
import time
import jax
from flax import nnx
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, cast
from jax.sharding import Mesh, NamedSharding, PartitionSpec


import huggingface_hub
import numpy as np
import safetensors.torch
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn

from sgl_jax.srt.configs.device_config import DeviceConfig
from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.utils.utils import get_bool_env_var


from sgl_jax.srt.model_loader.arch import (
    get_model_architecture,
)


logger = logging.getLogger(__name__)

class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError

class JAXModelLoader(BaseModelLoader):
    
    @dataclasses.dataclass
    class JAXSource:
        model_or_path: str
        revision: Optional[str]

        @classmethod
        def init_new(cls, model_config: ModelConfig):
            return cls(
                model_config.model_path,
                model_config.revision,
            )

    def __init__(self, load_config: LoadConfig, rng: jax.Array, mesh: jax.sharding.Mesh):
        super().__init__(load_config)
        if load_config.load_format != LoadFormat.JAX:
            raise ValueError(
                f"JAXModelLoader only supports JAX load format, "
                f"got {load_config.load_format}"
            )
        
        self.rng = rng
        self.mesh = mesh

    def download_model(self, model_config: ModelConfig) -> None:
        source = self.JAXSource.init_new(model_config)
        self._prepare_jax_weights(
            source.model_or_path, 
            source.revision
        )

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> Any:
        # Initialize JAX model
        model = self._initialize_model(model_config)
        
        # Load weights
        jit_model = self._get_model(model, model_config)
    
        return jit_model

    def _initialize_model(self, model_config: ModelConfig) -> Any:
        model_class, _ = get_model_architecture(model_config)
        
        # Check if this is a JAX model
        if not hasattr(model_class, 'load_weights'):
            raise ValueError(
                f"Model class {model_class.__name__} does not support weights loading. "
                "Please ensure you're using a JAX-compatible model."
            )
                
        return model_class

    def _get_model(self, model_class: Any, model_config: ModelConfig) -> nnx.Module:
        model = nnx.eval_shape(lambda: model_class(model_config, self.rng, self.mesh))
        model.load_weights(self.rng)

        @nnx.jit(donate_argnames=(0,))
        def create_jit_model(model):
            state = nnx.state(model)
            nnx.update(model, state)
            return model
        
        with self.mesh:
            jit_model = create_jit_model(model)
        return jit_model

    def _maybe_download_from_modelscope(
        self, model: str, revision: Optional[str]
    ) -> Optional[str]:
        if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=self.load_config.ignore_patterns,
                )
            else:
                model_path = model
            return model_path
        return None

    def _prepare_jax_weights(
        self, model_name_or_path: str, revision: Optional[str]
    ) -> Tuple[str, List[str]]:
        model_path = self._maybe_download_from_modelscope(model_name_or_path, revision)
        if model_path is not None:
            model_name_or_path = model_path

        is_local = os.path.isdir(model_name_or_path)
        
        if is_local:
            hf_folder = model_name_or_path
        else:
            from huggingface_hub import snapshot_download
            
            hf_folder = snapshot_download(
                model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                cache_dir=self.load_config.download_dir,
                tqdm_class=None,
                revision=revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

        hf_weights_files = []
        for file in os.listdir(hf_folder):
            if file.endswith('.msgpack'):
                hf_weights_files.append(os.path.join(hf_folder, file))
        
        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any JAX model weights (.msgpack files) in `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files


def get_model_loader(load_config: LoadConfig, rng: jax.Array, mesh: jax.sharding.Mesh) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.JAX:
        return JAXModelLoader(load_config, rng, mesh)

    return JAXModelLoader(load_config, rng, mesh)
