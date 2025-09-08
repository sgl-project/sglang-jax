import dataclasses
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import huggingface_hub
import jax
from flax import nnx

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader.arch import get_model_architecture
from sgl_jax.srt.utils.common_utils import get_bool_env_var

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
    ) -> Any:
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

    def __init__(
        self, load_config: LoadConfig, rngs: jax.Array, mesh: jax.sharding.Mesh
    ):
        super().__init__(load_config)
        if load_config.load_format != LoadFormat.JAX:
            raise ValueError(
                f"JAXModelLoader only supports JAX load format, "
                f"got {load_config.load_format}"
            )

        self.rng = rngs
        self.mesh = mesh

    def download_model(self, model_config: ModelConfig) -> str:
        source = self.JAXSource.init_new(model_config)
        hf_folder = self._prepare_weights(source.model_or_path, source.revision)
        return hf_folder

    def load_model(
        self,
        *,
        model_config: ModelConfig,
    ) -> Any:
        # prepare model file
        hf_folder = self.download_model(model_config)
        model_config.model_path = hf_folder
        # Initialize JAX model
        logger.info("start initialize model************************************")
        model = self._initialize_model(model_config)
        logger.info("end initialize model************************************")
        # Load weights
        jit_model = self._get_model(model, model_config)

        return jit_model

    def _initialize_model(self, model_config: ModelConfig) -> Any:
        model_class, _ = get_model_architecture(model_config)

        if not hasattr(model_class, "load_weights"):
            raise ValueError(
                f"Model class {model_class.__name__} does not support weights loading. "
                "Please ensure you're using a JAX-compatible model and implement load_weights method."
            )

        return model_class

    def _get_model(self, model_class: Any, model_config: ModelConfig) -> nnx.Module:
        # Create model directly - random initialization is fast and will be immediately overwritten
        # The "double initialization" cost is minimal since load_weights replaces all parameters
        logger.info("start create mode1l************************************")
        time.sleep(100)
        model = model_class(model_config, self.rng, self.mesh)
        logger.info("start load weights2************************************")
        time.sleep(100)
        # Load actual weights, which overwrites the random initialization
        rng_key = self.rng.default.key.value
        model.load_weights(rng_key)
        logger.info("end load weights3***********************************")
        time.sleep(100)
        # Apply sharding constraints after loading
        with self.mesh:
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)

        return model

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

    def _prepare_weights(
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

        return hf_folder


def get_model_loader(
    load_config: LoadConfig, rngs: jax.Array, mesh: jax.sharding.Mesh
) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.JAX:
        return JAXModelLoader(load_config, rngs, mesh)

    return JAXModelLoader(load_config, rngs, mesh)
