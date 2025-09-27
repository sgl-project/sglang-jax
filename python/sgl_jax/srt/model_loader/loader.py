import dataclasses
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import huggingface_hub
import jax
import jax.numpy as jnp
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
        model = self._initialize_model(model_config)

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
        @nnx.jit
        def create_model(rng: nnx.Rngs):
            model = model_class(model_config, rng, self.mesh)
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
            return model

        with self.mesh:
            model = create_model(self.rng)

        rng_key = self.rng.default.key.value
        model.load_weights(rng_key)

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


class JAXDummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values for JAX models."""

    def __init__(
        self, load_config: LoadConfig, rngs: jax.Array, mesh: jax.sharding.Mesh
    ):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )
        self.rng = rngs
        self.mesh = mesh

    def download_model(self, model_config: ModelConfig) -> None:
        # Nothing to download for dummy loader
        return None

    def _initialize_model(self, model_config: ModelConfig) -> Any:
        # Do not require a load_weights method for dummy loader
        model_class, _ = get_model_architecture(model_config)
        return model_class

    def _initialize_dummy_weights(
        self,
        model: nnx.Module,
        low: float = -1e-3,
        high: float = 1e-3,
        seed: int = 1234,
    ) -> None:
        """Initialize floating-point weights with random values.

        Values only depend on parameter size and dtype (and fixed seed),
        matching the PyTorch dummy loader behavior.
        """
        params = nnx.state(model)
        pspecs = nnx.get_partition_spec(params)
        key = jax.random.PRNGKey(seed)

        def init_value(x, pspec):
            if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
                target_dtype = x.dtype
                gen_dtype = jnp.float16 if jnp.finfo(target_dtype).bits < 16 else target_dtype
                values = jax.random.uniform(
                    key,
                    shape=x.shape,
                    minval=low,
                    maxval=high,
                    dtype=gen_dtype,
                ).astype(target_dtype)
                # Ensure arrays respect original sharding
                return jax.lax.with_sharding_constraint(values, pspec)
            return x

        new_params = jax.tree_util.tree_map(init_value, params, pspecs)
        nnx.update(model, new_params)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
    ) -> Any:
        # Initialize JAX model definition on mesh
        model_class = self._initialize_model(model_config)

        def create_model(rng: nnx.Rngs):
            model = model_class(model_config.hf_config, rng, self.mesh)
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)
            return model

        with self.mesh:
            model = create_model(self.rng)
            # Assign random weights deterministically
            self._initialize_dummy_weights(model)

        return model


def get_model_loader(
    load_config: LoadConfig, rngs: jax.Array, mesh: jax.sharding.Mesh
) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return JAXDummyModelLoader(load_config, rngs, mesh)

    if load_config.load_format == LoadFormat.JAX:
        return JAXModelLoader(load_config, rngs, mesh)

    return JAXModelLoader(load_config, rngs, mesh)
