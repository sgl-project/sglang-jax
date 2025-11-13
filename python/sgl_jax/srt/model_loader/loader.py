import dataclasses
import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
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
        revision: str | None

        @classmethod
        def init_new(cls, model_config: ModelConfig):
            return cls(
                model_config.model_path,
                model_config.revision,
            )

    def __init__(self, load_config: LoadConfig, rngs: jax.Array, mesh: jax.sharding.Mesh):
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
        with self.mesh:
            model = nnx.eval_shape(
                lambda: model_class(model_config.hf_config, model_config.dtype, self.rng, self.mesh)
            )

        model.load_weights(model_config, None if self.rng is None else self.rng.default.key.value)
        return model

    def _maybe_download_from_modelscope(self, model: str, revision: str | None) -> str | None:
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
        self, model_name_or_path: str, revision: str | None
    ) -> tuple[str, list[str]]:
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

    def __init__(self, load_config: LoadConfig, rngs: jax.Array, mesh: jax.sharding.Mesh):
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
        """
        Fill all floating arrays in nnx.state(model) from a NumPy RNG.
        For each array, the RNG is re-seeded with `seed`, we draw a flat
        stream of length `numel` in a generation dtype (fp16 if <16-bit,
        else native; bfloat16 generates in fp32), reshape to array.shape,
        cast to the target dtype, and (if present) re-apply the array's
        sharding spec so partitioning doesn't affect values.
        """
        params = nnx.state(model)
        pspecs = nnx.get_partition_spec(params)

        def _np_gen_dtype(jdtype) -> np.dtype:
            bits = jnp.finfo(jdtype).bits
            if bits < 16:
                return np.float16
            if jdtype == jnp.bfloat16:
                return np.float32
            return {
                jnp.float16: np.float16,
                jnp.float32: np.float32,
                jnp.float64: np.float64,
            }.get(jdtype, np.float32)

        def _init_leaf(x, pspec):
            is_array = isinstance(x, jax.Array)
            is_abstract = isinstance(x, jax.ShapeDtypeStruct)

            if (is_array or is_abstract) and jnp.issubdtype(x.dtype, jnp.floating):
                tgt_dtype = x.dtype
                shape = x.shape

                # Safely construct sharding
                if pspec is not None:
                    sharding = jax.sharding.NamedSharding(self.mesh, pspec)
                else:
                    # If no pspec, try to preserve original sharding; fallback to replicated
                    sharding = getattr(
                        x,
                        "sharding",
                        jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec()),
                    )

                def _make_shard(indices):
                    # Compute local shard shape from global shape and slice indices
                    shard_shape = []
                    for dim_size, idx in zip(shape, indices):
                        if isinstance(idx, slice):
                            start, stop, step = idx.indices(dim_size)
                            assert step == 1, f"Non-unit step in slice not supported: {idx}"
                            shard_shape.append(stop - start)
                        else:
                            # Integer index (e.g., from advanced indexing); size 1
                            shard_shape.append(1)
                    shard_shape = tuple(shard_shape)

                    # Generate random data
                    rng = np.random.default_rng(seed)
                    gen_dtype = _np_gen_dtype(tgt_dtype)
                    numel = int(np.prod(shard_shape))
                    flat = rng.uniform(low, high, size=numel)
                    arr_np = flat.reshape(shard_shape).astype(gen_dtype)
                    return jnp.asarray(arr_np, dtype=tgt_dtype)

                return jax.make_array_from_callback(shape, sharding, _make_shard)
            return x

        new_params = jax.tree_util.tree_map(_init_leaf, params, pspecs)

        # Do not alter rotary embedding caches
        def _preserve_rope_caches(path, old, new):
            # path is a tuple of keys; stringify for robust matching
            path_str = ".".join(str(k) for k in path)
            if ("cos_sin_cache" in path_str) or ("_cos_sin_cache" in path_str):
                return old
            return new

        new_params = jax.tree_util.tree_map_with_path(_preserve_rope_caches, params, new_params)
        nnx.update(model, new_params)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
    ) -> Any:
        model_class = self._initialize_model(model_config)

        with self.mesh:
            model = nnx.eval_shape(
                lambda: model_class(model_config.hf_config, model_config.dtype, self.rng, self.mesh)
            )
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
