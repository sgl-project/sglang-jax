import copy
import dataclasses
import glob
import inspect
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import huggingface_hub
import jax
from flax import nnx
from safetensors import safe_open

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.model_loader.arch import get_model_architecture
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.debug_utils import print_parameter_shardings

logger = logging.getLogger(__name__)


def _prepare_static_quantization(model_config, model):
    """Use the same quantized module structure for checkpoint and synthetic weights."""
    config = getattr(model_config, "quantization_config", None)
    if config is not None and config.is_static_checkpoint:
        from sgl_jax.srt.utils.quantization.quantization_utils import apply_quantization

        logger.info("Applying STATIC quantization structure preparation...")
        return apply_quantization(model_config, model, is_static_input=True)
    return model


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


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: str | None
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            )

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model_path,
            model_config.revision,
        )

    def load_model(
        self,
        *,
        model_config: ModelConfig,
    ) -> Any:
        pass

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

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[tuple[str, jax.Array], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder = self._prepare_weights(source.model_or_path, source.revision)
        weights_files = glob.glob(os.path.join(hf_folder, "*.safetensors"))

        if len(weights_files) == 0:
            raise RuntimeError(f"Cannot find any *.safetensors files in {hf_folder}")
        weights_files.sort()
        platform = os.getenv("JAX_PLATFORMS", None)
        backend = "cpu" if platform != "proxy" else "proxy"
        for st_file in weights_files:
            with (
                jax.default_device(jax.local_devices(backend=backend)[0]),
                safe_open(st_file, framework="flax") as f,
            ):
                for name in list(f.keys()):
                    yield source.prefix + name, f.get_tensor(name)


class JAXModelLoader(DefaultModelLoader):
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

    def __init__(self, load_config: LoadConfig, mesh: jax.sharding.Mesh):
        super().__init__(load_config)
        self.mesh = mesh

    def download_model(self, model_config: ModelConfig) -> str:
        source = self.JAXSource.init_new(model_config)
        hf_folder = self._prepare_weights(source.model_or_path, source.revision)
        return hf_folder

    def load_model(
        self,
        model_config: ModelConfig,
    ) -> Any:
        # prepare model file
        hf_folder = self.download_model(model_config)

        # if sub_dir is specified, use it
        if self.load_config.sub_dir is not None:
            hf_folder = os.path.join(hf_folder, self.load_config.sub_dir)
            model_config = copy.copy(model_config)

        model_config.model_path = hf_folder
        # Initialize JAX model
        model = self._initialize_model(model_config)

        from sgl_jax.srt.model_loader.weights import LocalSource

        config = copy.copy(model_config)
        with LocalSource(config, warmup=True) as source:
            config._weight_source = source
            return self._get_model(model, config)

    def _initialize_model(self, model_config: ModelConfig) -> Any:
        if not isinstance(model_config, ModelConfig) or self.load_config.model_class is not None:
            model_class = (
                model_config.model_class
                if hasattr(model_config, "model_class") and model_config.model_class is not None
                else self.load_config.model_class
            )
        else:
            model_class, _ = get_model_architecture(model_config)

        if not hasattr(model_class, "load_weights"):
            raise ValueError(
                f"Model class {model_class.__name__} does not support weights loading. "
                "Please ensure you're using a JAX-compatible model and implement load_weights method."
            )

        return model_class

    def _get_model(self, model_class: Any, model_config: ModelConfig) -> nnx.Module:
        if not isinstance(model_config, ModelConfig):
            config = model_config
        else:
            config = model_config.hf_config

        kwargs = {"dtype": model_config.dtype, "mesh": self.mesh}
        if "dtype_config" in inspect.signature(model_class.__init__).parameters:
            kwargs["dtype_config"] = getattr(model_config, "dtype_config", None)

        with jax.set_mesh(self.mesh):
            model = nnx.eval_shape(lambda: model_class(config, **kwargs))

        model = _prepare_static_quantization(model_config, model)
        model.load_weights(model_config)

        missing = [
            jax.tree_util.keystr(path)
            for path, value in jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))[0]
            if isinstance(value, jax.ShapeDtypeStruct)
        ]
        from sgl_jax.srt.model_loader.weights.source import coordinate_error

        coordinate_error(
            (ValueError(f"Unloaded model parameters: {missing[:20]}") if missing else None),
            "final parameter validation",
        )

        print_parameter_shardings(model)

        return model


class RunaiModelLoader(JAXModelLoader):
    """Keep JAX's existing mapping/sharding logic and replace only checkpoint I/O."""

    def __init__(self, load_config: LoadConfig, mesh: jax.sharding.Mesh):
        from sgl_jax.srt.utils.runai_utils import configure_runai

        BaseModelLoader.__init__(self, load_config)
        if load_config.decryption_key_file:
            raise ValueError("runai_streamer does not support encrypted checkpoints")
        extra = load_config.model_loader_extra_config
        configure_runai({} if extra is None else extra)
        self.mesh = mesh

    def download_model(self, model_config: ModelConfig) -> str:
        from sgl_jax.srt.utils.runai_utils import download_metadata, is_gcs_path

        source = getattr(model_config, "model_weights", None) or model_config.model_path
        if is_gcs_path(source):
            return download_metadata(source, self.load_config.download_dir)
        return self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(self, model_config: ModelConfig) -> Any:
        from sgl_jax.srt.model_loader.weights.source import RunaiWeightSource
        from sgl_jax.srt.utils.runai_utils import is_gcs_path

        model_type = getattr(getattr(model_config, "hf_config", None), "model_type", "")
        if getattr(model_config, "is_multimodal", False) and not model_type.startswith(
            ("gemma4", "qwen3_5")
        ):
            raise ValueError(
                "runai_streamer supports text models and the shared Gemma4/Qwen3.5 loaders. "
                "Other multimodal models may have additional local-file loaders; "
                "use a local checkpoint with --load-format auto for these models."
            )
        source = getattr(model_config, "model_weights", None) or model_config.model_path
        local_path = self.download_model(model_config)
        if not is_gcs_path(source):
            source = local_path
        if self.load_config.sub_dir:
            source = os.path.join(source, self.load_config.sub_dir)
            local_path = os.path.join(local_path, self.load_config.sub_dir)
        config = copy.copy(model_config)
        config.model_path = local_path
        model_class = self._initialize_model(config)
        # Skip filesystem warmup: only metadata is local, and callbacks read
        # addressable tensor ranges directly from the original source.
        with RunaiWeightSource(source, local_path) as weight_source:
            config._weight_source = weight_source
            try:
                return self._get_model(model_class, config)
            finally:
                del config._weight_source


class JAXDummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values for JAX models."""

    def __init__(self, load_config: LoadConfig, mesh: jax.sharding.Mesh):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )
        self.mesh = mesh

    def download_model(self, model_config: ModelConfig) -> None:
        # Nothing to download for dummy loader
        return None

    def _initialize_model(self, model_config: ModelConfig) -> Any:
        # Do not require a load_weights method for dummy loader
        model_class, _ = get_model_architecture(model_config)
        return model_class

    def load_model(
        self,
        *,
        model_config: ModelConfig,
    ) -> Any:
        model_class = self._initialize_model(model_config)

        kwargs = {"dtype": model_config.dtype, "mesh": self.mesh}
        if "dtype_config" in inspect.signature(model_class.__init__).parameters:
            kwargs["dtype_config"] = getattr(model_config, "dtype_config", None)

        if getattr(model_config, "_abstract_mode", False):
            from sgl_jax.srt.utils.quantization.quantization_utils import (
                apply_quantization,
            )

            model_config._dummy_mode = True
            quant_config = model_config.quantization_config
            is_static = quant_config is not None and quant_config.is_static_checkpoint

            def init_and_load():
                def initialize():
                    return model_class(model_config.hf_config, **kwargs)

                # Static preparation consumes shape descriptors, just as the
                # checkpoint loader does. Trace loading/post-load transforms
                # and online quantization without allocating parameter arrays.
                model = nnx.eval_shape(initialize) if is_static else initialize()
                model = _prepare_static_quantization(model_config, model)
                model.load_weights(model_config)
                if not is_static:
                    model = apply_quantization(model_config, model)
                return model

            with jax.set_mesh(self.mesh):
                return nnx.eval_shape(init_and_load)

        with jax.set_mesh(self.mesh):
            model = nnx.eval_shape(lambda: model_class(model_config.hf_config, **kwargs))

        model = _prepare_static_quantization(model_config, model)

        # Use model's load_weights with dummy mode to ensure correct sharding
        # Set a marker in model_config to indicate dummy mode
        model_config._dummy_mode = True
        model.load_weights(model_config)

        return model


def get_model_loader(load_config: LoadConfig, mesh: jax.sharding.Mesh) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return JAXDummyModelLoader(load_config, mesh)

    if load_config.load_format == LoadFormat.RUNAI_STREAMER:
        return RunaiModelLoader(load_config, mesh)

    if load_config.load_format == LoadFormat.JAX:
        return JAXModelLoader(load_config, mesh)

    return JAXModelLoader(load_config, mesh)
