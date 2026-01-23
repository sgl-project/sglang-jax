# TODO: also move pad_input_ids into this module
import importlib
import inspect
import logging
import pkgutil

from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

PROCESSOR_MAPPING = {}


def import_processors(package_name: str, overwrite: bool = False):
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning("Ignore import error when loading %s: %s", name, e)
                continue
            all_members = inspect.getmembers(module, inspect.isclass)
            classes = [
                member for name, member in all_members if member.__module__ == module.__name__
            ]
            for cls in (cls for cls in classes if issubclass(cls, BaseMultimodalProcessor)):
                assert hasattr(cls, "models")
                for arch in cls.models:
                    if overwrite:
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():
                            if model_cls.__name__ == arch.__name__:
                                del PROCESSOR_MAPPING[model_cls]
                                break
                    PROCESSOR_MAPPING[arch] = cls


def get_mm_processor(
    hf_config, server_args: ServerArgs, processor, transport_mode
) -> BaseMultimodalProcessor:
    for model_cls, processor_cls in PROCESSOR_MAPPING.items():
        if model_cls.__name__ in hf_config.architectures:
            return processor_cls(hf_config, server_args, processor, transport_mode)
    # FIXME this should be fixed cause we modify the architecture name in multimodal
    raise ValueError(
        f"No processor registered for architecture: {hf_config.architectures}.\n"
        f"Registered architectures: {[model_cls.__name__ for model_cls in PROCESSOR_MAPPING]}"
    )
