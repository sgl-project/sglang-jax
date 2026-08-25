import importlib
import inspect
import logging
import pkgutil

from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

logger = logging.getLogger(__name__)

PROCESSOR_MAPPING: dict[str, type[BaseMultimodalProcessor]] = {}


def import_processors(package_name: str) -> None:
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if ispkg:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as e:
            logger.warning("Ignore import error when loading %s: %s", name, e)
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if not issubclass(cls, BaseMultimodalProcessor) or cls is BaseMultimodalProcessor:
                continue
            for arch in getattr(cls, "models", ()):
                PROCESSOR_MAPPING[arch] = cls


def get_mm_processor_cls(hf_config) -> type[BaseMultimodalProcessor] | None:
    for arch in hf_config.architectures:
        processor_cls = PROCESSOR_MAPPING.get(arch)
        if processor_cls is not None:
            return processor_cls
    return None


def get_mm_processor(hf_config, server_args, processor) -> BaseMultimodalProcessor:
    processor_cls = get_mm_processor_cls(hf_config)
    if processor_cls is not None:
        return processor_cls(hf_config, server_args, processor)

    raise ValueError(
        f"No multimodal processor registered for architecture: {hf_config.architectures}.\n"
        f"Registered architectures: {sorted(PROCESSOR_MAPPING)}"
    )
