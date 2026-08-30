import importlib
import logging
import pkgutil
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: dict[str, type[Any] | str] = field(default_factory=dict)

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.models.keys()

    def _raise_for_unsupported(self, architectures: list[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details."
            )

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}"
        )

    def _try_load_model_cls(self, model_arch: str) -> type[Any] | None:
        if model_arch not in self.models:
            return None

        return self.models[model_arch]

    def _normalize_archs(
        self,
        architectures: str | list[str],
    ) -> list[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        # filter out support architectures
        normalized_arch = list(filter(lambda model: model in self.models, architectures))

        # make sure Transformers backend is put at the last as a fallback
        if len(normalized_arch) != len(architectures):
            normalized_arch.append("TransformersForCausalLM")
        return normalized_arch

    def resolve_model_cls(
        self,
        architectures: str | list[str],
    ) -> tuple[type[Any], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)

    def is_in_model_multimodal(self, architectures: str | list[str]) -> bool:
        from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract

        try:
            model_cls, _ = self.resolve_model_cls(architectures)
        except ValueError:
            return False
        return issubclass(model_cls, InModelMultimodalContract)


@lru_cache
def import_model_classes() -> dict[str, type[Any]]:
    model_arch_name_to_cls: dict[str, type[Any]] = {}

    def register(model_cls: type[Any]) -> None:
        assert (
            model_cls.__name__ not in model_arch_name_to_cls
        ), f"Duplicated model implementation for {model_cls.__name__}"
        model_arch_name_to_cls[model_cls.__name__] = model_cls

    package_name = "sgl_jax.srt.models"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning("Ignore import error when loading %s. %s", name, e)
                continue
            if hasattr(module, "EntryClass"):
                entry = module.EntryClass
                if isinstance(entry, list):  # To support multiple model classes in one module
                    for model_cls in entry:
                        register(model_cls)
                else:
                    register(entry)

    return model_arch_name_to_cls


ModelRegistry = _ModelRegistry(import_model_classes())
