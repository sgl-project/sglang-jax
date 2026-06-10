"""Multimodal processor base + arch-keyed registry + input caps (refactor M2 / design §3.5.1).

Per U3, the understanding plane needs exactly one multimodal-specific registry beyond the
standard srt ``ModelRegistry``: a processor registry. A processor declares the HF
architecture classes it serves via a ``models = [ArchClass, ...]`` class attribute and is
matched against ``hf_config.architectures`` — arch-keyed, no model-path substrings
(kills the as-is ``resolve_host_processor`` ad-hoc selection + ``"xxx" in model_path``).
Mirrors srt ``models/registry.py`` (pkgutil scan) and upstream ``import_processors`` (§2.3).

``MediaInputCaps`` carries the U1 input-side caps (design §3.3.4): bounding a single item's
encoder activation so it fits one in-model forward. Defaults ported from upstream
``multimodal/processors/qwen_vl.py``.

Skeleton (framework-first): the registry + caps + ABC interface are concrete; the heavy
processor internals (HF AutoProcessor wrapper, async io/cpu-executor parallel media
loading, smart_resize/smart_nframes algorithms) are filled in M3 when a real model is wired.
"""

from __future__ import annotations

import abc
import dataclasses
import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class MediaInputCaps:
    """Input-side caps that bound a single item to "one in-model forward" (U1, §3.3.4).

    Defaults ported from upstream qwen_vl.py. The smart_resize / smart_nframes algorithms
    that apply these (already partially present in the as-is ``_preprocess_qwen_video``,
    §1.12) are ported in M3; this dataclass is the single config home so processors apply
    them uniformly.
    """

    fps: float = 2.0
    fps_max_frames: int = 768
    video_max_pixels: int = 768 * 28 * 28
    max_aspect_ratio: int = 200
    # image_max_pixels intentionally omitted until the smart_resize port pins the value.


class BaseMultimodalProcessor(abc.ABC):
    """Base for multimodal input processors (mirrors upstream ``BaseMultimodalProcessor``).

    Subclasses:
      - declare ``models = [ArchClass, ...]`` (the registration / dispatch key);
      - implement :meth:`process` (raw media + text -> MultimodalDataItem(s) + input_ids).

    M3 fills: the HF AutoProcessor wrapper, async parallel media loading (io ThreadPool +
    cpu ProcessPool), and applying :attr:`caps`.
    """

    #: HF architecture classes this processor serves (registration key). Subclasses set it.
    models: list = []
    #: Input-side caps (U1). Subclasses may override.
    caps: MediaInputCaps = MediaInputCaps()

    @abc.abstractmethod
    def process(self, *, images=None, videos=None, audios=None, text=None):
        """Turn raw media + text into MultimodalDataItem(s) + input_ids (filled in M3)."""
        raise NotImplementedError


# ----- arch-keyed registry (mirrors srt models/registry.py) -----

_PROCESSOR_REGISTRY: dict[str, type[BaseMultimodalProcessor]] = {}


def register_processor(cls: type[BaseMultimodalProcessor]) -> None:
    """Register a processor under each arch class name in its ``models`` list."""
    for arch in getattr(cls, "models", []) or []:
        name = arch if isinstance(arch, str) else arch.__name__
        if name in _PROCESSOR_REGISTRY and _PROCESSOR_REGISTRY[name] is not cls:
            raise AssertionError(f"Duplicate processor registration for arch {name!r}")
        _PROCESSOR_REGISTRY[name] = cls


def import_processor_classes(package_name: str = "sgl_jax.srt.multimodal.processors") -> None:
    """Scan a package and self-register every ``BaseMultimodalProcessor`` subclass.

    Mirrors srt ``import_model_classes`` / upstream ``import_processors``. Tolerant: a
    missing package or a module that fails to import is skipped with a warning (the real
    processor package is populated as models are migrated in M3+).
    """
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        logger.warning("processor package %s not found yet; skipping scan", package_name)
        return
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if ispkg:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - match srt registry tolerance
            logger.warning("Ignore import error when loading %s. %s", name, e)
            continue
        import inspect

        for _, member in inspect.getmembers(module, inspect.isclass):
            if (
                member.__module__ == module.__name__
                and issubclass(member, BaseMultimodalProcessor)
                and member is not BaseMultimodalProcessor
            ):
                register_processor(member)


def get_processor_cls(architectures) -> type[BaseMultimodalProcessor] | None:
    """Return the processor class registered for any of ``architectures`` (by class name)."""
    if isinstance(architectures, str):
        architectures = [architectures]
    for arch in architectures or []:
        cls = _PROCESSOR_REGISTRY.get(arch)
        if cls is not None:
            return cls
    return None


def supported_processor_archs() -> set[str]:
    return set(_PROCESSOR_REGISTRY.keys())
