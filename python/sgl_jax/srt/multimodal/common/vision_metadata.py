from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VisionMetadataBuilderProtocol(Protocol):
    def __init__(self, vision_cfg: Any) -> None: ...

    def get_metadata(self, item: Any) -> Any: ...

    def stack_metadata(self, metas: list[Any], patch_k: int) -> Any: ...


_REGISTRY: dict[str, type] = {}


def register_vision_metadata_builder(arch: str, builder_cls: type) -> None:
    """Register ``builder_cls`` for architecture name ``arch``.

    A later registration for the same ``arch`` replaces the earlier one.
    """
    _REGISTRY[arch] = builder_cls


def resolve_vision_metadata_builder(arch_or_config: Any) -> type:
    """Resolve a builder class by arch name or hf_config-like object.

    Raises ``KeyError`` if the arch is not registered, or ``ValueError`` if a
    config-like object has no ``architectures`` list.
    """
    if isinstance(arch_or_config, str):
        arch = arch_or_config
    else:
        architectures = getattr(arch_or_config, "architectures", None)
        if not architectures:
            raise ValueError(
                "resolve_vision_metadata_builder: config has no non-empty "
                "'architectures' attribute"
            )
        arch = architectures[0]

    if arch not in _REGISTRY:
        raise KeyError(f"no vision metadata builder registered for arch {arch!r}")
    return _REGISTRY[arch]
