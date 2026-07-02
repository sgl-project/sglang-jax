"""Common vision-metadata protocol and registry."""

from __future__ import annotations

from typing import Protocol


class VisionMetadataPytree(Protocol):
    """Structural marker for a per-arch ViT aux registered pytree.

    Common code only carries this opaque payload. Concrete metadata types define
    their own fields and are consumed by their matching model encode body.
    """


class VisionMetadataBuilderProtocol(Protocol):
    """Per-arch host-side ViT aux builder interface."""

    def __init__(self, model_config) -> None: ...

    def get_metadata(self, item) -> VisionMetadataPytree:
        """Build native-size metadata for one multimodal item."""
        ...

    def stack_metadata(self, metas, patch_k) -> VisionMetadataPytree:
        """Pad and stack per-rank native metadata for one DP round."""
        ...


_BUILDERS: dict[str, type] = {}


def register_vision_metadata_builder(arch: str, builder_cls: type) -> None:
    """Register a per-arch vision-metadata builder class.

    Call at the concrete metadata module's top level so importing that module
    (via the main model file) registers the builder. Idempotent overwrite.
    """
    _BUILDERS[arch] = builder_cls


def resolve_vision_metadata_builder(model_config):
    """Resolve and instantiate the registered builder for ``model_config``.

    The concrete metadata module must already be imported by the model module.
    """
    hf_config = getattr(model_config, "hf_config", None)
    archs = getattr(hf_config, "architectures", None) or []
    arch = archs[0] if archs else None
    builder_cls = _BUILDERS.get(arch)
    if builder_cls is None:
        raise ValueError(
            f"No VisionMetadataBuilder registered for arch={arch!r}. "
            "Ensure the model file top-level imports its "
            "models/vision_metadata/<arch>.py module (which registers the "
            "builder at import time)."
        )
    return builder_cls(model_config)
