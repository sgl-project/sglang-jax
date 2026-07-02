"""Common vision-metadata protocol and registry."""

from __future__ import annotations

from typing import Protocol


class VisionMetadataPytree(Protocol):
    """Structural marker for a per-arch ViT-aux **registered pytree**.

    The concrete type is defined per model (e.g.
    ``models/vision_metadata/qwen2_5_vl.py``'s ``Qwen25VLVisionMetadata`` with
    children ``window_index`` / ``cu_window_seqlens`` / ``rotary_pos_emb``).
    Common code treats ``meta`` only through this marker -- a registered pytree
    whose leaves are ``np.ndarray`` (host, scheduler-built) or ``jax.Array``
    (device, after ``init_new`` device_put). No member contract is declared here
    on purpose: it names the opaque payload without coupling common -> models.
    Only the model's encode body reads the concrete fields.
    """


class VisionMetadataBuilderProtocol(Protocol):
    """Per-arch, config-only ViT-aux builder interface.

    Concrete builders live in ``models/vision_metadata/<arch>.py`` and are
    resolved from ``model_config``.
    """

    def __init__(self, model_config) -> None: ...

    def get_metadata(self, item) -> VisionMetadataPytree:
        """One ``MultimodalDataItem`` -> native-size per-arch meta (numpy).

        The builder pulls whatever geometry it needs FROM ``item`` (e.g. Qwen's
        ``image_grid_thw``); the interface does not assume a grid, so models
        that derive geometry differently (or need no host aux) still fit.
        """
        ...

    def stack_metadata(self, metas, patch_k) -> VisionMetadataPytree:
        """Cross-rank pad-by-role + stack single-image metas -> ``[dp, ...]``."""
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
