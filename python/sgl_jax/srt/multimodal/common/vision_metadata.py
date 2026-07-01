"""Common (arch-general) vision-metadata interface + registry (spec §3.2).

The vision-metadata *interface* (Protocols) and *registry* live here; the
concrete per-arch metadata pytree and builder live in
``models/vision_metadata/<arch>.py`` and register themselves at import time.
Common code (mm_plan / scheduler / encode-JIT plumbing) depends only on this
module, never on a concrete model/metadata file.

Registration is **import-triggered, NOT auto-scanned**: each VLM's main model
file (``models/<arch>.py``) imports its metadata module at top level; when
``ModelRegistry`` loads that model, the import chain runs the metadata module's
top-level ``register_vision_metadata_builder(...)``. ``resolve`` only looks up
the dict (does no import). Adding a new VLM = new model file + new metadata file
+ one import line in the model file, with no change to any central table.
"""

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
    """Per-arch, config-only ViT-aux builder interface (spec §3.2).

    Concrete builders live in ``models/vision_metadata/<arch>.py``. All methods
    are host-side numpy: no weights, no model instance. The scheduler
    instantiates the builder from ``model_config.vision_config`` and calls
    ``get_metadata`` per image, then ``stack_metadata`` per round.
    """

    def __init__(self, vision_cfg) -> None: ...

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


def resolve_vision_metadata_builder(arch_or_config):
    """Resolve the registered builder class for an arch name or ``.arch`` carrier.

    Pure dict lookup, no import. The concrete builder must already be registered
    -- that happens when the main model file was imported (it top-level imports
    its ``models/vision_metadata/<arch>.py``, whose top-level ``register(...)``
    fills the dict). ``ModelRegistry`` loads models well before the scheduler
    runs, so by ``build_mm_embed_plan`` time the entry is present.
    """
    arch = (
        arch_or_config if isinstance(arch_or_config, str) else getattr(arch_or_config, "arch", None)
    )
    builder = _BUILDERS.get(arch)
    if builder is None:
        raise ValueError(
            f"No VisionMetadataBuilder registered for arch={arch!r}. "
            "Ensure the model file top-level imports its "
            "models/vision_metadata/<arch>.py module (which registers the "
            "builder at import time)."
        )
    return builder
