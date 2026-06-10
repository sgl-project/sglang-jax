"""Multimodal capability declaration + is_multimodal derivation (refactor M2 / design §3.5.1).

Per U3: an understanding VLM registers in the standard srt ModelRegistry like any text
model (via ``EntryClass``). The only multimodal-specific facts it declares are a few
capability attributes on the model class; ``is_multimodal`` is DERIVED from that
self-declaration (NOT a separate static arch set), with a deployment-level
``server_args.multimodal`` override on top.

Capability attributes a multimodal model class may declare (all optional, sane defaults):
  - ``audio_kind``: ``"codes" | "features" | None`` — replaces the as-is per-item
    runtime flag ``_is_codes_audio`` (design §3.6.2).
  - ``has_deepstack``: ``bool`` — whether :func:`merge` produces a deepstack side-channel.
  - ``encode_image`` / ``encode_video`` / ``encode_audio`` — per-model tower encoders;
    their presence also derives ``supported_modalities`` and ``is_multimodal``.
  - ``supported_modalities`` (optional explicit override of the derivation).
  - ``is_multimodal`` (optional explicit marker; rarely needed).

Pure python — unit-testable on any interpreter.
"""

from __future__ import annotations

# modality name -> the method a model class implements to encode it.
_MODALITY_ENCODERS = {
    "image": "encode_image",
    "video": "encode_video",
    "audio": "encode_audio",
}


def supported_modalities(model_cls) -> set[str]:
    """Modalities a model class can encode.

    Derived from which ``encode_<modality>`` methods exist, unless the class declares an
    explicit (non-callable) ``supported_modalities`` attribute.
    """
    explicit = getattr(model_cls, "supported_modalities", None)
    if explicit is not None and not callable(explicit):
        return set(explicit)
    return {
        modality
        for modality, attr in _MODALITY_ENCODERS.items()
        if callable(getattr(model_cls, attr, None))
    }


def is_multimodal_arch(model_cls) -> bool:
    """Whether a registered model class is a multimodal (understanding) model.

    Source of truth = the model-class self-declaration (U3): multimodal iff it declares
    an explicit ``is_multimodal`` marker or at least one modality encoder. No separate
    static arch set (kills the as-is ``"xxx" in model_path`` substring checks). A
    deployment-level ``server_args.multimodal`` override can still force text-only serving.
    """
    if getattr(model_cls, "is_multimodal", False):
        return True
    return bool(supported_modalities(model_cls))


def audio_kind(model_cls) -> str | None:
    """Model-level audio kind: ``"codes"`` (discrete RVQ) | ``"features"`` (continuous) | None.

    Replaces the as-is per-item runtime flag ``_is_codes_audio`` (design §3.6.2): a model's
    audio is uniformly one kind, so dispatch keys off this declaration, not a runtime field.
    """
    return getattr(model_cls, "audio_kind", None)


def has_deepstack(model_cls) -> bool:
    """Whether :func:`merge` should emit a deepstack side-channel for this model."""
    return bool(getattr(model_cls, "has_deepstack", False))
