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
  - ``supported_modalities`` — the RECOMMENDED authoritative declaration (review code-review.md
    §11.6): an explicit set/tuple like ``("image", "video", "audio")``. Declaring it decouples
    capability from method names, so an encoder can be renamed (or video can ride ``encode_image``
    with no ``encode_video``) without ``is_multimodal`` silently going wrong. ``is_multimodal`` and
    the modality set derive from it when present.
  - ``encode_image`` / ``encode_video`` / ``encode_audio`` — per-model tower encoders. When a class
    does NOT declare ``supported_modalities``, their presence is the fallback derivation of
    ``supported_modalities`` and ``is_multimodal`` (ergonomic default; method-name dependent).
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

# The keyword params ModelRunner.encode_mm_reqs / jitted_embed_mm pass to EVERY model.embed_mm
# (beyond the positional input_ids). The "uniform embed_mm signature" the model docstrings promise
# is otherwise unenforced: V-2 added mm_real_* and M5 added mm_audio_codes without updating the
# other two models, so their media requests TypeError'd at encode (review H-1). reconcile_mm_capability
# asserts at startup that every mm-capable model accepts all of these (or **kwargs). Keep in lockstep
# with model_runner.jitted_embed_mm.
EMBED_MM_CONTRACT_PARAMS = (
    "mm_pixel_values",
    "mm_grid_thw",
    "mm_pixel_values_videos",
    "mm_video_grid_thw",
    "mm_audio_features",
    "mm_audio_feature_lengths",
    "mm_audio_codes",
    "mm_real_llm_dims",
    "mm_real_video_llm_dims",
)


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

    A model's audio is uniformly one kind, so this is the intended model-level declaration for audio
    routing (design §3.6.2). NOTE (not yet wired): the host assembler still dispatches per item via
    ``mm_assembly._is_codes_audio`` (the ``is_codes`` mm_item meta flag); consolidating routing onto
    this declaration is a follow-up.
    """
    return getattr(model_cls, "audio_kind", None)


def has_deepstack(model_cls) -> bool:
    """Whether :func:`merge` should emit a deepstack side-channel for this model."""
    return bool(getattr(model_cls, "has_deepstack", False))


# ----- ★5 startup reconciliation (design §3.5.5) -----


def find_capability_inconsistencies(
    registered_archs: dict,
    processor_archs,
    served_archs,
    served_is_multimodal,
) -> list[str]:
    """Pure core of :func:`reconcile_mm_capability` (unit-testable without the real registries).

    ``registered_archs`` maps arch name -> model class (the srt ModelRegistry contents);
    ``processor_archs`` is the set of arch names with a registered processor; ``served_archs`` is
    the served model's ``hf_config.architectures``; ``served_is_multimodal`` is the value the
    config-proxy (:func:`is_multimodal_model`) produced. Returns a list of human-readable
    inconsistency messages (empty = consistent).
    """
    errors: list[str] = []
    proc = set(processor_archs)

    # (1) Served model: the per-class capability (the U3 truth source) must agree with the
    # lightweight hf_config proxy that ModelConfig used. (A deployment override that forces a VLM
    # to text-only would legitimately diverge -- not modeled until the enable_multimodal tri-state
    # lands in G3; today no such override exists so they must match.)
    for arch in served_archs or []:
        cls = registered_archs.get(arch)
        if cls is not None:
            cap = is_multimodal_arch(cls)
            if cap != bool(served_is_multimodal):
                errors.append(
                    f"is_multimodal mismatch for {arch!r}: hf_config proxy="
                    f"{bool(served_is_multimodal)} but model-class capability={cap} -- the proxy "
                    "(is_multimodal_model) and the class capability (is_multimodal_arch) must agree"
                )
            break

    # (2) Global: every registered mm-capable model class must have a registered processor, else
    # its media inputs are silently dropped (the 'added a VLM, forgot its processor' failure).
    missing = sorted(
        arch
        for arch, cls in registered_archs.items()
        if is_multimodal_arch(cls) and arch not in proc
    )
    if missing:
        errors.append(
            f"multimodal model archs declare capability but have no registered processor: "
            f"{missing} -- register a BaseMultimodalProcessor(models=[...]) for each"
        )

    # (3) Uniform embed_mm contract (review H-1): ModelRunner.encode_mm_reqs calls model.embed_mm
    # with a fixed kwarg set (EMBED_MM_CONTRACT_PARAMS); every mm-capable model must accept all of
    # them (or **kwargs), else media requests TypeError at the first encode trace. Enforce here so
    # the "uniform signature" the docstrings promise can never silently drift again.
    import inspect

    for arch, cls in registered_archs.items():
        if not is_multimodal_arch(cls):
            continue
        fn = getattr(cls, "embed_mm", None)
        if fn is None:
            errors.append(f"{arch!r} is multimodal but defines no embed_mm")
            continue
        params = inspect.signature(fn).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue  # **kwargs accepts the whole contract
        missing_kw = [k for k in EMBED_MM_CONTRACT_PARAMS if k not in params]
        if missing_kw:
            errors.append(
                f"{arch!r}.embed_mm is missing encode-pass params {missing_kw}; the uniform "
                f"embed_mm contract requires {list(EMBED_MM_CONTRACT_PARAMS)} (or **kwargs)"
            )
    return errors


def reconcile_mm_capability(model_config) -> None:
    """★5 (design §3.5.5): assert the per-class capability truth source is coherent at startup.

    Run at TokenizerManager init after both the srt ModelRegistry and the mm ProcessorRegistry are
    populated. Makes ``mm_core.capability`` the *enforced* source of truth: the lightweight
    ``is_multimodal_model`` proxy (used early at ModelConfig build, before model-class resolution)
    is reconciled against the class capability, and every mm-capable model must have a processor.
    Raises ``AssertionError`` on any inconsistency rather than failing silently at request time.
    """
    from sgl_jax.srt.mm_core.processor import supported_processor_archs
    from sgl_jax.srt.models.registry import ModelRegistry

    served_archs = getattr(getattr(model_config, "hf_config", None), "architectures", None) or []
    errors = find_capability_inconsistencies(
        ModelRegistry.models,
        supported_processor_archs(),
        served_archs,
        bool(getattr(model_config, "is_multimodal", False)),
    )
    if errors:
        raise AssertionError(
            "multimodal capability reconciliation failed:\n  - " + "\n  - ".join(errors)
        )
