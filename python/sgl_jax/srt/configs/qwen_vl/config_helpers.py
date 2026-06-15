"""Qwen-VL vision-config normalization (srt-side, in-model).

The in-model Qwen2.5-VL wrapper (``srt/models/qwen2_5VL``) builds its ViT directly from the parsed
HF ``hf_config.vision_config`` -- no per-model dataclass with hardcoded defaults (review code-review
§2/§13). Variant-varying dims (``hidden_size`` / ``num_heads`` / ``depth`` / ...) MUST come from the
checkpoint; a missing one RAISES rather than silently using a wrong constant -- the deleted
``QwenVLModelVitConfig`` defaulted ``hidden_size=3584`` (the 7B post-merger LLM dim), which silently
loaded then crashed the 7B ViT's patch_embed at first forward. We read the *parsed* vision_config (an
HF config object, so HF has already filled its own field defaults), so for any real checkpoint every
field is present with the real value; the raise is a safety net for a genuinely malformed config.

Naming: Qwen2.5-VL checkpoints already use the canonical field names the ViT reads
(``depth`` / ``num_heads`` / ``out_hidden_size`` / ``fullatt_block_indexes``); the aliases below cover
sibling Qwen-VL configs. The one field HF's vision_config does not carry is the RMSNorm eps -- a fixed
architecture constant, not a per-variant dim.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# Qwen2.5-VL vision RMSNorm eps. HF's Qwen2_5_VLVisionConfig does not expose it; preserve the value
# the M3-validated path used (the old QwenVLModelVitConfig default). NOTE: the merger reads this via
# the wrapper's norm_eps kwarg and the blocks via config.rms_norm_eps -- keep them in sync here.
_QWEN_VL_VISION_RMS_NORM_EPS = 1e-5

# Canonical ViT field name -> HF vision_config key(s) it may appear under (first present wins). All
# are variant-varying dims: a missing one is an error, never defaulted.
_REQUIRED_INT_FIELDS = {
    "depth": ("depth", "num_hidden_layers"),
    "hidden_size": ("hidden_size",),
    "intermediate_size": ("intermediate_size",),
    "num_heads": ("num_heads", "num_attention_heads"),
    "in_channels": ("in_channels", "in_chans"),
    "patch_size": ("patch_size",),
    "spatial_merge_size": ("spatial_merge_size",),
    "temporal_patch_size": ("temporal_patch_size",),
    "window_size": ("window_size",),
}


def _first(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_qwen_vl_vision_config(hf_config) -> SimpleNamespace:
    """Normalize the parsed HF Qwen2.5-VL ``vision_config`` into the attribute set the in-model ViT
    reads. Sources every value from the checkpoint (no value defaults for variant dims -- a missing
    one raises). ``out_hidden_size`` falls back to the top-level LLM ``hidden_size`` (merger out dim
    == LLM in dim). ``rms_norm_eps`` is the one HF-absent architecture constant. Returns a
    SimpleNamespace so the ViT keeps reading ``config.<field>``.
    """
    vision = getattr(hf_config, "vision_config", None)
    if vision is None:
        raise ValueError(
            "Qwen2.5-VL hf_config has no vision_config -- cannot build the in-model ViT "
            "(a text-only checkpoint must not reach the VLM wrapper)."
        )
    vcfg = vision.to_dict() if hasattr(vision, "to_dict") else dict(vars(vision))

    out: dict = {}
    for field, keys in _REQUIRED_INT_FIELDS.items():
        val = _first(vcfg, keys)
        if val is None:
            raise ValueError(
                f"Qwen2.5-VL vision_config is missing required field {field!r} (looked under "
                f"{keys}); refusing to guess a default (the old hidden_size=3584 default was the "
                "bug this removes)."
            )
        out[field] = int(val)

    # hidden_act: present in HF vision_config (e.g. "silu").
    out["hidden_act"] = _first(vcfg, ("hidden_act",)) or "silu"

    # out_hidden_size (merger output dim): read, else fall back to the top-level LLM hidden_size.
    out_hidden = _first(vcfg, ("out_hidden_size", "output_hidden_size"))
    if out_hidden is None:
        out_hidden = getattr(hf_config, "hidden_size", None)
    if out_hidden is None:
        raise ValueError(
            "Qwen2.5-VL: neither vision_config.out_hidden_size nor top-level hidden_size present."
        )
    out["out_hidden_size"] = int(out_hidden)

    # fullatt_block_indexes: windowed-attention full-attn layer indices.
    fullatt = _first(vcfg, ("fullatt_block_indexes", "full_attn_block_indexes"))
    out["fullatt_block_indexes"] = list(fullatt) if fullatt is not None else []

    # rms_norm_eps: HF vision_config omits it -> fixed architecture constant.
    eps = _first(vcfg, ("rms_norm_eps",))
    out["rms_norm_eps"] = float(eps) if eps is not None else _QWEN_VL_VISION_RMS_NORM_EPS

    return SimpleNamespace(**out)
