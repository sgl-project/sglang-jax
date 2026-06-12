"""Qwen-VL vision-config helpers (srt-side).

Moved out of ``multimodal/configs/config_registry.py`` so the in-model Qwen2.5-VL
wrapper (``srt/models/qwen2_5VL``) can build its vision config without an
``srt -> multimodal`` import edge. The generation-side ``config_registry``
re-imports these helpers (``multimodal -> srt`` is allowed).
"""

import contextlib
import json
import logging
import os

from sgl_jax.srt.configs.qwen_vl.qwen_2_5_vl_config import QwenVLModelVitConfig

logger = logging.getLogger(__name__)


_QWEN_VL_VISION_KEY_MAP = {
    "depth": "depth",
    "num_hidden_layers": "depth",
    "hidden_size": "hidden_size",
    "intermediate_size": "intermediate_size",
    "num_attention_heads": "num_heads",
    "num_heads": "num_heads",
    "in_channels": "in_channels",
    "patch_size": "patch_size",
    "spatial_merge_size": "spatial_merge_size",
    "temporal_patch_size": "temporal_patch_size",
    "tokens_per_second": "tokens_per_second",
    "window_size": "window_size",
    "out_hidden_size": "out_hidden_size",
    "output_hidden_size": "out_hidden_size",
    "rms_norm_eps": "rms_norm_eps",
    "initializer_range": "initializer_range",
    "fullatt_block_indexes": "fullatt_block_indexes",
    "full_attn_block_indexes": "fullatt_block_indexes",
}
_QWEN_VL_VISION_INT_FIELDS = {
    "depth",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "in_channels",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "tokens_per_second",
    "window_size",
    "out_hidden_size",
}
_QWEN_VL_VISION_FLOAT_FIELDS = {
    "initializer_range",
    "rms_norm_eps",
}
_QWEN_VL_VISION_LIST_FIELDS = {
    "fullatt_block_indexes",
}


def _load_local_config_dict(model_path: str) -> dict | None:
    if not isinstance(model_path, str):
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to read config.json from %s: %s", config_path, exc)
        return None


def _apply_qwen_vl_vision_overrides(
    config: QwenVLModelVitConfig, model_path: str
) -> QwenVLModelVitConfig:
    config_dict = _load_local_config_dict(model_path)
    if not config_dict:
        return config

    vision_cfg = (
        config_dict.get("vision_config")
        or config_dict.get("vision_config_dict")
        or config_dict.get("vision_cfg")
    )
    if not isinstance(vision_cfg, dict):
        return config

    updated_fields: set[str] = set()
    for src_key, dst_attr in _QWEN_VL_VISION_KEY_MAP.items():
        if src_key not in vision_cfg:
            continue
        value = vision_cfg[src_key]
        if dst_attr in _QWEN_VL_VISION_INT_FIELDS:
            with contextlib.suppress(Exception):
                value = int(value)
        elif dst_attr in _QWEN_VL_VISION_FLOAT_FIELDS:
            with contextlib.suppress(Exception):
                value = float(value)
        elif dst_attr in _QWEN_VL_VISION_LIST_FIELDS and not isinstance(value, list):
            value = list(value)
        setattr(config, dst_attr, value)
        updated_fields.add(dst_attr)

    if (
        "out_hidden_size" not in updated_fields
        and (top_hidden_size := config_dict.get("hidden_size")) is not None
    ):
        with contextlib.suppress(Exception):
            top_hidden_size = int(top_hidden_size)
        config.out_hidden_size = top_hidden_size

    if updated_fields:
        logger.info("Loaded QwenVL vision config overrides from %s", model_path)
    return config


def qwen_vl_vision_config_from_hf(hf_config) -> QwenVLModelVitConfig:
    """Build a QwenVLModelVitConfig from a parsed HF Qwen2.5-VL config object.

    Mirrors :func:`_apply_qwen_vl_vision_overrides` but sources the already-parsed
    ``hf_config.vision_config`` (robust to ``model_path`` being a hub id rather than a
    local dir). Used by the in-model Qwen2.5-VL (refactor M3): without this override the
    bare ``QwenVLModelVitConfig()`` default ``hidden_size=3584`` is wrong for the 7B ViT
    (real vision width is 1280; 3584 is the post-merger LLM dim), so the ViT patch_embed
    reshape fails. ``out_hidden_size`` falls back to the top-level ``hidden_size``.
    """
    config = QwenVLModelVitConfig()
    vision = getattr(hf_config, "vision_config", None)
    if vision is None:
        return config
    vision_cfg = vision.to_dict() if hasattr(vision, "to_dict") else dict(vars(vision))
    if not isinstance(vision_cfg, dict):
        return config

    updated_fields: set[str] = set()
    for src_key, dst_attr in _QWEN_VL_VISION_KEY_MAP.items():
        if src_key not in vision_cfg:
            continue
        value = vision_cfg[src_key]
        if dst_attr in _QWEN_VL_VISION_INT_FIELDS:
            with contextlib.suppress(Exception):
                value = int(value)
        elif dst_attr in _QWEN_VL_VISION_FLOAT_FIELDS:
            with contextlib.suppress(Exception):
                value = float(value)
        elif dst_attr in _QWEN_VL_VISION_LIST_FIELDS and not isinstance(value, list):
            value = list(value)
        setattr(config, dst_attr, value)
        updated_fields.add(dst_attr)

    if "out_hidden_size" not in updated_fields:
        top_hidden_size = getattr(hf_config, "hidden_size", None)
        if top_hidden_size is not None:
            with contextlib.suppress(Exception):
                top_hidden_size = int(top_hidden_size)
            config.out_hidden_size = top_hidden_size
    return config
