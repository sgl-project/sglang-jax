"""MiMo-V2.5 config helpers, used by the in-model wrapper.

Originally extracted from the staged ``MiMoV2_5Embedding`` static methods (M6-S2) so the in-model
wrapper (``mimo_v2_5_inmodel.py``) didn't import the staged embed class for them; that staged embed
model has since been removed (M6-S5). Pure python.
"""

from __future__ import annotations

from types import SimpleNamespace

from transformers import PretrainedConfig


def get_config_value(config: PretrainedConfig, key: str, default=None):
    """Read ``key`` from the HF config, falling back to its ``processor_config``.

    MiMo-V2.5 keeps several token ids (e.g. ``audio_token_id``) only inside
    ``processor_config``, so fall back to it when the top-level attr is missing.
    """
    value = getattr(config, key, None)
    if value is not None:
        return value
    processor_config = getattr(config, "processor_config", None)
    if isinstance(processor_config, dict):
        return processor_config.get(key, default)
    if processor_config is not None:
        return getattr(processor_config, key, default)
    return default


def normalize_vision_config(vision_config):
    """Adapt the checkpoint's vision_config to what MiMoVisionTransformer expects.

    The HF checkpoint stores ``in_chans`` (not ``in_channels``) and omits ``qk_channels``;
    the real ViT head_dim is ``qk_channels`` default 64 (review D1-1: 1280/32=40 is wrong --
    the model uses getattr(config,"qk_channels",64) and vision_config has no qk_channels). Wrap
    the config to supply both without mutating the shared hf_config. A wrong head_dim is caught
    at weight load (the qkv shape won't match), so this is fail-safe.
    """

    def g(key, default=None):
        if isinstance(vision_config, dict):
            return vision_config.get(key, default)
        return getattr(vision_config, key, default)

    in_channels = g("in_channels", g("in_chans", 3))
    qk_channels = g("qk_channels", 64)
    # Carry every original field through, then override the two normalized names.
    base = dict(vision_config) if isinstance(vision_config, dict) else dict(vars(vision_config))
    base["in_channels"] = int(in_channels)
    base["qk_channels"] = int(qk_channels)
    return SimpleNamespace(**base)
