"""MiMo-V2.5 omni model package (text + audio + image + video understanding).

Exports use lazy attribute loading (PEP 562) so importing the package does not
pull the jax/flax model stack until a model class is actually accessed. The audio
tower (``audio_encoder``) and ViT (``vision_encoder``) feed the in-model wrapper
(``mimo_v2_5_inmodel``).
"""

from __future__ import annotations

__all__ = [
    "MiMoV25AudioUnderstandingEncoder",
    "MiMoV25AudioCodecProcessor",
    "MiMoV25AudioPayload",
    "MiMoV25Processor",
]

_LAZY = {
    "MiMoV25AudioUnderstandingEncoder": ("audio_encoder", "MiMoV25AudioUnderstandingEncoder"),
    "MiMoV25AudioCodecProcessor": ("audio_codec_processor", "MiMoV25AudioCodecProcessor"),
    "MiMoV25AudioPayload": ("audio_codec_processor", "MiMoV25AudioPayload"),
    "MiMoV25Processor": ("processor", "MiMoV25Processor"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{target[0]}")
    return getattr(module, target[1])


def __dir__():
    return sorted(__all__)
