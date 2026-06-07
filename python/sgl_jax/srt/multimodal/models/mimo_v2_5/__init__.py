"""MiMo-V2.5 omni model package (audio + text landed; vision is follow-up).

Exports use lazy attribute loading (PEP 562) so light submodules such as
``configuration`` can be imported without pulling the jax/flax model stack.
"""

from __future__ import annotations

__all__ = [
    "MiMoV2_5Embedding",
    "MiMoV25AudioUnderstandingEncoder",
    "MiMoV25AudioCodecProcessor",
    "MiMoV25AudioPayload",
    "MiMoV25Processor",
]

_LAZY = {
    "MiMoV2_5Embedding": ("embedding", "MiMoV2_5Embedding"),
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
