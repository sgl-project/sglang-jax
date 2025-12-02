from __future__ import annotations

# Backwards-compat import path for models/configs that reference
# `sgl_jax.srt.layers.fused_moe.FusedEPMoE`.
from .moe import FusedEPMoE

__all__ = ["FusedEPMoE"]
