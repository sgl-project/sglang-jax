"""Containers for scheduler-built multimodal encode/merge plans.

The scheduler creates these dataclasses with numpy leaves. ``ForwardBatch``
places those leaves on device before the model runner consumes the plan. The
``meta`` field is the only model-specific payload and is kept opaque to common
code via ``VisionMetadataPytree``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sgl_jax.srt.multimodal.common.vision_metadata import VisionMetadataPytree

if TYPE_CHECKING:
    import jax
    import numpy as np

    from sgl_jax.srt.multimodal.common.modality_enum import Modality


@dataclass
class VisionEncodeInputs:
    """Model/modality-specific encode payload for one owning-rank DP round.

    Two-state array fields (``np.ndarray`` host -> ``jax.Array`` device):

    - ``pixels``: ``[dp, patch_k, dim]`` -- per-rank round-k image patches.
    - ``valid``:  ``[dp]``               -- real patch-row count per rank's
      round-k image.
    - ``meta``:   per-arch ViT-aux registered pytree (opaque to common; see
      :class:`VisionMetadataPytree`). Crosses the encode JIT.
    """

    pixels: np.ndarray | jax.Array
    valid: np.ndarray | jax.Array
    meta: VisionMetadataPytree


@dataclass
class EmbedRound:
    """One owning-rank DP round: one single-image ViT per rank, then merge.

    ``src_idx``/``mask`` are integer/bool arrays produced by the scheduler that
    drive the ``where(mask, features[src_idx], running)`` merge -- no device
    cumsum. Two-state (``np.ndarray`` host -> ``jax.Array`` device).
    """

    encode_inputs: VisionEncodeInputs
    src_idx: np.ndarray | jax.Array  # [total_token] int   token -> features row
    mask: np.ndarray | jax.Array  # [total_token] bool


@dataclass
class MultimodalEmbedPlan:
    """Per-modality rounds. Empty ``rounds_by_modality`` => text-only forward."""

    rounds_by_modality: dict[Modality, list[EmbedRound]]
