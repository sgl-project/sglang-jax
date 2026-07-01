"""Modality-general containers for the in-model multimodal embed plan.

The scheduler builds a :class:`MultimodalEmbedPlan` host-side (numpy arrays), the
forward path device_puts its array leaves onto ``P("data")`` in
``ForwardBatch.init_new``, and ``general_mm_embed_routine`` -> ``embed_mm_inputs``
strings the per-round JIT segments using it.

The plan containers (``MultimodalEmbedPlan`` / ``EmbedRound`` /
``VisionEncodeInputs``) are plain dataclasses (not registered pytrees) and are
**two-state**: array fields are numpy when the scheduler builds them and are
swapped to ``jax.Array`` in place by ``init_new`` after device_put. That two-state
nature is declared explicitly as ``np.ndarray | jax.Array`` (not ``Any``); the
caller knows which state it holds by context (scheduler = numpy, post-``init_new``
= device).

The embedded ``meta`` payload *is* a registered pytree, but a PER-ARCH one
(e.g. ``Qwen25VLVisionMetadata`` in ``models/vision_metadata/qwen2_5_vl.py``).
Common code must not name its fields nor import the concrete class (keeps common
decoupled from models), so it is typed by the ``VisionMetadataPytree`` structural
marker defined in ``common/vision_metadata.py``. ``meta`` is the only opaque
per-arch pytree payload; ``pixels`` and ``valid`` are common tensor inputs to the
encode JIT.
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
