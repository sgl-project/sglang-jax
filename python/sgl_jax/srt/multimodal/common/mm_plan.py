"""Modality-general containers for the in-model multimodal embed plan.

The scheduler builds a :class:`MultimodalEmbedPlan` host-side (numpy arrays), the
forward path device_puts its array leaves onto ``P("data")`` in
``ForwardBatch.init_new``, and ``general_mm_embed_routine`` -> ``embed_mm_inputs``
strings the per-round JIT segments using it. Common code treats
``VisionEncodeInputs`` as an opaque pytree payload; only the model-provided
encode body interprets its fields.

These are plain dataclasses (not registered pytrees). They are host-side
containers whose array fields are swapped from numpy to device arrays in place by
``init_new``; nothing here is threaded through the backbone JIT.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class VisionEncodeInputs:
    """Model/modality-specific encode payload (opaque to common code).

    Fields hold one round's pad-stacked-across-ranks vision inputs:

    - ``pixels``: ``[dp, patch_k, dim]`` -- per-rank round-k image patches.
    - ``grid``:   ``[dp, 3]``           -- passthrough ``item.image_grid_thw``
      (dummy ranks = zeros).
    - ``valid``:  ``[dp]``              -- real patch-row count per rank's
      round-k image.

    No ``aux`` field: aux 落点 = Design X -- the model's ``get_image_feature``
    computes the ViT aux arrays from ``grid`` at encode time, so the plan carries
    only ``grid`` (host-side), never the derived aux.
    """

    pixels: Any
    grid: Any
    valid: Any


@dataclass
class EmbedRound:
    """One owning-rank DP round: one single-image ViT per rank, then merge.

    ``src_idx``/``mask`` are integer arrays produced by the scheduler that drive
    the ``where(mask, features[src_idx], running)`` merge -- no device cumsum.
    """

    encode_inputs: Any  # VisionEncodeInputs (opaque)
    src_idx: Any  # [total_token] int   token -> features row
    mask: Any  # [total_token] bool
    out_rows: int


@dataclass
class MultimodalEmbedPlan:
    """Per-modality rounds. Empty ``rounds_by_modality`` => text-only forward."""

    rounds_by_modality: dict  # {Modality.IMAGE: [EmbedRound, ...]}
