"""Containers for scheduler-built in-model multimodal encode/merge plans.

These are modality-agnostic: encode inputs are an opaque pytree and merge
arrays are plain index tensors, so the host orchestration does not need to know
what modality produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sgl_jax.srt.multimodal.common.modality_enum import Modality

if TYPE_CHECKING:
    import jax
    import numpy as np


@dataclass
class DeviceMergePlan:
    """Token-shaped routing from encoder rows to backbone embedding rows."""

    src_idx: np.ndarray | jax.Array  # [dp, tp, per_dp_token]
    mask: np.ndarray | jax.Array  # [dp, tp, per_dp_token]


@dataclass
class ModalityEmbedBatch:
    """One encoder invocation and its token merge routing."""

    # A registered pytree whose array leaves share leading ``[dp, tp]`` axes,
    # produced by a modality's plan builder. Kept modality-agnostic here: host
    # orchestration only tree-maps over it, never reads its fields.
    encode_inputs: Any
    merge: DeviceMergePlan
    source_capacity: int | None = None


# One encoder batch per modality for one language-model forward.
MultimodalEmbedPlan = dict[Modality, ModalityEmbedBatch]
