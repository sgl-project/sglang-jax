from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem

MultimodalEncodeFunc = Callable[[list[list[MultimodalDataItem]]], jax.Array]
MultimodalEncodeFuncs = Mapping[Modality, MultimodalEncodeFunc]


@dataclass(frozen=True)
class WaveformInputSpec:
    sample_rate: int
    channels: int
    dtype: np.dtype


@dataclass(frozen=True)
class MelInputSpec:
    num_mel_bins: int
    time_alignment: int
    dtype: np.dtype


@dataclass(frozen=True)
class VisionInputSpec:
    patch_dim: int
    spatial_merge_size: int
    dtype: np.dtype = np.dtype("float32")


class InModelMultimodalContract(ABC):
    mesh: Mesh | None = None

    deepstack_visual_layers: int = 0

    vision_input_spec: VisionInputSpec | None = None

    audio_input_spec = None

    @abstractmethod
    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        raise NotImplementedError

    def get_multimodal_encode_funcs(self) -> MultimodalEncodeFuncs:
        """Return encoders accepting items per lane, with outputs in lane-major item order."""
        return {}
