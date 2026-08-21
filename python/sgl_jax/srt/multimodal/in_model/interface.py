from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import jax
from jax.sharding import Mesh

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem

MultimodalEncodeFunc = Callable[[list[MultimodalDataItem]], jax.Array]
MultimodalEncodeFuncs = Mapping[Modality, MultimodalEncodeFunc]


class InModelMultimodalContract(ABC):
    mesh: Mesh | None = None
    deepstack_visual_layers: int = 0

    def precompile_multimodal(self) -> None:
        """Warm model-specific multimodal encoders."""
        return None

    def get_multimodal_embedding_packed_capacities(self) -> tuple[int, ...]:
        """Return finite packed-array capacities for warmup."""
        return ()

    @abstractmethod
    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        raise NotImplementedError

    def get_multimodal_encode_funcs(self) -> MultimodalEncodeFuncs:
        """Return per-modality encoders producing padded, item-ordered arrays."""
        return {}
