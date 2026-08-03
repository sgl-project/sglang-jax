from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax.sharding import Mesh
from jax.typing import ArrayLike

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem


class Placement(NamedTuple):

    row: int
    offset: int
    length: int


@dataclass(frozen=True)
class PackedMultimodalEmbedding:
    """The encoder's native, packed-to-device output (the sole output contract).

    ``output`` is the encoder result before it is sliced back to per-item true
    lengths.  Its leading dimension ``num_lanes`` is a mesh constant (one row per
    encoder lane == device) and its capacity dimension ``cap`` is bucketed, so
    the merge can gather from it with a fixed number of compiled shapes (never
    keyed on image size or request packing).  ``num_lanes`` and ``cap`` are
    carried explicitly so the merge (and the paged embedding pool) can allocate
    bucketed buffers without peeking at ``output.shape``.

    ``placements[i]`` locates the i-th input item inside ``output`` as a
    :class:`Placement` ``(row, offset, length)``.

    Deepstack (Qwen3-VL's auxiliary per-layer features) is carried as extra
    feature *planes* concatenated onto ``output``'s trailing axis rather than a
    separate tensor: the feature width is ``(1 + deepstack_dim) * H``, where the
    primary token embedding is ``output[..., :H]`` and the ``deepstack_dim``
    deepstack planes are ``output[..., H:]`` reshaped to ``[..., deepstack_dim,
    H]``. Both planes share the *same* ``placements`` and are gathered by one
    kernel; ``deepstack_dim == 0`` means no deepstack. This keeps a single output
    tensor, one gather, and one pool buffer.
    """

    output: ArrayLike  # [num_lanes, cap, (1 + deepstack_dim) * H], mesh-replicated
    placements: tuple[Placement, ...]
    num_lanes: int
    cap: int
    deepstack_dim: int = 0  # D; deepstack planes live in output[..., H:]


MultimodalEncodeFunc = Callable[[list[MultimodalDataItem]], PackedMultimodalEmbedding]
MultimodalEncodeFuncs = Mapping[Modality, MultimodalEncodeFunc]


class InModelMultimodalContract(ABC):
    mesh: Mesh | None = None
    deepstack_visual_layers: int = 0

    def precompile_multimodal(self) -> None:
        """Warm model-specific multimodal encoders."""
        return None

    def get_multimodal_embedding_packed_shapes(self) -> tuple[tuple[int, int], ...]:
        """Return finite ``(num_lanes, cap)`` encoder-output shapes for warmup."""
        return ()

    @abstractmethod
    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        raise NotImplementedError

    def get_multimodal_encode_funcs(self) -> MultimodalEncodeFuncs:
        """Per-modality encoders that produce the native packed output.

        Each function runs the model's visual encoder over a list of items and
        returns its ``PackedMultimodalEmbedding`` -- the sole embedding contract
        the merge gathers directly from, with a bounded number of compiled
        shapes.  A modality absent here has no embedding path and the merge
        raises for it.
        """
        return {}
