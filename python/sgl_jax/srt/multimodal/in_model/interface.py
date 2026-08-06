from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax.sharding import Mesh

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
    exposed as properties derived from ``output.shape`` -- they are contract, not
    state, so they can never disagree with the tensor they describe.

    ``placements[i]`` locates the i-th input item inside ``output`` as a
    :class:`Placement` ``(row, offset, length)``.

    Deepstack (Qwen3-VL's auxiliary per-layer features) is carried as extra
    feature *planes* concatenated onto ``output``'s trailing axis rather than a
    separate tensor: the feature width is ``(1 + deepstack_dim) * H``, where the
    primary token embedding is :attr:`primary` (``output[..., :H]``) and the
    ``deepstack_dim`` deepstack planes are :attr:`deepstack`
    (``output[..., H:]`` reshaped to ``[..., deepstack_dim, H]``). Both planes
    share the *same* ``placements`` and are gathered by one kernel;
    ``deepstack_dim == 0`` means no deepstack (:attr:`deepstack` is ``None``).
    This keeps a single output tensor, one gather, and one pool buffer.

    The invariants tying these fields together -- 3-D output, a feature width
    divisible by ``1 + deepstack_dim``, and placements within ``(num_lanes,
    cap)`` -- are enforced at construction, so a malformed encoder output fails
    at its source rather than deep inside the merge or the pool.
    """

    output: jax.Array  # [num_lanes, cap, (1 + deepstack_dim) * H], mesh-replicated
    placements: tuple[Placement, ...]
    deepstack_dim: int = 0  # D; deepstack planes live in output[..., H:]

    def __post_init__(self) -> None:
        if getattr(self.output, "ndim", None) != 3:
            raise ValueError(
                f"packed output must be [num_lanes, cap, (1+D)*H], got shape "
                f"{getattr(self.output, 'shape', None)}"
            )
        width = self.output.shape[2]
        if self.deepstack_dim < 0 or width % (1 + self.deepstack_dim) != 0:
            raise ValueError(
                f"feature width {width} not divisible by (1 + deepstack_dim={self.deepstack_dim})"
            )
        # Accept bare 3-tuples and normalise to Placement so downstream code can
        # rely on named access.
        object.__setattr__(
            self, "placements", tuple(Placement(*placement) for placement in self.placements)
        )
        num_lanes, cap = self.num_lanes, self.cap
        for placement in self.placements:
            if (
                not 0 <= placement.row < num_lanes
                or placement.offset < 0
                or placement.length < 0
                or placement.offset + placement.length > cap
            ):
                raise ValueError(f"placement {placement} out of packed bounds {(num_lanes, cap)}")

    @property
    def num_lanes(self) -> int:
        return self.output.shape[0]

    @property
    def cap(self) -> int:
        return self.output.shape[1]

    @property
    def hidden(self) -> int:
        """Per-token primary width ``H`` (feature width without deepstack planes)."""
        return self.output.shape[2] // (1 + self.deepstack_dim)

    @property
    def primary(self) -> jax.Array:
        """The primary token embedding, ``output[..., :H]``."""
        return self.output[..., : self.hidden]

    @property
    def deepstack(self) -> jax.Array | None:
        """Deepstack planes ``[..., deepstack_dim, H]``, or ``None`` when absent."""
        if self.deepstack_dim == 0:
            return None
        hidden = self.hidden
        return self.output[..., hidden:].reshape(
            *self.output.shape[:-1], self.deepstack_dim, hidden
        )


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
