from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import nnx
from jax.tree_util import register_pytree_node_class

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


def get_batch_page_indices(
    batch: ModelWorkerBatch, page_size: int, *, max_pages_per_dp: int | None = None
) -> np.ndarray:
    """Return host page IDs as DP rows, bounded before converting token locations."""
    page_indices = getattr(batch, "allocated_page_indices", None)
    if page_indices is not None:
        return page_indices.reshape(batch.dp_size, -1)[:, :max_pages_per_dp]
    cache_loc = batch.cache_loc.reshape(batch.dp_size, -1)
    return cache_loc[:, ::page_size][:, :max_pages_per_dp] // page_size


@register_pytree_node_class
@dataclass
class AttentionBackendMetadata:
    """Empty pytree base type for per-backend forward metadata.

    Concrete backends (FlashAttention, MLA, ...) subclass this so
    HybridLinearAttnBackendMetadata can type its `full_attn_metadata` field
    without depending on any specific concrete backend.
    """

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


class AttentionBackend(nnx.Module):
    """The base class of attention backends"""

    @abstractmethod
    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it"""
        raise NotImplementedError()

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        raise NotImplementedError()

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        raise NotImplementedError()
