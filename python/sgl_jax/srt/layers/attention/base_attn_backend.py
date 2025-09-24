from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import jax
from flax import nnx
from jax.sharding import Mesh

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


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
