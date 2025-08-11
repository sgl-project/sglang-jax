from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionBackend(nnx.Module):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
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
