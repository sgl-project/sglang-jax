"""GDN attention metadata: per-forward boundaries for packed ragged batches.

Built once on the host in :class:`GDNMetadataBuilder.get_forward_metadata` and
passed through :class:`ForwardBatch` as a JIT-traced pytree, so all GDN layers
in the model share one materialised metadata object instead of each layer
recomputing the cumsum.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array


@register_pytree_node_class
@dataclass
class GDNAttnMetadata:
    """Per-forward metadata for Gated-Delta-Net layers.

    For DECODE batches (one token per request) no boundary metadata is needed
    and ``cu_seqlens`` is ``None``.  For EXTEND/PREFILL batches, ``cu_seqlens``
    holds packed-buffer boundaries: ``cu_seqlens[i]`` is the start of request
    ``i`` and ``cu_seqlens[-1]`` is the total token count.
    """

    cu_seqlens: jax.Array | None = None  # [B+1] int32

    def tree_flatten(self):
        return (self.cu_seqlens,), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(cu_seqlens=children[0])


class GDNMetadataBuilder(nnx.Module):
    """Pre-computes per-forward GDN metadata.

    All GDN layers share one instance.  The builder runs on the host before
    ``jax.jit`` is entered, so the cumsum executes once per forward pass and
    the resulting ``cu_seqlens`` array is placed on the mesh as replicated
    metadata.
    """

    def __init__(self, mesh=None):
        self.mesh = mesh

    def get_forward_metadata(self, batch) -> GDNAttnMetadata:
        """Compute metadata for ``batch``.

        Accepts either a ``ModelWorkerBatch`` (host-side) or a ``ForwardBatch``
        — both expose ``forward_mode`` and ``extend_seq_lens``.  Mirrors
        ``LinearAttentionBackend.get_forward_metadata`` so the same call site
        in ``tp_worker`` works for both.
        """
        if batch.forward_mode == ForwardMode.DECODE:
            return GDNAttnMetadata()

        extend_seq_lens = np.asarray(batch.extend_seq_lens, dtype=np.int32)
        cu_seqlens = np.concatenate(
            [
                np.zeros(1, dtype=np.int32),
                np.cumsum(extend_seq_lens, dtype=np.int32),
            ]
        )

        # Single-host: place as replicated P() on the mesh. Multi-host: leave
        # sharding=None — shard_map's in_specs=P() will replicate.
        sharding = (
            NamedSharding(self.mesh, P())
            if self.mesh is not None and jax.process_count() == 1
            else None
        )
        (cu_seqlens_dev,) = device_array((cu_seqlens,), sharding=sharding)
        return GDNAttnMetadata(cu_seqlens=cu_seqlens_dev)
