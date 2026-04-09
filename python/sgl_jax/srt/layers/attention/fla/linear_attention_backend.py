"""LinearAttentionBackend: pre-computes scatter/gather metadata for linear attention prefill.

This module sits parallel to FlashAttentionBackend. All LinearAttention layers in the
model share one LinearAttentionBackend instance. It pre-computes metadata outside JIT
for prefill; decode is a no-op.
"""

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

_CHUNK_SIZE = 64


@register_pytree_node_class
@dataclass
class LinearAttentionMetadata:
    """Prefill metadata passed through ForwardBatch as a JIT-traced pytree.

    For DECODE batches, both fields are None (decode needs no scatter/gather).
    """

    cu_seqlens_dev: jax.Array = None  # [N_padded+1], chunk-aligned boundaries
    scatter_idx: jax.Array = None  # [T], tight-packed -> chunk-aligned mapping

    def tree_flatten(self):
        return (self.cu_seqlens_dev, self.scatter_idx), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(cu_seqlens_dev=children[0], scatter_idx=children[1])


class LinearAttentionBackend(nnx.Module):
    """Pre-computes scatter/gather index metadata for linear attention prefill.

    Design note: this backend is intentionally kept separate from FlashAttentionBackend.
    Ling-2.5 uses a hybrid architecture (MLA + Linear Attention layers). When the full
    model is integrated, these two backends may be combined into a single
    HybridAttnBackend that dispatches by layer_id (similar to sglang-python's
    HybridLinearAttnBackend). For now, keeping them independent simplifies development
    and unit testing of each attention type.

    Attributes:
        mesh: Optional JAX mesh for sharding.
        T_packed_bucket: Total chunk-aligned buffer length (static; changes trigger
            recompilation when used inside jit).
    """

    def __init__(self, mesh=None):
        self.mesh = mesh
        # Static attribute — enters NNX graphdef; changing it triggers recompilation.
        self.T_packed_bucket: int = 0

    def get_forward_metadata(self, batch) -> "LinearAttentionMetadata":
        """Pre-compute scatter/gather metadata for the current batch.

        For DECODE batches returns an empty LinearAttentionMetadata. For EXTEND
        batches it computes chunk-aligned cumulative sequence lengths and a scatter
        index array that maps outer (possibly padded) token positions into the
        packed buffer.
        """
        if batch.forward_mode == ForwardMode.DECODE:
            return LinearAttentionMetadata()

        extend_seq_lens = np.asarray(batch.extend_seq_lens, dtype=np.int32)
        N = len(extend_seq_lens)

        # Chunk-align each sequence length; zero-length sequences stay zero.
        aligned_lens = np.where(
            extend_seq_lens == 0,
            0,
            ((extend_seq_lens + _CHUNK_SIZE - 1) // _CHUNK_SIZE) * _CHUNK_SIZE,
        ).astype(np.int32)

        # Cumulative sum: shape [N+1], cu_seqlens[i] = start of request i in packed buf.
        cu_seqlens = np.concatenate(
            [np.array([0], dtype=np.int32), np.cumsum(aligned_lens, dtype=np.int32)]
        )

        T_pb = int(cu_seqlens[-1])  # total packed buffer size

        # T_outer comes from the outer padding bucket set by the scheduler.
        T_outer = len(batch.input_ids)

        # Initialise every outer position to the dummy slot (position T_pb in the
        # +1-padded buffer).  Real token positions are filled in below.
        scatter_idx = np.full(T_outer, T_pb, dtype=np.int32)

        offset_tight = 0
        for i in range(N):
            seq_len = int(extend_seq_lens[i])
            if seq_len == 0:
                continue
            # Real tokens for request i map to consecutive positions starting at
            # cu_seqlens[i] in the packed buffer.
            scatter_idx[offset_tight : offset_tight + seq_len] = np.arange(
                cu_seqlens[i], cu_seqlens[i] + seq_len, dtype=np.int32
            )
            offset_tight += seq_len

        self.T_packed_bucket = T_pb

        # Single-host: explicitly place as replicated P() on the mesh.
        # Multi-host (process_count > 1): leave sharding=None; the arrays are
        # small metadata (cu_seqlens, scatter_idx) and will be replicated by
        # shard_map's in_specs=P() when consumed by the model.
        sharding = (
            NamedSharding(self.mesh, P())
            if self.mesh is not None and jax.process_count() == 1
            else None
        )
        cu_seqlens_dev, scatter_idx_dev = device_array((cu_seqlens, scatter_idx), sharding=sharding)
        return LinearAttentionMetadata(cu_seqlens_dev=cu_seqlens_dev, scatter_idx=scatter_idx_dev)


def scatter_to_packed(x: jax.Array, scatter_idx: jax.Array, T_packed_bucket: int) -> jax.Array:
    """Scatter outer (padded) tokens into a chunk-aligned packed buffer.

    Args:
        x: Input activations of shape [T_outer, H, K].
        scatter_idx: Integer index array of shape [T_outer]. Each entry is either
            a position in the packed buffer or T_packed_bucket (dummy slot).
        T_packed_bucket: Total packed buffer length (not including the dummy slot).

    Returns:
        Packed buffer of shape [1, T_packed_bucket, H, K].
    """
    H, K = x.shape[1], x.shape[2]
    # Allocate buffer with one extra dummy slot at index T_packed_bucket.
    buf = jnp.zeros((1, T_packed_bucket + 1, H, K), dtype=x.dtype)
    buf = buf.at[0, scatter_idx].set(x)
    # Drop the dummy slot to get the final packed buffer.
    return buf[:, :T_packed_bucket]


def gather_from_packed(output_packed: jax.Array, scatter_idx: jax.Array) -> jax.Array:
    """Gather tokens from a packed buffer back to outer (padded) positions.

    Args:
        output_packed: Packed output of shape [1, T_packed_bucket, H, V].
        scatter_idx: Integer index array of shape [T_outer]. Positions equal to
            T_packed_bucket read from the zero-padded dummy column.

    Returns:
        Output of shape [T_outer, H, V].
    """
    # Pad one dummy column of zeros so that scatter_idx values == T_packed_bucket
    # safely read zero rather than out-of-bounds.
    padded = jnp.pad(output_packed, ((0, 0), (0, 1), (0, 0), (0, 0)))
    return padded[0, scatter_idx]
