from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ragged_paged_attention,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

CACHE_LOC_PADDING_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


@register_pytree_node_class
@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    num_seqs: jax.Array = None
    cu_q_lens: jax.Array = None
    cu_kv_lens: jax.Array = None
    page_indices: jax.Array = None
    num_kv_pages_per_block: int = None
    num_queries_per_block: int = None

    def tree_flatten(self):
        children = (
            self.num_seqs,
            self.cu_q_lens,
            self.cu_kv_lens,
            self.page_indices,
        )

        aux_data = {
            "num_kv_pages_per_block": self.num_kv_pages_per_block,
            "num_queries_per_block": self.num_queries_per_block,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.num_kv_pages_per_block = aux_data["num_kv_pages_per_block"]
        obj.num_queries_per_block = aux_data["num_queries_per_block"]

        obj.num_seqs = children[0]
        obj.cu_q_lens = children[1]
        obj.cu_kv_lens = children[2]
        obj.page_indices = children[3]

        return obj


@register_pytree_node_class
class FlashAttention(AttentionBackend):
    """Native Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        head_dim,
        vmem_limit_bytes: int = 32 * (1 << 20),  # 32MB
        max_num_kv_pages_per_block: int = 64,
        max_num_queries_per_block: int = 256,
        page_size: int = 1,
    ):
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        self.head_dim = head_dim
        self.max_num_kv_pages_per_block = max_num_kv_pages_per_block
        self.max_num_queries_per_block = max_num_queries_per_block
        self.page_size = page_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        metadata = FlashAttentionMetadata()

        indices = jnp.arange(0, len(forward_batch.cache_loc), self.page_size)
        selected_cache_locs = forward_batch.cache_loc[indices]
        metadata.page_indices = (selected_cache_locs // self.page_size).astype(
            jnp.int32
        )

        metadata.num_kv_pages_per_block = self.max_num_kv_pages_per_block
        metadata.num_queries_per_block = self.max_num_queries_per_block

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            metadata.cu_q_lens = jnp.concatenate(
                [
                    jnp.array([0], dtype=jnp.int32),
                    jnp.cumsum(forward_batch.extend_seq_lens),
                ]
            )
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            metadata.cu_q_lens = jnp.concatenate(
                [
                    jnp.array([0], dtype=jnp.int32),
                    jnp.cumsum(jnp.ones(forward_batch.batch_size, dtype=jnp.int32)),
                ]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

        metadata.cu_kv_lens = jnp.concatenate(
            [
                jnp.array([0], dtype=jnp.int32),
                jnp.cumsum(forward_batch.seq_lens),
            ]
        )
        metadata.num_seqs = jnp.sum(
            forward_batch.seq_lens > 0, dtype=jnp.int32
        ).reshape(
            1,
        )

        self.forward_metadata = metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "vmem_limit_bytes": self.vmem_limit_bytes,
            "head_dim": self.head_dim,
            "max_num_kv_pages_per_block": self.max_num_kv_pages_per_block,
            "max_num_queries_per_block": self.max_num_queries_per_block,
            "page_size": self.page_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            aux_data["num_heads"],
            aux_data["num_kv_heads"],
            aux_data["head_dim"],
            aux_data["vmem_limit_bytes"],
            aux_data["max_num_kv_pages_per_block"],
            aux_data["max_num_queries_per_block"],
            aux_data["page_size"],
        )

        obj.forward_metadata = children[0]

        return obj

    def __call__(
        self,
        q: jax.Array,  # [total_tokens, hidden_size]
        k: jax.Array,  # [total_tokens, hidden_size]
        v: jax.Array,  # [total_tokens, hidden_size]
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        attention_mask: jax.Array = None,
        kv_partition_axis: str = "tensor",
    ):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking
        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """
        if forward_batch.forward_mode == ForwardMode.IDLE:
            return (
                jnp.zeros_like(q),
                jnp.zeros_like(k),
                jnp.zeros_like(v),
            )

        k_buffer, v_buffer = self._get_and_set_kv_cache(
            k, v, forward_batch, layer.layer_id
        )

        if layer.scaling is None:
            scale = 1.0 / jnp.sqrt(layer.head_dim)
        else:
            scale = layer.scaling

        in_specs = (
            P(None, kv_partition_axis),  # q shape: [batched_tokens, head_num, head_dim]
            P(None, None, kv_partition_axis, None),  # k_buffer sha
            P(None, None, kv_partition_axis, None),  # v_buffer
            P(),  # page_indices
            P(),  # cu_q_lens
            P(),  # cu_kv_lens
            P(),  # num_seqs
        )
        out_specs = P(None, kv_partition_axis)

        def _ragged_paged_attention(*args):
            return ragged_paged_attention(
                *args,
                sm_scale=scale,
                sliding_window=None,
                soft_cap=None,
                mask_value=None,
                num_kv_pages_per_block=self.forward_metadata.num_kv_pages_per_block,
                num_queries_per_block=self.forward_metadata.num_queries_per_block,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

        attn_output = jax.shard_map(
            _ragged_paged_attention,
            mesh=jax.sharding.get_abstract_mesh(),
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            q.reshape(q.shape[0], -1, self.head_dim),
            k_buffer[:, None, :, :],
            v_buffer[:, None, :, :],
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.num_seqs,
        )

        return (
            attn_output.reshape(q.shape[0], -1),
            k_buffer,
            v_buffer,
        )

    def _get_and_set_kv_cache(
        self,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        layer_id: int,
    ) -> Tuple[jax.Array, jax.Array]:
        """
        Get the kv cache from the forward batch.
        """
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer_id, forward_batch.out_cache_loc, k, v
            )
        else:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer_id, forward_batch.out_cache_loc, k, v
            )

        return forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)


def get_cache_loc_padding_size_per_seq(max_seq_len_in_batch: int) -> int:
    """
    Get the padding size for each sequence in the bucket.
    """
    for bucket in CACHE_LOC_PADDING_BUCKETS:
        if bucket >= max_seq_len_in_batch:
            return bucket
    assert False, "No bucket is greater than max_seq_len_in_batch"
