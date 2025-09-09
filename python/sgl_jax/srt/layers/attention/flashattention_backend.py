from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.flash_attn_kernel.v2.kernel import (
    ragged_paged_attention,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array


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
    seq_lens: jax.Array = None
    distribution: jax.Array = None

    def tree_flatten(self):
        children = (
            self.num_seqs,
            self.cu_q_lens,
            self.cu_kv_lens,
            self.page_indices,
            self.seq_lens,
            self.distribution,
        )

        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.num_seqs = children[0]
        obj.cu_q_lens = children[1]
        obj.cu_kv_lens = children[2]
        obj.page_indices = children[3]
        obj.seq_lens = children[4]
        obj.distribution = children[5]

        return obj


@register_pytree_node_class
@dataclass
class FlashAttention(AttentionBackend):
    """Native Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        head_dim,
        vmem_limit_bytes: int = 32 * (1 << 20),  # 32MB
        page_size: int = 1,
        kv_partition_axis: str = "tensor",
    ):
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.kv_partition_axis = kv_partition_axis
        self.forward_metadata = FlashAttentionMetadata()

    def get_forward_metadata(self, batch: ModelWorkerBatch, mesh: Mesh):
        """Return the metadata for a forward pass."""
        metadata = FlashAttentionMetadata()

        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        selected_cache_locs = batch.cache_loc[indices]
        page_indices = (selected_cache_locs // self.page_size).astype(np.int32)

        if batch.forward_mode == ForwardMode.EXTEND:
            cu_q_lens = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(batch.extend_seq_lens),
                ]
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            cu_q_lens = jnp.concatenate(
                [
                    np.array([0], dtype=jnp.int32),
                    np.cumsum(jnp.ones(len(batch.seq_lens), dtype=np.int32)),
                ]
            )
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        seq_lens = np.copy(batch.seq_lens)

        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size
        cu_kv_lens = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(aligned_seq_lens),
            ]
        )

        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )

        # Construct distribution for V2 kernel: [decode_end, prefill_end, mixed_end]
        if batch.forward_mode == ForwardMode.DECODE:
            # All sequences are decode/mixed mode
            distribution = np.array([0, 0, num_seqs.item()], dtype=np.int32)
        elif batch.forward_mode == ForwardMode.EXTEND:
            # All sequences are prefill mode
            distribution = np.array(
                [0, num_seqs.item(), num_seqs.item()], dtype=np.int32
            )
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        (
            metadata.num_seqs,
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            mesh,
            (num_seqs, cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
        )
        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "vmem_limit_bytes": self.vmem_limit_bytes,
            "head_dim": self.head_dim,
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
        k_buffer, v_buffer = self._get_and_set_kv_cache(
            k, v, forward_batch, layer.layer_id
        )

        if layer.scaling is None:
            scale = 1.0 / jnp.sqrt(layer.head_dim)
        else:
            scale = layer.scaling

        # Prepare separate K/V caches for V2 kernel
        # k_buffer and v_buffer are already 3D: [total_tokens, num_kv_heads, head_dim]
        # Reshape to paged format: [num_pages, page_size, num_kv_heads, head_dim]
        total_tokens = k_buffer.shape[0]
        num_pages = total_tokens // self.page_size

        k_cache = k_buffer.reshape(
            num_pages, self.page_size, self.num_kv_heads, self.head_dim
        )
        v_cache = v_buffer.reshape(
            num_pages, self.page_size, self.num_kv_heads, self.head_dim
        )

        in_specs = (
            P(None, self.kv_partition_axis),  # queries
            P(None, self.kv_partition_axis),  # keys (new tokens)
            P(None, self.kv_partition_axis),  # values (new tokens)
            P(None, None, self.kv_partition_axis, None),  # k_cache
            P(None, None, self.kv_partition_axis, None),  # v_cache
            P(),  # kv_lens
            P(),  # page_indices
            P(),  # cu_q_lens
            P(),  # cu_kv_lens
            P(),  # distribution
        )
        out_specs = (
            P(None, self.kv_partition_axis),  # attention output
            P(None, None, self.kv_partition_axis, None),  # updated k_cache
            P(None, None, self.kv_partition_axis, None),  # updated v_cache
        )

        def _ragged_paged_attention_with_separate_kv(*args):
            queries, keys, values, k_cache, v_cache = args[:5]
            other_args = args[5:]

            # Directly call V2 kernel with separate k_cache and v_cache
            result, updated_k_cache, updated_v_cache = ragged_paged_attention(
                queries,
                keys,
                values,
                k_cache,
                v_cache,
                *other_args,
                sm_scale=scale,
                sliding_window=None,
                soft_cap=None,
                mask_value=None,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

            return result, updated_k_cache, updated_v_cache

        (
            attn_output,
            k_buffer,
            v_buffer,
        ) = jax.shard_map(  # V2 handles KV cache updates internally
            _ragged_paged_attention_with_separate_kv,
            mesh=jax.sharding.get_abstract_mesh(),
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            q.reshape(q.shape[0], -1, self.head_dim),
            k.reshape(k.shape[0], -1, self.head_dim),
            v.reshape(v.shape[0], -1, self.head_dim),
            k_cache,
            v_cache,
            self.forward_metadata.seq_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
        )

        # V2 kernel internally updates the KV cache, so we don't need to handle updated_kv_cache

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
        # if forward_batch.forward_mode == ForwardMode.EXTEND:
        #     forward_batch.token_to_kv_pool.set_kv_buffer(
        #         layer_id, forward_batch.out_cache_loc, k, v, is_decode=False
        #     )
        # else:
        #     forward_batch.token_to_kv_pool.set_kv_buffer(
        #         layer_id, forward_batch.out_cache_loc, k, v, is_decode=True
        #     )

        return forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        assert (
            res > 0
        ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
        return res
