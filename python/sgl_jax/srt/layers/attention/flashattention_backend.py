from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention_v3 import (
    ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.flash_attn_kernel.util import get_dtype_packing
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
        batch_size = len(batch.seq_lens)

        if batch.forward_mode == ForwardMode.EXTEND:
            cu_q_lens = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(batch.extend_seq_lens),
                ]
            )
            distribution = jnp.array([0, batch_size, batch_size], jnp.int32)
        elif batch.forward_mode == ForwardMode.DECODE:
            cu_q_lens = jnp.concatenate(
                [
                    np.array([0], dtype=jnp.int32),
                    np.cumsum(jnp.ones(len(batch.seq_lens), dtype=np.int32)),
                ]
            )
            distribution = jnp.array([batch_size, batch_size, batch_size], jnp.int32)
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

        metadata.distribution = device_array(mesh, distribution)
        metadata.num_seqs = device_array(mesh, num_seqs)
        metadata.cu_q_lens = device_array(mesh, cu_q_lens)
        metadata.cu_kv_lens = device_array(mesh, cu_kv_lens)
        metadata.page_indices = device_array(mesh, page_indices)
        metadata.seq_lens = device_array(mesh, seq_lens)
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
        if layer.scaling is None:
            scale = 1.0 / jnp.sqrt(layer.head_dim)
        else:
            scale = layer.scaling

        in_specs = (
            P(
                None, self.kv_partition_axis
            ),  # q shape: [batched_tokens, head_num, head_dim]
            P(None, self.kv_partition_axis),  # k
            P(None, self.kv_partition_axis),  # v
            P(None, None, self.kv_partition_axis, None),  # k_buffer
            P(None, None, self.kv_partition_axis, None),  # v_buffer
            P(),  # seq_lens
            P(),  # page_indices
            P(),  # cu_q_lens
            P(),  # distribution
        )
        out_specs = (
            P(None, self.kv_partition_axis),  # attn_weight
            P(None, None, self.kv_partition_axis, None),  # key cache
            P(None, None, self.kv_partition_axis, None),  # v cache
        )

        def _ragged_paged_attention(*args):
            q, k, v, k_cache, v_cache = args[:5]
            other_args = args[5:]

            kv_packing = get_dtype_packing(k_cache.dtype)
            k_cache = k_cache.reshape(
                *k_cache.shape[:2],
                k_cache.shape[2] // kv_packing,
                kv_packing,
                self.head_dim,
            )
            v_cache = v_cache.reshape(
                *v_cache.shape[:2],
                v_cache.shape[2] // kv_packing,
                kv_packing,
                self.head_dim,
            )

            # Since we now use pre-padded kv heads, ensure they are always even
            assert k_cache.shape[-2] % 2 == 0, (
                f"k_buffer kv_heads={k_cache.shape[-2]} should be even after pre-padding. "
                "This indicates a configuration issue with kv heads padding."
            )
            assert v_cache.shape[-2] % 2 == 0, (
                f"v_buffer kv_heads={v_cache.shape[-2]} should be even after pre-padding. "
                "This indicates a configuration issue with kv heads padding."
            )

            attn, k_cache, v_cache = ragged_paged_attention(
                q,
                k,
                v,
                k_cache,
                v_cache,
                *other_args,
                sm_scale=scale,
                sliding_window=None,
                soft_cap=None,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )
            return (
                attn,
                k_cache.reshape(*k_cache.shape[:2], -1, k_cache.shape[-1]),
                v_cache.reshape(*v_cache.shape[:2], -1, v_cache.shape[-1]),
            )

        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        attn_output, k_cache, v_cache = jax.shard_map(
            _ragged_paged_attention,
            mesh=jax.sharding.get_abstract_mesh(),
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            q.reshape(q.shape[0], -1, self.head_dim),
            k.reshape(q.shape[0], -1, self.head_dim),
            v.reshape(q.shape[0], -1, self.head_dim),
            k_cache.reshape(
                k_cache.shape[0] // self.page_size, self.page_size, -1, self.head_dim
            ),
            v_cache.reshape(
                v_cache.shape[0] // self.page_size, self.page_size, -1, self.head_dim
            ),
            self.forward_metadata.seq_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.distribution,
        )

        return (
            attn_output.reshape(q.shape[0], -1),
            k_cache,
            v_cache,
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
                layer_id, forward_batch.out_cache_loc, k, v, is_decode=False
            )
        else:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer_id, forward_batch.out_cache_loc, k, v, is_decode=True
            )

        return forward_batch.token_to_kv_pool.get_kv_buffer(layer_id)

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        assert (
            res > 0
        ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
        return res
