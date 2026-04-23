"""Attention backend for the absorbed MLA path (MLA v2 Pallas kernel).

This backend wraps `mla_ragged_paged_attention` with a 4D paged latent KV cache
(see `MLATokenToKVPool`). Unlike the MHA backend it takes a 4-tuple payload
`(ql_nope, q_pe, new_kv_c, new_k_pe)` and returns a latent output `o_latent` of
shape `[T, n_h, kv_lora_rank]`; the caller is responsible for projecting through
`W_UV → W_O` (see `docs/design/MLA.md` §3.9).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass
class MLAAttentionMetadata:
    """Per-forward metadata for the MLA v2 kernel.

    Drops `custom_mask` / `swa_page_indices` / `attention_sink` (RFC §3.8). The
    `cu_kv_lens` field carries the page-aligned KV cumulative lengths used by
    the kernel to locate each sequence's pages in the ragged `page_indices`
    layout (matches RPA v3 addressing).
    """

    cu_q_lens: jax.Array = None
    cu_kv_lens: jax.Array = None
    page_indices: jax.Array = None
    seq_lens: jax.Array = None
    distribution: jax.Array = None

    def tree_flatten(self):
        children = (
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
        obj.cu_q_lens = children[0]
        obj.cu_kv_lens = children[1]
        obj.page_indices = children[2]
        obj.seq_lens = children[3]
        obj.distribution = children[4]
        return obj


@dataclass
class MLAAttentionBackend(AttentionBackend):
    """Absorbed-MLA attention backend backed by the v2 Pallas kernel."""

    def __init__(
        self,
        num_attn_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        page_size: int = 1,
        mesh: jax.sharding.Mesh = None,
        vmem_limit_bytes: int = 100 * (1 << 20),
        num_kv_pages_per_block: tuple[int, int, int] = (3, 1, 1),
        num_queries_per_block: tuple[int, int, int] = (1, 16, 16),
        decode_batch_size: int = 4,
    ):
        self.num_heads = num_attn_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.page_size = page_size
        self.mesh = mesh
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_kv_pages_per_block = num_kv_pages_per_block
        self.num_queries_per_block = num_queries_per_block
        self.decode_batch_size = decode_batch_size

        self.forward_metadata = nnx.data(MLAAttentionMetadata())

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Build per-batch metadata. Mirrors `FlashAttention.get_forward_metadata`
        without the SWA / custom_mask branches.
        """
        metadata = MLAAttentionMetadata()

        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        page_indices = (batch.cache_loc[indices] // self.page_size).astype(np.int32)

        if batch.forward_mode == ForwardMode.EXTEND:
            cu_q_lens = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(batch.extend_seq_lens, dtype=np.int32),
                ]
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            cu_q_lens = np.arange(len(batch.seq_lens) + 1, dtype=np.int32)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        seq_lens = batch.seq_lens
        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size
        cu_kv_lens = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(aligned_seq_lens, dtype=np.int32),
            ]
        )

        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            distribution = np.array([num_seqs, num_seqs, num_seqs], dtype=np.int32)
        elif batch.forward_mode == ForwardMode.EXTEND:
            distribution = np.array([0, num_seqs, num_seqs], dtype=np.int32)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        (
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
        )
        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_attn_heads": self.num_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "page_size": self.page_size,
            "vmem_limit_bytes": self.vmem_limit_bytes,
            "num_kv_pages_per_block": self.num_kv_pages_per_block,
            "num_queries_per_block": self.num_queries_per_block,
            "decode_batch_size": self.decode_batch_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            num_attn_heads=aux_data["num_attn_heads"],
            kv_lora_rank=aux_data["kv_lora_rank"],
            qk_nope_head_dim=aux_data["qk_nope_head_dim"],
            qk_rope_head_dim=aux_data["qk_rope_head_dim"],
            v_head_dim=aux_data["v_head_dim"],
            page_size=aux_data["page_size"],
            mesh=None,
            vmem_limit_bytes=aux_data["vmem_limit_bytes"],
            num_kv_pages_per_block=aux_data["num_kv_pages_per_block"],
            num_queries_per_block=aux_data["num_queries_per_block"],
            decode_batch_size=aux_data["decode_batch_size"],
        )
        obj.forward_metadata = children[0]
        return obj

    @named_scope
    def __call__(
        self,
        payload: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **kwargs,
    ):
        """Absorbed-MLA forward.

        Args:
            payload: `(ql_nope, q_pe, new_kv_c, new_k_pe)`
                - ql_nope:  [T, n_h, kv_lora_rank]
                - q_pe:     [T, n_h, qk_rope_head_dim]
                - new_kv_c: [T, kv_lora_rank]
                - new_k_pe: [T, qk_rope_head_dim]
            layer: RadixAttention metadata holder.
            forward_batch / token_to_kv_pool: cache + scheduling info.

        Returns:
            (o_latent, updated_cache) where
              - o_latent: [T, n_h, kv_lora_rank] (caller projects via W_UV→W_O)
              - updated_cache: 4D paged buffer to feed back into the pool.

        The kernel handles `qk_rope_head_dim=64 → 128` and other
        `align_to(*, 128)` padding internally, so callers should pass tensors at
        the unpadded logical dim.
        """
        ql_nope, q_pe, new_kv_c, new_k_pe = payload

        cache = token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)
        sm_scale = (
            (1.0 / jnp.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )

        in_specs = (
            P(None, None, None),  # ql_nope    [T, n_h, lkv]
            P(None, None, None),  # q_pe       [T, n_h, r]
            P(None, None),  # new_kv_c   [T, lkv]
            P(None, None),  # new_k_pe   [T, r]
            P(None, None, None, None),  # cache (replicated, RFC §3.1)
            P(),  # seq_lens
            P(),  # page_indices
            P(),  # cu_q_lens
            P(),  # cu_kv_lens
            P(),  # distribution
        )
        out_specs = (
            P(None, None, None),  # o_latent       [T, n_h, lkv]
            P(None, None, None, None),  # updated cache  4D
        )

        def _run(
            ql_nope_,
            q_pe_,
            new_kv_c_,
            new_k_pe_,
            cache_,
            seq_lens_,
            page_indices_,
            cu_q_lens_,
            cu_kv_lens_,
            distribution_,
        ):
            return mla_ragged_paged_attention(
                ql_nope_,
                q_pe_,
                new_kv_c_,
                new_k_pe_,
                cache_,
                seq_lens_,
                page_indices_,
                cu_q_lens_,
                cu_kv_lens_,
                distribution_,
                sm_scale=sm_scale,
                num_kv_pages_per_block=self.num_kv_pages_per_block,
                num_queries_per_block=self.num_queries_per_block,
                decode_batch_size=self.decode_batch_size,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

        o_latent, updated_cache = jax.shard_map(
            _run,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache,
            self.forward_metadata.seq_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
        )

        return o_latent, updated_cache
