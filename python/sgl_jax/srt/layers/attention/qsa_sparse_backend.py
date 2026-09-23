"""QSA sparse attention backend (Qwen3.8-Flash-Next).

GQA plus an indexer, the way ``DSASparseAttentionBackend`` is MLA plus an
indexer -- so this subclasses ``FlashAttention`` and inherits its metadata, its
page-table construction and its dense path unchanged. What it adds is the
sparse route: scatter the step's compressed indexer keys, pick the blocks, and
attend over only those.

The indexer's weights stay in the model, as they do upstream and in DSA: the
attention module calls ``QSAIndexer.project`` and ``QSAIndexer.compress_batch``
and hands the results down as kwargs. This backend owns the paging and the
kernels, not the parameters.

Every piece of arithmetic here is covered by a test somewhere else --
``test_qsa_pool`` for the page geometry, ``test_paging`` for the scatter,
``test_qsa_indexer`` for the compression, ``test_qsa_pipeline`` for the whole
chain, ``test_sparse_gqa_parity`` for the kernel. What is **not** covered is
this file's own assembly: no test in the repo constructs an attention backend,
DSA's included. It is first exercised end to end when a model wires it up.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.qsa.paging import as_3d, as_4d, scatter_compressed
from sgl_jax.srt.kernels.qsa.sparse_gqa_attention import sparse_gqa_attention
from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing
from sgl_jax.srt.layers.attention.dsa_sparse_backend import _fixed_stride_pages
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.qsa_indexer import select_blocks

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

# Sparse prefill is opt in, matching DSA_PREFILL_SPARSE: per-query selection
# pays off in decode, where each query comes from a different sequence and
# there is nothing to amortise, and loses in prefill, where neighbouring
# queries select almost the same blocks.
QSA_PREFILL_SPARSE = os.environ.get("QSA_PREFILL_SPARSE", "0") == "1"


@register_pytree_node_class
@dataclass
class QSAFusedCache:
    """What a QSA layer hands back for the pool to absorb."""

    kv: jax.Array
    compressed: jax.Array | None = None
    ring: jax.Array | None = None

    def tree_flatten(self):
        return ((self.kv, self.compressed, self.ring), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class QSASparseAttentionBackend(FlashAttention):
    """Sparse GQA over the blocks the QSA indexer selects."""

    token_to_kv_pool_class = None  # set below, after the pool import resolves

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        head_dim,
        page_size: int = 1,
        kv_partition_axis: str = "tensor",
        attention_data_partition_axis: str = "data",
        mesh: jax.sharding.Mesh = None,
        *,
        compress_ratio: int = 4,
        block_topk: int = 512,
        full_slot: dict[int, int] | None = None,
    ):
        super().__init__(
            num_attn_heads,
            num_kv_heads,
            head_dim,
            page_size,
            kv_partition_axis=kv_partition_axis,
            attention_data_partition_axis=attention_data_partition_axis,
            mesh=mesh,
        )
        self.compress_ratio = compress_ratio
        self.block_topk = block_topk
        # layer id -> indexer slot; only full-attention layers have one.
        self.full_slot = full_slot or {}

    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        aux_data = {
            **aux_data,
            "compress_ratio": self.compress_ratio,
            "block_topk": self.block_topk,
            "full_slot": self.full_slot,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            aux_data["num_heads"],
            aux_data["num_kv_heads"],
            aux_data["head_dim"],
            aux_data["page_size"],
            kv_partition_axis=aux_data.get("kv_partition_axis", "tensor"),
            attention_data_partition_axis=aux_data.get("attention_data_partition_axis", "data"),
            mesh=aux_data.get("mesh"),
            compress_ratio=aux_data["compress_ratio"],
            block_topk=aux_data["block_topk"],
            full_slot=aux_data["full_slot"],
        )
        obj.forward_metadata = children[0]
        return obj

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        causal: int = 1,
        attention_sink: jax.Array = None,
        **qsa_kwargs,
    ):
        """Dense unless this layer has an indexer and the mode wants sparse.

        ``qsa_kwargs`` carries what the model's indexer produced this step:
        ``indexer_q`` (scoring queries), and ``compressed`` / ``groups`` /
        ``seq_ids`` / ``ring`` from ``QSAIndexer.compress_batch``.
        """
        slot = self.full_slot.get(getattr(layer, "layer_id", -1))
        indexer_q = qsa_kwargs.get("indexer_q")
        if slot is None or indexer_q is None:
            return super().__call__(
                q, k, v, layer, forward_batch, token_to_kv_pool, causal, attention_sink
            )

        compressed_cache, ring = self._absorb_indexer_step(token_to_kv_pool, slot, qsa_kwargs)

        is_decode = forward_batch.forward_mode.is_decode()
        if not is_decode and not QSA_PREFILL_SPARSE:
            # Dense still has to happen, and it writes the KV cache on the way.
            out, kv_fused = super().__call__(
                q, k, v, layer, forward_batch, token_to_kv_pool, causal, attention_sink
            )
            return out, QSAFusedCache(kv_fused, compressed_cache, ring)

        # Sparse: the step's own keys must be in the cache before the gather,
        # because a query attends to its own position and to its open group.
        token_to_kv_pool.set_kv_buffer(
            layer.layer_id, forward_batch.out_cache_loc, k, v, is_decode=is_decode
        )
        kv_fused = token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)

        out = self._run_sparse(q, indexer_q, compressed_cache, kv_fused, layer, forward_batch)
        return out.reshape(q.shape[0], -1), QSAFusedCache(kv_fused, compressed_cache, ring)

    def _absorb_indexer_step(self, token_to_kv_pool, slot, qsa_kwargs):
        """Write this step's completed groups into the compressed cache.

        Page arithmetic only, so it runs in the global scope rather than inside
        a ``shard_map``: the ring is indexed by request slot, and a request's
        slot says nothing about which shard holds its pages, so the ring is
        replicated and updated here.
        """
        md = self.forward_metadata
        cache4d = token_to_kv_pool.get_compressed_key_buffer(slot)
        packing = get_dtype_packing(cache4d.dtype)
        updated = scatter_compressed(
            as_3d(cache4d),
            qsa_kwargs["compressed"],
            qsa_kwargs["groups"],
            qsa_kwargs["seq_ids"],
            md.page_indices,
            md.cu_kv_lens,
            compress_ratio=self.compress_ratio,
        )
        return as_4d(updated, packing), qsa_kwargs.get("ring")

    def _run_sparse(self, q, indexer_q, compressed_cache, kv_fused, layer, forward_batch):
        """Select blocks, then attend over them, inside one shard_map."""
        md = self.forward_metadata
        n_seqs = md.seq_lens.shape[0]
        pages_per_seq = md.page_indices.shape[0] // n_seqs
        page_table = _fixed_stride_pages(
            md.page_indices, md.cu_kv_lens, self.page_size, pages_per_seq
        ).reshape(n_seqs, pages_per_seq)
        scale = (
            1.0 / jnp.sqrt(layer.head_dim)
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )
        token_to_req = jnp.clip(
            jnp.searchsorted(md.cu_q_lens[1:], jnp.arange(q.shape[0]), side="right"),
            0,
            n_seqs - 1,
        )
        dpa = self.attention_data_partition_axis

        def _select_and_attend(
            q_,
            indexer_q_,
            compressed_,
            kv_,
            positions_,
            token_to_req_,
            page_table_,
            seq_lens_,
            page_indices_,
            cu_q_lens_,
            distribution_,
        ):
            block_ids = select_blocks(
                indexer_q_,
                as_3d(compressed_),
                seq_lens_,
                page_indices_,
                cu_q_lens_,
                cu_q_lens_,  # unused on the kernel path, which walks the fixed-stride table
                distribution_,
                block_topk=self.block_topk,
                compress_ratio=self.compress_ratio,
                pages_per_seq=pages_per_seq,
                use_kernel=True,
            )
            return sparse_gqa_attention(
                q_,
                block_ids,
                positions_,
                token_to_req_,
                page_table_,
                kv_,
                sm_scale=scale,
                ratio=self.compress_ratio,
            )

        return jax.shard_map(
            _select_and_attend,
            in_specs=(
                P(dpa, "tensor", None),  # q
                P(dpa, None, None),  # indexer queries
                P(dpa, None, None, None),  # compressed cache
                P(dpa, None, "tensor", None, None),  # fused KV cache
                P(dpa),  # positions
                P(dpa),  # token -> request
                P(dpa, None),  # page table
                P(dpa),  # seq_lens
                P(dpa),  # page_indices
                P(dpa),  # cu_q_lens
                P(dpa),  # distribution
            ),
            out_specs=P(dpa, "tensor", None),
            check_vma=False,
        )(
            q,
            indexer_q,
            compressed_cache,
            kv_fused,
            forward_batch.positions,
            token_to_req,
            page_table,
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.distribution,
        )


def _resolve_pool_class():
    from sgl_jax.srt.mem_cache.memory_pool import QSATokenToKVPool

    return QSATokenToKVPool


QSASparseAttentionBackend.token_to_kv_pool_class = _resolve_pool_class()
