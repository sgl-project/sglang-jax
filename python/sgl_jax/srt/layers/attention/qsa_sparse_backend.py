"""QSA sparse attention backend (Qwen3.8-Flash-Next).

GQA plus an indexer, the way ``DSASparseAttentionBackend`` is MLA plus an
indexer -- so this subclasses ``FlashAttention`` and inherits its metadata, its
page-table construction and its dense path unchanged. What it adds is the
sparse route: compress the step's indexer keys and scatter them, pick the
blocks, and attend over only those.

The indexer's weights stay in the model, as they do upstream and in DSA: the
attention module calls ``QSAIndexer.project`` and hands down the projections
together with the indexer, whose ``compress_batch`` this backend runs. The
backend owns the paging, the kernels and where each of them runs, not the
parameters.

Every piece of arithmetic here is covered by a test somewhere else --
``test_qsa_pool`` for the page geometry, ``test_paging`` for the scatter,
``test_qsa_indexer`` for the compression, ``test_qsa_pipeline`` for the whole
chain, ``test_sparse_gqa_parity`` for the kernel. ``test_qsa_sparse_backend``
covers this file's own assembly: which inputs a QSA layer requires, what each
rank of a data-parallel batch reads, and the cache layout and page table each
kernel is handed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.qsa.paging import as_3d, as_4d, scatter_compressed
from sgl_jax.srt.kernels.qsa.sparse_gqa_attention import sparse_gqa_attention
from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing
from sgl_jax.srt.layers.attention.dsa_sparse_backend import _fixed_stride_pages
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.qsa_indexer import select_blocks, write_ring_rows

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

# Sparse prefill is opt in, matching DSA_PREFILL_SPARSE: per-query selection
# pays off in decode, where each query comes from a different sequence and
# there is nothing to amortise, and loses in prefill, where neighbouring
# queries select almost the same blocks.
QSA_PREFILL_SPARSE = os.environ.get("QSA_PREFILL_SPARSE", "0") == "1"

# What a QSA layer must be called with; see QSASparseAttentionBackend.__call__.
_INDEXER_INPUTS = ("indexer_q", "indexer_k", "indexer", "indexer_rotary_emb")


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

        ``qsa_kwargs`` carries the model's indexer outputs for this step,
        ``indexer_q`` (scoring queries) and ``indexer_k`` (raw keys, before
        compression), and the ``indexer`` with its ``indexer_rotary_emb``. The
        backend runs ``indexer.compress_batch`` itself, inside its
        ``shard_map``, where each rank sees its own requests.
        """
        slot = self.full_slot.get(getattr(layer, "layer_id", -1))
        if slot is None:
            return super().__call__(
                q, k, v, layer, forward_batch, token_to_kv_pool, causal, attention_sink
            )
        missing = [name for name in _INDEXER_INPUTS if qsa_kwargs.get(name) is None]
        if missing:
            raise ValueError(
                f"layer {layer.layer_id} has a QSA indexer but was called without "
                f"{', '.join(missing)}; the model must pass every indexer output"
            )

        compressed_cache, ring = self._absorb_indexer_step(
            token_to_kv_pool, slot, forward_batch, qsa_kwargs
        )

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

        out = self._run_sparse(
            q, qsa_kwargs["indexer_q"], compressed_cache, kv_fused, layer, forward_batch
        )
        return out.reshape(q.shape[0], -1), QSAFusedCache(kv_fused, compressed_cache, ring)

    def _absorb_indexer_step(self, token_to_kv_pool, slot, forward_batch, qsa_kwargs):
        """Compress this step's keys and write the completed groups.

        Runs per data-parallel rank. Each rank's ``cu_q_lens`` and
        ``cu_kv_lens`` start from zero and its page ids are local, so the batch
        is only addressable from inside the rank. The ring is the exception: it
        is indexed by ``ReqToTokenPool`` slot, which is unique across ranks, so
        it stays replicated and every rank writes every rank's updated rows.
        """
        md = self.forward_metadata
        dpa = self.attention_data_partition_axis
        # The indexer's parameters enter the shard_map as an argument: it cannot
        # close over arrays placed on explicit mesh axes.
        graphdef, params = nnx.split(qsa_kwargs["indexer"])
        rotary_emb = qsa_kwargs["indexer_rotary_emb"]
        cache = token_to_kv_pool.get_compressed_key_buffer(slot)
        packing = get_dtype_packing(cache.dtype)

        def _absorb(
            params,
            raw_keys,
            positions,
            req_slots,
            rings,
            cache_,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
        ):
            indexer = nnx.merge(graphdef, params)
            compressed, groups, seq_ids, local_rings = indexer.compress_batch(
                raw_keys, positions, cu_q_lens, req_slots, rings, rotary_emb
            )
            cache3d = scatter_compressed(
                as_3d(cache_),
                compressed,
                groups,
                seq_ids,
                page_indices,
                cu_kv_lens,
                compress_ratio=self.compress_ratio,
            )
            all_slots = jax.lax.all_gather(req_slots, dpa, tiled=True)
            all_rows = jax.lax.all_gather(local_rings[req_slots], dpa, tiled=True)
            return as_4d(cache3d, packing), write_ring_rows(rings, all_slots, all_rows)

        return jax.shard_map(
            _absorb,
            in_specs=(
                P(),  # indexer parameters, replicated
                P(dpa, None),  # raw indexer keys
                P(dpa),  # positions
                P(dpa),  # request slots
                P(None, None, None),  # ring
                P(dpa, None, None, None),  # compressed cache
                P(dpa),  # page_indices
                P(dpa),  # cu_q_lens
                P(dpa),  # cu_kv_lens
            ),
            out_specs=(P(dpa, None, None, None), P(None, None, None)),
            check_vma=False,
        )(
            params,
            qsa_kwargs["indexer_k"],
            forward_batch.positions,
            forward_batch.req_pool_indices,
            token_to_kv_pool.get_open_group_buffer(slot),
            cache,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
        )

    def _run_sparse(self, q, indexer_q, compressed_cache, kv_fused, layer, forward_batch):
        """Select blocks, then attend over them, inside one shard_map.

        The page table and each token's request come from the rank's own
        metadata, for the reason ``_absorb_indexer_step`` gives.
        """
        md = self.forward_metadata
        # A Python float: the attention kernel takes its scale as a static argument.
        scale = layer.head_dim**-0.5 if layer.scaling is None else layer.scaling
        dpa = self.attention_data_partition_axis

        def _select_and_attend(
            q_,
            indexer_q_,
            compressed_,
            kv_,
            positions_,
            seq_lens_,
            page_indices_,
            cu_q_lens_,
            cu_kv_lens_,
            distribution_,
        ):
            n_seqs = seq_lens_.shape[0]
            pages_per_seq = page_indices_.shape[0] // n_seqs
            page_table_ = _fixed_stride_pages(
                page_indices_, cu_kv_lens_, self.page_size, pages_per_seq
            ).reshape(n_seqs, pages_per_seq)
            token_to_req_ = jnp.clip(
                jnp.searchsorted(cu_q_lens_[1:], jnp.arange(q_.shape[0]), side="right"),
                0,
                n_seqs - 1,
            )
            block_ids = select_blocks(
                indexer_q_,
                compressed_,
                seq_lens_,
                page_table_,
                cu_q_lens_,
                distribution_,
                block_topk=self.block_topk,
                compress_ratio=self.compress_ratio,
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
                P(dpa),  # seq_lens
                P(dpa),  # page_indices
                P(dpa),  # cu_q_lens
                P(dpa),  # cu_kv_lens
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
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
        )


def _resolve_pool_class():
    from sgl_jax.srt.mem_cache.memory_pool import QSATokenToKVPool

    return QSATokenToKVPool


QSASparseAttentionBackend.token_to_kv_pool_class = _resolve_pool_class()
