"""DSA sparse attention backend (DeepSeek Sparse Attention + IndexShare).

Wraps the absorbed-MLA path with a lightning-indexer top-k selection so core
attention runs over at most ``index_topk`` KV positions per query. IndexShare
(GLM-5.2) is realised by threading the last full-layer's ``topk_indices``
through the model's per-layer loop and reusing it on ``shared`` layers.

The legacy path runs page-level sparse MLA for decode and dense MLA for
extend. The exact path uses a functional ``lax.top_k`` StreamIndex fallback,
maps sequence-local positions to physical cache slots, and runs the fused
SparseCore-gather + TensorCore-attention kernel for both extend and decode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.dsa.exact_attention import sparse_core_tensor_core_dsa
from sgl_jax.srt.kernels.dsa.ref import streamindex_page_topk_ref, streamindex_topk_ref
from sgl_jax.srt.kernels.dsa.sparse_mla import compute_topk_pages, sparse_mla_page_level
from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_SPARSE_PALLAS_MAX_T = 1
_DEBUG_N_HIT = False
# Page-budget for the page-scoring indexer path: the indexer max-pools token
# scores per page and selects this many pages directly (0 = off, use the
# token-topk + page-union path with k_pages_max=512). Bounds sparse-MLA cost
# to O(budget) flat vs the union path which saturates 512 pages at long ctx.
_PAGE_TOPK_BUDGET = int(os.environ.get("DSA_PAGE_TOPK", "0"))


@register_pytree_node_class
@dataclass
class DSAFusedCache:
    """Return payload from :class:`DSASparseAttentionBackend`.

    ``kv`` is the updated latent-KV page buffer (same as the plain MLA backend
    returns). ``idx`` is the updated indexer-key page buffer for full layers,
    ``None`` on shared layers. ``topk`` is the freshly computed top-k indices
    on full layers, ``None`` on shared layers — the model loop threads this
    into the next layer's ``dsa_topk_in`` to implement IndexShare.
    """

    kv: jax.Array
    idx: jax.Array | None
    topk: jax.Array | None
    topk_pages: jax.Array | None = None

    def tree_flatten(self):
        return ((self.kv, self.idx, self.topk, self.topk_pages), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@dataclass
class DSASparseAttentionBackend(MLAAttentionBackend):
    """Absorbed-MLA + DSA lightning-indexer top-k + IndexShare."""

    def __init__(
        self,
        *,
        index_topk: int,
        index_head_dim: int,
        index_n_heads: int,
        skip_offset: int,
        full_slot: dict[int, int],
        sparse_impl: str = "page",
        topk_impl: str = "approx",
        **mla_kwargs,
    ):
        super().__init__(**mla_kwargs)
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.skip_offset = skip_offset
        self.full_slot = full_slot
        if sparse_impl not in ("page", "exact"):
            raise ValueError(f"unknown DSA sparse implementation: {sparse_impl}")
        if topk_impl not in ("approx", "exact_lax"):
            raise ValueError(f"unknown DSA top-k implementation: {topk_impl}")
        self.sparse_impl = sparse_impl
        self.topk_impl = topk_impl

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        aux = {
            **aux,
            "index_topk": self.index_topk,
            "index_head_dim": self.index_head_dim,
            "index_n_heads": self.index_n_heads,
            "skip_offset": self.skip_offset,
            "full_slot": self.full_slot,
            "sparse_impl": self.sparse_impl,
            "topk_impl": self.topk_impl,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            index_topk=aux_data["index_topk"],
            index_head_dim=aux_data["index_head_dim"],
            index_n_heads=aux_data["index_n_heads"],
            skip_offset=aux_data["skip_offset"],
            full_slot=aux_data["full_slot"],
            sparse_impl=aux_data.get("sparse_impl", "page"),
            topk_impl=aux_data.get("topk_impl", "approx"),
            num_attn_heads=aux_data["num_attn_heads"],
            kv_lora_rank=aux_data["kv_lora_rank"],
            qk_nope_head_dim=aux_data["qk_nope_head_dim"],
            qk_rope_head_dim=aux_data["qk_rope_head_dim"],
            v_head_dim=aux_data["v_head_dim"],
            page_size=aux_data["page_size"],
            mesh=aux_data.get("mesh"),
            attention_data_partition_axis=aux_data.get("attention_data_partition_axis", "data"),
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
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **kwargs,
    ):
        del v
        q_rope = kwargs["q_rope"]
        k_rope = kwargs["k_rope"]
        indexer_type: str = kwargs.get("indexer_type", "full")
        q_idx: jax.Array | None = kwargs.get("q_idx")
        k_idx: jax.Array | None = kwargs.get("k_idx")
        idx_weights: jax.Array | None = kwargs.get("idx_weights")
        dsa_topk_in: jax.Array | None = kwargs.get("dsa_topk_in")
        dsa_topk_pages_in: jax.Array | None = kwargs.get("dsa_topk_pages_in")

        layer_id = layer.layer_id
        is_full = indexer_type == "full"
        slot = self.full_slot.get(layer_id) if is_full else None

        new_kv_c = k if k.ndim == 2 else jnp.squeeze(k, axis=1)
        new_k_pe = k_rope if k_rope.ndim == 2 else jnp.squeeze(k_rope, axis=1)

        kv_cache = token_to_kv_pool.get_fused_kv_buffer(layer_id)
        idx_cache = token_to_kv_pool.get_indexer_key_buffer(slot) if is_full else None

        sm_scale = (
            (1.0 / jnp.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim))
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )
        dpa = self.attention_data_partition_axis
        md = self.forward_metadata

        # ── dense short-circuit ────────────────────────────────────────────
        # skip_offset layers, or layers with no indexer projections wired in,
        # fall back to plain absorbed-MLA. The kv cache write still happens
        # inside the dense kernel via input_output_aliases.
        is_decode = forward_batch.forward_mode.is_decode()
        use_exact = self.sparse_impl == "exact"
        if layer_id < self.skip_offset or (is_full and q_idx is None):
            o, kv_cache = self._run_dense(
                q, q_rope, new_kv_c, new_k_pe, kv_cache, sm_scale, layer, dpa, md
            )
            # Only the final dense-prefix layer feeds an exact sparse layer.
            # The legacy decode path keeps its historical per-full-layer top-k.
            needs_prefix_topk = use_exact and layer_id == self.skip_offset - 1
            idx_cache, topk, topk_pages = self._maybe_index(
                is_full,
                q_idx,
                k_idx,
                idx_weights,
                idx_cache,
                dpa,
                md,
                compute_topk=needs_prefix_topk or (is_decode and not use_exact),
                compute_pages=is_decode and not use_exact,
                one_token_per_seq=is_decode,
            )
            return o, DSAFusedCache(kv=kv_cache, idx=idx_cache, topk=topk, topk_pages=topk_pages)

        # ── prefill/mixed → dense fallback: page_level is decode-only (one
        # query token per seq). EXTEND/MIXED chunks route to plain dense; the
        # indexer still writes idx cache so subsequent decode steps get valid
        # topk. Multi-request DECODE (T>1) does go through the sparse path.
        if not is_decode and not use_exact:
            o, kv_cache = self._run_dense(
                q, q_rope, new_kv_c, new_k_pe, kv_cache, sm_scale, layer, dpa, md
            )
            idx_cache, _, _ = self._maybe_index(
                is_full,
                q_idx,
                k_idx,
                idx_weights,
                idx_cache,
                dpa,
                md,
                compute_topk=False,
                compute_pages=False,
                one_token_per_seq=False,
            )
            return o, DSAFusedCache(kv=kv_cache, idx=idx_cache, topk=None, topk_pages=None)

        # ── indexer top-k (full) or reuse (shared) ─────────────────────────
        idx_cache, topk, topk_pages = self._maybe_index(
            is_full,
            q_idx,
            k_idx,
            idx_weights,
            idx_cache,
            dpa,
            md,
            compute_pages=not use_exact,
            one_token_per_seq=is_decode,
        )
        if not is_full:
            assert dsa_topk_in is not None, (
                f"shared layer {layer_id} requires dsa_topk_in from preceding full layer"
            )
            topk_use = dsa_topk_in
            topk_pages_use = dsa_topk_pages_in
        else:
            topk_use = topk
            topk_pages_use = topk_pages

        # ── sparse MLA over top-k ─────────────────────────────────────────
        if use_exact:
            o, kv_cache = self._run_exact(
                q,
                q_rope,
                new_kv_c,
                new_k_pe,
                kv_cache,
                topk_use,
                sm_scale,
                dpa,
                md,
            )
        else:
            o, kv_cache = self._run_sparse(
                q,
                q_rope,
                new_kv_c,
                new_k_pe,
                kv_cache,
                topk_use,
                topk_pages_use,
                sm_scale,
                dpa,
                md,
            )
        return o, DSAFusedCache(
            kv=kv_cache,
            idx=idx_cache,
            topk=topk if is_full else None,
            topk_pages=topk_pages if is_full else None,
        )

    # ────────────────────────────────────────────────────────────────────────
    # internals
    # ────────────────────────────────────────────────────────────────────────

    def _maybe_index(
        self,
        is_full,
        q_idx,
        k_idx,
        idx_weights,
        idx_cache,
        dpa,
        md,
        *,
        compute_topk=True,
        compute_pages=False,
        one_token_per_seq=False,
    ):
        """On full layers: write k_idx into paged indexer cache, compute top-k
        (and, when ``compute_pages``, the unique-page list for IndexShare).

        ``compute_topk=False`` (prefill/mixed) skips ``streamindex_topk_ref``
        entirely — the topk is only consumed by ``_run_sparse`` on DECODE, so
        during chunked prefill we only need the k_idx cache write."""
        if not is_full or q_idx is None:
            return idx_cache, None, None

        in_specs = (
            P(dpa, None, None),  # q_idx    [T, H_idx, D_idx] — replicated: softmax needs all heads
            P(dpa, None),  # k_idx    [T, D_idx]
            P(dpa, None),  # weights  [T, H_idx]
            P(dpa, None, None, None),  # idx_cache paged
            P(dpa),  # seq_lens
            P(dpa),  # page_indices
            P(dpa),  # cu_q_lens
            P(dpa),  # cu_kv_lens
            P(dpa),  # distribution
        )
        out_specs = (P(dpa, None, None, None), P(dpa, None), P(dpa, None))

        def _run(q_, k_, w_, cache_, seq_lens_, pi_, cuq_, cukv_, dist_):
            page_size = cache_.shape[1] * cache_.shape[2]
            idx_dim = cache_.shape[3]
            pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
            cache3d = cache_.reshape(cache_.shape[0], page_size, idx_dim)
            cache3d = _scatter_paged(cache3d, k_, seq_lens_, pi_, cuq_, cukv_, pages_per_seq)
            if compute_topk and _PAGE_TOPK_BUDGET > 0:
                # Page-scoring path: budget pages picked directly by max-pooled
                # page score; token-level topk is not materialized (sparse MLA
                # attends whole pages via topk_pages, so it never reads it).
                topk = jnp.full((q_.shape[0], 1), -1, jnp.int32)
                topk_pages = streamindex_page_topk_ref(
                    q_,
                    w_,
                    cache3d,
                    seq_lens_,
                    pi_,
                    cuq_,
                    cukv_,
                    dist_,
                    k_pages=min(_PAGE_TOPK_BUDGET, pages_per_seq),
                    pages_per_seq=pages_per_seq,
                    # compute_topk is decode-only: T == num_seqs, token i
                    # belongs to seq i — enables the O(S * max_kv) fast path.
                    one_token_per_seq=True,
                )
                return cache3d.reshape(cache_.shape), topk, topk_pages
            if compute_topk:
                topk = streamindex_topk_ref(
                    q_,
                    w_,
                    cache3d,
                    seq_lens_,
                    pi_,
                    cuq_,
                    cukv_,
                    dist_,
                    k=self.index_topk,
                    pages_per_seq=pages_per_seq,
                    one_token_per_seq=one_token_per_seq,
                    exact=self.topk_impl == "exact_lax",
                )
            else:
                topk = jnp.full((q_.shape[0], 1), -1, jnp.int32)
            if compute_pages:
                topk_pages = compute_topk_pages(
                    topk,
                    page_size=self.page_size,
                    pages_per_seq=pages_per_seq,
                    k_pages_max=512,
                )
                if _DEBUG_N_HIT:
                    jax.debug.print(
                        "dsa_n_hit min={} max={} mean={}",
                        jnp.min(jnp.sum(topk_pages >= 0, -1)),
                        jnp.max(jnp.sum(topk_pages >= 0, -1)),
                        jnp.mean(jnp.sum(topk_pages >= 0, -1).astype(jnp.float32)),
                    )
            else:
                topk_pages = jnp.full((topk.shape[0], 1), -1, jnp.int32)
            return cache3d.reshape(cache_.shape), topk, topk_pages

        idx_cache, topk, topk_pages = jax.shard_map(
            _run, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )(
            q_idx,
            k_idx,
            idx_weights,
            idx_cache,
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
        )
        return idx_cache, topk, (topk_pages if compute_pages else None)

    def _run_exact(self, ql, qpe, kvc, kpe, cache, topk, sm_scale, dpa, md):
        """Write the fused KV cache, map logical top-k to slots, then attend."""
        in_specs = (
            P(dpa, "tensor", None),
            P(dpa, "tensor", None),
            P(dpa, None),
            P(dpa, None),
            P(dpa, None, None, None),
            P(dpa, None),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
        )
        out_specs = (P(dpa, "tensor", None), P(dpa, None, None, None))

        def _run(ql_, qpe_, kvc_, kpe_, cache_, topk_, seq_lens_, pi_, cuq_, cukv_, dist_):
            del dist_
            page_size = cache_.shape[1] * cache_.shape[2]
            cache3d = cache_.reshape(cache_.shape[0], page_size, cache_.shape[-1])
            cache3d = _scatter_fused_kv_paged(
                cache3d,
                kvc_,
                kpe_,
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
                kv_lora_rank=self.kv_lora_rank,
            )
            physical_slots, selected_counts = _logical_topk_to_physical_slots(
                topk_, seq_lens_, pi_, cuq_, cukv_, page_size
            )

            # SparseCore requires bq_sparse*K to be a gather-wave multiple.
            # For C=2 decode, bq_sparse=2 is the smallest valid choice at K=2048.
            # Larger extend batches use the kernel's validated 128-query tile.
            bq_sparse = 2 if ql_.shape[0] < 128 else 128
            bq = 1 if bq_sparse == 2 else 32
            q_padded = _pad_first_axis(ql_, bq_sparse)
            qpe_padded = _pad_first_axis(qpe_, bq_sparse)
            slots_padded = _pad_first_axis(physical_slots, bq_sparse)
            counts_padded = _pad_first_axis(selected_counts, bq_sparse)
            output = sparse_core_tensor_core_dsa(
                q_padded,
                qpe_padded,
                cache3d.reshape(-1, cache3d.shape[-1]),
                slots_padded,
                counts_padded,
                sm_scale,
                bq_sparse=bq_sparse,
                bq=bq,
                b_topk=128,
            )
            return output[: ql_.shape[0]], cache3d.reshape(cache_.shape)

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            ql,
            qpe,
            kvc,
            kpe,
            cache,
            topk,
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
        )

    def _run_sparse(self, ql, qpe, kvc, kpe, cache, topk, topk_pages, sm_scale, dpa, md):
        has_pages = topk_pages is not None
        if not has_pages:
            topk_pages = jnp.full((topk.shape[0], 1), -1, jnp.int32)
        in_specs = (
            P(dpa, "tensor", None),
            P(dpa, "tensor", None),
            P(dpa, None),
            P(dpa, None),
            P(dpa, None, None, None),
            P(dpa, None),  # topk [T, k]
            P(dpa, None),  # topk_pages [T, k_pages_max]
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
        )
        out_specs = (P(dpa, "tensor", None), P(dpa, None, None, None))

        def _run(ql_, qpe_, kvc_, kpe_, cache_, topk_, tpages_, seq_lens_, pi_, cuq_, cukv_, dist_):
            page_size = cache_.shape[1] * cache_.shape[2]
            pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
            return sparse_mla_page_level(
                ql_,
                qpe_,
                kvc_,
                kpe_,
                cache_,
                seq_lens_,
                topk_,
                pi_,
                cuq_,
                cukv_,
                dist_,
                tpages_ if has_pages else None,
                sm_scale=float(sm_scale),
                page_size=page_size,
                pages_per_seq=pages_per_seq,
                kv_lora_rank=self.kv_lora_rank,
                k_pages_max=(
                    min(_PAGE_TOPK_BUDGET, pages_per_seq) + 1 if _PAGE_TOPK_BUDGET > 0 else 512
                ),
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            ql,
            qpe,
            kvc,
            kpe,
            cache,
            topk,
            topk_pages,
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
        )

    def _run_dense(self, ql, qpe, kvc, kpe, cache, sm_scale, layer, dpa, md):
        in_specs = (
            P(dpa, "tensor", None),
            P(dpa, "tensor", None),
            P(dpa, None),
            P(dpa, None),
            P(dpa, None, None, None),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
        )
        out_specs = (P(dpa, "tensor", None), P(dpa, None, None, None))
        sw = layer.sliding_window_size if layer is not None else None
        sc = layer.logit_cap if layer is not None else None

        def _run(ql_, qpe_, kvc_, kpe_, cache_, seq_lens_, pi_, cuq_, cukv_, dist_):
            return mla_ragged_paged_attention(
                ql_,
                qpe_,
                kvc_,
                kpe_,
                cache_,
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
                dist_,
                sm_scale=sm_scale,
                sliding_window=sw,
                soft_cap=sc,
                num_kv_pages_per_block=self.num_kv_pages_per_block,
                num_queries_per_block=self.num_queries_per_block,
                decode_batch_size=self.decode_batch_size,
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            ql,
            qpe,
            kvc,
            kpe,
            cache,
            md.seq_lens,
            md.page_indices,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.distribution,
        )


def _scatter_paged(
    cache3d: jax.Array,
    new_tokens: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    pages_per_seq: int,
) -> jax.Array:
    """Write new_tokens[t] into cache at (page, offset) for each seq's tail slots.

    Jit-compatible reference for the paged cache write that the Pallas kernel
    does via input_output_aliases. For seq i with q tokens cu_q_lens[i]..[i+1),
    token j lands at absolute position seq_lens[i] - (q_end - q_start) + j.
    """
    page_size = cache3d.shape[1]
    T = new_tokens.shape[0]
    S = seq_lens.shape[0]

    t = jnp.arange(T)
    seq_id = jnp.searchsorted(cu_q_lens[1:], t, side="right")
    seq_id = jnp.clip(seq_id, 0, S - 1)
    q_start = cu_q_lens[seq_id]
    q_end = cu_q_lens[seq_id + 1]
    kv_len = seq_lens[seq_id]
    abs_pos = jnp.maximum(kv_len - (q_end - q_start) + (t - q_start), 0)
    valid = (t >= q_start) & (t < q_end) & (kv_len > 0)

    page_local = abs_pos // page_size
    offset = abs_pos % page_size
    page = page_indices[cu_kv_lens[seq_id] // page_size + page_local]

    sentinel = cache3d.shape[0] - 1
    safe_page = jnp.where(valid, page, sentinel)
    safe_off = jnp.where(valid, offset, 0)
    return cache3d.at[safe_page, safe_off].set(new_tokens.astype(cache3d.dtype))


def _scatter_fused_kv_paged(
    cache3d: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    *,
    kv_lora_rank: int,
) -> jax.Array:
    """Pack latent KV + RoPE exactly as ``MLATokenToKVPool`` and scatter it."""
    rope_offset = (kv_lora_rank + 127) // 128 * 128
    packed = jnp.zeros((new_kv_c.shape[0], cache3d.shape[-1]), dtype=cache3d.dtype)
    packed = packed.at[:, :kv_lora_rank].set(new_kv_c.astype(cache3d.dtype))
    packed = packed.at[:, rope_offset : rope_offset + new_k_pe.shape[-1]].set(
        new_k_pe.astype(cache3d.dtype)
    )
    pages_per_seq = page_indices.shape[0] // seq_lens.shape[0]
    return _scatter_paged(
        cache3d,
        packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        pages_per_seq,
    )


def _logical_topk_to_physical_slots(
    topk: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Map per-sequence logical token positions to flattened cache rows.

    ``page_indices`` is ragged-packed. Sequence ``i`` starts at
    ``cu_kv_lens[i] // page_size``; using a fixed ``i * pages_per_seq`` stride
    would silently cross sequence boundaries for unequal context lengths.
    """
    num_tokens, topk_size = topk.shape
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_ids = jnp.searchsorted(cu_q_lens[1:], token_ids, side="right")
    seq_ids = jnp.clip(seq_ids, 0, seq_lens.shape[0] - 1)

    logical = jnp.maximum(topk, 0)
    page_ptr = cu_kv_lens[seq_ids, None] // page_size + logical // page_size
    ptr_in_bounds = (page_ptr >= 0) & (page_ptr < page_indices.shape[0])
    safe_ptr = jnp.clip(page_ptr, 0, page_indices.shape[0] - 1)
    physical_pages = page_indices[safe_ptr]
    query_valid = token_ids < cu_q_lens[-1]
    valid = (
        query_valid[:, None]
        & (topk >= 0)
        & (logical < seq_lens[seq_ids, None])
        & ptr_in_bounds
        & (physical_pages >= 0)
    )
    physical_slots = physical_pages * page_size + logical % page_size
    physical_slots = jnp.where(valid, physical_slots, jnp.int32(0))
    selected_counts = jnp.sum(valid, axis=1, dtype=jnp.int32)

    # ``lax.top_k`` places all finite values before -inf padding, so valid
    # entries are a prefix as required by selected_counts. Keep this helper
    # pure/JIT-compatible; correctness tests cover the prefix invariant.
    del topk_size
    return physical_slots.astype(jnp.int32), selected_counts


def _pad_first_axis(x: jax.Array, multiple: int) -> jax.Array:
    pad = (-x.shape[0]) % multiple
    return jnp.pad(x, ((0, pad), *((0, 0),) * (x.ndim - 1)))
