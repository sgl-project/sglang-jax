"""DSA sparse attention backend (DeepSeek Sparse Attention + IndexShare).

Wraps the absorbed-MLA path with a lightning-indexer top-k selection so core
attention runs over at most ``index_topk`` KV positions per query. IndexShare
(GLM-5.2) is realised by threading the last full-layer's ``topk_indices``
through the model's per-layer loop and reusing it on ``shared`` layers.

Phase A path uses jnp reference kernels (:mod:`sgl_jax.srt.kernels.dsa.ref`);
DECODE runs the Pallas ``sparse_mla_page_level``; EXTEND falls back to
plain dense (the indexer still writes idx cache for later decode steps).
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

from sgl_jax.srt.kernels.dsa.ref import streamindex_page_topk_ref, streamindex_topk_ref
from sgl_jax.srt.kernels.dsa.sparse_mla import compute_topk_pages, sparse_mla_page_level
from sgl_jax.srt.kernels.dsa.sparse_mla_prefill import prefill_write_and_attend_ragged
from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import (
    paged_write_back,
    prefill_write_and_attend_ragged_qblock,
)
from sgl_jax.srt.kernels.dsa.streamindex_topk import (
    streamindex_page_topk,
    streamindex_topk,
)
from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.utils.profiling_utils import named_scope

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_attention import RadixAttention
    from sgl_jax.srt.mem_cache.memory_pool import KVCache
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_SPARSE_PALLAS_MAX_T = 1
# Page-budget for the page-scoring indexer path: the indexer max-pools token
# scores per page and selects this many pages directly (0 = off, use the
# token-topk + page-union path with k_pages_max=512). Bounds sparse-MLA cost
# to O(budget) flat vs the union path which saturates 512 pages at long ctx.
_PAGE_TOPK_BUDGET = int(os.environ.get("DSA_PAGE_TOPK", "0"))
# Opt-in: score+topk via the Pallas streamindex kernel instead of the jnp
# reference. The kernel reads O(actual kv_len) pages (vs the ref's O(max_ctx)
# padded gather) and shares the ref's exact scoring semantics
# (sum_h relu(q·k)·w_h, approx_max_k). Token-topk path only; DSA_PAGE_TOPK
# takes precedence when both are set.
_INDEXER_KERNEL = os.environ.get("DSA_INDEXER_KERNEL", "0") == "1"
# Opt-in: prefill/extend page-level indexer top-k via the page-pooled Pallas
# streamindex kernel instead of the jnp reference. The kernel scores O(actual
# kv_len) pages (ref is O(max_ctx) padded gather) and never materializes
# [T, max_kv] token scores in HBM — the dominant HLO-temporaries term at long
# context. Same selection semantics as the ref (parity-gated). Default OFF.
_INDEXER_KERNEL_PREFILL = os.environ.get("DSA_INDEXER_KERNEL_PREFILL", "0") == "1"
# bkv block of 64 pages (4K tokens @ page 64) benched fastest across
# B=8..64, ctx 8K..128K on v7x/v6e.
_INDEXER_KERNEL_KV_PAGES_PER_BLOCK = 64

# Opt-in: run EXTEND/prefill through the fused sparse-MLA prefill kernel
# (``sparse_mla_prefill.sparse_mla_attention``) at page granularity instead of the
# dense fallback. Default OFF ⇒ prefill behaviour is unchanged. Page-level
# selection (read_block == page_size) consumes the indexer's page-topk directly.
# Scope: packed-ragged extend — supports the full serving surface via the same
# ragged metadata the dense path uses: multi-request batching (max_running>1),
# radix/prefix caching (a cache hit is an extend with a non-zero prefix), and
# chunked prefill (each chunk is an extend over the growing prefix). All three
# reduce to the same per-query-token kernel contract and are validated by the
# A1/A2 parity gates in test/srt/kernels/dsa/test_sparse_mla_prefill_parity.py.
_PREFILL_SPARSE = int(os.environ.get("DSA_PREFILL_SPARSE", "0"))
# Within DSA_PREFILL_SPARSE=1, the sparse extend runs through the query-BLOCK
# kernel (``sparse_mla_prefill_qblock``) by DEFAULT — QB queries share one
# program and each block DMAs its selected-page *union* once, instead of one
# program (and K page DMAs) per query. Semantics are parity-gated against the
# per-query kernel (same masked-softmax math); it wins whenever neighbouring
# queries' selections overlap (sinks + local windows). The default sparse=0
# path is completely unaffected. ``DSA_PREFILL_QBLOCK=0`` is an escape hatch
# back to the per-query kernel (e.g. for pathological no-locality selections).
_PREFILL_QBLOCK = os.environ.get("DSA_PREFILL_QBLOCK", "1") == "1"
# Default 256 from the QB sweep on GLM-5.2 (v7x tp16 and v6e-64): saturated
# attend work scales as (T/QB) * ctx_pages, and 64->256 was uniformly positive
# (110k TTFT -10%, 16k -11%) with paired-accuracy gates clean at every step.
# 512 projects <2% further and grows the per-block union tail, so 256 is the
# sweet spot. Override per deployment via DSA_PREFILL_QBLOCK_QB.
_PREFILL_QBLOCK_QB = int(os.environ.get("DSA_PREFILL_QBLOCK_QB", "256"))
# Opt-in (W7): run speculative TARGET_VERIFY / DRAFT_EXTEND batches through the
# DECODE machinery instead of the sparse-prefill kernels. Every spec token
# becomes its own one-query "sequence" (kv_len = its own position + 1), the new
# KV rows are written up front with paged_write_back, then the decode indexer
# (one_token_per_seq) and sparse_mla_page_level run exactly as for decode. The
# extend-shaped prefill path costs ~10 ms/step at T=4 (bq_512 indexer blocks +
# qblock machinery); this trades it for n decode queries. Default OFF.
_SPEC_AS_DECODE = os.environ.get("DSA_SPEC_AS_DECODE", "0") == "1"


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
        **mla_kwargs,
    ):
        super().__init__(**mla_kwargs)
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.skip_offset = skip_offset
        self.full_slot = full_slot

    def tree_flatten(self):
        children, aux = super().tree_flatten()
        aux = {
            **aux,
            "index_topk": self.index_topk,
            "index_head_dim": self.index_head_dim,
            "index_n_heads": self.index_n_heads,
            "skip_offset": self.skip_offset,
            "full_slot": self.full_slot,
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
        # GLM-5.2 MTP IndexShare (index_share_for_mtp_iteration): a FULL layer
        # asked to reuse a threaded selection skips its own top-k (still writes
        # its indexer-key cache) and attends over the caller's indices instead.
        dsa_topk_reuse: bool = bool(kwargs.get("dsa_topk_reuse", False))

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
        # Only a "full" layer with no indexer projections wired (q_idx is None)
        # falls back to plain absorbed-MLA. Every full layer that HAS an indexer
        # runs its own top-k + sparse attention, matching config.indexer_types
        # (for GLM-5.2 layers 0..skip_offset-1 are declared "full" ⇒ sparse, not
        # dense). We intentionally do NOT gate on `layer_id < self.skip_offset`
        # here: index_skip_topk_offset selects which leading layers own their
        # indexer vs share one (IndexShare), not which layers skip sparsity.
        # Routing those full layers dense (a) diverged from the reference and
        # (b) cascaded in sparse prefill — their shared followers then had no
        # pages to reuse and fell to dense too (a 6-layer dense O(T²) block).
        # The kv cache write still happens inside the dense kernel via
        # input_output_aliases in the remaining (no-indexer) dense case.
        is_decode = forward_batch.forward_mode.is_decode()
        if is_full and q_idx is None:
            o, kv_cache = self._run_dense(
                q, q_rope, new_kv_c, new_k_pe, kv_cache, sm_scale, layer, dpa, md
            )
            idx_cache, topk, topk_pages = self._maybe_index(
                is_full,
                q_idx,
                k_idx,
                idx_weights,
                idx_cache,
                dpa,
                md,
                compute_topk=is_decode,
                compute_pages=is_decode,
            )
            return o, DSAFusedCache(kv=kv_cache, idx=idx_cache, topk=topk, topk_pages=topk_pages)

        # ── W7: spec verify / draft-extend in decode form ─────────────────
        mode = forward_batch.forward_mode
        if (
            not is_decode
            and _SPEC_AS_DECODE
            and (mode.is_target_verify() or mode.is_draft_extend())
        ):
            return self._run_spec_as_decode(
                q,
                q_rope,
                new_kv_c,
                new_k_pe,
                kv_cache,
                idx_cache,
                q_idx,
                k_idx,
                idx_weights,
                is_full,
                dsa_topk_in,
                dsa_topk_pages_in,
                dsa_topk_reuse,
                sm_scale,
                dpa,
                md,
            )

        # ── prefill/mixed ─────────────────────────────────────────────────
        # Default: dense fallback (page_level is decode-only; the indexer still
        # writes the idx cache so later decode steps get valid topk).
        # Opt-in (DSA_PREFILL_SPARSE): run the fused sparse-MLA *prefill* kernel
        # over the indexer's page-topk — this is the EXTEND-sparsity gap PR.
        # MIXED (chunked-prefill continuous batching: extend chunks + decodes in one
        # batch) is is_decode()==False, so it flows here too; the packed-ragged path
        # treats each decode request as a 1-query extend (extend_seq_lens==1), which
        # the per-query-token kernel handles uniformly.
        if not is_decode:
            if _PREFILL_SPARSE:
                reuse_pages = is_full and dsa_topk_reuse and dsa_topk_pages_in is not None
                idx_cache, topk_pages = self._maybe_index_prefill_pages(
                    is_full,
                    q_idx,
                    k_idx,
                    idx_weights,
                    idx_cache,
                    dpa,
                    md,
                    compute_pages=not reuse_pages,
                )
                # Page selection for the sparse-prefill attend. A FULL (indexer)
                # layer just produced [T, k_pages] page-topk; a SHARED layer gets
                # None from `_maybe_index_prefill_pages` and reuses the preceding
                # full layer's pages threaded via `dsa_topk_pages_in` (IndexShare).
                # During prefill that thread is ALSO [T, k_pages] over the SAME T
                # query rows (the full layer above ran the same single-shot prefill)
                # — NOT the decode one-query-per-seq shape. None (a shared layer
                # before any full layer) ⇒ fall through to the dense attend below.
                topk_pages_use = topk_pages if (is_full and not reuse_pages) else dsa_topk_pages_in
                if topk_pages_use is not None:
                    o, kv_cache = self._run_sparse_prefill(
                        q,
                        q_rope,
                        new_kv_c,
                        new_k_pe,
                        kv_cache,
                        topk_pages_use,
                        sm_scale,
                        dpa,
                        md,
                        forward_batch,
                    )
                    return o, DSAFusedCache(
                        kv=kv_cache,
                        idx=idx_cache,
                        topk=None,
                        topk_pages=topk_pages if is_full else None,
                    )
                # no selection available (shared layer before any full) → dense
                # fallback; idx cache already written above.
                o, kv_cache = self._run_dense(
                    q, q_rope, new_kv_c, new_k_pe, kv_cache, sm_scale, layer, dpa, md
                )
                return o, DSAFusedCache(kv=kv_cache, idx=idx_cache, topk=None, topk_pages=None)

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
            )
            return o, DSAFusedCache(kv=kv_cache, idx=idx_cache, topk=None, topk_pages=None)

        # ── indexer top-k (full) or reuse (shared / MTP IndexShare) ────────
        reuse_topk = is_full and dsa_topk_reuse and dsa_topk_in is not None
        idx_cache, topk, topk_pages = self._maybe_index(
            is_full,
            q_idx,
            k_idx,
            idx_weights,
            idx_cache,
            dpa,
            md,
            compute_topk=not reuse_topk,
            compute_pages=not reuse_topk,
        )
        if not is_full or reuse_topk:
            assert (
                dsa_topk_in is not None
            ), f"shared layer {layer_id} requires dsa_topk_in from preceding full layer"
            topk_use = dsa_topk_in
            topk_pages_use = dsa_topk_pages_in
        else:
            topk_use = topk
            topk_pages_use = topk_pages

        # ── sparse MLA over top-k ─────────────────────────────────────────
        o, kv_cache = self._run_sparse(
            q, q_rope, new_kv_c, new_k_pe, kv_cache, topk_use, topk_pages_use, sm_scale, dpa, md
        )
        return o, DSAFusedCache(
            kv=kv_cache,
            idx=idx_cache,
            topk=topk if (is_full and not reuse_topk) else None,
            topk_pages=topk_pages if (is_full and not reuse_topk) else None,
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
            if compute_topk and _INDEXER_KERNEL:
                topk = streamindex_topk(
                    q_,
                    w_,
                    cache3d.reshape(cache_.shape),
                    seq_lens_,
                    _fixed_stride_pages(pi_, cukv_, page_size, pages_per_seq),
                    cuq_,
                    dist_,
                    k=self.index_topk,
                    # GLM indexer keys are uncompressed (one key per token).
                    compression_ratio=1,
                    num_kv_pages_per_block=_INDEXER_KERNEL_KV_PAGES_PER_BLOCK,
                    num_queries_per_block=1,
                )
            elif compute_topk:
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
                    # compute_topk is decode-only: token i belongs to seq i.
                    one_token_per_seq=True,
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

    def _maybe_index_prefill_pages(
        self, is_full, q_idx, k_idx, idx_weights, idx_cache, dpa, md, *, compute_pages=True
    ):
        """Prefill page-topk: scatter ``k_idx`` into the paged indexer cache and,
        on full layers, compute per-query **causal** page-level top-k via the
        general (non-decode, ``one_token_per_seq=False``) indexer path.

        ``compute_pages=False`` (MTP IndexShare reuse steps) only performs the
        indexer-key cache write and returns ``topk_pages=None``.

        Returns ``(idx_cache, topk_pages)`` where ``topk_pages`` is ``[T, k_pages]``
        seq-local page ids (-1 padded) — exactly the sparse kernel's per-query
        unit ids at ``read_block == page_size``. ``topk_pages`` is ``None`` on
        shared layers / when no indexer is wired (caller reuses the threaded one).
        """
        if not is_full or q_idx is None:
            return idx_cache, None
        k_pages = (self.index_topk + self.page_size - 1) // self.page_size
        in_specs = (
            P(dpa, None, None),  # q_idx    [T, H_idx, D_idx] replicated
            P(dpa, None),  # k_idx    [T, D_idx]
            P(dpa, None),  # weights  [T, H_idx]
            P(dpa, None, None, None),  # idx_cache paged
            P(dpa),  # seq_lens
            P(dpa),  # page_indices
            P(dpa),  # cu_q_lens
            P(dpa),  # cu_kv_lens
            P(dpa),  # distribution
        )
        out_specs = (P(dpa, None, None, None), P(dpa, None))

        def _run(q_, k_, w_, cache_, seq_lens_, pi_, cuq_, cukv_, dist_):
            page_size = cache_.shape[1] * cache_.shape[2]
            idx_dim = cache_.shape[3]
            pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
            cache3d = cache_.reshape(cache_.shape[0], page_size, idx_dim)
            cache3d = _scatter_paged(cache3d, k_, seq_lens_, pi_, cuq_, cukv_, pages_per_seq)
            if not compute_pages:
                return cache3d.reshape(cache_.shape), jnp.full(
                    (q_.shape[0], k_pages), -1, jnp.int32
                )
            if _INDEXER_KERNEL_PREFILL:
                # dist_[2] == number of real (seq_len > 0) sequences in this
                # EXTEND batch: mla_backend builds distribution = [0, 0, N] for
                # ForwardMode.EXTEND (decode runs as a separate forward), so the
                # kernel's per-seq grid over [0, N) matches the ref's masked
                # full-batch loop exactly; padded seqs stay -1 on both paths.
                topk_pages = streamindex_page_topk(
                    q_,
                    w_,
                    cache3d.reshape(cache_.shape),
                    seq_lens_,
                    # the kernel indexes page_indices[seq_id * pages_per_seq + p]
                    # (fixed stride), but sglang packs seq i's pages at
                    # cu_kv_lens[i]//page_size (variable stride) — repack, same
                    # as the decode call site, or a multi-request EXTEND batch
                    # with unequal lengths reads another sequence's pages.
                    _fixed_stride_pages(pi_, cukv_, page_size, pages_per_seq),
                    cuq_,
                    dist_[2],
                    k_pages=k_pages,
                )
            else:
                topk_pages = streamindex_page_topk_ref(
                    q_,
                    w_,
                    cache3d,
                    seq_lens_,
                    pi_,
                    cuq_,
                    cukv_,
                    dist_,
                    k_pages=k_pages,
                    pages_per_seq=pages_per_seq,
                    one_token_per_seq=False,  # prefill: T>1 tokens/seq, per-query causal
                )
            return cache3d.reshape(cache_.shape), topk_pages

        idx_cache, topk_pages = jax.shard_map(
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
        if not compute_pages:
            return idx_cache, None
        return idx_cache, topk_pages

    def _spec_metadata_as_decode(self, md, num_tokens: int, dpa):
        """Pseudo-decode metadata for a spec batch (rank-local build, P(dpa) out)."""
        page_size = self.page_size
        t_local = num_tokens // self.mesh.shape[dpa]
        in_specs = (P(dpa), P(dpa), P(dpa), P(dpa))
        out_specs = (P(dpa), P(dpa), P(dpa), P(dpa), P(dpa))

        def _run(sl_, cuq_, cukv_, pi_):
            return _spec_pseudo_decode_metadata(sl_, cuq_, cukv_, pi_, t_local, page_size)

        kv_len, cu_q, cu_kv, pi, dist = jax.shard_map(
            _run, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )(md.seq_lens, md.cu_q_lens, md.cu_kv_lens, md.page_indices)
        return type(md)(
            cu_q_lens=cu_q, cu_kv_lens=cu_kv, page_indices=pi, seq_lens=kv_len, distribution=dist
        )

    def _prewrite_spec_kv(self, kvc, kpe, cache, dpa, md):
        """Write every spec token's latent KV row to its slot before attention.

        The decode kernel writes only the token it attends for; token t must
        already see tokens < t of the same request, so land all rows first (the
        kernel then rewrites its own row with identical bytes).
        """
        page_size = self.page_size
        kv_lora_rank = self.kv_lora_rank
        in_specs = (
            P(dpa, None),
            P(dpa, None),
            P(dpa, None, None, None),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
        )
        out_specs = P(dpa, None, None, None)

        def _run(kvc_, kpe_, cache_, sl_, cuq_, cukv_, pi_):
            t = kvc_.shape[0]
            rope = kpe_.shape[-1]
            loc = _spec_token_slots(sl_, cuq_, cukv_, pi_, t, page_size)
            row = jnp.zeros((t, cache_.shape[3]), cache_.dtype)
            row = row.at[:, :kv_lora_rank].set(kvc_.astype(cache_.dtype))
            row = row.at[:, kv_lora_rank : kv_lora_rank + rope].set(
                kpe_.reshape(t, rope).astype(cache_.dtype)
            )
            return paged_write_back(cache_, row, loc, page_size=page_size)

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            kvc, kpe, cache, md.seq_lens, md.cu_q_lens, md.cu_kv_lens, md.page_indices
        )

    def _run_spec_as_decode(
        self,
        q,
        q_rope,
        new_kv_c,
        new_k_pe,
        kv_cache,
        idx_cache,
        q_idx,
        k_idx,
        idx_weights,
        is_full,
        dsa_topk_in,
        dsa_topk_pages_in,
        dsa_topk_reuse,
        sm_scale,
        dpa,
        md,
    ):
        """W7: TARGET_VERIFY / DRAFT_EXTEND through the decode path (see _SPEC_AS_DECODE)."""
        num_tokens = q.shape[0]
        pmd = self._spec_metadata_as_decode(md, num_tokens, dpa)
        kv_cache = self._prewrite_spec_kv(new_kv_c, new_k_pe, kv_cache, dpa, md)
        # IndexShare on the draft side threads page-topk only; either form counts as reuse.
        reuse = (
            is_full
            and dsa_topk_reuse
            and (dsa_topk_in is not None or dsa_topk_pages_in is not None)
        )
        idx_cache, topk, topk_pages = self._maybe_index(
            is_full,
            q_idx,
            k_idx,
            idx_weights,
            idx_cache,
            dpa,
            pmd,
            compute_topk=not reuse,
            compute_pages=not reuse,
        )
        if not is_full or reuse:
            assert (
                dsa_topk_in is not None or dsa_topk_pages_in is not None
            ), "shared layer requires dsa_topk_in / dsa_topk_pages_in from a preceding full layer"
            topk_use = dsa_topk_in
            topk_pages_use = dsa_topk_pages_in
        else:
            topk_use = topk
            topk_pages_use = topk_pages
        if topk_use is None:
            topk_use = _placeholder_topk_like(topk_pages_use)
        o, kv_cache = self._run_sparse(
            q, q_rope, new_kv_c, new_k_pe, kv_cache, topk_use, topk_pages_use, sm_scale, dpa, pmd
        )
        return o, DSAFusedCache(
            kv=kv_cache,
            idx=idx_cache,
            topk=topk if (is_full and not reuse) else None,
            topk_pages=topk_pages if (is_full and not reuse) else None,
        )

    def _run_sparse_prefill(
        self, ql, qpe, kvc, kpe, cache, topk_pages, sm_scale, dpa, md, forward_batch
    ):
        """Fused sparse-MLA prefill: self-write the current chunk's latent into the
        paged fused cache, then attend only the page-topk pages.

        Packed-ragged: threads the same ragged metadata the dense/indexer paths use
        (``seq_lens``/``cu_q_lens``/``cu_kv_lens``/``page_indices``) so a batch of
        multiple requests (``max_running>1``) prefills in one call. With a single
        request this reduces to the previous single-shot behaviour. Returns
        ``(o_latent, updated_cache)``.
        """
        positions = forward_batch.positions.astype(jnp.int32)
        page_size = self.page_size
        kv_lora_rank = self.kv_lora_rank
        sm = float(sm_scale)
        # Speculative verify / draft-extend batches carry no per-token
        # out_cache_loc: the scheduler hands the 2 x draft_token_num
        # allocation-extension list (-1 padded, schedule_batch spec decode), and
        # the dense MLA / FA kernels never read it -- they place query token i
        # of sequence s at kv index seq_lens[s] - q_len[s] + i. Derive the
        # self-write slots the same way from the ragged metadata (rank-local,
        # inside the shard_map) so both paths write the same cells.
        mode = forward_batch.forward_mode
        derive_loc = mode.is_target_verify() or mode.is_draft_extend()
        if derive_loc:
            loc = None
        else:
            loc = forward_batch.out_cache_loc.astype(jnp.int32)
            if loc.shape[0] != ql.shape[0]:
                raise ValueError(
                    "sparse prefill self-write: out_cache_loc has "
                    f"{loc.shape[0]} entries for {ql.shape[0]} tokens ({mode.name})"
                )

        in_specs = (
            P(dpa, "tensor", None),  # ql   [T, H, kv_lora_rank]
            P(dpa, "tensor", None),  # qpe  [T, H, rope]
            P(dpa, None),  # kvc  [T, kv_lora_rank]
            P(dpa, None),  # kpe  [T, rope]
            P(dpa, None, None, None),  # cache
            P(dpa, None),  # topk_pages [T, K]
            P(dpa),  # positions [T]
            P(dpa),  # seq_lens [S]
            P(dpa),  # cu_q_lens [S+1]
            P(dpa),  # cu_kv_lens [S+1]
            P(dpa),  # page_indices [total_pages]
        )
        args = [
            ql,
            qpe,
            kvc,
            kpe,
            cache,
            topk_pages,
            positions,
            md.seq_lens,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.page_indices,
        ]
        if loc is not None:
            in_specs = in_specs + (P(dpa),)  # loc [T]
            args.append(loc)
        out_specs = (P(dpa, "tensor", None), P(dpa, None, None, None))

        def _run(ql_, qpe_, kvc_, kpe_, cache_, tp_, pos_, sl_, cuq_, cukv_, pi_, *rest):
            if rest:
                loc_ = rest[0]
            else:
                loc_ = _spec_token_slots(sl_, cuq_, cukv_, pi_, ql_.shape[0], page_size)
            if _PREFILL_QBLOCK:
                o, cache_new = prefill_write_and_attend_ragged_qblock(
                    ql_,
                    qpe_,
                    kvc_,
                    kpe_,
                    cache_,
                    tp_,
                    pos_,
                    loc_,
                    sl_,
                    cuq_,
                    cukv_,
                    pi_,
                    kv_lora_rank=kv_lora_rank,
                    page_size=page_size,
                    sm_scale=sm,
                    query_block=_PREFILL_QBLOCK_QB,
                )
            else:
                o, cache_new = prefill_write_and_attend_ragged(
                    ql_,
                    qpe_,
                    kvc_,
                    kpe_,
                    cache_,
                    tp_,
                    pos_,
                    loc_,
                    sl_,
                    cuq_,
                    cukv_,
                    pi_,
                    kv_lora_rank=kv_lora_rank,
                    page_size=page_size,
                    sm_scale=sm,
                )
            return o.astype(ql_.dtype), cache_new

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(*args)

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


def _spec_token_slots(
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    page_indices: jax.Array,
    num_tokens: int,
    page_size: int,
) -> jax.Array:
    """Per-token KV slot for a verify / draft-extend batch, from ragged metadata.

    Rank-local (call inside the shard_map). Query token ``i`` of sequence ``s``
    lands at kv index ``seq_lens[s] - q_len[s] + i`` -- the placement the dense
    MLA v2 / FA kernels use -- inside the packed page table (sequence ``s``'s
    pages start at ``cu_kv_lens[s] // page_size``). Tokens past
    ``cu_q_lens[-1]`` (bucket padding) and tokens of empty sequences get -1,
    which the self-write drops.
    """
    num_seqs = seq_lens.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seg = jnp.searchsorted(cu_q_lens, tok, side="right").astype(jnp.int32) - 1
    seg_c = jnp.clip(seg, 0, num_seqs - 1)
    q_start = cu_q_lens[seg_c]
    q_len = cu_q_lens[seg_c + 1] - q_start
    kv_pos = seq_lens[seg_c] - q_len + (tok - q_start)
    page_slot = cu_kv_lens[seg_c] // page_size + kv_pos // page_size
    page_slot = jnp.clip(page_slot, 0, page_indices.shape[0] - 1)
    slot = page_indices[page_slot] * page_size + kv_pos % page_size
    valid = (
        (tok < cu_q_lens[num_seqs])
        & (seg >= 0)
        & (seg < num_seqs)
        & (seq_lens[seg_c] > 0)
        & (q_len > 0)
        & (kv_pos >= 0)
    )
    return jnp.where(valid, slot, -1).astype(jnp.int32)


def _spec_pseudo_decode_metadata(
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    page_indices: jax.Array,
    num_tokens: int,
    page_size: int,
):
    """View every spec token as its own one-query decode sequence (rank-local).

    Token ``i`` (origin sequence ``s``, query offset ``t``) becomes pseudo-sequence
    ``i`` with ``kv_len = seq_lens[s] - q_len[s] + t + 1`` (its own slot + 1, so
    the decode causal bound ``pos < kv_len`` equals the prefill bound
    ``pos <= abs_q``), a fixed-stride copy of ``s``'s page segment of width
    ``W = len(page_indices) // num_seqs`` (the ``pages_per_seq`` bound the decode
    path already assumes), ``cu_q = arange``, ``cu_kv = i * W * page_size``.
    Padding tokens and empty sequences get ``kv_len 0`` and point at page 0.
    Returns ``(kv_len, cu_q_lens, cu_kv_lens, page_indices, distribution)``.
    """
    num_seqs = seq_lens.shape[0]
    width = max(page_indices.shape[0] // num_seqs, 1)
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seg = jnp.searchsorted(cu_q_lens, tok, side="right").astype(jnp.int32) - 1
    seg_c = jnp.clip(seg, 0, num_seqs - 1)
    q_start = cu_q_lens[seg_c]
    q_len = cu_q_lens[seg_c + 1] - q_start
    valid = (
        (tok < cu_q_lens[num_seqs])
        & (seg >= 0)
        & (seg < num_seqs)
        & (seq_lens[seg_c] > 0)
        & (q_len > 0)
    )
    kv_len = jnp.where(valid, seq_lens[seg_c] - q_len + (tok - q_start) + 1, 0).astype(jnp.int32)
    seg_start = cu_kv_lens[seg_c] // page_size
    src = jnp.clip(
        seg_start[:, None] + jnp.arange(width, dtype=jnp.int32)[None, :],
        0,
        page_indices.shape[0] - 1,
    )
    pi = jnp.where(valid[:, None], page_indices[src], 0).astype(jnp.int32).reshape(-1)
    cu_q = jnp.arange(num_tokens + 1, dtype=jnp.int32)
    cu_kv = (jnp.arange(num_tokens + 1, dtype=jnp.int32) * (width * page_size)).astype(jnp.int32)
    n_valid = jnp.sum(valid).astype(jnp.int32)
    dist = jnp.stack([n_valid, n_valid, n_valid]).astype(jnp.int32)
    return kv_len, cu_q, cu_kv, pi, dist


def _placeholder_topk_like(topk_pages: jax.Array) -> jax.Array:
    """``[T, 1]`` all -1 token-topk placeholder in ``topk_pages``' placement.

    ``_run_sparse`` shard_maps the token topk with ``P(dpa, None)``; a fresh
    ``jnp.full`` inside the JIT is replicated on the explicit mesh and fails the
    in_specs check, so derive the placeholder from the data-sharded page topk.
    """
    return jnp.full_like(topk_pages[:, :1], -1, dtype=jnp.int32)


def _fixed_stride_pages(
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    page_size: int,
    pages_per_seq: int,
) -> jax.Array:
    """Repack the packed page table into the kernel's fixed-stride layout.

    sglang packs seq i's pages starting at ``cu_kv_lens[i] // page_size``
    (cu_kv_lens is page-aligned by construction); the streamindex kernel
    indexes ``page_indices[seq_id * pages_per_seq + page_id]``. The two
    coincide only when every sequence occupies exactly ``pages_per_seq``
    slots (e.g. single-seq decode), so gather-repack here. Tail entries
    beyond seq i's actual pages may alias a later sequence's pages — same
    as the reference's ``dynamic_slice`` over the packed table — which is
    harmless because both mask scores past ``kv_len``.
    """
    starts = cu_kv_lens[:-1] // page_size
    gidx = starts[:, None] + jnp.arange(pages_per_seq)[None, :]
    gidx = jnp.minimum(gidx, page_indices.shape[0] - 1)
    return page_indices[gidx].reshape(-1)


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

    # Padding rows must land on the reserved page: the allocator hands out
    # local pages 1..pages_per_rank and keeps page 0 (reads through it are
    # masked past kv_len), while the LAST page is allocatable — near-full
    # pools would otherwise get offset 0 of a live page's keys clobbered.
    sentinel = 0
    safe_page = jnp.where(valid, page, sentinel)
    safe_off = jnp.where(valid, offset, 0)
    return cache3d.at[safe_page, safe_off].set(new_tokens.astype(cache3d.dtype))
