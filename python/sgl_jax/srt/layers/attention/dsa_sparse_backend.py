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
    sparse_mla_attention_qblock,
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
# Merge DCP partial attention with reduce-scatter (``ag_rs``) instead of
# all-gathering every rank's all-head output and keeping 1/dcp_size of it. The
# gather path materializes dcp_size copies of ``[T, H_all, Dv]`` f32 — at
# 8k/DCP=16 that single collective was 71.9% of device self time. Set
# ``DSA_DCP_MERGE_SCATTER=0`` to fall back (both paths are pinned equal by
# ``test_dcp_merge_scatter``).
_DCP_MERGE_SCATTER = os.environ.get("DSA_DCP_MERGE_SCATTER", "1") == "1"
# Prefill: gather KV pages and attend local heads instead of gathering Q and merging.
_DCP_PREFILL_LOCAL_HEADS = os.environ.get("DSA_DCP_PREFILL_LOCAL_HEADS", "1") == "1"
_DCP_PREFILL_LOCAL_QB = int(os.environ.get("DSA_DCP_PREFILL_LOCAL_QB", "256"))
# Decode merge with one all-to-all instead of pmax + psum + psum_scatter.
_DCP_MERGE_A2A = os.environ.get("DSA_DCP_MERGE_A2A", "1") == "1"
# Send the ``ag_rs`` reduce-scatter in bf16 instead of f32, halving the bytes of
# the largest remaining DCP collective. Worth 9% of prefill wall time at 128k and
# 512k and 10.5% of device self time, with the ``reduce-scatter`` category exactly
# halved and nothing else in the profile moving. The feared loss -- the reduction
# accumulates dcp_size partials weighted by exp(lse_r - max_lse) <= 1, so the
# shards that round away are the ones contributing least -- does not show up
# against the bf16 the rest of the model already runs in: 8k tokens stay
# bit-identical to dcp=1, the 1M needle keeps 3/3 recall at 975k tokens, and
# max|dlogprob| vs dcp=1 is *lower* than the f32 path's. Set
# ``DSA_DCP_MERGE_BF16=0`` to go back to f32.
_DCP_MERGE_SCATTER_BF16 = os.environ.get("DSA_DCP_MERGE_BF16", "1") == "1"
_MERGE_SCATTER_DTYPE = jnp.bfloat16 if _DCP_MERGE_SCATTER_BF16 else None


def _dcp_prefill_local_heads(
    ql,
    qpe,
    kvc,
    kpe,
    cache,
    global_pages,
    positions,
    loc,
    seq_lens,
    cu_q_lens,
    cu_kv_lens,
    page_indices,
    *,
    kv_lora_rank,
    page_size,
    sm_scale,
    dcp_size,
    dcp_rank,
    query_block,
):
    """DCP prefill on this rank's heads over an all-gathered view of the batch's KV pages."""
    from sgl_jax.srt.layers.dcp.write import physical_write_loc_jax

    T, H, Dv = ql.shape
    rope = qpe.shape[-1]
    S = seq_lens.shape[0]
    row = jnp.zeros((T, cache.shape[-1]), cache.dtype)
    row = row.at[:, :Dv].set(kvc.astype(cache.dtype))
    row = row.at[:, Dv : Dv + rope].set(kpe.reshape(T, rope).astype(cache.dtype))
    write_loc = physical_write_loc_jax(loc, dcp_size, dcp_rank, page_size).astype(jnp.int32)
    cache = paged_write_back(
        cache, row, write_loc, page_size=page_size, r_cap=T // page_size + S + 34
    )
    # Global DSA page d is local page d // dcp_size on rank d % dcp_size.
    view = jax.lax.all_gather(cache[page_indices], "tensor", axis=1)
    view = view.reshape((-1,) + cache.shape[1:])
    t = jnp.arange(T, dtype=jnp.int32)
    q_seq_id = jnp.clip(jnp.searchsorted(cu_q_lens[1:], t, side="right"), 0, S - 1).astype(
        jnp.int32
    )
    q = jnp.concatenate([ql, qpe], axis=-1)
    out = sparse_mla_attention_qblock(
        q.reshape(1, T, H, q.shape[-1]),
        view,
        global_pages.reshape(1, T, -1),
        positions.reshape(1, T),
        kv_lora_rank=Dv,
        read_block=page_size,
        query_block=query_block,
        sm_scale=float(sm_scale),
        page_size=page_size,
        q_seq_id=q_seq_id,
        seq_lens=seq_lens,
        cu_kv_lens=cu_kv_lens * dcp_size,
        page_indices=jnp.arange(view.shape[0], dtype=jnp.int32),
    )
    return out.reshape(T, H, Dv), cache


def _squeeze_dcp_cache(cache_):
    """Drop the size-1 leading DCP shard so kernels see 4D ``[pages, ...]``."""
    if cache_.ndim == 5:
        return cache_[0], True
    return cache_, False


def _unsqueeze_dcp_cache(cache4d, had_dcp_axis: bool):
    return cache4d[None, ...] if had_dcp_axis else cache4d


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
            dcp_size=aux_data.get("dcp_size", 1),
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
                idx_cache, topk_pages, topk_tokens = self._maybe_index_prefill_pages(
                    is_full,
                    q_idx,
                    k_idx,
                    idx_weights,
                    idx_cache,
                    dpa,
                    md,
                    forward_batch.positions.astype(jnp.int32),
                )
                # Page selection for the sparse-prefill attend. A FULL (indexer)
                # layer just produced [T, k_pages] page-topk; a SHARED layer gets
                # None from `_maybe_index_prefill_pages` and reuses the preceding
                # full layer's pages threaded via `dsa_topk_pages_in` (IndexShare).
                # During prefill that thread is ALSO [T, k_pages] over the SAME T
                # query rows (the full layer above ran the same single-shot prefill)
                # — NOT the decode one-query-per-seq shape. None (a shared layer
                # before any full layer) ⇒ fall through to the dense attend below.
                # dcp>1 threads token-level owned∩global topk via `dsa_topk_in`.
                topk_pages_use = topk_pages if is_full else dsa_topk_pages_in
                topk_use = topk_tokens if is_full else dsa_topk_in
                if topk_pages_use is not None or (self.dcp_size > 1 and topk_use is not None):
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
                        topk=topk_use,
                    )
                    return o, DSAFusedCache(
                        kv=kv_cache,
                        idx=idx_cache,
                        topk=topk_use if is_full else None,
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

        # ── indexer top-k (full) or reuse (shared) ─────────────────────────
        idx_cache, topk, topk_pages = self._maybe_index(
            is_full, q_idx, k_idx, idx_weights, idx_cache, dpa, md, compute_pages=True
        )
        if not is_full:
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
            forward_batch,
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
    ):
        """On full layers: write k_idx into paged indexer cache, compute top-k
        (and, when ``compute_pages``, the unique-page list for IndexShare).

        ``compute_topk=False`` (prefill/mixed) skips ``streamindex_topk_ref``
        entirely — the topk is only consumed by ``_run_sparse`` on DECODE, so
        during chunked prefill we only need the k_idx cache write."""
        if not is_full or q_idx is None:
            return idx_cache, None, None

        cache_spec = self.paged_cache_spec(dpa)
        in_specs = (
            P(dpa, None, None),  # q_idx    [T, H_idx, D_idx] — replicated: softmax needs all heads
            P(dpa, None),  # k_idx    [T, D_idx]
            P(dpa, None),  # weights  [T, H_idx]
            cache_spec,  # idx_cache paged
            P(dpa),  # seq_lens
            P(dpa),  # page_indices
            P(dpa),  # cu_q_lens
            P(dpa),  # cu_kv_lens
            P(dpa),  # distribution
        )
        out_specs = (cache_spec, P(dpa, None), P(dpa, None))

        def _run(q_, k_, w_, cache_, seq_lens_, pi_, cuq_, cukv_, dist_):
            cache_, had_dcp = _squeeze_dcp_cache(cache_)
            page_size = cache_.shape[1] * cache_.shape[2]
            idx_dim = cache_.shape[3]
            pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
            cache3d = cache_.reshape(cache_.shape[0], page_size, idx_dim)
            dcp_rank = 0 if self.dcp_size <= 1 else jax.lax.axis_index("tensor")
            cache3d = _scatter_paged(
                cache3d,
                k_,
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
                pages_per_seq,
                dcp_size=self.dcp_size,
                dcp_rank=dcp_rank,
                dcp_interleave=page_size if self.dcp_size > 1 else 1,
            )
            if compute_topk and self.dcp_size > 1:
                from sgl_jax.srt.layers.dcp.indexer import (
                    page_topk_blocked_jax,
                    score_decode_local_jax,
                )

                dcp_rank = jax.lax.axis_index("tensor")
                local_scores = score_decode_local_jax(
                    q_,
                    w_,
                    cache3d,
                    seq_lens_,
                    pi_,
                    cukv_,
                    pages_per_seq,
                    page_size,
                    self.dcp_size,
                    dcp_rank,
                    interleave=page_size,
                )
                n_dsa_pages = max(pages_per_seq * self.dcp_size, 1)
                # Match the dcp=1 budget exactly: same page scores, same max-pool,
                # same top-k width => bit-identical selection, which is what makes
                # the greedy-match gate meaningful.
                k_pages = (
                    min(_PAGE_TOPK_BUDGET, n_dsa_pages)
                    if _PAGE_TOPK_BUDGET > 0
                    else (self.index_topk + page_size - 1) // page_size
                )
                # Block-interleaved decode selects at page granularity too: the
                # attend reads whole physical pages, so a token-level topk is
                # never materialized (same as the _PAGE_TOPK_BUDGET path).
                _global_pages, topk_pages = page_topk_blocked_jax(
                    local_scores,
                    self.dcp_size,
                    dcp_rank,
                    page_size,
                    k_pages,
                    seq_lens_ - 1,
                )
                topk = jnp.full((q_.shape[0], 1), -1, jnp.int32)
                return (
                    _unsqueeze_dcp_cache(cache3d.reshape(cache_.shape), had_dcp),
                    topk,
                    topk_pages,
                )
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
                return (
                    _unsqueeze_dcp_cache(cache3d.reshape(cache_.shape), had_dcp),
                    topk,
                    topk_pages,
                )
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
            return _unsqueeze_dcp_cache(cache3d.reshape(cache_.shape), had_dcp), topk, topk_pages

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

    def _run_sparse(
        self, ql, qpe, kvc, kpe, cache, topk, topk_pages, sm_scale, dpa, md, forward_batch
    ):
        has_pages = topk_pages is not None
        if not has_pages:
            topk_pages = jnp.full((topk.shape[0], 1), -1, jnp.int32)
        cache_spec = self.paged_cache_spec(dpa)
        dcp_size = self.dcp_size
        return_lse = dcp_size > 1
        loc = forward_batch.out_cache_loc.astype(jnp.int32)
        positions = forward_batch.positions.astype(jnp.int32)
        in_specs = (
            P(dpa, "tensor", None),
            P(dpa, "tensor", None),
            P(dpa, None),
            P(dpa, None),
            cache_spec,
            P(dpa, None),  # topk [T, k]
            P(dpa, None),  # topk_pages [T, k_pages_max]
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),  # loc
            P(dpa),  # positions
        )
        out_specs = (P(dpa, "tensor", None), cache_spec)

        def _run(
            ql_,
            qpe_,
            kvc_,
            kpe_,
            cache_,
            topk_,
            tpages_,
            seq_lens_,
            pi_,
            cuq_,
            cukv_,
            dist_,
            loc_,
            pos_,
        ):
            cache_, had_dcp = _squeeze_dcp_cache(cache_)
            page_size = cache_.shape[1] * cache_.shape[2]
            pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
            dcp_rank = 0 if dcp_size <= 1 else jax.lax.axis_index("tensor")
            h_local = ql_.shape[1]
            if dcp_size > 1:
                from sgl_jax.srt.layers.dcp.comm import (
                    a2a_merge_dcp_attention,
                    allgather_heads,
                    gather_merge_dcp_attention,
                    merge_scatter_dcp_attention,
                    slice_local_heads,
                )

                # Block-interleaved: ``tpages_`` is this rank's physical page ids,
                # so the same page-level kernel the dcp=1 path uses attends here.
                # Q is all-gathered because the LSE merge only composes the same
                # heads over different KV shards.
                ql_use, qpe_use = allgather_heads(ql_, qpe_)
                o, cache_new, lse = sparse_mla_page_level(
                    ql_use,
                    qpe_use,
                    kvc_,
                    kpe_,
                    cache_,
                    seq_lens_,
                    topk_,
                    pi_,
                    cuq_,
                    cukv_,
                    dist_,
                    tpages_,
                    sm_scale=float(sm_scale),
                    page_size=page_size,
                    pages_per_seq=pages_per_seq,
                    kv_lora_rank=self.kv_lora_rank,
                    # A rank owns at most `pages_per_seq` physical pages, and the
                    # indexer compacts its owned ids to the front of each row, so
                    # truncating there is lossless — and keeps the kernel's attend
                    # span at ~4 pages instead of the full global page budget.
                    k_pages_max=min(tpages_.shape[-1], pages_per_seq) + 1,
                    vmem_limit_bytes=self.vmem_limit_bytes,
                    return_lse=True,
                    dcp_size=dcp_size,
                    dcp_rank=dcp_rank,
                    dcp_interleave=page_size,
                )
                if _DCP_MERGE_A2A:
                    o = a2a_merge_dcp_attention(o, lse, h_local)
                elif _DCP_MERGE_SCATTER:
                    o = merge_scatter_dcp_attention(
                        o, lse, h_local, scatter_dtype=_MERGE_SCATTER_DTYPE
                    )
                else:
                    o = slice_local_heads(gather_merge_dcp_attention(o, lse), h_local)
                return o.astype(ql_.dtype), _unsqueeze_dcp_cache(cache_new, had_dcp)
            result = sparse_mla_page_level(
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
                return_lse=return_lse,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
            o, cache_out = result
            return o, _unsqueeze_dcp_cache(cache_out, had_dcp)

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
            loc,
            positions,
        )

    def _maybe_index_prefill_pages(
        self, is_full, q_idx, k_idx, idx_weights, idx_cache, dpa, md, positions
    ):
        """Prefill page-topk: scatter ``k_idx`` into the paged indexer cache and,
        on full layers, compute per-query **causal** page-level top-k via the
        general (non-decode, ``one_token_per_seq=False``) indexer path.

        Returns ``(idx_cache, topk_pages, topk_tokens)``. ``topk_pages`` is
        ``[T, k_pages]`` seq-local page ids (-1 padded). ``topk_tokens`` is
        virtual owned∩global top-k when ``dcp_size>1``, else ``None``.
        Both are ``None`` on shared layers / when no indexer is wired.
        """
        if not is_full or q_idx is None:
            return idx_cache, None, None
        k_pages = (self.index_topk + self.page_size - 1) // self.page_size
        cache_spec = self.paged_cache_spec(dpa)
        in_specs = (
            P(dpa, None, None),  # q_idx    [T, H_idx, D_idx] replicated
            P(dpa, None),  # k_idx    [T, D_idx]
            P(dpa, None),  # weights  [T, H_idx]
            cache_spec,  # idx_cache paged
            P(dpa),  # seq_lens
            P(dpa),  # page_indices
            P(dpa),  # cu_q_lens
            P(dpa),  # cu_kv_lens
            P(dpa),  # distribution
            P(dpa),  # positions
        )
        out_specs = (cache_spec, P(dpa, None), P(dpa, None))

        def _run(q_, k_, w_, cache_, seq_lens_, pi_, cuq_, cukv_, dist_, pos_):
            cache_, had_dcp = _squeeze_dcp_cache(cache_)
            page_size = cache_.shape[1] * cache_.shape[2]
            idx_dim = cache_.shape[3]
            pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
            k_eff = min(k_pages, pages_per_seq)
            cache3d = cache_.reshape(cache_.shape[0], page_size, idx_dim)
            dcp_rank = 0 if self.dcp_size <= 1 else jax.lax.axis_index("tensor")
            cache3d = _scatter_paged(
                cache3d,
                k_,
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
                pages_per_seq,
                dcp_size=self.dcp_size,
                dcp_rank=dcp_rank,
                dcp_interleave=page_size if self.dcp_size > 1 else 1,
            )
            if self.dcp_size > 1:
                from sgl_jax.srt.layers.dcp.indexer import (
                    page_topk_blocked_jax,
                    page_topk_from_gathered,
                    score_prefill_local_jax,
                )

                if _INDEXER_KERNEL_PREFILL:
                    # Streaming page maxima: never materializes [T, local_kv],
                    # which is 2.1 GiB/layer at 1M and the long-context OOM.
                    local_max = streamindex_page_topk(
                        q_,
                        w_,
                        cache3d.reshape(cache_.shape),
                        seq_lens_,
                        _fixed_stride_pages(pi_, cukv_, page_size, pages_per_seq),
                        cuq_,
                        dist_[2],
                        k_pages=k_eff,
                        dcp_size=self.dcp_size,
                        dcp_rank=dcp_rank,
                        dcp_interleave=page_size,
                        return_page_scores=True,
                    )
                    global_pages, phys_pages = page_topk_from_gathered(
                        jax.lax.all_gather(local_max, "tensor", axis=0),
                        self.dcp_size,
                        dcp_rank,
                        page_size,
                        k_pages,
                        pos_,
                    )
                else:
                    local_scores = score_prefill_local_jax(
                        q_,
                        w_,
                        cache3d,
                        seq_lens_,
                        pi_,
                        cuq_,
                        cukv_,
                        pos_,
                        pages_per_seq,
                        page_size,
                        self.dcp_size,
                        dcp_rank,
                        interleave=page_size,
                    )
                    # Block-interleaved: rank r's physical page P *is* global page
                    # P*dcp+r, so the global top-k needs no cross-rank max and each
                    # selected page is one whole physical page on exactly one rank.
                    global_pages, phys_pages = page_topk_blocked_jax(
                        local_scores,
                        self.dcp_size,
                        dcp_rank,
                        page_size,
                        k_pages,
                        pos_,
                    )
                return (
                    _unsqueeze_dcp_cache(cache3d.reshape(cache_.shape), had_dcp),
                    phys_pages,
                    global_pages,
                )
            dummy_topk = jnp.full((q_.shape[0], 1), -1, jnp.int32)
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
                    k_pages=k_eff,
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
                    k_pages=k_eff,
                    pages_per_seq=pages_per_seq,
                    one_token_per_seq=False,  # prefill: T>1 tokens/seq, per-query causal
                )
            return (
                _unsqueeze_dcp_cache(cache3d.reshape(cache_.shape), had_dcp),
                topk_pages,
                dummy_topk,
            )

        idx_cache, topk_pages, topk_tokens = jax.shard_map(
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
            positions,
        )
        if self.dcp_size <= 1:
            topk_tokens = None
        return idx_cache, topk_pages, topk_tokens

    def _run_sparse_prefill(
        self,
        ql,
        qpe,
        kvc,
        kpe,
        cache,
        topk_pages,
        sm_scale,
        dpa,
        md,
        forward_batch,
        topk=None,
    ):
        """Fused sparse-MLA prefill: self-write the current chunk's latent into the
        paged fused cache, then attend only the page-topk pages.

        Packed-ragged: threads the same ragged metadata the dense/indexer paths use
        (``seq_lens``/``cu_q_lens``/``cu_kv_lens``/``page_indices``) so a batch of
        multiple requests (``max_running>1``) prefills in one call. With a single
        request this reduces to the previous single-shot behaviour. Returns
        ``(o_latent, updated_cache)``.
        """
        loc = forward_batch.out_cache_loc.astype(jnp.int32)
        positions = forward_batch.positions.astype(jnp.int32)
        page_size = self.page_size
        kv_lora_rank = self.kv_lora_rank
        sm = float(sm_scale)
        dcp_size = self.dcp_size
        cache_spec = self.paged_cache_spec(dpa)
        # Derive the -1 placeholders from `loc`, which is already sharded on `dpa`:
        # a bare jnp.full is replicated and does not match in_specs P(dpa, None).
        if topk_pages is None:
            topk_pages = jnp.full_like(loc[:, None], -1)
        if topk is None:
            topk = jnp.full_like(loc[:, None], -1)

        in_specs = (
            P(dpa, "tensor", None),  # ql   [T, H, kv_lora_rank]
            P(dpa, "tensor", None),  # qpe  [T, H, rope]
            P(dpa, None),  # kvc  [T, kv_lora_rank]
            P(dpa, None),  # kpe  [T, rope]
            cache_spec,  # cache (5D leading DCP axis when dcp_size>1)
            P(dpa, None),  # topk_pages [T, K]
            P(dpa, None),  # topk tokens [T, k]
            P(dpa),  # positions [T]
            P(dpa),  # loc [T]
            P(dpa),  # seq_lens [S]
            P(dpa),  # cu_q_lens [S+1]
            P(dpa),  # cu_kv_lens [S+1]
            P(dpa),  # page_indices [total_pages]
        )
        out_specs = (P(dpa, "tensor", None), cache_spec)

        def _run(ql_, qpe_, kvc_, kpe_, cache_, tp_, topk_, pos_, loc_, sl_, cuq_, cukv_, pi_):
            cache_, had_dcp = _squeeze_dcp_cache(cache_)
            dcp_rank = 0 if dcp_size <= 1 else jax.lax.axis_index("tensor")
            kwargs = dict(
                kv_lora_rank=kv_lora_rank,
                page_size=page_size,
                sm_scale=sm,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
            if dcp_size > 1:
                from sgl_jax.srt.layers.dcp.comm import (
                    allgather_heads,
                    gather_merge_dcp_attention,
                    merge_scatter_dcp_attention,
                    slice_local_heads,
                )
                from sgl_jax.srt.layers.dcp.write import (
                    owned_len_jax,
                    physical_positions_jax,
                )

                # Block-interleaved KV: ``tp_`` is already this rank's physical
                # page ids, so the SAME kernel the dcp=1 path uses reads whole
                # pages here — the only DCP-specific work is (a) all-gathering Q
                # so every rank runs all heads (the LSE merge is only valid for
                # the same heads over different KV shards), (b) mapping the causal
                # bounds into physical slot space, and (c) the merge.
                if _DCP_PREFILL_LOCAL_HEADS:
                    o, cache_new = _dcp_prefill_local_heads(
                        ql_,
                        qpe_,
                        kvc_,
                        kpe_,
                        cache_,
                        topk_,
                        pos_,
                        loc_,
                        sl_,
                        cuq_,
                        cukv_,
                        pi_,
                        query_block=_DCP_PREFILL_LOCAL_QB,
                        **kwargs,
                    )
                    return o.astype(ql_.dtype), _unsqueeze_dcp_cache(cache_new, had_dcp)
                h_local = ql_.shape[1]
                ql_use, qpe_use = allgather_heads(ql_, qpe_)
                o, cache_new, lse = prefill_write_and_attend_ragged_qblock(
                    ql_use,
                    qpe_use,
                    kvc_,
                    kpe_,
                    cache_,
                    tp_,
                    physical_positions_jax(pos_, dcp_size, dcp_rank, page_size),
                    loc_,
                    owned_len_jax(sl_, dcp_size, dcp_rank, page_size),
                    cuq_,
                    cukv_,
                    pi_,
                    query_block=_PREFILL_QBLOCK_QB,
                    dcp_interleave=page_size,
                    return_lse=True,
                    **kwargs,
                )
                if _DCP_MERGE_SCATTER:
                    o = merge_scatter_dcp_attention(
                        o, lse, h_local, scatter_dtype=_MERGE_SCATTER_DTYPE
                    )
                else:
                    o = slice_local_heads(gather_merge_dcp_attention(o, lse), h_local)
                return o.astype(ql_.dtype), _unsqueeze_dcp_cache(cache_new, had_dcp)
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
                    query_block=_PREFILL_QBLOCK_QB,
                    **kwargs,
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
                    **kwargs,
                )
            return o.astype(ql_.dtype), _unsqueeze_dcp_cache(cache_new, had_dcp)

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            ql,
            qpe,
            kvc,
            kpe,
            cache,
            topk_pages,
            topk,
            positions,
            loc,
            md.seq_lens,
            md.cu_q_lens,
            md.cu_kv_lens,
            md.page_indices,
        )

    def _run_dense(self, ql, qpe, kvc, kpe, cache, sm_scale, layer, dpa, md):
        cache_spec = self.paged_cache_spec(dpa)
        in_specs = (
            P(dpa, "tensor", None),
            P(dpa, "tensor", None),
            P(dpa, None),
            P(dpa, None),
            cache_spec,
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
            P(dpa),
        )
        out_specs = (P(dpa, "tensor", None), cache_spec)
        sw = layer.sliding_window_size if layer is not None else None
        sc = layer.logit_cap if layer is not None else None

        def _run(ql_, qpe_, kvc_, kpe_, cache_, seq_lens_, pi_, cuq_, cukv_, dist_):
            cache_, had_dcp = _squeeze_dcp_cache(cache_)
            o, cache_out = mla_ragged_paged_attention(
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
            return o, _unsqueeze_dcp_cache(cache_out, had_dcp)

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
    dcp_size: int = 1,
    dcp_rank: int | jax.Array = 0,
    dcp_interleave: int = 1,
) -> jax.Array:
    """Write new_tokens[t] into cache at (page, offset) for each seq's tail slots.

    Jit-compatible reference for the paged cache write that the Pallas kernel
    does via input_output_aliases. For seq i with q tokens cu_q_lens[i]..[i+1),
    token j lands at absolute position seq_lens[i] - (q_end - q_start) + j.

    Under DCP, ``abs_pos`` is the token's **virtual** position; ownership and the
    physical slot follow ``dcp/layout.py`` for the given ``dcp_interleave``.
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
    if dcp_size > 1:
        if dcp_interleave == 1:
            valid = valid & ((abs_pos % dcp_size) == dcp_rank)
            abs_pos = abs_pos // dcp_size
        else:
            i = dcp_interleave
            blk = abs_pos // i
            valid = valid & ((blk % dcp_size) == dcp_rank)
            abs_pos = (blk // dcp_size) * i + (abs_pos % i)

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
