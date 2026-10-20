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

from sgl_jax.srt.environ import envs
from sgl_jax.srt.kernels.dsa.ref import (
    streamindex_page_topk_ref,
    streamindex_page_topk_ref_batched,
    streamindex_page_topk_ref_grouped,
    streamindex_topk_ref,
)
from sgl_jax.srt.kernels.dsa.scatter_paged_inplace import scatter_rows_inplace
from sgl_jax.srt.kernels.dsa.sparse_mla import compute_topk_pages, sparse_mla_page_level
from sgl_jax.srt.kernels.dsa.sparse_mla_prefill import prefill_write_and_attend_ragged
from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import (
    paged_write_back,
    prefill_write_and_attend_ragged_qblock,
    prefill_write_and_attend_ragged_qblock_tsa,
)
from sgl_jax.srt.kernels.dsa.streamindex_topk import (
    streamindex_page_topk,
    streamindex_topk,
)
from sgl_jax.srt.kernels.mla.v2.kernel import mla_ragged_paged_attention
from sgl_jax.srt.layers.attention.mla_backend import (
    MLAAttentionBackend,
    MLAAttentionMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.profiling_utils import named_scope

_MLA_Q_HEAD_MAJOR = envs.SGLANG_JAX_MLA_Q_HEAD_MAJOR.get()
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
# Page top-k identity shortcut: when the page budget covers the whole KV bucket
# (k_pages >= pages_per_seq, static per decode bucket: budget 128 pages = 16k tokens), the
# page top-k selects every valid page anyway, so skip the gather + scoring + top_k and return
# the identity page list (valid pages in order, -1 padded). Same page SET as the reference;
# the attention kernel visits the pages in position order instead of score order (summation
# order change only -> paired accuracy gate, not bitwise). cc64 1k/1k census 10-04: the
# reference path is 4.9 ms/step (ref.py:447 scoring 2.8 + ref.py:433 page gather 1.75).
# DSA_PAGE_TOPK_IDENTITY_MAX_PAGES=<n> (default 32): only take the identity shortcut when the bucket
# has <= n pages (10-04 gp380: at the 32-page bucket (4096/128 cc8) v22 lost 15 % throughput while
# 16-page buckets gained; the attend kernel's x1.7 under identity grows with pages). 0 = no cap,
# -1 = never.
_PAGE_TOPK_IDENTITY_MAX_PAGES = envs.DSA_PAGE_TOPK_IDENTITY_MAX_PAGES.get()


def _identity_allowed(pages_per_seq: int) -> bool:
    if _PAGE_TOPK_IDENTITY_MAX_PAGES < 0:
        return False
    return _PAGE_TOPK_IDENTITY_MAX_PAGES == 0 or pages_per_seq <= _PAGE_TOPK_IDENTITY_MAX_PAGES


def _identity_page_topk_grouped(
    seq_lens,
    cu_q_lens,
    num_tokens: int,
    k_pages: int,
    pages_per_seq: int,
    page_size: int,
    q_group: int,
):
    """Grouped (spec verify) counterpart of :func:`_identity_page_topk`: request ``s`` has ``G``
    query tokens at absolute positions ``seq_len - G + g``; query ``g`` sees pages
    ``0 .. ceil((seq_len - G + g + 1) / page_size) - 1`` (the set ``streamindex_page_topk_ref_grouped``
    selects when ``k_pages >= pages_per_seq``). Output assembly mirrors the reference
    (rows ``cu_q[s] + g``, padded requests -> -1, sentinel row ``T``)."""
    S = seq_lens.shape[0]
    G = q_group
    T = num_tokens
    q_len = cu_q_lens[1 : S + 1] - cu_q_lens[:S]
    valid = (seq_lens > 0) & (q_len == G)  # [S]
    rows = jnp.minimum(cu_q_lens[:S, None] + jnp.arange(G)[None, :], T - 1)  # [S, G]
    abs_q = seq_lens[:, None] - G + jnp.arange(G)[None, :]  # [S, G]
    nvalid = (abs_q + page_size) // page_size  # ceil((abs_q + 1) / page_size)
    pos = jnp.arange(k_pages, dtype=jnp.int32)[None, None, :]
    ok = valid[:, None, None] & (pos < nvalid[:, :, None]) & (pos < pages_per_seq)
    pidx = jnp.where(ok, pos, -1).astype(jnp.int32).reshape(S * G, k_pages)
    row_valid = jnp.broadcast_to(valid[:, None], (S, G)).reshape(-1)
    dest = jnp.where(row_valid, rows.reshape(-1), T)
    out = jnp.full((T + 1, k_pages), -1, dtype=jnp.int32)
    return out.at[dest].set(pidx)[:T]


def _identity_page_topk(
    seq_lens, num_tokens: int, k_pages: int, pages_per_seq: int, page_size: int
):
    """``i32[num_tokens, k_pages]``: local page positions ``0..nvalid-1`` of token i's sequence
    (token i belongs to seq i), ``-1`` beyond ``nvalid = ceil(seq_len / page_size)``."""
    n = min(num_tokens, seq_lens.shape[0])
    nvalid = (seq_lens[:n] + page_size - 1) // page_size  # [n]
    pos = jnp.arange(k_pages, dtype=jnp.int32)[None, :]  # k_pages >= pages_per_seq
    valid = (pos < nvalid[:, None]) & (pos < pages_per_seq)
    pidx = jnp.where(valid, pos, -1).astype(jnp.int32)
    out = jnp.full((num_tokens, k_pages), -1, dtype=jnp.int32)
    return out.at[:n].set(pidx)


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
# Opt-in: token-shard the prefill indexer scoring across the "tensor" axis.
# Baseline scores replicated: every rank computes all T query rows (q_idx is
# P(dpa, None, None)), 16x redundant under tp16. Each rank instead scores its
# contiguous T/tp row slice against the (replicated) k cache with per-rank
# shifted seq metadata — same causal q_pos = seq_len - q_len + row_offset
# convention as the kernel/ref — and the per-rank page-topk is all-gathered
# back. Zero extra communication beyond the [T, k_pages] i32 gather. Falls
# back to the replicated path when T % tp != 0.
# Decode page-topk vectorized over sequences (one big idx-cache gather + batched
# head-chunked scoring with static head-chunk unroll + batched top_k) instead of
# the per-seq fori_loop reference, from this decode batch size up: at bs1 the
# head-chunked path costs more than the single per-seq einsum (nap dpb9: bs1
# TPOT 12.07 -> 12.21), while at T=64 it takes the indexer path 21 -> ~8 ms/step
# (09-21 trace: while 12.65 + 64x 1 MiB gathers 5.14 + dynamic_slice 3.35 ms).
_IDX_DECODE_BATCHED_MIN_BS = 8
# a2a-DP attend (DSA_DP_DECODE_BS, default 64): decode buckets with bs >= this threshold reshard
# queries from head-TP ([T, H/tp]) to request-sharded ([T/tp, H]) via
# all_to_all inside the shard_map, attend only the local request slice, and
# a2a back — the decode twin of the prefill TSA sandwich. Per-seq attend is
# embarrassingly parallel over requests, so this cuts the batched-decode
# attend ~T/tp-fold (cc32: 15.8 -> ~4 ms/step) at the cost of two tiny a2a
# ([T,4,576] ~147KB). Cache-write replica consistency: every rank pre-writes
# ALL T new kv rows (jnp scatter, ~64 rows) before the kernel, because the
# kernel's in-place write only covers the local slice and request->rank
# assignment is not stable across steps. bs < threshold keeps head-TP
# (two a2a are pure latency at bs1). 0 = off. The pre-write is the 4D-native
# scatter_rows_inplace kernel (a .at[].set on the 3D view materializes a
# full-cache gather+scatter plus the 4D<->3D relayout: nap e2e cc64 TPOT 351 ms,
# 09-21).
_DP_DECODE_BS = envs.DSA_DP_DECODE_BS.get()
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
# DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT=<L> (default 2048, clamped to index_topk): EXTEND batches
# whose longest sequence is <= L run plain dense attention; the page top-k would select
# every page anyway, so this is the same function without the indexer top-k and the mask
# (the indexer keys are still written for later decode steps).
_PREFILL_DENSE_EXACT_MAX = envs.DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT.get()
_LOGGED_ONCE: set[str] = set()


def _log_once(msg: str) -> None:
    if msg not in _LOGGED_ONCE:
        _LOGGED_ONCE.add(msg)
        logger.info(msg)


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
# TSA (token-sharded attention) prefill, opt-in via DSA_TSA_PREFILL=1: inside
# the sparse-prefill shard_map, an all_to_all reshards queries from head-TP
# ([T, H/tp] per device) to token-sharded ([T/tp, H] per device), each device
# attends only its contiguous token slice's page union, and a second
# all_to_all reshards the output back. Head-TP makes QB queries span 4-head
# fragments of 64 unrelated positions, so a 256-row MXU group unions ~177
# pages; 16 ADJACENT tokens x all 64 heads union ~66 pages at the same M.
# QB=16 from the P0 shape sweep on real 110k indices: QB=4 x1.90 / QB=8
# x2.32 / QB=16 x2.72 vs head-TP at the same depth (smaller groups have
# denser unions but pay per-block launch/DMA overhead). The two all_to_alls
# move ~71MB/device/layer/chunk — well under the attend time they unlock.
_TSA_PREFILL = os.environ.get("DSA_TSA_PREFILL", "0") == "1"
# Opt-in: prefill indexer-key cache write through the 4D-native run-table
# write-back kernel (the one the KV write uses) instead of reshape-to-3D +
# XLA scatter + reshape-back. gp101 4k trace (10-02): the 3D round trip costs a
# full relayout of the [P, page, D] indexer cache per full layer, 12.2 ms per
# 4k chunk (scatter_fusion bf16[267648,128] x 45), plus the kernel path
# reshapes the 3D view back to 4D. Bitwise-identical writes (paged_write_back
# contract + test_idx_prefill_inplace.py). Indexer-cache writes of <= SCATTER_ROWS_MAX rows (decode /
# verify steps) take the 4D-native XLA scatter instead of the run-table kernel (bit-identical). The
# exact-dense arm writes its keys per page (``_dense_arm_idx_write_pages``: whole-page gather /
# overlay / scatter, no 3D view -- gp794e 8k two-window trace 2026-10-10: the 3D-view scatter was a
# full relayout of the pool, 21.5 ms per 395 ms module).
_TSA_PREFILL_QB = int(os.environ.get("DSA_TSA_PREFILL_QB", "16"))
# Block-strided (zigzag) query assignment inside TSA. The
# contiguous split hands device k the k-th T/tp token slice, so under a causal
# mask device 0 attends the shortest prefixes and device tp-1 the longest
# (measured v7x tp8: 16 vs 125 ms per 4k chunk, 49 vs 349 ms on the first 8k
# chunk) and the o-side all_to_all absorbs the difference as wait. Zigzag gives
# device k query blocks k, k+tp, k+2tp, ... (each block still QB consecutive
# tokens, so every block's union / loop order -- and its output -- is
# unchanged); only the two local token permutations around the a2a are added.


# TSA absorb-after-a2a: run the W_UK absorb and the W_UV
# projection on the token-sharded side of the TSA all_to_all instead of the
# head-sharded side, so the q-side a2a carries q_nope (192 dims) instead of
# the absorbed latent query (512) and the o-side a2a carries the W_UV output
# (256) instead of the latent output (512): -55% a2a bytes with no extra
# pass. Same contraction per element on another chip -> bit identical. Needs
# replicated full-head W_UK / W_UV copies (model params `w_uk_all` /
# `w_uv_all`, ~29 MiB bf16 per layer per device).
# Head-major (with the absorb path): run the attend
# kernel with head-major q/o so the W_UK dot output ([H, T/tp, R], batch-major)
# feeds the kernel and the kernel output feeds the W_UV dot without the two
# tile-level relayouts XLA otherwise inserts around the custom call (gp246 4k
# trace: 7.95 ms q-side copy + 1.56 ms convert, 4.55 ms o-side). The kernel
# rows are a permutation of the token-major order (bit-identical, see
# test_qblock_kernel_parity::test_head_major_*). The return a2a then carries
# [H, T/tp, Dv] (split heads, concat tokens) and the one remaining relayout is
# the [H/tp, T, Dv] -> token-major transpose for o_proj.


def _absorb_dot(lhs, rhs, dims, out_dtype):
    """Batched absorb dot (W_UK / W_UV): f32 accumulate + explicit ``astype`` (XLA fuses the
    cast into the following transpose)."""
    return jax.lax.dot_general(lhs, rhs, dims, preferred_element_type=jnp.float32).astype(out_dtype)


def tsa_zigzag_permute(x, tp_size: int, qb: int):
    """[T, ...] token order -> block-strided order so that a contiguous
    ``all_to_all(split_axis=0)`` hands device k the query blocks k, k+tp, k+2tp, ...
    (block = ``qb`` consecutive tokens). Requires ``T % (tp_size * qb) == 0``."""
    T = x.shape[0]
    x = x.reshape(T // (tp_size * qb), tp_size, qb, *x.shape[1:])
    return jnp.swapaxes(x, 0, 1).reshape(T, *x.shape[3:])


def tsa_zigzag_unpermute(x, tp_size: int, qb: int):
    """Inverse of :func:`tsa_zigzag_permute` on the rank-major a2a output [T, ...]."""
    T = x.shape[0]
    x = x.reshape(tp_size, T // (tp_size * qb), qb, *x.shape[1:])
    return jnp.swapaxes(x, 0, 1).reshape(T, *x.shape[3:])


def tsa_zigzag_token_index(t_local: int, tp_size: int, qb: int, rank):
    """Global chunk token index of local row g on device ``rank`` under the zigzag
    assignment: ((g // qb) * tp + rank) * qb + g % qb."""
    g = jnp.arange(t_local, dtype=jnp.int32)
    return ((g // qb) * tp_size + rank) * qb + (g % qb)


# Opt-in (W7): run speculative TARGET_VERIFY / DRAFT_EXTEND batches through the
# DECODE machinery instead of the sparse-prefill kernels. Every spec token
# becomes its own one-query "sequence" (kv_len = its own position + 1), the new
# KV rows are written up front with paged_write_back, then the decode indexer
# (one_token_per_seq) and sparse_mla_page_level run exactly as for decode. The
# extend-shaped prefill path costs ~10 ms/step at T=4 (bq_512 indexer blocks +
# qblock machinery); this trades it for n decode queries. Default OFF.
_SPEC_AS_DECODE = os.environ.get("DSA_SPEC_AS_DECODE", "0") == "1"
# Query block for the page-topk indexer kernel on spec batches (T = a few tokens
# per request; the prefill default of 512 pads 4 queries to a 512 block).
_SPEC_INDEXER_QB = int(os.environ.get("DSA_SPEC_INDEXER_QB", "16"))
# A/B knob (opt-in): decode-form verify tokens of one request share the last token's page list
# (see sparse_mla_page_level page_share_group). Default 1 = per-token pages.
_SPEC_VERIFY_PAGE_SHARE = int(os.environ.get("DSA_SPEC_VERIFY_PAGE_SHARE", "1") or "1")
# Opt-in: decode-form verify runs each request's draft tokens as ONE ragged
# sequence (G query rows over the shared page list) instead of G one-query
# pseudo-sequences -- the request's KV is read once (see sparse_mla_page_level
# group_queries). Implies the page-share page set. Default OFF.
_SPEC_VERIFY_QGROUP = os.environ.get("DSA_SPEC_VERIFY_QGROUP", "0") == "1"
# Grouped verify only pays off when many requests share a step: at bs1 the one
# grouped sequence on the mixed kernel matches the four one-query pseudo
# sequences on the batched decode kernel (12.4 vs 12.9 ms attention) while the
# step lost ~0.8 ms elsewhere, so keep the per-token path below this batch.
_SPEC_VERIFY_QGROUP_MIN_BS = int(os.environ.get("DSA_SPEC_VERIFY_QGROUP_MIN_BS", "8"))
# Grouped (G queries per request) verify attention reads the last q_len positions of each
# sequence from the new-KV buffer and writes them back to their true slots itself (mla/v2
# update path), so the backend pre-write of the same rows is skipped there.


def _use_verify_qgroup(is_verify: bool, draft_group: int | None, num_tokens: int) -> bool:
    """Grouped (G queries per request) verify attention: opt-in, decode-form
    verify batches only, and at least ``_SPEC_VERIFY_QGROUP_MIN_BS`` requests."""
    if not (is_verify and _SPEC_VERIFY_QGROUP and draft_group):
        return False
    return num_tokens // int(draft_group) >= _SPEC_VERIFY_QGROUP_MIN_BS


# Opt-in: on decode-form verify batches (S requests x G draft tokens) run the
# indexer page-topk as ONE batched XLA op over all requests
# (``streamindex_page_topk_ref_grouped``) instead of the per-sequence kernel
# grid, whose per-seq launch/DMA fixed cost dominates at cc64 (14.5 ms/step,
# 64 grid steps x ~21 full layers). Gated by a min batch (below it the kernel
# wins) and a max kv span (the batched gather materialises
# [S, pages_per_seq*page_size, D] keys; the kernel only DMAs up to kv_len).
# Default OFF.
_IDX_VERIFY_BATCHED = os.environ.get("DSA_IDX_VERIFY_BATCHED", "0") == "1"
_IDX_VERIFY_BATCHED_MIN_BS = int(os.environ.get("DSA_IDX_VERIFY_BATCHED_MIN_BS", "8"))
_IDX_VERIFY_BATCHED_MAX_KV = int(os.environ.get("DSA_IDX_VERIFY_BATCHED_MAX_KV", "8192"))


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
        # exact-dense arm bound: never above index_topk (beyond it the page top-k would drop pages
        # and dense attention would no longer be the same function); the host-side flag mirrors
        # this via ``dsa_dense_arm_flags(index_topk=...)``
        self.dense_exact_max_context = min(_PREFILL_DENSE_EXACT_MAX, int(index_topk))
        if self.dense_exact_max_context != _PREFILL_DENSE_EXACT_MAX:
            logger.warning(
                "DSA_PREFILL_DENSE_EXACT_MAX_CONTEXT=%d > index_topk=%d: exact-dense bound clamped to %d",
                _PREFILL_DENSE_EXACT_MAX,
                int(index_topk),
                self.dense_exact_max_context,
            )
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
        # (q_nope [T, H/tp, qk_nope], w_uk_all [R, H, qk_nope], w_uv_all [R, H, v]) for
        # the TSA absorb-after-a2a path; ignored on every other path.
        dsa_absorb = kwargs.get("dsa_absorb")

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
        # pure prefill only: is_extend() is also true for TARGET_VERIFY / DRAFT_EXTEND
        # (4- and 3-token sequences), which must keep the token-major kernel path
        q_head_major = _MLA_Q_HEAD_MAJOR and forward_batch.forward_mode == ForwardMode.EXTEND
        if is_full and q_idx is None:
            o, kv_cache = self._run_dense(
                q,
                q_rope,
                new_kv_c,
                new_k_pe,
                kv_cache,
                sm_scale,
                layer,
                dpa,
                md,
                q_head_major=q_head_major,
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
                forward_batch,
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
            if (
                _PREFILL_SPARSE
                and self.dense_exact_max_context > 0
                and forward_batch.forward_mode.is_extend()
                and getattr(forward_batch, "dsa_dense_exact", False)
            ):
                # Exact dense arm (context <= index_topk): write the indexer keys, skip the
                # page top-k, plain causal dense attention; latent output (caller applies W_UV).
                _log_once(
                    f"DSA prefill: exact-dense arm taken (static, max_context<="
                    f"{self.dense_exact_max_context}, tokens={q.shape[0]})"
                )
                # same call as the no-indexer dense short-circuit: write the indexer keys,
                # no scoring / top-k (``_maybe_index_prefill_pages(compute_pages=False)``
                # still ran the scoring kernel)
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
                o, kv_cache = self._run_dense(
                    q,
                    q_rope,
                    new_kv_c,
                    new_k_pe,
                    kv_cache,
                    sm_scale,
                    layer,
                    dpa,
                    dense_arm_metadata(md),
                    q_head_major=q_head_major,
                )
                return o, DSAFusedCache(kv=kv_cache, idx=idx_cache, topk=None, topk_pages=None)
            if _PREFILL_SPARSE:
                reuse_pages = is_full and dsa_topk_reuse and dsa_topk_pages_in is not None
                if reuse_pages and bool(getattr(forward_batch, "spec_kvshare_readonly", False)):
                    topk_pages = None  # KVShare: no indexer-key write for draft steps >= 1
                else:
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
                    # TSA needs the (padded) chunk length divisible by tp so the
                    # all_to_all tiles evenly; shapes are static at trace time,
                    # odd buckets just keep the head-TP layout.
                    tp_size = self.mesh.shape["tensor"]
                    use_tsa = _TSA_PREFILL and _tsa_eligible(
                        q.shape[0], self.mesh.shape[dpa], tp_size
                    )
                    run_prefill = (
                        self._run_sparse_prefill_tsa if use_tsa else self._run_sparse_prefill
                    )
                    absorb_kw = (
                        {"absorb": dsa_absorb} if (use_tsa and dsa_absorb is not None) else {}
                    )

                    o, kv_cache = run_prefill(
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
                        **absorb_kw,
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
            if not compute_topk:
                # key write only (exact-dense / dense-fallback prefill): per-page whole-page
                # gather / overlay / scatter on the 4D pool, no 3D view (see the note above)
                _log_once(
                    "DSA indexer key write (dense arm): per-page dynamic_update_slice"
                    + f" (rows={k_.shape[0]})"
                )
                cache4d = _dense_arm_idx_write_pages(
                    cache_, k_, page_size, seq_lens_, pi_, cuq_, cukv_, pages_per_seq
                )
                topk = jnp.full((q_.shape[0], 1), -1, jnp.int32)
                if compute_pages:
                    topk_pages = compute_topk_pages(
                        topk,
                        page_size=self.page_size,
                        pages_per_seq=pages_per_seq,
                        # sparse_mla_page_level keeps 511 hit pages + the new-token page (#1785)
                        k_pages_max=511,
                    )
                else:
                    topk_pages = jnp.full((topk.shape[0], 1), -1, jnp.int32)
                return cache4d, topk, topk_pages
            cache3d = cache_.reshape(cache_.shape[0], page_size, idx_dim)
            cache3d = _scatter_paged(cache3d, k_, seq_lens_, pi_, cuq_, cukv_, pages_per_seq)
            if compute_topk and _PAGE_TOPK_BUDGET > 0:
                # Page-scoring path: budget pages picked directly by max-pooled
                # page score; token-level topk is not materialized (sparse MLA
                # attends whole pages via topk_pages, so it never reads it).
                topk = jnp.full((q_.shape[0], 1), -1, jnp.int32)
                k_pages_eff = min(_PAGE_TOPK_BUDGET, pages_per_seq)
                if _identity_allowed(pages_per_seq) and k_pages_eff >= pages_per_seq:
                    # budget covers the bucket: every valid page is selected -> no scoring
                    topk_pages = _identity_page_topk(
                        seq_lens_, q_.shape[0], k_pages_eff, pages_per_seq, page_size
                    )
                elif q_.shape[0] >= _IDX_DECODE_BATCHED_MIN_BS:
                    topk_pages = streamindex_page_topk_ref_batched(
                        q_,
                        w_,
                        cache3d,
                        seq_lens_,
                        pi_,
                        cukv_,
                        k_pages=min(_PAGE_TOPK_BUDGET, pages_per_seq),
                        pages_per_seq=pages_per_seq,
                        unroll_heads=True,
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
                    # sparse_mla_page_level keeps 511 hit pages + the new-token page
                    k_pages_max=511,
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

    def _run_sparse(
        self,
        ql,
        qpe,
        kvc,
        kpe,
        cache,
        topk,
        topk_pages,
        sm_scale,
        dpa,
        md,
        page_share_group: int = 1,
        group_queries: bool = False,
    ):
        has_pages = topk_pages is not None
        if not has_pages:
            topk_pages = jnp.full((topk.shape[0], 1), -1, jnp.int32)
        tp_size = self.mesh.shape["tensor"]
        g_size = tp_size  # a2a group: the whole tensor axis
        dp_decode = 0 < _DP_DECODE_BS <= ql.shape[0] and tp_size > 1 and ql.shape[0] % g_size == 0
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
            k_pages_max = (
                min(_PAGE_TOPK_BUDGET, pages_per_seq) + 1 if _PAGE_TOPK_BUDGET > 0 else 512
            )
            if has_pages:
                # Page lists from the prefill-form indexer (spec decode-form path)
                # are index_topk/page_size wide; the page-level wrapper expects
                # k_pages_max - 1 columns.
                tpages_ = _pad_topk_pages(tpages_, k_pages_max - 1)
            if dp_decode:
                t_local = ql_.shape[0] // g_size
                off = (jax.lax.axis_index("tensor") % g_size) * t_local
                # Replica-consistent cache write of ALL T new rows up front:
                # the kernel below only in-place-writes this rank's slice and
                # request->rank assignment is not stable across steps.
                kv_dim = cache_.shape[3]
                new_full = jnp.concatenate([kvc_, kpe_], axis=-1)
                new_full = jnp.pad(new_full, ((0, 0), (0, kv_dim - new_full.shape[1])))
                # 4D-native in-place scatter: the cache must never be
                # reshaped here — every 3D/2D view is a full relayout copy.
                cache_ = _scatter_paged_inplace(
                    cache_, new_full, seq_lens_, pi_, cuq_, cukv_, pages_per_seq
                )
                # head-TP -> request-sharded (rank r owns rows [off, off+Tl))
                ql_ = jax.lax.all_to_all(ql_, "tensor", split_axis=0, concat_axis=1, tiled=True)
                qpe_ = jax.lax.all_to_all(qpe_, "tensor", split_axis=0, concat_axis=1, tiled=True)
                kvc_ = jax.lax.dynamic_slice_in_dim(kvc_, off, t_local, 0)
                kpe_ = jax.lax.dynamic_slice_in_dim(kpe_, off, t_local, 0)
                seq_lens_ = jax.lax.dynamic_slice_in_dim(seq_lens_, off, t_local, 0)
                topk_ = jax.lax.dynamic_slice_in_dim(topk_, off, t_local, 0)
                tpages_ = jax.lax.dynamic_slice_in_dim(tpages_, off, t_local, 0)
                cukv_ = jax.lax.dynamic_slice_in_dim(cukv_, off, t_local + 1, 0)
                cuq_ = jnp.arange(t_local + 1, dtype=jnp.int32)
                dist_ = jnp.full((3,), t_local, jnp.int32)
            o_, cache_out = sparse_mla_page_level(
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
                k_pages_max=k_pages_max,
                vmem_limit_bytes=self.vmem_limit_bytes,
                page_share_group=page_share_group,
                group_queries=group_queries,
            )
            if dp_decode:
                # request-sharded -> head-TP: [T/tp, H, Dv] -> [T, H/tp, Dv]
                o_ = jax.lax.all_to_all(
                    o_.astype(ql_.dtype), "tensor", split_axis=1, concat_axis=0, tiled=True
                )
            return o_, cache_out

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
        self,
        is_full,
        q_idx,
        k_idx,
        idx_weights,
        idx_cache,
        dpa,
        md,
        *,
        compute_pages=True,
        num_queries_per_block=None,
        q_group=None,
    ):
        """Prefill page-topk: scatter ``k_idx`` into the paged indexer cache and,
        on full layers, compute per-query **causal** page-level top-k via the
        general (non-decode, ``one_token_per_seq=False``) indexer path.

        ``compute_pages=False`` (MTP IndexShare reuse steps) only performs the
        indexer-key cache write and returns ``topk_pages=None``.

        ``q_group`` (spec verify: draft tokens per request, every valid request
        has exactly that many query rows) enables the opt-in batched grouped
        page-topk, see ``_IDX_VERIFY_BATCHED``.

        Returns ``(idx_cache, topk_pages)`` where ``topk_pages`` is ``[T, k_pages]``
        seq-local page ids (-1 padded) — exactly the sparse kernel's per-query
        unit ids at ``read_block == page_size``. ``topk_pages`` is ``None`` on
        shared layers / when no indexer is wired (caller reuses the threaded one).
        """
        if not is_full or q_idx is None:
            return idx_cache, None
        k_pages = (self.index_topk + self.page_size - 1) // self.page_size
        tp_size = self.mesh.shape["tensor"]
        tshard = tp_size > 1 and q_idx.shape[0] % tp_size == 0
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
            pages_per_seq = pi_.shape[0] // seq_lens_.shape[0]
            # 4D-native in-place write with the run-table DMA kernel the KV write
            # uses (bit-identical to the flat scatter, handles consecutive tokens
            # in one slab); no 3D view (every reshape of the paged cache is a
            # full relayout copy)
            page_, off_, valid_ = _paged_landing(
                k_.shape[0], page_size, seq_lens_, pi_, cuq_, cukv_
            )
            loc_ = jnp.where(valid_, page_ * page_size + off_, -1).astype(jnp.int32)
            # explicit run capacity like the KV write: the default
            # 2*(T//ps)+130 grows the SMEM run table to [3*r_cap, 4] int32,
            # which Mosaic pads to 512 B per row -> 1.5 MiB at r_cap=1024
            # (gp112 10-02, super-computers §20.62); tokens here are
            # contiguous per sequence so runs <= pages touched + seqs
            cache4d = paged_write_back(
                cache_,
                k_.astype(cache_.dtype),
                loc_,
                page_size=page_size,
                r_cap=k_.shape[0] // page_size + seq_lens_.shape[0] + 34,
                interpret=jax.default_backend() == "cpu",
                # 10-06 (closed-levers review): the verify step writes <=256 indexer rows
                # per layer here; the run-table kernel's per-run DMAs are latency-bound at
                # that size (its own measurement: 0.27 ms vs 0.065 ms for the 4D-native
                # scatter at 256 rows; cc64 trace 21 x 153 us = 3.2 ms/step). Same switch
                # the KV write uses; prefill chunks (> SCATTER_ROWS_MAX rows) keep the kernel.
                small_rows_scatter=True,
            )
            cache_out = cache4d
            if not compute_pages:
                return cache_out, jnp.full((q_.shape[0], k_pages), -1, jnp.int32)
            if tshard:
                # Score only this rank's contiguous T/tp query rows. The k-idx
                # cache stays replicated (scatter above is full-chunk), so the
                # only reshard is the topk_pages all-gather at the exit.
                t_local = q_.shape[0] // tp_size
                off = jax.lax.axis_index("tensor") * t_local
                q_ = jax.lax.dynamic_slice_in_dim(q_, off, t_local, axis=0)
                w_ = jax.lax.dynamic_slice_in_dim(w_, off, t_local, axis=0)
                # Per-rank pseudo seq metadata (TSA q_token_offset precedent):
                # clip cu_q_lens to the local window; shrink each seq's kv_len
                # by its query rows falling after the window so the scorer's
                # q_pos = kv_len - q_len + row_offset stays the global
                # absolute position for every local row.
                seq_lens_ = jnp.maximum(seq_lens_ - jnp.maximum(cuq_[1:] - off - t_local, 0), 0)
                cuq_ = jnp.clip(cuq_ - off, 0, t_local)
            topk_pages = _prefill_page_topk(
                q_,
                w_,
                None,  # no 3-D cache view on this path
                seq_lens_,
                pi_,
                cuq_,
                cukv_,
                dist_,
                k_pages=k_pages,
                pages_per_seq=pages_per_seq,
                num_queries_per_block=num_queries_per_block,
                q_group=q_group,
                kernel_cache_shape=cache_.shape,
                cache4d=cache4d,
            )
            if tshard:
                topk_pages = jax.lax.all_gather(topk_pages, "tensor", axis=0, tiled=True)
            return cache_out, topk_pages

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

    def _gather_spec_kv(self, cache, dpa, md, num_tokens, rope):
        """Current latent rows at each spec token's slot: (kv_c [T, lkv], k_pe [T, rope])."""
        page_size = self.page_size
        kv_lora_rank = self.kv_lora_rank
        t_local = num_tokens // self.mesh.shape[dpa]
        in_specs = (P(dpa, None, None, None), P(dpa), P(dpa), P(dpa), P(dpa))
        out_specs = (P(dpa, None), P(dpa, None))

        def _run(cache_, sl_, cuq_, cukv_, pi_):
            loc = _spec_token_slots(sl_, cuq_, cukv_, pi_, t_local, page_size)
            row = _gather_cache_rows(cache_, loc, page_size)
            return row[:, :kv_lora_rank], row[:, kv_lora_rank : kv_lora_rank + rope]

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            cache, md.seq_lens, md.cu_q_lens, md.cu_kv_lens, md.page_indices
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
            return paged_write_back(cache_, row, loc, page_size=page_size, small_rows_scatter=True)

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
        forward_batch=None,
    ):
        """W7: TARGET_VERIFY / DRAFT_EXTEND through the decode path (see _SPEC_AS_DECODE)."""
        num_tokens = q.shape[0]
        pmd = self._spec_metadata_as_decode(md, num_tokens, dpa)
        readonly = bool(getattr(forward_batch, "spec_kvshare_readonly", False))
        if readonly:
            # KVShare: no new KV for draft steps >= 1. The page-level kernel always
            # rewrites its own row, so hand it the row currently in the cache
            # (step 0's) -- an idempotent write. No pre-write, no indexer-key write.
            new_kv_c, new_k_pe = self._gather_spec_kv(
                kv_cache, dpa, md, num_tokens, new_k_pe.shape[-1]
            )
        else:
            # Grouped verify (G query rows per request on the mixed kernel) needs no
            # pre-write: the kernel reads the group's rows from new_kv and writes them
            # back in place. Per-token pseudo-sequences still need it (token t must see
            # tokens < t of the same request from the cache).
            _grp = _use_verify_qgroup(
                forward_batch.forward_mode.is_target_verify(),
                _spec_draft_group(forward_batch, num_tokens),
                num_tokens,
            )
            if not _grp:
                kv_cache = self._prewrite_spec_kv(new_kv_c, new_k_pe, kv_cache, dpa, md)
        # IndexShare on the draft side threads page-topk only; either form counts as reuse.
        reuse = (
            is_full
            and dsa_topk_reuse
            and (dsa_topk_in is not None or dsa_topk_pages_in is not None)
        )
        # Indexer: one vectorised page-topk over the request's T query rows on the
        # ORIGINAL metadata (the prefill-form path; k_pages = index_topk/page_size),
        # not T one-query decode passes -- the ref decode loop cost ~9 ms/step at
        # T=4 (dpa40nap). Only the query block shrinks (512 -> _SPEC_INDEXER_QB).
        draft_group = _spec_draft_group(forward_batch, num_tokens)
        if readonly and reuse:
            topk_pages = None
        else:
            idx_cache, topk_pages = self._maybe_index_prefill_pages(
                is_full,
                q_idx,
                k_idx,
                idx_weights,
                idx_cache,
                dpa,
                md,
                compute_pages=not reuse,
                num_queries_per_block=_SPEC_INDEXER_QB,
                q_group=draft_group,
            )
        topk = None
        if not is_full or reuse:
            assert (
                dsa_topk_in is not None or dsa_topk_pages_in is not None
            ), "shared layer requires dsa_topk_in / dsa_topk_pages_in from a preceding full layer"
            topk_use = dsa_topk_in
            topk_pages_use = dsa_topk_pages_in
        else:
            topk_use = None
            topk_pages_use = topk_pages
        if topk_use is None:
            topk_use = _placeholder_topk_like(topk_pages_use)
        is_verify = forward_batch.forward_mode.is_target_verify()
        page_share_group = _SPEC_VERIFY_PAGE_SHARE if is_verify else 1
        group_queries = False
        if _use_verify_qgroup(is_verify, draft_group, num_tokens):
            page_share_group = draft_group
            group_queries = True
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
            pmd,
            page_share_group=page_share_group,
            group_queries=group_queries,
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
        # KVShare (GLM-5.2 MTP): draft steps >= 1 must not add their own KV to the
        # cache -- they attend the step-0 selection over step-0's KV. A loc of -1
        # makes paged_write_back drop every row (the attend is unchanged).
        readonly = bool(getattr(forward_batch, "spec_kvshare_readonly", False))
        if readonly or derive_loc:
            loc = None  # built rank-locally inside the shard_map (see _run)
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
            elif readonly:
                # KVShare read-only draft step: all rows dropped by paged_write_back.
                loc_ = jnp.full((ql_.shape[0],), -1, jnp.int32)
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

    def _run_sparse_prefill_tsa(
        self, ql, qpe, kvc, kpe, cache, topk_pages, sm_scale, dpa, md, forward_batch, absorb=None
    ):
        """Token-sharded-attention sparse prefill (same call contract as
        :meth:`_run_sparse_prefill`).

        Inside the shard_map an ``all_to_all`` reshards the queries from
        head-TP ([T, H/tp] per device) to token-sharded ([T/tp, H] per device),
        each device attends only its contiguous token slice — whose page union
        is ~2.7x denser than a head-TP query block's — and a second
        ``all_to_all`` reshards the output back to head-TP for the unchanged
        w_uv/o_proj epilogue. The write side keeps the replicated full-chunk
        behaviour, so every device's cache copy stays complete. Projections,
        weights and the decode path are untouched."""
        positions = forward_batch.positions.astype(jnp.int32)
        page_size = self.page_size
        kv_lora_rank = self.kv_lora_rank
        sm = float(sm_scale)
        tp_size = self.mesh.shape["tensor"]
        # Same write contract as _run_sparse_prefill: speculative verify /
        # draft-extend batches derive their self-write slots from the ragged
        # metadata (out_cache_loc is the 2 x draft_token_num allocation list,
        # not per token), and KVShare read-only draft steps write nothing.
        mode = forward_batch.forward_mode
        derive_loc = mode.is_target_verify() or mode.is_draft_extend()
        readonly = bool(getattr(forward_batch, "spec_kvshare_readonly", False))
        if readonly or derive_loc:
            loc = None  # built rank-locally inside the shard_map (see _run)
        else:
            loc = forward_batch.out_cache_loc.astype(jnp.int32)
            if loc.shape[0] != ql.shape[0]:
                raise ValueError(
                    "TSA sparse prefill self-write: out_cache_loc has "
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
        if absorb is not None:
            # q_nope head-sharded, replicated full-head W_UK / W_UV
            in_specs = in_specs + (P(dpa, "tensor", None), P(None, None, None), P(None, None, None))
            args.extend((absorb[0], absorb[1], absorb[2]))
        out_specs = (P(dpa, "tensor", None), P(dpa, None, None, None))

        qb = _TSA_PREFILL_QB
        zigzag = ql.shape[0] % (tp_size * qb) == 0

        def _run(ql_, qpe_, kvc_, kpe_, cache_, tp_, pos_, sl_, cuq_, cukv_, pi_, *rest):
            rest = list(rest)
            absorb_ = ()
            if absorb is not None:
                absorb_, rest = tuple(rest[-3:]), rest[:-3]
            if rest:
                loc_ = rest[0]
            elif readonly:
                # KVShare read-only draft step: all rows dropped by paged_write_back.
                loc_ = jnp.full((ql_.shape[0],), -1, jnp.int32)
            else:
                loc_ = _spec_token_slots(sl_, cuq_, cukv_, pi_, ql_.shape[0], page_size)
            t_local = ql_.shape[0] // tp_size
            if absorb_:
                qn_, wuk_, wuv_ = absorb_
                # the absorbed ql_ is unused on this path (dead after tracing)
                ql_ = qn_
            if zigzag:
                ql_ = tsa_zigzag_permute(ql_, tp_size, qb)
                qpe_ = tsa_zigzag_permute(qpe_, tp_size, qb)
            # head-TP -> token-sharded: [T, H/tp, D] -> [T/tp, H, D]; ranks hold
            # contiguous token slices in rank order, heads concat in rank order
            # (matching the contiguous head-block sharding of q_b_proj).
            # one all_to_all for q_nope ++ q_rope (same permutation, pure data movement ->
            # bit-identical): one SparseCore a2a fewer per layer-window than two separate
            # a2a (gp316 4k trace: the second q a2a waited ~65 us median at its done, 12.8 ms
            # per window); the lane split on the token-sharded side touches 1/tp of the tokens
            nope_w = ql_.shape[-1]
            q_cat_t = jax.lax.all_to_all(
                jnp.concatenate([ql_, qpe_], axis=-1),
                "tensor",
                split_axis=0,
                concat_axis=1,
                tiled=True,
            )
            ql_t = q_cat_t[..., :nope_w]
            qpe_t = q_cat_t[..., nope_w:]
            head_major = bool(absorb_)
            if absorb_:
                # W_UK absorb on the token-sharded rows (all heads): same contraction
                # per element as the model's head-sharded form -> bit identical.
                # Batch-major result [H, T/tp, R]; head-major mode hands it to the
                # kernel as is instead of transposing to token-major.
                ql_t = _absorb_dot(ql_t, wuk_, (((2,), (2,)), ((1,), (1,))), qn_.dtype)
                if not head_major:
                    ql_t = ql_t.transpose(1, 0, 2)
            if head_major:
                qpe_t = qpe_t.transpose(1, 0, 2)  # [H, T/tp, rope] (64 lanes; small)
            rank = jax.lax.axis_index("tensor")
            if zigzag:
                q_idx = tsa_zigzag_token_index(t_local, tp_size, qb, rank)
                tp_loc = jnp.take(tp_, q_idx, axis=0)
                pos_loc = jnp.take(pos_, q_idx, axis=0)
                off_kw = dict(q_token_index=q_idx)
            else:
                off = rank * t_local
                tp_loc = jax.lax.dynamic_slice_in_dim(tp_, off, t_local, axis=0)
                pos_loc = jax.lax.dynamic_slice_in_dim(pos_, off, t_local, axis=0)
                off_kw = dict(q_token_offset=off)
            o_t, cache_new = prefill_write_and_attend_ragged_qblock_tsa(
                ql_t,
                qpe_t,
                kvc_,
                kpe_,
                cache_,
                tp_loc,
                pos_loc,
                loc_,
                sl_,
                cuq_,
                cukv_,
                pi_,
                kv_lora_rank=kv_lora_rank,
                page_size=page_size,
                sm_scale=sm,
                query_block=qb,
                # the first consumer casts o_t to bf16 (W_UV dot operand / a2a payload), so
                # let the kernel store bf16 directly: same rounding point, no f32 pass
                # (gp217 4k trace: convert fusions on the attend output 4.6 + 2.6 ms)
                out_dtype=qn_.dtype if absorb_ else ql_.dtype,
                head_major=head_major,
                **off_kw,
            )
            if head_major:
                # W_UV on the head-major kernel output: [H, T/tp, R] -> [H, T/tp, v_head_dim]
                # (batch-major in and out: no relayout on either side of the dot).
                o_t = _absorb_dot(
                    o_t.astype(qn_.dtype), wuv_, (((2,), (0,)), ((0,), (1,))), qn_.dtype
                )
            elif absorb_:
                # W_UV on the token-sharded rows before the return a2a:
                # [T/tp, H, R] -> [T/tp, H, v_head_dim]
                o_t = _absorb_dot(
                    o_t.astype(qn_.dtype), wuv_, (((2,), (0,)), ((1,), (1,))), qn_.dtype
                )
                o_t = o_t.transpose(1, 0, 2)
            # token-sharded -> head-TP: [T/tp, H, Dv] -> [T, H/tp, Dv]
            if head_major:
                # [H, T/tp, Dv] -> [H/tp, T, Dv] -> token-major [T, H/tp, Dv] for o_proj
                o = jax.lax.all_to_all(
                    o_t.astype(ql_.dtype), "tensor", split_axis=0, concat_axis=1, tiled=True
                )
                o = o.transpose(1, 0, 2)
            else:
                o = jax.lax.all_to_all(
                    o_t.astype(ql_.dtype), "tensor", split_axis=1, concat_axis=0, tiled=True
                )
            if zigzag:
                o = tsa_zigzag_unpermute(o, tp_size, qb)
            return o, cache_new

        return jax.shard_map(_run, in_specs=in_specs, out_specs=out_specs, check_vma=False)(*args)

    def _run_dense(
        self,
        ql,
        qpe,
        kvc,
        kpe,
        cache,
        sm_scale,
        layer,
        dpa,
        md,
        q_head_major=False,
    ):
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
                # head-major needs the decode cases skipped (EXTEND: dist[0] == 0)
                skip_decode_cases=q_head_major,
                q_head_major=q_head_major,
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
    ``pos <= abs_q``) and ``cu_q = arange``.

    Page table: a stride view, not a copy. The page-level decode kernel locates a
    sequence's pages as ``page_indices[cu_kv_lens[seq] // page_size + local]`` and
    never assumes the segments are disjoint, so every pseudo-sequence simply
    points at its origin sequence's segment: ``cu_kv[i] = cu_kv_lens[s]`` and
    ``page_indices`` is passed through unchanged. (The previous per-token copy of
    the full ``pages_per_seq`` window was a ``[T, W]`` gather per layer: 256 x 1056
    int32 at cc64, ~40 ms of a 168 ms verify step.) Padding tokens and empty
    sequences get ``kv_len 0`` and point at segment 0.
    Returns ``(kv_len, cu_q_lens, cu_kv_lens, page_indices, distribution)``.
    """
    num_seqs = seq_lens.shape[0]
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
    cu_kv = jnp.where(valid, cu_kv_lens[seg_c], 0).astype(jnp.int32)
    cu_kv = jnp.concatenate([cu_kv, cu_kv[-1:]])  # [T + 1]: the kernel indexes cu_kv[seq_id] only
    cu_q = jnp.arange(num_tokens + 1, dtype=jnp.int32)
    n_valid = valid.sum().astype(jnp.int32)
    dist = jnp.stack([n_valid, n_valid, n_valid]).astype(jnp.int32)
    return kv_len, cu_q, cu_kv, page_indices.astype(jnp.int32), dist


def _gather_cache_rows(cache: jax.Array, loc: jax.Array, page_size: int) -> jax.Array:
    """Rows ``cache[loc[t]]`` of a ``[P, ps//pk, pk, D]`` paged cache; loc < 0 -> zeros."""
    pk = cache.shape[2]
    safe = jnp.maximum(loc, 0)
    page = safe // page_size
    off = safe % page_size
    rows = cache[page, off // pk, off % pk]
    return jnp.where((loc >= 0)[:, None], rows, jnp.zeros_like(rows))


def _tsa_eligible(num_tokens_global: int, dp_size: int, tp_size: int) -> bool:
    """TSA prefill needs the per-rank (after DP sharding) token count divisible by tp so the
    all_to_all tiles evenly; checking the global count is wrong (DP=2, tp=2, T=6 passes
    globally but each rank holds 3 tokens). Shapes are static at trace time."""
    if tp_size <= 1 or dp_size <= 0:
        return False
    return (num_tokens_global // dp_size) % tp_size == 0


def _spec_draft_group(forward_batch, num_tokens: int) -> int | None:
    """Draft tokens per request (G) on a decode-form TARGET_VERIFY batch whose
    token count is a whole number of requests; ``None`` otherwise."""
    if not forward_batch.forward_mode.is_target_verify():
        return None
    dtn = getattr(getattr(forward_batch, "spec_info", None), "draft_token_num", None)
    if dtn and int(dtn) > 1 and num_tokens % int(dtn) == 0:
        return int(dtn)
    return None


def _prefill_page_topk(
    q_,
    w_,
    cache3d,
    seq_lens_,
    pi_,
    cuq_,
    cukv_,
    dist_,
    *,
    k_pages: int,
    pages_per_seq: int,
    num_queries_per_block,
    q_group,
    kernel_cache_shape: tuple[int, ...] | None = None,
    cache4d: jax.Array | None = None,
):
    """Per-query causal page-topk over ``cache3d`` [P, page_size, D] for one
    shard of a prefill-form batch; ``i32[T, k_pages]`` (-1 padded).
    ``kernel_cache_shape`` is the paged 4-D cache shape the Pallas kernel takes;
    ``cache4d`` (native 4-D cache) feeds the kernel path directly without a
    relayout, and the 3-D view is only materialized for the reference paths.

    Path order: batched grouped XLA op (opt-in, verify batches only) ->
    per-sequence Pallas kernel (``DSA_INDEXER_KERNEL_PREFILL``) -> loop ref.
    The page budget may exceed the (batch-sized) page table width; top_k needs
    k <= pages_per_seq. Clamp and right-pad back to the [T, k_pages] contract.
    """
    if cache3d is None:
        if not _INDEXER_KERNEL_PREFILL or q_group is not None:
            cache3d = cache4d.reshape(cache4d.shape[0], -1, cache4d.shape[-1])
        page_size = cache4d.shape[1] * cache4d.shape[2]
    else:
        page_size = cache3d.shape[1]
    k_eff = min(k_pages, pages_per_seq)
    grouped = (
        q_group is not None
        and _IDX_VERIFY_BATCHED
        and seq_lens_.shape[0] >= _IDX_VERIFY_BATCHED_MIN_BS
        and pages_per_seq * page_size <= _IDX_VERIFY_BATCHED_MAX_KV
    )
    if grouped and _identity_allowed(pages_per_seq) and k_eff >= pages_per_seq:
        # budget covers the bucket: every page visible to each query is selected -> no scoring
        # (cc64 1k/1k census 10-04: this grouped reference path is 4.9 ms/step)
        topk_pages = _identity_page_topk_grouped(
            seq_lens_, cuq_, q_.shape[0], k_eff, pages_per_seq, page_size, int(q_group)
        )
    elif grouped:
        topk_pages = streamindex_page_topk_ref_grouped(
            q_,
            w_,
            cache3d,
            seq_lens_,
            pi_,
            cuq_,
            cukv_,
            k_pages=k_eff,
            pages_per_seq=pages_per_seq,
            q_group=int(q_group),
        )
    elif _INDEXER_KERNEL_PREFILL:
        # dist_[2] == number of real (seq_len > 0) sequences in this
        # EXTEND batch: mla_backend builds distribution = [0, 0, N] for
        # ForwardMode.EXTEND (decode runs as a separate forward), so the
        # kernel's per-seq grid over [0, N) matches the ref's masked
        # full-batch loop exactly; padded seqs stay -1 on both paths.
        assert kernel_cache_shape is not None, "kernel path needs the paged 4-D cache shape"
        topk_pages = streamindex_page_topk(
            q_,
            w_,
            cache4d if cache4d is not None else cache3d.reshape(kernel_cache_shape),
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
            **(
                {}
                if num_queries_per_block is None
                else {"num_queries_per_block": num_queries_per_block}
            ),
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
    if k_eff < k_pages:
        topk_pages = _pad_topk_pages(topk_pages, k_pages)
    return topk_pages


def _pad_topk_pages(topk_pages: jax.Array, width: int) -> jax.Array:
    """Right-pad a ``[T, k]`` page-topk with -1 to ``width`` columns (no-op if wide enough)."""
    k = topk_pages.shape[1]
    if k >= width:
        return topk_pages
    return jnp.pad(topk_pages, ((0, 0), (0, width - k)), constant_values=-1)


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


def dense_arm_metadata(md):
    """Metadata for the exact-dense attend kernel (``mla_ragged_paged_attention``, MIXED case).

    The mixed kernel sizes each sequence's KV loop by ``kv_len`` and its query-block pipeline by the
    query rows, so a sequence with ``q_len == 0`` but ``kv_len > 0`` is a shape it never handles
    (see ``sparse_mla.py``: the next sequence is never prefetched -> device hang). Such sequences
    exist only in prefill micro-batch windows (``window_forward_metadata``: a request whose rows all
    lie *before* the window keeps ``kv_len`` = its prefix with ``q_len == 0``); the single-window
    batch only has ``q_len == kv_len == 0`` padding. Zero ``kv_len`` for every ``q_len == 0``
    sequence: the kernel then skips it, and it has no rows to attend or write anyway. (knife 1 V2,
    2026-10-07: the 8 x 1k two-window exact-dense program hung at the first batch on three arms.)
    """
    cuq = getattr(md, "cu_q_lens", None)
    seq_lens = getattr(md, "seq_lens", None)
    if cuq is None or seq_lens is None or cuq.shape[0] != seq_lens.shape[0] + 1:
        return md
    q_lens = cuq[1:] - cuq[:-1]
    out = MLAAttentionMetadata()
    for name in ("cu_q_lens", "cu_kv_lens", "page_indices"):
        setattr(out, name, getattr(md, name))
    out.seq_lens = jnp.where(q_lens > 0, seq_lens, 0).astype(seq_lens.dtype)
    # The MIXED case iterates sequences [distribution[1], distribution[2]); sequences[distribution[0]:distribution[1]]
    # ("chunked-prefill-only" slot) are never launched. A window's sequences with rows are contiguous, so hand the
    # kernel exactly that range: it then never sees a q_len == 0 sequence at all (leading ones in window B, trailing
    # ones in window A, bs padding). Masking kv_len alone was not enough: 8 x 1k two-window still hung (gp785 2026-10-08,
    # py-spy: scheduler waiting on the fused-prefill output) while 4 x 1k passed.
    has_q = q_lens > 0
    any_q = jnp.any(has_q)
    first = jnp.argmax(has_q).astype(jnp.int32)
    last = (has_q.shape[0] - 1 - jnp.argmax(has_q[::-1])).astype(jnp.int32)
    dist = getattr(md, "distribution", None)
    if dist is not None and dist.shape == (3,):
        # arithmetic on the original array keeps its (Explicit-mesh, P(data)) sharding; a freshly stacked array is
        # replicated and fails _run_dense's shard_map in_specs=P(data) (gp786 2026-10-08), and .at[].set needs an
        # explicit out_sharding under Explicit axes
        idx = jnp.arange(3, dtype=dist.dtype)
        vals = jnp.where(idx == 1, first, jnp.where(idx == 2, last + 1, 0)).astype(dist.dtype)
        vals = jnp.where(any_q, vals, 0).astype(dist.dtype)
        out.distribution = dist * 0 + vals
    else:
        out.distribution = dist
    return out


def window_forward_metadata(md, offset: int, length: int):
    """Attention metadata for the query-row window ``[offset, offset + length)``
    of a ragged EXTEND batch (prefill micro-batching: the chunk's rows are run
    as consecutive micro-batches inside one forward).

    ``cu_kv_lens`` / ``page_indices`` / ``distribution`` are kept, so every
    sequence's pages stay where the full-chunk metadata put them; ``cu_q_lens``
    is clipped to the window and each sequence's kv_len shrinks by the number
    of its query rows that fall *after* the window. Then
    ``q_pos = kv_len - q_len + row`` (the convention of ``_paged_landing``, the
    indexer scorer and the qblock kernels) is the same absolute position for
    every window row as in the full chunk, a request cut by the window boundary
    becomes an ordinary chunked-prefill continuation in the later window, and a
    request with no rows in the window has ``q_len == 0`` (kv_len == its prefix,
    or the full length once written) -- the shape the per-rank (token-sharded)
    metadata already produces. ``dp_size`` must be 1 (flat [S+1] / [S] arrays).
    """
    cuq = md.cu_q_lens
    seq_lens = md.seq_lens
    if cuq.shape[0] != seq_lens.shape[0] + 1:
        raise ValueError(
            "window_forward_metadata needs dp_size == 1 metadata: "
            f"cu_q_lens {cuq.shape} vs seq_lens {seq_lens.shape}"
        )
    q_lens = cuq[1:] - cuq[:-1]
    rows_after = jnp.minimum(jnp.maximum(cuq[1:] - (offset + length), 0), q_lens)
    out = MLAAttentionMetadata()
    out.cu_q_lens = jnp.clip(cuq - offset, 0, length).astype(cuq.dtype)
    out.seq_lens = jnp.maximum(seq_lens - rows_after, 0).astype(seq_lens.dtype)
    out.cu_kv_lens = md.cu_kv_lens
    out.page_indices = md.page_indices
    out.distribution = md.distribution
    return out


def _paged_landing(
    T: int,
    page_size: int,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
):
    """(page, offset, valid) landing slots for the T new tail tokens.

    For seq i with q tokens cu_q_lens[i]..[i+1), token j lands at absolute
    position seq_lens[i] - (q_end - q_start) + j.
    """
    page, offset, valid, _, _ = _paged_landing_bounds(
        T, page_size, seq_lens, page_indices, cu_q_lens, cu_kv_lens
    )
    return page, offset, valid


def _paged_landing_bounds(
    T: int,
    page_size: int,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
):
    """``_paged_landing`` plus each token's request bounds ``(q_start, q_end)``."""
    S = seq_lens.shape[0]
    t = jnp.arange(T)
    seq_id = jnp.searchsorted(cu_q_lens[1:], t, side="right")
    seq_id = jnp.clip(seq_id, 0, S - 1)
    q_start = cu_q_lens[seq_id]
    q_end = cu_q_lens[seq_id + 1]
    kv_len = seq_lens[seq_id]
    abs_pos = jnp.maximum(kv_len - (q_end - q_start) + (t - q_start), 0)
    valid = (t >= q_start) & (t < q_end) & (kv_len > 0)
    page = page_indices[cu_kv_lens[seq_id] // page_size + abs_pos // page_size]
    return page, abs_pos % page_size, valid, q_start, q_end


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
    does via input_output_aliases.
    """
    page_size = cache3d.shape[1]
    page, offset, valid = _paged_landing(
        new_tokens.shape[0], page_size, seq_lens, page_indices, cu_q_lens, cu_kv_lens
    )

    # Padding rows must land on the reserved page: the allocator hands out
    # local pages 1..pages_per_rank and keeps page 0 (reads through it are
    # masked past kv_len), while the LAST page is allocatable — near-full
    # pools would otherwise get offset 0 of a live page's keys clobbered.
    sentinel = 0
    safe_page = jnp.where(valid, page, sentinel)
    safe_off = jnp.where(valid, offset, 0)
    return cache3d.at[safe_page, safe_off].set(new_tokens.astype(cache3d.dtype))


def _dense_arm_idx_write_pages(
    cache4d, new_tokens, page_size, seq_lens, page_indices, cu_q_lens, cu_kv_lens, pages_per_seq
):
    """Indexer-key write of the dense prefill arms as whole-page gather / overlay / scatter.

    Units = the pages the window's T new rows can touch: at most ``T // page_size + 2 * S`` (each sequence
    adds at most one partial page at each end), enumerated by the prefix sum of per-sequence touched-page
    counts -- NOT ``S x pages_per_seq`` (pages_per_seq is the max-context page count, 1056 at 135k: the
    first form reserved 17.6 GB of temporaries in the 8192-token program, gp811 2026-10-10). Per unit:
    gather its page, overlay the new rows that land in it (rows [q_start, q_end) of the sequence at
    absolute positions kv_len - q_len + r), scatter it back by physical page index (unique per live unit,
    dead units dropped). Same slot semantics as ``_scatter_paged`` / ``_paged_landing``; bit-identical on
    live pages (test_dense_arm_write_parity_pages)."""
    Pn, pspk, pk, D = cache4d.shape
    assert pspk * pk == page_size, (cache4d.shape, page_size)
    S = seq_lens.shape[0]
    T = new_tokens.shape[0]
    U = T // page_size + 2 * S  # ceil(q / page_size) + 1 pages per sequence, summed
    rows = new_tokens.astype(cache4d.dtype)
    rows_pad = jnp.concatenate([rows, jnp.zeros((1, D), rows.dtype)], axis=0)  # index T = zero row
    q_len_s = cu_q_lens[1:] - cu_q_lens[:-1]  # [S]
    pos0_s = seq_lens - q_len_s  # absolute position of each sequence's first new token
    live_s = (q_len_s > 0) & (seq_lens > 0)
    first_s = jnp.where(live_s, pos0_s // page_size, 0)
    last_s = jnp.where(live_s, (seq_lens - 1) // page_size, -1)
    touched = jnp.where(live_s, last_s - first_s + 1, 0)  # pages touched per sequence
    cum = jnp.cumsum(touched)  # [S]
    units = jnp.arange(U)
    i = jnp.minimum(jnp.searchsorted(cum, units, side="right"), S - 1)
    j = units - jnp.where(i > 0, cum[jnp.maximum(i - 1, 0)], 0)
    live = units < cum[S - 1]
    q_start = cu_q_lens[i]
    q_len = q_len_s[i]
    pos0 = pos0_s[i]
    slot = first_s[i] + j
    page = page_indices[cu_kv_lens[i] // page_size + jnp.clip(slot, 0, pages_per_seq - 1)]
    page = jnp.where(live & (slot < pages_per_seq), page, -1)
    tok = (q_start + slot * page_size - pos0)[:, None] + jnp.arange(page_size)[None, :]  # [U, page]
    valid = (page >= 0)[:, None] & (tok >= q_start[:, None]) & (tok < (q_start + q_len)[:, None])
    new_rows = jnp.take(rows_pad, jnp.where(valid, tok, T), axis=0)  # [U, page, D]
    old = cache4d.at[jnp.maximum(page, 0)].get(mode="clip").reshape(U, page_size, D)
    merged = jnp.where(valid[:, :, None], new_rows, old).reshape(U, pspk, pk, D)
    return cache4d.at[page].set(
        merged, mode="drop", wrap_negative_indices=False, unique_indices=True
    )


def _scatter_paged_inplace(
    cache4d: jax.Array,
    new_tokens: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    pages_per_seq: int,
) -> jax.Array:
    """In-place Pallas twin of ``_scatter_paged`` on the NATIVE 4D cache.

    Takes the pool's ``[P, page_size // packing, packing, D]`` cache directly
    (no 3D view: any reshape of this array is an XLA relayout copy, see
    ``scatter_paged_inplace``). Position ``o`` within a page lives at
    ``(sub, pair) = (o // packing, o % packing)`` — the same row-major
    convention the 3D ``[P, page_size, D]`` view implies.

    The kernel read-modify-writes every slab exactly once, so tokens of one
    request that share a slab (consecutive positions ``packing * k + r``,
    e.g. the 4 verify positions per request) are merged here: the first such
    token in the batch is the slab OWNER and carries a ``[packing, D]``
    payload whose row ``r`` is the new row of token ``i - pair_i + r`` (its
    same-slab partner), written for the row range ``[lo, hi)`` =
    ``[pair_i, min(packing, q_end - (i - pair_i)))``. Partner tokens and
    padding rows are routed to distinct slabs of reserved page 0
    (``sub = t % S``) with an empty row range ``lo == hi == 0``, so they fetch
    and store page 0 unchanged (concurrent same-slab RMWs of identical content).
    """
    P, S, packing, _ = cache4d.shape
    page_size = S * packing
    T = new_tokens.shape[0]
    page, offset, valid, q_start, q_end = _paged_landing_bounds(
        T, page_size, seq_lens, page_indices, cu_q_lens, cu_kv_lens
    )
    t = jnp.arange(T)
    pair = offset % packing
    slab_first = t - pair  # batch index that would hold row 0 of this slab
    # Owner = first token of its request inside the slab: row 0 of the slab,
    # or the request's first token when it starts mid-slab.
    owner = valid & ((pair == 0) | (t == q_start))
    lo = jnp.where(owner, pair, 0)
    hi = jnp.where(owner, jnp.minimum(packing, q_end - slab_first), 0)
    slots = jnp.stack(
        [
            jnp.where(owner, page, 0),
            jnp.where(owner, offset // packing, t % S),
            lo,
            hi,
        ]
    ).astype(jnp.int32)
    # payload[i, r] = new_tokens[i - pair_i + r]; rows outside [lo, hi) are never read.
    src = jnp.clip(slab_first[:, None] + jnp.arange(packing)[None, :], 0, T - 1)
    new_slabs = jnp.take(new_tokens.astype(cache4d.dtype), src, axis=0)
    return scatter_rows_inplace(slots, new_slabs, cache4d)
