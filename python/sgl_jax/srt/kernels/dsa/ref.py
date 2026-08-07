"""JNP reference implementations for DSA sparse attention.

These are correctness oracles for the Pallas kernels in this directory, and
also serve as the Phase-A e2e path (``--dsa-use-pallas=false``). All functions
are jit-compatible with static shapes.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.dsa.topk import select_indexer_topk

_NEG_INF = float("-inf")
# v7x optimum for H=32 and the 135168-token GLM-5.2 score bucket. Sixteen
# rows cross the dense all-head materialization guard below and fall back to
# the much slower per-head accumulator loop.
_INDEXER_QUERY_BLOCK_SIZE = 14


def _mask_and_compact_topk_indices(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Mask invalid candidates and compact them without sorting valid scores."""

    valid = values > _NEG_INF
    masked = jnp.where(valid, indices, -1)

    def compact_invalid_to_end() -> jax.Array:
        num_rows, k = indices.shape
        destination = jnp.where(
            valid,
            jnp.cumsum(valid, axis=-1, dtype=jnp.int32) - 1,
            k,
        )
        row = jnp.arange(num_rows, dtype=jnp.int32)[:, None]
        compacted = jnp.full((num_rows, k + 1), -1, dtype=indices.dtype)
        compacted = compacted.at[row, destination].set(masked)
        return compacted[:, :k]

    # Long-context rows have K finite candidates, so the serving hot path
    # bypasses compaction. Short-context/padded rows take the O(K) fallback to
    # preserve the valid-prefix ABI required by exact sparse attention.
    return jax.lax.cond(jnp.all(valid), lambda: masked, compact_invalid_to_end)


def _run_score_select_pipeline(
    num_tiles: jax.Array,
    score_tile: Callable[[jax.Array], jax.Array],
    select_and_store_tile: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    out: jax.Array,
) -> jax.Array:
    """Ping-pong score construction with selection of the preceding tile."""

    def run_nonempty(pipeline_out):
        score_buffer_0 = score_tile(jnp.int32(0))
        score_buffer_1 = jnp.empty_like(score_buffer_0)

        def pipeline_body(tile_id, pipeline_carry):
            def run_step(current_scores, current_out):
                # Scoring the next tile and selecting the current tile have no
                # data dependency, allowing TensorCore/SparseCore overlap.
                next_scores = score_tile(tile_id + 1)
                current_out = select_and_store_tile(
                    tile_id,
                    current_scores,
                    current_out,
                )
                return next_scores, current_out

            def even_step(buffers):
                scores_0, scores_1, current_out = buffers
                next_scores, current_out = run_step(scores_0, current_out)
                return scores_0, next_scores, current_out

            def odd_step(buffers):
                scores_0, scores_1, current_out = buffers
                next_scores, current_out = run_step(scores_1, current_out)
                return next_scores, scores_1, current_out

            return jax.lax.cond(
                jax.lax.bitwise_and(tile_id, 1) == 0,
                even_step,
                odd_step,
                pipeline_carry,
            )

        score_buffer_0, score_buffer_1, pipeline_out = jax.lax.fori_loop(
            0,
            num_tiles - 1,
            pipeline_body,
            (score_buffer_0, score_buffer_1, pipeline_out),
        )

        final_tile = num_tiles - 1

        def drain_buffer_0(buffers):
            scores_0, _, current_out = buffers
            return select_and_store_tile(final_tile, scores_0, current_out)

        def drain_buffer_1(buffers):
            _, scores_1, current_out = buffers
            return select_and_store_tile(final_tile, scores_1, current_out)

        return jax.lax.cond(
            jax.lax.bitwise_and(final_tile, 1) == 0,
            drain_buffer_0,
            drain_buffer_1,
            (score_buffer_0, score_buffer_1, pipeline_out),
        )

    return jax.lax.cond(
        num_tiles > 0,
        run_nonempty,
        lambda pipeline_out: pipeline_out,
        out,
    )


def build_index_share_map(
    indexer_types: list[str] | None,
    skip_offset: int,
    num_layers: int,
) -> tuple[dict[int, int], dict[int, int], int]:
    """Static IndexShare layer→slot mapping.

    Returns:
      full_slot: layer_id → slot_id for layers with indexer_type == "full"
      src_slot:  layer_id → slot_id whose topk this layer consumes
                 (its own slot if full, nearest preceding full's slot if shared)
      num_full:  number of full layers (== number of indexer_key buffers)
    """
    if indexer_types is None:
        indexer_types = ["full"] * num_layers
    assert len(indexer_types) == num_layers

    full_slot: dict[int, int] = {}
    src_slot: dict[int, int] = {}
    last_slot = -1
    for layer_id, itype in enumerate(indexer_types):
        if itype == "full":
            last_slot = len(full_slot)
            full_slot[layer_id] = last_slot
            src_slot[layer_id] = last_slot
        elif itype == "shared":
            assert last_slot >= 0, f"layer {layer_id} is shared but no preceding full"
            src_slot[layer_id] = last_slot
        else:
            raise ValueError(f"unknown indexer_type {itype!r} at layer {layer_id}")
    return full_slot, src_slot, len(full_slot)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "pages_per_seq",
        "one_token_per_seq",
        "topk_impl",
        "score_query_block_size",
    ),
)
def score_and_select_index_tokens(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    one_token_per_seq: bool = False,
    topk_impl: str,
    score_query_block_size: int = _INDEXER_QUERY_BLOCK_SIZE,
) -> jax.Array:
    """Score cached index keys and select the top-k token positions.

    ``one_token_per_seq=True`` (decode: T == num_seqs, token i belongs to
    seq i) scores only each iteration's own query row — O(S * max_kv) total
    instead of O(S * T * max_kv). Each sequence's ``[1, max_kv]`` score row is
    one pipeline tile, so scoring sequence ``n + 1`` can overlap selection for
    sequence ``n`` without combining their page tables; see
    ``streamindex_page_topk_ref``.

    Scores are ``sum_h relu(q_h · k) * w_h`` per DSA semantics. Because ReLU
    sits before the head sum, the naive ``[T, H, max_kv]`` intermediate OOMs
    at chunked-prefill sizes. Extend streams fixed-size query blocks through a
    score/selection software pipeline: while SparseCore selects block ``n``,
    TensorCore can construct scores for block ``n + 1``. Only the two pipeline
    score buffers are live instead of one packed ``[T, max_kv]`` allocation.

    Args:
      q_idx:         f[T, H, D]  indexer query heads
      idx_weights:   f[T, H]     per-head mixing weights
      index_key_cache: f[P, page_size, D]  paged indexer keys
      seq_lens:     i32[S]      kv length per sequence
      page_indices: i32[N_pages]  packed; seq i's pages start at cu_kv_lens[i]//page_size
      cu_q_lens:    i32[S+1]
      cu_kv_lens:   i32[S+1]    cumsum of page-aligned kv lens (page_indices stride)
      distribution: i32[3]      (decode_end, prefill_end, num_seqs)
      k:            top-k budget
      pages_per_seq: static, maximum pages materialized per sequence
      topk_impl: selection backend: approximate XLA, exact XLA, or exact
        SparseCore radix selection (``approx``, ``exact_lax``, or ``radix``).
      score_query_block_size: query rows carried in one score buffer. At long
        context, a sufficiently small block keeps the all-head score temporary
        below the dense-path threshold, allowing XLA to fuse the head reduction
        instead of updating a loop-carried score matrix once per head.

    Returns:
      i32[T, k]  top-k kv positions per query token; -1 for padding.
    """
    T, H, D = q_idx.shape
    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]
    active_num_seqs = jnp.clip(distribution[2], 0, num_seqs)
    if score_query_block_size < 1:
        raise ValueError(f"score_query_block_size must be positive, got {score_query_block_size}")

    idx_weights_f32 = idx_weights.astype(jnp.float32)
    kv_pos = jnp.arange(max_kv, dtype=jnp.int32)

    def score_index_tile(q_tile, weights_tile, seq_k_idx):
        if q_tile.shape[0] * H * max_kv <= 1 << 26:
            similarities = jnp.einsum(
                "thd,kd->thk",
                q_tile,
                seq_k_idx,
                preferred_element_type=jnp.float32,
            )
            return jnp.einsum("th,thk->tk", weights_tile, jax.nn.relu(similarities))

        def accumulate_head(head_id, scores):
            q_head = jax.lax.dynamic_index_in_dim(q_tile, head_id, axis=1, keepdims=False)
            weight_head = jax.lax.dynamic_index_in_dim(weights_tile, head_id, axis=1, keepdims=True)
            similarities = jnp.einsum(
                "td,kd->tk",
                q_head,
                seq_k_idx,
                preferred_element_type=jnp.float32,
            )
            return scores + jax.nn.relu(similarities) * weight_head

        return jax.lax.fori_loop(
            0,
            H,
            accumulate_head,
            jnp.zeros((q_tile.shape[0], max_kv), jnp.float32),
        )

    def select_topk_indices(scores):
        values, indices = select_indexer_topk(
            scores,
            k=k,
            implementation=topk_impl,
        )
        return _mask_and_compact_topk_indices(values, indices)

    if one_token_per_seq:
        out = jnp.full((T, k), -1, dtype=jnp.int32)

        def score_decode_tile(seq_id):
            kv_len = seq_lens[seq_id]
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
            )
            seq_k_idx = index_key_cache[seq_pages].reshape(max_kv, D)
            q_idx_i = jax.lax.dynamic_slice_in_dim(q_idx, seq_id, 1, axis=0)  # [1, H, D]
            idx_weights_i = jax.lax.dynamic_slice_in_dim(
                idx_weights_f32, seq_id, 1, axis=0
            )  # [1, H]

            with jax.named_scope("dsa_indexer_decode_score_tile"):
                scores = score_index_tile(q_idx_i, idx_weights_i, seq_k_idx)

            return jnp.where(kv_pos[None, :] < kv_len, scores, _NEG_INF)

        def select_and_store_decode_tile(seq_id, scores, decode_out):
            with jax.named_scope("dsa_indexer_decode_topk_tile"):
                idx = select_topk_indices(scores)
            return jax.lax.dynamic_update_slice_in_dim(
                decode_out,
                idx,
                seq_id,
                axis=0,
            )

        return _run_score_select_pipeline(
            active_num_seqs,
            score_decode_tile,
            select_and_store_decode_tile,
            out,
        )

    # Keep each loop-carried score tile small enough to avoid materializing the
    # OOM-sized [T, H, max_kv] tensor. Padding the query axis prevents
    # dynamic_slice from clamping the final ragged block back into valid rows.
    query_block_size = min(score_query_block_size, T)
    q_idx_padded = jnp.pad(q_idx, ((0, query_block_size), (0, 0), (0, 0)))
    idx_weights_padded = jnp.pad(idx_weights_f32, ((0, query_block_size), (0, 0)))
    out_padded = jnp.full((T + query_block_size, k), -1, dtype=jnp.int32)

    def body(seq_id, packed_out):
        q_start = cu_q_lens[seq_id]
        q_end = cu_q_lens[seq_id + 1]
        q_len = jnp.maximum(q_end - q_start, 0)
        kv_len = seq_lens[seq_id]
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        seq_k_idx = index_key_cache[seq_pages].reshape(max_kv, D)

        num_query_blocks = (q_len + query_block_size - 1) // query_block_size

        def score_block(block_id):
            block_start = q_start + block_id * query_block_size
            q_block = jax.lax.dynamic_slice_in_dim(
                q_idx_padded, block_start, query_block_size, axis=0
            )
            weights_block = jax.lax.dynamic_slice_in_dim(
                idx_weights_padded, block_start, query_block_size, axis=0
            )

            local_q_pos = block_id * query_block_size + jnp.arange(
                query_block_size, dtype=jnp.int32
            )
            query_valid = local_q_pos < q_len
            abs_q = kv_len - q_len + local_q_pos
            mask = (
                query_valid[:, None]
                & (kv_pos[None, :] < kv_len)
                & (kv_pos[None, :] <= abs_q[:, None])
            )

            with jax.named_scope("dsa_indexer_score_block"):
                scores_block = score_index_tile(q_block, weights_block, seq_k_idx)

            scores_block = jnp.where(mask, scores_block, _NEG_INF)
            return scores_block

        def select_and_store(block_id, scores_block, block_out):
            with jax.named_scope("dsa_indexer_topk_block"):
                # The adapter owns candidate selection. Radix candidates remain
                # unordered; invalid/padded entries are compacted behind them.
                idx = select_topk_indices(scores_block)
            block_start = q_start + block_id * query_block_size
            return jax.lax.dynamic_update_slice_in_dim(
                block_out,
                idx,
                block_start,
                axis=0,
            )

        return _run_score_select_pipeline(
            num_query_blocks,
            score_block,
            select_and_store,
            packed_out,
        )

    out_padded = jax.lax.fori_loop(0, active_num_seqs, body, out_padded)
    return out_padded[:T]


@functools.partial(jax.jit, static_argnames=("k_pages", "pages_per_seq", "one_token_per_seq"))
def streamindex_page_topk_ref(
    q: jax.Array,
    weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k_pages: int,
    pages_per_seq: int,
    one_token_per_seq: bool = False,
) -> jax.Array:
    """Page-level lightning-indexer: max-pool token scores within each page,
    then top-k over pages_per_seq page scores.

    ``one_token_per_seq=True`` (decode batches: T == num_seqs, token i belongs
    to seq i) switches to a fast path where each loop iteration scores only
    its own single query row — O(S * max_kv) total instead of the general
    path's O(S * T * max_kv), which collapses under multi-request decode
    (each iteration would score the full [T, max_kv] and mask all but one row).

    Vs the token-level path (``score_and_select_index_tokens`` + page union, which
    saturates k_pages_max at long context): the page budget is exact, the
    top_k runs over [T, pages_per_seq] instead of [T, max_kv] (page_size×
    smaller), and sparse-MLA cost becomes a true O(k_pages) flat.

    Returns:
      i32[T, k_pages]  seq-local page ids per query token; -1 for padding.
    """
    T, H, D = q.shape
    page_size = cache_kv.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]

    w = weights.astype(jnp.float32)
    out = jnp.full((T, k_pages), -1, dtype=jnp.int32)

    if one_token_per_seq:

        def body_decode(seq_id, out):
            kv_len = seq_lens[seq_id]
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
            )
            keys = cache_kv[seq_pages].reshape(max_kv, D)
            q_i = jax.lax.dynamic_slice_in_dim(q, seq_id, 1, axis=0)  # [1, H, D]
            w_i = jax.lax.dynamic_slice_in_dim(w, seq_id, 1, axis=0)  # [1, H]
            s = jnp.einsum("thd,kd->thk", q_i, keys, preferred_element_type=jnp.float32)
            scores = jnp.einsum("th,thk->tk", w_i, jax.nn.relu(s))  # [1, max_kv]
            # decode: query is the last token (abs pos kv_len-1), so the causal
            # bound coincides with the kv_len bound
            mask = (jnp.arange(max_kv) < kv_len)[None, :]
            scores = jnp.where(mask, scores, _NEG_INF)
            page_scores = scores.reshape(1, pages_per_seq, page_size).max(axis=-1)
            vals, pidx = jax.lax.top_k(page_scores, k_pages)
            pidx = jnp.where(vals > _NEG_INF, pidx, -1)
            return jax.lax.dynamic_update_slice_in_dim(out, pidx, seq_id, axis=0)

        return jax.lax.fori_loop(0, num_seqs, body_decode, out)

    def body(seq_id, out):
        q_start = cu_q_lens[seq_id]
        q_end = cu_q_lens[seq_id + 1]
        kv_len = seq_lens[seq_id]
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        keys = cache_kv[seq_pages].reshape(max_kv, D)

        q_pos = jnp.arange(T)
        kv_pos = jnp.arange(max_kv)
        in_seq_q = (q_pos >= q_start) & (q_pos < q_end)
        abs_q = kv_len - (q_end - q_start) + (q_pos - q_start)
        mask = in_seq_q[:, None] & (kv_pos[None, :] < kv_len) & (kv_pos[None, :] <= abs_q[:, None])

        if T * H * max_kv <= 1 << 26:
            s = jnp.einsum("thd,kd->thk", q, keys, preferred_element_type=jnp.float32)
            scores = jnp.einsum("th,thk->tk", w, jax.nn.relu(s))
        else:

            def h_step(h, acc):
                q_h = jax.lax.dynamic_index_in_dim(q, h, axis=1, keepdims=False)
                w_h = jax.lax.dynamic_index_in_dim(w, h, axis=1, keepdims=True)
                s_h = jnp.einsum("td,kd->tk", q_h, keys, preferred_element_type=jnp.float32)
                return acc + jax.nn.relu(s_h) * w_h

            scores = jax.lax.fori_loop(0, H, h_step, jnp.zeros((T, max_kv), jnp.float32))
        scores = jnp.where(mask, scores, _NEG_INF)
        page_scores = scores.reshape(T, pages_per_seq, page_size).max(axis=-1)
        vals, pidx = jax.lax.top_k(page_scores, k_pages)
        pidx = jnp.where(vals > _NEG_INF, pidx, -1)
        return jnp.where(in_seq_q[:, None], pidx, out)

    return jax.lax.fori_loop(0, num_seqs, body, out)


@functools.partial(jax.jit, static_argnames=("sm_scale", "pages_per_seq", "v_dim"))
def sparse_mla_ref(
    q: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    pages_per_seq: int,
    v_dim: int,
) -> jax.Array:
    """Reference sparse absorbed-MLA over per-query top-k positions.

    Args:
      q:            f[T, H, D_qk]  packed [nope|rope] query in latent space
      cache_kv:     f[P, page_size, D_kv]  packed [c_kv|k_rope] latent cache
      kv_lens:      i32[S]
      topk_indices: i32[T, k]  kv positions per query, -1 = ignore
      page_indices: i32[S * pages_per_seq]
      cu_q_lens:    i32[S+1]
      distribution: i32[3]
      sm_scale:     softmax scale
      pages_per_seq: static
      v_dim:        latent value dim (kv_lora_rank), <= D_kv

    Returns:
      f[T, H, v_dim]
    """
    T, H, _ = q.shape
    _, page_size, Dkv = cache_kv.shape
    num_seqs = kv_lens.shape[0]
    max_kv = pages_per_seq * page_size

    out = jnp.zeros((T, H, v_dim), dtype=q.dtype)

    def body(seq_id, out):
        q_start = cu_q_lens[seq_id]
        q_end = cu_q_lens[seq_id + 1]
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        kv_flat = cache_kv[seq_pages].reshape(max_kv, Dkv)

        q_pos = jnp.arange(T)
        in_seq = (q_pos >= q_start) & (q_pos < q_end)

        kv_len = kv_lens[seq_id]
        idx = jnp.where(topk_indices >= 0, topk_indices, 0)
        valid = (topk_indices >= 0) & (topk_indices < kv_len) & in_seq[:, None]
        kv_sel = kv_flat[idx]  # [T, k, Dkv]

        logits = jnp.einsum("thd,tkd->thk", q.astype(jnp.float32), kv_sel.astype(jnp.float32))
        logits = logits * sm_scale
        logits = jnp.where(valid[:, None, :], logits, _NEG_INF)
        p = jax.nn.softmax(logits, axis=-1)
        p = jnp.where(valid[:, None, :], p, 0.0)

        v_sel = kv_sel[..., :v_dim]
        o = jnp.einsum("thk,tkd->thd", p, v_sel.astype(jnp.float32)).astype(q.dtype)
        return jnp.where(in_seq[:, None, None], o, out)

    return jax.lax.fori_loop(0, num_seqs, body, out)
