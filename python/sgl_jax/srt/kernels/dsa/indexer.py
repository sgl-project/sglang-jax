"""DSA indexer scoring, masking, selection, and software pipelining."""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.dsa.paged_score import paged_decode_scores_pallas
from sgl_jax.srt.kernels.dsa.topk import (
    select_indexer_radix_topk_indices,
    select_indexer_topk,
)

_NEG_INF = float("-inf")
# v7x optimum for H=32 and the 135168-token GLM-5.2 score bucket. Thirty-two
# query rows expose 1024 query-head rows to MXU while keeping score/top-k
# pipeline buffers bounded.
_INDEXER_QUERY_BLOCK_SIZE = 32


def _compact_topk_indices(valid: jax.Array, indices: jax.Array) -> jax.Array:
    """Compact valid unordered candidates before ``-1`` padding."""

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


def _mask_and_compact_topk_indices(values: jax.Array, indices: jax.Array) -> jax.Array:
    """Mask invalid candidates and compact them without sorting valid scores."""

    return _compact_topk_indices(values > _NEG_INF, indices)


def _compute_score_tile(
    q_tile: jax.Array,
    weights_tile: jax.Array,
    sequence_keys: jax.Array,
) -> jax.Array:
    """Compute DSA index scores for one query tile and one sequence."""

    num_queries, num_heads, _ = q_tile.shape
    max_kv = sequence_keys.shape[0]
    if num_queries * num_heads * max_kv <= 1 << 28:
        similarities = jnp.einsum(
            "thd,kd->thk",
            q_tile,
            sequence_keys,
            preferred_element_type=jnp.float32,
        )
        return jnp.einsum("th,thk->tk", weights_tile, jax.nn.relu(similarities))

    def accumulate_head(head_id, scores):
        q_head = jax.lax.dynamic_index_in_dim(q_tile, head_id, axis=1, keepdims=False)
        weight_head = jax.lax.dynamic_index_in_dim(weights_tile, head_id, axis=1, keepdims=True)
        similarities = jnp.einsum(
            "td,kd->tk",
            q_head,
            sequence_keys,
            preferred_element_type=jnp.float32,
        )
        return scores + jax.nn.relu(similarities) * weight_head

    return jax.lax.fori_loop(
        0,
        num_heads,
        accumulate_head,
        jnp.zeros((num_queries, max_kv), jnp.float32),
    )


def _select_topk_indices(
    scores: jax.Array,
    valid_lengths: jax.Array,
    *,
    k: int,
    topk_impl: str,
) -> jax.Array:
    if valid_lengths.shape != scores.shape[:1]:
        raise ValueError(
            f"valid_lengths must have shape {scores.shape[:1]}, got {valid_lengths.shape}"
        )
    if topk_impl == "radix":
        indices = select_indexer_radix_topk_indices(scores, k=k)
        valid = (indices >= 0) & (indices < valid_lengths[:, None])
        return _compact_topk_indices(valid, indices)

    values, indices = select_indexer_topk(
        scores,
        k=k,
        implementation=topk_impl,
    )
    return _mask_and_compact_topk_indices(values, indices)


def _pipeline_score_and_topk_tiles(
    num_tiles: jax.Array,
    score_tile: Callable[[jax.Array], jax.Array],
    topk_tile: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    out: jax.Array,
) -> jax.Array:
    """Ping-pong score construction with top-k on the preceding tile."""

    def run_nonempty(pipeline_out):
        score_buffer_0 = score_tile(jnp.int32(0))
        score_buffer_1 = jnp.empty_like(score_buffer_0)

        def pipeline_body(tile_id, pipeline_carry):
            def run_step(current_scores, current_out):
                # Scoring the next tile and selecting the current tile have no
                # data dependency, allowing TensorCore/SparseCore overlap.
                next_scores = score_tile(tile_id + 1)
                current_out = topk_tile(
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
            return topk_tile(final_tile, scores_0, current_out)

        def drain_buffer_1(buffers):
            _, scores_1, current_out = buffers
            return topk_tile(final_tile, scores_1, current_out)

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


def _compute_decode_scores_and_select_topk_indices(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    topk_impl: str,
) -> jax.Array:
    """Score one query per sequence and select all decode rows in one top-k call.

    Decode requires ``T == num_seqs`` and query row ``i`` to belong to sequence
    ``i``. On TPU, a Pallas scorer follows the page table directly and only
    materializes bounded key tiles in VMEM. A two-row decode uses one persistent
    program that prefetches both page streams and scores them through batched
    matrix products. The resulting ``[T, max_kv]`` FP32 score matrix is
    submitted to top-k as one batch so two-row decode can use both SparseCores
    in the radix implementation. Non-TPU backends keep the JAX gather
    implementation as a correctness fallback.

    Args:
      q_idx:         f[T, H, D]  indexer query heads
      idx_weights:   f[T, H]     per-head mixing weights
      index_key_cache: f[P, page_size, D]  paged indexer keys
      seq_lens:     i32[S]      kv length per sequence
      page_indices: i32[N_pages]  packed; seq i's pages start at cu_kv_lens[i]//page_size
      cu_kv_lens:   i32[S+1]    cumsum of page-aligned kv lens (page_indices stride)
      distribution: i32[3]      (decode_end, prefill_end, num_seqs)
      k:            top-k budget
      pages_per_seq: static, maximum pages materialized per sequence
      topk_impl: selection backend: approximate XLA, exact XLA, or exact
        SparseCore radix selection (``approx``, ``exact_lax``, or ``radix``).

    Returns:
      i32[T, k]  top-k kv positions per query token; -1 for padding.
    """
    T, _, D = q_idx.shape
    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]
    if num_seqs != T:
        raise ValueError(
            "decode indexer requires one query row per sequence, "
            f"got q_idx.shape[0]={T} and num_seqs={num_seqs}"
        )

    active_num_seqs = jnp.clip(distribution[2], 0, num_seqs)
    idx_weights_f32 = idx_weights.astype(jnp.float32)
    kv_pos = jnp.arange(max_kv, dtype=jnp.int32)

    with jax.named_scope("Decode"), jax.named_scope("PagedScore"):
        pallas_compatible = (
            D % 128 == 0
            and max_kv % 128 == 0
            and page_size <= 2048
            and q_idx.dtype in (jnp.bfloat16, jnp.float32)
            and idx_weights.dtype in (jnp.bfloat16, jnp.float32)
            and index_key_cache.dtype in (jnp.bfloat16, jnp.float32)
        )
        if jax.default_backend() == "tpu" and pallas_compatible:
            batched_scores = paged_decode_scores_pallas(
                q_idx,
                idx_weights,
                index_key_cache,
                seq_lens,
                page_indices,
                cu_kv_lens,
                distribution,
                pages_per_seq=pages_per_seq,
                persistent_two_seq=T == 2,
                coalesce_page_dma=True,
            )
        else:
            batched_scores = jnp.full((T, max_kv), _NEG_INF, dtype=jnp.float32)

            def score_sequence(seq_id, scores):
                kv_len = seq_lens[seq_id]
                seq_pages = jax.lax.dynamic_slice_in_dim(
                    page_indices,
                    cu_kv_lens[seq_id] // page_size,
                    pages_per_seq,
                )
                sequence_keys = index_key_cache[seq_pages].reshape(max_kv, D)
                q_i = jax.lax.dynamic_slice_in_dim(q_idx, seq_id, 1, axis=0)
                weights_i = jax.lax.dynamic_slice_in_dim(
                    idx_weights_f32,
                    seq_id,
                    1,
                    axis=0,
                )
                row_scores = _compute_score_tile(q_i, weights_i, sequence_keys)
                row_scores = jnp.where(
                    kv_pos[None, :] < kv_len,
                    row_scores,
                    _NEG_INF,
                )
                return jax.lax.dynamic_update_slice_in_dim(
                    scores,
                    row_scores,
                    seq_id,
                    axis=0,
                )

            batched_scores = jax.lax.fori_loop(
                0,
                active_num_seqs,
                score_sequence,
                batched_scores,
            )

    with (
        jax.named_scope("Decode"),
        jax.named_scope("TopK"),
        jax.named_scope(topk_impl),
    ):
        valid_lengths = jnp.where(
            jnp.arange(T, dtype=jnp.int32) < active_num_seqs,
            jnp.clip(seq_lens, 0, max_kv),
            0,
        )
        return _select_topk_indices(
            batched_scores,
            valid_lengths,
            k=k,
            topk_impl=topk_impl,
        )


def _compute_extend_scores_and_select_topk_indices(
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
    topk_impl: str,
    score_query_block_size: int = _INDEXER_QUERY_BLOCK_SIZE,
) -> jax.Array:
    """Score packed extend queries through a bounded score/top-k pipeline.

    Each sequence gathers its key pages once and reuses them across all of its
    query blocks. Fixed-size score tiles bound peak memory and allow scoring
    block ``n + 1`` to overlap top-k selection for block ``n``.
    """
    T, _, D = q_idx.shape
    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]
    active_num_seqs = jnp.clip(distribution[2], 0, num_seqs)
    if score_query_block_size < 1:
        raise ValueError(f"score_query_block_size must be positive, got {score_query_block_size}")

    idx_weights_f32 = idx_weights.astype(jnp.float32)
    kv_pos = jnp.arange(max_kv, dtype=jnp.int32)

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
        with jax.named_scope("Extend"), jax.named_scope("IndexKeyGather"):
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
            )
            seq_k_idx = index_key_cache[seq_pages].reshape(max_kv, D)

        num_query_blocks = (q_len + query_block_size - 1) // query_block_size

        def score_block(block_id):
            with jax.named_scope("Extend"), jax.named_scope("ScoreBlock"):
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
                mask = query_valid[:, None] & (kv_pos[None, :] <= abs_q[:, None])

                scores_block = _compute_score_tile(
                    q_block,
                    weights_block,
                    seq_k_idx,
                )

                scores_block = jnp.where(mask, scores_block, _NEG_INF)
                return scores_block

        def topk_block(block_id, scores_block, block_out):
            with (
                jax.named_scope("Extend"),
                jax.named_scope("TopKBlock"),
                jax.named_scope(topk_impl),
            ):
                # The adapter owns candidate selection. Radix candidates remain
                # unordered; invalid/padded entries are compacted behind them.
                local_q_pos = block_id * query_block_size + jnp.arange(
                    query_block_size, dtype=jnp.int32
                )
                query_valid = local_q_pos < q_len
                valid_lengths = jnp.where(
                    query_valid,
                    jnp.clip(kv_len - q_len + local_q_pos + 1, 0, max_kv),
                    0,
                )
                idx = _select_topk_indices(
                    scores_block,
                    valid_lengths,
                    k=k,
                    topk_impl=topk_impl,
                )
                block_start = q_start + block_id * query_block_size
                return jax.lax.dynamic_update_slice_in_dim(
                    block_out,
                    idx,
                    block_start,
                    axis=0,
                )

        return _pipeline_score_and_topk_tiles(
            num_query_blocks,
            score_block,
            topk_block,
            packed_out,
        )

    out_padded = jax.lax.fori_loop(0, active_num_seqs, body, out_padded)
    return out_padded[:T]


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
def compute_scores_and_select_topk_indices(
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
    """Score indexer queries and select sequence-local top-k token positions.

    ``one_token_per_seq=True`` selects the decode implementation: it scores
    each sequence without materializing ``[T, max_kv, D]`` keys, then submits
    the resulting ``[T, max_kv]`` score matrix to one batched top-k call.

    Otherwise the extend implementation gathers each sequence's keys once and
    streams its query blocks through the score/top-k ping-pong pipeline.
    """
    if one_token_per_seq:
        return _compute_decode_scores_and_select_topk_indices(
            q_idx,
            idx_weights,
            index_key_cache,
            seq_lens,
            page_indices,
            cu_kv_lens,
            distribution,
            k=k,
            pages_per_seq=pages_per_seq,
            topk_impl=topk_impl,
        )

    return _compute_extend_scores_and_select_topk_indices(
        q_idx,
        idx_weights,
        index_key_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        k=k,
        pages_per_seq=pages_per_seq,
        topk_impl=topk_impl,
        score_query_block_size=score_query_block_size,
    )
