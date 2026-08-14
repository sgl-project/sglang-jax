"""DSA indexer scoring, masking, selection, and software pipelining."""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.dsa.paged_score import (
    paged_decode_scores_pallas,
    paged_extend_score_and_map_block_pallas,
    paged_extend_score_block_pallas,
)
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
        indices = select_indexer_radix_topk_indices(
            scores,
            k=k,
        )
        valid = (indices >= 0) & (indices < valid_lengths[:, None])
        return _compact_topk_indices(valid, indices)

    values, indices = select_indexer_topk(
        scores,
        k=k,
        implementation=topk_impl,
    )
    return _mask_and_compact_topk_indices(values, indices)


def _map_logical_topk_to_physical_slots(
    logical_topk: jax.Array,
    seq_pages: jax.Array,
    valid_lengths: jax.Array,
    page_size: int,
) -> jax.Array:
    """Map selected logical positions through one or more sequence page tables."""

    if valid_lengths.shape != logical_topk.shape[:1]:
        raise ValueError(
            f"valid_lengths must have shape {logical_topk.shape[:1]}, got {valid_lengths.shape}"
        )
    logical = jnp.maximum(logical_topk, 0)
    logical_page = logical // page_size
    page_in_bounds = logical_page < seq_pages.shape[-1]
    safe_logical_page = jnp.clip(logical_page, 0, seq_pages.shape[-1] - 1)
    if seq_pages.ndim == 1:
        physical_page = seq_pages[safe_logical_page]
    elif seq_pages.ndim == 2:
        if seq_pages.shape[0] != logical_topk.shape[0]:
            raise ValueError(
                "a batched page table must have one row per top-k row, "
                f"got seq_pages.shape={seq_pages.shape} and "
                f"logical_topk.shape={logical_topk.shape}"
            )
        physical_page = jnp.take_along_axis(
            seq_pages,
            safe_logical_page,
            axis=-1,
        )
    else:
        raise ValueError(f"seq_pages must be rank 1 or 2, got shape={seq_pages.shape}")

    valid = (
        (logical_topk >= 0)
        & page_in_bounds
        & (logical < valid_lengths[:, None])
        & (physical_page >= 0)
    )
    physical_slot = physical_page * page_size + logical % page_size
    return jnp.where(valid, physical_slot, -1).astype(jnp.int32)


def _map_packed_logical_topk_to_physical_slots(
    logical_topk: jax.Array,
    page_indices: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    active_num_seqs: jax.Array,
    *,
    pages_per_seq: int,
    page_size: int,
) -> jax.Array:
    """Resolve a packed extend batch after all score/Top-K work completes."""

    num_queries = logical_topk.shape[0]
    num_seqs = seq_lens.shape[0]
    query_pos = jnp.arange(num_queries, dtype=jnp.int32)

    # cu_q_lens partitions the packed query rows. Derive one sequence id and
    # page-table base per row without materializing [T, pages_per_seq].
    query_seq_id = jnp.sum(
        query_pos[:, None] >= cu_q_lens[1:][None, :],
        axis=1,
        dtype=jnp.int32,
    )
    safe_seq_id = jnp.clip(query_seq_id, 0, num_seqs - 1)
    query_start = cu_q_lens[safe_seq_id]
    query_end = cu_q_lens[safe_seq_id + 1]
    query_len = jnp.maximum(query_end - query_start, 0)
    local_query_pos = query_pos - query_start
    query_valid = (
        (query_seq_id < active_num_seqs)
        & (query_seq_id < num_seqs)
        & (query_pos >= query_start)
        & (query_pos < query_end)
    )
    valid_lengths = jnp.where(
        query_valid,
        jnp.clip(
            seq_lens[safe_seq_id] - query_len + local_query_pos + 1,
            0,
            pages_per_seq * page_size,
        ),
        0,
    )

    logical = jnp.maximum(logical_topk, 0)
    logical_page = logical // page_size
    page_table_start = cu_kv_lens[safe_seq_id] // page_size
    page_ptr = page_table_start[:, None] + logical_page
    page_in_bounds = (
        (logical_page < pages_per_seq) & (page_ptr >= 0) & (page_ptr < page_indices.shape[0])
    )
    safe_page_ptr = jnp.clip(page_ptr, 0, page_indices.shape[0] - 1)
    physical_page = page_indices[safe_page_ptr]
    valid = (
        (logical_topk >= 0)
        & query_valid[:, None]
        & page_in_bounds
        & (logical < valid_lengths[:, None])
        & (physical_page >= 0)
    )
    physical_slot = physical_page * page_size + logical % page_size
    return jnp.where(valid, physical_slot, -1).astype(jnp.int32)


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

    # XLA may clone the enclosed calls for ping, pong, and drain branches.
    # Keep every clone under one stable scope so XProf presents them as one
    # score/Top-K pipeline instead of unrelated while/conditional operations.
    with jax.named_scope("ScoreTopKPipeline"):
        return jax.lax.cond(
            num_tiles > 0,
            run_nonempty,
            lambda pipeline_out: pipeline_out,
            out,
        )


def _pipeline_score_topk_and_mapping_tiles(
    num_tiles: jax.Array,
    score_tile: Callable[[jax.Array], jax.Array],
    score_and_map_tile: Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
    select_tile: Callable[[jax.Array, jax.Array], jax.Array],
    map_tile: Callable[[jax.Array, jax.Array], jax.Array],
    write_tile: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    out: jax.Array,
) -> jax.Array:
    """Pipeline TensorCore Score/Mapping with pure SparseCore Top-K.

    Score and Top-K exchange two score banks. Top-K and Mapping exchange two
    logical-index banks. In steady state, one TensorCore program scores tile
    ``i + 1`` and maps tile ``i - 1`` while SparseCore selects tile ``i``.
    """

    def run_nonempty(pipeline_out):
        score_buffer_0 = score_tile(jnp.int32(0))

        def run_single(single_out):
            logical = select_tile(jnp.int32(0), score_buffer_0)
            physical = map_tile(jnp.int32(0), logical)
            return write_tile(jnp.int32(0), physical, single_out)

        def run_multiple(multi_out):
            # Fill both producer/consumer edges. These calls are independent:
            # TensorCore scores tile 1 while SparseCore selects tile 0.
            score_buffer_1 = score_tile(jnp.int32(1))
            logical_buffer_0 = select_tile(jnp.int32(0), score_buffer_0)
            logical_buffer_1 = jnp.empty_like(logical_buffer_0)

            def pipeline_body(tile_id, pipeline_carry):
                scores_0, scores_1, logical_0, logical_1, current_out = pipeline_carry

                def odd_step(buffers):
                    scores_0, scores_1, logical_0, logical_1, current_out = buffers
                    next_scores, mapped_previous = score_and_map_tile(
                        tile_id + 1,
                        tile_id - 1,
                        logical_0,
                    )
                    current_logical = select_tile(tile_id, scores_1)
                    current_out = write_tile(tile_id - 1, mapped_previous, current_out)
                    return next_scores, scores_1, logical_0, current_logical, current_out

                def even_step(buffers):
                    scores_0, scores_1, logical_0, logical_1, current_out = buffers
                    next_scores, mapped_previous = score_and_map_tile(
                        tile_id + 1,
                        tile_id - 1,
                        logical_1,
                    )
                    current_logical = select_tile(tile_id, scores_0)
                    current_out = write_tile(tile_id - 1, mapped_previous, current_out)
                    return scores_0, next_scores, current_logical, logical_1, current_out

                return jax.lax.cond(
                    jax.lax.bitwise_and(tile_id, 1) == 1,
                    odd_step,
                    even_step,
                    pipeline_carry,
                )

            (
                score_buffer_0_final,
                score_buffer_1_final,
                logical_buffer_0_final,
                logical_buffer_1_final,
                multi_out,
            ) = jax.lax.fori_loop(
                1,
                num_tiles - 1,
                pipeline_body,
                (
                    score_buffer_0,
                    score_buffer_1,
                    logical_buffer_0,
                    logical_buffer_1,
                    multi_out,
                ),
            )

            final_tile = num_tiles - 1

            def drain_odd(buffers):
                scores_0, scores_1, logical_0, _, current_out = buffers
                # Mapping tile N-2 and Top-K tile N-1 use different engines.
                mapped_previous = map_tile(final_tile - 1, logical_0)
                final_logical = select_tile(final_tile, scores_1)
                current_out = write_tile(final_tile - 1, mapped_previous, current_out)
                return final_logical, current_out

            def drain_even(buffers):
                scores_0, _, _, logical_1, current_out = buffers
                mapped_previous = map_tile(final_tile - 1, logical_1)
                final_logical = select_tile(final_tile, scores_0)
                current_out = write_tile(final_tile - 1, mapped_previous, current_out)
                return final_logical, current_out

            final_logical, multi_out = jax.lax.cond(
                jax.lax.bitwise_and(final_tile, 1) == 1,
                drain_odd,
                drain_even,
                (
                    score_buffer_0_final,
                    score_buffer_1_final,
                    logical_buffer_0_final,
                    logical_buffer_1_final,
                    multi_out,
                ),
            )
            final_physical = map_tile(final_tile, final_logical)
            return write_tile(final_tile, final_physical, multi_out)

        return jax.lax.cond(
            num_tiles == 1,
            run_single,
            run_multiple,
            pipeline_out,
        )

    with jax.named_scope("ScoreTopKMappingPipeline"):
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
    output_physical_slots: bool,
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
                persistent_two_seq=T >= 2 and T % 2 == 0,
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

    with jax.named_scope("Decode"):
        valid_lengths = jnp.where(
            jnp.arange(T, dtype=jnp.int32) < active_num_seqs,
            jnp.clip(seq_lens, 0, max_kv),
            0,
        )
        if output_physical_slots:
            page_offsets = jnp.arange(pages_per_seq, dtype=jnp.int32)
            page_starts = cu_kv_lens[:-1] // page_size
            page_ptrs = page_starts[:, None] + page_offsets[None, :]
            page_ptr_valid = (page_ptrs >= 0) & (page_ptrs < page_indices.shape[0])
            safe_page_ptrs = jnp.clip(page_ptrs, 0, page_indices.shape[0] - 1)
            seq_pages = page_indices[safe_page_ptrs]
            seq_pages = jnp.where(page_ptr_valid, seq_pages, -1)
        selection_scope = "RadixTopK" if topk_impl == "radix" else f"TopK_{topk_impl}"
        with jax.named_scope(selection_scope):
            selected_topk = _select_topk_indices(
                batched_scores,
                valid_lengths,
                k=k,
                topk_impl=topk_impl,
            )
        if not output_physical_slots:
            return selected_topk
        # Decode has one page table per score row. SparseCore currently supports
        # only a shared rank-1 page table in its fused output epilogue, so keep
        # this small [T, K] mapping adjacent to selection without materializing
        # a full [T, max_kv] slot LUT.
        with jax.named_scope("LogicalToPhysicalSlots"):
            return _map_logical_topk_to_physical_slots(
                selected_topk,
                seq_pages,
                valid_lengths,
                page_size,
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
    output_physical_slots: bool,
    score_query_block_size: int = _INDEXER_QUERY_BLOCK_SIZE,
) -> jax.Array:
    """Score packed extend queries through a bounded score/top-k pipeline.

    TPU query blocks follow the sequence page table directly and double-buffer
    bounded key tiles in VMEM. Other backends gather each sequence's keys once
    as a correctness fallback. Fixed-size score tiles bound peak memory and
    allow scoring block ``n + 1`` to overlap top-k selection for block ``n``.
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
    use_paged_score = (
        jax.default_backend() == "tpu"
        and D % 128 == 0
        and max_kv % 128 == 0
        and page_size <= 2048
        and query_block_size <= _INDEXER_QUERY_BLOCK_SIZE
        and q_idx.dtype in (jnp.bfloat16, jnp.float32)
        and idx_weights.dtype in (jnp.bfloat16, jnp.float32)
        and index_key_cache.dtype in (jnp.bfloat16, jnp.float32)
    )
    score_tile_scope = "PagedScoreTile" if use_paged_score else "ScoreTile"
    use_score_topk_mapping_pipeline = (
        use_paged_score and output_physical_slots and topk_impl == "radix" and k % 128 == 0
    )
    pipeline_scope = (
        "ScoreTopKMappingPipeline" if use_score_topk_mapping_pipeline else "ScoreTopKPipeline"
    )
    q_idx_padded = jnp.pad(q_idx, ((0, query_block_size), (0, 0), (0, 0)))
    idx_weights_padded = jnp.pad(idx_weights_f32, ((0, query_block_size), (0, 0)))
    out_padded = jnp.full((T + query_block_size, k), -1, dtype=jnp.int32)

    def body(seq_id, packed_out):
        q_start = cu_q_lens[seq_id]
        q_end = cu_q_lens[seq_id + 1]
        q_len = jnp.maximum(q_end - q_start, 0)
        kv_len = seq_lens[seq_id]
        with jax.named_scope("Extend"), jax.named_scope("IndexKeyPages"):
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
            )
            if not use_paged_score:
                seq_k_idx = index_key_cache[seq_pages].reshape(max_kv, D)

        num_query_blocks = (q_len + query_block_size - 1) // query_block_size

        def block_inputs(block_id):
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
            valid_lengths = jnp.where(
                query_valid,
                jnp.clip(kv_len - q_len + local_q_pos + 1, 0, max_kv),
                0,
            )
            return q_block, weights_block, valid_lengths

        def score_block(block_id):
            with (
                jax.named_scope("Extend"),
                jax.named_scope(pipeline_scope),
                jax.named_scope(score_tile_scope),
            ):
                q_block, weights_block, valid_lengths = block_inputs(block_id)
                if use_paged_score:
                    scores_block = paged_extend_score_block_pallas(
                        q_block,
                        weights_block,
                        index_key_cache,
                        valid_lengths,
                        seq_pages,
                        pages_per_seq=pages_per_seq,
                        coalesce_page_dma=True,
                    )
                else:
                    scores_block = _compute_score_tile(
                        q_block,
                        weights_block,
                        seq_k_idx,
                    )
                    mask = kv_pos[None, :] < valid_lengths[:, None]
                    scores_block = jnp.where(mask, scores_block, _NEG_INF)
                return scores_block

        def score_and_map_block(score_block_id, mapping_block_id, logical_topk):
            q_block, weights_block, score_valid_lengths = block_inputs(score_block_id)
            _, _, mapping_valid_lengths = block_inputs(mapping_block_id)
            with (
                jax.named_scope("Extend"),
                jax.named_scope(pipeline_scope),
                jax.named_scope("PagedScoreAndPhysicalMappingTile"),
            ):
                return paged_extend_score_and_map_block_pallas(
                    q_block,
                    weights_block,
                    index_key_cache,
                    score_valid_lengths,
                    seq_pages,
                    logical_topk,
                    mapping_valid_lengths,
                    pages_per_seq=pages_per_seq,
                    coalesce_page_dma=True,
                )

        def select_block(block_id, scores_block):
            # The adapter owns candidate selection. Radix candidates remain
            # unordered; invalid/padded entries are compacted behind them.
            _, _, valid_lengths = block_inputs(block_id)
            selection_scope = "RadixTopK" if topk_impl == "radix" else f"TopK_{topk_impl}"
            with (
                jax.named_scope("Extend"),
                jax.named_scope(pipeline_scope),
                jax.named_scope(selection_scope),
            ):
                return _select_topk_indices(
                    scores_block,
                    valid_lengths,
                    k=k,
                    topk_impl=topk_impl,
                )

        def map_block(block_id, logical_topk):
            _, _, valid_lengths = block_inputs(block_id)
            with (
                jax.named_scope("Extend"),
                jax.named_scope(pipeline_scope),
                jax.named_scope("LogicalToPhysicalSlotsTile"),
            ):
                return _map_logical_topk_to_physical_slots(
                    logical_topk,
                    seq_pages,
                    valid_lengths,
                    page_size,
                )

        def write_block(block_id, block_indices, block_out):
            block_start = q_start + block_id * query_block_size
            with (
                jax.named_scope("Extend"),
                jax.named_scope(pipeline_scope),
                jax.named_scope("TopKOutputUpdate"),
            ):
                return jax.lax.dynamic_update_slice_in_dim(
                    block_out,
                    block_indices,
                    block_start,
                    axis=0,
                )

        def select_and_write_block(block_id, scores_block, block_out):
            return write_block(
                block_id,
                select_block(block_id, scores_block),
                block_out,
            )

        with jax.named_scope("Extend"):
            if use_score_topk_mapping_pipeline:
                return _pipeline_score_topk_and_mapping_tiles(
                    num_query_blocks,
                    score_block,
                    score_and_map_block,
                    select_block,
                    map_block,
                    write_block,
                    packed_out,
                )
            return _pipeline_score_and_topk_tiles(
                num_query_blocks,
                score_block,
                select_and_write_block,
                packed_out,
            )

    out_padded = jax.lax.fori_loop(0, active_num_seqs, body, out_padded)
    selected_topk = out_padded[:T]
    if not output_physical_slots or use_score_topk_mapping_pipeline:
        return selected_topk

    # Keep the pure selector pure: finish and drain every score/Top-K block
    # before resolving the packed logical result through the page tables.
    with jax.named_scope("Extend"), jax.named_scope("LogicalToPhysicalSlots"):
        return _map_packed_logical_topk_to_physical_slots(
            selected_topk,
            page_indices,
            seq_lens,
            cu_q_lens,
            cu_kv_lens,
            active_num_seqs,
            pages_per_seq=pages_per_seq,
            page_size=page_size,
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "pages_per_seq",
        "one_token_per_seq",
        "topk_impl",
        "output_physical_slots",
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
    output_physical_slots: bool = False,
    score_query_block_size: int = _INDEXER_QUERY_BLOCK_SIZE,
) -> jax.Array:
    """Score indexer queries and select top-k token positions.

    ``one_token_per_seq=True`` selects the decode implementation: it scores
    each sequence without materializing ``[T, max_kv, D]`` keys, then submits
    the resulting ``[T, max_kv]`` score matrix to one batched top-k call.

    Otherwise the extend implementation gathers each sequence's keys once and
    streams its query blocks through the score/top-k ping-pong pipeline.

    By default the result contains sequence-local logical positions. When
    ``output_physical_slots=True``, selection also resolves those positions to
    flattened KV-cache slots; radix extend does this inside its SparseCore
    output epilogue.
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
            output_physical_slots=output_physical_slots,
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
        output_physical_slots=output_physical_slots,
        score_query_block_size=score_query_block_size,
    )
