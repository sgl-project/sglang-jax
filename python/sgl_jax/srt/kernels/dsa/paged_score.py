"""Paged-cache Pallas scorers for DSA decode and extend query blocks."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG_INF = float("-inf")
_DEFAULT_BLOCK_K = 22528
_DEFAULT_EXTEND_BLOCK_K = 2048
_DECODE_SEQS_PER_GROUP = 2
_MAX_MAPPING_PAGE_RUNS = 2


def _resolve_block_k(max_kv: int, page_size: int, requested_block_k: int) -> int:
    """Choose the largest TPU-aligned divisor at or below the requested tile."""
    candidate = min(max_kv, requested_block_k)
    candidate -= candidate % 128
    while candidate >= 128:
        if max_kv % candidate == 0 and candidate % page_size == 0:
            return candidate
        candidate -= 128
    raise ValueError(
        f"cannot tile max_kv={max_kv} with page_size={page_size} into 128-aligned key blocks"
    )


def _weighted_relu_scores(
    query: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    *,
    batched_keys: bool,
) -> jax.Array:
    """Apply the common DSA score formula for one resident key tile.

    ``query`` is either ``[H, D]`` or ``[Bq, H, D]``. Shared keys have shape
    ``[Bk, D]``; batched keys have shape ``[Bq, Bk, D]``. Keeping this
    mathematical core shared lets decode and extend specialize only their page
    DMA and launch schedules.
    """

    if query.ndim == 2:
        if batched_keys or keys.ndim != 2 or weights.ndim != 1:
            raise ValueError(
                "a rank-2 query requires rank-2 shared keys and rank-1 weights, "
                f"got query={query.shape}, keys={keys.shape}, weights={weights.shape}"
            )
        similarities = lax.dot_general(
            query,
            keys,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        return lax.dot_general(
            weights.astype(jnp.float32),
            jnp.maximum(similarities, jnp.float32(0.0)),
            dimension_numbers=(((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    if query.ndim != 3 or weights.ndim != 2:
        raise ValueError(
            "a query block requires query [Bq, H, D] and weights [Bq, H], "
            f"got query={query.shape}, weights={weights.shape}"
        )
    if batched_keys:
        if keys.ndim != 3 or keys.shape[0] != query.shape[0]:
            raise ValueError(
                "batched keys must have shape [Bq, Bk, D], "
                f"got query={query.shape}, keys={keys.shape}"
            )
        similarities = lax.dot_general(
            query,
            keys,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
    else:
        if keys.ndim != 2:
            raise ValueError(f"shared keys must have shape [Bk, D], got {keys.shape}")
        similarities = lax.dot_general(
            query,
            keys,
            dimension_numbers=(((2,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    return lax.dot_general(
        weights.astype(jnp.float32),
        jnp.maximum(similarities, jnp.float32(0.0)),
        dimension_numbers=(((1,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def _paged_decode_score_kernel(
    seq_lens_ref,  # i32[S], SMEM
    page_indices_ref,  # i32[S * pages_per_seq], SMEM
    cu_kv_lens_ref,  # i32[S + 1], SMEM
    q_vmem_ref,  # f[1, H, D], VMEM
    weights_vmem_ref,  # f[1, 1, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    scores_out_vmem_ref,  # f32[1, 1, max_kv], VMEM
    keys_x2_vmem_ref,  # f[2, block_k, D] or f[2, pages_per_block, page_size, D]
    key_dma_sems,  # DMA semaphore [2]
    *,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
):
    """Score one sequence while double-buffering paged key DMA."""
    seq_id = pl.program_id(0)
    block_k = pages_per_block * page_size if coalesce_page_dma else keys_x2_vmem_ref.shape[1]
    page_table_start = cu_kv_lens_ref[seq_id] // page_size
    kv_len = seq_lens_ref[seq_id]

    def fetch_key_block(block_id, buffer, *, wait: bool):
        dst = keys_x2_vmem_ref.at[buffer]
        sem = key_dma_sems.at[buffer]
        if wait:
            pltpu.make_async_copy(dst, dst, sem).wait()
            return

        first_page = page_table_start + block_id * pages_per_block

        def fetch_page(page_in_block, _):
            physical_page = page_indices_ref[first_page + page_in_block]
            if coalesce_page_dma:
                page_dst = dst.at[page_in_block]
            else:
                page_dst = dst.at[pl.ds(page_in_block * page_size, page_size)]
            pltpu.make_async_copy(
                cache_hbm_ref.at[physical_page],
                page_dst,
                sem,
            ).start()
            return None

        if coalesce_page_dma:
            first_physical_page = page_indices_ref[first_page]
            pltpu.make_async_copy(
                cache_hbm_ref.at[pl.ds(first_physical_page, pages_per_block)],
                dst,
                sem,
            ).start()
        else:
            lax.fori_loop(0, pages_per_block, fetch_page, None, unroll=False)

    # Prologue: seed the first key buffer. Query and head weights are already
    # resident in per-program VMEM through their BlockSpecs.
    fetch_key_block(0, 0, wait=False)

    def score_block(block_id, _):
        buffer = block_id % 2
        next_buffer = 1 - buffer

        fetch_key_block(block_id, buffer, wait=True)

        @pl.when(block_id + 1 < num_k_blocks)
        def prefetch_next_keys():
            fetch_key_block(block_id + 1, next_buffer, wait=False)

        # Mosaic's matmul lowering does not reliably support mixed FP32/BF16
        # operands.  The production query is FP32 after the Hadamard transform,
        # while the paged index-key cache is BF16, so make the promotion that
        # jnp.einsum applies in the reference path explicit here.
        if first_dot_bf16:
            # Performance ablation: keep both MXU operands in BF16 while
            # accumulating into FP32. This avoids promoting the complete
            # [block_k, D] key tile to FP32, but may change candidates near the
            # top-k boundary when the production query arrives in FP32.
            query = q_vmem_ref[0].astype(jnp.bfloat16)
            keys = keys_x2_vmem_ref[buffer]
            if coalesce_page_dma:
                keys = keys.reshape(block_k, keys_x2_vmem_ref.shape[-1])
            keys = keys.astype(jnp.bfloat16)
        else:
            query = q_vmem_ref[0]
            keys = keys_x2_vmem_ref[buffer]
            if coalesce_page_dma:
                keys = keys.reshape(block_k, keys_x2_vmem_ref.shape[-1])
            keys = keys.astype(query.dtype)
        # Match the two reference einsums exactly. A plain FP32 multiply plus
        # reduce for the head mixture uses a different TPU numerical path and
        # can perturb candidates around the top-k boundary.
        scores = _weighted_relu_scores(
            query,
            weights_vmem_ref[0, 0].astype(jnp.float32),
            keys,
            batched_keys=False,
        )
        positions = block_id * block_k + jnp.arange(block_k, dtype=jnp.int32)
        scores_out_vmem_ref[0, 0, pl.ds(block_id * block_k, block_k)] = jnp.where(
            positions < kv_len,
            scores,
            _NEG_INF,
        )
        return None

    lax.fori_loop(0, num_k_blocks, score_block, None, unroll=False)


def _paged_decode_score_two_seq_kernel(
    seq_lens_ref,  # i32[S], SMEM
    page_indices_ref,  # i32[S * pages_per_seq], SMEM
    cu_kv_lens_ref,  # i32[S + 1], SMEM
    q_vmem_ref,  # f[2, H, D], VMEM
    weights_vmem_ref,  # f[2, 1, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    scores_out_vmem_ref,  # f32[2, 1, max_kv], VMEM
    keys_vmem_ref,  # flat or page-shaped two-sequence ping-pong buffers
    key_dma_sems,  # DMA semaphore [2 sequences, 2 buffers]
    *,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
):
    """Score one pair of sequences with a shared prefetch schedule."""
    seq_base = pl.program_id(0) * _DECODE_SEQS_PER_GROUP
    block_k = pages_per_block * page_size if coalesce_page_dma else keys_vmem_ref.shape[2]

    def fetch_key_block(seq_id, block_id, buffer, *, wait: bool):
        dst = keys_vmem_ref.at[seq_id, buffer]
        sem = key_dma_sems.at[seq_id, buffer]
        if wait:
            pltpu.make_async_copy(dst, dst, sem).wait()
            return

        global_seq_id = seq_base + seq_id
        page_table_start = cu_kv_lens_ref[global_seq_id] // page_size
        first_page = page_table_start + block_id * pages_per_block

        def fetch_page(page_in_block, _):
            physical_page = page_indices_ref[first_page + page_in_block]
            if coalesce_page_dma:
                page_dst = dst.at[page_in_block]
            else:
                page_dst = dst.at[pl.ds(page_in_block * page_size, page_size)]
            pltpu.make_async_copy(
                cache_hbm_ref.at[physical_page],
                page_dst,
                sem,
            ).start()
            return None

        if coalesce_page_dma:
            first_physical_page = page_indices_ref[first_page]
            pltpu.make_async_copy(
                cache_hbm_ref.at[pl.ds(first_physical_page, pages_per_block)],
                dst,
                sem,
            ).start()
        else:
            lax.fori_loop(0, pages_per_block, fetch_page, None, unroll=False)

    def score_key_block_pair(block_id, buffer):
        if first_dot_bf16:
            query = q_vmem_ref[...].astype(jnp.bfloat16)
            keys = keys_vmem_ref[:, buffer]
            if coalesce_page_dma:
                keys = keys.reshape(2, block_k, keys_vmem_ref.shape[-1])
            keys = keys.astype(jnp.bfloat16)
        else:
            query = q_vmem_ref[...]
            keys = keys_vmem_ref[:, buffer]
            if coalesce_page_dma:
                keys = keys.reshape(2, block_k, keys_vmem_ref.shape[-1])
            keys = keys.astype(query.dtype)
        scores = _weighted_relu_scores(
            query,
            weights_vmem_ref[:, 0].astype(jnp.float32),
            keys,
            batched_keys=True,
        )
        positions = block_id * block_k + jnp.arange(block_k, dtype=jnp.int32)
        for seq_id in range(2):
            scores_out_vmem_ref[
                seq_id,
                0,
                pl.ds(block_id * block_k, block_k),
            ] = jnp.where(
                positions < seq_lens_ref[seq_base + seq_id],
                scores[seq_id],
                _NEG_INF,
            )

    # Seed both current buffers. Both page streams can make progress before
    # either sequence reaches its first wait.
    for seq_id in range(2):
        fetch_key_block(seq_id, 0, 0, wait=False)

    def score_block_pair(block_id, _):
        buffer = block_id % 2
        next_buffer = 1 - buffer

        # Make both key blocks resident, then present both sequences to Mosaic
        # as one pair of batched matrix products. This gives the compiler an
        # explicit batch axis instead of two sequential H=32 dot operations.
        for seq_id in range(2):
            fetch_key_block(seq_id, block_id, buffer, wait=True)

            @pl.when(block_id + 1 < num_k_blocks)
            def prefetch_next_keys(seq_id=seq_id):
                fetch_key_block(seq_id, block_id + 1, next_buffer, wait=False)

        score_key_block_pair(block_id, buffer)
        return None

    lax.fori_loop(0, num_k_blocks, score_block_pair, None, unroll=False)


def _score_paged_extend_query_block(
    valid_lengths_ref,  # i32[Bq], SMEM
    page_indices_ref,  # i32[pages_per_seq], SMEM
    q_vmem_ref,  # f[Bq, H, D], VMEM
    weights_vmem_ref,  # f[Bq, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    scores_out_vmem_ref,  # f32[Bq, max_kv], VMEM
    keys_x2_vmem_ref,  # f[2, block_k, D] or f[2, pages_per_block, page_size, D]
    key_dma_sems,  # DMA semaphore [2]
    *,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
):
    """Score one extend query block against one shared paged key stream."""

    block_k = pages_per_block * page_size if coalesce_page_dma else keys_x2_vmem_ref.shape[1]

    def fetch_key_block(block_id, buffer, *, wait: bool):
        dst = keys_x2_vmem_ref.at[buffer]
        sem = key_dma_sems.at[buffer]
        if wait:
            pltpu.make_async_copy(dst, dst, sem).wait()
            return

        first_page = block_id * pages_per_block

        def fetch_page(page_in_block, _):
            physical_page = page_indices_ref[first_page + page_in_block]
            if coalesce_page_dma:
                page_dst = dst.at[page_in_block]
            else:
                page_dst = dst.at[pl.ds(page_in_block * page_size, page_size)]
            pltpu.make_async_copy(
                cache_hbm_ref.at[physical_page],
                page_dst,
                sem,
            ).start()
            return None

        if coalesce_page_dma:
            first_physical_page = page_indices_ref[first_page]
            pltpu.make_async_copy(
                cache_hbm_ref.at[pl.ds(first_physical_page, pages_per_block)],
                dst,
                sem,
            ).start()
        else:
            lax.fori_loop(0, pages_per_block, fetch_page, None, unroll=False)

    fetch_key_block(0, 0, wait=False)

    def score_block(block_id, _):
        buffer = block_id % 2
        next_buffer = 1 - buffer
        fetch_key_block(block_id, buffer, wait=True)

        @pl.when(block_id + 1 < num_k_blocks)
        def prefetch_next_keys():
            fetch_key_block(block_id + 1, next_buffer, wait=False)

        keys = keys_x2_vmem_ref[buffer]
        if coalesce_page_dma:
            keys = keys.reshape(block_k, keys_x2_vmem_ref.shape[-1])
        if first_dot_bf16:
            query = q_vmem_ref[...].astype(jnp.bfloat16)
            keys = keys.astype(jnp.bfloat16)
        else:
            query = q_vmem_ref[...]
            keys = keys.astype(query.dtype)
        scores = _weighted_relu_scores(
            query,
            weights_vmem_ref[...],
            keys,
            batched_keys=False,
        )
        positions = block_id * block_k + jnp.arange(block_k, dtype=jnp.int32)
        # Mosaic SMEM supports scalar loads only. Keep score construction
        # batched, then use static row stores so each causal length is loaded
        # as one scalar instead of materializing valid_lengths[Bq] in VMEM.
        for query_id in range(q_vmem_ref.shape[0]):
            valid = positions < valid_lengths_ref[query_id]
            scores_out_vmem_ref[query_id, pl.ds(block_id * block_k, block_k)] = jnp.where(
                valid,
                scores[query_id],
                _NEG_INF,
            )
        return None

    lax.fori_loop(0, num_k_blocks, score_block, None, unroll=False)


def _paged_extend_score_block_kernel(
    valid_lengths_ref,  # i32[Bq], SMEM
    page_indices_ref,  # i32[pages_per_seq], SMEM
    q_vmem_ref,  # f[Bq, H, D], VMEM
    weights_vmem_ref,  # f[Bq, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    scores_out_vmem_ref,  # f32[Bq, max_kv], VMEM
    keys_x2_vmem_ref,  # f[2, block_k, D] or f[2, pages_per_block, page_size, D]
    key_dma_sems,  # DMA semaphore [2]
    *,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
):
    _score_paged_extend_query_block(
        valid_lengths_ref,
        page_indices_ref,
        q_vmem_ref,
        weights_vmem_ref,
        cache_hbm_ref,
        scores_out_vmem_ref,
        keys_x2_vmem_ref,
        key_dma_sems,
        page_size=page_size,
        pages_per_block=pages_per_block,
        num_k_blocks=num_k_blocks,
        first_dot_bf16=first_dot_bf16,
        coalesce_page_dma=coalesce_page_dma,
    )


def _paged_extend_score_and_map_block_kernel(
    score_valid_lengths_ref,  # i32[Bq], SMEM
    mapping_valid_lengths_ref,  # i32[Bq], SMEM
    score_page_indices_ref,  # i32[pages_per_seq], SMEM
    mapping_run_starts_ref,  # i32[max_page_runs], SMEM
    mapping_run_bases_ref,  # i32[max_page_runs], SMEM
    mapping_num_runs_ref,  # i32[1], SMEM
    q_vmem_ref,  # f[Bq, H, D], VMEM
    weights_vmem_ref,  # f[Bq, H], VMEM
    cache_hbm_ref,  # f[P, page_size, D], HBM
    logical_topk_vmem_ref,  # i32[Bq, K], VMEM
    scores_out_vmem_ref,  # f32[Bq, max_kv], VMEM
    physical_slots_out_vmem_ref,  # i32[Bq, K], VMEM
    keys_x2_vmem_ref,  # f[2, block_k, D] or f[2, pages_per_block, page_size, D]
    key_dma_sems,  # DMA semaphore [2]
    *,
    page_size: int,
    pages_per_block: int,
    num_k_blocks: int,
    first_dot_bf16: bool,
    coalesce_page_dma: bool,
):
    """Map the preceding Top-K block while scoring the current query block.

    TensorCore Pallas cannot perform vector-indexed VMEM loads. The caller
    therefore compresses the valid page table into a small set of contiguous
    runs. Mapping becomes vector comparisons plus affine address arithmetic,
    which Mosaic can schedule beside key DMA and MXU work.
    """

    num_pages = score_page_indices_ref.shape[0]
    logical = logical_topk_vmem_ref[...]
    safe_logical = jnp.maximum(logical, 0)
    logical_page = safe_logical // page_size
    page_in_bounds = logical_page < num_pages
    physical_page = mapping_run_bases_ref[0] + logical_page - mapping_run_starts_ref[0]
    physical_page = jnp.where(
        mapping_num_runs_ref[0] > 0,
        physical_page,
        jnp.int32(-1),
    )
    for run_id in range(1, mapping_run_starts_ref.shape[0]):
        run_start = mapping_run_starts_ref[run_id]
        run_base = mapping_run_bases_ref[run_id]
        use_run = (run_id < mapping_num_runs_ref[0]) & (logical_page >= run_start)
        physical_page = jnp.where(
            use_run,
            run_base + logical_page - run_start,
            physical_page,
        )
    mapping_valid_lengths = jnp.stack(
        [mapping_valid_lengths_ref[query_id] for query_id in range(logical_topk_vmem_ref.shape[0])]
    )
    valid = (
        (logical >= 0)
        & page_in_bounds
        & (safe_logical < mapping_valid_lengths[:, None])
        & (physical_page >= 0)
    )
    physical_slot = physical_page * page_size + safe_logical % page_size
    physical_slots_out_vmem_ref[...] = jnp.where(
        valid,
        physical_slot,
        jnp.int32(-1),
    )

    _score_paged_extend_query_block(
        score_valid_lengths_ref,
        score_page_indices_ref,
        q_vmem_ref,
        weights_vmem_ref,
        cache_hbm_ref,
        scores_out_vmem_ref,
        keys_x2_vmem_ref,
        key_dma_sems,
        page_size=page_size,
        pages_per_block=pages_per_block,
        num_k_blocks=num_k_blocks,
        first_dot_bf16=first_dot_bf16,
        coalesce_page_dma=coalesce_page_dma,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "coalesce_page_dma",
        "interpret",
    ),
)
def _paged_extend_score_block_pallas_impl(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    valid_lengths: jax.Array,
    page_indices: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_EXTEND_BLOCK_K,
    first_dot_bf16: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Score a fixed query block directly from one sequence's page table."""

    if q_idx.ndim != 3:
        raise ValueError(f"q_idx must have shape [Bq, H, D], got {q_idx.shape}")
    if idx_weights.shape != q_idx.shape[:2]:
        raise ValueError(f"idx_weights must have shape {q_idx.shape[:2]}, got {idx_weights.shape}")
    if index_key_cache.ndim != 3:
        raise ValueError(
            f"index_key_cache must have shape [P, page_size, D], got {index_key_cache.shape}"
        )
    num_queries, _, head_dim = q_idx.shape
    if valid_lengths.shape != (num_queries,):
        raise ValueError(
            f"valid_lengths must have shape {(num_queries,)}, got {valid_lengths.shape}"
        )
    if page_indices.shape != (pages_per_seq,):
        raise ValueError(
            f"page_indices must have shape {(pages_per_seq,)}, got {page_indices.shape}"
        )
    if index_key_cache.shape[-1] != head_dim:
        raise ValueError(
            f"cache head dim {index_key_cache.shape[-1]} does not match q head dim {head_dim}"
        )
    if head_dim % 128:
        raise ValueError(f"head_dim must be divisible by 128, got {head_dim}")
    if pages_per_seq < 1:
        raise ValueError(f"pages_per_seq must be positive, got {pages_per_seq}")
    for name, value in (("valid_lengths", valid_lengths), ("page_indices", page_indices)):
        if value.dtype != jnp.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")

    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    if block_k < 128 or block_k % 128:
        raise ValueError(f"block_k must be a positive multiple of 128, got {block_k}")
    block_k = _resolve_block_k(max_kv, page_size, block_k)
    pages_per_block = block_k // page_size
    num_k_blocks = max_kv // block_k

    key_shape = (
        (2, pages_per_block, page_size, head_dim) if coalesce_page_dma else (2, block_k, head_dim)
    )
    scores = pl.pallas_call(
        functools.partial(
            _paged_extend_score_block_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
        ),
        out_shape=jax.ShapeDtypeStruct((num_queries, max_kv), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(q_idx.shape, lambda *_: (0, 0, 0)),
                pl.BlockSpec(idx_weights.shape, lambda *_: (0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (num_queries, max_kv),
                lambda *_: (0, 0),
            ),
            scratch_shapes=[
                pltpu.VMEM(key_shape, index_key_cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "dsa_paged_extend_score_block_contiguous"
            if coalesce_page_dma
            else "dsa_paged_extend_score_block_paged"
        ),
    )(
        valid_lengths,
        page_indices,
        q_idx,
        idx_weights,
        index_key_cache,
    )
    return scores


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "coalesce_page_dma",
        "interpret",
    ),
)
def _paged_extend_score_and_map_block_pallas_impl(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    score_valid_lengths: jax.Array,
    page_indices: jax.Array,
    logical_topk: jax.Array,
    mapping_valid_lengths: jax.Array,
    mapping_run_starts: jax.Array,
    mapping_run_bases: jax.Array,
    mapping_num_runs: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_EXTEND_BLOCK_K,
    first_dot_bf16: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Score one block and map a preceding block's logical Top-K positions."""

    if q_idx.ndim != 3:
        raise ValueError(f"q_idx must have shape [Bq, H, D], got {q_idx.shape}")
    if idx_weights.shape != q_idx.shape[:2]:
        raise ValueError(f"idx_weights must have shape {q_idx.shape[:2]}, got {idx_weights.shape}")
    if index_key_cache.ndim != 3:
        raise ValueError(
            f"index_key_cache must have shape [P, page_size, D], got {index_key_cache.shape}"
        )
    num_queries, _, head_dim = q_idx.shape
    if score_valid_lengths.shape != (num_queries,):
        raise ValueError(
            f"score_valid_lengths must have shape {(num_queries,)}, got {score_valid_lengths.shape}"
        )
    if logical_topk.ndim != 2 or logical_topk.shape[0] != num_queries:
        raise ValueError(f"logical_topk must have shape [Bq, K], got {logical_topk.shape}")
    if mapping_valid_lengths.shape != (num_queries,):
        raise ValueError(
            "mapping_valid_lengths must have shape "
            f"{(num_queries,)}, got {mapping_valid_lengths.shape}"
        )
    mapping_run_capacity = mapping_run_starts.shape[0]
    if not 1 <= mapping_run_capacity <= _MAX_MAPPING_PAGE_RUNS:
        raise ValueError(
            "mapping_run_starts capacity must be in "
            f"[1, {_MAX_MAPPING_PAGE_RUNS}], got {mapping_run_starts.shape}"
        )
    if mapping_run_bases.shape != mapping_run_starts.shape:
        raise ValueError(
            "mapping_run_bases must match mapping_run_starts, got "
            f"{mapping_run_bases.shape} and {mapping_run_starts.shape}"
        )
    if mapping_num_runs.shape != (1,):
        raise ValueError(f"mapping_num_runs must have shape (1,), got {mapping_num_runs.shape}")
    if page_indices.shape != (pages_per_seq,):
        raise ValueError(
            f"page_indices must have shape {(pages_per_seq,)}, got {page_indices.shape}"
        )
    if index_key_cache.shape[-1] != head_dim:
        raise ValueError(
            f"cache head dim {index_key_cache.shape[-1]} does not match q head dim {head_dim}"
        )
    if head_dim % 128:
        raise ValueError(f"head_dim must be divisible by 128, got {head_dim}")
    if pages_per_seq < 1:
        raise ValueError(f"pages_per_seq must be positive, got {pages_per_seq}")
    if logical_topk.shape[1] % 128:
        raise ValueError(
            "logical_topk K dimension must be divisible by 128 for the TPU vector mapping, "
            f"got {logical_topk.shape[1]}"
        )
    for name, value in (
        ("score_valid_lengths", score_valid_lengths),
        ("page_indices", page_indices),
        ("logical_topk", logical_topk),
        ("mapping_valid_lengths", mapping_valid_lengths),
        ("mapping_run_starts", mapping_run_starts),
        ("mapping_run_bases", mapping_run_bases),
        ("mapping_num_runs", mapping_num_runs),
    ):
        if value.dtype != jnp.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")

    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    if block_k < 128 or block_k % 128:
        raise ValueError(f"block_k must be a positive multiple of 128, got {block_k}")
    block_k = _resolve_block_k(max_kv, page_size, block_k)
    pages_per_block = block_k // page_size
    num_k_blocks = max_kv // block_k
    key_shape = (
        (2, pages_per_block, page_size, head_dim) if coalesce_page_dma else (2, block_k, head_dim)
    )
    scores, physical_slots = pl.pallas_call(
        functools.partial(
            _paged_extend_score_and_map_block_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((num_queries, max_kv), jnp.float32),
            jax.ShapeDtypeStruct(logical_topk.shape, jnp.int32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(q_idx.shape, lambda *_: (0, 0, 0)),
                pl.BlockSpec(idx_weights.shape, lambda *_: (0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(logical_topk.shape, lambda *_: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((num_queries, max_kv), lambda *_: (0, 0)),
                pl.BlockSpec(logical_topk.shape, lambda *_: (0, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM(key_shape, index_key_cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "dsa_paged_extend_score_map_contiguous"
            if coalesce_page_dma
            else "dsa_paged_extend_score_map_paged"
        ),
    )(
        score_valid_lengths,
        mapping_valid_lengths,
        page_indices,
        mapping_run_starts,
        mapping_run_bases,
        mapping_num_runs,
        q_idx,
        idx_weights,
        index_key_cache,
        logical_topk,
    )
    return scores, physical_slots


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "coalesce_page_dma",
        "interpret",
    ),
)
def paged_extend_score_and_map_block_pallas(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    score_valid_lengths: jax.Array,
    page_indices: jax.Array,
    logical_topk: jax.Array,
    mapping_valid_lengths: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_EXTEND_BLOCK_K,
    first_dot_bf16: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run current Score and preceding logical-to-physical mapping together.

    Page tables with at most ``_MAX_MAPPING_PAGE_RUNS`` contiguous physical
    runs use affine mapping inside the Score Pallas program. More fragmented
    tables retain exact behavior through a regular XLA gather next to Score.
    """

    kwargs = dict(
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        first_dot_bf16=first_dot_bf16,
        interpret=interpret,
    )

    page_size = index_key_cache.shape[1]
    num_valid_pages = jnp.clip(
        (jnp.max(mapping_valid_lengths) + page_size - 1) // page_size,
        0,
        pages_per_seq,
    )
    page_id = jnp.arange(pages_per_seq, dtype=jnp.int32)
    previous_page = jnp.concatenate((page_indices[:1] - 1, page_indices[:-1]))
    run_start_mask = (page_id < num_valid_pages) & (
        (page_id == 0) | (page_indices != previous_page + 1)
    )
    num_page_runs = jnp.sum(run_start_mask, dtype=jnp.int32)
    mapping_run_starts = jnp.nonzero(
        run_start_mask,
        size=_MAX_MAPPING_PAGE_RUNS,
        fill_value=0,
    )[0].astype(jnp.int32)
    mapping_run_bases = page_indices[mapping_run_starts]
    mapping_num_runs = jnp.minimum(num_page_runs, _MAX_MAPPING_PAGE_RUNS)[None]

    def run_compressed(run_capacity: int, *, contiguous_page_table: bool):
        return _paged_extend_score_and_map_block_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            score_valid_lengths,
            page_indices,
            logical_topk,
            mapping_valid_lengths,
            mapping_run_starts[:run_capacity],
            mapping_run_bases[:run_capacity],
            mapping_num_runs,
            coalesce_page_dma=contiguous_page_table,
            **kwargs,
        )

    def run_fallback():
        scores = paged_extend_score_block_pallas(
            q_idx,
            idx_weights,
            index_key_cache,
            score_valid_lengths,
            page_indices,
            coalesce_page_dma=coalesce_page_dma,
            **kwargs,
        )
        safe_logical = jnp.maximum(logical_topk, 0)
        logical_page = safe_logical // page_size
        page_in_bounds = logical_page < pages_per_seq
        physical_page = page_indices[jnp.clip(logical_page, 0, pages_per_seq - 1)]
        valid = (
            (logical_topk >= 0)
            & page_in_bounds
            & (safe_logical < mapping_valid_lengths[:, None])
            & (physical_page >= 0)
        )
        physical_slot = physical_page * page_size + safe_logical % page_size
        return scores, jnp.where(valid, physical_slot, -1).astype(jnp.int32)

    def run_compressed_with_score_layout(run_capacity: int):
        if not coalesce_page_dma:
            return run_compressed(run_capacity, contiguous_page_table=False)

        expected_pages = page_indices[:1] + jnp.arange(pages_per_seq, dtype=jnp.int32)
        page_table_in_bounds = jnp.all(
            (page_indices >= 0) & (page_indices < index_key_cache.shape[0])
        )
        contiguous_page_table = page_table_in_bounds & jnp.all(page_indices == expected_pages)
        return jax.lax.cond(
            contiguous_page_table,
            lambda: run_compressed(run_capacity, contiguous_page_table=True),
            lambda: run_compressed(run_capacity, contiguous_page_table=False),
        )

    return jax.lax.cond(
        num_page_runs <= _MAX_MAPPING_PAGE_RUNS,
        lambda: run_compressed_with_score_layout(_MAX_MAPPING_PAGE_RUNS),
        run_fallback,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "coalesce_page_dma",
        "interpret",
    ),
)
def paged_extend_score_block_pallas(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    valid_lengths: jax.Array,
    page_indices: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_EXTEND_BLOCK_K,
    first_dot_bf16: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Score one extend query block against a shared paged key stream.

    This is the general page-aware score schedule used by extend. Decode keeps
    its persistent two-sequence schedule because its query rows use different
    page tables; both schedules call the same weighted-ReLU score core.
    """

    kwargs = dict(
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        first_dot_bf16=first_dot_bf16,
        interpret=interpret,
    )
    if not coalesce_page_dma:
        return _paged_extend_score_block_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            valid_lengths,
            page_indices,
            coalesce_page_dma=False,
            **kwargs,
        )

    expected_pages = page_indices[:1] + jnp.arange(pages_per_seq, dtype=jnp.int32)
    page_table_in_bounds = jnp.all((page_indices >= 0) & (page_indices < index_key_cache.shape[0]))
    contiguous_page_table = page_table_in_bounds & jnp.all(page_indices == expected_pages)
    return jax.lax.cond(
        contiguous_page_table,
        lambda: _paged_extend_score_block_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            valid_lengths,
            page_indices,
            coalesce_page_dma=True,
            **kwargs,
        ),
        lambda: _paged_extend_score_block_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            valid_lengths,
            page_indices,
            coalesce_page_dma=False,
            **kwargs,
        ),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "persistent_two_seq",
        "coalesce_page_dma",
        "interpret",
    ),
)
def _paged_decode_scores_pallas_impl(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_BLOCK_K,
    first_dot_bf16: bool = False,
    persistent_two_seq: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Compute decode index scores directly from a paged key cache.

    ``coalesce_page_dma=True`` is an internal fast-path assertion: callers
    must verify that every active sequence page table is contiguous.

    Query row ``i`` belongs to sequence ``i``. The kernel never constructs a
    contiguous ``[T, max_kv, D]`` or ``[max_kv, D]`` gathered-key array; it
    follows ``page_indices`` while double-buffering page-aligned key blocks in
    VMEM and writes only the FP32 ``[T, max_kv]`` score matrix to HBM. The
    default block is tuned for the 135168-position v7x bucket and is reduced to
    the largest compatible divisor for other score shapes.
    """
    if q_idx.ndim != 3:
        raise ValueError(f"q_idx must have shape [T, H, D], got {q_idx.shape}")
    if idx_weights.shape != q_idx.shape[:2]:
        raise ValueError(f"idx_weights must have shape {q_idx.shape[:2]}, got {idx_weights.shape}")
    if index_key_cache.ndim != 3:
        raise ValueError(
            f"index_key_cache must have shape [P, page_size, D], got {index_key_cache.shape}"
        )
    num_queries, _, head_dim = q_idx.shape
    if seq_lens.shape != (num_queries,):
        raise ValueError(
            "decode scorer requires one query per sequence, "
            f"got q_idx.shape[0]={num_queries} and seq_lens.shape={seq_lens.shape}"
        )
    if index_key_cache.shape[-1] != head_dim:
        raise ValueError(
            f"cache head dim {index_key_cache.shape[-1]} does not match q head dim {head_dim}"
        )
    if head_dim % 128:
        raise ValueError(f"head_dim must be divisible by 128, got {head_dim}")
    if pages_per_seq < 1:
        raise ValueError(f"pages_per_seq must be positive, got {pages_per_seq}")

    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    if block_k < 128:
        raise ValueError(f"block_k must be at least 128, got {block_k}")
    if block_k % 128:
        raise ValueError(f"block_k must be a positive multiple of 128, got {block_k}")
    block_k = _resolve_block_k(max_kv, page_size, block_k)
    for name, value in (
        ("seq_lens", seq_lens),
        ("page_indices", page_indices),
        ("cu_kv_lens", cu_kv_lens),
        ("distribution", distribution),
    ):
        if value.dtype != jnp.int32:
            raise TypeError(f"{name} must be int32, got {value.dtype}")

    pages_per_block = block_k // page_size
    num_k_blocks = max_kv // block_k
    active_num_seqs = jnp.clip(distribution[2], 0, num_queries)
    scalar_prefetches = (seq_lens, page_indices, cu_kv_lens)
    idx_weights_3d = idx_weights[:, None, :]
    use_persistent_two_seq = (
        persistent_two_seq
        and num_queries >= _DECODE_SEQS_PER_GROUP
        and num_queries % _DECODE_SEQS_PER_GROUP == 0
    )
    if use_persistent_two_seq:
        kernel = functools.partial(
            _paged_decode_score_two_seq_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
        )
        grid = (num_queries // _DECODE_SEQS_PER_GROUP,)
        q_spec = pl.BlockSpec(
            (_DECODE_SEQS_PER_GROUP, *q_idx.shape[1:]),
            lambda pair_id, *_: (pair_id, 0, 0),
        )
        weights_spec = pl.BlockSpec(
            (_DECODE_SEQS_PER_GROUP, 1, idx_weights.shape[1]),
            lambda pair_id, *_: (pair_id, 0, 0),
        )
        score_spec = pl.BlockSpec(
            (_DECODE_SEQS_PER_GROUP, 1, max_kv),
            lambda pair_id, *_: (pair_id, 0, 0),
        )
        persistent_key_shape = (
            (
                _DECODE_SEQS_PER_GROUP,
                2,
                pages_per_block,
                page_size,
                head_dim,
            )
            if coalesce_page_dma
            else (_DECODE_SEQS_PER_GROUP, 2, block_k, head_dim)
        )
        scratch_shapes = [
            pltpu.VMEM(persistent_key_shape, index_key_cache.dtype),
            pltpu.SemaphoreType.DMA((_DECODE_SEQS_PER_GROUP, 2)),
        ]
    else:
        kernel = functools.partial(
            _paged_decode_score_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
        )
        grid = (active_num_seqs,)
        q_spec = pl.BlockSpec(
            (1, *q_idx.shape[1:]),
            lambda seq_id, *_: (seq_id, 0, 0),
        )
        weights_spec = pl.BlockSpec(
            (1, 1, idx_weights.shape[1]),
            lambda seq_id, *_: (seq_id, 0, 0),
        )
        score_spec = pl.BlockSpec(
            (1, 1, max_kv),
            lambda seq_id, *_: (seq_id, 0, 0),
        )
        key_shape = (
            (2, pages_per_block, page_size, head_dim)
            if coalesce_page_dma
            else (2, block_k, head_dim)
        )
        scratch_shapes = [
            pltpu.VMEM(key_shape, index_key_cache.dtype),
            pltpu.SemaphoreType.DMA((2,)),
        ]

    scores = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_queries, 1, max_kv), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            grid=grid,
            in_specs=[
                q_spec,
                weights_spec,
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=score_spec,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "dsa_paged_decode_score_contiguous"
            if coalesce_page_dma
            else "dsa_paged_decode_score_paged"
        ),
    )(
        *scalar_prefetches,
        q_idx,
        idx_weights_3d,
        index_key_cache,
    )
    active_rows = jnp.arange(num_queries, dtype=jnp.int32) < active_num_seqs
    return jnp.where(active_rows[:, None], scores[:, 0, :], _NEG_INF)


@functools.partial(
    jax.jit,
    static_argnames=(
        "pages_per_seq",
        "block_k",
        "first_dot_bf16",
        "persistent_two_seq",
        "coalesce_page_dma",
        "interpret",
    ),
)
def paged_decode_scores_pallas(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    pages_per_seq: int,
    block_k: int = _DEFAULT_BLOCK_K,
    first_dot_bf16: bool = False,
    persistent_two_seq: bool = False,
    coalesce_page_dma: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Score decode rows directly from paged keys.

    When requested, block DMA is used only after verifying complete active
    sequence page tables; fragmented or malformed tables take the exact
    per-page fallback.
    """

    kwargs = dict(
        pages_per_seq=pages_per_seq,
        block_k=block_k,
        first_dot_bf16=first_dot_bf16,
        persistent_two_seq=persistent_two_seq,
        interpret=interpret,
    )
    if not coalesce_page_dma:
        return _paged_decode_scores_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            seq_lens,
            page_indices,
            cu_kv_lens,
            distribution,
            coalesce_page_dma=False,
            **kwargs,
        )

    num_queries = q_idx.shape[0]
    page_size = index_key_cache.shape[1]
    active_num_seqs = jnp.clip(distribution[2], 0, num_queries)
    page_table_starts = cu_kv_lens[:num_queries] // page_size

    def read_sequence_pages(page_table_start):
        return jax.lax.dynamic_slice_in_dim(
            page_indices,
            page_table_start,
            pages_per_seq,
        )

    sequence_pages = jax.vmap(read_sequence_pages)(page_table_starts)
    expected_pages = (
        sequence_pages[:, :1]
        + jnp.arange(
            pages_per_seq,
            dtype=jnp.int32,
        )[None, :]
    )
    page_table_in_bounds = (page_table_starts >= 0) & (
        page_table_starts + pages_per_seq <= page_indices.shape[0]
    )
    contiguous_rows = page_table_in_bounds & jnp.all(
        sequence_pages == expected_pages,
        axis=1,
    )
    active_rows = jnp.arange(num_queries, dtype=jnp.int32) < active_num_seqs
    all_active_rows_contiguous = jnp.all(~active_rows | contiguous_rows)

    def score(*, contiguous_page_table: bool):
        return _paged_decode_scores_pallas_impl(
            q_idx,
            idx_weights,
            index_key_cache,
            seq_lens,
            page_indices,
            cu_kv_lens,
            distribution,
            coalesce_page_dma=contiguous_page_table,
            **kwargs,
        )

    return jax.lax.cond(
        all_active_rows_contiguous,
        lambda: score(contiguous_page_table=True),
        lambda: score(contiguous_page_table=False),
    )
