"""Paged-cache Pallas scorer for one-token-per-sequence DSA decode."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG_INF = float("-inf")
_DEFAULT_BLOCK_K = 22528


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
        similarities = lax.dot_general(
            query,
            keys,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        # Match the reference einsum's second DEFAULT-precision dot.  A plain
        # FP32 multiply + reduce uses a different TPU numerical path and can
        # perturb candidates around the top-k boundary.
        scores = lax.dot_general(
            weights_vmem_ref[0, 0].astype(jnp.float32),
            jnp.maximum(similarities, jnp.float32(0.0)),
            dimension_numbers=(((0,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
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
    seq_lens_ref,  # i32[2], SMEM
    page_indices_ref,  # i32[2 * pages_per_seq], SMEM
    cu_kv_lens_ref,  # i32[3], SMEM
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
    """Score two sequences in one program with a shared prefetch schedule."""
    block_k = pages_per_block * page_size if coalesce_page_dma else keys_vmem_ref.shape[2]

    def fetch_key_block(seq_id, block_id, buffer, *, wait: bool):
        dst = keys_vmem_ref.at[seq_id, buffer]
        sem = key_dma_sems.at[seq_id, buffer]
        if wait:
            pltpu.make_async_copy(dst, dst, sem).wait()
            return

        page_table_start = cu_kv_lens_ref[seq_id] // page_size
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
        similarities = lax.dot_general(
            query,
            keys,
            dimension_numbers=(((2,), (2,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        scores = lax.dot_general(
            weights_vmem_ref[:, 0].astype(jnp.float32),
            jnp.maximum(similarities, jnp.float32(0.0)),
            dimension_numbers=(((1,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        positions = block_id * block_k + jnp.arange(block_k, dtype=jnp.int32)
        for seq_id in range(2):
            scores_out_vmem_ref[
                seq_id,
                0,
                pl.ds(block_id * block_k, block_k),
            ] = jnp.where(
                positions < seq_lens_ref[seq_id],
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
    use_persistent_two_seq = persistent_two_seq and num_queries == 2
    if use_persistent_two_seq:
        kernel = functools.partial(
            _paged_decode_score_two_seq_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            num_k_blocks=num_k_blocks,
            first_dot_bf16=first_dot_bf16,
            coalesce_page_dma=coalesce_page_dma,
        )
        grid = (1,)
        q_spec = pl.BlockSpec(q_idx.shape, lambda *_: (0, 0, 0))
        weights_spec = pl.BlockSpec(idx_weights_3d.shape, lambda *_: (0, 0, 0))
        score_spec = pl.BlockSpec(
            (num_queries, 1, max_kv),
            lambda *_: (0, 0, 0),
        )
        persistent_key_shape = (
            (
                num_queries,
                2,
                pages_per_block,
                page_size,
                head_dim,
            )
            if coalesce_page_dma
            else (num_queries, 2, block_k, head_dim)
        )
        scratch_shapes = [
            pltpu.VMEM(persistent_key_shape, index_key_cache.dtype),
            pltpu.SemaphoreType.DMA((num_queries, 2)),
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
