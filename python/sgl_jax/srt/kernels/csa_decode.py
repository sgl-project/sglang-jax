"""Request-local paged CSA decode scores with multi-row, double-buffered DMA.

Each grid step owns several queries and prefetches two key tiles per query.
Host-built page-run segments combine contiguous physical pages into larger DMAs;
callers without segments use the device-side single-page segmentation.
Both schedules compute completed-group scores with an FP32 weighted head reduction.
"""

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG_INF = jnp.finfo(jnp.float32).min

MAX_BLOCK_K = 2048
DEFAULT_ROWS_PER_STEP = 8
ROWS_PER_STEP_ENV = "DSV4_CSA_SCORER_ROWS"

# One int32 per DMA segment: physical page << 10 | page offset inside the block << 3 | log2(pages)
_SEG_SIZE_BITS = 3
_SEG_OFFSET_BITS = 7
_SEG_SHIFT = _SEG_SIZE_BITS + _SEG_OFFSET_BITS
_SEG_SIZE_MASK = (1 << _SEG_SIZE_BITS) - 1
_SEG_OFFSET_MASK = (1 << _SEG_OFFSET_BITS) - 1
MAX_PHYSICAL_PAGE = (1 << (31 - _SEG_SHIFT)) - 1
MAX_PAGES_PER_BLOCK = 1 << _SEG_OFFSET_BITS


def scorer_block_k(capacity: int) -> int:
    """Key rows one scorer tile holds for a request-local capacity of ``capacity`` entries.

    The largest power of two (at most ``MAX_BLOCK_K``) that divides ``capacity``, so
    fine capacity buckets such as 2560 tile as 5 x 512 rather than failing.
    """
    block = min(MAX_BLOCK_K, capacity)
    while block > 128 and capacity % block:
        block //= 2
    return block


def scorer_pages_per_block(capacity: int, page_size: int) -> int:
    """Pages per scorer tile; the host page-run segmentation must use the same value."""
    block_k = scorer_block_k(capacity)
    if block_k % page_size or capacity % block_k:
        raise ValueError("CSA decode capacity must be a multiple of the key tile and page size")
    return block_k // page_size


def encode_segments(physical_page, offset, log2_pages):
    """Pack ``(physical page, page offset in block, log2 pages)`` into the segment int32."""
    return (physical_page << _SEG_SHIFT) | (offset << _SEG_SIZE_BITS) | log2_pages


def decode_segments(segment):
    """Inverse of ``encode_segments`` (numpy / jnp)."""
    return (
        segment >> _SEG_SHIFT,
        (segment >> _SEG_SIZE_BITS) & _SEG_OFFSET_MASK,
        segment & _SEG_SIZE_MASK,
    )


def page_run_segments(pages, page_counts, pages_per_block: int):
    """Host (numpy) page-run segmentation of a request-local page table.

    ``pages`` [T, N] physical page per logical page (any integer dtype; entries at or
    beyond ``page_counts[t]`` are ignored), ``page_counts`` [T] valid pages per row.
    Returns ``(segments [T, N] int32, counts [T, N // pages_per_block] int32)``: for
    every block of ``pages_per_block`` logical pages, ``counts`` DMA segments, each a run
    of ``2**k`` physically consecutive pages, larger pieces first, in block order.
    ``popcount(run) <= run`` keeps every block's segment list inside its ``pages_per_block``
    slots.
    """
    pages = np.asarray(pages)
    counts = np.asarray(page_counts)
    if pages.ndim != 2 or counts.shape != (pages.shape[0],):
        raise ValueError("pages must be [T, N] and page_counts [T]")
    rows, table = pages.shape
    if pages_per_block <= 0 or table % pages_per_block:
        raise ValueError("the page table width must be a multiple of pages_per_block")
    if pages_per_block > MAX_PAGES_PER_BLOCK:
        raise ValueError(f"pages_per_block must be <= {MAX_PAGES_PER_BLOCK}")
    blocks = table // pages_per_block
    pages64 = pages.astype(np.int64).reshape(rows, blocks, pages_per_block)
    logical = np.arange(table, dtype=np.int64).reshape(1, blocks, pages_per_block)
    valid = logical < counts.astype(np.int64)[:, None, None]
    if valid.any() and (
        pages64[valid].min() < 0 or pages64[valid].max() + pages_per_block > MAX_PHYSICAL_PAGE
    ):
        raise ValueError(f"physical pages must lie in [0, {MAX_PHYSICAL_PAGE - pages_per_block}]")
    contiguous = np.zeros_like(valid)
    contiguous[..., 1:] = (
        valid[..., 1:] & valid[..., :-1] & (pages64[..., 1:] == pages64[..., :-1] + 1)
    )
    start = valid & ~contiguous
    offsets = np.arange(pages_per_block, dtype=np.int64)
    run_of = np.maximum.accumulate(np.where(start, offsets, -1), axis=-1)
    # lengths[t, b, k] = pages in the run that starts at k (0 where no run starts):
    # count, per (row, block), how many valid positions point at each run start.
    flat_key = np.arange(rows * blocks)[:, None] * pages_per_block + run_of.reshape(
        rows * blocks, -1
    )
    counts_flat = np.bincount(
        flat_key[valid.reshape(rows * blocks, -1)], minlength=rows * blocks * pages_per_block
    )
    lengths = counts_flat.reshape(rows, blocks, pages_per_block)
    lengths = np.where(start, lengths, 0)
    pieces_has, pieces_val = [], []
    for bit in range(pages_per_block.bit_length() - 1, -1, -1):
        has = start & (((lengths >> bit) & 1) == 1)
        placed = lengths & ~((1 << (bit + 1)) - 1)  # larger pieces already placed before
        pieces_has.append(has)
        pieces_val.append(encode_segments(pages64 + placed, offsets + placed, bit))
    has = np.stack(pieces_has, -1).reshape(rows * blocks, -1)
    val = np.stack(pieces_val, -1).reshape(rows * blocks, -1)
    slot = np.cumsum(has, axis=1) - 1
    out = np.zeros((rows * blocks, pages_per_block), np.int64)
    src_rows, src_cols = np.nonzero(has)
    out[src_rows, slot[src_rows, src_cols]] = val[src_rows, src_cols]
    segments = out.reshape(rows, table).astype(np.int32)
    seg_counts = has.sum(1).reshape(rows, blocks).astype(np.int32)
    return segments, seg_counts


def trivial_segments(pages, page_counts, pages_per_block: int):
    """Device (jnp) segmentation with one single-page segment per valid page.

    Same contract as ``page_run_segments``; used when a caller has no host-built
    segments (including tests). Both feed the same kernel.
    """
    pages = jnp.asarray(pages, jnp.int32)
    rows, table = pages.shape
    blocks = table // pages_per_block
    logical = jnp.arange(table, dtype=jnp.int32)
    valid = logical[None, :] < jnp.asarray(page_counts, jnp.int32)[:, None]
    offsets = logical % pages_per_block
    segments = jnp.where(valid, encode_segments(pages, offsets[None, :], 0), 0)
    remaining = (
        jnp.asarray(page_counts, jnp.int32)[:, None]
        - (jnp.arange(blocks, dtype=jnp.int32) * pages_per_block)[None, :]
    )
    counts = jnp.clip(remaining, 0, pages_per_block)
    return segments.astype(jnp.int32), counts.astype(jnp.int32)


def _score_tile(query, resident, head_weights, length, block, block_k, out_ref, out_row):
    similarities = lax.dot_general(
        query,
        resident.astype(query.dtype),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    scores = jnp.sum(jax.nn.relu(similarities) * head_weights[:, None], axis=0)
    positions = block * block_k + jnp.arange(block_k, dtype=jnp.int32)
    out_ref[out_row, 0, pl.ds(pl.multiple_of(block * block_k, block_k), block_k)] = jnp.where(
        positions < length, scores, _NEG_INF
    )


def _segments_kernel(
    lengths_ref,
    counts_ref,
    segments_ref,
    q_ref,
    weights_ref,
    cache_ref,
    out_ref,
    keys_ref,
    sems_ref,
    *,
    page_size,
    pages_per_block,
    rows_per_step,
    max_log2,
):
    step = pl.program_id(0)
    block_k = keys_ref.shape[2]
    out_ref[...] = jnp.full(out_ref.shape, _NEG_INF, jnp.float32)

    def segment_copy(r, buffer, physical, offset, num_pages):
        src = cache_ref.at[
            pl.ds(pl.multiple_of(physical * page_size, page_size), num_pages * page_size)
        ]
        dst = keys_ref.at[
            r, buffer, pl.ds(pl.multiple_of(offset * page_size, page_size), num_pages * page_size)
        ]
        return pltpu.make_async_copy(src, dst, sems_ref.at[r, buffer])

    def for_segments(row, block, apply):
        base = block * pages_per_block

        def body(i, carry):
            segment = segments_ref[row, base + i]
            physical, offset, log2_pages = decode_segments(segment)
            for size_log2 in range(max_log2 + 1):

                @pl.when(log2_pages == size_log2)
                def _apply(size_log2=size_log2):
                    apply(physical, offset, 1 << size_log2)

            return carry

        lax.fori_loop(0, counts_ref[row, block], body, 0)

    def fetch(r, row, block):
        buffer = block % 2
        keys_ref[r, buffer] = jnp.zeros(keys_ref.shape[2:], keys_ref.dtype)
        for_segments(row, block, lambda p, o, n: segment_copy(r, buffer, p, o, n).start())

    def wait(r, row, block):
        buffer = block % 2
        for_segments(row, block, lambda p, o, n: segment_copy(r, buffer, p, o, n).wait())

    rows = [step * rows_per_step + r for r in range(rows_per_step)]
    lengths = [lengths_ref[row] for row in rows]
    num_blocks = [(length + block_k - 1) // block_k for length in lengths]
    max_blocks = functools.reduce(jnp.maximum, num_blocks)

    # Two tiles per row in flight before any wait: every row's DMAs overlap.
    for first in (0, 1):
        for r, row in enumerate(rows):

            @pl.when(num_blocks[r] > first)
            def _prefetch(r=r, row=row, first=first):
                fetch(r, row, first)

    def step_blocks(block, carry):
        # Waits and prefetches are per-row scalar branches; the tile maths below is
        # branch-free straight-line code over the rows so the scheduler can overlap
        # one row's MXU pass with another's reduction. Rows without this block score
        # whatever their buffer holds and mask every position past their length.
        for r, row in enumerate(rows):

            @pl.when(block < num_blocks[r])
            def _wait_row(r=r, row=row):
                wait(r, row, block)

        for r in range(rows_per_step):
            _score_tile(
                q_ref[r],
                keys_ref[r, block % 2],
                weights_ref[r, 0],
                lengths[r],
                block,
                block_k,
                out_ref,
                r,
            )

        for r, row in enumerate(rows):

            @pl.when(block + 2 < num_blocks[r])
            def _prefetch_row(r=r, row=row):
                fetch(r, row, block + 2)

        return carry

    lax.fori_loop(0, max_blocks, step_blocks, 0)


def resolve_rows_per_step(rows_per_step=None) -> int:
    if rows_per_step is None:
        rows_per_step = int(os.environ.get(ROWS_PER_STEP_ENV, DEFAULT_ROWS_PER_STEP))
    if rows_per_step < 1:
        raise ValueError("rows_per_step must be positive")
    return rows_per_step


def paged_csa_decode_scores(
    q,
    weights,
    cache,
    lengths,
    pages,
    *,
    page_size=None,
    segments=None,
    segment_counts=None,
    rows_per_step=None,
    interpret=False,
):
    """Score only each query's completed compressed entries.

    q [T,H,D], weights [T,H], cache [P,page_size,D] or flat [P*page_size,D] (then
    ``page_size`` is required), lengths [T], pages [T,N]. ``segments`` / ``segment_counts``
    are the host page-run segmentation of ``pages`` (``page_run_segments``); without them
    the kernel reads one page per DMA. The result [T,N*page_size] uses finite minimum
    FP32 for every invalid entry. Request isolation is encoded in the allocator-derived
    page table.
    """
    tokens, heads, dim = q.shape
    if cache.ndim == 3:
        if page_size is not None and page_size != cache.shape[1]:
            raise ValueError("page_size disagrees with the 3-D cache")
        page_size = cache.shape[1]
        cache = cache.reshape(-1, dim)
    elif cache.ndim != 2 or page_size is None:
        raise ValueError("a flat CSA decode cache needs page_size")
    capacity = pages.shape[1] * page_size
    if weights.shape != (tokens, heads) or lengths.shape != (tokens,) or pages.shape[0] != tokens:
        raise ValueError("CSA decode query, weight, length and page-table shapes disagree")
    if cache.shape[-1] != dim or dim % 128 or page_size % 8 or capacity % 128:
        raise ValueError("CSA decode requires aligned cache pages and 128-wide key dimensions")
    block_k = scorer_block_k(capacity)
    pages_per_block = scorer_pages_per_block(capacity, page_size)
    blocks = capacity // block_k
    lengths = jnp.asarray(lengths, jnp.int32)
    if segments is None:
        page_counts = (lengths + page_size - 1) // page_size
        segments, segment_counts = trivial_segments(pages, page_counts, pages_per_block)
    segments = jnp.asarray(segments, jnp.int32)
    segment_counts = jnp.asarray(segment_counts, jnp.int32)
    if segments.shape != (tokens, pages.shape[1]) or segment_counts.shape != (tokens, blocks):
        raise ValueError("CSA decode segments do not match the page table")

    rows_per_step = resolve_rows_per_step(rows_per_step)
    padded = -(-tokens // rows_per_step) * rows_per_step
    if padded != tokens:
        pad = ((0, padded - tokens),)
        q = jnp.pad(q, pad + ((0, 0), (0, 0)))
        weights = jnp.pad(weights, pad + ((0, 0),))
        lengths = jnp.pad(lengths, pad)
        segments = jnp.pad(segments, pad + ((0, 0),))
        segment_counts = jnp.pad(segment_counts, pad + ((0, 0),))
    itemsize = jnp.dtype(cache.dtype).itemsize
    keys_bytes = rows_per_step * 2 * block_k * dim * itemsize
    out_bytes = 2 * rows_per_step * capacity * 4
    vmem_limit = max(16, min(96, 8 + (keys_bytes + out_bytes) * 2 // (1024 * 1024))) * 1024 * 1024
    scores = pl.pallas_call(
        functools.partial(
            _segments_kernel,
            page_size=page_size,
            pages_per_block=pages_per_block,
            rows_per_step=rows_per_step,
            max_log2=pages_per_block.bit_length() - 1,
        ),
        out_shape=jax.ShapeDtypeStruct((padded, 1, capacity), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(padded // rows_per_step,),
            in_specs=[
                pl.BlockSpec((rows_per_step, heads, dim), lambda s, *_: (s, 0, 0)),
                pl.BlockSpec((rows_per_step, 1, heads), lambda s, *_: (s, 0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((rows_per_step, 1, capacity), lambda s, *_: (s, 0, 0)),
            scratch_shapes=[
                pltpu.VMEM((rows_per_step, 2, block_k, dim), cache.dtype),
                pltpu.SemaphoreType.DMA((rows_per_step, 2)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=vmem_limit
        ),
        interpret=interpret,
        name=f"csa_request_local_decode_scores_r{rows_per_step}",
    )(lengths, segment_counts, segments, q, weights[:, None, :], cache)
    return scores[:tokens, 0, :]
