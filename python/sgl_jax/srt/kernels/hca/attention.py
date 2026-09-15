"""Shared-KV multi-query attention over HCA's window and compressed cache tiers.

The ABI mirrors SGLang-JAX ragged paged attention: physical cache pages are
separate from flattened per-request ``page_indices`` tables, while
``cu_*_kv_lens`` locate each request's page-table segment.  No request-major KV
snapshot or per-entry physical-slot map crosses the kernel boundary.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.hca.tuned_block_sizes import HCAKernelSchedule


def _align(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


# Offset no real query length can reach, so padded schedule rows fail the
# kernels' ``query_base < q_len`` check and skip their KV loops.
INERT_QUERY_OFFSET = 1 << 30


def _get_interpret() -> bool:
    requested = os.environ.get("PALLAS_INTERPRET", "").strip().lower()
    return requested in ("1", "true") or jax.default_backend() != "tpu"


def _data_out_sharding(rank: int):
    """Return an explicit leading-data spec only inside a named mesh."""
    mesh = jax.sharding.get_abstract_mesh()
    if "data" not in mesh.axis_names:
        return None
    data_axis = mesh.axis_names.index("data")
    if mesh.axis_types[data_axis] is not jax.sharding.AxisType.Explicit:
        return None
    return P("data", *(None for _ in range(rank - 1)))


def _cache_layout(cache, head_dim):
    if cache.ndim != 4 or cache.shape[-1] < head_dim:
        raise ValueError(
            f"paged cache must be [pages,page_size/packing,packing,head_dim], got {cache.shape}"
        )
    page_size = cache.shape[1] * cache.shape[2]
    return cache.reshape(-1, cache.shape[-1]), page_size


def _page_table_locations(
    page_indices,
    cu_kv_lens,
    logical_positions,
    valid,
    *,
    page_size,
    physical_rows,
    sequence_ids=None,
):
    """Resolve logical positions through SGLang's flattened ragged page table."""
    starts = (cu_kv_lens[:-1] // jnp.int32(page_size)).astype(jnp.int32)
    if sequence_ids is not None:
        starts = starts[sequence_ids.astype(jnp.int32)]
    while starts.ndim < logical_positions.ndim:
        starts = starts[..., None]
    logical_pages = jnp.floor_divide(logical_positions, page_size).astype(jnp.int32)
    table_locs = starts + logical_pages
    table_valid = valid & (table_locs >= 0) & (table_locs < page_indices.shape[0])
    safe_table_locs = jnp.where(table_valid, table_locs, 0)
    page_ref = page_indices.at[safe_table_locs]
    page_out_sharding = _data_out_sharding(safe_table_locs.ndim)
    if page_out_sharding is None:
        physical_pages = page_ref.get(mode="promise_in_bounds")
    else:
        physical_pages = page_ref.get(mode="promise_in_bounds", out_sharding=page_out_sharding)
    physical_locs = physical_pages.astype(jnp.int32) * jnp.int32(page_size) + jnp.mod(
        logical_positions, page_size
    ).astype(jnp.int32)
    resolved_valid = table_valid & (physical_pages > 0) & (physical_locs < physical_rows)
    return physical_locs, resolved_valid


def _window_locations(page_indices, cu_kv_lens, clamped_positions, *, page_size):
    """Resolve ring positions through the flattened window table.

    Unlike ``_page_table_locations`` this skips the table-bound checks, which
    lower to several 128-wide reductions; callers must clamp positions into
    their request's table segment and mask invalid rows themselves.
    """
    table_locs = (cu_kv_lens[:-1] // jnp.int32(page_size))[:, None] + jnp.floor_divide(
        clamped_positions, page_size
    )
    pages = page_indices.at[table_locs].get(mode="promise_in_bounds")
    return pages, pages * jnp.int32(page_size) + jnp.mod(clamped_positions, page_size)


def _gather_physical_rows(flat_cache, locations, valid, head_dim):
    safe = jnp.where(valid, locations, 0).astype(jnp.int32)
    selected_ref = flat_cache.at[safe, :head_dim]
    out_sharding = _data_out_sharding(safe.ndim + 1)
    if out_sharding is None:
        selected = selected_ref.get(mode="promise_in_bounds")
    else:
        selected = selected_ref.get(mode="promise_in_bounds", out_sharding=out_sharding)
    return jnp.where(valid[..., None], selected, 0)


def _scatter_physical_rows(flat_cache, locations, values, valid):
    safe = jnp.where(valid, locations, flat_cache.shape[0]).astype(jnp.int32).reshape(-1)
    values = values.reshape(-1, values.shape[-1])
    padded = jnp.pad(
        values.astype(flat_cache.dtype),
        ((0, 0), (0, flat_cache.shape[-1] - values.shape[-1])),
    )
    update = flat_cache.at[safe]
    kwargs = {"mode": "drop", "wrap_negative_indices": False}
    out_sharding = _data_out_sharding(flat_cache.ndim)
    if out_sharding is not None:
        kwargs["out_sharding"] = out_sharding
    return update.set(padded, **kwargs)


def _commit_window_rows(
    window_flat,
    window_page_indices,
    window_cu_kv_lens,
    positions,
    new_kv,
    valid_token_mask,
    query_seq_ids,
    seq_lens,
    *,
    window_size,
    page_size,
):
    """Write each token's KV into its sliding-window ring slot.

    A long chunk can wrap the ring more than once; earlier writes are dead and
    their physical destinations collide, so only each request's final
    ``window_size`` positions are committed.
    """
    locations, valid = _page_table_locations(
        window_page_indices,
        window_cu_kv_lens,
        positions,
        valid_token_mask & (positions >= 0),
        page_size=page_size,
        physical_rows=window_flat.shape[0],
        sequence_ids=query_seq_ids,
    )
    final_window = valid & (positions >= seq_lens[query_seq_ids] - window_size)
    return _scatter_physical_rows(window_flat, locations, new_kv, final_window)


def _write_cache_rows_kernel(
    locations_ref,
    valid_ref,
    values_ref,
    cache_hbm_ref,
    _,
    packed_rows_ref,
    semaphore,
    *,
    page_size: int,
    packing: int,
    tile_n: int,
):
    """DMA selected BF16 rows into an input-output-aliased physical cache."""
    block = pl.program_id(0)
    for row in range(tile_n):
        token = block * tile_n + row
        valid = valid_ref[token]

        @pl.when(valid)
        def _write_valid_row(row=row, token=token):
            location = locations_ref[token]
            page = jnp.floor_divide(location, page_size)
            page_row = jnp.mod(location, page_size)
            packed_row = jnp.floor_divide(page_row, packing)
            packed_lane = jnp.mod(page_row, packing)
            load = pltpu.make_async_copy(
                cache_hbm_ref.at[page, packed_row],
                packed_rows_ref.at[row],
                semaphore,
            )
            load.start()
            load.wait()
            first_lane = jnp.stack((values_ref[row], packed_rows_ref[row, 1]), axis=0)
            second_lane = jnp.stack((packed_rows_ref[row, 0], values_ref[row]), axis=0)
            packed_rows_ref[row] = jnp.where(packed_lane == 0, first_lane, second_lane)
            store = pltpu.make_async_copy(
                packed_rows_ref.at[row],
                cache_hbm_ref.at[page, packed_row],
                semaphore,
            )
            store.start()
            store.wait()


@functools.partial(jax.jit, static_argnames=("schedule",))
def _write_cache_rows(cache, locations, values, valid, *, schedule: HCAKernelSchedule):
    """Return ``cache`` after aliased writes, without copying untouched pages."""
    tokens, head_dim = values.shape
    tile_n = schedule.cache_write_tile
    padded = _align(tokens, tile_n)
    pad = padded - tokens
    values = jnp.pad(values.astype(cache.dtype), ((0, pad), (0, 0)))
    locations = jnp.pad(locations.astype(jnp.int32), (0, pad))
    valid = jnp.pad(valid.astype(jnp.bool_), (0, pad))
    page_size = cache.shape[1] * cache.shape[2]
    packing = cache.shape[2]
    if packing != 2:
        raise ValueError("production HCA alias writer requires BF16 packing=2")
    return pl.pallas_call(
        functools.partial(
            _write_cache_rows_kernel,
            page_size=page_size,
            packing=packing,
            tile_n=tile_n,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(padded // tile_n,),
            in_specs=(
                pl.BlockSpec((tile_n, head_dim), lambda block, *_: (block, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((tile_n, packing, head_dim), cache.dtype),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={3: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=_get_interpret(),
        name=f"hca-cache-row-write-n{tile_n}-d{head_dim}",
    )(locations, valid, values, cache)


def _streaming_attention_kernel(
    compressed_page_indices_ref,
    compressed_page_starts_ref,
    compressed_lens_ref,
    q_ref,
    window_kv_ref,
    window_len_ref,
    compressed_cache_hbm_ref,
    attention_sink_ref,
    out_ref,
    compressed_kv_x2_ref,
    dma_semaphores,
    m_ref,
    l_ref,
    acc_ref,
    *,
    pages_per_block: int,
    page_size: int,
    tile_k: int,
    compressed_tile: int,
    softmax_scale: float,
):
    """One FlashAttention-style program over SWA then compressed HCA tiles."""
    head_dim = q_ref.shape[2]
    q = q_ref[0, ...].astype(jnp.bfloat16)
    sink = attention_sink_ref[...].astype(jnp.float32)[:, None]
    # Compute SWA without the virtual sink, round its unnormalised numerator to BF16
    # at the SWA/compressed-cache boundary, and add the sink to the final denominator.
    negative_finite = jnp.finfo(jnp.float32).min
    m_ref[...] = jnp.full(m_ref.shape, negative_finite, jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    def consume(kv, valid):
        scores = jax.lax.dot_general(
            q,
            kv.astype(jnp.bfloat16),
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ) * jnp.float32(softmax_scale)
        scores = jnp.where(valid[None, :], scores, negative_finite)
        block_maximum = jnp.max(scores, axis=1, keepdims=True)
        previous_maximum = m_ref[...][:, :1]
        next_maximum = jnp.maximum(previous_maximum, block_maximum)
        alpha = jnp.exp(previous_maximum - next_maximum)
        probabilities = jnp.exp(scores - next_maximum)
        next_denominator = alpha * l_ref[...][:, :1] + jnp.sum(probabilities, axis=1, keepdims=True)
        value = jax.lax.dot_general(
            probabilities,
            kv.astype(jnp.bfloat16),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        acc_ref[...] = alpha * acc_ref[...] + value
        m_ref[...] = jnp.broadcast_to(next_maximum, m_ref.shape)
        l_ref[...] = jnp.broadcast_to(next_denominator, l_ref.shape)

    token = pl.program_id(0)
    window_len = window_len_ref[0, 0, 0]
    # Decode splits its ring by lane width, independently of the chunk path's
    # ``swa_compute_tile``; two 64-row blocks fix the reduction order.
    swa_tile = tile_k // 2
    window_valid = jnp.arange(tile_k, dtype=jnp.int32) < window_len
    consume(window_kv_ref[0, :swa_tile], window_valid[:swa_tile])
    consume(window_kv_ref[0, swa_tile:], window_valid[swa_tile:])
    acc_ref[...] = acc_ref[...].astype(jnp.bfloat16).astype(jnp.float32)

    compressed_len = compressed_lens_ref[token]
    compressed_page_start = compressed_page_starts_ref[token]
    num_blocks = pl.cdiv(compressed_len, compressed_tile)
    cache_rows = compressed_cache_hbm_ref.reshape(-1, head_dim)

    def fetch(block, buffer, *, wait):
        semaphore = dma_semaphores.at[buffer]
        kv_buffer = compressed_kv_x2_ref.at[buffer]
        if wait:
            destination = kv_buffer.at[pl.ds(0, compressed_tile)]
            pltpu.make_async_copy(destination, destination, semaphore).wait()
            return
        for page_in_block in range(pages_per_block):
            logical_page = block * pages_per_block + page_in_block
            table_location = jnp.minimum(
                compressed_page_start + logical_page,
                compressed_page_indices_ref.shape[0] - 1,
            )
            physical_page = compressed_page_indices_ref[table_location]
            pltpu.make_async_copy(
                cache_rows.at[pl.ds(physical_page * page_size, page_size)],
                kv_buffer.at[pl.ds(page_in_block * page_size, page_size)],
                semaphore,
            ).start()

    @pl.when(num_blocks > 0)
    def _start_first_block():
        fetch(0, 0, wait=False)

    def consume_compressed(block, buffer):
        fetch(block, buffer, wait=True)
        next_block = block + 1
        next_buffer = jnp.bitwise_xor(buffer, 1)

        @pl.when(next_block < num_blocks)
        def _start_next_block():
            fetch(next_block, next_buffer, wait=False)

        valid = block * compressed_tile + jnp.arange(compressed_tile, dtype=jnp.int32)
        consume(compressed_kv_x2_ref[buffer, ...], valid < compressed_len)
        return next_buffer

    jax.lax.fori_loop(0, num_blocks, consume_compressed, jnp.int32(0), unroll=False)

    sink_term = jnp.exp(sink - m_ref[...][:, :1])
    denominator = l_ref[...][:, :1] + sink_term
    out_ref[...] = (acc_ref[...] * pl.reciprocal(denominator, approx=True)).astype(jnp.bfloat16)[
        None, ...
    ]


@functools.partial(
    jax.jit,
    static_argnames=("softmax_scale", "interpret", "schedule"),
)
def _streaming_attention(
    q,
    window_rows,
    window_lens,
    compressed_cache,
    compressed_page_indices,
    compressed_page_starts,
    compressed_lens,
    attention_sink,
    *,
    schedule: HCAKernelSchedule,
    softmax_scale: float,
    interpret: bool | None = None,
):
    """Stream both HCA segments through one Pallas online-softmax program."""
    if q.ndim != 3 or window_rows.ndim != 3 or compressed_cache.ndim != 4:
        raise ValueError("q/window/cache must be [T,H,D]/[T,K,D]/physical 4D")
    tokens, heads, head_dim = q.shape
    if q.dtype != jnp.bfloat16:
        raise ValueError("HCA streaming query must be BF16")
    if window_rows.shape[0] != tokens:
        raise ValueError("q and window rows must share T")
    if window_rows.shape[2] != head_dim or compressed_cache.shape[-1] < head_dim:
        raise ValueError("all HCA streaming inputs must cover head_dim")
    if window_lens.shape != (tokens,) or compressed_lens.shape != (tokens,):
        raise ValueError("window_lens and compressed_lens must be [T]")
    if compressed_page_starts.shape != (tokens,):
        raise ValueError("compressed_page_starts must be [T]")
    if compressed_page_indices.ndim != 1:
        raise ValueError("compressed_page_indices must be flattened")
    if attention_sink.shape != (heads,):
        raise ValueError(f"attention_sink must be [{heads}]")
    tile_k = schedule.mxu_lanes
    compressed_tile = schedule.compressed_tile
    if head_dim % schedule.mxu_lanes:
        raise ValueError(f"head_dim must be a multiple of {schedule.mxu_lanes}")
    if window_rows.shape[1] > tile_k:
        raise ValueError("SWA rows must fit one reduction tile")

    padded_heads = _align(heads, schedule.sublanes)
    page_size = compressed_cache.shape[1] * compressed_cache.shape[2]
    if compressed_tile % page_size:
        raise ValueError("physical cache page_size must divide the compressed tile")
    pages_per_block = compressed_tile // page_size
    q = jnp.pad(q, ((0, 0), (0, padded_heads - heads), (0, 0)))
    # The native 128-row path pads the final reduction block on the right.
    if window_rows.shape[1] < tile_k:
        window_rows = jnp.pad(
            window_rows,
            ((0, 0), (0, tile_k - window_rows.shape[1]), (0, 0)),
        )
    swa_rows = window_rows.shape[1]
    attention_sink = jnp.pad(
        attention_sink.astype(jnp.float32),
        (0, padded_heads - heads),
        constant_values=-jnp.inf,
    )
    if interpret is None:
        interpret = _get_interpret()

    scalar_prefetches = (
        compressed_page_indices.astype(jnp.int32),
        compressed_page_starts.astype(jnp.int32),
        compressed_lens.astype(jnp.int32),
    )
    output = pl.pallas_call(
        functools.partial(
            _streaming_attention_kernel,
            pages_per_block=pages_per_block,
            page_size=page_size,
            tile_k=tile_k,
            compressed_tile=compressed_tile,
            softmax_scale=float(softmax_scale),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            grid=(tokens,),
            in_specs=(
                pl.BlockSpec((1, padded_heads, head_dim), lambda token, *_: (token, 0, 0)),
                pl.BlockSpec((1, swa_rows, head_dim), lambda token, *_: (token, 0, 0)),
                pl.BlockSpec(
                    (1, schedule.sublanes, schedule.mxu_lanes),
                    lambda token, *_: (token, 0, 0),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((padded_heads,), lambda token, *_: (0,)),
            ),
            out_specs=pl.BlockSpec((1, padded_heads, head_dim), lambda token, *_: (token, 0, 0)),
            scratch_shapes=(
                pltpu.VMEM((2, compressed_tile, head_dim), compressed_cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM((padded_heads, schedule.mxu_lanes), jnp.float32),
                pltpu.VMEM((padded_heads, schedule.mxu_lanes), jnp.float32),
                pltpu.VMEM((padded_heads, head_dim), jnp.float32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((tokens, padded_heads, head_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            f"hca-paged-stream-swa{tile_k}-hca{compressed_tile}"
            f"-p{page_size}-h{padded_heads}-d{head_dim}"
        ),
    )(
        *scalar_prefetches,
        q,
        window_rows,
        jnp.broadcast_to(
            window_lens.astype(jnp.int32)[:, None, None],
            (tokens, schedule.sublanes, schedule.mxu_lanes),
        ),
        compressed_cache,
        attention_sink,
    )
    return output[:, :heads]


def _chunk_attention_kernel(
    compressed_page_indices_ref,
    compressed_page_starts_ref,
    q_lens_ref,
    prefix_lens_ref,
    compressed_lens_ref,
    query_block_request_ids_ref,
    query_block_offsets_ref,
    cu_q_lens_ref,
    q_hbm_ref,
    q_tail_hbm_ref,
    combined_kv_hbm_ref,
    compressed_cache_hbm_ref,
    attention_sink_ref,
    out_ref,
    out_tail_ref,
    q_ref,
    out_stage_ref,
    swa_kv_x2_ref,
    compressed_kv_ref,
    dma_semaphores,
    m_ref,
    l_ref,
    acc_ref,
    *,
    queries_per_block: int,
    compressed_tile: int,
    compressed_pages_per_tile: int,
    page_size: int,
    softmax_scale: float,
    q_compute_block_size: int,
    swa_dma_tile: int,
    swa_compute_tile: int,
    sublanes: int,
    may_cross_end: bool,
):
    """Request-level ragged HCA using TPU's production q/KV tile schedule."""
    # dma_semaphores: [0,1] SWA double buffer, [2] compressed, [3] output,
    # [4,5] q double buffer.
    query_block = pl.program_id(0)
    request = query_block_request_ids_ref[query_block]
    heads = q_ref.shape[2]  # padded to the sublane multiple, not the real count
    head_dim = q_ref.shape[3]
    tokens = q_hbm_ref.shape[0]
    query_base = query_block_offsets_ref[query_block]
    q_row = cu_q_lens_ref[request] + query_base
    tail_start = max(tokens - queries_per_block, 0)

    # q streams double-buffered across grid steps: step b waits b-1's fetch and
    # prefetches b+1.  Blocks running past ``q``'s end read the staged tail copy.
    def fetch_q(block, slot):
        row = cu_q_lens_ref[query_block_request_ids_ref[block]] + query_block_offsets_ref[block]
        semaphore = dma_semaphores.at[4 + slot]
        destination = q_ref.at[slot]
        if not may_cross_end:
            pltpu.make_async_copy(
                q_hbm_ref.at[pl.ds(row, queries_per_block)], destination, semaphore
            ).start()
            return

        @pl.when(row + queries_per_block <= tokens)
        def _from_tokens():
            pltpu.make_async_copy(
                q_hbm_ref.at[pl.ds(row, queries_per_block)], destination, semaphore
            ).start()

        @pl.when(row + queries_per_block > tokens)
        def _from_tail():
            tail_row = jnp.clip(row - tail_start, 0, queries_per_block)
            pltpu.make_async_copy(
                q_tail_hbm_ref.at[pl.ds(tail_row, queries_per_block)], destination, semaphore
            ).start()

    slot = jnp.bitwise_and(query_block, 1)

    @pl.when(query_block == 0)
    def _fetch_first_q():
        fetch_q(0, 0)

    q_len = q_lens_ref[request]
    prefix_len = prefix_lens_ref[request]
    valid_query_block = query_base < q_len
    query_locals = query_base + jnp.arange(queries_per_block, dtype=jnp.int32)
    query_positions = prefix_len + query_locals
    negative_finite = jnp.finfo(jnp.float32).min
    m_ref[...] = jnp.full(m_ref.shape, negative_finite, jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
    current_q = q_ref.at[slot]
    pltpu.make_async_copy(current_q, current_q, dma_semaphores.at[4 + slot]).wait()
    q = q_ref[slot].astype(jnp.bfloat16)
    q_flat = q.reshape(queries_per_block * heads, head_dim)

    def broadcast_minor(value, width):
        target = _align(width, value.shape[1])
        return jnp.concatenate([value for _ in range(target // value.shape[1])], axis=1)[:, :width]

    def consume(kv, key_positions, *, compressed):
        # Split the query schedule into platform-selected compute tiles so score/value
        # contractions raise MXU occupancy without changing KV reduction groups.
        for query_chunk in range(queries_per_block // q_compute_block_size):
            query_start = query_chunk * q_compute_block_size
            query_stop = query_start + q_compute_block_size
            row_start = query_start * heads
            row_count = q_compute_block_size * heads
            rows = pl.ds(row_start, row_count)
            chunk_locals = query_locals[query_start:query_stop]
            chunk_positions = query_positions[query_start:query_stop]
            if compressed:
                keep = key_positions[None, :] < (chunk_positions[:, None] + 1) // 128
            else:
                query_rows = 128 + chunk_locals
                delta = query_rows[:, None] - key_positions[None, :]
                keep = (
                    (delta.astype(jnp.uint32) < jnp.uint32(128))
                    & (chunk_locals[:, None] < q_len)
                    & (key_positions[None, :] >= jnp.maximum(128 - prefix_len, 0))
                )
            scores = jax.lax.dot_general(
                q_flat[row_start : row_start + row_count],
                kv.astype(jnp.bfloat16),
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ) * jnp.float32(softmax_scale)
            scores = scores.reshape(q_compute_block_size, heads, -1)
            scores = jnp.where(keep[:, None, :], scores, negative_finite).reshape(row_count, -1)
            block_maximum = jnp.max(scores, axis=1, keepdims=True)
            previous_maximum = m_ref[rows, ...]
            next_maximum = jnp.maximum(previous_maximum, block_maximum)
            alpha = jnp.exp(previous_maximum - next_maximum)
            probabilities = jnp.exp(scores - broadcast_minor(next_maximum, scores.shape[1]))
            next_l = alpha * l_ref[rows, ...] + jnp.sum(probabilities, axis=1, keepdims=True)
            value = jax.lax.dot_general(
                probabilities,
                kv.astype(jnp.bfloat16),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            acc_ref[rows, :] = broadcast_minor(alpha, head_dim) * acc_ref[rows, :] + value
            m_ref[rows, :] = next_maximum
            l_ref[rows, :] = next_l

    # ``combined_kv`` holds the previous 128 logical rows then the current chunk.
    # Start at the first real row: masking leading rows would shift reduction bounds.
    swa_start = jnp.maximum(query_base + 1, 128 - prefix_len)
    swa_end = jnp.minimum(128 + query_base + queries_per_block, 128 + q_len)
    num_half_blocks = jnp.where(
        valid_query_block,
        pl.cdiv(jnp.maximum(swa_end - swa_start, 0), swa_compute_tile),
        0,
    )
    num_swa_blocks = pl.cdiv(num_half_blocks, 2)

    def fetch_swa(block, buffer, *, wait):
        semaphore = dma_semaphores.at[buffer]
        destination = swa_kv_x2_ref.at[buffer]
        if wait:
            pltpu.make_async_copy(destination, destination, semaphore).wait()
        else:
            row = swa_start + block * swa_dma_tile
            pltpu.make_async_copy(
                combined_kv_hbm_ref.at[request, pl.ds(row, swa_dma_tile)],
                destination,
                semaphore,
            ).start()

    @pl.when(num_swa_blocks > 0)
    def _start_swa():
        fetch_swa(0, 0, wait=False)

    def consume_swa(block, buffer):
        fetch_swa(block, buffer, wait=True)
        next_block = block + 1
        next_buffer = jnp.bitwise_xor(buffer, 1)

        @pl.when(next_block < num_swa_blocks)
        def _prefetch():
            fetch_swa(next_block, next_buffer, wait=False)

        key_positions = swa_start + block * swa_dma_tile + jnp.arange(swa_dma_tile, dtype=jnp.int32)
        swa_kv = pltpu.bitcast(swa_kv_x2_ref[buffer, ...], jnp.bfloat16).reshape(
            swa_dma_tile, head_dim
        )
        consume(
            swa_kv[:swa_compute_tile],
            key_positions[:swa_compute_tile],
            compressed=False,
        )

        @pl.when(block * 2 + 1 < num_half_blocks)
        def _consume_second_half():
            consume(
                swa_kv[swa_compute_tile:],
                key_positions[swa_compute_tile:],
                compressed=False,
            )

        return next_buffer

    jax.lax.fori_loop(0, num_swa_blocks, consume_swa, jnp.int32(0), unroll=False)
    acc_ref[...] = acc_ref[...].astype(jnp.bfloat16).astype(jnp.float32)

    query_end = jnp.minimum(query_base + queries_per_block, q_len)
    max_compressed = jnp.where(
        valid_query_block,
        jnp.minimum(
            (prefix_len + query_end) // 128,
            compressed_lens_ref[request],
        ),
        0,
    )
    num_compressed_blocks = pl.cdiv(max_compressed, compressed_tile)
    cache_rows = compressed_cache_hbm_ref.reshape(-1, head_dim)
    compressed_start = compressed_page_starts_ref[request]

    @pl.when(num_compressed_blocks > 0)
    def _zero_compressed_tile():
        # Preserve TPU's 2048-wide reduction ABI without reading nonexistent
        # pages: masked tail rows are finite zeros, as in the block-KV prologue.
        compressed_kv_ref[...] = jnp.zeros(compressed_kv_ref.shape, compressed_kv_ref.dtype)

    def fetch_compressed(block, *, wait):
        # One buffer, unlike the double-buffered SWA and q fetches: the tile is
        # large and the loop short enough that VMEM is better spent elsewhere.
        semaphore = dma_semaphores.at[2]
        destination = compressed_kv_ref.at[0]
        valid_pages = pl.cdiv(
            jnp.maximum(max_compressed - block * compressed_tile, 0),
            page_size,
        )
        transfer_rows = valid_pages * page_size
        if wait:
            if page_size < sublanes:
                # A Mosaic DMA wait slice must be tile-aligned, so small pages
                # issue a full tile below; query masking hides the padded rows.
                pltpu.make_async_copy(destination, destination, semaphore).wait()
            else:
                transferred = destination.at[pl.ds(0, transfer_rows)]
                pltpu.make_async_copy(transferred, transferred, semaphore).wait()
        else:
            for page in range(compressed_pages_per_tile):
                if page_size < sublanes:
                    table_location = jnp.minimum(
                        compressed_start + block * compressed_pages_per_tile + page,
                        compressed_page_indices_ref.shape[0] - 1,
                    )
                    physical_page = compressed_page_indices_ref[table_location]
                    pltpu.make_async_copy(
                        cache_rows.at[pl.ds(physical_page * page_size, page_size)],
                        destination.at[pl.ds(page * page_size, page_size)],
                        semaphore,
                    ).start()
                    continue

                @pl.when(page < valid_pages)
                def _copy_valid_page(page=page):
                    table_location = compressed_start + block * compressed_pages_per_tile + page
                    physical_page = compressed_page_indices_ref[table_location]
                    pltpu.make_async_copy(
                        cache_rows.at[pl.ds(physical_page * page_size, page_size)],
                        destination.at[pl.ds(page * page_size, page_size)],
                        semaphore,
                    ).start()

    @pl.when(num_compressed_blocks > 0)
    def _start_compressed():
        fetch_compressed(0, wait=False)

    # Issue the next block's q fetch once the latency-critical SWA chain is
    # done; the compressed stage, epilogue, and step boundary hide it.
    @pl.when(query_block + 1 < pl.num_programs(0))
    def _prefetch_next_q():
        fetch_q(query_block + 1, 1 - slot)

    def consume_compressed(block, carry):
        fetch_compressed(block, wait=True)
        key_positions = block * compressed_tile + jnp.arange(compressed_tile, dtype=jnp.int32)
        consume(compressed_kv_ref[0, ...], key_positions, compressed=True)

        @pl.when(block + 1 < num_compressed_blocks)
        def _fetch_next():
            fetch_compressed(block + 1, wait=False)

        return carry

    jax.lax.fori_loop(0, num_compressed_blocks, consume_compressed, None, unroll=False)

    sink = attention_sink_ref[...].astype(jnp.float32)
    for query in range(queries_per_block):
        head_slice = pl.ds(query * heads, heads)
        denominator = l_ref[head_slice, ...] + jnp.exp(sink[:, None] - m_ref[head_slice, ...])
        # Stage into a dedicated buffer: writing back into the q slots would
        # order these stores against the in-flight next-block q prefetch.
        out_stage_ref[query, ...] = (
            acc_ref[head_slice] * pl.reciprocal(broadcast_minor(denominator, head_dim), approx=True)
        ).astype(jnp.bfloat16)
    # Spill into a later request's rows is safe: the grid runs in order and each
    # DMA is waited, so the owner overwrites it.  End-crossers use the tail output.
    if not may_cross_end:
        pltpu.make_async_copy(
            out_stage_ref,
            out_ref.at[pl.ds(q_row, queries_per_block)],
            dma_semaphores.at[3],
        ).start()
    else:

        @pl.when(q_row + queries_per_block <= tokens)
        def _store_tokens():
            pltpu.make_async_copy(
                out_stage_ref,
                out_ref.at[pl.ds(q_row, queries_per_block)],
                dma_semaphores.at[3],
            ).start()

        @pl.when(q_row + queries_per_block > tokens)
        def _store_tail():
            tail_row = jnp.clip(q_row - tail_start, 0, queries_per_block)
            pltpu.make_async_copy(
                out_stage_ref,
                out_tail_ref.at[pl.ds(tail_row, queries_per_block)],
                dma_semaphores.at[3],
            ).start()

    pltpu.make_async_copy(out_stage_ref, out_stage_ref, dma_semaphores.at[3]).wait()


@functools.partial(
    jax.jit,
    static_argnames=(
        "softmax_scale",
        "queries_per_block",
        "may_cross_end",
        "interpret",
        "schedule",
    ),
)
def _chunk_attention(
    q,
    cu_q_lens,
    combined_kv,
    q_lens,
    prefix_lens,
    compressed_lens,
    query_block_request_ids,
    query_block_offsets,
    compressed_cache,
    compressed_page_indices,
    compressed_page_starts,
    attention_sink,
    *,
    schedule: HCAKernelSchedule,
    softmax_scale: float,
    queries_per_block: int | None = None,
    may_cross_end: bool = True,
    interpret: bool | None = None,
):
    if queries_per_block is None:
        queries_per_block = schedule.query_block_size
    tokens, heads, head_dim = q.shape
    num_query_blocks = query_block_request_ids.shape[0]
    batch = combined_kv.shape[0]
    padded_heads = _align(heads, schedule.sublanes)
    page_size = compressed_cache.shape[1] * compressed_cache.shape[2]
    if padded_heads != heads:
        # Only sublane-unaligned head counts pay a staging pad.
        q = jnp.pad(q, ((0, 0), (0, padded_heads - heads), (0, 0)))
    # A request's tail block can run past ``q``'s end, so re-stage that span.
    # Aligned uniform prefill cannot cross it and skips this (may_cross_end=False).
    tail_start = max(tokens - queries_per_block, 0)
    if may_cross_end:
        q_tail = jnp.pad(
            q[tail_start:],
            ((0, 2 * queries_per_block - (tokens - tail_start)), (0, 0), (0, 0)),
        )
    else:
        q_tail = jnp.zeros((2 * queries_per_block, padded_heads, head_dim), q.dtype)
    # Each window DMA reads ``swa_dma_tile`` rows from an arbitrary start, so
    # the buffer needs a tile of slack behind the last query row.
    combined_kv = jnp.pad(combined_kv, ((0, 0), (0, schedule.swa_dma_tile), (0, 0)))
    packed_u16 = jax.lax.bitcast_convert_type(combined_kv, jnp.uint16).reshape(
        batch,
        combined_kv.shape[1],
        head_dim // schedule.mxu_lanes,
        schedule.mxu_lanes,
    )
    low = jnp.bitwise_and(packed_u16, jnp.uint16(0xFF)).astype(jnp.uint8)
    high = jnp.right_shift(packed_u16, jnp.uint16(8)).astype(jnp.uint8)
    combined_kv = jnp.stack((low, high), axis=3).reshape(
        batch,
        combined_kv.shape[1],
        2 * head_dim // schedule.mxu_lanes,  # two bytes per bf16, lane-major
        schedule.mxu_lanes,
    )
    attention_sink = jnp.pad(
        attention_sink.astype(jnp.float32),
        (0, padded_heads - heads),
        constant_values=-jnp.inf,
    )
    if interpret is None:
        interpret = _get_interpret()
    compressed_tile = schedule.compressed_tile
    if compressed_tile % page_size:
        raise ValueError("compressed_tile must contain whole cache pages")
    output, output_tail = pl.pallas_call(
        functools.partial(
            _chunk_attention_kernel,
            queries_per_block=queries_per_block,
            compressed_tile=compressed_tile,
            compressed_pages_per_tile=compressed_tile // page_size,
            page_size=page_size,
            softmax_scale=float(softmax_scale),
            q_compute_block_size=min(schedule.query_compute_block_size, queries_per_block),
            swa_dma_tile=schedule.swa_dma_tile,
            swa_compute_tile=schedule.swa_compute_tile,
            sublanes=schedule.sublanes,
            may_cross_end=may_cross_end,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=8,
            grid=(num_query_blocks,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((padded_heads,), lambda block, *_: (0,)),
            ),
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            scratch_shapes=(
                pltpu.VMEM((2, queries_per_block, padded_heads, head_dim), jnp.bfloat16),
                pltpu.VMEM((queries_per_block, padded_heads, head_dim), jnp.bfloat16),
                pltpu.VMEM(
                    (
                        2,
                        schedule.swa_dma_tile,
                        2 * head_dim // schedule.mxu_lanes,
                        schedule.mxu_lanes,
                    ),
                    jnp.uint8,
                ),
                pltpu.VMEM((1, compressed_tile, head_dim), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((6,)),
                pltpu.VMEM((queries_per_block * padded_heads, schedule.mxu_lanes), jnp.float32),
                pltpu.VMEM((queries_per_block * padded_heads, schedule.mxu_lanes), jnp.float32),
                pltpu.VMEM((queries_per_block * padded_heads, head_dim), jnp.float32),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, padded_heads, head_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct((2 * queries_per_block, padded_heads, head_dim), jnp.bfloat16),
        ),
        compiler_params=pltpu.CompilerParams(
            # "arbitrary": grid steps run in order, which the cross-step q
            # prefetch depends on.
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"hca-chunk-prefill-q{queries_per_block}-c{compressed_tile}",
    )(
        compressed_page_indices.astype(jnp.int32),
        compressed_page_starts.astype(jnp.int32),
        q_lens.astype(jnp.int32),
        prefix_lens.astype(jnp.int32),
        compressed_lens.astype(jnp.int32),
        query_block_request_ids.astype(jnp.int32),
        query_block_offsets.astype(jnp.int32),
        cu_q_lens.astype(jnp.int32),
        q,
        q_tail,
        combined_kv,
        compressed_cache,
        attention_sink,
    )
    if not may_cross_end:
        return output[:, :heads]
    # Rows owned by a block whose span crosses the array end live in the tail
    # output; merge them back.  Ownership mirrors the kernel's branch exactly.
    lo = max(tokens - queries_per_block, 0)
    rows = lo + jnp.arange(tokens - lo, dtype=jnp.int32)
    row_request = jnp.clip(
        jnp.searchsorted(cu_q_lens, rows, side="right").astype(jnp.int32) - 1,
        0,
        cu_q_lens.shape[0] - 2,
    )
    starts = cu_q_lens[row_request]
    owner_base = starts + ((rows - starts) // queries_per_block) * queries_per_block
    tail_owned = owner_base + queries_per_block > tokens
    output = output.at[lo:].set(
        jnp.where(tail_owned[:, None, None], output_tail[: tokens - lo], output[lo:])
    )
    return output[:, :heads]


@functools.partial(
    jax.jit,
    static_argnames=(
        "softmax_scale",
        "window_size",
        "compress_ratio",
        "schedule",
    ),
    donate_argnums=(2, 3),
)
def ragged_attention(
    q,
    new_kv,
    window_cache,
    compressed_cache,
    positions,
    compressed_write_values,
    compressed_write_mask,
    attention_sink,
    metadata,
    *,
    schedule: HCAKernelSchedule,
    softmax_scale: float,
    window_size: int = 128,
    compress_ratio: int = 128,
):
    """Cache-aware ragged HCA for fresh/chunked prefill, decode, and mixed batches.

    Queries and new KV are flattened in request-major order. Historical SWA rows
    are read through the framework page table while current-chunk rows are consumed
    directly, so updating a wrapped ring cannot corrupt an early query in the same
    chunk. One Pallas program consumes SWA first and then double-buffered physical
    compressed pages while retaining the same online-softmax state. Neither scores,
    ``[SWA | compressed]``, nor request-major compressed KV is built in HBM.
    """
    window_page_indices = metadata.window_page_indices
    window_cu_kv_lens = metadata.window_cu_kv_lens
    compressed_page_indices = metadata.compressed_page_indices
    compressed_cu_kv_lens = metadata.compressed_cu_kv_lens
    query_seq_ids = metadata.query_seq_ids
    cu_q_lens = metadata.cu_q_lens
    seq_lens = metadata.seq_lens
    compressed_kv_lens = metadata.compressed_kv_lens
    valid_token_mask = metadata.valid_token_mask
    query_block_request_ids = metadata.query_block_request_ids
    query_block_offsets = metadata.query_block_offsets
    decode_request_ids = metadata.decode_request_ids
    max_queries_per_request = metadata.max_queries_per_request

    tokens, heads, head_dim = q.shape
    batch = seq_lens.shape[0]
    # q/new_kv/attention_sink shapes and the D/window/ratio constants are the
    # backend's contract, checked there. What is worth re-checking here is that
    # the bucketed metadata agrees with the token count, because a padding or
    # capacity mistake would otherwise surface deep inside Pallas.
    if query_seq_ids.shape != (tokens,) or positions.shape != (tokens,):
        raise ValueError("query_seq_ids and positions must both be [T]")
    if valid_token_mask.shape != (tokens,):
        raise ValueError("valid_token_mask must be [T]")
    if cu_q_lens.shape != (batch + 1,):
        raise ValueError(f"cu_q_lens must be [{batch + 1}]")
    if compressed_write_values.shape != (tokens, head_dim):
        raise ValueError(f"compressed_write_values must be [{tokens},{head_dim}]")
    if compressed_write_mask.shape != (tokens,):
        raise ValueError("compressed_write_mask must be [T]")
    if compressed_kv_lens.shape != (batch,):
        raise ValueError("compressed_kv_lens must be [B]")

    query_seq_ids = query_seq_ids.astype(jnp.int32)
    valid_token_mask = valid_token_mask.astype(jnp.bool_)
    window_flat, window_page_size = _cache_layout(window_cache, head_dim)
    compressed_flat, compressed_page_size = _cache_layout(compressed_cache, head_dim)
    if window_page_size != compressed_page_size:
        raise ValueError("window and compressed cache page sizes must match")
    # The q-block DMA path slices VMEM/HBM on TPU's eight-row tile boundary.
    if window_page_size % schedule.sublanes:
        raise ValueError(f"HCA page_size must be a multiple of {schedule.sublanes}")

    # Install all new compressed entries first. The per-query prefix mask below
    # prevents a query from observing a later boundary in its own EXTEND chunk.
    compressed_entries = jnp.floor_divide(positions + 1, compress_ratio) - 1
    compressed_write_locs, compressed_write_valid = _page_table_locations(
        compressed_page_indices,
        compressed_cu_kv_lens,
        compressed_entries,
        valid_token_mask & compressed_write_mask & (compressed_entries >= 0),
        page_size=compressed_page_size,
        physical_rows=compressed_flat.shape[0],
        sequence_ids=query_seq_ids,
    )
    # HCA emits at most one value for each absolute compression boundary, so
    # compressed destinations are unique inside a forward call.
    if jax.default_backend() == "tpu":
        compressed_cache = _write_cache_rows(
            compressed_cache,
            compressed_write_locs,
            compressed_write_values,
            compressed_write_valid,
            schedule=schedule,
        )
        compressed_flat = compressed_cache.reshape(-1, compressed_cache.shape[-1])
    else:
        compressed_flat = _scatter_physical_rows(
            compressed_flat,
            compressed_write_locs,
            compressed_write_values,
            compressed_write_valid,
        )
        compressed_cache = compressed_flat.reshape(compressed_cache.shape)

    q_lens = jnp.diff(cu_q_lens).astype(jnp.int32)
    prefix_lens = seq_lens.astype(jnp.int32) - q_lens
    max_queries = int(max_queries_per_request)
    if max_queries < 1:
        raise ValueError("max_queries_per_request must be positive")
    compressed_page_starts_by_request = compressed_cu_kv_lens[:-1] // jnp.int32(
        compressed_page_size
    )

    if max_queries == 1:
        # Row ``j`` holds absolute position ``window_start + j``: below
        # ``prefix_lens`` from the ring, the decode token from ``new_kv``.
        window_start = jnp.maximum(seq_lens.astype(jnp.int32) - window_size, 0)
        window_positions = window_start[:, None] + jnp.arange(window_size, dtype=jnp.int32)[None, :]
        # Clamp so rows past the newest position cannot index outside this
        # request's table segment; ``cache_valid`` discards what they read.
        lookup_positions = jnp.minimum(window_positions, prefix_lens[:, None])
        window_pages, window_locs = _window_locations(
            window_page_indices,
            window_cu_kv_lens,
            lookup_positions,
            page_size=window_page_size,
        )
        cache_valid = (window_positions < prefix_lens[:, None]) & (window_pages > 0)
        cache_rows = _gather_physical_rows(window_flat, window_locs, cache_valid, head_dim)
        is_new = window_positions == prefix_lens[:, None]
        window_rows = jnp.where(is_new[..., None], new_kv[:, None, :], cache_rows)
        # TPU's 2048-row reduction contains the same first 128-row subtree;
        # use it only once multiple 128-row groups can affect online softmax.
        output = _streaming_attention(
            q,
            window_rows,
            jnp.minimum(seq_lens, window_size),
            compressed_cache,
            compressed_page_indices,
            compressed_page_starts_by_request,
            jnp.minimum(seq_lens // compress_ratio, compressed_kv_lens),
            attention_sink,
            schedule=schedule,
            softmax_scale=softmax_scale,
        )
        window_flat = _commit_window_rows(
            window_flat,
            window_page_indices,
            window_cu_kv_lens,
            positions,
            new_kv,
            valid_token_mask,
            query_seq_ids,
            seq_lens,
            window_size=window_size,
            page_size=window_page_size,
        )
        return (
            output,
            window_flat.reshape(window_cache.shape),
            compressed_cache,
        )

    history_positions = (
        prefix_lens[:, None] - window_size + jnp.arange(window_size, dtype=jnp.int32)[None, :]
    )
    history_pages, history_locs = _window_locations(
        window_page_indices,
        window_cu_kv_lens,
        jnp.maximum(history_positions, 0),
        page_size=window_page_size,
    )
    history_valid = (history_positions >= 0) & (history_pages > 0)
    history = _gather_physical_rows(window_flat, history_locs, history_valid, head_dim)

    local_queries = jnp.arange(tokens, dtype=jnp.int32) - cu_q_lens[query_seq_ids]
    kv_padded = (
        jnp.zeros((batch, max_queries, head_dim), new_kv.dtype)
        .at[query_seq_ids, local_queries]
        .set(new_kv)
    )
    combined_kv = jnp.concatenate((history, kv_padded), axis=1)
    if query_block_request_ids.shape[0]:
        mixed_output = _chunk_attention(
            q,
            cu_q_lens,
            combined_kv,
            q_lens,
            prefix_lens,
            compressed_kv_lens,
            query_block_request_ids,
            query_block_offsets,
            compressed_cache,
            compressed_page_indices,
            compressed_page_starts_by_request,
            attention_sink,
            schedule=schedule,
            softmax_scale=softmax_scale,
        )
    else:
        mixed_output = jnp.zeros_like(q)
    if decode_request_ids.shape[0]:
        # Ids padded with -1 get an inert offset (kernel skips their KV loops)
        # and an out-of-bounds scatter target that ``mode="drop"`` discards.
        decode_valid = decode_request_ids >= 0
        safe_decode_ids = jnp.maximum(decode_request_ids, 0)
        decode_output = _chunk_attention(
            q,
            cu_q_lens,
            combined_kv,
            q_lens,
            prefix_lens,
            compressed_kv_lens,
            safe_decode_ids,
            jnp.where(decode_valid, 0, jnp.int32(INERT_QUERY_OFFSET)),
            compressed_cache,
            compressed_page_indices,
            compressed_page_starts_by_request,
            attention_sink,
            schedule=schedule,
            softmax_scale=softmax_scale,
            queries_per_block=1,
        )
        decode_scatter_indices = jnp.where(decode_valid, cu_q_lens[safe_decode_ids], tokens)
        decode_rows = decode_output[jnp.clip(decode_scatter_indices, 0, tokens - 1)]
        output = mixed_output.at[decode_scatter_indices].set(decode_rows, mode="drop")
    else:
        output = mixed_output
    output = jnp.where(valid_token_mask[:, None, None], output, 0.0)
    window_flat = _commit_window_rows(
        window_flat,
        window_page_indices,
        window_cu_kv_lens,
        positions,
        new_kv,
        valid_token_mask,
        query_seq_ids,
        seq_lens,
        window_size=window_size,
        page_size=window_page_size,
    )
    return (
        output,
        window_flat.reshape(window_cache.shape),
        compressed_cache,
    )


@functools.partial(
    jax.jit,
    static_argnames=("softmax_scale", "window_size", "compress_ratio", "schedule"),
    donate_argnums=(2, 3),
)
def uniform_prefill_attention(
    q,
    new_kv,
    window_cache,
    compressed_cache,
    compressed_write_values,
    attention_sink,
    metadata,
    *,
    schedule: HCAKernelSchedule,
    softmax_scale: float,
    window_size: int = 128,
    compress_ratio: int = 128,
):
    """Run fresh-prompt HCA over physical SGLang pages.

    The current prompt KV remains a direct attention input, matching chunk-prefill
    attention in serving frameworks.  The final sliding-window state and every HCA
    boundary record are nevertheless written through their page tables inside this
    call.  Compressed records are then gathered back through the physical page table
    before attention, so an incorrect page mapping cannot pass validation.
    """
    batch = metadata.seq_lens.shape[0]
    tokens, heads, head_dim = q.shape
    # q/new_kv/attention_sink shapes are the backend's contract. This one is
    # load-bearing here: the dense [B, S] view below is only valid when the
    # token dimension really is B equal-length requests with no padding.
    if batch < 1 or tokens % batch:
        raise ValueError("uniform prefill requires T divisible by B")
    sequence = tokens // batch
    if compressed_write_values.shape != (tokens, head_dim):
        raise ValueError(f"compressed_write_values must be [{tokens},{head_dim}]")

    q = q.reshape(batch, sequence, heads, head_dim)
    new_kv = new_kv.reshape(batch, sequence, head_dim)
    complete_entries = sequence // compress_ratio
    entry = jnp.arange(complete_entries, dtype=jnp.int32)[None, :]
    boundary_tokens = (
        jnp.arange(batch, dtype=jnp.int32)[:, None] * sequence + (entry + 1) * compress_ratio - 1
    )
    compressed_write_values = compressed_write_values[boundary_tokens]
    compressed_write_lens = jnp.full((batch,), complete_entries, jnp.int32)
    window_page_indices = metadata.window_page_indices
    window_cu_kv_lens = metadata.window_cu_kv_lens
    compressed_page_indices = metadata.compressed_page_indices
    compressed_cu_kv_lens = metadata.compressed_cu_kv_lens

    window_flat, window_page_size = _cache_layout(window_cache, head_dim)
    compressed_flat, compressed_page_size = _cache_layout(compressed_cache, head_dim)
    if window_page_size != compressed_page_size:
        raise ValueError("window and compressed cache page sizes must match")

    # Only the last window survives in serving state; limiting the scatter to
    # those rows also avoids duplicate destinations when the prompt wraps the ring.
    window_start = max(0, sequence - window_size)
    window_positions = jnp.broadcast_to(
        jnp.arange(window_start, sequence, dtype=jnp.int32)[None, :],
        (batch, sequence - window_start),
    )
    window_locs, window_valid = _page_table_locations(
        window_page_indices,
        window_cu_kv_lens,
        window_positions,
        jnp.ones_like(window_positions, dtype=jnp.bool_),
        page_size=window_page_size,
        physical_rows=window_flat.shape[0],
    )
    window_flat = _scatter_physical_rows(
        window_flat,
        window_locs,
        new_kv[:, window_start:],
        window_valid,
    )

    compressed_locs, resolved_write_valid = _page_table_locations(
        compressed_page_indices,
        compressed_cu_kv_lens,
        jnp.broadcast_to(entry, (batch, complete_entries)),
        jnp.ones((batch, complete_entries), jnp.bool_),
        page_size=compressed_page_size,
        physical_rows=compressed_flat.shape[0],
    )
    compressed_flat = _scatter_physical_rows(
        compressed_flat,
        compressed_locs,
        compressed_write_values,
        resolved_write_valid,
    )
    compressed_page_starts = compressed_cu_kv_lens[:-1] // jnp.int32(compressed_page_size)
    queries_per_block = schedule.query_block_size
    padded_sequence = _align(sequence, queries_per_block)
    blocks_per_request = padded_sequence // queries_per_block
    request_ids = jnp.repeat(jnp.arange(batch, dtype=jnp.int32), blocks_per_request)
    block_offsets = jnp.tile(
        jnp.arange(blocks_per_request, dtype=jnp.int32) * queries_per_block,
        batch,
    )
    combined_kv = jnp.concatenate(
        (jnp.zeros((batch, window_size, head_dim), new_kv.dtype), new_kv),
        axis=1,
    )
    output = _chunk_attention(
        q.reshape(batch * sequence, heads, head_dim),
        jnp.arange(batch + 1, dtype=jnp.int32) * sequence,
        combined_kv,
        jnp.full((batch,), sequence, jnp.int32),
        jnp.zeros((batch,), jnp.int32),
        compressed_write_lens,
        request_ids,
        block_offsets,
        compressed_flat.reshape(compressed_cache.shape),
        compressed_page_indices,
        compressed_page_starts,
        attention_sink,
        schedule=schedule,
        softmax_scale=softmax_scale,
        queries_per_block=queries_per_block,
        may_cross_end=sequence % queries_per_block != 0,
    )
    return (
        output,
        window_flat.reshape(window_cache.shape),
        compressed_flat.reshape(compressed_cache.shape),
    )


__all__ = ["INERT_QUERY_OFFSET", "ragged_attention", "uniform_prefill_attention"]
