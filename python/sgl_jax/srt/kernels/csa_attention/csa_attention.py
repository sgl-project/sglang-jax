"""Read-only paged CSA: compact gather and joint SWA/compressed attention."""

import functools
import math
from typing import NamedTuple

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .tune import (
    APPEND_TILE,
    DMA_DEPTH,
    LANES,
    MAX_QUERY_TILE,
    ROUTE_QUERY_TILE,
    SUBLANES,
    CSAAttentionSchedule,
)


class CSAAttentionMetadata(NamedTuple):
    """Caller-owned int32 arrays; token/request padding uses -1, page zero is dummy.

    query_seq_ids:[T], cu_q_lens:[B+1], seq_lens:[B]. Flattened page tables
    use page-aligned cu_*_kv_lens:[B+1] in tokens/entries, as in HCA.
    Window tables address a window_size-token ring before this chunk. Compressed
    lengths:[B] count completed entries. Nonnegative Top-K entries must be unique.
    """

    query_seq_ids: jax.Array
    cu_q_lens: jax.Array
    seq_lens: jax.Array
    window_page_indices: jax.Array
    window_cu_kv_lens: jax.Array
    compressed_page_indices: jax.Array
    compressed_cu_kv_lens: jax.Array
    compressed_kv_lens: jax.Array


def _broadcast_minor(value, width):
    return jnp.concatenate((value,) * pl.cdiv(width, value.shape[1]), axis=1)[:, :width]


def attention_update(q, kv, valid, m_ref, l_ref, acc_ref, *, scale, kv_transposed=False):
    scores = jax.lax.dot_general(
        q,
        kv,
        (((1,), (0 if kv_transposed else 1,)), ((), ())),
        preferred_element_type=jnp.float32,
    ) * jnp.float32(scale)
    scores = jnp.where(valid, scores, -jnp.inf)
    maximum = jnp.maximum(m_ref[...], jnp.max(scores, axis=1, keepdims=True))
    correction = jnp.exp(m_ref[...] - maximum)
    probability = jnp.exp(scores - _broadcast_minor(maximum, scores.shape[1]))
    denominator = correction * l_ref[...] + jnp.sum(probability, axis=1, keepdims=True)
    product = jax.lax.dot_general(
        probability.astype(jnp.bfloat16),
        kv,
        (((1,), (1 if kv_transposed else 0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = _broadcast_minor(correction, q.shape[1]) * acc_ref[...] + product
    m_ref[...] = jnp.broadcast_to(maximum, m_ref.shape)
    l_ref[...] = jnp.broadcast_to(denominator, l_ref.shape)


@functools.partial(jax.jit, static_argnames=("panel_size", "interpret"))
def prepare_routes(indices, *, panel_size, interpret=False):
    tokens, selected = indices.shape
    blocks = selected // LANES
    levels = selected.bit_length() - 1

    def kernel(src, dst):
        order = jnp.arange(selected, dtype=jnp.int32).reshape(1, blocks, LANES)
        ordered = jnp.where(src[...] >= 0, src[...], jnp.iinfo(jnp.int32).max).reshape(
            ROUTE_QUERY_TILE, blocks, LANES
        )
        # Bitonic compare-exchange: lane permutations within each sublane, then across them.
        for level in range(1, levels + 1):
            span = 1 << level
            for power in range(level - 1, -1, -1):
                stride = 1 << power
                if stride < LANES:
                    partner = jnp.take_along_axis(
                        ordered,
                        jnp.broadcast_to(
                            (jnp.arange(LANES) ^ stride)[None, None, :], ordered.shape
                        ),
                        axis=2,
                    )
                else:
                    partner = jnp.concatenate(
                        tuple(
                            ordered[:, (i ^ (stride // LANES)) : (i ^ (stride // LANES)) + 1]
                            for i in range(blocks)
                        ),
                        axis=1,
                    )
                take_min = ((order & span) == 0) == ((order & stride) == 0)
                ordered = jnp.where(
                    take_min, jnp.minimum(ordered, partner), jnp.maximum(ordered, partner)
                )
        ordered = ordered.reshape(ROUTE_QUERY_TILE, selected)
        panels = ordered // panel_size
        positions = jnp.arange(selected)[None, :]
        next_panel = jnp.concatenate(
            (panels[:, 1:], jnp.full((ROUTE_QUERY_TILE, 1), -1, jnp.int32)), axis=1
        )
        ends = jnp.where(panels != next_panel, positions + 1, selected)
        for power in range(levels):
            stride = 1 << power
            following = pltpu.roll(ends, selected - stride, axis=1)
            ends = jnp.minimum(ends, jnp.where(positions + stride < selected, following, selected))
        dst[:, 0:1] = ordered[:, None, :]
        dst[:, 1:2] = ends[:, None, :]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tokens, 2, selected), jnp.int32),
        grid=(pl.cdiv(tokens, ROUTE_QUERY_TILE),),
        in_specs=[pl.BlockSpec((ROUTE_QUERY_TILE, selected), lambda b: (b, 0))],
        out_specs=pl.BlockSpec((ROUTE_QUERY_TILE, 2, selected), lambda b: (b, 0, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="csa-route-plan",
    )(indices)


def _decode_cache(codes, rope_bytes, *, nope_dim, fp8_scale_block, feature_axis):
    # Specialize the feature axis without transposing either gather layout.
    def features(array, start, end):
        return jax.lax.slice_in_dim(array, start, end, axis=feature_axis)

    values = pltpu.bitcast(features(codes, 0, nope_dim), jnp.float8_e4m3fn).astype(jnp.bfloat16)
    scale_count = nope_dim // fp8_scale_block
    scales = pltpu.bitcast(
        features(codes, nope_dim, nope_dim + scale_count), jnp.float8_e8m0fnu
    ).astype(jnp.bfloat16)
    block_shape = list(codes.shape)
    block_shape[feature_axis] = fp8_scale_block
    expanded = jnp.concatenate(
        tuple(
            jnp.broadcast_to(features(scales, i, i + 1), block_shape) for i in range(scale_count)
        ),
        axis=feature_axis,
    )
    rope_bytes = rope_bytes.astype(jnp.int32)
    rope_dim = rope_bytes.shape[feature_axis] // jnp.dtype(jnp.bfloat16).itemsize
    bits = (
        (features(rope_bytes, 0, rope_dim) << 8) | features(rope_bytes, rope_dim, 2 * rope_dim)
    ).astype(jnp.uint16)
    rope = pltpu.bitcast(bits, jnp.bfloat16)
    return jnp.concatenate((values * expanded, rope), axis=feature_axis)


def _kernel(
    requests,
    cu_q,
    seq_lens,
    window_pages,
    window_cu,
    compressed_pages,
    compressed_cu,
    compressed_lens,
    q_ref,
    new_kv,
    window_cache,
    nope_cache,
    rope_cache,
    topk,
    sink_ref,
    out_ref,
    window_ref,
    fresh_ref,
    nope_ref,
    rope_ref,
    page_active,
    sem,
    m_ref,
    l_ref,
    acc_ref,
    kv_tile,
    valid_tile,
    *gather_scratch,
    query_tile,
    selected_tile,
    dma_tile,
    page_size,
    window_page_size,
    window_size,
    compression_ratio,
    fp8_scale_block,
    rows_per_group,
    scale,
):
    if query_tile == 1:
        compact_codes, compact_valid, route_vector, route_scalar, route_state, route_sem = (
            gather_scratch
        )
    else:
        (selected_valid,) = gather_scratch
    block = pl.program_id(0)
    heads, head_dim = q_ref.shape[1:]
    word_bytes = jnp.dtype(jnp.uint32).itemsize
    top_k = topk.shape[-1]
    nope_record_bytes = math.prod(nope_cache.shape[-2:])
    rope_record_bytes = rope_cache.shape[-1]
    rope_dim = rope_cache.shape[-1] // jnp.dtype(jnp.bfloat16).itemsize
    nope_dim = head_dim - rope_dim
    token_ids = block * query_tile + jnp.arange(query_tile)
    request_ids = jnp.stack(
        tuple(
            requests[jnp.minimum(block * query_tile + i, requests.shape[0] - 1)]
            for i in range(query_tile)
        )
    )
    out_ref[...] = jnp.zeros(out_ref.shape, jnp.bfloat16)

    def page_location(table, offsets, request, logical, size, num_pages):
        start, end = offsets[request] // size, offsets[request + 1] // size
        location = start + logical // size
        valid = (logical >= 0) & (location >= start) & (location < end)
        valid &= (location >= 0) & (location < table.shape[0])
        page = table[jnp.where(valid, location, 0)]
        valid &= (page > 0) & (page < num_pages)
        return jnp.where(valid, page, 0), logical % size, valid

    def request_group(local, _):
        token = block * query_tile + local
        request = requests[jnp.minimum(token, requests.shape[0] - 1)]
        safe_request = jnp.clip(request, 0, seq_lens.shape[0] - 1)
        active = (token < requests.shape[0]) & (request >= 0) & (request < seq_lens.shape[0])
        active &= (token >= cu_q[safe_request]) & (token < cu_q[safe_request + 1])
        active &= (local == 0) | (
            requests[jnp.clip(token - 1, 0, requests.shape[0] - 1)] != request
        )

        @pl.when(active)
        def attend():
            prefix = seq_lens[safe_request] - (cu_q[safe_request + 1] - cu_q[safe_request])
            position = prefix + token_ids - cu_q[safe_request]
            members = (request_ids == request) & (token_ids < requests.shape[0])
            members &= (token_ids >= cu_q[safe_request]) & (token_ids < cu_q[safe_request + 1])
            member_bits = members.astype(jnp.int32)
            q = jnp.where(member_bits[:, None, None] != 0, q_ref[...], 0).reshape(
                query_tile * heads, head_dim
            )
            selected = jnp.where(
                member_bits[:, None] != 0, topk[:, :1].reshape(query_tile, top_k), -1
            )
            if query_tile == 1:

                @pl.when((prefix > 0) & (token - cu_q[safe_request] < window_size - 1))
                def prefetch_history():
                    page, _, exists = page_location(
                        window_pages,
                        window_cu,
                        safe_request,
                        0,
                        window_page_size,
                        window_cache.shape[0],
                    )

                    @pl.when(exists)
                    def transfer_history():
                        pltpu.make_async_copy(window_cache.at[page], window_ref, sem.at[0]).start()

            m_ref[...] = jnp.broadcast_to(
                sink_ref[...][None, :, None], (query_tile, heads, LANES)
            ).reshape(m_ref.shape)
            l_ref[...] = jnp.ones(l_ref.shape, jnp.float32)
            acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

            def consume(kv, valid):
                # Normalize the gathered layout once before both MXU contractions.
                count = kv.shape[0]
                kv_tile[:count] = kv.astype(jnp.float32)
                valid_tile[:count] = jnp.broadcast_to(
                    jnp.max(valid.astype(jnp.int32), axis=0)[:, None], (count, LANES)
                )
                row_valid = _broadcast_minor(valid_tile[:count], head_dim) != 0
                kv = jnp.where(row_valid, kv_tile[:count], 0).astype(jnp.bfloat16)
                attention_update(
                    q, kv, jnp.repeat(valid, heads, axis=0), m_ref, l_ref, acc_ref, scale=scale
                )

            capacity = compressed_cu[safe_request + 1] - compressed_cu[safe_request]
            visible_per_query = jnp.maximum(
                0,
                jnp.minimum(
                    jnp.minimum(compressed_lens[safe_request], (position + 1) // compression_ratio),
                    capacity,
                ),
            )
            visible = jnp.max(jnp.where(members, visible_per_query, 0))
            steps = pl.cdiv(visible, selected_tile)

            if query_tile == 1:
                keep_selected = (selected >= 0) & (selected < visible)
                count = jnp.sum(keep_selected.astype(jnp.int32))
                ordered = jnp.where(keep_selected, selected, jnp.iinfo(jnp.int32).max).reshape(
                    1, top_k
                )
                panel_size = math.gcd(page_size, LANES)
                ends = jnp.minimum(topk[0, 1:2], count)
                route_vector[0:1] = ordered
                route_vector[1:2] = ends
                copy = pltpu.make_async_copy(route_vector, route_scalar, route_sem)
                copy.start()
                route_state[0] = count
                route_state[1] = 0
                compact_codes[...] = jnp.zeros(compact_codes.shape, jnp.uint32)
                compact_valid[...] = jnp.zeros(compact_valid.shape, jnp.int32)

                def fetch_panel(position, buffer):
                    logical = (route_scalar[0, position] // panel_size) * panel_size
                    page, offset, exists = page_location(
                        compressed_pages,
                        compressed_cu,
                        safe_request,
                        logical,
                        page_size,
                        nope_cache.shape[0],
                    )
                    page_active[buffer, 0] = exists.astype(jnp.int32)

                    @pl.when(exists)
                    def transfer():
                        pltpu.make_async_copy(
                            nope_cache.at[page, pl.ds(offset, panel_size)],
                            nope_ref.at[buffer, pl.ds(0, panel_size)],
                            sem.at[2 + 2 * buffer],
                        ).start()
                        pltpu.make_async_copy(
                            rope_cache.at[
                                page, pl.ds(offset // rows_per_group, panel_size // rows_per_group)
                            ],
                            rope_ref.at[buffer, pl.ds(0, panel_size // rows_per_group)],
                            sem.at[3 + 2 * buffer],
                        ).start()

                copy.wait()

                route_state[2] = 0
                for slot in range(DMA_DEPTH - 1):

                    @pl.when(route_state[2] < count)
                    def prime(slot=slot):
                        cursor = route_state[2]
                        fetch_panel(cursor, slot)
                        route_state[2] = route_scalar[1, cursor]

                def consume_compact():
                    @pl.when(route_state[0] > route_state[1] * APPEND_TILE)
                    def attention():
                        # Restore feature order once per collected tile, not per source page.
                        codes = (
                            pltpu.bitcast(
                                compact_codes[: nope_record_bytes // word_bytes], jnp.uint8
                            )
                            .reshape(nope_record_bytes // word_bytes, word_bytes, APPEND_TILE)
                            .transpose(1, 0, 2)
                            .reshape(nope_record_bytes, APPEND_TILE)
                        )
                        rope_bytes = pltpu.bitcast(
                            compact_codes[nope_record_bytes // word_bytes :], jnp.uint8
                        )
                        kv_t = _decode_cache(
                            codes,
                            rope_bytes,
                            nope_dim=nope_dim,
                            fp8_scale_block=fp8_scale_block,
                            feature_axis=0,
                        )
                        kv_t = jnp.where(compact_valid[...] != 0, kv_t, 0)
                        attention_update(
                            q,
                            kv_t,
                            compact_valid[...] != 0,
                            m_ref,
                            l_ref,
                            acc_ref,
                            scale=scale,
                            kv_transposed=True,
                        )

            def next_block(previous):
                # Walk the union of selected blocks, not the entire context/page table.
                ids = selected // selected_tile
                keep = (selected >= 0) & (selected < visible_per_query[:, None]) & (ids > previous)
                return jnp.min(jnp.where(keep, ids, steps))

            def fetch(step, buffer):
                logical = step * selected_tile
                bits = jnp.iinfo(jnp.int32).bits
                keep = (selected >= 0) & (selected < visible_per_query[:, None])
                weights = jnp.where(keep, jnp.int32(1) << (selected & (bits - 1)), 0)
                # Unique Top-K entries set distinct bits: exact int32 sum is bitwise OR.
                mask = jnp.zeros((query_tile, selected_tile), jnp.int32)
                positions = jnp.arange(selected_tile)[None, :]
                for word_id in range(pl.cdiv(selected_tile, bits)):
                    word = jnp.sum(
                        jnp.where((selected & -bits) == logical + word_id * bits, weights, 0),
                        axis=1,
                        keepdims=True,
                    )
                    # Keep each word lane-replicated instead of packing then slicing.
                    word = jnp.broadcast_to(word, (query_tile, LANES))
                    unpacked = (
                        jax.lax.shift_right_logical(
                            _broadcast_minor(word, selected_tile), positions % bits
                        )
                        & 1
                    )
                    mask = mask | jnp.where(positions // bits == word_id, unpacked, 0)
                # Aggregate physical-page DMAs into one MXU-width attention tile.
                for part in range(selected_tile // dma_tile):
                    rows = pl.ds(part * dma_tile, dma_tile)
                    page, offset, exists = page_location(
                        compressed_pages,
                        compressed_cu,
                        safe_request,
                        logical + part * dma_tile,
                        page_size,
                        nope_cache.shape[0],
                    )
                    keep = mask[:, part * dma_tile : (part + 1) * dma_tile] * exists.astype(
                        jnp.int32
                    )
                    selected_valid[buffer, :, rows] = keep
                    # Reuse the scalar decision for DMA start, wait, and tile activity.
                    active = jnp.any(keep)
                    page_active[buffer, part] = active.astype(jnp.int32)

                    @pl.when(active)
                    def transfer(part=part, page=page, offset=offset, rows=rows):
                        semaphore = 2 + 2 * (buffer * (selected_tile // dma_tile) + part)
                        pltpu.make_async_copy(
                            nope_cache.at[page, pl.ds(offset, dma_tile)],
                            nope_ref.at[buffer, rows],
                            sem.at[semaphore],
                        ).start()
                        pltpu.make_async_copy(
                            rope_cache.at[
                                page, pl.ds(offset // rows_per_group, dma_tile // rows_per_group)
                            ],
                            rope_ref.at[
                                buffer,
                                pl.ds(
                                    part * dma_tile // rows_per_group, dma_tile // rows_per_group
                                ),
                            ],
                            sem.at[semaphore + 1],
                        ).start()

            if query_tile != 1:
                first_selected = next_block(-1)

                @pl.when(first_selected < steps)
                def start_compressed():
                    fetch(first_selected, 0)

            # Historical ring: whole physical pages, never one DMA per token.
            @pl.when((prefix > 0) & (token - cu_q[safe_request] < window_size - 1))
            def history():
                def page_step(part, _):
                    page, _, exists = page_location(
                        window_pages,
                        window_cu,
                        safe_request,
                        part * window_page_size,
                        window_page_size,
                        window_cache.shape[0],
                    )

                    @pl.when(exists)
                    def attend_page():
                        copy = pltpu.make_async_copy(window_cache.at[page], window_ref, sem.at[0])

                        @pl.when((part != 0) | (query_tile != 1))
                        def start_history():
                            copy.start()

                        copy.wait()
                        slots = part * window_page_size + jnp.arange(window_page_size)
                        absolute = prefix - 1 - (prefix - 1 - slots) % window_size
                        keep = (
                            (absolute[None, :] >= 0)
                            & (absolute[None, :] > position[:, None] - window_size)
                            & (member_bits[:, None] != 0)
                        )
                        consume(window_ref[...].reshape(window_page_size, head_dim), keep)

                jax.lax.fori_loop(0, window_size // window_page_size, page_step, None)

            # Current chunk remains separate from the historical ring during prefill.
            first = jnp.maximum(cu_q[safe_request], token - window_size + 1) // window_size
            last = (
                jnp.minimum((block + 1) * query_tile, cu_q[safe_request + 1]) - 1
            ) // window_size

            def fresh_step(part, _):
                copy = pltpu.make_async_copy(
                    new_kv.at[pl.ds(part * window_size, window_size)], fresh_ref, sem.at[1]
                )
                copy.start()
                copy.wait()
                rows = part * window_size + jnp.arange(window_size)
                keep = (
                    (rows[None, :] >= cu_q[safe_request])
                    & (rows[None, :] > token_ids[:, None] - window_size)
                    & (rows[None, :] <= token_ids[:, None])
                    & (member_bits[:, None] != 0)
                )
                consume(fresh_ref[...], keep)

            jax.lax.fori_loop(first, last + 1, fresh_step, None)

            if query_tile == 1:

                def panel_step(state):
                    position, buffer = state
                    end = route_scalar[1, position]
                    logical = (route_scalar[0, position] // panel_size) * panel_size
                    exists = page_active[buffer, 0] != 0

                    @pl.when(route_state[2] < route_state[0])
                    def prefetch():
                        cursor = route_state[2]
                        fetch_panel(cursor, (buffer + DMA_DEPTH - 1) % DMA_DEPTH)
                        route_state[2] = route_scalar[1, cursor]

                    @pl.when(exists)
                    def wait():
                        pltpu.make_async_copy(
                            nope_ref.at[buffer, pl.ds(0, panel_size)],
                            nope_ref.at[buffer, pl.ds(0, panel_size)],
                            sem.at[2 + 2 * buffer],
                        ).wait()
                        pltpu.make_async_copy(
                            rope_ref.at[buffer, pl.ds(0, panel_size // rows_per_group)],
                            rope_ref.at[buffer, pl.ds(0, panel_size // rows_per_group)],
                            sem.at[3 + 2 * buffer],
                        ).wait()

                    raw_rope = rope_ref[buffer, : LANES // rows_per_group].reshape(
                        LANES, rope_record_bytes
                    )
                    packed = jnp.concatenate(
                        (
                            pltpu.bitcast(nope_ref[buffer, :LANES], jnp.uint32)
                            .reshape(LANES, nope_record_bytes // word_bytes)
                            .T,
                            pltpu.bitcast(raw_rope.T, jnp.uint32),
                        ),
                        axis=0,
                    )

                    def gather_group():
                        group = route_state[1]
                        for part in range(APPEND_TILE // LANES):
                            first = group * APPEND_TILE + part * LANES

                            @pl.when((end > first) & (position < first + LANES))
                            def update(part=part, first=first):
                                selected_group = route_vector[0:1, pl.ds(first, LANES)]
                                relative = selected_group - logical
                                offsets = jnp.clip(relative, 0, LANES - 1)
                                belongs = (relative >= 0) & (relative < panel_size)
                                gathered = jnp.take_along_axis(
                                    packed,
                                    jnp.broadcast_to(offsets, (packed.shape[0], LANES)),
                                    axis=1,
                                )
                                compact_codes[:, part * LANES : (part + 1) * LANES] = jnp.where(
                                    belongs,
                                    gathered,
                                    compact_codes[:, part * LANES : (part + 1) * LANES],
                                )
                                compact_valid[:, part * LANES : (part + 1) * LANES] = compact_valid[
                                    :, part * LANES : (part + 1) * LANES
                                ] | belongs.astype(jnp.int32)

                    @pl.when(exists)
                    def collect():
                        gather_group()

                    boundary = (route_state[1] + 1) * APPEND_TILE

                    @pl.when(end >= boundary)
                    def flush():
                        consume_compact()
                        route_state[1] = route_state[1] + 1
                        compact_codes[...] = jnp.zeros(compact_codes.shape, jnp.uint32)
                        compact_valid[...] = jnp.zeros(compact_valid.shape, jnp.int32)

                        @pl.when((end > boundary) & exists)
                        def carry():
                            gather_group()

                    return end, (buffer + 1) % DMA_DEPTH

                jax.lax.while_loop(
                    lambda state: state[0] < route_state[0],
                    panel_step,
                    (jnp.int32(0), jnp.int32(0)),
                )
                consume_compact()
            else:

                def selected_step(state):
                    step, buffer = state
                    valid = selected_valid[buffer] != 0
                    nonempty = jnp.bool_(False)

                    following = next_block(step)

                    @pl.when(following < steps)
                    def prefetch():
                        # The alternate buffer is free; start it before waiting for this one.
                        fetch(following, 1 - buffer)

                    for part in range(selected_tile // dma_tile):
                        rows = pl.ds(part * dma_tile, dma_tile)
                        active = page_active[buffer, part] != 0
                        nonempty = nonempty | active

                        @pl.when(active)
                        def wait(part=part, rows=rows):
                            semaphore = 2 + 2 * (buffer * (selected_tile // dma_tile) + part)
                            pltpu.make_async_copy(
                                nope_ref.at[buffer, rows],
                                nope_ref.at[buffer, rows],
                                sem.at[semaphore],
                            ).wait()
                            rope_rows = pl.ds(
                                part * dma_tile // rows_per_group, dma_tile // rows_per_group
                            )
                            pltpu.make_async_copy(
                                rope_ref.at[buffer, rope_rows],
                                rope_ref.at[buffer, rope_rows],
                                sem.at[semaphore + 1],
                            ).wait()

                    def decode_and_consume(count):
                        codes = nope_ref[buffer, :count].reshape(count, nope_record_bytes)
                        rope_bytes = rope_ref[buffer, : count // rows_per_group].reshape(
                            count, rope_record_bytes
                        )
                        consume(
                            _decode_cache(
                                codes,
                                rope_bytes,
                                nope_dim=nope_dim,
                                fp8_scale_block=fp8_scale_block,
                                feature_axis=1,
                            ),
                            valid[:, :count],
                        )

                    @pl.when(nonempty)
                    def attend_page():
                        # Omit a causally empty suffix, down to one hardware lane-width tile.
                        def consume_prefix(count):
                            if count == LANES:
                                decode_and_consume(count)
                            else:
                                jax.lax.cond(
                                    visible > step * selected_tile + count // 2,
                                    lambda: decode_and_consume(count),
                                    lambda: consume_prefix(count // 2),
                                )

                        consume_prefix(selected_tile)

                    return following, 1 - buffer

                jax.lax.while_loop(
                    lambda state: state[0] < steps, selected_step, (first_selected, jnp.int32(0))
                )
            value = (
                (acc_ref[...] / _broadcast_minor(l_ref[...], head_dim))
                .reshape(query_tile, heads, head_dim)
                .astype(jnp.bfloat16)
            )
            out_ref[...] = jnp.where(member_bits[:, None, None] != 0, value, out_ref[...])

    jax.lax.fori_loop(0, query_tile, request_group, None)


@functools.partial(
    jax.jit,
    static_argnames=(
        "scale",
        "schedule",
        "window_size",
        "compression_ratio",
        "fp8_scale_block",
        "rows_per_group",
        "interpret",
    ),
)
def csa_joint_attention(
    q,
    new_kv,
    window_cache,
    compressed_nope,
    compressed_rope,
    topk_indices,
    attention_sink,
    metadata: CSAAttentionMetadata,
    *,
    scale: float,
    schedule: CSAAttentionSchedule,
    window_size: int,
    compression_ratio: int,
    fp8_scale_block: int,
    rows_per_group: int,
    interpret: bool = False,
):
    """Return BF16 [T,H,D]; all input buffers, including caches, are read-only.

    Queries are packed by request, followed by optional inactive padding.
    Queries in a block share page reads, with independent causal/Top-K masks.
    new_kv contains this chunk, not yet committed to the historical SWA ring.
    Cache scale blocks and RoPE row grouping must match the producer's format.
    """
    if q.ndim != 3 or q.dtype != jnp.bfloat16:
        raise ValueError("q must be BF16 [T,H,D]")
    tokens, heads, head_dim = q.shape
    word_bytes = jnp.dtype(jnp.uint32).itemsize
    if not isinstance(window_size, int) or window_size <= 0 or window_size % LANES:
        raise ValueError("window_size must be a positive multiple of the lane width")
    if not isinstance(compression_ratio, int) or compression_ratio <= 0:
        raise ValueError("compression_ratio must be a positive integer")
    if not isinstance(fp8_scale_block, int) or fp8_scale_block <= 0:
        raise ValueError("fp8_scale_block must be a positive integer")
    if not isinstance(rows_per_group, int) or rows_per_group <= 0 or LANES % rows_per_group:
        raise ValueError("rows_per_group must be a positive divisor of the lane width")
    if not heads or heads % SUBLANES:
        raise ValueError("heads must be a positive multiple of eight")
    if new_kv.shape != (tokens, head_dim) or new_kv.dtype != jnp.bfloat16:
        raise ValueError("new_kv must be BF16 [T,D]")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != tokens or topk_indices.dtype != jnp.int32:
        raise ValueError("topk_indices must be int32 [T,K]")
    top_k = topk_indices.shape[-1]
    if top_k < APPEND_TILE or top_k & (top_k - 1):
        raise ValueError("Top-K width must be a power of two and at least one append tile")
    if attention_sink.shape != (heads,) or attention_sink.dtype != jnp.float32:
        raise ValueError("attention_sink must be finite FP32 [H]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if schedule.query_tile < 1 or schedule.query_tile > MAX_QUERY_TILE:
        raise ValueError(f"query_tile must be between one and {MAX_QUERY_TILE}")
    if (
        schedule.selected_tile <= 0
        or schedule.selected_tile % LANES
        or top_k % schedule.selected_tile
    ):
        raise ValueError("selected_tile must be lane-aligned and divide the Top-K width")
    if (
        window_cache.ndim != 4
        or window_cache.shape[-2:] != (2, head_dim)
        or window_cache.dtype != jnp.bfloat16
    ):
        raise ValueError("window_cache must be BF16 [pages,page_size/2,2,D]")
    if (
        compressed_nope.ndim != 4
        or compressed_nope.shape[-1] != LANES
        or compressed_nope.dtype != jnp.uint8
    ):
        raise ValueError("compressed_nope must be uint8 [pages,page_size,record_blocks,128]")
    nope_record_bytes = math.prod(compressed_nope.shape[-2:])
    # Compact gather currently requires one uint32 word per lane in each NoPE row.
    if nope_record_bytes != word_bytes * LANES or head_dim != nope_record_bytes:
        raise ValueError("unsupported NoPE record width or head dimension")
    pages, page_size = compressed_nope.shape[:2]
    if not pages or page_size <= 0 or page_size % rows_per_group:
        raise ValueError("compressed page size must be positive and divisible by rows_per_group")
    if (
        compressed_rope.ndim != 4
        or compressed_rope.shape != (pages, page_size // rows_per_group, rows_per_group, LANES)
        or compressed_rope.dtype != jnp.uint8
    ):
        raise ValueError(
            "compressed_rope must be uint8 [pages,page_size/rows_per_group,rows_per_group,128]"
        )
    rope_record_bytes = compressed_rope.shape[-1]
    nope_dim = head_dim - rope_record_bytes // jnp.dtype(jnp.bfloat16).itemsize
    if nope_dim % fp8_scale_block or nope_dim + nope_dim // fp8_scale_block > nope_record_bytes:
        raise ValueError("fp8_scale_block must divide NoPE width and leave room for scales")
    window_page_size = window_cache.shape[1] * 2
    if not window_cache.shape[0] or not window_page_size or window_size % window_page_size:
        raise ValueError("window page size must divide window_size")
    batch = metadata.seq_lens.shape[0]
    expected = (tokens, batch + 1, batch, None, batch + 1, None, batch + 1, batch)
    for array, length in zip(metadata, expected, strict=True):
        if (
            array.ndim != 1
            or array.dtype != jnp.int32
            or (length is not None and array.shape != (length,))
        ):
            raise ValueError("metadata must contain matching one-dimensional int32 arrays")
    if not tokens or not batch:
        return jnp.zeros_like(q)
    if not metadata.window_page_indices.size or not metadata.compressed_page_indices.size:
        raise ValueError("page tables must include at least a dummy page")
    bt = schedule.query_tile
    # A cache fitting one lane-width panel needs no wider aggregation.
    tile = LANES if (pages - 1) * page_size <= LANES else schedule.selected_tile
    dma_tile = math.gcd(page_size, tile)
    buffer_count = DMA_DEPTH if bt == 1 else 2
    buffer_rows = LANES if bt == 1 else tile
    dma_parts = 1 if bt == 1 else tile // dma_tile
    if bt == 1:
        route_input = prepare_routes(
            topk_indices,
            panel_size=math.gcd(page_size, LANES),
            interpret=interpret,
        )
        gather_scratch = (
            pltpu.VMEM(
                ((nope_record_bytes + rope_record_bytes) // word_bytes, APPEND_TILE), jnp.uint32
            ),
            pltpu.VMEM((1, APPEND_TILE), jnp.int32),
            pltpu.VMEM((2, top_k), jnp.int32),
            pltpu.SMEM((2, top_k), jnp.int32),
            pltpu.SMEM((3,), jnp.int32),
            pltpu.SemaphoreType.DMA,
        )
    else:
        route_input = topk_indices.reshape(tokens, 1, top_k)
        gather_scratch = (pltpu.VMEM((2, bt, tile), jnp.int32),)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            query_tile=bt,
            selected_tile=tile,
            dma_tile=dma_tile,
            page_size=page_size,
            window_page_size=window_page_size,
            window_size=window_size,
            compression_ratio=compression_ratio,
            fp8_scale_block=fp8_scale_block,
            rows_per_group=rows_per_group,
            scale=scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(metadata),
            grid=(pl.cdiv(tokens, bt),),
            in_specs=(
                pl.BlockSpec((bt, heads, head_dim), lambda b, *_: (b, 0, 0)),
                *(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(4)),
                pl.BlockSpec((bt, 2 if bt == 1 else 1, top_k), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((heads,), lambda b, *_: (0,)),
            ),
            out_specs=pl.BlockSpec((bt, heads, head_dim), lambda b, *_: (b, 0, 0)),
            scratch_shapes=(
                pltpu.VMEM((window_page_size // 2, 2, head_dim), jnp.bfloat16),
                pltpu.VMEM((window_size, head_dim), jnp.bfloat16),
                pltpu.VMEM(
                    (buffer_count, buffer_rows, *compressed_nope.shape[-2:]),
                    jnp.uint8,
                ),
                pltpu.VMEM(
                    (buffer_count, buffer_rows // rows_per_group, *compressed_rope.shape[-2:]),
                    jnp.uint8,
                ),
                pltpu.SMEM((buffer_count, dma_parts), jnp.int32),
                pltpu.SemaphoreType.DMA((2 + 2 * buffer_count * dma_parts,)),
                pltpu.VMEM((bt * heads, LANES), jnp.float32),
                pltpu.VMEM((bt * heads, LANES), jnp.float32),
                pltpu.VMEM((bt * heads, head_dim), jnp.float32),
                pltpu.VMEM((max(window_size, tile, APPEND_TILE), head_dim), jnp.float32),
                pltpu.VMEM((max(window_size, tile, APPEND_TILE), LANES), jnp.int32),
                *gather_scratch,
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="csa-joint-attention",
    )(
        *metadata,
        q,
        jnp.pad(new_kv, ((0, -tokens % window_size), (0, 0))),
        window_cache,
        compressed_nope,
        compressed_rope,
        route_input,
        attention_sink,
    )
