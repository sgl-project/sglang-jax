"""Fused ratio-4 KV/index compression using the V4 ring-state and BF16 cache ABI."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.dsv4.paged_row_write import paged_row_write

from .tune import CompressorSchedule


class CompressorBlock(NamedTuple):
    """Backend work: contiguous token ids [R,Q], request ids [R]; -1 pads a row's tail.

    Cover real tokens once, in request order, with distinct live requests/state slots per block.
    Valid tokens form a prefix with consecutive, in-range positions; the backend guarantees this.
    """

    token_indices: jax.Array
    request_indices: jax.Array


class CompressorMetadata(NamedTuple):
    """Device metadata; cache locations are physical compressed-record offsets.

    positions/cache_locations [T], cu_q_lens [B+1], state_indices [B].
    Slot zero is valid; the last state slot and cache page zero are padding.
    Negative locations suppress cache writes, not continuation-state updates.
    """

    positions: jax.Array
    cu_q_lens: jax.Array
    state_indices: jax.Array
    cache_locations: jax.Array
    blocks: tuple[CompressorBlock, ...]


def _take_rows(value, indices):
    indices = jnp.broadcast_to(indices, (indices.size, value.shape[1]))
    return jnp.take_along_axis(value, indices, axis=0, mode="promise_in_bounds")


def _normalize_record(pooled, norm, cos, sin, norm_eps):
    pooled *= jax.lax.rsqrt(jnp.mean(pooled * pooled, axis=1, keepdims=True) + norm_eps)
    pooled *= norm[None, :]
    rope_dim = 2 * cos.shape[-1]
    pairs = pooled[:, -rope_dim:].reshape(pooled.shape[0], rope_dim // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rope = jnp.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1)
    return jnp.concatenate(
        (pooled[:, :-rope_dim], rope.reshape(pooled.shape[0], rope_dim)), axis=1
    ).astype(jnp.bfloat16)


def _pool_decode_group(projected, state, ape, norm, cos, sin, starts, lengths, norm_eps):
    batch, width = projected.shape
    dim = width // 4
    projected += jnp.concatenate(
        (jnp.zeros((batch, 2 * dim), jnp.float32), _take_rows(ape, (starts % 4)[:, None])), axis=1
    )
    row = jnp.arange(8)[None, :, None]
    state = jnp.where(
        (row == starts[:, None, None] % 8) & (lengths[:, None, None] > 0),
        projected[:, None, :],
        state,
    )
    # Reduce in physical ring order; avoid per-request dynamic window rotations.
    age = (row - (starts[:, None, None] + 1)) % 8
    values = jnp.where(age >= 4, state[..., dim : 2 * dim], state[..., :dim])
    scores = jnp.where(age >= 4, state[..., 3 * dim :], state[..., 2 * dim : 3 * dim])
    live = (starts[:, None, None] - 7 + age >= 0) & (lengths[:, None, None] > 0)
    scores = jnp.where(live, scores, -jnp.inf)
    maximum = jnp.max(scores, axis=1, keepdims=True)
    weights = jnp.exp(scores - jnp.where(jnp.isfinite(maximum), maximum, 0))
    weights /= jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), jnp.finfo(jnp.float32).tiny)
    pooled = jnp.sum(values * weights, axis=1)
    return _normalize_record(pooled, norm, cos, sin, norm_eps), state


def _pool_and_save(projected, state, ape, norm, cos, sin, start, length, norm_eps):
    rows, width = projected.shape
    dim = width // 4
    pos = start + jnp.arange(rows)[:, None]
    projected += jnp.concatenate(
        (jnp.zeros((rows, 2 * dim), jnp.float32), _take_rows(ape, pos % 4)), axis=1
    )

    # Pool before overwriting ring rows needed by this tile.
    row = jnp.arange(2 * rows)[:, None]
    offsets = (-start - 1) % 4 + 4 * (row // 8) - 7 + row % 8
    previous = pltpu.roll(state, (8 - start % 8) % 8, axis=0)
    source = jnp.concatenate((previous, projected), axis=0)
    shift = (-start - 1) % 4 + 1
    # Align eight-token chunks locally, then select across neighboring chunks.
    chunks = pltpu.roll(source.reshape(rows // 8 + 1, 8, width), 8 - shift, axis=1)
    aligned = jnp.concatenate(
        (
            jnp.where(
                jnp.arange(rows)[:, None] % 8 < 8 - shift,
                chunks[:-1].reshape(rows, width),
                chunks[1:].reshape(rows, width),
            ),
            chunks[-1, :4],
        ),
        axis=0,
    )
    live = (start + offsets >= 0) & ((-start - 1) % 4 + 4 * (row // 8) < length)
    groups = rows // 4
    # Select each window half before expanding overlap; do not duplicate all four fields.
    values = jnp.concatenate(
        (
            aligned[:rows, :dim].reshape(groups, 4, dim),
            aligned[4 : rows + 4, dim : 2 * dim].reshape(groups, 4, dim),
        ),
        axis=1,
    )
    scores = jnp.concatenate(
        (
            aligned[:rows, 2 * dim : 3 * dim].reshape(groups, 4, dim),
            aligned[4 : rows + 4, 3 * dim :].reshape(groups, 4, dim),
        ),
        axis=1,
    )
    scores = jnp.where(live, scores.reshape(2 * rows, dim), -jnp.inf).reshape(groups, 8, dim)
    maximum = jnp.max(scores, axis=1, keepdims=True)
    weights = jnp.exp(scores - jnp.where(jnp.isfinite(maximum), maximum, 0))
    weights /= jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), jnp.finfo(jnp.float32).tiny)
    pooled = jnp.sum(values * weights, axis=1)
    record = _normalize_record(pooled, norm, cos[:groups], sin[:groups], norm_eps)

    tail = pltpu.roll(source, (rows + 8 - length) % (rows + 8), axis=0)[:8]
    state = pltpu.roll(tail, (start + length) % 8, axis=0)
    return record, state


def _projection_kernel(x, weight, output, acc, *, k_steps):
    k = pl.program_id(1)
    product = jax.lax.dot_general(
        x[...],
        weight[...],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc[...] = jnp.where(k == 0, product, acc[...] + product)

    @pl.when(k == k_steps - 1)
    def store():
        output[...] = acc[...]


def _project_decode(x, weight, schedule, interpret):
    # One MXU batch across requests; do not stream the entire weight once per token.
    tokens, hidden = x.shape
    rows = schedule.token_tile(tokens)
    padded = pl.cdiv(tokens, rows) * rows
    tile_k = schedule.projection_k_tile
    return pl.pallas_call(
        functools.partial(_projection_kernel, k_steps=hidden // tile_k),
        out_shape=jax.ShapeDtypeStruct((padded, weight.shape[1]), jnp.float32),
        grid=(pl.cdiv(tokens, rows), hidden // tile_k),
        in_specs=(
            pl.BlockSpec((rows, tile_k), lambda b, k: (b, k)),
            pl.BlockSpec((tile_k, weight.shape[1]), lambda b, k: (k, 0)),
        ),
        out_specs=pl.BlockSpec((rows, weight.shape[1]), lambda b, k: (b, 0)),
        scratch_shapes=(pltpu.VMEM((rows, weight.shape[1]), jnp.float32),),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
        name="csa_decode_projection",
    )(jnp.pad(x, ((0, padded - tokens), (0, 0))), weight)


def _decode_pool_kernel(
    metadata,
    projection,
    main_state,
    index_state,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    main_records,
    index_records,
    main_out,
    index_out,
    main_buffer,
    index_buffer,
    sems,
    *,
    requests,
    norm_eps,
):
    group = pl.program_id(0)
    banks = main_buffer.shape[0]
    bank = group % banks
    families = (
        (main_state, main_ape, main_norm, main_records, main_out, main_buffer),
        (index_state, index_ape, index_norm, index_records, index_out, index_buffer),
    )
    active = []
    slots = []
    for r in range(requests):
        slot = metadata[group * requests + r, 2]
        slots.append(slot)
        active.append(
            (slot >= 0) & (slot < main_state.shape[0] - 1) & (metadata[group * requests + r, 1] > 0)
        )

    def prefetch(g, b):
        for f, (source, _, _, _, _, buffers) in enumerate(families):
            buffer = buffers.at[b]
            for r in range(requests):
                slot = metadata[g * requests + r, 2]
                valid = (
                    (slot >= 0)
                    & (slot < main_state.shape[0] - 1)
                    & (metadata[g * requests + r, 1] > 0)
                )
                buffer[r] = jnp.zeros(buffer.shape[1:], jnp.float32)
                copy = pltpu.make_async_copy(source.at[slot], buffer.at[r], sems.at[b, f, r])
                pl.when(valid)(copy.start)

    pl.when(group == 0)(lambda: prefetch(group, bank))
    # Distinct request slots allow next-group reads to overlap current-group writes.
    pl.when(group + 1 < pl.num_programs(0))(lambda: prefetch(group + 1, (bank + 1) % banks))
    starts = jnp.stack([metadata[group * requests + r, 0] for r in range(requests)])
    lengths = jnp.stack([metadata[group * requests + r, 1] for r in range(requests)])
    offset = 0
    for f, (source, ape, norm, records, target, buffers) in enumerate(families):
        buffer = buffers.at[bank]
        for r in range(requests):
            load = pltpu.make_async_copy(source.at[slots[r]], buffer.at[r], sems.at[bank, f, r])
            pl.when(active[r])(load.wait)
        width = buffer.shape[-1]
        record, updated = _pool_decode_group(
            projection[:, offset : offset + width],
            buffer[...],
            ape[...],
            norm[...],
            cos[...],
            sin[...],
            starts,
            lengths,
            norm_eps,
        )
        records[...] = record
        buffer[...] = updated
        for r in range(requests):
            store = pltpu.make_async_copy(buffer.at[r], target.at[slots[r]], sems.at[bank, f, r])
            pl.when(active[r])(store.start)
        offset += width
    for f, (_, _, _, _, target, buffers) in enumerate(families):
        buffer = buffers.at[bank]
        for r in range(requests):
            store = pltpu.make_async_copy(buffer.at[r], target.at[slots[r]], sems.at[bank, f, r])
            pl.when(active[r])(store.wait)


def _decode_pool(
    x,
    main_state,
    index_state,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    starts,
    lengths,
    slots,
    token_starts,
    *,
    schedule,
    norm_eps,
    interpret,
):
    batch = starts.size
    requests = schedule.decode_request_tile(batch)
    padded = pl.cdiv(batch, requests) * requests
    banks = min(2, padded // requests)
    dims = (main_norm.size, index_norm.size)
    ends = ((-starts - 1) % 4)[:, None]
    positions = jnp.clip(starts + ends[:, 0] - 3, 0, cos.shape[0] - 1)
    metadata = jnp.stack((starts, lengths, slots), axis=1)
    metadata = jnp.pad(metadata, ((0, padded - batch), (0, 0)), constant_values=-1)

    def pad(value):
        return jnp.pad(value, ((0, padded - batch), (0, 0)))

    states = (main_state, index_state)
    outputs = pl.pallas_call(
        functools.partial(_decode_pool_kernel, requests=requests, norm_eps=norm_eps),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(padded // requests,),
            in_specs=(
                pl.BlockSpec((requests, x.shape[1]), lambda g, m: (g, 0)),
                *[pl.BlockSpec(memory_space=pltpu.HBM) for _ in dims],
                *[pl.BlockSpec(a.shape, lambda g, m: (0, 0)) for a in (main_ape, index_ape)],
                *[pl.BlockSpec(n.shape, lambda g, m: (0,)) for n in (main_norm, index_norm)],
                *[pl.BlockSpec((requests, cos.shape[-1]), lambda g, m: (g, 0)) for _ in dims],
            ),
            out_specs=(
                *[pl.BlockSpec((requests, d), lambda g, m: (g, 0)) for d in dims],
                *[pl.BlockSpec(memory_space=pltpu.HBM) for _ in dims],
            ),
            scratch_shapes=(
                *[pltpu.VMEM((banks, requests, 8, 4 * d), jnp.float32) for d in dims],
                pltpu.SemaphoreType.DMA((banks, 2, requests)),
            ),
        ),
        out_shape=(
            *[jax.ShapeDtypeStruct((padded, d), jnp.bfloat16) for d in dims],
            *[jax.ShapeDtypeStruct(s.shape, s.dtype) for s in states],
        ),
        input_output_aliases={2: 2, 3: 3},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="csa_decode_pool_grouped",
    )(
        metadata,
        pad(x[token_starts]),
        *[s if interpret else pltpu.with_memory_space_constraint(s, pltpu.HBM) for s in states],
        main_ape,
        index_ape,
        main_norm,
        index_norm,
        pad(cos[positions]),
        pad(sin[positions]),
    )
    return outputs[0][:batch, None], outputs[1][:batch, None], outputs[2], outputs[3], ends


def _compress_kernel(
    metadata_ref,
    x_ref,
    weight_ref,
    main_state_ref,
    index_state_ref,
    main_ape_ref,
    index_ape_ref,
    main_norm_ref,
    index_norm_ref,
    cos_ref,
    sin_ref,
    main_ref,
    index_ref,
    main_out_ref,
    index_out_ref,
    projection_ref,
    main_buffer,
    index_buffer,
    sems,
    *,
    k_steps,
    rows,
    input_rows,
    dma_rows,
    state_slots,
    norm_eps,
):
    request, tile, k = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    slot = metadata_ref[request, 2]
    active = (slot >= 0) & (slot < state_slots - 1) & (metadata_ref[request, 1] > 0)

    @pl.when((tile == 0) & (k == 0))
    def initialize():
        for source, buffer in (
            (main_state_ref, main_buffer),
            (index_state_ref, index_buffer),
        ):
            buffer[...] = jnp.where(active, source[0], 0)

    live_tile = active & (tile * rows < metadata_ref[request, 1])

    @pl.when(live_tile)
    def compute():
        token = metadata_ref[request, 3] + tile * rows
        first = jnp.minimum(token // dma_rows * dma_rows, input_rows - x_ref.shape[0])
        shift = (x_ref.shape[0] - (token - first)) % x_ref.shape[0]
        values = jax.lax.cond(
            token == first,
            lambda: x_ref[:rows],
            lambda: pltpu.roll(x_ref[...].astype(jnp.float32), shift, axis=0)[:rows].astype(
                x_ref.dtype
            ),
        )
        remaining = metadata_ref[request, 1] - tile * rows
        values = jax.lax.cond(
            remaining >= rows,
            lambda: values,
            lambda: jnp.where(jnp.arange(rows)[:, None] < remaining, values, 0),
        )
        product = jax.lax.dot_general(
            values.astype(jnp.bfloat16),
            weight_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        projection_ref[...] = jnp.where(k == 0, product, projection_ref[...] + product)

        @pl.when(k == k_steps - 1)
        def finish():
            start = metadata_ref[request, 0] + tile * rows
            length = jnp.clip(metadata_ref[request, 1] - tile * rows, 0, rows)
            split = main_state_ref.shape[-1]
            for projection, state, ape, norm, out, saved in (
                (
                    projection_ref[:, :split],
                    main_buffer[...],
                    main_ape_ref[...],
                    main_norm_ref[...],
                    main_ref,
                    main_buffer,
                ),
                (
                    projection_ref[:, split:],
                    index_buffer[...],
                    index_ape_ref[...],
                    index_norm_ref[...],
                    index_ref,
                    index_buffer,
                ),
            ):
                record, updated = _pool_and_save(
                    projection,
                    state,
                    ape,
                    norm,
                    cos_ref[0],
                    sin_ref[0],
                    start,
                    length,
                    norm_eps,
                )
                out[0] = record
                saved[...] = updated

            @pl.when(tile == (metadata_ref[request, 1] - 1) // rows)
            def save():
                copies = []
                for buffer, target, sem in (
                    (main_buffer, main_out_ref, sems.at[0]),
                    (index_buffer, index_out_ref, sems.at[1]),
                ):
                    copy = pltpu.make_async_copy(buffer, target.at[slot], sem)
                    copy.start()
                    copies.append(copy)
                for copy in copies:
                    copy.wait()

    @pl.when((~live_tile) & (k == k_steps - 1))
    def clear_padding():
        main_ref[...] = jnp.zeros(main_ref.shape, main_ref.dtype)
        index_ref[...] = jnp.zeros(index_ref.shape, index_ref.dtype)


def _compress_block(
    x,
    weight,
    main_state,
    index_state,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    starts,
    lengths,
    slots,
    token_starts,
    *,
    sequence,
    projected,
    schedule,
    norm_eps,
    interpret,
):
    if projected:
        return _decode_pool(
            x,
            main_state,
            index_state,
            main_ape,
            index_ape,
            main_norm,
            index_norm,
            cos,
            sin,
            starts,
            lengths,
            slots,
            token_starts,
            schedule=schedule,
            norm_eps=norm_eps,
            interpret=interpret,
        )
    batch, hidden = starts.size, x.shape[1]
    rows, dma_rows, load_rows, input_rows = schedule.input_geometry(
        x.shape[0], sequence, x.dtype.itemsize
    )
    tile_k = schedule.projection_k_tile
    tiles, k_steps = pl.cdiv(sequence, rows), hidden // tile_k
    groups_per_tile = rows // 4
    groups = tiles * groups_per_tile
    x = jnp.pad(x, ((0, input_rows - x.shape[0]), (0, 0)))

    def input_map(r, q, k, m):
        first = (m[r, 3] + q * rows) // dma_rows * dma_rows
        return pl.multiple_of(jnp.minimum(first, input_rows - load_rows), dma_rows), k * tile_k

    if not interpret:
        weight = pltpu.with_memory_space_constraint(weight, pltpu.VMEM)
    ends = ((-starts - 1) % 4)[:, None] + 4 * jnp.arange(groups)[None, :]
    rope_positions = jnp.clip(starts[:, None] + ends - 3, 0, cos.shape[0] - 1)
    dims = (main_norm.size, index_norm.size)
    outputs = pl.pallas_call(
        functools.partial(
            _compress_kernel,
            k_steps=k_steps,
            rows=rows,
            input_rows=input_rows,
            dma_rows=dma_rows,
            state_slots=main_state.shape[0],
            norm_eps=norm_eps,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(batch, tiles, k_steps),
            in_specs=(
                pl.BlockSpec((pl.Element(load_rows), pl.Element(tile_k)), input_map),
                pl.BlockSpec((tile_k, weight.shape[1]), lambda r, q, k, m: (k, 0)),
                *[
                    pl.BlockSpec(
                        (1, 8, 4 * d),
                        lambda r, q, k, m: (jnp.clip(m[r, 2], 0, main_state.shape[0] - 1), 0, 0),
                    )
                    for d in dims
                ],
                *[pl.BlockSpec(a.shape, lambda r, q, k, m: (0, 0)) for a in (main_ape, index_ape)],
                *[pl.BlockSpec(n.shape, lambda r, q, k, m: (0,)) for n in (main_norm, index_norm)],
                *[
                    pl.BlockSpec((1, groups_per_tile, cos.shape[-1]), lambda r, q, k, m: (r, q, 0))
                    for _ in range(2)
                ],
            ),
            out_specs=(
                *[
                    pl.BlockSpec((1, groups_per_tile, d), lambda r, q, k, m: (r, q, 0))
                    for d in dims
                ],
                *[pl.BlockSpec(memory_space=pltpu.HBM) for _ in dims],
            ),
            scratch_shapes=(
                pltpu.VMEM((rows, main_state.shape[-1] + index_state.shape[-1]), jnp.float32),
                *[pltpu.VMEM((8, 4 * d), jnp.float32) for d in dims],
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(
            *[jax.ShapeDtypeStruct((batch, groups, d), jnp.bfloat16) for d in dims],
            *[jax.ShapeDtypeStruct(s.shape, s.dtype) for s in (main_state, index_state)],
        ),
        input_output_aliases={3: 2, 4: 3},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
        interpret=interpret,
        name="csa_dual_compressor_bf16",
    )(
        jnp.stack((starts, lengths, slots, token_starts), axis=1),
        x,
        weight,
        main_state if interpret else pltpu.with_memory_space_constraint(main_state, pltpu.HBM),
        index_state if interpret else pltpu.with_memory_space_constraint(index_state, pltpu.HBM),
        main_ape,
        index_ape,
        main_norm,
        index_norm,
        cos[rope_positions],
        sin[rope_positions],
    )
    return (*outputs, ends)


@functools.partial(jax.jit, static_argnames=("schedule", "norm_eps", "interpret"))
def csa_compressor(
    x,
    fused_weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    main_state,
    index_state,
    main_cache,
    index_cache,
    metadata: CompressorMetadata,
    *,
    schedule: CompressorSchedule,
    norm_eps: float = 1e-6,
    interpret: bool = False,
):
    """Return complete (main_state, index_state, c4_cache, indexer_cache) replacements.

    Buffers are local to one data shard: FP32 state [slots,8,4D], BF16 cache [pages,P/4,D].
    D=512/128; owns state[c4/indexer] and KV[c4/indexer], leaving SWA/c128 untouched.
    Caller resets fresh/recycled state to contents=0, scores=-inf; chunks reuse state.
    Caller owns allocation/metadata/commit; this operator owns all four returned families.
    """
    if x.ndim != 2 or x.dtype != jnp.bfloat16 or not x.shape[1]:
        raise ValueError("x must be BF16 [T,hidden]")
    tokens, hidden = x.shape
    if fused_weight.shape != (hidden, 2560) or fused_weight.dtype != jnp.bfloat16:
        raise ValueError(
            "fused_weight must be BF16 [hidden,2560]: main KV/score then index KV/score"
        )
    if (
        schedule.projection_k_tile <= 0
        or schedule.projection_k_tile % 128
        or hidden % schedule.projection_k_tile
    ):
        raise ValueError("projection_k_tile must be a lane-aligned divisor of hidden")
    if schedule.query_tile <= 0 or schedule.query_tile % 8:
        raise ValueError("query_tile must be a positive multiple of eight")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    for state, cache, ape, norm, dim in (
        (main_state, main_cache, main_ape, main_norm, 512),
        (index_state, index_cache, index_ape, index_norm, 128),
    ):
        if state.ndim != 3 or state.shape[1:] != (8, 4 * dim) or state.dtype != jnp.float32:
            raise ValueError("state must be FP32 [slots,8,4D]")
        if cache.ndim != 3 or cache.shape[-1] != dim or cache.dtype != jnp.bfloat16:
            raise ValueError("cache must be BF16 [pages,P/4,D]")
        if (
            ape.shape != (4, 2 * dim)
            or norm.shape != (dim,)
            or ape.dtype != jnp.float32
            or norm.dtype != jnp.float32
        ):
            raise ValueError("APE/norm must be FP32 [4,2D]/[D]")
    if main_state.shape[0] < 2 or main_state.shape[0] != index_state.shape[0]:
        raise ValueError("state pools need matching capacity including a last padding slot")
    if main_cache.shape[:2] != index_cache.shape[:2] or min(main_cache.shape[:2]) <= 0:
        raise ValueError("cache pools need matching nonempty page geometry")
    if (
        cos.ndim != 2
        or cos.shape[0] == 0
        or cos.shape[1] != 32
        or sin.shape != cos.shape
        or cos.dtype != jnp.float32
        or sin.dtype != jnp.float32
    ):
        raise ValueError("cos/sin must be FP32 [max_position,32]")
    requests = metadata.state_indices.size
    for value, shape in (
        (metadata.positions, (tokens,)),
        (metadata.cache_locations, (tokens,)),
        (metadata.cu_q_lens, (requests + 1,)),
        (metadata.state_indices, (requests,)),
    ):
        if value.shape != shape or value.dtype != jnp.int32:
            raise ValueError(f"metadata must be int32 {shape}")
    if not tokens or not requests:
        return main_state, index_state, main_cache, index_cache

    for block in metadata.blocks:
        ids = block.token_indices
        if ids.ndim != 2 or min(ids.shape) <= 0 or ids.dtype != jnp.int32:
            raise ValueError("block token_indices must be nonempty int32 [R,Q]")
        if (
            block.request_indices.shape != (ids.shape[0],)
            or block.request_indices.dtype != jnp.int32
        ):
            raise ValueError("block request_indices must be int32 [R]")
    projected = bool(metadata.blocks) and all(
        b.token_indices.shape[1] == 1 for b in metadata.blocks
    )
    inputs = _project_decode(x, fused_weight, schedule, interpret) if projected else x
    for block in metadata.blocks:
        ids = block.token_indices
        req = jnp.clip(block.request_indices, 0, requests - 1)
        safe = jnp.clip(ids, 0, tokens - 1)
        slots = metadata.state_indices[req]
        active = (
            (block.request_indices >= 0)
            & (block.request_indices < requests)
            & (slots >= 0)
            & (slots < main_state.shape[0] - 1)
        )
        valid = (
            active[:, None]
            & (ids >= metadata.cu_q_lens[req, None])
            & (ids < metadata.cu_q_lens[req + 1, None])
            & (ids >= 0)
            & (ids < tokens)
        )
        # Contiguous token/position prefixes need only one position lookup per row.
        first_position = metadata.positions[safe[:, 0]]
        positions = first_position[:, None] + safe - safe[:, :1]
        valid &= (positions >= 0) & (positions < cos.shape[0])
        length = jnp.sum(valid, axis=1, dtype=jnp.int32)
        start = jnp.where(length > 0, first_position, 0)
        main, index, ms, ins, ends = _compress_block(
            inputs,
            fused_weight,
            main_state,
            index_state,
            main_ape,
            index_ape,
            main_norm,
            index_norm,
            cos,
            sin,
            start,
            length,
            jnp.where(active, slots, -1),
            safe[:, 0],
            sequence=ids.shape[1],
            projected=projected,
            schedule=schedule,
            norm_eps=norm_eps,
            interpret=interpret,
        )
        main_state, index_state = ms, ins
        end_ids = jnp.clip(safe[:, :1] + jnp.clip(ends, 0, ids.shape[1] - 1), 0, tokens - 1)
        locations = metadata.cache_locations[end_ids]
        pages, page_rows = main_cache.shape[:2]
        write = (
            (ends < length[:, None]) & (locations >= page_rows) & (locations < pages * page_rows)
        )
        updated = []
        for cache, records in ((main_cache, main), (index_cache, index)):
            flat = cache.reshape(-1, cache.shape[-1])
            row_alignment = pltpu.Tiling.COMPACT.shape[0] * (4 // cache.dtype.itemsize)
            # Constrain the Pallas path, not the shared writer's unaligned XLA fallback.
            if not interpret and flat.shape[0] % row_alignment == 0:
                flat = pltpu.with_memory_space_constraint(flat, pltpu.HBM)
            flat = paged_row_write(
                flat,
                records.reshape(-1, records.shape[-1]),
                locations.reshape(-1),
                write.reshape(-1),
                run=schedule.cache_write_run(records.shape[1], cache.dtype.itemsize),
                interpret=interpret,
            )
            updated.append(flat.reshape(cache.shape))
        main_cache, index_cache = updated
    return main_state, index_state, main_cache, index_cache
