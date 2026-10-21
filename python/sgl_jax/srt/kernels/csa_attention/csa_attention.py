"""BF16 paged CSA joint attention with complete SWA-family replacement."""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.dsv4.paged_row_write import paged_row_write
from sgl_jax.srt.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2

from .tune import LANES, CSAAttentionSchedule


class CSAAttentionMetadata(NamedTuple):
    """Caller-owned int32 arrays; token/request padding uses -1, page zero is dummy.

    query_seq_ids:[T], cu_q_lens:[B+1], seq_lens:[B]. Flattened page tables
    use page-aligned cu_*_kv_lens:[B+1] in tokens/entries, as in HCA.
    Live queries are request-contiguous in cu_q_lens order.
    Window tables start at the page containing max(0, prefix-window_size+1).
    Compressed lengths:[B] count completed entries. Top-K entries must be unique.
    window_write_locations:[T] gives raw rank-local row addresses; -1 suppresses a write.
    Valid write locations are unique physical rows in caller-owned SWA pages.
    """

    query_seq_ids: jax.Array
    cu_q_lens: jax.Array
    seq_lens: jax.Array
    window_page_indices: jax.Array
    window_cu_kv_lens: jax.Array
    compressed_page_indices: jax.Array
    compressed_cu_kv_lens: jax.Array
    compressed_kv_lens: jax.Array
    window_write_locations: jax.Array


def _broadcast_minor(value, width):
    return jnp.concatenate((value,) * pl.cdiv(width, value.shape[1]), axis=1)[:, :width]


def _attention_update(q, kv, valid, m_ref, l_ref, acc_ref, *, scale):
    scores = jax.lax.dot_general(
        q,
        kv,
        (((1,), (1,)), ((), ())),
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
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = _broadcast_minor(correction, q.shape[1]) * acc_ref[...] + product
    m_ref[...] = jnp.broadcast_to(maximum, m_ref.shape)
    l_ref[...] = jnp.broadcast_to(denominator, l_ref.shape)


def _decode_pages(table, offsets):
    sublanes = pltpu.Tiling.COMPACT.shape[0]
    table = jnp.pad(table, (0, -table.size % LANES))

    def kernel(pages, indices, output):
        index = indices[...]
        result = jnp.zeros_like(index)
        for start in range(0, table.size, LANES):
            local = index - start
            panel = jnp.broadcast_to(pages[start : start + LANES], (sublanes, LANES))
            values = jnp.take_along_axis(panel, jnp.clip(local, 0, LANES - 1), axis=1)
            result = jnp.where((local >= 0) & (local < LANES), values, result)
        output[...] = result

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(offsets.shape, jnp.int32),
        grid=(pl.cdiv(offsets.shape[0], sublanes),),
        in_specs=(
            pl.BlockSpec(table.shape, lambda b: (0,)),
            pl.BlockSpec((sublanes, offsets.shape[1]), lambda b: (b, 0)),
        ),
        out_specs=pl.BlockSpec((sublanes, offsets.shape[1]), lambda b: (b, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="csa_decode_pages",
    )(table, offsets)


def _decode_kernel(
    position,
    active,
    q,
    new,
    compressed,
    valid,
    sink,
    window,
    window_valid,
    output,
    maximum,
    denominator,
    accumulator,
    *,
    scale,
    window_size,
):
    token = pl.program_id(0)
    row = jax.lax.broadcasted_iota(jnp.int32, (window_size, window.shape[-1]), 0)
    history = jnp.where(row == window_size - 1, new[0], window[0])
    keep = (jnp.concatenate((window_valid[0, 0], valid[0, 0])) != 0) & (active[token] != 0)
    kv = jnp.where(
        keep.astype(jnp.int32)[:, None] != 0, jnp.concatenate((history, compressed[0]), axis=0), 0
    )
    maximum[...] = jnp.broadcast_to(sink[...][:, None], maximum.shape)
    denominator[...] = jnp.ones(denominator.shape, jnp.float32)
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
    _attention_update(
        jnp.where(active[token] != 0, q[0], 0),
        kv,
        keep[None],
        maximum,
        denominator,
        accumulator,
        scale=scale,
    )
    output[0] = jnp.where(active[token] != 0, accumulator[...] / denominator[...][:, :1], 0).astype(
        jnp.bfloat16
    )


def _decode_attention(
    q,
    new,
    window,
    cache,
    indices,
    sink,
    md,
    *,
    scale,
    window_size,
    window_page_size,
    compression_ratio,
):
    tokens, heads, dim = q.shape
    page_size, selected = cache.shape[1], indices.shape[1]
    request = md.query_seq_ids
    safe = jnp.clip(request, 0, md.seq_lens.size - 1)
    active = (request >= 0) & (request < md.seq_lens.size)
    active &= (jnp.arange(tokens) >= md.cu_q_lens[safe]) & (
        jnp.arange(tokens) < md.cu_q_lens[safe + 1]
    )
    position = md.seq_lens[safe] - 1
    start, end = (
        md.compressed_cu_kv_lens[safe] // page_size,
        md.compressed_cu_kv_lens[safe + 1] // page_size,
    )
    entry = start[:, None] + indices // page_size
    visible = jnp.minimum(md.compressed_kv_lens[safe], (position + 1) // compression_ratio)
    valid = active[:, None] & (indices >= 0) & (indices < visible[:, None])
    valid &= (entry >= start[:, None]) & (entry < end[:, None])
    valid &= (entry >= 0) & (entry < md.compressed_page_indices.size)
    pages = _decode_pages(md.compressed_page_indices, jnp.where(valid, entry, -1))
    valid &= (pages > 0) & (pages < cache.shape[0])
    padding = jnp.arange(tokens * selected).reshape(tokens, selected) % page_size
    locations = jnp.where(valid, pages * page_size + indices % page_size, padding).reshape(-1)
    # Contiguous page/row axes form a bitcast view, not a persistent-pool copy.
    gathered = ragged_gather_v2(
        cache.reshape(-1, dim), locations, jnp.int32(0), jnp.int32(locations.size)
    )
    absolute = position[:, None] - window_size + 1 + jnp.arange(window_size)[None]
    first_page = jnp.maximum(0, position - window_size + 1) // window_page_size
    wp_start = md.window_cu_kv_lens[safe] // window_page_size
    wp_end = md.window_cu_kv_lens[safe + 1] // window_page_size
    wp_entry = wp_start[:, None] + absolute // window_page_size - first_page[:, None]
    historical = active[:, None] & (absolute >= 0) & (absolute < position[:, None])
    historical &= (wp_entry >= wp_start[:, None]) & (wp_entry < wp_end[:, None])
    historical &= (wp_entry >= 0) & (wp_entry < md.window_page_indices.size)
    wp = _decode_pages(md.window_page_indices, jnp.where(historical, wp_entry, -1))
    historical &= (wp > 0) & (wp < window.shape[0] // window_page_size)
    window_locations = jnp.where(
        historical, wp * window_page_size + absolute % window_page_size, 0
    ).reshape(-1)
    history = ragged_gather_v2(
        window, window_locations, jnp.int32(0), jnp.int32(window_locations.size)
    )
    window_valid = historical | (active[:, None] & (absolute == position[:, None]))
    return pl.pallas_call(
        functools.partial(_decode_kernel, scale=scale, window_size=window_size),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(tokens,),
            in_specs=(
                pl.BlockSpec((1, heads, dim), lambda t, *_: (t, 0, 0)),
                pl.BlockSpec((1, 1, dim), lambda t, *_: (t, 0, 0)),
                pl.BlockSpec((1, selected, dim), lambda t, *_: (t, 0, 0)),
                pl.BlockSpec((1, 1, selected), lambda t, *_: (t, 0, 0)),
                pl.BlockSpec((heads,), lambda t, *_: (0,)),
                pl.BlockSpec((1, window_size, dim), lambda t, *_: (t, 0, 0)),
                pl.BlockSpec((1, 1, window_size), lambda t, *_: (t, 0, 0)),
            ),
            out_specs=pl.BlockSpec((1, heads, dim), lambda t, *_: (t, 0, 0)),
            scratch_shapes=(
                pltpu.VMEM((heads, LANES), jnp.float32),
                pltpu.VMEM((heads, LANES), jnp.float32),
                pltpu.VMEM((heads, dim), jnp.float32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="csa_bf16_decode_attention",
    )(
        position,
        active.astype(jnp.int32),
        q,
        new[:, None],
        gathered.reshape(tokens, selected, dim),
        valid[:, None].astype(jnp.int32),
        sink,
        history.reshape(tokens, window_size, dim),
        window_valid[:, None].astype(jnp.int32),
    )


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
    compressed_cache,
    topk,
    sink_ref,
    out_ref,
    window_ref,
    fresh_ref,
    compressed_ref,
    page_active,
    sem,
    m_ref,
    l_ref,
    acc_ref,
    kv_tile,
    valid_tile,
    selected_valid,
    query_tile,
    selected_tile,
    dma_tile,
    page_size,
    window_page_size,
    window_size,
    compression_ratio,
    scale,
):
    block = pl.program_id(0)
    heads, head_dim = q_ref.shape[1:]
    top_k = topk.shape[-1]
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
                _attention_update(
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
                        compressed_cache.shape[0],
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
                        semaphore = 2 + buffer * (selected_tile // dma_tile) + part
                        pltpu.make_async_copy(
                            compressed_cache.at[page, pl.ds(offset, dma_tile)],
                            compressed_ref.at[buffer, rows],
                            sem.at[semaphore],
                        ).start()

            first_selected = next_block(-1)

            @pl.when(first_selected < steps)
            def start_compressed():
                fetch(first_selected, 0)

            # Backend tables begin at the first retained historical page.
            @pl.when((prefix > 0) & (token - cu_q[safe_request] < window_size - 1))
            def history():
                first_position = jnp.maximum(0, prefix - window_size + 1)
                first_page = first_position // window_page_size * window_page_size

                def page_step(part, _):
                    page, _, exists = page_location(
                        window_pages,
                        window_cu,
                        safe_request,
                        part * window_page_size,
                        window_page_size,
                        window_cache.shape[0] // window_page_size,
                    )

                    @pl.when(exists)
                    def attend_page():
                        copy = pltpu.make_async_copy(
                            window_cache.at[pl.ds(page * window_page_size, window_page_size)],
                            window_ref,
                            sem.at[0],
                        )

                        copy.start()
                        copy.wait()
                        absolute = (
                            first_page + part * window_page_size + jnp.arange(window_page_size)
                        )
                        keep = (
                            (absolute[None, :] >= 0)
                            & (absolute[None, :] < prefix)
                            & (absolute[None, :] > position[:, None] - window_size)
                            & (member_bits[:, None] != 0)
                        )
                        consume(window_ref[...].reshape(window_page_size, head_dim), keep)

                jax.lax.fori_loop(
                    0, pl.cdiv(prefix - first_page, window_page_size), page_step, None
                )

            # Current chunk stays separate until historical reads complete.
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
                        semaphore = 2 + buffer * (selected_tile // dma_tile) + part
                        pltpu.make_async_copy(
                            compressed_ref.at[buffer, rows],
                            compressed_ref.at[buffer, rows],
                            sem.at[semaphore],
                        ).wait()

                def consume_rows(count):
                    consume(compressed_ref[buffer, :count], valid[:, :count])

                @pl.when(nonempty)
                def attend_page():
                    # Omit a causally empty suffix, down to one hardware lane-width tile.
                    def consume_prefix(count):
                        if count == LANES:
                            consume_rows(count)
                        else:
                            jax.lax.cond(
                                visible > step * selected_tile + count // 2,
                                lambda: consume_rows(count),
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

    # Request rows are contiguous; interior tiles need only one group dispatch.
    first_request = requests[block * query_tile]
    last_request = requests[jnp.minimum((block + 1) * query_tile - 1, requests.shape[0] - 1)]
    groups = jnp.where(first_request == last_request, 1, query_tile)
    jax.lax.fori_loop(0, groups, request_group, None)


@functools.partial(
    jax.jit,
    static_argnames=(
        "scale",
        "schedule",
        "window_size",
        "window_page_size",
        "compression_ratio",
        "interpret",
    ),
)
def csa_joint_attention(
    q,
    new_kv,
    window_cache,
    compressed_cache,
    topk_indices,
    attention_sink,
    metadata: CSAAttentionMetadata,
    *,
    scale: float,
    schedule: CSAAttentionSchedule,
    window_size: int = 128,
    window_page_size: int = 128,
    compression_ratio: int = 4,
    interpret: bool = False,
):
    """Return (BF16 [T,H,D] output, complete updated SWA [slots,D]).

    c4 [pages,P/4,D] is read-only and includes completed records from this chunk.
    Backend supplies rank-local read tables and write locations; it must skip its
    own SWA scatter. This chunk's KV remains separate until old history is consumed.
    Top-K entries are unique request-local compressed-record indices, with -1 padding.
    Decode uses SparseCore gather and requires at most one live query per request.
    """
    if q.ndim != 3 or q.dtype != jnp.bfloat16:
        raise ValueError("q must be BF16 [T,H,D]")
    tokens, heads, dim = q.shape
    if not heads or heads % 8 or dim != 512:
        raise ValueError("heads must be a positive multiple of eight; head_dim must be 512")
    if new_kv.shape != (tokens, dim) or new_kv.dtype != jnp.bfloat16:
        raise ValueError("new_kv must be BF16 [T,D]")
    if window_page_size not in (128, 256) or window_size <= 0 or window_size % LANES:
        raise ValueError("window_size must be lane-aligned; page size must be 128 or 256")
    if compression_ratio != 4:
        raise ValueError("CSA requires compression_ratio=4")
    if window_cache.ndim != 2 or window_cache.dtype != jnp.bfloat16 or window_cache.shape[1] != dim:
        raise ValueError("window_cache must be BF16 [slots,D]")
    if window_cache.shape[0] < window_page_size or window_cache.shape[0] % window_page_size:
        raise ValueError("SWA capacity must include aligned page zero")
    if (
        compressed_cache.ndim != 3
        or compressed_cache.dtype != jnp.bfloat16
        or compressed_cache.shape[2] != dim
        or compressed_cache.shape[1] != window_page_size // 4
        or compressed_cache.shape[0] < 1
    ):
        raise ValueError("c4 cache must be BF16 [pages,P/4,D]")
    if (
        topk_indices.ndim != 2
        or topk_indices.shape[0] != tokens
        or topk_indices.dtype != jnp.int32
        or topk_indices.shape[1] <= 0
        or topk_indices.shape[1] % LANES
    ):
        raise ValueError("Top-K must be int32 [T,K], K a positive multiple of 128")
    if attention_sink.shape != (heads,) or attention_sink.dtype != jnp.float32:
        raise ValueError("attention_sink must be FP32 [H]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    bt, tile = schedule.query_tile, schedule.selected_tile
    if schedule.decode and bt != 1:
        raise ValueError("SparseCore decode requires query_tile=1")
    if bt <= 0 or bt > 32 or bt & (bt - 1) or tile not in (128, 256):
        raise ValueError("schedule requires power-of-two query_tile <=32 and selected_tile 128/256")
    batch = metadata.seq_lens.size
    sizes = (tokens, batch + 1, batch, None, batch + 1, None, batch + 1, batch, tokens)
    for value, size in zip(metadata, sizes, strict=True):
        if value.ndim != 1 or value.dtype != jnp.int32 or (size is not None and value.size != size):
            raise ValueError("metadata must contain correctly sized int32 vectors")
    if not metadata.window_page_indices.size or not metadata.compressed_page_indices.size:
        raise ValueError("read tables must include a padding entry")
    if not tokens or not batch:
        return jnp.zeros_like(q), window_cache
    if schedule.decode and not interpret:
        output = _decode_attention(
            q,
            new_kv,
            window_cache,
            compressed_cache,
            topk_indices,
            attention_sink,
            metadata,
            scale=scale,
            window_size=window_size,
            window_page_size=window_page_size,
            compression_ratio=compression_ratio,
        )
    else:
        cps = compressed_cache.shape[1]
        dma_tile = math.gcd(cps, tile)
        parts = tile // dma_tile
        output = pl.pallas_call(
            functools.partial(
                _kernel,
                query_tile=bt,
                selected_tile=tile,
                dma_tile=dma_tile,
                page_size=cps,
                window_page_size=window_page_size,
                window_size=window_size,
                compression_ratio=compression_ratio,
                scale=scale,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=8,
                grid=(pl.cdiv(tokens, bt),),
                in_specs=(
                    pl.BlockSpec((bt, heads, dim), lambda b, *_: (b, 0, 0)),
                    *[pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(3)],
                    pl.BlockSpec((bt, 1, topk_indices.shape[1]), lambda b, *_: (b, 0, 0)),
                    pl.BlockSpec((heads,), lambda b, *_: (0,)),
                ),
                out_specs=pl.BlockSpec((bt, heads, dim), lambda b, *_: (b, 0, 0)),
                scratch_shapes=(
                    pltpu.VMEM((window_page_size, dim), jnp.bfloat16),
                    pltpu.VMEM((window_size, dim), jnp.bfloat16),
                    pltpu.VMEM((2, tile, dim), jnp.bfloat16),
                    pltpu.SMEM((2, parts), jnp.int32),
                    pltpu.SemaphoreType.DMA((2 + 2 * parts,)),
                    pltpu.VMEM((bt * heads, LANES), jnp.float32),
                    pltpu.VMEM((bt * heads, LANES), jnp.float32),
                    pltpu.VMEM((bt * heads, dim), jnp.float32),
                    pltpu.VMEM((max(window_page_size, window_size, tile), dim), jnp.float32),
                    pltpu.VMEM((max(window_page_size, window_size, tile), LANES), jnp.int32),
                    pltpu.VMEM((2, bt, tile), jnp.int32),
                ),
            ),
            out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            interpret=interpret,
            name="csa_bf16_joint_attention",
        )(
            *metadata[:8],
            q,
            jnp.pad(new_kv, ((0, -tokens % window_size), (0, 0))),
            window_cache,
            compressed_cache,
            topk_indices[:, None],
            attention_sink,
        )
    ids = jnp.arange(tokens)
    req = jnp.clip(metadata.query_seq_ids, 0, batch - 1)
    valid = (metadata.query_seq_ids >= 0) & (metadata.query_seq_ids < batch)
    valid &= (ids >= metadata.cu_q_lens[req]) & (ids < metadata.cu_q_lens[req + 1])
    loc = metadata.window_write_locations
    valid &= (loc >= window_page_size) & (loc < window_cache.shape[0])
    if not interpret:
        # Keep the aliased pool in HBM instead of staging the whole pool for a few row writes.
        window_cache = pltpu.with_memory_space_constraint(window_cache, pltpu.HBM)
    updated = paged_row_write(
        window_cache, new_kv, loc, valid, run=schedule.write_run, interpret=interpret
    )
    return output, updated
