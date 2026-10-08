"""Cache-aware native V4 attention, including the CSA compressor and indexer.

Models supply projections and weight values. This executor owns rank-local cache
views, slot resets, sharding and update packaging; no model parameters live here.
The same native implementation supports SWA and small-geometry HCA validation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.csa_decode import page_run_segments, scorer_pages_per_block
from sgl_jax.srt.kernels.dsv4.state_init import (
    init_state_slots,
    state_init_kernel_enabled,
)
from sgl_jax.srt.layers.attention.dsv4.dispatch import (
    ReadTables,
    read_tables,
    run_layer,
)


class CompressorWeights(NamedTuple):
    wkv: jax.Array
    wgate: jax.Array
    ape: jax.Array
    norm_weight: jax.Array
    cos_sin_cache: jax.Array
    # Optional pre-split halves of ``cos_sin_cache`` (built once at load): the HCA
    # kernel takes separate cos/sin tables, and slicing the full table inside the
    # jitted step costs a relayout of the whole table every step.
    cos_table: jax.Array | None = None
    sin_table: jax.Array | None = None
    # Optional ``[hidden, 2*D]`` bf16 ``[Wkv|Wgate]^T`` built once at load for the HCA
    # kernels (see ``DeepseekV4Compressor.prepare_fused_projection``).
    fused: jax.Array | None = None


_COMPRESS_FIELDS = ("wkv", "wgate", "ape", "norm_weight", "cos_sin_cache")


def _compress_kwargs(weights):
    """The `compress_chunk` keyword arguments of a CompressorWeights (drops the pre-split tables)."""
    d = weights._asdict()
    return {k: d[k] for k in _COMPRESS_FIELDS}


class IndexerInputs(NamedTuple):
    q: jax.Array
    weights: jax.Array
    compressor: CompressorWeights


def padded_read_tables(
    *,
    request_pool,
    allocator,
    slots,
    lengths,
    q_lens,
    ratio,
    window_size,
    page_size,
    max_context_len,
    token_capacity,
    rank,
    compressed_capacity=None,
    decode_capacity=None,
    minimal=False,
):
    if decode_capacity is not None or minimal:
        # Request-local decode: the layer reads through ``decode_page_indices`` /
        # ``decode_window_rows`` only, so the flat shared-history tables (every
        # compressed entry of every request: ~150k rows at bs=64 / 9K, 7 ms of host
        # time per step to enumerate and 3 MB to upload) are not built; one inert row
        # keeps the pytree shape. ``minimal`` asks for the same when the ratio's layers
        # never read these tables (ratio 128 under the Pallas HCA backend).
        tables = ReadTables(
            window_rows=np.zeros((1,), np.int32),
            window_positions=np.full((1,), -1, np.int32),
            window_request_ids=np.full((1,), -1, np.int32),
            compressed_rows=np.zeros((1,), np.int32),
            compressed_entry_ids=np.full((1,), -1, np.int32),
            compressed_request_ids=np.full((1,), -1, np.int32),
        )
        window_capacity = compressed_capacity = 1
    else:
        tables = read_tables(
            request_pool=request_pool,
            allocator=allocator,
            slots=slots,
            lengths=lengths,
            q_lens=q_lens,
            ratio=ratio,
            window_size=window_size,
            page_size=page_size,
            rank=rank,
        )
        # Window reads are bounded by the query bucket and sliding-window halo.
        window_capacity = max(
            1, min(len(slots) * max_context_len, token_capacity + len(slots) * (window_size - 1))
        )
    if compressed_capacity is None:
        # Bucket actual gathered history, not padded request slots times the
        # configured maximum context. The caller shares this bucket across DP.
        count = len(tables.compressed_rows)
        compressed_capacity = max(128, 1 << (max(1, count) - 1).bit_length()) if ratio else 1
    values = {}
    for name in ReadTables.__slots__:
        value = getattr(tables, name)
        if value is None:
            continue
        capacity = window_capacity if name.startswith("window") else compressed_capacity
        if len(value) > capacity:
            raise ValueError(f"{name} exceeds the configured V4 metadata capacity")
        fill = 0 if name.endswith("rows") else -1
        value = np.asarray(value)
        padded = np.full((capacity,) + value.shape[1:], fill, value.dtype)
        padded[: len(value)] = value
        values[name] = padded
    if decode_capacity is not None:
        if ratio != 4 or np.any((q_lens != 0) & (q_lens != 1)):
            raise ValueError("request-local CSA tables require one decode query per request")
        compressed_page_size = page_size // ratio
        table_width = decode_capacity // compressed_page_size
        pages = np.zeros((token_capacity, table_width), np.int32)
        page_counts = np.zeros((token_capacity,), np.int32)
        windows = np.zeros((token_capacity, window_size), np.int32)
        mapping = allocator.full_to_swa_index_mapping
        mapping = mapping[rank] if isinstance(mapping, list) else mapping
        # One decode query per live request, packed in request order (rows past the
        # live count stay zero); vectorised across requests.
        live = np.flatnonzero(np.asarray(q_lens) > 0)
        if live.size:
            live_slots = np.asarray(slots, np.int64)[live]
            live_lengths = np.asarray(lengths, np.int64)[live]
            complete = live_lengths // ratio
            counts = (complete + compressed_page_size - 1) // compressed_page_size  # [L]
            if np.any(counts > table_width):
                raise ValueError("CSA decode page table narrower than a request's history")
            starts = np.arange(table_width, dtype=np.int64) * page_size  # [N]
            page_mask = starts[None, :] < counts[:, None] * page_size
            anchors = np.asarray(
                request_pool.req_to_token[live_slots[:, None], starts[None, :]], np.int64
            )
            if np.any(page_mask & ((anchors < page_size) | (anchors % page_size != 0))):
                raise ValueError("CSA compressed pages must start at allocated page boundaries")
            pages[: live.size] = np.where(page_mask, anchors // page_size, 0)
            page_counts[: live.size] = counts
            positions = live_lengths[:, None] - window_size + np.arange(window_size)[None, :]
            in_range = positions >= 0
            locations = np.asarray(
                request_pool.req_to_token[live_slots[:, None], np.maximum(positions, 0)], np.int64
            )
            windows[: live.size] = np.where(in_range, mapping[locations], 0)
        values["decode_page_indices"] = pages
        values["decode_window_rows"] = windows
        # DMA segments for the request-local scorer (runs of consecutive physical pages),
        # built once per step here rather than per CSA layer on device.
        segments, segment_counts = page_run_segments(
            pages, page_counts, scorer_pages_per_block(decode_capacity, compressed_page_size)
        )
        values["decode_page_segments"] = segments
        values["decode_page_segment_counts"] = segment_counts
    return ReadTables(**values)


def _reset_state(state, metadata):
    limit = state.shape[0] - 1  # The last local slot is padding, not a request.
    slots = metadata.request_slots
    valid = metadata.state_init_mask & metadata.request_valid_mask & (slots >= 0) & (slots < limit)
    destinations = jnp.where(valid, slots, state.shape[0])
    if state_init_kernel_enabled():
        # One template slot, DMA'd only into the valid destinations (none in decode).
        template = (
            jnp.zeros(state.shape[1:], state.dtype).at[..., state.shape[-1] // 2 :].set(-jnp.inf)
        )
        return init_state_slots(state, destinations, template)
    empty = jnp.zeros((slots.shape[0], *state.shape[1:]), state.dtype)
    empty = empty.at[..., state.shape[-1] // 2 :].set(-jnp.inf)
    return state.at[destinations].set(empty, mode="drop")


def run_dsv4_attention(
    mesh,
    q,
    new_kv,
    *,
    hidden_states,
    layer_id,
    ratio,
    metadata,
    tables,
    token_to_kv_pool,
    compressor_state_pool,
    compressor,
    indexer,
    attention_sink,
    softmax_scale,
    rope_head_dim,
    norm_eps,
    index_topk,
    hidden_local=False,
):
    kv_pool = token_to_kv_pool
    states = compressor_state_pool
    family = f"c{ratio}"
    window = kv_pool.get_swa_buffer(layer_id)
    compressed = kv_pool.get_compressed_buffer(layer_id) if ratio else None
    state = states.get_buffer(family, layer_id) if ratio else None
    index_cache = kv_pool.get_indexer_buffer(layer_id) if ratio == 4 else None
    index_state = states.get_buffer("indexer", layer_id) if ratio == 4 else None
    if ratio and compressor is None:
        raise ValueError("compressed attention requires model compressor weights")
    if ratio == 4 and indexer is None:
        raise ValueError("CSA requires model indexer projections and compressor weights")

    def local(
        q_,
        kv_,
        x_,
        window_,
        compressed_,
        state_,
        index_cache_,
        index_state_,
        md,
        read,
        cw,
        iq,
        iw,
        icw,
        sink,
    ):
        valid = md.valid_token_mask[:, None]
        if not hidden_local:
            # row-block input: the row-sharded compressors never read padded rows
            x_ = jnp.where(valid, x_, 0)
        kv_ = jnp.where(valid, kv_, 0)
        q_ = jnp.where(valid[:, :, None], q_, 0)
        buffers = {"swa": window_.reshape(-1, window_.shape[-1])}
        if ratio:
            buffers["compressed"] = compressed_.reshape(-1, compressed_.shape[-1])
            state_shape = state_.shape
            # C128 retains HCA's native [slots, 128, 2, D] physical layout.
            # The generic/reference compressor consumes the equivalent packed
            # [slots, 128, 2*D] view, including when resetting recycled slots.
            state_ = state_.reshape(state_.shape[:2] + (-1,))
            state_ = _reset_state(state_, md)
        idx = None
        if ratio == 4:
            buffers["indexer"] = index_cache_.reshape(-1, index_cache_.shape[-1])
            idx = dict(
                q=iq,
                weights=iw,
                compressor_input=x_,
                state=_reset_state(index_state_, md),
                head_dim=iq.shape[-1],
                rope_head_dim=rope_head_dim,
                compressor_weights=_compress_kwargs(icw),
            )
        output, updates = run_layer(
            q=q_,
            new_kv=kv_,
            compressor_input=x_,
            layer_id=layer_id,
            ratio=ratio,
            metadata=md,
            tables=read,
            kv_buffers=buffers,
            state=state_,
            compressor_weights=None if cw is None else _compress_kwargs(cw),
            indexer=idx,
            attention_sink=sink,
            softmax_scale=softmax_scale,
            window_size=md.window_size,
            head_dim=q_.shape[-1],
            index_topk=index_topk,
            rope_head_dim=rope_head_dim,
            norm_eps=norm_eps,
        )
        out = {"swa": updates["swa"].reshape(window_.shape)}
        if ratio:
            out["compressed"] = updates["compressed"].reshape(compressed_.shape)
            out["state"] = updates["state"].reshape(state_shape)
        if ratio == 4:
            out["indexer"] = updates["indexer"].reshape(index_cache_.shape)
            out["indexer_state"] = updates["indexer_state"]
        return output, out

    replica = lambda tree: jax.tree.map(lambda a: P(*([None] * a.ndim)), tree)
    data_spec = lambda array: P("data", *([None] * (array.ndim - 1)))
    outputs = {"swa": data_spec(window)}
    if ratio:
        outputs.update({"compressed": data_spec(compressed), "state": data_spec(state)})
    if ratio == 4:
        outputs.update({"indexer": P("data", None, None), "indexer_state": P("data", None, None)})
    specs = (P("data", "tensor", None), outputs)
    named = jax.tree.map(lambda p: NamedSharding(mesh, p), specs)
    fn = jax.shard_map(
        local,
        mesh=None,
        in_specs=(
            P("data", "tensor", None),
            P("data", None),
            # hidden: full chunk on every device, or this device's row block only
            # (DSV4_LOWRANK_AG: the compressors run on local rows + a ppermute halo)
            P("tensor", None) if hidden_local else P("data", None),
            P("data", None),
            data_spec(compressed) if ratio else None,
            data_spec(state) if ratio else None,
            P("data", None, None) if ratio == 4 else None,
            P("data", None, None) if ratio == 4 else None,
            jax.tree.map(lambda _: P("data"), metadata),
            jax.tree.map(lambda _: P("data"), tables),
            replica(compressor),
            P("data", None, None) if indexer else None,
            P("data", None) if indexer else None,
            replica(indexer.compressor) if indexer else None,
            P("tensor"),
        ),
        out_specs=specs,
        check_vma=False,
    )
    fn = jax.sharding.auto_axes(fn, axes=mesh.axis_names, out_sharding=named)
    return fn(
        q,
        new_kv,
        hidden_states,
        window,
        compressed,
        state,
        index_cache,
        index_state,
        metadata,
        tables,
        compressor,
        indexer.q if indexer else None,
        indexer.weights if indexer else None,
        indexer.compressor if indexer else None,
        attention_sink,
    )
