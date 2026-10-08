"""CSA decode over request-local pages and exact selected cache slots."""

import os

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.csa_decode import paged_csa_decode_scores
from sgl_jax.srt.kernels.dsa.streamindex_topk import select_topk_indices
from sgl_jax.srt.kernels.dsv4.csa_decode_attention import gathered_decode_attention

_NEG_INF = jnp.finfo(jnp.float32).min

# Which indexer scores + selects the decode query's compressed entries:
#   "kernel": kernels/dsa/streamindex_topk over the request-local page table (same kernel
#             the prefill indexer uses; Pallas scoring + exact SparseCore selection);
#   "p370":   paged_csa_decode_scores (request-local Pallas scorer) + exact selector;
#   "auto":   "kernel" on TPU, "p370" elsewhere (the p370 scorer can run interpreted).
DECODE_INDEXER_BACKEND_ENV = "DSV4_DECODE_INDEXER_BACKEND"
_KERNEL_KV_PAGES_PER_BLOCK = int(os.environ.get("DSV4_INDEXER_KV_PAGES_PER_BLOCK", "64"))
_KERNEL_QUERIES_PER_BLOCK = (1, 64, 64)


def resolve_decode_indexer_backend(backend: str = "auto") -> str:
    if backend not in ("auto", "kernel", "p370"):
        raise ValueError(f"unknown CSA decode indexer backend {backend!r}")
    if backend == "auto":
        backend = os.environ.get(DECODE_INDEXER_BACKEND_ENV, "p370")
        if backend not in ("auto", "kernel", "p370"):
            raise ValueError(
                f"{DECODE_INDEXER_BACKEND_ENV}={backend!r} must be auto, kernel or p370"
            )
    if backend == "auto":
        return "kernel" if jax.default_backend() == "tpu" else "p370"
    return backend


def csa_decode_select_p370(
    index_q,
    index_weights,
    index_cache,
    pages,
    lengths,
    *,
    take,
    ratio,
    compressed_page_size,
    topk_backend="auto",
    page_segments=None,
    page_segment_counts=None,
):
    """Request-local Pallas scorer + exact selector: ``(selected [B, take], valid [B, take])``.

    ``page_segments`` / ``page_segment_counts`` are the host page-run segmentation of
    ``pages`` (``kernels.csa_decode.page_run_segments``); without them the scorer splits
    the table into single pages on device.
    """
    del ratio
    scores = paged_csa_decode_scores(
        index_q,
        index_weights,
        index_cache.reshape(-1, index_cache.shape[-1]),
        lengths,
        pages,
        page_size=compressed_page_size,
        segments=page_segments,
        segment_counts=page_segment_counts,
        interpret=jax.default_backend() != "tpu",
    )
    return select_decode_entries(scores, lengths, take=take, topk_backend=topk_backend)


def csa_decode_select_kernel(
    index_q,
    index_weights,
    index_cache,
    pages,
    lengths,
    *,
    take,
    ratio,
    compressed_page_size,
    topk_backend="auto",
):
    """Same contract as `csa_decode_select_p370` through ``kernels/dsa/streamindex_topk``.

    ``pages`` is the request-local page table (``decode_page_indices``, one row per packed
    decode query, zeros for padded rows), which is exactly the paged layout the kernel
    reads; ``lengths`` are completed compressed groups, so ``seq_lens = lengths * ratio``
    makes the kernel's completed-groups rule admit exactly those entries. Selections come
    back in request-local entry coordinates, the same as the p370 path.
    """
    from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk

    if compressed_page_size % 2:
        raise ValueError("compressed_page_size must be even for the paged indexer layout")
    num_queries = index_q.shape[0]
    lengths = jnp.asarray(lengths, jnp.int32)
    cache_kv = jnp.asarray(index_cache).reshape(-1, compressed_page_size // 2, 2, index_q.shape[-1])
    selected = streamindex_topk(
        q=jnp.asarray(index_q).astype(jnp.bfloat16),
        indexer_weights=jnp.asarray(index_weights, jnp.float32),
        cache_kv=cache_kv,
        seq_lens=lengths * ratio,
        page_indices=jnp.asarray(pages, jnp.int32).reshape(-1),
        cu_q_lens=jnp.arange(num_queries + 1, dtype=jnp.int32),
        distribution=jnp.asarray((num_queries, num_queries, num_queries), jnp.int32),
        k=take,
        compression_ratio=ratio,
        num_kv_pages_per_block=_KERNEL_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_KERNEL_QUERIES_PER_BLOCK,
        topk_backend=topk_backend,
    )
    valid = (selected >= 0) & (selected < lengths[:, None])
    return selected, valid


def select_decode_entries(scores, lengths, *, take: int, topk_backend: str = "auto"):
    """Exact top-``take`` entries per decode query, ``(selected [B, take], valid [B, take])``.

    Uses the shared DSA exit-stage selector (SparseCore radix select on chips that have
    one, XLA otherwise) instead of a full-row sort. ``scores`` marks unusable entries
    with ``_NEG_INF``; entries at or beyond ``lengths`` are also invalid.
    """
    masked = jnp.where(scores > _NEG_INF, scores, -jnp.inf)
    selected = select_topk_indices(masked, take, backend=topk_backend)
    valid = (selected >= 0) & (selected < lengths[:, None])
    return selected, valid


def short_kv_kernel_enabled() -> bool:
    """``DSV4_DECODE_SHORT_KV_KERNEL=1``: stream pages through the HCA Pallas kernel."""
    return os.environ.get("DSV4_DECODE_SHORT_KV_KERNEL", "1") == "1"


def _short_kv_streaming_attention(
    q,
    compressed_cache,
    window_cache,
    pages,
    window_rows,
    *,
    lengths,
    query_positions,
    valid_token_mask,
    attention_sink,
    softmax_scale,
    compressed_page_size,
    capacity,
):
    """All entries attend: reuse the HCA paged streaming kernel over the C4 pool.

    The kernel reads compressed pages by page table (no gather of the whole
    bucket), keeps the SWA rows as one gathered tile, and treats the first
    ``window_len`` window rows as valid, so the CSA window (newest last,
    invalid rows first for positions < window) is rotated valid-first.
    """
    from sgl_jax.srt.kernels.hca.attention import _streaming_attention
    from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule

    tokens, heads, head_dim = q.shape
    window = window_rows.shape[1]
    window_len = jnp.where(valid_token_mask, jnp.clip(query_positions + 1, 0, window), 0).astype(
        jnp.int32
    )
    shift = window - window_len
    order = (jnp.arange(window, dtype=jnp.int32)[None, :] + shift[:, None]) % window
    rotated_rows = jnp.take_along_axis(window_rows, order, axis=1)
    window_kv = jnp.take(window_cache, rotated_rows, axis=0).astype(jnp.bfloat16)
    # Off-TPU (CPU interpret tests) there is no device kind to key the schedule
    # table; use the v7x row, which is what the interpret tests exercise.
    device_kind = jax.devices()[0].device_kind if jax.default_backend() == "tpu" else "TPU7x"
    schedule = get_hca_kernel_schedule(
        device_kind,
        page_size=compressed_page_size,
        max_compressed_entries=capacity,
        local_heads=heads,
        head_dim=head_dim,
    )
    out = _streaming_attention(
        q.astype(jnp.bfloat16),
        window_kv,
        window_len,
        compressed_cache,
        pages.reshape(-1).astype(jnp.int32),
        (jnp.arange(tokens, dtype=jnp.int32) * pages.shape[1]),
        lengths.astype(jnp.int32),
        attention_sink.astype(jnp.float32),
        schedule=schedule,
        softmax_scale=softmax_scale,
        compressed_page_size=compressed_page_size,
    )
    return jnp.where(valid_token_mask[:, None, None], out.astype(jnp.float32), 0.0)


def _take_pages_onehot(pages, page_index):
    """Page lookup as three small MXU matmuls (exact for page ids below 2**24).

    ``onehot[t, k, n] = (n == page_index[t, k])`` in bf16 against the page table split
    into three bytes (each byte is an integer <= 255, exactly representable in bf16; a
    single non-zero product per output element keeps the f32 sum exact). The one-hot
    tensor is ``T x K x N`` bf16 (64 x 512 x 512 = 32 MiB at 64K context), which XLA
    streams at HBM rate, against the per-element gather slow path it replaces.
    """
    rows, table = pages.shape
    pages = pages.astype(jnp.int32)
    digits = jnp.stack(
        ((pages & 0xFF), ((pages >> 8) & 0xFF), ((pages >> 16) & 0xFF)), axis=-1
    ).astype(jnp.bfloat16)
    onehot = (
        jnp.arange(table, dtype=jnp.int32)[None, None, :]
        == page_index.astype(jnp.int32)[:, :, None]
    ).astype(jnp.bfloat16)
    values = jnp.einsum("tkn,tnc->tkc", onehot, digits, preferred_element_type=jnp.float32)
    values = values.astype(jnp.int32)
    return values[..., 0] | (values[..., 1] << 8) | (values[..., 2] << 16)


def _take_pages(pages, page_index):
    """``pages[b, page_index[b, k]]`` for every decode row.

    XLA lowers both the batched ``take_along_axis(pages, idx, axis=1)`` and the flat
    1-D gather on TPU into a per-element s32 gather slow path (64 rows x 512 entries:
    4.2 ms and 3.9 ms per step at bs=64 in v7x decode profiles). The default expresses
    the lookup as one-hot matmuls instead (``_take_pages_onehot``). ``DSV4_DECODE_PAGE_TAKE``
    selects a fallback: ``gather`` (flat 1-D take), ``2d`` (batched take_along_axis).
    A Pallas SMEM scalar-lookup kernel measured 3 ms/step slower than the gather
    and was removed.
    """
    mode = os.environ.get("DSV4_DECODE_PAGE_TAKE", "onehot")
    if mode == "2d":
        return jnp.take_along_axis(pages, page_index, axis=1)
    if mode == "gather":
        rows, table = pages.shape
        offsets = (jnp.arange(rows, dtype=jnp.int32) * table)[:, None]
        return jnp.take(pages.reshape(-1), offsets + page_index.astype(jnp.int32), axis=0)
    return _take_pages_onehot(pages, page_index)


def csa_decode_attention(
    q,
    index_q,
    index_weights,
    index_cache,
    compressed_cache,
    window_cache,
    pages,
    window_rows,
    *,
    query_positions,
    valid_token_mask,
    attention_sink,
    softmax_scale,
    compressed_page_size,
    index_topk,
    ratio,
    page_segments=None,
    page_segment_counts=None,
):
    """Read selected compressed slots plus the SWA union, including the sink.

    Page and SWA addresses have one row per packed decode query; padded queries
    own no entries. Both caches already contain this step's compressor/KV writes.
    This keeps prefill on its shared-history path and avoids cross-request decode
    score matrices, global top-k, and gathering every compressed KV candidate.
    """
    lengths = jnp.where(valid_token_mask, (query_positions + 1) // ratio, 0)
    capacity = pages.shape[1] * compressed_page_size
    if capacity <= index_topk:
        if short_kv_kernel_enabled():
            return _short_kv_streaming_attention(
                q,
                compressed_cache,
                window_cache,
                pages,
                window_rows,
                lengths=lengths,
                query_positions=query_positions,
                valid_token_mask=valid_token_mask,
                attention_sink=attention_sink,
                softmax_scale=softmax_scale,
                compressed_page_size=compressed_page_size,
                capacity=capacity,
            )
        # Every entry of this decode bucket fits in the top-k budget, so exact
        # top-k selects all valid entries: skip indexer scoring, selection, and
        # the sort (the GPU serving path takes the same kv_len <= topk shortcut).
        selected = jnp.broadcast_to(
            jnp.arange(capacity, dtype=jnp.int32)[None, :], (lengths.shape[0], capacity)
        )
        selected_valid = selected < lengths[:, None]
    else:
        take = min(index_topk, capacity)
        select = (
            csa_decode_select_kernel
            if resolve_decode_indexer_backend() == "kernel"
            else csa_decode_select_p370
        )
        extra = {}
        if select is csa_decode_select_p370:
            extra = dict(page_segments=page_segments, page_segment_counts=page_segment_counts)
        selected, selected_valid = select(
            index_q,
            index_weights,
            index_cache,
            pages,
            lengths,
            take=take,
            ratio=ratio,
            compressed_page_size=compressed_page_size,
            **extra,
        )
        # Preserve original entry order during attention and gather nearby slots
        # together. Exact top-k is unchanged; invalid selections sort to the end.
        selected = jnp.sort(jnp.where(selected_valid, selected, capacity), axis=-1)
        selected_valid = selected < lengths[:, None]
    safe_selected = jnp.where(selected_valid, selected, 0)
    physical_pages = _take_pages(pages, safe_selected // compressed_page_size)
    slots = physical_pages * compressed_page_size + safe_selected % compressed_page_size

    # Both index sets are in bounds by construction (page table entries and window
    # rows are allocator slots, padded rows point at row 0); ``promise_in_bounds``
    # drops the fill select XLA otherwise runs over the gathered [T, 512+128, D] block
    # (0.7 ms per bs=64 step over the CSA layers on v7x).
    compressed = compressed_cache.at[slots].get(mode="promise_in_bounds")
    window = window_cache.at[window_rows].get(mode="promise_in_bounds")
    window_positions = (
        query_positions[:, None] - window_rows.shape[1] + 1 + jnp.arange(window_rows.shape[1])
    )
    window_valid = valid_token_mask[:, None] & (window_positions >= 0)
    out = gathered_decode_attention(
        q,
        window,
        compressed,
        window_valid,
        selected_valid,
        attention_sink,
        softmax_scale=softmax_scale,
    )
    return jnp.where(valid_token_mask[:, None, None], out, 0.0)
