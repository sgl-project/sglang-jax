"""Host-side metadata preparation for the single-device CSA operator.

The caller supplies page tables and state slots; this module does not allocate
cache storage or register a model-serving backend.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.csa import CSAMetadata
from sgl_jax.srt.kernels.csa_attention import CSAAttentionMetadata
from sgl_jax.srt.kernels.csa_compressor import CompressorBlock, CompressorMetadata
from sgl_jax.srt.kernels.csa_compressor.tune import CSA_COMPRESSION_RATIO


def prepare_csa_metadata(
    query_lengths,
    prefix_lengths,
    state_indices,
    window_pages,
    compressed_pages,
    *,
    num_tokens,
    window_size,
    window_page_size,
    compressed_page_size,
):
    """Build metadata from host arrays; page 0 and token/request -1 are padding.

    Page tables are [B, pages_per_request], with exclusive physical page ownership.
    State slots of active requests must be unique. Token padding is trailing only.
    """
    lengths, prefixes, slots, wp, cp = [
        np.asarray(v, np.int32)
        for v in (query_lengths, prefix_lengths, state_indices, window_pages, compressed_pages)
    ]
    batch = lengths.size
    if (
        lengths.ndim != 1
        or batch == 0
        or prefixes.shape != lengths.shape
        or slots.shape != lengths.shape
    ):
        raise ValueError(
            "lengths, prefixes and state_indices must have matching nonempty [B] shapes"
        )
    if (lengths < 0).any() or (prefixes < 0).any() or num_tokens < max(1, lengths.sum()):
        raise ValueError("invalid lengths or insufficient token capacity")
    if (
        min(window_size, window_page_size, compressed_page_size) <= 0
        or window_size % window_page_size
    ):
        raise ValueError("window_size must be a positive multiple of window_page_size")
    if any(p.ndim != 2 or p.shape[0] != batch or p.shape[1] == 0 for p in (wp, cp)):
        raise ValueError("page tables must be fixed-stride [B, pages_per_request]")
    active = lengths > 0
    if (slots[active] < 0).any() or len(np.unique(slots[active])) != active.sum():
        raise ValueError("active requests must own distinct nonnegative state slots")
    ratio = CSA_COMPRESSION_RATIO
    ends = prefixes + lengths
    if (
        wp.shape[1] * window_page_size != window_size
        or (ends // ratio > cp.shape[1] * compressed_page_size).any()
    ):
        raise ValueError("page table capacity does not cover the context")
    for pages, required in (
        (wp, np.where(active, wp.shape[1], 0)),
        (
            cp,
            np.where(active, (ends // ratio + compressed_page_size - 1) // compressed_page_size, 0),
        ),
    ):
        used = np.concatenate([p[:n] for p, n in zip(pages, required, strict=True)])
        if (used <= 0).any() or len(np.unique(used)) != len(used):
            raise ValueError("active page tables must contain distinct positive physical pages")
    cu = np.asarray((0, *np.cumsum(lengths)), np.int32)
    positions = np.full(num_tokens, -1, np.int32)
    reqs = np.full(num_tokens, -1, np.int32)
    locations = np.full(num_tokens, -1, np.int32)
    window_locations = np.full(num_tokens, -1, np.int32)
    for r, (n, prefix) in enumerate(zip(lengths, prefixes, strict=True)):
        ids = np.arange(cu[r], cu[r + 1])
        pos = prefix + np.arange(n)
        positions[ids], reqs[ids] = pos, r
        emit = (pos + 1) % ratio == 0
        entries = pos[emit] // ratio
        locations[ids[emit]] = (
            cp[r, entries // compressed_page_size] * compressed_page_size
            + entries % compressed_page_size
        )
        # Keep only the newest writer for each circular slot, even for long prefill.
        keep = np.arange(n) >= max(0, n - window_size)
        ring = pos[keep] % window_size
        window_locations[ids[keep]] = (
            wp[r, ring // window_page_size] * window_page_size + ring % window_page_size
        )
    blocks = []
    for length, start in sorted(set(zip(lengths, prefixes % ratio, strict=True))):
        if not length:
            continue
        requests = np.flatnonzero((lengths == length) & (prefixes % ratio == start)).astype(
            np.int32
        )
        leading = min(length, (-start) % ratio)
        stop = leading + (length - leading) // ratio * ratio
        intervals = [(i, i + 1) for i in range(leading)]
        if stop > leading:
            intervals.append((leading, stop))
        intervals.extend((i, i + 1) for i in range(stop, length))
        for begin, end in intervals:
            blocks.append(
                CompressorBlock(
                    cu[requests, None] + np.arange(begin, end, dtype=np.int32), requests
                )
            )
    slots = np.where(active, slots, -1)
    decode = 0
    while decode < batch and lengths[decode] == 1:
        decode += 1
    metadata = CSAMetadata(
        CompressorMetadata(positions, cu, slots, locations, tuple(blocks)),
        CSAAttentionMetadata(
            reqs,
            cu,
            ends,
            wp.ravel(),
            np.arange(batch + 1, dtype=np.int32) * window_size,
            cp.ravel(),
            np.arange(batch + 1, dtype=np.int32) * cp.shape[1] * compressed_page_size,
            ends // ratio,
            window_locations,
        ),
        np.asarray((decode, decode, batch), np.int32),
    )
    return jax.tree.map(jnp.asarray, metadata)
