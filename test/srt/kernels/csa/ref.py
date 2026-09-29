"""NumPy-only CSA composition oracle, including ring/state/cache updates."""

from test.srt.kernels.csa_attention.ref import reference as attention_reference
from test.srt.kernels.csa_attention.ref import update_window
from test.srt.kernels.csa_compressor import ref as compressor

import numpy as np


def topk(q, weights, index_cache, metadata, k):
    q, weights = np.asarray(q, np.float32), np.asarray(weights, np.float32)
    attention = metadata.attention
    reqs, cu, ends, _, _, pages, offsets, _ = map(np.asarray, attention[:8])
    positions = np.asarray(metadata.compressor.positions)
    page_size = index_cache.shape[1] * index_cache.shape[2]
    keys = compressor.decode_index(np.asarray(index_cache).reshape(-1, 256))
    keys = keys.reshape(index_cache.shape[0], page_size, 128)
    result = np.full((q.shape[0], k), -1, np.int32)
    for token, request in enumerate(reqs):
        if request < 0:
            continue
        count = (positions[token] + 1) // 4
        if count <= k:
            result[token, :count] = np.arange(count)
            continue
        table = pages[offsets[request] // page_size : offsets[request + 1] // page_size]
        kv = keys[table].reshape(-1, 128)[:count]
        logits = np.maximum(q[token] @ kv.T, 0)
        scores = (weights[token, :, None] * logits).sum(axis=0, dtype=np.float32)
        result[token] = np.argsort(-scores, kind="stable")[:k]
    return result


def reference(
    inputs, cache, metadata, *, projection_k_tile, query_tile, top_k=512, window_size=128
):
    x, weight, ma, ia, mn, inn, cos, sin, iq, iw, q, new, sink = inputs
    states = [np.array(v, copy=True) for v in cache[:2]]
    caches = [np.array(v, copy=True) for v in cache[2:5]]
    projection = compressor.compressor_projection(
        x,
        weight,
        [np.asarray(b.token_indices) for b in metadata.compressor.blocks],
        k_tile=projection_k_tile,
        query_tile=query_tile,
    )
    positions, cu, slots, locs = map(np.asarray, metadata.compressor[:4])
    for r, slot in enumerate(slots):
        if slot < 0:
            continue
        for t in range(cu[r], cu[r + 1]):
            pooled = []
            for s, part, ape, norm in (
                (0, projection[t : t + 1, :2048], ma, mn),
                (1, projection[t : t + 1, 2048:], ia, inn),
            ):
                value, emit, state = compressor.compressor_step(
                    part,
                    states[s][slot : slot + 1],
                    ape,
                    norm,
                    cos,
                    sin,
                    positions[t : t + 1],
                )
                states[s][slot] = state[0]
                pooled.append(value)
            if emit[0] and locs[t] >= 0:
                records = (*compressor.pack_main(pooled[0]), compressor.pack_index(pooled[1]))
                for target, record in zip(caches, records, strict=True):
                    target.reshape(-1, record.shape[-1])[locs[t]] = record[0]
    indices = topk(iq, iw, caches[-1], metadata, top_k)
    output = attention_reference(
        q,
        new,
        cache[-1],
        *caches[:2],
        indices,
        sink,
        metadata.attention,
        scale=q.shape[-1] ** -0.5,
        window_size=window_size,
    )
    window = update_window(new, cache[-1], metadata.attention, window_size=window_size)
    return output, (*states, *caches, window), indices
