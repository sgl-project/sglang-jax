"""Independent NumPy dense joint attention over native BF16 cache inputs."""

import numpy as np


def update_window(new_kv, window_cache, metadata, *, window_size, page_size):
    result = np.array(window_cache, copy=True)
    requests, cu, ends, pages, offsets = map(np.asarray, metadata[:5])
    for request in range(len(ends)):
        begin, end = cu[request], cu[request + 1]
        prefix = ends[request] - (end - begin)
        for token in range(max(begin, end - window_size), end):
            if token >= len(requests) or requests[token] != request:
                continue
            if metadata[-1][token] < 0:
                continue
            slot = (prefix + token - begin) % window_size
            entry = offsets[request] // page_size + slot // page_size
            if not (0 <= entry < len(pages) and entry < offsets[request + 1] // page_size):
                continue
            page = pages[entry]
            if 0 < page < len(result) // page_size:
                row = slot % page_size
                result[page * page_size + row] = new_kv[token]
    return result


def reference(
    q,
    new_kv,
    window,
    compressed,
    indices,
    sink,
    metadata,
    *,
    scale,
    window_size=128,
    compression_ratio=4,
    window_page_size=128,
):
    q, new_kv, window = (np.asarray(x, np.float32) for x in (q, new_kv, window))
    compressed, indices, sink = (np.asarray(x) for x in (compressed, indices, sink))
    reqs, cu, ends, wp, wc, cp, cc, compressed_lens = (np.asarray(x) for x in metadata[:8])
    output = np.zeros(q.shape, np.float32)
    wps, cps = window_page_size, compressed.shape[1]

    def resolve(table, offsets, request, logical, size, pages):
        at = offsets[request] // size + logical // size
        if logical < 0 or not offsets[request] // size <= at < offsets[request + 1] // size:
            return None
        if not 0 <= at < len(table) or not 0 < table[at] < pages:
            return None
        return int(table[at]), logical % size

    for token, request in enumerate(reqs):
        if not 0 <= request < len(ends) or not cu[request] <= token < cu[request + 1]:
            continue
        prefix = int(ends[request] - (cu[request + 1] - cu[request]))
        position = prefix + token - cu[request]
        rows = []
        for absolute in range(max(0, position - window_size + 1), position + 1):
            if absolute >= prefix:
                rows.append(new_kv[cu[request] + absolute - prefix])
            else:
                location = resolve(wp, wc, request, absolute % window_size, wps, len(window) // wps)
                if location is not None:
                    page, slot = location
                    rows.append(window[page * wps + slot])
        for index in indices[token]:
            if not 0 <= index < min(compressed_lens[request], (position + 1) // compression_ratio):
                continue
            location = resolve(cp, cc, request, int(index), cps, len(compressed))
            if location is None:
                continue
            page, slot = location
            rows.append(compressed[page, slot].astype(np.float32))
        if not rows:
            continue
        kv = np.asarray(rows, np.float32)
        scores = q[token] @ kv.T * np.float32(scale)
        maximum = np.maximum(scores.max(axis=-1, keepdims=True), sink[:, None])
        weight = np.exp(scores - maximum)
        denominator = weight.sum(axis=-1, keepdims=True) + np.exp(sink[:, None] - maximum)
        output[token] = (weight @ kv) / denominator
    return output
