"""Independent NumPy dense joint attention over the actual encoded cache inputs."""

import ml_dtypes
import numpy as np


def reference(
    q,
    new_kv,
    window,
    nope,
    rope,
    indices,
    sink,
    metadata,
    *,
    scale,
    window_size=128,
    compression_ratio=4,
    fp8_scale_block=64,
    rows_per_group=4,
):
    q, new_kv, window = (np.asarray(x, np.float32) for x in (q, new_kv, window))
    nope, rope, indices, sink = (np.asarray(x) for x in (nope, rope, indices, sink))
    reqs, cu, ends, wp, wc, cp, cc, compressed_lens = (np.asarray(x) for x in metadata)
    output = np.zeros(q.shape, np.float32)
    wps, cps = window.shape[1] * 2, nope.shape[1]

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
                location = resolve(wp, wc, request, absolute % window_size, wps, len(window))
                if location is not None:
                    page, slot = location
                    rows.append(window[page, slot // 2, slot % 2])
        for index in indices[token]:
            if not 0 <= index < min(compressed_lens[request], (position + 1) // compression_ratio):
                continue
            location = resolve(cp, cc, request, int(index), cps, len(nope))
            if location is None:
                continue
            page, slot = location
            record = nope[page, slot].reshape(512)
            values = record[:448].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
            scales = (
                record[448 : 448 + 448 // fp8_scale_block]
                .view(ml_dtypes.float8_e8m0fnu)
                .astype(np.float32)
            )
            values = values * np.repeat(scales, fp8_scale_block)
            r = rope[page, slot // rows_per_group, slot % rows_per_group].astype(np.uint16)
            rotated = ((r[:64] << 8) | r[64:]).view(ml_dtypes.bfloat16).astype(np.float32)
            rows.append(np.concatenate((values, rotated)))
        if not rows:
            continue
        kv = np.asarray(rows, np.float32)
        scores = q[token] @ kv.T * np.float32(scale)
        maximum = np.maximum(scores.max(axis=-1, keepdims=True), sink[:, None])
        weight = np.exp(scores - maximum)
        denominator = weight.sum(axis=-1, keepdims=True) + np.exp(sink[:, None] - maximum)
        output[token] = (weight @ kv) / denominator
    return output
