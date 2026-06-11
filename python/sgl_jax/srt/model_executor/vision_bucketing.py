"""V-2 vision patch bucketing (host side, pure NumPy — no JAX).

Pad each image's LLM-grid ``(h // merge, w // merge)`` up to a multiple of the bucket edge so the
in-model vision encode jit sees a canonical (bounded) set of grid geometries instead of one per
resolution. Lives in its own JAX-free module so the merge-unit padding logic — the bug-prone part
— is unit-testable on CPU without importing the model runner (and its JAX dependency).

Design: §5.3 (V-2). Consumed by ``ModelRunner.encode_mm_reqs``.
"""

from __future__ import annotations

import numpy as np


def bucket_pad_images(
    pixel_values: np.ndarray,
    grids: list[tuple[int, int, int]] | tuple,
    merge: int,
    bucket: int,
) -> tuple[np.ndarray, tuple, np.ndarray]:
    """Pad each image's grid to a canonical bucket at merge-unit granularity.

    Args:
      pixel_values: ``[sum_i t_i*h_i*w_i, dim]`` patches, items concatenated in order. Patch order
        within an item is ``(t, llm_h, llm_w, merge*merge)`` row-major (the processor's layout that
        the ViT's ``reshape(seq // mu, mu, -1)`` assumes).
      grids: per-item ``(t, h, w)`` in PATCH units.
      merge: spatial_merge_size ``m`` (so the LLM-grid is ``(h//m, w//m)`` and a merge-unit is
        ``m*m`` patches).
      bucket: bucket edge ``S`` in LLM-grid units; each LLM dim is rounded UP to a multiple of S.

    Returns ``(padded_pixels, padded_grids, real_llm_dims)``:
      - padded_pixels: ``[sum_i t_i*H_i*W_i, dim]`` with bucket padding zero-filled at the
        bottom/right of each item's LLM-grid (so real units keep canonical row-major order).
      - padded_grids: per-item ``(t, H, W)`` PATCH-unit grids where ``H,W`` are bucket multiples.
        An item already on a bucket multiple is passed through unchanged.
      - real_llm_dims: ``int32[num_items, 2]`` of each item's true ``(llm_h, llm_w)`` — fed to the
        ViT as a *traced* array so the jit keys only on the padded grid, never the real size.
    """
    px = np.asarray(pixel_values)
    dim = px.shape[-1]
    out_px: list[np.ndarray] = []
    padded_grids: list[tuple[int, int, int]] = []
    real_dims: list[tuple[int, int]] = []
    cur = 0
    for t, h, w in grids:
        t, h, w = int(t), int(h), int(w)
        size = t * h * w
        unit = px[cur : cur + size]
        cur += size
        llm_h, llm_w = h // merge, w // merge
        pad_llm_h = ((llm_h + bucket - 1) // bucket) * bucket
        pad_llm_w = ((llm_w + bucket - 1) // bucket) * bucket
        real_dims.append((llm_h, llm_w))
        if pad_llm_h == llm_h and pad_llm_w == llm_w:
            out_px.append(unit)
            padded_grids.append((t, h, w))
            continue
        mu = merge * merge
        u = unit.reshape(t, llm_h, llm_w, mu, dim)
        padded = np.zeros((t, pad_llm_h, pad_llm_w, mu, dim), dtype=u.dtype)
        padded[:, :llm_h, :llm_w, :, :] = u
        out_px.append(padded.reshape(t * pad_llm_h * pad_llm_w * mu, dim))
        padded_grids.append((t, pad_llm_h * merge, pad_llm_w * merge))
    return (
        np.concatenate(out_px, axis=0),
        tuple(padded_grids),
        np.asarray(real_dims, dtype=np.int32),
    )
