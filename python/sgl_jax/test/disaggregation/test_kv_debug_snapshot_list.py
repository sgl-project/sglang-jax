"""build_kv_debug_snapshot handles a per-layer list equivalently to a stack."""
from __future__ import annotations

import numpy as np

from sgl_jax.srt.disaggregation.debug_utils import build_kv_debug_snapshot


def test_list_matches_stacked():
    rng = np.random.default_rng(0)
    layers = [rng.standard_normal((3, 2, 4)).astype(np.float32) for _ in range(5)]
    stacked = np.stack(layers, axis=0)
    snap_list = build_kv_debug_snapshot(layers)
    snap_stack = build_kv_debug_snapshot(stacked)
    assert snap_list.shape == snap_stack.shape == (5, 3, 2, 4)
    assert snap_list.global_digest == snap_stack.global_digest
    assert snap_list.page_digests == snap_stack.page_digests
