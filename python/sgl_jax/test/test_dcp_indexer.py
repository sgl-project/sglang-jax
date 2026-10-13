"""Slice 4 I1: global indexer top-k over striped local scores."""

from __future__ import annotations

import numpy as np

from sgl_jax.srt.layers.dcp.indexer import global_topk_from_shards
from sgl_jax.srt.layers.dcp.layout import owner, virtual_index


def test_virtual_index_inverts_physical():
    dcp = 16
    for v in range(64):
        r = int(owner(v, dcp))
        p = v // dcp
        assert int(virtual_index(p, dcp, r)) == v


def test_dcp1_topk_matches_local_order():
    scores = np.array([[[0.1, 0.9, 0.2, 0.35]]], dtype=np.float32)
    global_ids, owned = global_topk_from_shards(scores, k=2)
    np.testing.assert_array_equal(global_ids, [[1, 3]])
    np.testing.assert_array_equal(owned[0], [[1, 3]])


def test_i1_planted_needle_page_on_owner_in_global_topk():
    # N=2, 4 physical slots/rank. Needle at virtual 5 → rank 1, physical 2.
    dcp, t, local_kv, k = 2, 1, 4, 3
    scores = np.full((dcp, t, local_kv), -np.inf, dtype=np.float32)
    scores[0, 0, 0] = 1.0  # virtual 0
    scores[0, 0, 1] = 2.0  # virtual 2
    scores[1, 0, 2] = 100.0  # virtual 5 — planted needle
    scores[1, 0, 0] = 1.5  # virtual 1
    global_ids, owned = global_topk_from_shards(scores, k=k)
    assert 5 in global_ids[0]
    assert owner(5, dcp) == 1
    assert 5 in owned[1][0]
    assert 5 not in owned[0][0]
    # Non-owners of the needle are -1 on rank 0.
    assert owned[0][0][list(global_ids[0]).index(5)] == -1
