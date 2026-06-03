from __future__ import annotations

import numpy as np
import pytest

from sgl_jax.srt.disaggregation.debug_utils import (
    build_kv_debug_snapshot,
    count_kv_debug_mismatches,
    find_first_kv_debug_mismatch,
    kv_debug_enabled,
)


def test_kv_debug_enabled_honors_flag_and_req_filter(monkeypatch):
    monkeypatch.delenv("SGL_JAX_PD_DEBUG_KV", raising=False)
    monkeypatch.delenv("SGL_JAX_PD_DEBUG_REQ_ID", raising=False)
    assert not kv_debug_enabled("req-1")

    monkeypatch.setenv("SGL_JAX_PD_DEBUG_KV", "1")
    assert kv_debug_enabled("req-1")

    monkeypatch.setenv("SGL_JAX_PD_DEBUG_REQ_ID", "bucket16")
    assert kv_debug_enabled("probe-bucket16-req")
    assert not kv_debug_enabled("probe-bucket8-req")


def test_build_kv_debug_snapshot_captures_global_and_page_digests():
    kv = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)

    snapshot = build_kv_debug_snapshot(kv)

    assert snapshot.shape == (2, 3, 4)
    assert snapshot.dtype == "int16"
    assert len(snapshot.global_digest) == 16
    assert len(snapshot.page_digests) == 2
    assert len(snapshot.page_digests[0]) == 3
    assert snapshot.page_digests[0][0] != snapshot.page_digests[0][1]
    assert snapshot.sample_page_digests(max_layers=1, max_pages=2) == (
        (snapshot.page_digests[0][0], snapshot.page_digests[0][1]),
    )


def test_compare_kv_debug_snapshots_finds_first_mismatch():
    base = np.arange(2 * 2 * 3, dtype=np.int16).reshape(2, 2, 3)
    modified = base.copy()
    modified[1, 0, 2] += 7

    left = build_kv_debug_snapshot(base)
    right = build_kv_debug_snapshot(modified)

    assert count_kv_debug_mismatches(left, right) == 1
    assert find_first_kv_debug_mismatch(left, right) == (1, 0)


def test_build_kv_debug_snapshot_rejects_inputs_without_layer_and_page_axes():
    with pytest.raises(ValueError, match="at least 2 dims"):
        build_kv_debug_snapshot(np.arange(4, dtype=np.int16))
