from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.speculative.draft_extend_fused import (
    _build_decode_loop_cache_loc,
    _build_decode_loop_out_cache_loc,
)


def test_decode_loop_cache_loc_is_dp_segmented():
    req_to_token = np.zeros((8, 16), dtype=np.int32)
    req_to_token[2] = 200 + np.arange(16, dtype=np.int32)
    req_to_token[5] = 500 + np.arange(16, dtype=np.int32)
    req_to_token[6] = 600 + np.arange(16, dtype=np.int32)
    pool = SimpleNamespace(req_to_token=req_to_token)

    cache_loc = _build_decode_loop_cache_loc(
        pool,
        np.asarray([2, -1, 5, 6], dtype=np.int32),
        np.asarray([3, 0, 5, 2], dtype=np.int32),
        dp_size=2,
        per_dp_bs=2,
        page_size=4,
    )

    assert cache_loc.shape == (24,)
    np.testing.assert_array_equal(cache_loc[:3], [200, 201, 202])
    assert np.all(cache_loc[3:12] == 0)
    np.testing.assert_array_equal(cache_loc[12:17], [500, 501, 502, 503, 504])
    assert np.all(cache_loc[17:20] == 0)
    np.testing.assert_array_equal(cache_loc[20:22], [600, 601])
    assert np.all(cache_loc[22:24] == 0)


def test_decode_loop_cache_loc_has_page_for_each_padded_slot():
    req_to_token = np.zeros((8, 16), dtype=np.int32)
    req_to_token[2] = 200 + np.arange(16, dtype=np.int32)
    pool = SimpleNamespace(req_to_token=req_to_token)

    page_size = 4
    per_dp_bs = 8
    cache_loc = _build_decode_loop_cache_loc(
        pool,
        np.asarray([2, -1, -1, -1, -1, -1, -1, -1], dtype=np.int32),
        np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
        dp_size=1,
        per_dp_bs=per_dp_bs,
        page_size=page_size,
    )

    assert cache_loc.shape == (per_dp_bs * page_size,)
    assert cache_loc[0] == 200
    assert np.all(cache_loc[1:] == 0)


def test_decode_loop_out_cache_loc_uses_current_positions():
    req_to_token = np.zeros((8, 16), dtype=np.int32)
    req_to_token[2] = 200 + np.arange(16, dtype=np.int32)
    req_to_token[5] = 500 + np.arange(16, dtype=np.int32)
    req_to_token[6] = 600 + np.arange(16, dtype=np.int32)
    pool = SimpleNamespace(req_to_token=req_to_token)

    out_cache_loc = _build_decode_loop_out_cache_loc(
        pool,
        np.asarray([2, -1, 5, 6], dtype=np.int32),
        np.asarray([3, 0, 5, 2], dtype=np.int32),
        np.asarray([True, False, True, True]),
    )

    np.testing.assert_array_equal(out_cache_loc, [203, -1, 505, 602])
