"""CPU unit tests for DFlashWorker state-advance helpers.

The full worker construction needs real target/draft ModelWorkers (TPU); these
tests exercise pure array helpers by instantiating the worker via ``__new__``
and injecting the minimal attributes each helper reads.
"""

from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.layers.attention.flashattention_backend import (
    _pad_page_indices,
    _repack_row_padded_page_indices,
)
from sgl_jax.srt.speculative.dflash_info import DFlashDraftInput
from sgl_jax.srt.speculative.dflash_worker import DFlashWorker


def _bare_worker(**attrs):
    w = object.__new__(DFlashWorker)
    for k, v in attrs.items():
        object.__setattr__(w, k, v)
    return w


def test_committed_positions_prefill_and_decode():
    w = _bare_worker()

    prefill_di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([3, 2], dtype=np.int32),
        draft_seq_lens=np.array([5, 0], dtype=np.int32),
    )
    pos = w._committed_positions(prefill_di, is_prefill=True)
    np.testing.assert_array_equal(pos, np.array([5, 6, 7, 0, 1], dtype=np.int32))

    decode_di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([2, 1], dtype=np.int32),
        draft_seq_lens=np.array([10, 7], dtype=np.int32),
    )
    pos = w._committed_positions(decode_di, is_prefill=False)
    np.testing.assert_array_equal(pos, np.array([8, 9, 6], dtype=np.int32))


def test_committed_cache_loc_reads_req_to_token():
    # req_to_token[req_pool_index, position] -> global cache slot.
    req_to_token = np.arange(100, dtype=np.int32).reshape(5, 20)
    draft_model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
    )
    w = _bare_worker(draft_model_runner=draft_model_runner)

    mwb = SimpleNamespace(req_pool_indices=np.array([1, 3], dtype=np.int32))
    di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([2, 1], dtype=np.int32),
        draft_seq_lens=np.array([4, 2], dtype=np.int32),  # decode: start = len - ctx
    )
    locs = w._committed_cache_loc(mwb, di, is_prefill=False)
    # req1: positions [2,3] -> req_to_token[1, 2:4] = [22, 23]
    # req3: position  [1]   -> req_to_token[3, 1:2] = [61]
    np.testing.assert_array_equal(locs, np.array([22, 23, 61], dtype=np.int32))


def test_committed_decode_row_padded_cache_loc_positions():
    req_to_token = np.arange(100, dtype=np.int32).reshape(5, 20)
    draft_model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
    )
    w = _bare_worker(draft_model_runner=draft_model_runner, block_size=4)

    mwb = SimpleNamespace(req_pool_indices=np.array([1, 3], dtype=np.int32))
    di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([2, 1], dtype=np.int32),
        draft_seq_lens=np.array([10, 7], dtype=np.int32),  # decode starts: [8, 6]
    )

    locs, positions = w._committed_decode_row_padded_cache_loc_positions(mwb, di)

    np.testing.assert_array_equal(
        locs,
        np.array([28, 29, -1, -1, 66, -1, -1, -1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        positions,
        np.array([8, 9, 9, 9, 6, 6, 6, 6], dtype=np.int32),
    )


def test_committed_decode_row_padded_cache_loc_ignores_padded_slots():
    req_to_token = np.arange(100, dtype=np.int32).reshape(5, 20)
    draft_model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
    )
    w = _bare_worker(draft_model_runner=draft_model_runner, block_size=4)

    mwb = SimpleNamespace(
        req_pool_indices=np.array([1, 2, 3, 0], dtype=np.int32),
        real_bs=3,
        real_bs_per_dp=[3],
        per_dp_bs_size=4,
    )
    di = DFlashDraftInput(
        verified_id=np.array([0, 0, 0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([1, 1, 1, 1], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7, 0], dtype=np.int32),
    )

    locs, positions = w._committed_decode_row_padded_cache_loc_positions(mwb, di)

    np.testing.assert_array_equal(
        locs,
        np.array([24, -1, -1, -1, 45, -1, -1, -1, 66, -1, -1, -1, -1, -1, -1, -1]),
    )
    np.testing.assert_array_equal(positions[-4:], np.zeros((4,), dtype=np.int32))


def test_trim_dflash_draft_input_drops_stale_tail_state():
    w = _bare_worker()
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 30, 40], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7, 8], dtype=np.int32),
    )

    w._trim_dflash_draft_input_to_decode_batch(di, bs=3)

    np.testing.assert_array_equal(di.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))


def test_page_indices_capacity_is_bounded_by_request_and_pool():
    w = _bare_worker(
        _page_indices_pool_capacity=8192,
        _page_indices_per_seq_capacity=1024,
    )

    assert w._page_indices_capacity(1) == 1024
    assert w._page_indices_capacity(4) == 4096
    assert w._page_indices_capacity(16) == 8192


def test_prefill_precompile_variants_use_runtime_extend_buckets():
    manager = SimpleNamespace(
        max_padded_batch_size=128,
        token_buckets=[64, 128, 256, 1024, 2048],
    )

    assert DFlashWorker._prefill_precompile_variants(manager) == [
        (128, 128),
        (128, 256),
        (128, 1024),
        (128, 2048),
    ]


def test_pack_cache_loc_rows_uses_bucket_stable_row_width():
    w = _bare_worker(page_size=1)
    rows = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6, 7, 8], dtype=np.int32),
    ]

    packed = w._pack_kv_cache(rows, [len(r) for r in rows])

    assert packed.shape == (32,)
    np.testing.assert_array_equal(packed[:16], np.array([1, 2, 3] + [3] * 13, dtype=np.int32))
    np.testing.assert_array_equal(
        packed[16:32], np.array([4, 5, 6, 7, 8] + [8] * 11, dtype=np.int32)
    )


def test_repack_row_padded_page_indices_removes_dflash_row_padding():
    row0 = np.array(list(range(100, 123)) + [122] * 9, dtype=np.int32)
    row1 = np.array(list(range(200, 223)) + [222] * 9, dtype=np.int32)
    cache_loc = np.concatenate([row0, row1])

    page_indices = _repack_row_padded_page_indices(
        cache_loc,
        np.array([23, 23], dtype=np.int32),
        page_size=1,
        dp_size=1,
        per_dp_bs=2,
    )

    assert page_indices.shape == (46,)
    np.testing.assert_array_equal(page_indices[:23], np.arange(100, 123, dtype=np.int32))
    np.testing.assert_array_equal(page_indices[23:], np.arange(200, 223, dtype=np.int32))


def test_pad_page_indices_uses_fixed_dflash_capacity():
    page_indices = np.array([3, 5, 7], dtype=np.int32)

    padded = _pad_page_indices(page_indices, max_num_seqs=2, fixed_capacity=8)

    np.testing.assert_array_equal(
        padded,
        np.array([3, 5, 7, 0, 0, 0, 0, 0], dtype=np.int32),
    )


def test_pad_page_indices_rejects_fixed_capacity_overflow():
    with np.testing.assert_raises_regex(ValueError, "exceed fixed capacity"):
        _pad_page_indices(
            np.arange(9, dtype=np.int32),
            max_num_seqs=2,
            fixed_capacity=8,
        )
