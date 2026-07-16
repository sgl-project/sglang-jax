"""CPU unit tests for DFlashWorker state-advance helpers.

The full worker construction needs real target/draft ModelWorkers (TPU); these
tests exercise pure array helpers by instantiating the worker via ``__new__``
and injecting the minimal attributes each helper reads.
"""

from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.layers.attention.flashattention_backend import _pad_page_indices
from sgl_jax.srt.speculative.dflash_info import DFlashDraftInput
from sgl_jax.srt.speculative.dflash_worker import DFlashWorker


def _bare_worker(**attrs):
    w = object.__new__(DFlashWorker)
    for k, v in attrs.items():
        object.__setattr__(w, k, v)
    return w


def test_prefill_draft_extend_metadata_preserves_dp_rank_sections():
    # DP=2, four token rows per rank. Rank-local padding must stay between
    # rank 0's real rows and rank 1's real rows.
    mwb = SimpleNamespace(
        positions=np.array([5, 6, 0, 0, 9, 0, 0, 0], dtype=np.int32),
        out_cache_loc=np.array([20, 21, -1, -1, 40, -1, -1, -1], dtype=np.int32),
    )
    target_hidden = np.zeros((8, 16), dtype=np.float32)

    positions, cache_loc = DFlashWorker._prefill_draft_extend_metadata(mwb, target_hidden)

    np.testing.assert_array_equal(positions, mwb.positions)
    np.testing.assert_array_equal(cache_loc, mwb.out_cache_loc)


def test_prefill_draft_extend_metadata_rejects_bucket_mismatch():
    mwb = SimpleNamespace(
        positions=np.array([0, 1, 2], dtype=np.int32),
        out_cache_loc=np.array([10, 11, 12], dtype=np.int32),
    )
    target_hidden = np.zeros((2, 16), dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "must match the target hidden bucket"):
        DFlashWorker._prefill_draft_extend_metadata(mwb, target_hidden)


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


def test_committed_decode_row_padded_cache_loc_handles_uneven_dp_ranks():
    req_to_token = np.arange(160, dtype=np.int32).reshape(8, 20)
    draft_model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
    )
    w = _bare_worker(draft_model_runner=draft_model_runner, block_size=2)
    mwb = SimpleNamespace(
        req_pool_indices=np.array([1, 2, 0, 4, 0, 0], dtype=np.int32),
        real_bs=3,
        real_bs_per_dp=[2, 1],
        per_dp_bs_size=3,
    )
    di = DFlashDraftInput(
        verified_id=np.zeros(6, dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.ones(6, dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 0, 8, 0, 0], dtype=np.int32),
    )

    locs, _ = w._committed_decode_row_padded_cache_loc_positions(mwb, di)

    np.testing.assert_array_equal(
        locs.reshape(6, 2),
        np.array(
            [
                [24, -1],
                [45, -1],
                [-1, -1],
                [87, -1],
                [-1, -1],
                [-1, -1],
            ],
            dtype=np.int32,
        ),
    )


def test_build_page_indices_preserves_dp_rank_sections():
    req_to_token = np.arange(200, dtype=np.int32).reshape(10, 20)
    w = _bare_worker(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        block_size=2,
        page_size=1,
        _page_indices_pool_capacity=16,
        _page_indices_per_seq_capacity=4,
    )
    mwb = SimpleNamespace(
        req_pool_indices=np.array([1, 2, 3, 4], dtype=np.int32),
        logits_indices_selector=np.arange(4, dtype=np.int32),
        dp_size=2,
        per_dp_bs_size=2,
    )

    page_indices = w._build_dflash_page_indices(
        mwb,
        np.array([1, 1, 1, 1], dtype=np.int32),
        bs=4,
    )

    np.testing.assert_array_equal(
        page_indices.reshape(2, 8),
        np.array(
            [
                [20, 21, 22, 40, 41, 42, 0, 0],
                [60, 61, 62, 80, 81, 82, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_build_page_indices_handles_uneven_dp_ranks():
    req_to_token = np.arange(160, dtype=np.int32).reshape(8, 20)
    w = _bare_worker(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        block_size=2,
        page_size=1,
        _page_indices_pool_capacity=16,
        _page_indices_per_seq_capacity=4,
    )
    mwb = SimpleNamespace(
        req_pool_indices=np.array([1, 2, 0, 4, 0, 0], dtype=np.int32),
        logits_indices_selector=np.array([0, 1, 3], dtype=np.int32),
        dp_size=2,
        per_dp_bs_size=3,
    )

    page_indices = w._build_dflash_page_indices(
        mwb,
        np.array([2, 1, 0, 3, 0, 0], dtype=np.int32),
        bs=6,
    )

    np.testing.assert_array_equal(
        page_indices.reshape(2, 8),
        np.array(
            [
                [20, 21, 22, 23, 40, 41, 42, 0],
                [80, 81, 82, 83, 84, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_compact_dflash_state_removes_dp_padding_but_keeps_new_seq_lens():
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 0, 30, 0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([1, 2, 0, 3, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 0, 7, 0, 0], dtype=np.int32),
    )
    di.new_seq_lens = np.array([6, 8, 0, 10, 0, 0], dtype=np.int32)

    DFlashWorker._compact_dflash_state_to_real_slots(
        di,
        np.array([0, 1, 3], dtype=np.int32),
    )

    np.testing.assert_array_equal(di.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))
    np.testing.assert_array_equal(di.new_seq_lens, np.array([6, 8, 0, 10, 0, 0]))


def test_verify_write_cache_loc_selects_valid_half_per_dp_rank():
    w = _bare_worker(block_size=2)
    batch = SimpleNamespace(
        dp_size=2,
        per_dp_bs_size=2,
        out_cache_loc=np.array(
            [1, 2, 3, 4, -1, -1, -1, -1, 5, 6, 7, 8, -1, -1, -1, -1],
            dtype=np.int32,
        ),
    )

    np.testing.assert_array_equal(
        w._verify_write_cache_loc(batch),
        np.arange(1, 9, dtype=np.int32),
    )


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


def test_build_page_indices_reads_noncontiguous_physical_pages():
    req_to_token = np.array(
        [
            np.arange(8),
            [20, 21, 40, 41, 60, 61, 80, 81],
            [100, 101, 120, 121, 140, 141, 160, 161],
        ],
        dtype=np.int32,
    )
    w = _bare_worker(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        block_size=2,
        page_size=2,
        _page_indices_pool_capacity=8,
        _page_indices_per_seq_capacity=4,
    )
    mwb = SimpleNamespace(
        req_pool_indices=np.array([1, 2], dtype=np.int32),
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        dp_size=1,
        per_dp_bs_size=2,
    )

    page_indices = w._build_dflash_page_indices(
        mwb,
        np.array([3, 2], dtype=np.int32),
        bs=2,
    )

    np.testing.assert_array_equal(
        page_indices,
        np.array([10, 20, 30, 50, 60, 0, 0, 0], dtype=np.int32),
    )


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
