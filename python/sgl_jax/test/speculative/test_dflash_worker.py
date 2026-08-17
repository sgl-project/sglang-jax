from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np

from sgl_jax.srt.layers.attention.flashattention_metadata import (
    PagedKVLayout,
    build_target_verify_metadata,
    pad_page_indices,
)
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.dflash_info import DFlashDraftInput, _mask_draft_kv_writes
from sgl_jax.srt.speculative.dflash_worker import DFlashWorker
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _bare_worker(**attrs):
    w = object.__new__(DFlashWorker)
    for k, v in attrs.items():
        object.__setattr__(w, k, v)
    return w


def test_no_overlap_dflash_uses_explicit_fused_orchestration():
    worker = object.__new__(BaseSpecWorker)
    worker.server_args = SimpleNamespace(disable_overlap_schedule=True)
    worker.speculative_algorithm = SpeculativeAlgorithm.DFLASH
    worker._can_use_fused_eagle_verify = False
    worker._get_cur_allocate_lens = Mock(return_value=np.array([16], dtype=np.int32))
    worker._draft_worker = Mock()
    batch_output = object()
    worker._draft_worker.verify.return_value = batch_output
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        launch_done=None,
    )

    result = worker.forward_batch_speculative_generation(batch, launch_done=Mock())

    assert result is batch_output
    worker._draft_worker.draft.assert_called_once_with(batch)
    verify_batch, verify_allocate_lens = worker._draft_worker.verify.call_args.args
    assert verify_batch is batch
    np.testing.assert_array_equal(verify_allocate_lens, np.array([16], dtype=np.int32))
    worker._draft_worker.draft_extend_for_decode.assert_called_once_with(
        batch,
        batch_output,
    )


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


def test_draft_extend_masks_unaccepted_and_padded_rows():
    cache_loc = np.arange(12, dtype=np.int32)
    masked = _mask_draft_kv_writes(
        jax.numpy.asarray(cache_loc),
        jax.numpy.asarray([2, 4, 3], dtype=jax.numpy.int32),
        jax.numpy.asarray([True, False, True]),
    )

    np.testing.assert_array_equal(
        np.asarray(masked),
        np.array([0, 1, -1, -1, -1, -1, -1, -1, 8, 9, 10, -1], dtype=np.int32),
    )


def test_verify_metadata_uses_explicit_active_slots():
    metadata = build_target_verify_metadata(
        PagedKVLayout(page_indices=jax.numpy.arange(32, dtype=jax.numpy.int32)),
        prefix_lens=jax.numpy.array([4, 0, 0, 0], dtype=jax.numpy.int32),
        allocated_lens=jax.numpy.array([8, 8, 8, 8], dtype=jax.numpy.int32),
        active_mask=jax.numpy.array([True, True, False, False]),
        draft_width=4,
        page_size=1,
        dp_size=1,
    )

    np.testing.assert_array_equal(np.asarray(metadata.cu_q_lens), np.array([0, 4, 8, 8, 8]))
    np.testing.assert_array_equal(np.asarray(metadata.cu_kv_lens), np.array([0, 8, 12, 12, 12]))
    np.testing.assert_array_equal(np.asarray(metadata.seq_lens), np.array([8, 4, 0, 0]))
    np.testing.assert_array_equal(np.asarray(metadata.distribution), np.array([0, 2, 2]))


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


def test_unpad_draft_state_removes_dp_padding():
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 0, 30, 0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([1, 2, 0, 3, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 0, 7, 0, 0], dtype=np.int32),
    )
    DFlashWorker._unpad_draft_state(
        di,
        np.array([0, 1, 3], dtype=np.int32),
    )

    np.testing.assert_array_equal(di.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))


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


def test_trim_draft_state_drops_stale_tail():
    w = _bare_worker()
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 30, 40], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7, 8], dtype=np.int32),
    )

    w._trim_draft_state_to_bs(di, bs=3)

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

    padded = pad_page_indices(page_indices, max_num_seqs=2, fixed_capacity=8)

    np.testing.assert_array_equal(
        padded,
        np.array([3, 5, 7, 0, 0, 0, 0, 0], dtype=np.int32),
    )


def test_pad_page_indices_rejects_fixed_capacity_overflow():
    with np.testing.assert_raises_regex(ValueError, "exceed fixed capacity"):
        pad_page_indices(
            np.arange(9, dtype=np.int32),
            max_num_seqs=2,
            fixed_capacity=8,
        )
