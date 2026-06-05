from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.managers.scheduler import Scheduler, SpecVerifyPhaseResult
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput


def test_spec_verify_phase_result_keeps_dp_padded_accept_layout():
    per_dp_bs = 4
    dp_size = 2
    stride = 4
    total_bs = per_dp_bs * dp_size
    accept_lens = np.array([4, 2, 0, 0, 3, 1, 0, 0], dtype=np.int32)
    next_token_ids = np.arange(total_bs * stride, dtype=np.int32)
    draft = EagleDraftInput(
        verified_id=np.arange(total_bs, dtype=np.int32),
        new_seq_lens=np.arange(total_bs, dtype=np.int32) + 100,
        allocate_lens=np.arange(total_bs, dtype=np.int32) + 128,
        hidden_states=np.zeros((total_bs, 8), dtype=np.float32),
    )

    result = SpecVerifyPhaseResult(
        logits_output=None,
        next_token_ids=next_token_ids,
        accept_lens=accept_lens,
        allocate_lens=draft.allocate_lens,
        scheduler_next_draft_input=draft,
        draft_extend_state={"stride": stride},
        bid=7,
        cache_miss_count=0,
    )

    assert result.accept_lens.shape == (total_bs,)
    assert result.next_token_ids.shape == (total_bs * stride,)
    assert result.scheduler_next_draft_input.new_seq_lens.shape == (total_bs,)


def test_split_phase_entrypoints_import():
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
    from sgl_jax.srt.speculative.draft_extend_fused import spec_decode_verify_phase

    assert callable(spec_decode_verify_phase)
    assert hasattr(BaseSpecWorker, "forward_batch_speculative_verify_phase")


def test_split_phase_wrapper_entrypoints_import():
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
    from sgl_jax.srt.speculative.draft_extend_fused import (
        spec_decode,
        spec_decode_draft_extend_phase,
        spec_decode_verify_phase,
    )

    assert callable(spec_decode)
    assert callable(spec_decode_verify_phase)
    assert callable(spec_decode_draft_extend_phase)
    assert hasattr(BaseSpecWorker, "forward_batch_speculative_draft_extend_phase")


def test_publish_spec_verify_phase_updates_lengths_without_overwriting_spec_info():
    scheduler = Scheduler.__new__(Scheduler)
    rank0_spec = object()
    rank1_spec = object()
    batch = SimpleNamespace(
        dp_size=2,
        reqs_info=[
            SimpleNamespace(
                reqs=[object(), object()],
                seq_lens=np.array([10, 20], dtype=np.int32),
                spec_info=rank0_spec,
            ),
            SimpleNamespace(
                reqs=[object()],
                seq_lens=np.array([30], dtype=np.int32),
                spec_info=rank1_spec,
            ),
        ],
    )
    model_worker_batch = SimpleNamespace(
        real_bs_per_dp=[2, 1],
        per_dp_bs_size=2,
    )
    scheduler_next_draft_input = EagleDraftInput(
        verified_id=np.array([11, 22, 33], dtype=np.int32),
        new_seq_lens=np.array([12, 24, 33], dtype=np.int32),
        allocate_lens=np.array([64, 64, 64], dtype=np.int32),
    )
    result = SpecVerifyPhaseResult(
        logits_output=None,
        next_token_ids=np.arange(8, dtype=np.int32),
        accept_lens=np.array([2, 4, 3, 0], dtype=np.int32),
        allocate_lens=scheduler_next_draft_input.allocate_lens,
        scheduler_next_draft_input=scheduler_next_draft_input,
        draft_extend_state=None,
        bid=1,
        cache_miss_count=0,
    )

    scheduler._publish_spec_verify_phase_lengths_to_batch(
        batch,
        model_worker_batch,
        result,
    )

    np.testing.assert_array_equal(batch.reqs_info[0].seq_lens, np.array([12, 24]))
    np.testing.assert_array_equal(batch.reqs_info[1].seq_lens, np.array([33]))
    assert batch.reqs_info[0].spec_info is rank0_spec
    assert batch.reqs_info[1].spec_info is rank1_spec


def test_write_back_spec_draft_state_requires_complete_next_draft_input():
    scheduler = Scheduler.__new__(Scheduler)
    batch = SimpleNamespace(
        reqs_info=[
            SimpleNamespace(spec_info=None),
            SimpleNamespace(spec_info=None),
        ],
    )
    model_worker_batch = SimpleNamespace(real_bs=3, real_bs_per_dp=[2, 1])
    incomplete = EagleDraftInput(
        topk_index=None,
        topk_p=np.zeros((3, 1, 1), dtype=np.float32),
        hidden_states=np.zeros((3, 8), dtype=np.float32),
        verified_id=np.arange(3, dtype=np.int32),
        allocate_lens=np.full((3,), 64, dtype=np.int32),
    )

    try:
        scheduler._write_back_spec_draft_state_to_batch(
            batch,
            model_worker_batch,
            incomplete,
        )
    except AssertionError as exc:
        assert "topk_index" in str(exc)
    else:
        raise AssertionError("expected incomplete draft state to be rejected")


def test_write_back_spec_draft_state_splits_complete_next_draft_input():
    scheduler = Scheduler.__new__(Scheduler)
    batch = SimpleNamespace(
        reqs_info=[
            SimpleNamespace(spec_info=None),
            SimpleNamespace(spec_info=None),
        ],
    )
    model_worker_batch = SimpleNamespace(real_bs=3, real_bs_per_dp=[2, 1])
    complete = EagleDraftInput(
        topk_index=np.arange(3, dtype=np.int32).reshape(3, 1, 1),
        topk_p=np.ones((3, 1, 1), dtype=np.float32),
        hidden_states=np.zeros((3, 8), dtype=np.float32),
        verified_id=np.arange(10, 13, dtype=np.int32),
        allocate_lens=np.full((3,), 64, dtype=np.int32),
        new_seq_lens=np.array([12, 24, 33], dtype=np.int32),
    )

    scheduler._write_back_spec_draft_state_to_batch(
        batch,
        model_worker_batch,
        complete,
    )

    np.testing.assert_array_equal(
        batch.reqs_info[0].spec_info.topk_index,
        np.array([[[0]], [[1]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.reqs_info[0].spec_info.new_seq_lens,
        np.array([12, 24], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.reqs_info[1].spec_info.verified_id,
        np.array([12], dtype=np.int32),
    )


def test_fused_greedy_draft_state_requires_topk_and_verified_id():
    from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

    worker = BaseSpecWorker.__new__(BaseSpecWorker)
    batch_without_topk = SimpleNamespace(
        spec_info_padded=SimpleNamespace(
            verified_id=np.array([1], dtype=np.int32),
            topk_index=None,
        )
    )
    batch_with_state = SimpleNamespace(
        spec_info_padded=SimpleNamespace(
            verified_id=np.array([1], dtype=np.int32),
            topk_index=np.array([[[2]]], dtype=np.int32),
        )
    )

    assert not worker._has_fused_greedy_draft_state(batch_without_topk)
    assert worker._has_fused_greedy_draft_state(batch_with_state)
