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


def test_phase_a_prebuild_builds_same_batch_candidate_from_padded_new_seq_lens():
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    req_pool = np.array([10, 11, 12, 13], dtype=np.int32)
    spec_info = EagleDraftInput(
        allocate_lens=np.full((4,), 64, dtype=np.int32),
    )
    spec_info.verify_write_lens = np.array([20, 21, 22, 23], dtype=np.int32)
    model_worker_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        req_pool_indices=req_pool,
        same_batch_chain_req_pool_indices=req_pool.copy(),
        same_batch_chain_out_cache_loc_chunks=[
            np.arange(4, dtype=np.int32),
            np.arange(4, 8, dtype=np.int32),
        ],
        same_batch_chain_verify_write_lens=np.array([24, 25, 26, 27], dtype=np.int32),
        same_batch_chain_allocate_lens=np.full((4,), 64, dtype=np.int32),
        spec_info_padded=spec_info,
        seq_lens=np.array([16, 17, 18, 19], dtype=np.int32),
        seq_lens_sum=70,
        out_cache_loc=np.arange(8, dtype=np.int32),
        bid=9,
    )
    verify_result = SimpleNamespace(
        padded_new_seq_lens_host=np.array([21, 22, 23, 24], dtype=np.int32)
    )

    candidate = worker._prebuild_same_batch_spec_chain_candidate_after_phase_a(
        model_worker_batch,
        verify_result,
    )

    assert candidate is not None
    np.testing.assert_array_equal(candidate.req_pool_indices, req_pool)
    np.testing.assert_array_equal(candidate.seq_lens, verify_result.padded_new_seq_lens_host)
    assert candidate.seq_lens_sum == int(verify_result.padded_new_seq_lens_host.sum())
    assert candidate.skip_fused_verify_padding_for_decode is True


def test_same_batch_chain_peek_prefers_scheduler_reserved_frontier(monkeypatch):
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 4)
    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    req_pool = np.array([1, 2], dtype=np.int32)
    stale_spec_info = EagleDraftInput(
        allocate_lens=np.array([10, 10], dtype=np.int32),
    )
    stale_spec_info.verify_write_lens = np.array([10, 10], dtype=np.int32)
    req_to_token = np.arange(3 * 80, dtype=np.int32).reshape(3, 80)
    worker.worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
    )
    model_worker_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        req_pool_indices=req_pool,
        same_batch_chain_req_pool_indices=req_pool.copy(),
        same_batch_chain_out_cache_loc_chunks=None,
        same_batch_chain_verify_write_lens=np.array([20, 20], dtype=np.int32),
        same_batch_chain_allocate_lens=np.array([64, 64], dtype=np.int32),
        spec_info_padded=stale_spec_info,
        seq_lens=np.array([16, 16], dtype=np.int32),
        seq_lens_sum=32,
        out_cache_loc=np.arange(8, dtype=np.int32),
        bid=11,
    )
    pending = SimpleNamespace(
        padded_next_draft_input=EagleDraftInput(
            new_seq_lens=np.array([21, 22], dtype=np.int32),
        ),
        padded_req_pool_indices=req_pool.copy(),
        padded_new_seq_lens_host=np.array([21, 22], dtype=np.int32),
    )

    candidate = worker._build_same_batch_spec_chain_candidate_batch(
        model_worker_batch,
        pending,
    )

    assert candidate is not None
    np.testing.assert_array_equal(
        candidate.spec_info_padded.allocate_lens,
        np.array([64, 64], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        candidate.spec_info_padded.verify_write_lens,
        np.array([24, 25], dtype=np.int32),
    )


def test_eagle_prepare_for_decode_reserves_extra_chain_slack(monkeypatch):
    from sgl_jax.srt.speculative import eagle_util

    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 4)
    allocated = {}

    def fake_alloc_paged_token_slots_extend(
        tree_cache,
        prefix_lens,
        seq_lens,
        last_loc,
        extend_num_tokens,
        dp_rank=0,
    ):
        allocated["prefix_lens"] = np.asarray(prefix_lens, dtype=np.int32).copy()
        allocated["seq_lens"] = np.asarray(seq_lens, dtype=np.int32).copy()
        allocated["extend_num_tokens"] = int(extend_num_tokens)
        return np.arange(128, 192, dtype=np.int32)

    monkeypatch.setattr(
        eagle_util,
        "alloc_paged_token_slots_extend",
        fake_alloc_paged_token_slots_extend,
    )
    req_to_token = np.zeros((1, 256), dtype=np.int32)
    req_to_token[0, 127] = 127
    schedule_batch = SimpleNamespace(
        batch_size=lambda: 1,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
        tree_cache=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        reqs_info=[
            SimpleNamespace(
                seq_lens=np.array([117], dtype=np.int32),
                req_pool_indices=np.array([0], dtype=np.int32),
            )
        ],
    )
    draft = EagleDraftInput(
        allocate_lens=np.array([128], dtype=np.int32),
    )
    draft.verify_write_lens = np.array([120], dtype=np.int32)

    draft.prepare_for_decode(schedule_batch)

    np.testing.assert_array_equal(draft.allocate_lens, np.array([192], dtype=np.int32))
    np.testing.assert_array_equal(draft.verify_write_lens, np.array([120], dtype=np.int32))
    np.testing.assert_array_equal(allocated["prefix_lens"], np.array([128], dtype=np.int32))
    np.testing.assert_array_equal(allocated["seq_lens"], np.array([192], dtype=np.int32))
    assert allocated["extend_num_tokens"] == 64


def test_same_batch_chain_prewarm_uses_current_layout_without_stashing():
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    calls = []
    req_pool = np.array([1, 2], dtype=np.int32)

    def fake_prepare(candidate):
        calls.append(candidate)
        return SimpleNamespace(target_forward_batch=object())

    worker._prepare_chained_verify_launch_after_phase_a = fake_prepare
    model_worker_batch = SimpleNamespace(
        allow_same_batch_spec_chain=True,
        req_pool_indices=req_pool,
        same_batch_chain_req_pool_indices=req_pool.copy(),
        same_batch_chain_out_cache_loc_chunks=[
            np.arange(4, dtype=np.int32),
            np.arange(4, 8, dtype=np.int32),
        ],
        same_batch_chain_verify_write_lens=np.array([24, 25], dtype=np.int32),
        same_batch_chain_allocate_lens=np.array([64, 64], dtype=np.int32),
        spec_info_padded=EagleDraftInput(allocate_lens=np.array([64, 64], dtype=np.int32)),
        seq_lens=np.array([21, 22], dtype=np.int32),
        seq_lens_sum=43,
        out_cache_loc=np.arange(8, dtype=np.int32),
        bid=12,
    )

    worker._prewarm_same_batch_spec_chain_prepare_cache(model_worker_batch)

    assert len(calls) == 1
    assert getattr(worker, "pending_same_batch_spec_chain_candidate", None) is None
    np.testing.assert_array_equal(calls[0].seq_lens, np.array([21, 22], dtype=np.int32))
    np.testing.assert_array_equal(
        calls[0].spec_info_padded.allocate_lens,
        np.array([64, 64], dtype=np.int32),
    )


def test_prebuilt_candidate_keeps_prepared_verify_launch_payload():
    payload = object()
    candidate = SimpleNamespace(prepared_fused_greedy_verify_launch=payload)

    assert candidate.prepared_fused_greedy_verify_launch is payload


def test_phase_b_relay_patches_only_device_dependent_fields():
    relayed = EagleDraftInput(
        topk_index=np.array([[[1]], [[2]]], dtype=np.int32),
        topk_p=np.ones((2, 1, 1), dtype=np.float32),
        hidden_states=np.ones((2, 4), dtype=np.float32),
        verified_id=np.array([7, 8], dtype=np.int32),
        new_seq_lens=np.array([100, 200], dtype=np.int32),
        allocate_lens=np.array([300, 400], dtype=np.int32),
    )
    relayed.previous_token_list = np.array([[1], [2]], dtype=np.int32)
    candidate_spec = EagleDraftInput(
        new_seq_lens=np.array([10, 20], dtype=np.int32),
        allocate_lens=np.array([30, 40], dtype=np.int32),
    )

    for field in ("topk_index", "topk_p", "verified_id", "hidden_states", "previous_token_list"):
        setattr(candidate_spec, field, getattr(relayed, field))

    np.testing.assert_array_equal(candidate_spec.new_seq_lens, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(candidate_spec.allocate_lens, np.array([30, 40], dtype=np.int32))
    np.testing.assert_array_equal(candidate_spec.verified_id, np.array([7, 8], dtype=np.int32))


def test_swa_radix_cache_finished_req_uses_eagle_page_aligned_kv_len():
    from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache

    inserted = {}
    freed = []

    class FakeAllocator:
        def free(self, indices, dp_rank=0):
            freed.append((np.asarray(indices, dtype=np.int32).copy(), dp_rank))

    class FakeReq:
        req_pool_idx = 3
        dp_rank = 0
        origin_input_ids = [1, 2, 3]
        output_ids = [4, 5, 6]
        extra_key = None
        cache_protected_len = 0
        last_matched_prefix_len = 0
        last_node = None
        swa_uuid_for_lock = None
        swa_evicted_seqlen = 0

        def pop_committed_kv_cache(self):
            return 5

    cache = SWARadixCache.__new__(SWARadixCache)
    cache.disable = False
    cache.page_size = 4
    cache.is_eagle = True
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=np.array(
            [
                np.zeros(16, dtype=np.int32),
                np.zeros(16, dtype=np.int32),
                np.zeros(16, dtype=np.int32),
                np.array([11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ]
        )
    )
    cache.token_to_kv_pool_allocator = FakeAllocator()

    def fake_insert(key, value, prev_prefix_len, swa_evicted_seqlen=0):
        inserted["key_len"] = len(key.token_ids)
        inserted["value"] = np.asarray(value, dtype=np.int32).copy()
        inserted["prev_prefix_len"] = prev_prefix_len
        inserted["swa_evicted_seqlen"] = swa_evicted_seqlen

    cache.insert = fake_insert
    cache.dec_lock_ref = lambda *args, **kwargs: None

    cache.cache_finished_req(FakeReq())

    assert inserted["key_len"] == 5
    np.testing.assert_array_equal(inserted["value"], np.array([11, 12, 13, 14]))
    assert len(freed) == 1
    np.testing.assert_array_equal(freed[0][0], np.array([15], dtype=np.int32))
