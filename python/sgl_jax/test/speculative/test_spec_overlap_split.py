from collections import namedtuple
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.scheduler import Scheduler, SpecVerifyPhaseResult
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
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


def test_deferred_verify_uses_non_donated_target_memory_pools(monkeypatch):
    from sgl_jax.srt.speculative import draft_extend_fused

    built = []

    def fake_build(*, donate_target_memory_pools=True):
        built.append(donate_target_memory_pools)
        return f"jit_donate_{donate_target_memory_pools}"

    monkeypatch.setattr(draft_extend_fused, "_build_fused_greedy_verify_jit", fake_build)
    draft_worker = SimpleNamespace()

    deferred_jit = draft_extend_fused._get_fused_greedy_verify_jit_fn(
        draft_worker,
        defer_target_pool_updates=True,
    )
    normal_jit = draft_extend_fused._get_fused_greedy_verify_jit_fn(
        draft_worker,
        defer_target_pool_updates=False,
    )
    deferred_jit_again = draft_extend_fused._get_fused_greedy_verify_jit_fn(
        draft_worker,
        defer_target_pool_updates=True,
    )

    assert deferred_jit == "jit_donate_False"
    assert normal_jit == "jit_donate_True"
    assert deferred_jit_again == deferred_jit
    assert built == [False, True]


def test_flashattention_metadata_clone_uses_shallow_fast_path():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        _clone_attn_backend_with_metadata,
    )

    FlashAttention = type("FlashAttention", (), {})
    original = FlashAttention()
    original.num_heads = 8
    original.forward_metadata = object()
    new_metadata = object()

    cloned = _clone_attn_backend_with_metadata(original, new_metadata)

    assert cloned is not original
    assert cloned.num_heads == 8
    assert cloned.forward_metadata is new_metadata
    assert original.forward_metadata is not new_metadata


def test_apply_pending_exact_padded_layout_does_not_overwrite_frontier_fields():
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    req_pool = np.array([11, 12], dtype=np.int32)
    model_worker_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=req_pool.copy(),
        spec_info_padded=EagleDraftInput(
            topk_index=np.full((2, 1, 1), -1, dtype=np.int32),
            topk_p=np.zeros((2, 1, 1), dtype=np.float32),
            hidden_states=np.zeros((2, 4), dtype=np.float32),
            verified_id=np.zeros((2,), dtype=np.int32),
            new_seq_lens=np.array([100, 200], dtype=np.int32),
            allocate_lens=np.array([128, 256], dtype=np.int32),
            verify_write_lens=np.array([103, 203], dtype=np.int32),
        ),
    )
    pending = SimpleNamespace(
        next_draft_input=EagleDraftInput(),
        req_pool_indices=req_pool.copy(),
        padded_req_pool_indices=req_pool.copy(),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.arange(2, dtype=np.int32).reshape(2, 1, 1),
            topk_p=np.ones((2, 1, 1), dtype=np.float32),
            hidden_states=np.ones((2, 4), dtype=np.float32),
            verified_id=np.array([7, 8], dtype=np.int32),
            new_seq_lens=np.array([104, 205], dtype=np.int32),
            allocate_lens=np.array([192, 320], dtype=np.int32),
            verify_write_lens=np.array([107, 208], dtype=np.int32),
        ),
    )
    worker.pending_spec_draft_extend_result = pending

    worker._apply_pending_spec_draft_extend_to_batch(model_worker_batch)

    spec_info = model_worker_batch.spec_info_padded
    np.testing.assert_array_equal(spec_info.verified_id, np.array([7, 8], dtype=np.int32))
    np.testing.assert_array_equal(spec_info.new_seq_lens, np.array([100, 200], dtype=np.int32))
    np.testing.assert_array_equal(spec_info.allocate_lens, np.array([128, 256], dtype=np.int32))
    np.testing.assert_array_equal(spec_info.verify_write_lens, np.array([103, 203], dtype=np.int32))
    assert worker.pending_spec_draft_extend_result is None


def test_materialize_verify_phase_rewinds_draft_extended_seq_lens():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        FusedGreedyDraftExtendState,
        FusedGreedyVerifyPhaseAsync,
        spec_decode_materialize_verify_phase,
    )

    speculative_num_draft_tokens = 4
    seq_lens_after_draft_extend = np.array([104, 204, 0, 0], dtype=np.int32)
    accept_lens = np.array([3, 1, 0, 0], dtype=np.int32)
    predict = np.array(
        [
            10,
            11,
            12,
            13,
            20,
            21,
            22,
            23,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.int32,
    )
    draft_output = SimpleNamespace(
        next_draft_input=EagleDraftInput(
            verified_id=np.zeros(
                (len(seq_lens_after_draft_extend) * speculative_num_draft_tokens,), dtype=np.int32
            )
        )
    )
    async_result = FusedGreedyVerifyPhaseAsync(
        logits_output=None,
        next_token_ids_prefetch=predict,
        accept_lens_prefetch=accept_lens,
        accept_lens_device=jnp.asarray(accept_lens),
        allocate_lens=np.array([128, 256, 0, 0], dtype=np.int32),
        scheduler_next_draft_input_allocate_lens=np.array([128, 256], dtype=np.int32),
        selector=np.array([0, 1], dtype=np.int32),
        seq_lens_host=seq_lens_after_draft_extend,
        draft_extend_state=FusedGreedyDraftExtendState(
            batch_output=draft_output,
            positions=jnp.zeros((len(seq_lens_after_draft_extend),), dtype=jnp.int32),
            predispatched=None,
        ),
        bid=5,
        cache_miss_count=0,
    )

    result = spec_decode_materialize_verify_phase(async_result)

    np.testing.assert_array_equal(result.scheduler_next_draft_input.verified_id, np.array([12, 20]))
    np.testing.assert_array_equal(
        result.scheduler_next_draft_input.new_seq_lens, np.array([104, 202])
    )
    np.testing.assert_array_equal(result.padded_new_seq_lens_host[:2], np.array([104, 202]))


def test_greedy_chain_verify_masks_padding_rows():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        _greedy_sample_and_prepare_draft_inputs_chain_from_predict,
    )

    out = _greedy_sample_and_prepare_draft_inputs_chain_from_predict(
        target_hidden=jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
        positions=jnp.arange(4, dtype=jnp.int32),
        seq_lens=jnp.array([10, 0], dtype=jnp.int32),
        draft_tokens=jnp.array([100, 101, 102, 103, 200, 201, 202, 203], dtype=jnp.int32),
        target_predict=jnp.array([100, 101, 999, 999, 200, 201, 202, 203], dtype=jnp.int32),
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )

    np.testing.assert_array_equal(np.asarray(out.accept_lens), np.array([1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(out.new_seq_lens), np.array([11, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(out.sel_pos), np.array([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(out.select_index), np.array([0, 4], dtype=np.int32))


def test_materialize_draft_extend_all_padding_accept_zero_noop():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        FusedDraftExtendDispatch,
        _materialize_draft_extend_for_decode_fused,
    )

    batch_output = SimpleNamespace(
        next_draft_input=EagleDraftInput(verified_id=np.empty((0,), dtype=np.int32)),
        accept_lens=np.array([0, 0], dtype=np.int32),
        allocate_lens=np.array([0, 0], dtype=np.int32),
    )
    dispatch = FusedDraftExtendDispatch(
        batch_output=batch_output,
        selector=np.array([0, 1], dtype=np.int32),
        selected_layer0_hidden=jnp.zeros((2, 4), dtype=jnp.float32),
        topk_index_stacked=jnp.zeros((2, 1, 1), dtype=jnp.int32),
        previous_token_list=jnp.zeros((2, 1), dtype=jnp.int32),
        selected_verified_id=jnp.zeros((2,), dtype=jnp.int32),
        verified_id_arr=jnp.zeros((8,), dtype=jnp.int32),
        accept_lens_device=jnp.array([0, 0], dtype=jnp.int32),
        materialize_hidden=False,
        materialize_topk=False,
        accept_lens_host=np.array([0, 0], dtype=np.int32),
        verified_id_host=np.empty((0,), dtype=np.int32),
    )
    draft_worker = SimpleNamespace(speculative_num_steps=3)
    model_worker_batch = SimpleNamespace(seq_lens=np.array([0, 0], dtype=np.int32))

    _materialize_draft_extend_for_decode_fused(draft_worker, model_worker_batch, dispatch)

    np.testing.assert_array_equal(
        batch_output.next_draft_input.verified_id, np.empty((0,), dtype=np.int32)
    )
    np.testing.assert_array_equal(batch_output.accept_lens, np.array([0, 0], dtype=np.int32))


def test_materialize_draft_extend_precompile_dummy_accept_zero_noop():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        FusedDraftExtendDispatch,
        _materialize_draft_extend_for_decode_fused,
    )

    batch_output = SimpleNamespace(
        next_draft_input=EagleDraftInput(verified_id=np.empty((0,), dtype=np.int32)),
        accept_lens=np.array([0, 0], dtype=np.int32),
        allocate_lens=np.array([65, 65], dtype=np.int32),
    )
    dispatch = FusedDraftExtendDispatch(
        batch_output=batch_output,
        selector=np.array([0, 1], dtype=np.int32),
        selected_layer0_hidden=jnp.zeros((2, 4), dtype=jnp.float32),
        topk_index_stacked=jnp.zeros((2, 1, 1), dtype=jnp.int32),
        previous_token_list=jnp.zeros((2, 1), dtype=jnp.int32),
        selected_verified_id=jnp.zeros((2,), dtype=jnp.int32),
        verified_id_arr=jnp.zeros((8,), dtype=jnp.int32),
        accept_lens_device=jnp.array([0, 0], dtype=jnp.int32),
        materialize_hidden=False,
        materialize_topk=False,
        accept_lens_host=np.array([0, 0], dtype=np.int32),
        verified_id_host=np.empty((0,), dtype=np.int32),
    )
    draft_worker = SimpleNamespace(speculative_num_steps=3)
    model_worker_batch = SimpleNamespace(
        seq_lens=np.array([1, 1], dtype=np.int32),
        is_precompile_dummy=True,
    )

    _materialize_draft_extend_for_decode_fused(draft_worker, model_worker_batch, dispatch)

    np.testing.assert_array_equal(
        batch_output.next_draft_input.verified_id, np.empty((0,), dtype=np.int32)
    )
    np.testing.assert_array_equal(batch_output.allocate_lens, np.empty((0,), dtype=np.int32))
    np.testing.assert_array_equal(batch_output.accept_lens, np.array([0, 0], dtype=np.int32))


def test_materialize_draft_extend_drops_padding_rows_from_scheduler_state():
    from sgl_jax.srt.speculative.draft_extend_fused import (
        FusedDraftExtendDispatch,
        _materialize_draft_extend_for_decode_fused,
    )

    batch_output = SimpleNamespace(
        next_draft_input=EagleDraftInput(verified_id=np.empty((0,), dtype=np.int32)),
        accept_lens=np.array([2, 0], dtype=np.int32),
        allocate_lens=np.array([128, 0], dtype=np.int32),
    )
    dispatch = FusedDraftExtendDispatch(
        batch_output=batch_output,
        selector=np.array([0, 1], dtype=np.int32),
        selected_layer0_hidden=jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        topk_index_stacked=jnp.arange(2, dtype=jnp.int32).reshape(2, 1, 1),
        previous_token_list=jnp.zeros((2, 1), dtype=jnp.int32),
        selected_verified_id=jnp.zeros((2,), dtype=jnp.int32),
        verified_id_arr=jnp.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=jnp.int32),
        accept_lens_device=jnp.array([2, 0], dtype=jnp.int32),
        accept_lens_host=np.array([2, 0], dtype=np.int32),
    )
    draft_worker = SimpleNamespace(speculative_num_steps=3)
    model_worker_batch = SimpleNamespace(seq_lens=np.array([100, 0], dtype=np.int32))

    _materialize_draft_extend_for_decode_fused(draft_worker, model_worker_batch, dispatch)

    np.testing.assert_array_equal(
        batch_output.next_draft_input.verified_id, np.array([11], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        batch_output.next_draft_input.hidden_states, np.array([[0, 1, 2, 3]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        batch_output.next_draft_input.topk_index, np.array([[[0]]], dtype=np.int32)
    )
    np.testing.assert_array_equal(batch_output.allocate_lens, np.array([128], dtype=np.int32))
    np.testing.assert_array_equal(batch_output.accept_lens, np.array([2, 0], dtype=np.int32))


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
        speculative_num_draft_tokens=4,
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
        speculative_num_draft_tokens=4,
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
        return np.arange(128, 128 + int(extend_num_tokens), dtype=np.int32)

    monkeypatch.setattr(
        eagle_util,
        "alloc_paged_token_slots_extend",
        fake_alloc_paged_token_slots_extend,
    )
    req_to_token = np.zeros((1, 256), dtype=np.int32)
    req_to_token[0, 127] = 127
    req = SimpleNamespace(kv_committed_len=116, kv_allocated_len=128, decode_batch_idx=0)
    schedule_batch = SimpleNamespace(
        batch_size=lambda: 1,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
        tree_cache=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
        ),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        reqs_info=[
            SimpleNamespace(
                reqs=[req],
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
    assert req.decode_batch_idx == 1
    assert req.kv_committed_len == 117
    assert req.kv_allocated_len == 192


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
        speculative_num_draft_tokens=4,
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


def test_stash_prebuilt_same_batch_chain_keeps_pre_padded_verify_layout():
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    calls = []
    worker.spec_worker = SimpleNamespace(
        forward_batch_speculative_verify_phase_enqueue=lambda batch, prepared_launch=None: (
            calls.append((batch, prepared_launch)) or object()
        )
    )
    req_pool = np.array([1, 2], dtype=np.int32)
    candidate = SimpleNamespace(
        bid=21,
        req_pool_indices=req_pool.copy(),
        seq_lens=np.array([21, 22], dtype=np.int32),
        skip_fused_verify_padding_for_decode=True,
        prepared_fused_greedy_verify_launch=None,
        same_batch_chain_verify_write_lens=np.array([24, 25], dtype=np.int32),
        same_batch_chain_allocate_lens=np.array([64, 64], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            new_seq_lens=np.array([21, 22], dtype=np.int32),
            allocate_lens=np.array([64, 64], dtype=np.int32),
            verify_write_lens=np.array([24, 25], dtype=np.int32),
        ),
    )
    pending = SimpleNamespace(
        padded_req_pool_indices=req_pool.copy(),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.zeros((2, 1, 1), dtype=np.int32),
            topk_p=np.ones((2, 1, 1), dtype=np.float32),
            hidden_states=np.ones((2, 4), dtype=np.float32),
            verified_id=np.array([7, 8], dtype=np.int32),
            new_seq_lens=np.array([21, 22], dtype=np.int32),
        ),
    )

    worker._stash_prebuilt_same_batch_spec_chain_candidate(candidate, pending)

    assert len(calls) == 1
    enqueued_batch, prepared_launch = calls[0]
    assert prepared_launch is None
    assert enqueued_batch.skip_fused_verify_padding_for_decode is True
    np.testing.assert_array_equal(enqueued_batch.seq_lens, np.array([21, 22], dtype=np.int32))


def test_stash_reuses_phase_b_prepared_same_batch_chain_launch():
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    calls = []
    worker.spec_worker = SimpleNamespace(
        forward_batch_speculative_verify_phase_enqueue=lambda batch, prepared_launch=None: (
            calls.append((batch, prepared_launch)) or object()
        )
    )
    Prepared = namedtuple("Prepared", ["previous_verified_id", "previous_token_list"])
    prepared = Prepared(previous_verified_id=None, previous_token_list=None)
    req_pool = np.array([1, 2], dtype=np.int32)
    candidate = SimpleNamespace(
        bid=22,
        req_pool_indices=req_pool.copy(),
        seq_lens=np.array([21, 22], dtype=np.int32),
        skip_fused_verify_padding_for_decode=True,
        prepared_fused_greedy_verify_launch=prepared,
        same_batch_chain_prepared_layout="phase_b",
        same_batch_chain_verify_write_lens=np.array([24, 25], dtype=np.int32),
        same_batch_chain_allocate_lens=np.array([64, 64], dtype=np.int32),
        spec_info_padded=EagleDraftInput(
            new_seq_lens=np.array([21, 22], dtype=np.int32),
            allocate_lens=np.array([64, 64], dtype=np.int32),
            verify_write_lens=np.array([24, 25], dtype=np.int32),
        ),
    )
    pending = SimpleNamespace(
        padded_req_pool_indices=req_pool.copy(),
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[3]], [[4]]], dtype=np.int32),
            topk_p=np.ones((2, 1, 1), dtype=np.float32),
            hidden_states=np.ones((2, 4), dtype=np.float32),
            verified_id=np.array([7, 8], dtype=np.int32),
            new_seq_lens=np.array([21, 22], dtype=np.int32),
        ),
    )

    worker._stash_prebuilt_same_batch_spec_chain_candidate(candidate, pending)

    assert len(calls) == 1
    enqueued_batch, prepared_launch = calls[0]
    assert prepared_launch is not prepared
    assert enqueued_batch.prepared_fused_greedy_verify_launch is prepared_launch
    np.testing.assert_array_equal(
        prepared_launch.previous_verified_id,
        np.array([7, 8], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        prepared_launch.previous_token_list,
        np.array([[3], [4]], dtype=np.int32),
    )


def test_phase_b_dispatch_prebuild_waits_for_host_new_seq_lens(monkeypatch):
    from sgl_jax.srt.managers import tp_worker_overlap_thread
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    prepared = object()
    req_pool = np.array([1, 2], dtype=np.int32)

    class FakeDeviceArray:
        pass

    monkeypatch.setattr(tp_worker_overlap_thread.jax, "Array", FakeDeviceArray)
    device_seq_lens = FakeDeviceArray()

    def fake_prepare(candidate):
        raise AssertionError("device-only seq_lens must not build a chain candidate")
        return prepared

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
        seq_lens=np.array([19, 20], dtype=np.int32),
        seq_lens_sum=39,
        speculative_num_draft_tokens=4,
        out_cache_loc=np.arange(8, dtype=np.int32),
        bid=13,
    )
    pending = SimpleNamespace(
        padded_next_draft_input=EagleDraftInput(new_seq_lens=device_seq_lens),
        padded_req_pool_indices=req_pool.copy(),
    )

    candidate = worker._prebuild_same_batch_spec_chain_candidate_after_phase_b_dispatch(
        model_worker_batch,
        pending,
    )

    assert candidate is None


def test_phase_b_dispatch_prebuild_uses_host_new_seq_lens_and_marks_phase_b(monkeypatch):
    from sgl_jax.srt.managers import tp_worker_overlap_thread
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    prepared = object()
    req_pool = np.array([1, 2], dtype=np.int32)

    class FakeDeviceArray:
        pass

    monkeypatch.setattr(tp_worker_overlap_thread.jax, "Array", FakeDeviceArray)
    worker._prepare_chained_verify_launch_after_phase_a = lambda candidate: prepared
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
        seq_lens=np.array([19, 20], dtype=np.int32),
        seq_lens_sum=39,
        speculative_num_draft_tokens=4,
        out_cache_loc=np.arange(8, dtype=np.int32),
        bid=23,
    )
    pending = SimpleNamespace(
        padded_next_draft_input=EagleDraftInput(
            topk_index=np.array([[[3]], [[4]]], dtype=np.int32),
            topk_p=np.ones((2, 1, 1), dtype=np.float32),
            hidden_states=np.ones((2, 4), dtype=np.float32),
            verified_id=np.array([7, 8], dtype=np.int32),
            new_seq_lens=FakeDeviceArray(),
        ),
        padded_req_pool_indices=req_pool.copy(),
        padded_new_seq_lens_host=np.array([21, 22], dtype=np.int32),
    )

    candidate = worker._prebuild_same_batch_spec_chain_candidate_after_phase_b_dispatch(
        model_worker_batch,
        pending,
    )

    assert candidate is not None
    assert candidate.prepared_fused_greedy_verify_launch is prepared
    assert candidate.same_batch_chain_prepared_layout == "phase_b"
    np.testing.assert_array_equal(candidate.seq_lens, np.array([21, 22], dtype=np.int32))


def test_same_batch_chain_requires_host_lens_for_materialize_frontier(monkeypatch):
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    monkeypatch.setattr(EagleDraftInput, "ALLOC_LEN_PER_DECODE", 4)
    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    req_pool = np.array([1, 2], dtype=np.int32)
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
        same_batch_chain_verify_write_lens=np.array([20, 24], dtype=np.int32),
        same_batch_chain_allocate_lens=np.array([64, 64], dtype=np.int32),
        spec_info_padded=EagleDraftInput(allocate_lens=np.array([64, 64], dtype=np.int32)),
        seq_lens=np.array([19, 23], dtype=np.int32),
        seq_lens_sum=42,
        speculative_num_draft_tokens=4,
        out_cache_loc=np.arange(8, dtype=np.int32),
        bid=14,
    )
    pending = SimpleNamespace(
        padded_next_draft_input=EagleDraftInput(new_seq_lens=jnp.array([21, 25], dtype=jnp.int32)),
        padded_req_pool_indices=req_pool.copy(),
    )

    candidate = worker._build_same_batch_spec_chain_candidate_batch(
        model_worker_batch,
        pending,
    )

    assert candidate is None


def test_same_batch_chain_pop_requires_matching_frontiers():
    from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient

    worker = ModelWorkerClient.__new__(ModelWorkerClient)
    req_pool = np.array([1, 2, -1, -1], dtype=np.int32)
    model_worker_batch = SimpleNamespace(
        req_pool_indices=req_pool.copy(),
        seq_lens=np.array([21, 22, 0, 0], dtype=np.int32),
        speculative_num_draft_tokens=4,
        spec_info_padded=SimpleNamespace(
            verify_write_lens=np.array([24, 25, 0, 0], dtype=np.int32),
            allocate_lens=np.array([64, 64, 0, 0], dtype=np.int32),
        ),
    )
    candidate_batch = SimpleNamespace(bid=17)

    def make_candidate(**overrides):
        values = dict(
            req_pool_indices=req_pool.copy(),
            expected_seq_lens=np.array([21, 22, 0, 0], dtype=np.int32),
            expected_verify_write_lens=np.array([24, 25, 0, 0], dtype=np.int32),
            expected_allocate_lens=np.array([64, 64, 0, 0], dtype=np.int32),
            verify_async_result=object(),
            model_worker_batch=candidate_batch,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    worker.pending_same_batch_spec_chain_candidate = make_candidate()
    assert worker._pop_matching_same_batch_spec_chain_candidate(model_worker_batch) is not None

    worker.pending_same_batch_spec_chain_candidate = make_candidate(
        expected_verify_write_lens=np.array([24, 26, 0, 0], dtype=np.int32),
    )
    assert worker._pop_matching_same_batch_spec_chain_candidate(model_worker_batch) is None

    worker.pending_same_batch_spec_chain_candidate = make_candidate(
        expected_seq_lens=np.array([21, 26, 0, 0], dtype=np.int32),
    )
    assert worker._pop_matching_same_batch_spec_chain_candidate(model_worker_batch) is None


def test_flashattention_target_verify_metadata_uses_device_seq_lens():
    from jax.sharding import Mesh

    from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1), axis_names=("data",))
    attn = FlashAttention(
        num_attn_heads=1,
        num_kv_heads=1,
        head_dim=16,
        page_size=64,
        mesh=mesh,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=np.array([20, 0], dtype=np.int32),
        target_verify_seq_lens_device=jnp.array([21, 0], dtype=jnp.int32),
        cache_loc=np.arange(128, dtype=np.int32),
        dp_size=1,
        per_dp_bs_size=2,
        logits_indices_selector=np.array([0], dtype=np.int32),
        spec_info_padded=SimpleNamespace(custom_mask=None, draft_token_num=4),
    )

    metadata = attn.get_eagle_forward_metadata(batch)

    np.testing.assert_array_equal(np.asarray(metadata.seq_lens), np.array([21, 0], dtype=np.int32))
    np.testing.assert_array_equal(
        np.asarray(metadata.cu_q_lens),
        np.array([0, 4, 4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.cu_kv_lens),
        np.array([0, 64, 64], dtype=np.int32),
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
        swa_prefix_lock_released = False
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


def test_spec_decode_accept_updates_req_kv_accounting_for_swa_eviction():
    from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[5, 6, 7],
        decode_batch_idx=11,
        kv_committed_len=4,
        kv_allocated_len=8,
    )

    SchedulerOutputProcessorMixin._advance_spec_decode_req_kv_accounting(
        req,
        accepted_len=3,
        finished=False,
    )

    assert req.decode_batch_idx == 11
    assert req.kv_committed_len == 6
    assert req.kv_allocated_len == 8


def test_eagle_prepare_for_decode_advances_req_kv_lifecycle_from_req_fields(monkeypatch):
    import sgl_jax.srt.speculative.eagle_util as eagle_util

    class FakeReq:
        req_pool_idx = 1
        kv_committed_len = 100
        kv_allocated_len = 100
        decode_batch_idx = 7

    class FakeBatch:
        dp_size = 1
        tree_cache = object()
        req_to_token_pool = SimpleNamespace(req_to_token=np.zeros((4, 256), dtype=np.int32))
        token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        reqs_info = [
            SimpleNamespace(
                reqs=[FakeReq()],
                seq_lens=np.array([101], dtype=np.int32),
                req_pool_indices=np.array([1], dtype=np.int32),
            )
        ]

        def batch_size(self):
            return 1

    old_alloc_len = EagleDraftInput.ALLOC_LEN_PER_DECODE
    EagleDraftInput.ALLOC_LEN_PER_DECODE = 4
    monkeypatch.setattr(
        eagle_util,
        "alloc_token_slots",
        lambda tree_cache, n, dp_rank=0: np.arange(1000, 1000 + n, dtype=np.int32),
    )
    try:
        draft = EagleDraftInput(
            allocate_lens=np.array([100], dtype=np.int32),
            verify_write_lens=np.array([100], dtype=np.int32),
        )
        batch = FakeBatch()

        draft.prepare_for_decode(batch)
    finally:
        EagleDraftInput.ALLOC_LEN_PER_DECODE = old_alloc_len

    req = batch.reqs_info[0].reqs[0]
    assert req.decode_batch_idx == 8
    assert req.kv_committed_len == 101
    assert req.kv_allocated_len == 116
    np.testing.assert_array_equal(draft.allocate_lens, np.array([116], dtype=np.int32))
    np.testing.assert_array_equal(draft.verify_write_lens, np.array([104], dtype=np.int32))
    np.testing.assert_array_equal(
        batch.reqs_info[0].out_cache_loc,
        np.arange(1000, 1004, dtype=np.int32),
    )


def test_release_kv_cache_allows_spec_overallocated_tail():
    from sgl_jax.srt.mem_cache.common import release_kv_cache

    class FakeReq:
        req_pool_idx = 0
        dp_rank = 0
        kv_committed_len = 5
        kv_allocated_len = 12
        kv_committed_freed = False
        kv_overallocated_freed = False

        def pop_committed_kv_cache(self):
            assert not self.kv_committed_freed
            self.kv_committed_freed = True
            return self.kv_committed_len

        def pop_overallocated_kv_cache(self):
            assert not self.kv_overallocated_freed
            self.kv_overallocated_freed = True
            return self.kv_committed_len, self.kv_allocated_len

    freed = []
    req_pool_freed = []
    req = FakeReq()
    tree_cache = SimpleNamespace(
        page_size=4,
        req_to_token_pool=SimpleNamespace(
            req_to_token=np.arange(32, dtype=np.int32).reshape(1, 32),
            free=lambda req_arg: req_pool_freed.append(req_arg),
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda indices, dp_rank=0: freed.append((np.asarray(indices).copy(), dp_rank))
        ),
        cache_finished_req=lambda req_arg, is_insert=True: req_arg.pop_committed_kv_cache(),
    )

    release_kv_cache(req, tree_cache, allow_overallocated=True)

    assert req.kv_committed_freed is True
    assert req.kv_overallocated_freed is True
    assert req_pool_freed == [req]
    assert len(freed) == 1
    np.testing.assert_array_equal(freed[0][0], np.array([8, 9, 10, 11], dtype=np.int32))
    assert freed[0][1] == 0


def test_swa_radix_cache_can_release_only_swa_prefix_lock_for_leaf():
    from sgl_jax.srt.mem_cache.radix_cache import RadixKey
    from sgl_jax.srt.mem_cache.swa_radix_cache import LRUList, SWARadixCache, TreeNode

    freed = []
    root = TreeNode()
    leaf = TreeNode()
    leaf.parent = root
    leaf.key = RadixKey([1, 2, 3, 4], None)
    leaf.value = np.array([11, 12, 13, 14], dtype=np.int32)
    leaf.full_lock_ref = 1
    leaf.swa_lock_ref = 1
    leaf.swa_uuid = 123
    root.children[1] = leaf

    cache = SWARadixCache.__new__(SWARadixCache)
    cache.disable = False
    cache.root_node = root
    cache.swa_lru_list = LRUList(swa=True)
    cache.swa_lru_list.insert_mru(leaf)
    mapping = np.zeros(32, dtype=np.int32)
    mapping[leaf.value] = leaf.value
    cache.token_to_kv_pool_allocator = SimpleNamespace(
        full_to_swa_index_mapping=mapping,
        free_swa=lambda indices, dp_rank=0: freed.append((np.asarray(indices).copy(), dp_rank)),
    )
    cache.swa_protected_size_ = {0: 4}
    cache.swa_evictable_size_ = {0: 0}

    cache.dec_swa_lock_only(leaf, leaf.swa_uuid)

    assert leaf.full_lock_ref == 1
    assert leaf.swa_lock_ref == 0
    assert leaf.swa_tombstone is True
    assert cache.swa_protected_size_[0] == 0
    assert cache.swa_evictable_size_[0] == 0
    assert leaf.id not in cache.swa_lru_list.cache
    assert len(freed) == 1
    np.testing.assert_array_equal(freed[0][0], leaf.value)
    assert freed[0][1] == 0
