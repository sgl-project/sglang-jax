"""CPU tests for Frozen-KV MTP's non-overlap batch-merge contract."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.frozen_kv_mtp_seed import FrozenKvMtpSeedState
from sgl_jax.srt.speculative.frozen_kv_mtp_worker import (
    FrozenKvMtpDraftInput,
    FrozenKvMtpDraftWorker,
)
from sgl_jax.srt.speculative.overlap_utils import can_merge_spec_non_overlap_prefill
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def _state(rows: list[tuple[int, int, int, int]]) -> FrozenKvMtpDraftInput:
    """Build rows of (verified_token, hidden_tag, allocated, committed)."""
    verified, hidden, allocated, committed = map(np.asarray, zip(*rows, strict=True))
    return FrozenKvMtpDraftInput(
        topk_p=verified[:, None].astype(np.float32) / 100,
        topk_index=(verified + 1)[:, None].astype(np.int32),
        hidden_states=hidden[:, None].astype(np.float32),
        verified_id=verified.astype(np.int32),
        accept_length=np.full(len(rows), 2, dtype=np.int32),
        accept_length_cpu=np.full(len(rows), 2, dtype=np.int32),
        allocate_lens=allocated.astype(np.int32),
        new_seq_lens=committed.astype(np.int32),
    )


def _batch(rows: list[tuple[int, int, int, int]]) -> ScheduleBatch:
    """Minimal single-rank batch exercising the production merge method."""

    class Req:
        lora_id = "0"

    info = ScheduleReqsInfo()
    info.reqs = [Req() for _ in rows]
    info.req_pool_indices = np.arange(len(rows), dtype=np.int32)
    info.seq_lens = np.asarray([row[3] for row in rows], dtype=np.int32)
    info.seq_lens_sum = int(info.seq_lens.sum())
    info.out_cache_loc = None
    info.output_ids = None
    info.top_logprobs_nums = None
    info.token_ids_logprobs = None
    info.sampling_info = None
    info.spec_info = _state(rows)

    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.__dict__.update(
        dp_size=1,
        reqs_info=[info],
        forward_mode=ForwardMode.DECODE,
        return_logprob=False,
        return_output_logprob_only=False,
        return_hidden_states=False,
        has_stream=False,
        has_grammar=False,
    )
    return batch


def test_frozen_kv_merge_keeps_token_seed_and_kv_length_rows_together():
    running = _state([(11, 101, 128, 127)])
    prefill = _state([(22, 202, 256, 255), (33, 303, 384, 383)])

    running.merge_batch(prefill)

    np.testing.assert_array_equal(running.verified_id, [11, 22, 33])
    np.testing.assert_array_equal(running.hidden_states[:, 0], [101, 202, 303])
    np.testing.assert_array_equal(running.allocate_lens, [128, 256, 384])
    np.testing.assert_array_equal(running.new_seq_lens, [127, 255, 383])
    np.testing.assert_array_equal(running.accept_length, [2, 2, 2])


def test_frozen_kv_merge_allows_prefill_without_accept_length_and_preserves_its_input():
    """GSM8K c32 merges a verified decode batch with a fresh prefill batch."""
    running = _state([(11, 101, 128, 127)])
    running.seed_state = FrozenKvMtpSeedState(
        bonus_token=jnp.asarray([99], dtype=jnp.int32),
        target_hidden=jnp.asarray([[909]], dtype=jnp.float32),
        committed_lens=jnp.asarray([127], dtype=jnp.int32),
        allocate_lens=jnp.asarray([128], dtype=jnp.int32),
        request_indices=jnp.asarray([5], dtype=jnp.int32),
        valid_mask=jnp.asarray([True]),
    )
    prefill = _state([(22, 202, 256, 255)])
    prefill.accept_length = None
    prefill.accept_length_cpu = None

    running.merge_batch(prefill)

    assert running.accept_length is None
    assert running.accept_length_cpu is None
    np.testing.assert_array_equal(running.seed_state.valid_mask, [True, False])

    worker = FrozenKvMtpDraftWorker.__new__(FrozenKvMtpDraftWorker)
    batch = type(
        "Batch",
        (),
        {
            "spec_info_padded": running,
            "logits_indices_selector": np.asarray([0, 1], dtype=np.int32),
        },
    )()
    worker._prepare_seed_proposal(batch)

    # Existing decoded request consumes its target seed; the new prefill keeps
    # its own generic draft token/hidden state until it is verified once.
    np.testing.assert_array_equal(running.verified_id, [99, 22])
    np.testing.assert_array_equal(running.hidden_states[:, 0], [909, 202])
    assert running.seed_state is None


def test_frozen_kv_state_remains_a_jax_pytree_after_its_specialization():
    state = _state([(11, 101, 128, 127)])

    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(restored, FrozenKvMtpDraftInput)
    np.testing.assert_array_equal(restored.verified_id, [11])
    np.testing.assert_array_equal(restored.allocate_lens, [128])
    np.testing.assert_array_equal(restored.new_seq_lens, [127])
    np.testing.assert_array_equal(restored.accept_length, [2])
    np.testing.assert_array_equal(restored.accept_length_cpu, [2])


def test_frozen_kv_filter_retract_keeps_page_boundary_metadata_aligned():
    # The entries deliberately land in distinct 128-token page ranges.  A
    # retraction must remove every matching state row, not only draft logits.
    state = _state(
        [
            (11, 101, 128, 127),
            (22, 202, 256, 255),
            (33, 303, 384, 383),
        ]
    )

    state.filter_batch(np.array([2, 0], dtype=np.int32), has_been_filtered=False)

    np.testing.assert_array_equal(state.verified_id, [33, 11])
    np.testing.assert_array_equal(state.hidden_states[:, 0], [303, 101])
    np.testing.assert_array_equal(state.allocate_lens, [384, 128])
    np.testing.assert_array_equal(state.new_seq_lens, [383, 127])
    np.testing.assert_array_equal(state.accept_length_cpu, [2, 2])


def test_schedule_batch_merge_uses_the_frozen_contract_for_prefill_admission():
    running = _batch([(11, 101, 128, 127)])
    prefill = _batch([(22, 202, 256, 255)])

    running.merge_batch(prefill)

    info = running.reqs_info[0]
    assert len(info.reqs) == 2
    assert isinstance(info.spec_info, FrozenKvMtpDraftInput)
    np.testing.assert_array_equal(info.spec_info.verified_id, [11, 22])
    np.testing.assert_array_equal(info.spec_info.allocate_lens, [128, 256])
    np.testing.assert_array_equal(info.seq_lens, [127, 255])


def test_frozen_kv_state_round_trips_through_dp_scatter_and_split():
    flat = _state([(11, 101, 128, 127), (22, 202, 256, 255), (33, 303, 384, 383)])
    padded = ScheduleBatch._scatter_spec_info_to_dp_slots(
        flat, selector=np.array([0, 1, 3], dtype=np.int32), total_bs=4
    )
    assert isinstance(padded, FrozenKvMtpDraftInput)

    # The worker compacts its padded output using the same selector before the
    # scheduler splits it back to per-DP-rank state.
    selector = np.array([0, 1, 3], dtype=np.int32)
    compact = FrozenKvMtpDraftInput(
        topk_p=np.asarray(padded.topk_p)[selector],
        topk_index=np.asarray(padded.topk_index)[selector],
        hidden_states=np.asarray(padded.hidden_states)[selector],
        verified_id=np.asarray(padded.verified_id)[selector],
        accept_length=np.asarray(padded.accept_length)[selector],
        accept_length_cpu=np.asarray(padded.accept_length_cpu)[selector],
        allocate_lens=np.asarray(padded.allocate_lens)[selector],
        new_seq_lens=np.asarray(padded.new_seq_lens)[selector],
    )
    rank0, rank1 = ScheduleBatch._split_spec_info_per_rank(compact, [2, 1])
    for state, tokens, allocated in ((rank0, [11, 22], [128, 256]), (rank1, [33], [384])):
        assert isinstance(state, FrozenKvMtpDraftInput)
        np.testing.assert_array_equal(state.verified_id, tokens)
        np.testing.assert_array_equal(state.allocate_lens, allocated)


def test_frozen_kv_non_overlap_admission_is_allowed_only_for_its_checked_state_path():
    assert can_merge_spec_non_overlap_prefill(False, SpeculativeAlgorithm.FROZEN_KV_MTP)
    assert not can_merge_spec_non_overlap_prefill(True, SpeculativeAlgorithm.FROZEN_KV_MTP)
    assert not can_merge_spec_non_overlap_prefill(False, SpeculativeAlgorithm.NEXTN)


def test_frozen_kv_merge_rejects_missing_or_mismatched_state():
    running = _state([(11, 101, 128, 127)])
    incomplete = FrozenKvMtpDraftInput(
        topk_p=np.ones((1, 1), dtype=np.float32),
        topk_index=np.ones((1, 1), dtype=np.int32),
        hidden_states=np.ones((1, 1), dtype=np.float32),
        verified_id=np.ones((1,), dtype=np.int32),
        allocate_lens=np.array([127], dtype=np.int32),
        new_seq_lens=np.array([128], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="shorter than its committed sequence length"):
        running.merge_batch(incomplete)


def test_frozen_kv_relay_descriptor_merges_and_scatter_without_host_model_state():
    """The generic future_indices lifecycle is valid for Frozen-KV too."""
    left = FrozenKvMtpDraftInput(
        future_indices=np.asarray([7], dtype=np.int32),
        allocate_lens=np.asarray([128], dtype=np.int32),
        new_seq_lens=np.asarray([120], dtype=np.int32),
        accept_length_cpu=np.asarray([3], dtype=np.int32),
    )
    right = FrozenKvMtpDraftInput(
        future_indices=np.asarray([11], dtype=np.int32),
        allocate_lens=np.asarray([256], dtype=np.int32),
        new_seq_lens=np.asarray([240], dtype=np.int32),
        accept_length_cpu=np.asarray([2], dtype=np.int32),
    )

    left.merge_batch(right)
    left.filter_batch(np.asarray([1], dtype=np.int32), has_been_filtered=False)
    padded = ScheduleBatch._scatter_spec_info_to_dp_slots(
        left, selector=np.asarray([2], dtype=np.int32), total_bs=4
    )

    assert isinstance(padded, FrozenKvMtpDraftInput)
    assert padded.topk_p is None
    assert padded.hidden_states is None
    np.testing.assert_array_equal(padded.future_indices, [0, 0, 11, 0])
    np.testing.assert_array_equal(padded.allocate_lens, [0, 0, 256, 0])
    np.testing.assert_array_equal(padded.new_seq_lens, [0, 0, 240, 0])


def test_frozen_kv_relay_descriptor_rejects_mixed_nonrelay_merge():
    relay = FrozenKvMtpDraftInput(
        future_indices=np.asarray([7], dtype=np.int32),
        allocate_lens=np.asarray([128], dtype=np.int32),
        new_seq_lens=np.asarray([120], dtype=np.int32),
    )
    with pytest.raises(AssertionError, match="future_indices"):
        relay.merge_batch(_state([(22, 202, 256, 255)]))


def test_frozen_kv_relay_descriptor_trim_keeps_seed_mask_request_aligned():
    state = FrozenKvMtpDraftInput(
        future_indices=np.asarray([7, 11, 13], dtype=np.int32),
        allocate_lens=np.asarray([128, 256, 384], dtype=np.int32),
        new_seq_lens=np.asarray([120, 240, 360], dtype=np.int32),
        relay_seed_mask=np.asarray([True, False, True]),
    )

    state.trim_to_length(2)

    np.testing.assert_array_equal(state.future_indices, [7, 11])
    np.testing.assert_array_equal(state.relay_seed_mask, [True, False])
