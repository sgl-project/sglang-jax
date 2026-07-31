import jax
import numpy as np
import pytest

from sgl_jax.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput


def test_eagle_draft_input_pytree_round_trip():
    draft_input = EagleDraftInput(
        topk_index=np.array([[3], [5]], dtype=np.int32),
        hidden_states=np.ones((2, 4), dtype=np.float32),
        verified_id=np.array([3, 5], dtype=np.int32),
        accept_length=np.array([1, 2], dtype=np.int32),
        accept_length_cpu=np.array([1, 2], dtype=np.int32),
        kv_indices=np.array([7, 11], dtype=np.int32),
        future_indices=np.array([0, 1], dtype=np.int32),
        num_tokens_per_batch=1,
        num_tokens_for_logprob_per_batch=1,
    )

    leaves, tree = jax.tree_util.tree_flatten(draft_input)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    np.testing.assert_array_equal(restored.topk_index, draft_input.topk_index)
    np.testing.assert_array_equal(restored.verified_id, draft_input.verified_id)
    np.testing.assert_array_equal(restored.kv_indices, draft_input.kv_indices)
    np.testing.assert_array_equal(restored.future_indices, draft_input.future_indices)


def test_eagle_draft_input_merge_downgrades_recurrent_chain_for_prefill_seed():
    running = EagleDraftInput(
        topk_index=np.array([[11, 12, 13]], dtype=np.int32),
        hidden_states=np.ones((1, 4), dtype=np.float32),
        verified_id=np.array([10], dtype=np.int32),
        allocate_lens=np.array([32], dtype=np.int32),
    )
    prefetched = EagleDraftInput(
        topk_index=np.array([[21]], dtype=np.int32),
        hidden_states=np.full((1, 4), 2, dtype=np.float32),
        verified_id=np.array([20], dtype=np.int32),
        allocate_lens=np.array([48], dtype=np.int32),
    )

    running.merge_batch(prefetched)

    assert running.topk_index.shape == (2, 1)
    np.testing.assert_array_equal(running.topk_index, np.array([[11], [21]], dtype=np.int32))
    np.testing.assert_array_equal(running.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(running.allocate_lens, np.array([32, 48], dtype=np.int32))


def test_eagle_draft_input_relay_merge_and_filter_preserve_request_indices():
    running = EagleDraftInput(
        future_indices=np.array([3, 5], dtype=np.int32),
        allocate_lens=np.array([32, 48], dtype=np.int32),
    )
    joined = EagleDraftInput(
        future_indices=np.array([7], dtype=np.int32),
        allocate_lens=np.array([64], dtype=np.int32),
    )

    running.merge_batch(joined)
    running.filter_batch(np.array([2, 0], dtype=np.int32), has_been_filtered=False)

    np.testing.assert_array_equal(running.future_indices, np.array([7, 3], dtype=np.int32))
    np.testing.assert_array_equal(running.allocate_lens, np.array([64, 32], dtype=np.int32))


def test_eagle_draft_input_rejects_bootstrap_relay_merge():
    relay = EagleDraftInput(future_indices=np.array([3], dtype=np.int32))
    bootstrap = EagleDraftInput(
        topk_index=np.array([[11]], dtype=np.int32),
        hidden_states=np.ones((1, 4), dtype=np.float32),
        verified_id=np.array([10], dtype=np.int32),
    )

    with pytest.raises(AssertionError, match="future_indices"):
        relay.merge_batch(bootstrap)


def test_eagle_verify_input_pytree_round_trip():
    verify_input = EagleVerifyInput(
        draft_token=np.arange(8, dtype=np.int32),
        positions=np.arange(8, dtype=np.int32),
    )

    leaves, tree = jax.tree_util.tree_flatten(verify_input)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    np.testing.assert_array_equal(restored.draft_token, verify_input.draft_token)
    np.testing.assert_array_equal(restored.positions, verify_input.positions)
