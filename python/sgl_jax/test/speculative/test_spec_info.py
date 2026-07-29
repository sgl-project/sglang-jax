import jax
import numpy as np

from sgl_jax.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput


def test_eagle_draft_input_pytree_round_trip():
    draft_input = EagleDraftInput(
        topk_p=np.ones((2, 1), dtype=np.float32),
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
        topk_p=np.ones((1, 3), dtype=np.float32),
        topk_index=np.array([[11, 12, 13]], dtype=np.int32),
        hidden_states=np.ones((1, 4), dtype=np.float32),
        verified_id=np.array([10], dtype=np.int32),
        allocate_lens=np.array([32], dtype=np.int32),
    )
    prefetched = EagleDraftInput(
        topk_p=np.ones((1, 1), dtype=np.float32),
        topk_index=np.array([[21]], dtype=np.int32),
        hidden_states=np.full((1, 4), 2, dtype=np.float32),
        verified_id=np.array([20], dtype=np.int32),
        allocate_lens=np.array([48], dtype=np.int32),
    )

    running.merge_batch(prefetched)

    assert running.topk_p.shape == (2, 1)
    np.testing.assert_array_equal(running.topk_index, np.array([[11], [21]], dtype=np.int32))
    np.testing.assert_array_equal(running.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(running.allocate_lens, np.array([32, 48], dtype=np.int32))


def test_eagle_verify_input_pytree_round_trip():
    verify_input = EagleVerifyInput(
        draft_token=np.arange(8, dtype=np.int32),
        custom_mask=None,
        positions=np.arange(8, dtype=np.int32),
        retrive_index=np.arange(8, dtype=np.int32).reshape(2, 4),
        retrive_next_token=np.full((2, 4), -1, dtype=np.int32),
        retrive_next_sibling=np.full((2, 4), -1, dtype=np.int32),
        spec_steps=3,
        draft_token_num=4,
    )

    leaves, tree = jax.tree_util.tree_flatten(verify_input)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert restored.spec_steps == 3
    assert restored.draft_token_num == 4
    np.testing.assert_array_equal(restored.draft_token, verify_input.draft_token)
    np.testing.assert_array_equal(restored.retrive_index, verify_input.retrive_index)
