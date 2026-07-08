import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashVerifyInput,
    compute_dflash_accept_len_and_bonus,
)
from sgl_jax.srt.speculative.spec_info import SpecInput


def test_compute_dflash_accept_len_and_bonus():
    candidates = jnp.array(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=jnp.int32,
    )
    target_predict = jnp.array(
        [
            [11, 12, 13, 99],  # accept 3, bonus target_predict[3]
            [21, 77, 88, 99],  # accept 1, bonus target_predict[1]
            [55, 31, 32, 33],  # accept 0, bonus target_predict[0]
        ],
        dtype=jnp.int32,
    )

    accept_len, bonus = compute_dflash_accept_len_and_bonus(candidates, target_predict)

    np.testing.assert_array_equal(np.asarray(accept_len), np.array([3, 1, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(bonus), np.array([99, 77, 55], dtype=np.int32))


def test_dflash_verify_input_is_spec_input_and_pytree():
    vi = DFlashVerifyInput(
        draft_token=jnp.arange(8, dtype=jnp.int32),
        positions=jnp.arange(8, dtype=jnp.int32),
        custom_mask=None,
        draft_token_num=4,
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )

    assert isinstance(vi, SpecInput)
    assert vi.is_verify_input() and not vi.is_draft_input()
    assert vi.get_spec_adjust_token_coefficient() == 4
    assert vi.get_verify_token_num(bs=2) == 8

    leaves, treedef = jax.tree_util.tree_flatten(vi)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, DFlashVerifyInput)
    assert restored.draft_token_num == 4
    assert restored.capture_hidden_mode == CaptureHiddenMode.FULL


def test_dflash_verify_input_verify_greedy():
    vi = DFlashVerifyInput(
        draft_token=jnp.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=jnp.int32),
        positions=jnp.arange(8, dtype=jnp.int32),
        draft_token_num=4,
    )
    target_predict = jnp.array([11, 12, 99, 0, 99, 0, 0, 0], dtype=jnp.int32)

    accept_len, bonus = vi.verify_greedy(target_predict)
    np.testing.assert_array_equal(np.asarray(accept_len), np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(bonus), np.array([99, 99], dtype=np.int32))


def test_dflash_draft_input_is_spec_input():
    di = DFlashDraftInput(
        verified_id=np.array([1, 2], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 7], dtype=np.int32),
    )

    assert isinstance(di, SpecInput)
    assert di.is_draft_input() and not di.is_verify_input()
    np.testing.assert_array_equal(di.get_logical_token_num(bs=2), np.ones(2, dtype=np.int32))
    assert di.get_verify_token_num(bs=2) == 0
