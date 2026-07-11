import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashVerifyInput,
    build_dflash_draft_block,
    compute_dflash_accept_len_and_bonus,
    compute_new_kv_slices,
    dflash_greedy_verify_outputs,
    gather_dflash_accepted_hidden_padded,
)
from sgl_jax.srt.speculative.overlap_utils import (
    can_merge_spec_non_overlap_prefill,
    use_legacy_eagle3_non_overlap,
)
from sgl_jax.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm


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


def test_dflash_verify_input_verify_from_logits():
    # bs=2, block_size=4, vocab=100. draft_token[:,0] is the seed.
    vi = DFlashVerifyInput(
        draft_token=jnp.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=jnp.int32),
        positions=jnp.arange(8, dtype=jnp.int32),
        draft_token_num=4,
    )
    # Build target logits whose argmax reproduces a chosen target_predict.
    # req0 target_predict = [11, 12, 13, 99] -> accept all 3 drafts, bonus 99
    # req1 target_predict = [21, 77, 88, 99] -> accept 1 draft, bonus 77
    target_predict = np.array([[11, 12, 13, 99], [21, 77, 88, 99]], dtype=np.int32)
    logits = np.full((8, 100), -1.0, dtype=np.float32)
    for i, row in enumerate(target_predict.reshape(-1)):
        logits[i, row] = 10.0
    logits = jnp.asarray(logits)

    accept_lens_out, next_token_ids_flat, new_verified_id, accept_len_draft = vi.verify(logits)

    np.testing.assert_array_equal(np.asarray(accept_lens_out), np.array([4, 2], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(accept_len_draft), np.array([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(new_verified_id), np.array([99, 77], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(next_token_ids_flat).reshape(2, 4), target_predict)


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


def test_dflash_draft_input_new_tokens_required_next_decode_page_aligned():
    class Req:
        def __init__(self, committed, allocated):
            self.kv_committed_len = committed
            self.kv_allocated_len = allocated

    di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0], dtype=np.int32),
        draft_seq_lens=np.array([0, 0], dtype=np.int32),
        block_size=16,
    )

    requests = [
        Req(committed=120, allocated=120),  # needs slots through 136 -> one new page
        Req(committed=16, allocated=128),  # already has enough page capacity
    ]

    assert di.new_tokens_required_next_decode(requests, page_size=128) == 128


def test_dflash_draft_input_align_to_reqs_appends_merged_request_state():
    class Req:
        def __init__(self, origin_input_ids, output_ids):
            self.origin_input_ids = origin_input_ids
            self.output_ids = output_ids

    di = DFlashDraftInput(
        verified_id=np.array([10, 20], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6], dtype=np.int32),
        block_size=16,
    )
    reqs = [
        Req([1, 10], []),
        Req([1, 20], []),
        Req([1, 2, 3], [30]),
    ]

    di._align_to_reqs(reqs, np.array([5, 6, 7], dtype=np.int32))

    np.testing.assert_array_equal(di.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))


def test_dflash_draft_input_scatter_pads_to_spec_decode_bucket():
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 30], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7], dtype=np.int32),
        block_size=16,
    )

    padded = ScheduleBatch._scatter_spec_info_to_dp_slots(
        di,
        selector=np.array([0, 1, 2], dtype=np.int32),
        total_bs=4,
    )

    np.testing.assert_array_equal(padded.verified_id, np.array([10, 20, 30, 0], dtype=np.int32))
    np.testing.assert_array_equal(padded.ctx_lens, np.array([0, 0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(padded.draft_seq_lens, np.array([5, 6, 7, 0], dtype=np.int32))

    [unpadded] = ScheduleBatch._split_spec_info_per_rank(padded, [3])
    np.testing.assert_array_equal(unpadded.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(unpadded.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))


def test_dflash_non_overlap_can_merge_without_legacy_eagle3_accounting():
    assert can_merge_spec_non_overlap_prefill(False, SpeculativeAlgorithm.DFLASH)
    assert not use_legacy_eagle3_non_overlap(False, SpeculativeAlgorithm.DFLASH)


def test_build_dflash_draft_block():
    verified_id = np.array([7, 8], dtype=np.int32)
    target_prefix_lens = np.array([5, 3], dtype=np.int32)

    block_ids, positions = build_dflash_draft_block(
        verified_id=verified_id,
        mask_token_id=99,
        target_prefix_lens=target_prefix_lens,
        block_size=4,
    )

    np.testing.assert_array_equal(
        np.asarray(block_ids),
        np.array([[7, 99, 99, 99], [8, 99, 99, 99]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(positions),
        np.array([[5, 6, 7, 8], [3, 4, 5, 6]], dtype=np.int32),
    )


def test_dflash_greedy_verify_outputs():
    # draft_token[:, 0] is the seed (already-emitted verified id from last step).
    draft_token = jnp.array(
        [
            [10, 11, 12, 13],  # accept 3 drafts + bonus -> emit 4
            [20, 21, 22, 23],  # accept 1 draft + bonus  -> emit 2
            [30, 31, 32, 33],  # accept 0 drafts + bonus -> emit 1
        ],
        dtype=jnp.int32,
    )
    target_predict = jnp.array(
        [
            [11, 12, 13, 99],
            [21, 77, 88, 99],
            [55, 31, 32, 33],
        ],
        dtype=jnp.int32,
    )

    accept_lens_out, next_token_ids_flat, new_verified_id, accept_len_draft = (
        dflash_greedy_verify_outputs(draft_token, target_predict)
    )

    # emitted token count per req = accepted drafts + 1 bonus
    np.testing.assert_array_equal(np.asarray(accept_lens_out), np.array([4, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(accept_len_draft), np.array([3, 1, 0], dtype=np.int32))
    # next seed for the following block is the bonus token
    np.testing.assert_array_equal(
        np.asarray(new_verified_id), np.array([99, 77, 55], dtype=np.int32)
    )
    # flat ids are target_predict flattened; scheduler slices the first
    # accept_lens_out entries per req block.
    np.testing.assert_array_equal(
        np.asarray(next_token_ids_flat),
        np.array([11, 12, 13, 99, 21, 77, 88, 99, 55, 31, 32, 33], dtype=np.int32),
    )


def test_gather_dflash_accepted_hidden_padded_matches_valid_concat_layout():
    block_size = 4
    hidden = jnp.arange(3 * block_size * 2, dtype=jnp.float32).reshape(3 * block_size, 2)
    accept_lens = jnp.array([4, 2, 1], dtype=jnp.int32)

    out = gather_dflash_accepted_hidden_padded(
        hidden,
        accept_lens,
        block_size=block_size,
    )

    hidden_np = np.asarray(hidden).reshape(3, block_size, 2)
    expected_valid = np.concatenate(
        [
            hidden_np[0, :4],
            hidden_np[1, :2],
            hidden_np[2, :1],
        ],
        axis=0,
    )
    expected = np.concatenate(
        [
            expected_valid,
            np.repeat(hidden_np[0, 0:1], 3 * block_size - len(expected_valid), axis=0),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_dflash_committed_slices_prefill():
    # prefill: commit whole new-prompt span [prefix_len : prefix_len + extend_len]
    ctx_lens = np.array([3, 2], dtype=np.int32)  # new prompt tokens per req
    draft_seq_lens = np.array([5, 0], dtype=np.int32)  # cached prefix length per req
    starts, lengths = compute_new_kv_slices(ctx_lens, draft_seq_lens, is_prefill=True)
    np.testing.assert_array_equal(starts, np.array([5, 0], dtype=np.int32))
    np.testing.assert_array_equal(lengths, np.array([3, 2], dtype=np.int32))


def test_dflash_committed_slices_decode():
    # decode: commit last accept_len tokens [new_seq_len - accept_len : new_seq_len]
    ctx_lens = np.array([2, 1], dtype=np.int32)  # committed this step per req
    draft_seq_lens = np.array([10, 7], dtype=np.int32)  # new total length per req
    starts, lengths = compute_new_kv_slices(ctx_lens, draft_seq_lens, is_prefill=False)
    np.testing.assert_array_equal(starts, np.array([8, 6], dtype=np.int32))
    np.testing.assert_array_equal(lengths, np.array([2, 1], dtype=np.int32))
