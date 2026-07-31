import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashVerifyInput,
    build_dflash_draft_block,
)
from sgl_jax.srt.speculative.overlap_utils import (
    can_merge_spec_non_overlap_prefill,
    uses_host_eagle_state,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


def test_dflash_verify_input_pytree_round_trip():
    vi = DFlashVerifyInput(
        draft_token=jnp.arange(8, dtype=jnp.int32),
        draft_token_num=4,
    )

    leaves, treedef = jax.tree_util.tree_flatten(vi)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, DFlashVerifyInput)
    assert restored.draft_token_num == 4


def test_dflash_draft_input_filter_batch():
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 30], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([1, 2, 3], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7], dtype=np.int32),
    )

    di.filter_batch(np.array([2, 0], dtype=np.int32), has_been_filtered=False)

    np.testing.assert_array_equal(di.verified_id, np.array([30, 10], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([7, 5], dtype=np.int32))


def test_dflash_draft_input_trim_to_length():
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 30], dtype=np.int32),
        target_hidden=jnp.arange(12, dtype=jnp.float32).reshape(3, 4),
        ctx_lens=np.array([1, 2, 3], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7], dtype=np.int32),
        allocate_lens=np.array([8, 9, 10], dtype=np.int32),
        reservation_base_lens=np.array([4, 5, 6], dtype=np.int32),
    )

    di.trim_to_length(2)

    np.testing.assert_array_equal(di.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([5, 6], dtype=np.int32))
    np.testing.assert_array_equal(di.allocate_lens, np.array([8, 9], dtype=np.int32))
    np.testing.assert_array_equal(
        di.reservation_base_lens,
        np.array([4, 5], dtype=np.int32),
    )
    assert di.target_hidden.shape == (2, 4)


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


def test_dflash_draft_input_aligns_dp_ranks_without_cross_rank_truncation():
    class Req:
        def __init__(self, token, committed):
            self.origin_input_ids = [token]
            self.output_ids = []
            self.kv_committed_len = committed

    rank0 = DFlashDraftInput(
        verified_id=np.array([10], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=np.array([5], dtype=np.int32),
        block_size=16,
    )
    rank1 = DFlashDraftInput(
        verified_id=np.array([20], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=np.array([7], dtype=np.int32),
        block_size=16,
    )
    flat = DFlashDraftInput(
        verified_id=np.array([10, 20], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 7], dtype=np.int32),
        block_size=16,
    )
    schedule_batch = type(
        "Batch",
        (),
        {
            "reqs_info": [
                type("Info", (), {"reqs": [Req(10, 5)], "spec_info": rank0})(),
                type("Info", (), {"reqs": [Req(20, 7)], "spec_info": rank1})(),
            ]
        },
    )()

    flat._align_dp_state_to_reqs(schedule_batch)

    np.testing.assert_array_equal(flat.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(flat.ctx_lens, np.array([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(flat.draft_seq_lens, np.array([5, 7], dtype=np.int32))


def test_dflash_dp_scatter_rejects_incomplete_state():
    incomplete = DFlashDraftInput(
        verified_id=np.array([10], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=None,
    )

    with np.testing.assert_raises_regex(ValueError, "draft_seq_lens.*missing"):
        ScheduleBatch._scatter_spec_info_to_dp_slots(
            incomplete,
            selector=np.array([0], dtype=np.int32),
            total_bs=2,
        )


def test_dflash_concat_normalizes_empty_and_none_target_hidden():
    rank0 = DFlashDraftInput(
        verified_id=np.array([10], dtype=np.int32),
        target_hidden=jnp.zeros((0, 8), dtype=jnp.bfloat16),
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=np.array([5], dtype=np.int32),
    )
    rank1 = DFlashDraftInput(
        verified_id=np.array([20], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0], dtype=np.int32),
        draft_seq_lens=np.array([7], dtype=np.int32),
    )

    flat = ScheduleBatch._concat_spec_info_per_rank([rank0, rank1])

    assert flat.target_hidden is None
    np.testing.assert_array_equal(flat.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(flat.draft_seq_lens, np.array([5, 7], dtype=np.int32))


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


def test_dflash_draft_input_dp_scatter_and_compact_split_round_trip():
    compact = DFlashDraftInput(
        verified_id=np.array([10, 20, 30], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7], dtype=np.int32),
        block_size=4,
    )
    selector = np.array([0, 1, 3], dtype=np.int32)  # rank0: 2/3, rank1: 1/3

    padded = ScheduleBatch._scatter_spec_info_to_dp_slots(
        compact,
        selector=selector,
        total_bs=6,
    )
    np.testing.assert_array_equal(
        padded.verified_id,
        np.array([10, 20, 0, 30, 0, 0], dtype=np.int32),
    )

    # The worker compacts verify output with the same selector before the
    # scheduler stores per-rank cross-round state.
    compact_again = DFlashDraftInput(
        verified_id=np.asarray(padded.verified_id)[selector],
        target_hidden=None,
        ctx_lens=np.asarray(padded.ctx_lens)[selector],
        draft_seq_lens=np.asarray(padded.draft_seq_lens)[selector],
        block_size=4,
    )
    rank0, rank1 = ScheduleBatch._split_spec_info_per_rank(compact_again, [2, 1])
    np.testing.assert_array_equal(rank0.verified_id, np.array([10, 20], dtype=np.int32))
    np.testing.assert_array_equal(rank1.verified_id, np.array([30], dtype=np.int32))


def test_dflash_non_overlap_can_merge_without_host_eagle_accounting():
    assert can_merge_spec_non_overlap_prefill(False, SpeculativeAlgorithm.DFLASH)
    assert not uses_host_eagle_state(False, SpeculativeAlgorithm.DFLASH)


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
