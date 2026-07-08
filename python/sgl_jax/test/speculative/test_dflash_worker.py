"""CPU unit tests for DFlashWorker state-advance helpers.

The full worker construction needs real target/draft ModelWorkers (TPU); these
tests exercise the pure array helpers by instantiating the worker via ``__new__``
and injecting the minimal attributes each helper reads. End-to-end draft/verify
is validated by the TPU v6e smoke command in ``docs/design/dflash_stage_c.md``.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.dflash_info import DFlashDraftInput
from sgl_jax.srt.speculative.dflash_worker import DFlashWorker


def _bare_worker(**attrs):
    w = object.__new__(DFlashWorker)
    for k, v in attrs.items():
        object.__setattr__(w, k, v)
    return w


def test_greedy_sample_from_head_picks_argmax():
    # lm_head: [vocab=4, hidden=3]. Row v is a one-hot at dim v%3 scaled by v+1,
    # so hidden aligned to a given row maximizes that row's logit.
    lm_head = jnp.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    w = _bare_worker(_target_lm_head=lm_head)
    hidden = jnp.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=jnp.float32)
    out = np.asarray(w._greedy_sample_from_head(hidden))
    np.testing.assert_array_equal(out, np.array([0, 1], dtype=np.int32))


def test_slice_committed_target_hidden_gathers_front_of_block():
    # bs=2, block_size=3, feat=2. Commit 2 tokens for req0, 1 for req1.
    block_size = 3
    w = _bare_worker(block_size=block_size)
    hs = jnp.arange(2 * block_size * 2, dtype=jnp.float32).reshape(2 * block_size, 2)
    accept_lens = np.array([2, 1], dtype=np.int32)
    out = np.asarray(w._slice_committed_target_hidden(hs, accept_lens, bs=2))
    # req0 rows 0,1 ; req1 row 3 (block start = 3)
    expected = np.stack(
        [np.asarray(hs)[0], np.asarray(hs)[1], np.asarray(hs)[3]], axis=0
    )
    np.testing.assert_array_equal(out, expected)


def test_committed_positions_prefill_and_decode():
    w = _bare_worker()

    prefill_di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([3, 2], dtype=np.int32),
        draft_seq_lens=np.array([5, 0], dtype=np.int32),
    )
    pos = w._committed_positions(prefill_di, is_prefill=True)
    np.testing.assert_array_equal(pos, np.array([5, 6, 7, 0, 1], dtype=np.int32))

    decode_di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([2, 1], dtype=np.int32),
        draft_seq_lens=np.array([10, 7], dtype=np.int32),
    )
    pos = w._committed_positions(decode_di, is_prefill=False)
    np.testing.assert_array_equal(pos, np.array([8, 9, 6], dtype=np.int32))


def test_committed_cache_loc_reads_req_to_token():
    # req_to_token[req_pool_index, position] -> global cache slot.
    req_to_token = np.arange(100, dtype=np.int32).reshape(5, 20)
    draft_model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
    )
    w = _bare_worker(draft_model_runner=draft_model_runner)

    mwb = SimpleNamespace(req_pool_indices=np.array([1, 3], dtype=np.int32))
    di = DFlashDraftInput(
        verified_id=np.array([0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([2, 1], dtype=np.int32),
        draft_seq_lens=np.array([4, 2], dtype=np.int32),  # decode: start = len - ctx
    )
    locs = w._committed_cache_loc(mwb, di, is_prefill=False)
    # req1: positions [2,3] -> req_to_token[1, 2:4] = [22, 23]
    # req3: position  [1]   -> req_to_token[3, 1:2] = [61]
    np.testing.assert_array_equal(locs, np.array([22, 23, 61], dtype=np.int32))


def test_pack_cache_loc_rows_uses_bucket_stable_row_width():
    w = _bare_worker(page_size=1)
    rows = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6, 7, 8], dtype=np.int32),
    ]

    packed = w._pack_cache_loc_rows(rows, [len(r) for r in rows])

    assert packed.shape == (32,)
    np.testing.assert_array_equal(packed[:5], np.array([1, 2, 3, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(packed[16:21], np.array([4, 5, 6, 7, 8], dtype=np.int32))
