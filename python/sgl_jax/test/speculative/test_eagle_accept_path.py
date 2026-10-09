"""After an EAGLE verify: compacting the accepted tree path, and emitting it.

An accepted tree path skips siblings, so its nodes are not a prefix of the
flat verify window. Both the KV behind each position and the emitted tokens
must follow the path, not the window prefix. A chain is the identity case.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.eagle_info import EagleDraftInput
from sgl_jax.srt.speculative.eagle_util import (
    accepted_path_kv_copies,
    compact_accepted_paths,
    front_pack_accepted_tokens,
)
from sgl_jax.srt.speculative.overlap_utils import resolve_spec_decode_token_ids
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

N = 6  # verify draft tokens per request
WIDTH = 4  # accept window: speculative_num_steps + 1


def _pool():
    """Two requests in pool rows 2 and 0, mirrored into a cache_loc snapshot."""
    req_to_token = np.zeros((4, 64), dtype=np.int32)
    req_to_token[2] = 100 + np.arange(64)
    req_to_token[0] = 500 + np.arange(64)
    req_pool_indices = np.array([2, 0], dtype=np.int32)
    cache_loc_starts = np.array([0, 40, -1])
    cache_loc = np.zeros(128, dtype=np.int32)
    cache_loc[0:40] = req_to_token[2, :40]
    cache_loc[40:80] = req_to_token[0, :40]
    window_starts = np.array([10, 20, 0])
    return req_to_token, req_pool_indices, cache_loc, cache_loc_starts, window_starts


def _accept_index(paths):
    """`(padded_bs, WIDTH)` flat node ids; padding slots get an empty path."""
    out = np.full((len(paths), WIDTH), -1, dtype=np.int32)
    for s, path in enumerate(paths):
        out[s, : len(path)] = s * N + np.asarray(path)
    return out


def _compact(paths, pool=None):
    req_to_token, req_pool_indices, cache_loc, cache_loc_starts, window_starts = pool or _pool()
    compact_accepted_paths(
        req_to_token,
        req_pool_indices,
        cache_loc,
        cache_loc_starts,
        window_starts,
        _accept_index(paths),
        np.array([0, 1]),
        N,
    )
    return req_to_token, cache_loc


def test_tree_path_moves_to_the_window_front():
    before, before_loc = _pool()[0].copy(), _pool()[2].copy()
    after, after_loc = _compact([[0, 2, 5], [0, 1, 3], []])

    # Slot 0: pool row 2, window [10, 16), cache_loc segment at 0.
    np.testing.assert_array_equal(after[2, 10:16], before[2, 10:16][[0, 2, 5, 1, 3, 4]])
    # Slot 1: pool row 0, window [20, 26), cache_loc segment at 40.
    np.testing.assert_array_equal(after[0, 20:26], before[0, 20:26][[0, 1, 3, 2, 4, 5]])
    np.testing.assert_array_equal(after_loc[10:16], after[2, 10:16])
    np.testing.assert_array_equal(after_loc[60:66], after[0, 20:26])

    # Same slots per window; nothing outside the windows moves.
    for row, start in ((2, 10), (0, 20)):
        assert sorted(after[row, start : start + N]) == sorted(before[row, start : start + N])
    outside = np.ones_like(before, dtype=bool)
    outside[2, 10:16] = outside[0, 20:26] = False
    np.testing.assert_array_equal(after[outside], before[outside])
    outside_loc = np.ones_like(before_loc, dtype=bool)
    outside_loc[10:16] = outside_loc[60:66] = False
    np.testing.assert_array_equal(after_loc[outside_loc], before_loc[outside_loc])


def test_chain_path_is_left_alone():
    before, _, before_loc, _, _ = _pool()
    after, after_loc = _compact([[0, 1, 2], [0, 1, 2, 3], []])
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(after_loc, before_loc)


def test_cache_loc_must_mirror_the_page_table():
    pool = list(_pool())
    pool[2][12] += 1  # cache_loc drifted from req_to_token inside slot 0's window
    with pytest.raises(AssertionError, match="does not mirror"):
        _compact([[0, 2, 5], [0, 1, 3], []], pool)


def test_repeated_node_is_not_a_path():
    with pytest.raises(AssertionError, match="not a path"):
        _compact([[0, 2, 2], [0, 1], []])


def _kv_pool(page_size, tp=1, size=640, layer_num=2):
    """A KV pool whose every row holds ``slot + 1000 * layer``."""
    mesh = Mesh(
        np.array(jax.devices()[:tp]).reshape(1, tp),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    pool = MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=jnp.float32,
        head_num=tp,
        head_dim=128,
        layer_num=layer_num,
        mesh=mesh,
    )
    for layer, kv in enumerate(pool.kv_buffer):
        values = np.arange(kv.shape[0] * kv.shape[1]).reshape(kv.shape[:2]) + 1000 * layer
        filled = np.broadcast_to(values[:, :, None, None, None], kv.shape).astype(np.float32)
        pool.kv_buffer[layer] = jax.device_put(filled, pool.kv_sharding)
    return pool


def _rows(pool, slots):
    """``(layer_num, len(slots))``: which slot's original KV each slot now holds."""
    out = []
    for layer, kv in enumerate(pool.kv_buffer):
        kv = np.asarray(kv)
        rows = kv.reshape(-1, *kv.shape[2:])[np.asarray(slots)]
        assert np.all(rows == rows[:, :1, :1, :1]), "a row mixes KV from several slots"
        out.append(rows[:, 0, 0, 0] - 1000 * layer)
    return np.stack(out).astype(np.int64)


@pytest.mark.parametrize(
    "tp",
    [
        1,
        pytest.param(4, marks=pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 devices")),
    ],
)
def test_kv_rows_are_copied_all_at_once(tp):
    pool = _kv_pool(page_size=8, tp=tp)
    num_slots = pool.kv_buffer[0].shape[0] * pool.kv_buffer[0].shape[1]
    # 11 and 20 are both read and written; 0 -> 0 is padding.
    pool.copy_kv_rows(np.array([10, 11, 20, 0]), np.array([11, 20, 5, 0]))

    expected = np.arange(num_slots)
    expected[[11, 20, 5]] = [10, 11, 20]
    np.testing.assert_array_equal(_rows(pool, np.arange(num_slots)), [expected, expected])


def test_chain_path_needs_no_kv_copies():
    req_to_token, req_pool_indices, _, _, window_starts = _pool()
    src, dst = accepted_path_kv_copies(
        req_to_token,
        req_pool_indices,
        window_starts,
        _accept_index([[0, 1, 2], [0, 1, 2, 3], []]),
        np.array([0, 1]),
        N,
    )
    assert src.size == dst.size == 0


@pytest.mark.parametrize("page_size", [1, 8])
def test_accepted_path_kv_lands_at_the_window_front(page_size):
    """Pointer compaction (page_size 1) and KV copies (paged) agree."""
    paths = [[0, 2, 5], [0, 1, 3], []]
    req_to_token, req_pool_indices, cache_loc, cache_loc_starts, window_starts = _pool()
    before = req_to_token.copy()
    pool = _kv_pool(page_size)
    worker = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        page_size=page_size,
        speculative_num_draft_tokens=N,
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(token_to_kv_pool=pool)),
    )
    mwb = SimpleNamespace(
        req_pool_indices=req_pool_indices,
        cache_loc=cache_loc,
        draft_cache_loc_starts=cache_loc_starts,
        seq_lens=window_starts,
        logits_indices_selector=np.array([0, 1]),
    )
    BaseSpecWorker._move_accepted_paths_to_front(worker, mwb, _accept_index(paths))

    for s, path in enumerate(paths[:2]):
        req, start = req_pool_indices[s], window_starts[s]
        committed = req_to_token[req, : start + len(path)]
        expected = np.concatenate([before[req, :start], before[req, start + np.asarray(path)]])
        np.testing.assert_array_equal(_rows(pool, committed), [expected, expected])
    if page_size > 1:
        # Paged tables stay page-contiguous: only the KV moves.
        np.testing.assert_array_equal(req_to_token, before)


@pytest.mark.parametrize("draft_token_num", [N, 3])
def test_emitted_tokens_follow_the_accepted_path(draft_token_num):
    """verify publishes, the scheduler resolves: the two strides must agree."""
    paths = [[0, 2], [0, 1, 2], [0]]
    accept_lens = np.array([len(p) for p in paths])
    # The target's prediction at node k of slot s; the accepted path's
    # predictions, in order, are the tokens to emit.
    predict = 1000 * (np.arange(len(paths))[:, None] + 1) + np.arange(draft_token_num)
    verified_id = np.zeros((len(paths), WIDTH), dtype=np.int32)
    for s, path in enumerate(paths):
        verified_id[s, : len(path)] = predict[s, path]

    emitted = front_pack_accepted_tokens(verified_id.reshape(-1), WIDTH, draft_token_num)
    batch = SimpleNamespace(
        per_dp_bs_size=len(paths),
        dp_size=1,
        reqs_info=[SimpleNamespace(reqs=[object()] * len(paths))],
    )
    tokens, lens = resolve_spec_decode_token_ids(
        SimpleNamespace(next_token_ids=emitted, accept_lens=accept_lens), batch, draft_token_num
    )
    assert lens == accept_lens.tolist()
    assert tokens == [predict[s, path].tolist() for s, path in enumerate(paths)]


def _extend_inputs(bs, draft_token_num):
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
    mwb = SimpleNamespace(
        seq_lens=np.array([9, 14, 0], dtype=np.int32)[:bs],
        logits_indices_selector=np.array([0, 1]),
        input_ids=np.zeros(bs * draft_token_num, dtype=np.int32),
        positions=np.zeros(bs * draft_token_num, dtype=np.int32),
        dp_size=1,
        per_dp_bs_size=bs,
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        capture_hidden_mode=CaptureHiddenMode.LAST,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_input_logprob_token_ids=None,
    )
    verified_id = np.arange(bs * WIDTH, dtype=np.int32)
    batch_output = SimpleNamespace(
        accept_lens=np.array([2, 3, 0]),
        next_draft_input=SimpleNamespace(
            hidden_states=np.zeros((bs * WIDTH, 8)), verified_id=verified_id, positions=None
        ),
    )
    runner = SimpleNamespace(
        mesh=mesh,
        attn_backend=SimpleNamespace(get_eagle_forward_metadata=lambda batch: "metadata"),
    )
    return mwb, batch_output, runner


def test_draft_extend_feeds_the_accept_window():
    """Each request's draft extend reads WIDTH tokens of verified_id."""
    mwb, batch_output, runner = _extend_inputs(3, N)
    mwb, _ = EagleDraftInput().prepare_for_extend_after_verify(
        mwb, runner, batch_output, N, accept_width=WIDTH
    )
    np.testing.assert_array_equal(mwb.extend_seq_lens, [WIDTH, WIDTH, 0])
    np.testing.assert_array_equal(mwb.seq_lens, [9 + WIDTH - 1, 14 + WIDTH - 1, 0])
    np.testing.assert_array_equal(mwb.logits_indices, np.cumsum([WIDTH, WIDTH, 0]) - 1)
    assert mwb.input_ids.shape == (3 * WIDTH,)


def test_draft_extend_without_accept_width_follows_input_ids():
    mwb, batch_output, runner = _extend_inputs(3, N)
    mwb, _ = EagleDraftInput().prepare_for_extend_after_verify(mwb, runner, batch_output, N)
    np.testing.assert_array_equal(mwb.extend_seq_lens, [N, N, 0])
    np.testing.assert_array_equal(mwb.seq_lens, [9 + N - 1, 14 + N - 1, 0])
