"""Recurrent track metadata must survive the pytree round-trip in lockstep.

The extra-buffer recurrent path publishes a recurrent snapshot only at page/track boundaries.
Two per-request arrays -- recurrent_track_indices / recurrent_track_mask --
ride the same plumbing as recurrent_cow_src_indices from the scheduler through
ModelWorkerBatch -> ForwardBatch -> LinearRecurrentAttnBackendMetadata. These
tests cover both the builder (_recurrent_track_entry / _build_recurrent_track_entries
populate the arrays only on a track boundary, None otherwise) and the wiring:
each array flattens and unflattens to the same field (distinct sentinels catch a
swapped index), and None round-trips to None.
"""

import unittest

import numpy as np

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.managers.schedule_batch import (
    Req,
    _build_recurrent_track_entries,
    _recurrent_track_entry,
)
from sgl_jax.srt.mem_cache.base_prefix_cache import MatchResult
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class TestLinearRecurrentAttnBackendMetadataTrack(unittest.TestCase):
    def test_track_arrays_round_trip_distinct_sentinels(self):
        # Distinct values per field so a swapped flatten/unflatten index is caught.
        md = LinearRecurrentAttnBackendMetadata(
            cu_q_lens=np.array([0, 1], dtype=np.int32),
            recurrent_indices=np.array([10, 11], dtype=np.int32),
            has_initial_state=np.array([1, 0], dtype=np.int32),
            recurrent_track_indices=np.array([20, 21], dtype=np.int32),
            recurrent_track_mask=np.array([1, 0], dtype=np.int32),
        )

        children, aux_data = md.tree_flatten()
        rebuilt = LinearRecurrentAttnBackendMetadata.tree_unflatten(aux_data, children)

        np.testing.assert_array_equal(rebuilt.cu_q_lens, md.cu_q_lens)
        np.testing.assert_array_equal(rebuilt.recurrent_indices, md.recurrent_indices)
        np.testing.assert_array_equal(rebuilt.has_initial_state, md.has_initial_state)
        np.testing.assert_array_equal(
            rebuilt.recurrent_track_indices, np.array([20, 21], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            rebuilt.recurrent_track_mask, np.array([1, 0], dtype=np.int32)
        )

    def test_track_arrays_none_round_trip(self):
        md = LinearRecurrentAttnBackendMetadata()
        rebuilt = LinearRecurrentAttnBackendMetadata.tree_unflatten(*reversed(md.tree_flatten()))

        self.assertIsNone(rebuilt.recurrent_track_indices)
        self.assertIsNone(rebuilt.recurrent_track_mask)


def _minimal_forward_batch(**overrides):
    kwargs = dict(
        bid=0,
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        input_ids=np.array([1, 2], dtype=np.int32),
        req_pool_indices=np.array([0, 1], dtype=np.int32),
        seq_lens=np.array([4, 5], dtype=np.int32),
        out_cache_loc=np.array([0, 1], dtype=np.int32),
    )
    kwargs.update(overrides)
    return ForwardBatch(**kwargs)


class TestForwardBatchTrackPytree(unittest.TestCase):
    def test_track_arrays_round_trip_distinct_sentinels(self):
        fb = _minimal_forward_batch(
            recurrent_indices=np.array([10, 11], dtype=np.int32),
            recurrent_cow_src_indices=np.array([30, 31], dtype=np.int32),
            recurrent_track_indices=np.array([20, 21], dtype=np.int32),
            recurrent_track_mask=np.array([1, 0], dtype=np.int32),
        )

        children, aux_data = fb.tree_flatten()
        rebuilt = ForwardBatch.tree_unflatten(aux_data, children)

        # Existing recurrent fields stay aligned.
        np.testing.assert_array_equal(rebuilt.recurrent_indices, np.array([10, 11]))
        np.testing.assert_array_equal(rebuilt.recurrent_cow_src_indices, np.array([30, 31]))
        # New track fields keep their distinct sentinels (no index swap).
        np.testing.assert_array_equal(
            rebuilt.recurrent_track_indices, np.array([20, 21], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            rebuilt.recurrent_track_mask, np.array([1, 0], dtype=np.int32)
        )

    def test_track_arrays_none_round_trip(self):
        fb = _minimal_forward_batch()

        children, aux_data = fb.tree_flatten()
        rebuilt = ForwardBatch.tree_unflatten(aux_data, children)

        self.assertIsNone(rebuilt.recurrent_track_indices)
        self.assertIsNone(rebuilt.recurrent_track_mask)


class _PingPongPool:
    """Minimal pool exposing the ping-pong flip helper the track builder uses."""

    @staticmethod
    def get_recurrent_ping_pong_other_idx(next_idx: int) -> int:
        return 1 - next_idx


class _FakeReq:
    def __init__(self, *, extend_input_len, buffer, next_idx=0):
        self.extend_input_len = extend_input_len
        self.recurrent_ping_pong_track_buffer = list(buffer)
        self.recurrent_next_track_idx = next_idx
        self.recurrent_last_track_seqlen = None


class TestRecurrentTrackEntryBuilder(unittest.TestCase):
    """Per-req track entries with read-then-flip bookkeeping."""

    def test_extend_on_boundary_reads_then_flips(self):
        # next_idx=0 -> read slot buffer[0]=40, then flip to 1.
        req = _FakeReq(extend_input_len=128, buffer=[40, 41], next_idx=0)
        entry = _recurrent_track_entry(req, 256, interval=128, pool=_PingPongPool(), is_extend=True)
        self.assertTrue(entry.track_mask)
        self.assertEqual(entry.track_index, 40)  # slot BEFORE the flip
        self.assertEqual(req.recurrent_last_track_seqlen, 256)
        self.assertEqual(req.recurrent_next_track_idx, 1)  # flipped once

    def test_extend_off_boundary_dormant_no_flip(self):
        req = _FakeReq(extend_input_len=100, buffer=[40, 41], next_idx=0)
        entry = _recurrent_track_entry(req, 200, interval=128, pool=_PingPongPool(), is_extend=True)
        self.assertFalse(entry.track_mask)
        self.assertEqual(entry.track_index, 0)
        self.assertIsNone(req.recurrent_last_track_seqlen)
        self.assertEqual(req.recurrent_next_track_idx, 0)  # untouched

    def test_extend_zero_input_never_tracks(self):
        # A zero-extend (e.g. already-committed prefix) must not snapshot even
        # if the seqlen happens to land on a boundary.
        req = _FakeReq(extend_input_len=0, buffer=[40, 41], next_idx=0)
        entry = _recurrent_track_entry(req, 256, interval=128, pool=_PingPongPool(), is_extend=True)
        self.assertFalse(entry.track_mask)
        self.assertEqual(req.recurrent_next_track_idx, 0)

    def test_decode_on_boundary_tracks(self):
        # Decode advances by 1; mask is purely seqlen % interval == 0.
        req = _FakeReq(extend_input_len=0, buffer=[40, 41], next_idx=1)
        entry = _recurrent_track_entry(
            req, 128, interval=128, pool=_PingPongPool(), is_extend=False
        )
        self.assertTrue(entry.track_mask)
        self.assertEqual(entry.track_index, 41)  # buffer[next_idx=1] before flip
        self.assertEqual(req.recurrent_next_track_idx, 0)
        self.assertEqual(req.recurrent_last_track_seqlen, 128)

    def test_builder_none_when_no_boundary(self):
        reqs = [
            _FakeReq(extend_input_len=100, buffer=[40, 41]),
            _FakeReq(extend_input_len=50, buffer=[42, 43]),
        ]
        out = _build_recurrent_track_entries(
            reqs, [200, 150], interval=128, pool=_PingPongPool(), is_extend=True
        )
        self.assertEqual(out, (None, None))
        # No req mutated when nothing hit a boundary.
        self.assertTrue(all(r.recurrent_last_track_seqlen is None for r in reqs))

    def test_builder_padded_one_shot_on_boundary(self):
        reqs = [
            _FakeReq(extend_input_len=128, buffer=[40, 41], next_idx=0),  # hits 256
            _FakeReq(extend_input_len=50, buffer=[42, 43], next_idx=0),  # misses
        ]
        indices, mask = _build_recurrent_track_entries(
            reqs, [256, 150], interval=128, pool=_PingPongPool(), is_extend=True
        )
        np.testing.assert_array_equal(mask, np.array([1, 0], dtype=np.int32))
        np.testing.assert_array_equal(indices, np.array([40, 0], dtype=np.int32))
        # Only the hitting req flipped.
        self.assertEqual(reqs[0].recurrent_next_track_idx, 1)
        self.assertEqual(reqs[1].recurrent_next_track_idx, 0)


class _CapturingTreeCache:
    """Records the MatchPrefixParams init_next_round_input builds."""

    disable = False

    class _Root:
        pass

    def __init__(self):
        self.root_node = self._Root()
        self.last_params = None

    def supports_recurrent(self) -> bool:
        return True

    def match_prefix(self, params):
        self.last_params = params
        return MatchResult(
            device_indices=np.empty((0,), dtype=np.int32),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
            host_hit_length=0,
        )


def _bare_req(rid):
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_new_tokens=8),
        eos_token_ids={9},
        vocab_size=32000,
    )


class TestFullOnlyRematch(unittest.TestCase):
    """A running recurrent req re-matches FULL-only, no re-clone."""

    def test_running_recurrent_req_full_only_no_clone(self):
        cache = _CapturingTreeCache()
        req = _bare_req("running")
        req.recurrent_pool_idx = 7  # already owns a running slot
        req.init_next_round_input(cache)
        self.assertTrue(cache.last_params.full_only)
        self.assertFalse(cache.last_params.cow_recurrent)

    def test_new_prefill_clones_and_matches_all_components(self):
        cache = _CapturingTreeCache()
        req = _bare_req("fresh")
        req.recurrent_pool_idx = None  # no running slot yet
        req.init_next_round_input(cache)
        self.assertFalse(cache.last_params.full_only)
        self.assertTrue(cache.last_params.cow_recurrent)


if __name__ == "__main__":
    unittest.main()
