"""Recurrent track metadata must survive the pytree round-trip in lockstep.

Stage 2 (PR#2) publishes a recurrent snapshot only at page/track boundaries.
Three new per-request arrays -- recurrent_track_indices / recurrent_track_mask /
recurrent_track_seqlens -- ride the same plumbing as recurrent_cow_src_indices
from the scheduler through ModelWorkerBatch -> ForwardBatch ->
LinearRecurrentAttnBackendMetadata. They are dormant for now (no builder fills
them), so they stay None end-to-end. These tests pin the wiring: each array
flattens and unflattens to the same field (distinct sentinels catch a swapped
index), and None round-trips to None.
"""

import unittest

import numpy as np

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class TestLinearRecurrentAttnBackendMetadataTrack(unittest.TestCase):
    def test_track_arrays_round_trip_distinct_sentinels(self):
        # Distinct values per field so a swapped flatten/unflatten index is caught.
        md = LinearRecurrentAttnBackendMetadata(
            cu_q_lens=np.array([0, 1], dtype=np.int32),
            recurrent_indices=np.array([10, 11], dtype=np.int32),
            has_initial_state=np.array([1, 0], dtype=np.int32),
            recurrent_track_indices=np.array([20, 21], dtype=np.int32),
            recurrent_track_mask=np.array([1, 0], dtype=np.int32),
            recurrent_track_seqlens=np.array([128, 256], dtype=np.int32),
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
        np.testing.assert_array_equal(
            rebuilt.recurrent_track_seqlens, np.array([128, 256], dtype=np.int32)
        )

    def test_track_arrays_none_round_trip(self):
        md = LinearRecurrentAttnBackendMetadata()
        rebuilt = LinearRecurrentAttnBackendMetadata.tree_unflatten(*reversed(md.tree_flatten()))

        self.assertIsNone(rebuilt.recurrent_track_indices)
        self.assertIsNone(rebuilt.recurrent_track_mask)
        self.assertIsNone(rebuilt.recurrent_track_seqlens)


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
            recurrent_track_seqlens=np.array([128, 256], dtype=np.int32),
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
        np.testing.assert_array_equal(
            rebuilt.recurrent_track_seqlens, np.array([128, 256], dtype=np.int32)
        )

    def test_track_arrays_none_round_trip(self):
        fb = _minimal_forward_batch()

        children, aux_data = fb.tree_flatten()
        rebuilt = ForwardBatch.tree_unflatten(aux_data, children)

        self.assertIsNone(rebuilt.recurrent_track_indices)
        self.assertIsNone(rebuilt.recurrent_track_mask)
        self.assertIsNone(rebuilt.recurrent_track_seqlens)


if __name__ == "__main__":
    unittest.main()
