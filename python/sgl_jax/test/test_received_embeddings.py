import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
    ReceivedEmbeddings,
    release_received_embeddings,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.processors.encoder import DisaggregatedInputMixin


class _Processor(DisaggregatedInputMixin):
    @staticmethod
    def _to_grid_list(value):
        return [] if value is None else np.asarray(value).reshape(-1, 3).tolist()


class TestReceivedEmbeddings(unittest.TestCase):
    def setUp(self):
        self.buffer = jnp.arange(32).reshape(8, 4)
        self.session = Mock()
        self.received = ReceivedEmbeddings(self.buffer, np.asarray([6, 7, 2, 3]), 4, [self.session])

    def test_row_slices_and_concatenation_preserve_buffer_and_ownership(self):
        first = self.received.slice_rows(0, 2)
        second = self.received.slice_rows(2, 4)
        self.assertTrue(np.shares_memory(first.row_indices, self.received.row_indices))
        combined = ReceivedEmbeddings.concatenate([second, first])
        self.assertIs(combined.buffer, self.buffer)
        np.testing.assert_array_equal(combined.row_indices, [2, 3, 6, 7])
        self.assertEqual(combined.sessions, [self.session])
        self.session.release.assert_not_called()

    def test_concatenation_rejects_different_buffers_and_widths(self):
        for other in [
            ReceivedEmbeddings(jnp.zeros_like(self.buffer), np.asarray([0]), 4, []),
            ReceivedEmbeddings(self.buffer, np.asarray([0]), 3, []),
        ]:
            with self.assertRaisesRegex(ValueError, "share one pool and width"):
                ReceivedEmbeddings.concatenate([self.received, other])

    def test_parts_keep_order_and_release_sessions_before_admission(self):
        accumulator = MultiModalEmbeddingData(2)
        other_session = Mock()
        second = ReceivedEmbeddings(self.buffer, np.asarray([1]), 4, [other_session])
        for index, embedding in [(1, second), (0, self.received)]:
            accumulator.add(EmbeddingData("req", 2, index, Modality.IMAGE), embedding)
        combined = accumulator.get_embedding()[Modality.IMAGE]
        np.testing.assert_array_equal(combined.row_indices, [6, 7, 2, 3, 1])
        self.assertEqual(combined.sessions, [self.session, other_session])
        accumulator.release()
        self.session.release.assert_called_once()
        other_session.release.assert_called_once()

    def test_reconstruction_preserves_received_rows_and_positions(self):
        processor = _Processor()
        processor.hf_config = SimpleNamespace(
            image_token_id=10, vision_config=SimpleNamespace(spatial_merge_size=1)
        )
        metadata = dict(
            image_grid_thw=[[1, 1, 2], [1, 1, 2]],
            item_hashes={Modality.IMAGE: [100, 200]},
        )
        prompt = [1, 10, 2, 10, 3]
        inputs = processor.get_mm_data(prompt, {Modality.IMAGE: self.received}, **metadata)
        self.assertEqual(inputs.input_ids, [1, 10, 10, 2, 10, 10, 3])
        self.assertEqual(inputs.radix_input_ids, [1, -101, -101, 2, -201, -201, 3])
        for item, rows, span in zip(inputs.mm_items, [[6, 7], [2, 3]], [(1, 3), (4, 6)]):
            received = item.precomputed_embeddings
            self.assertIs(received.buffer, self.buffer)
            self.assertEqual(item.placeholder_ranges, [span])
            np.testing.assert_array_equal(received.row_indices, rows)
        release_received_embeddings(inputs)
        self.assertTrue(all(item.precomputed_embeddings is None for item in inputs.mm_items))
        calls = self.session.release.call_count
        release_received_embeddings(inputs)
        self.assertEqual(self.session.release.call_count, calls)

    def test_reconstruction_rejects_missing_or_extra_rows(self):
        processor = _Processor()
        processor.hf_config = SimpleNamespace(
            image_token_id=10, vision_config=SimpleNamespace(spatial_merge_size=1)
        )
        for count, error in [(5, "incomplete"), (3, "unused")]:
            with self.assertRaisesRegex(ValueError, error):
                processor.get_mm_data(
                    [10],
                    {Modality.IMAGE: self.received},
                    image_grid_thw=[[1, 1, count]],
                    item_hashes={Modality.IMAGE: [100]},
                )


if __name__ == "__main__":
    unittest.main()
