import unittest

import numpy as np

from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor


class TestQwenVLProcessor(unittest.TestCase):
    def test_compute_image_offsets_uses_expanded_image_token_spans(self):
        image_token_id = 151655
        input_ids = [
            1,
            2,
            image_token_id,
            image_token_id,
            3,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            4,
        ]
        grids = [(1, 2, 4), (1, 4, 4)]

        offsets = QwenVLProcessor._compute_image_offsets(
            input_ids=input_ids,
            grids=grids,
            image_token_id=image_token_id,
            spatial_merge_size=2,
        )

        self.assertEqual(offsets, [(2, 3), (5, 8)])

    def test_build_items_attaches_per_image_offsets(self):
        features = np.arange(24).reshape(24, 1)
        grids = [(1, 2, 4), (1, 4, 4)]
        offsets = [(2, 3), (5, 8)]

        items = QwenVLProcessor._build_items(features, grids, offsets)

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].offsets, [(2, 3)])
        self.assertEqual(items[1].offsets, [(5, 8)])
        self.assertEqual(items[0].feature.shape, (8, 1))
        self.assertEqual(items[1].feature.shape, (16, 1))


if __name__ == "__main__":
    unittest.main()
