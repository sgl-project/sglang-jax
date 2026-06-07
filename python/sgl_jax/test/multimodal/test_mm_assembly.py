"""Tests for the shared mm_items -> kwargs assembler (numpy-only)."""

import unittest

import numpy as np

from sgl_jax.srt.multimodal.manager.mm_assembly import assemble_mm_inputs


class _Item:
    """Duck-typed stand-in for MultimodalDataItem (assembler only needs these)."""

    def __init__(self, modality, feature, model_specific_data=None):
        self._modality = modality
        self.feature = feature
        self.model_specific_data = model_specific_data or {}

    def is_image(self):
        return self._modality == "image"

    def is_video(self):
        return self._modality == "video"

    def is_audio(self):
        return self._modality == "audio"


class TestAssembleMMInputs(unittest.TestCase):
    def test_empty_and_non_dict(self):
        self.assertEqual(assemble_mm_inputs(None)["pixel_values_images"], None)
        self.assertEqual(assemble_mm_inputs({})["audio_codes"], None)

    def test_image_video_concat(self):
        mm = {
            "mm_items": [
                _Item("image", np.ones((2, 4))),
                _Item("image", np.ones((3, 4))),
                _Item("video", np.zeros((5, 4))),
            ]
        }
        out = assemble_mm_inputs(mm)
        self.assertEqual(out["pixel_values_images"].shape, (5, 4))
        self.assertEqual(out["pixel_values_videos"].shape, (5, 4))
        self.assertIsNone(out["audio_codes"])
        self.assertIsNone(out["audio_features"])

    def test_codes_audio_routes_to_audio_codes(self):
        mm = {
            "mm_items": [
                _Item("audio", np.zeros((8, 20)), {"is_codes": True, "token_lengths": [2]}),
            ]
        }
        out = assemble_mm_inputs(mm)
        self.assertEqual(out["audio_codes"].shape, (8, 20))
        self.assertIsNone(out["audio_features"])

    def test_continuous_audio_routes_to_audio_features(self):
        mm = {
            "mm_items": [_Item("audio", np.zeros((1, 128, 4)))],
            "audio_feature_attention_mask": np.ones((1, 4)),
        }
        out = assemble_mm_inputs(mm)
        self.assertIsNone(out["audio_codes"])
        self.assertEqual(out["audio_features"].shape, (1, 128, 4))
        self.assertEqual(out["audio_feature_attention_mask"].shape, (1, 4))

    def test_grids_passed_through(self):
        mm = {"mm_items": [], "image_grid_thw": [(1, 2, 2)], "video_grid_thw": [(1, 4, 4)]}
        out = assemble_mm_inputs(mm)
        self.assertEqual(out["image_grid_thw"], [(1, 2, 2)])
        self.assertEqual(out["video_grid_thw"], [(1, 4, 4)])


if __name__ == "__main__":
    unittest.main()
