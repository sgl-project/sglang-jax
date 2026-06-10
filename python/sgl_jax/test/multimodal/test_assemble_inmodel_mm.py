"""M3 regression: ScheduleBatch._assemble_inmodel_mm raw-mm assembly + pad ordering.

Requires the jax stack (imports ScheduleBatch). The helper is called unbound (self=None)
since it touches no instance state. Fake duck-typed items satisfy both
assemble_mm_inputs's surface (is_image/is_video/is_audio/feature/model_specific_data) and
the helper's pad routing (set_pad_value/pad_value).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch


class _FakeItem:
    def __init__(self, kind, feature, pad_value=None, model_specific_data=None):
        self._kind = kind
        self.feature = feature
        self.pad_value = pad_value
        self.model_specific_data = model_specific_data or {}

    def is_image(self):
        return self._kind == "image"

    def is_video(self):
        return self._kind == "video"

    def is_audio(self):
        return self._kind == "audio"

    def set_pad_value(self):
        if self.pad_value is None:
            self.pad_value = 999  # sentinel proving the fallback path ran


def _req(mm_inputs=None, multimodal_embedding=None):
    return SimpleNamespace(mm_inputs=mm_inputs, multimodal_embedding=multimodal_embedding)


class TestAssembleInModelMM(unittest.TestCase):
    assemble = staticmethod(ScheduleBatch._assemble_inmodel_mm)  # unbound

    def test_text_only_all_none(self):
        self.assertEqual(self.assemble(None, [_req(), _req()]), (None,) * 8)

    def test_staged_precomputed_embedding_excluded(self):
        staged = _req(
            mm_inputs={
                "mm_items": [_FakeItem("image", np.zeros((4, 2)))],
                "image_grid_thw": [[1, 2, 2]],
            },
            multimodal_embedding=object(),
        )
        self.assertEqual(self.assemble(None, [staged]), (None,) * 8)

    def test_single_image(self):
        img = _FakeItem("image", np.ones((4, 2), np.float32), pad_value=1000007)
        px, pxv, gthw, vthw, pads, visual_pads, aud, aud_lens = self.assemble(
            None, [_req(mm_inputs={"mm_items": [img], "image_grid_thw": [[1, 2, 2]]})]
        )
        self.assertIsNotNone(px)
        self.assertEqual(px.shape, (4, 2))
        self.assertIsNone(pxv)
        self.assertEqual(gthw, ((1, 2, 2),))
        self.assertIsNone(vthw)
        self.assertEqual(pads, (1000007,))
        self.assertEqual(visual_pads, (1000007,))  # image-only: visual == all pad_values
        self.assertIsNone(aud)
        self.assertIsNone(aud_lens)

    def test_image_video_pad_ordering(self):
        # Video listed FIRST -> pads must come out image-then-video (matches the model
        # forward's encode_image->encode_video concatenation and merge() keying).
        vid = _FakeItem("video", np.ones((6, 2), np.float32), pad_value=2000009)
        img = _FakeItem("image", np.ones((4, 2), np.float32), pad_value=1000007)
        px, pxv, gthw, vthw, pads, visual_pads, aud, aud_lens = self.assemble(
            None,
            [
                _req(
                    mm_inputs={
                        "mm_items": [vid, img],
                        "image_grid_thw": [[1, 2, 2]],
                        "video_grid_thw": [[1, 2, 3]],
                    }
                )
            ],
        )
        self.assertEqual(px.shape, (4, 2))  # image feats
        self.assertEqual(pxv.shape, (6, 2))  # video feats
        self.assertEqual(gthw, ((1, 2, 2),))
        self.assertEqual(vthw, ((1, 2, 3),))
        self.assertEqual(pads, (1000007, 2000009))  # image pad first, video pad second

    def test_set_pad_value_fallback(self):
        img = _FakeItem("image", np.ones((4, 2), np.float32), pad_value=None)
        out = self.assemble(
            None, [_req(mm_inputs={"mm_items": [img], "image_grid_thw": [[1, 2, 2]]})]
        )
        self.assertEqual(out[4], (999,))
        self.assertEqual(img.pad_value, 999)

    def test_multi_req_concat_and_global_pad_order(self):
        r_a = _req(
            mm_inputs={
                "mm_items": [_FakeItem("image", np.ones((4, 2), np.float32), pad_value=11)],
                "image_grid_thw": [[1, 2, 2]],
            }
        )
        r_b = _req(
            mm_inputs={
                "mm_items": [_FakeItem("image", np.ones((2, 2), np.float32), pad_value=22)],
                "image_grid_thw": [[1, 1, 2]],
            }
        )
        px, pxv, gthw, vthw, pads, visual_pads, aud, aud_lens = self.assemble(None, [r_a, r_b])
        self.assertEqual(px.shape, (6, 2))  # 4 + 2 rows concatenated
        self.assertEqual(gthw, ((1, 2, 2), (1, 1, 2)))
        self.assertEqual(pads, (11, 22))

    def test_audio_only(self):
        # Continuous-mel audio item -> audio_features [f, t] + per-audio mel length; audio
        # pad_value is in pad_values (for merge) but NOT in visual_pad_values (deepstack).
        aud = _FakeItem("audio", np.ones((8, 5), np.float32), pad_value=3000001)
        px, pxv, gthw, vthw, pads, visual_pads, aud_feats, aud_lens = self.assemble(
            None,
            [_req(mm_inputs={"mm_items": [aud], "audio_feature_attention_mask": np.ones((1, 5))})],
        )
        self.assertIsNone(px)
        self.assertIsNone(pxv)
        self.assertIsNotNone(aud_feats)
        self.assertEqual(aud_feats.shape, (8, 5))  # [f, t]
        self.assertEqual(aud_lens, (5,))  # mel length = mask.sum()
        self.assertEqual(pads, (3000001,))  # audio pad in all pad_values
        self.assertIsNone(visual_pads)  # deepstack excludes audio


if __name__ == "__main__":
    unittest.main()
