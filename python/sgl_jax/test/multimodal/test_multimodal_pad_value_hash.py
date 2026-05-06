import base64
import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    pad_input_tokens,
)
from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import MultimodalTokenizer


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 8), color=color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class TestMultimodalPadValueHash(unittest.TestCase):
    def test_precomputed_hash_skips_feature_hash(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=123456789,
            feature=np.ones((4, 4), dtype=np.float32),
        )

        with patch("sgl_jax.srt.multimodal.common.modality_enum.hash_feature") as hash_feature:
            item.set_pad_value()

        hash_feature.assert_not_called()
        self.assertEqual(item.pad_value, 123456789 % (1 << 24))

    def test_set_pad_value_falls_back_to_feature_hash(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=np.ones((4, 4), dtype=np.float32),
        )

        with patch(
            "sgl_jax.srt.multimodal.common.modality_enum.hash_feature",
            return_value=987654321,
        ) as hash_feature:
            item.set_pad_value()

        hash_feature.assert_called_once()
        self.assertEqual(item.hash, 987654321)
        self.assertEqual(item.pad_value, 987654321 % (1 << 24))

    def test_pad_input_tokens_uses_precomputed_hash(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=42,
            feature=np.ones((2, 2), dtype=np.float32),
        )

        with patch("sgl_jax.srt.multimodal.common.modality_enum.hash_feature") as hash_feature:
            padded = pad_input_tokens([1, 999, 2], [item], im_token_id=999)

        hash_feature.assert_not_called()
        self.assertEqual(padded, [1, 42, 2])

    def test_dict_round_trip_preserves_precomputed_hash(self):
        item = MultimodalDataItem.from_dict(
            {
                "modality": "IMAGE",
                "hash": 20260506,
                "feature": np.ones((2, 2), dtype=np.float32),
            }
        )

        with patch("sgl_jax.srt.multimodal.common.modality_enum.hash_feature") as hash_feature:
            item.set_pad_value()

        hash_feature.assert_not_called()
        self.assertEqual(item.pad_value, 20260506 % (1 << 24))

    def test_image_payload_hash_is_stable_and_content_sensitive(self):
        tokenizer = object.__new__(MultimodalTokenizer)
        red_image = _png_bytes((255, 0, 0))
        red_image_again = bytes(red_image)
        blue_image = _png_bytes((0, 0, 255))

        red_payload = tokenizer._load_image_with_hash(red_image)
        red_payload_again = tokenizer._load_image_with_hash(red_image_again)
        blue_payload = tokenizer._load_image_with_hash(blue_image)

        self.assertEqual(red_payload.hash, red_payload_again.hash)
        self.assertNotEqual(red_payload.hash, blue_payload.hash)
        self.assertEqual(red_payload.data.mode, "RGB")

    def test_data_url_hash_matches_decoded_payload_hash(self):
        tokenizer = object.__new__(MultimodalTokenizer)
        image_bytes = _png_bytes((16, 32, 48))
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()

        direct_payload = tokenizer._load_image_with_hash(image_bytes)
        data_url_payload = tokenizer._load_image_with_hash(data_url)

        self.assertEqual(direct_payload.hash, data_url_payload.hash)


if __name__ == "__main__":
    unittest.main()
