import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.multimodal.models.mimo_v2_5.audio_codec_processor import (
    MiMoV25AudioCodecProcessor,
    MiMoV25AudioPayload,
)


class TestMiMoV25AudioCodecProcessor(unittest.TestCase):
    audio_token_id = 151669

    def test_build_payload_normalizes_channel_first_codes(self):
        payload = MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.zeros((20, 5), dtype=np.int32),
            audio_token_id=self.audio_token_id,
            source="unit",
        )

        self.assertEqual(payload.codes.shape, (8, 20))
        self.assertEqual(payload.token_lengths, [2])
        self.assertEqual(payload.audio_token_id, self.audio_token_id)

    def test_build_payload_pads_each_audio_segment_before_concat(self):
        first = np.full((5, 20), 1, dtype=np.int32)
        second = np.full((6, 20), 2, dtype=np.int32)

        payload = MiMoV25AudioCodecProcessor.build_payload_from_codes(
            [first, second],
            audio_token_id=self.audio_token_id,
            source="unit",
        )

        self.assertEqual(payload.codes.shape, (16, 20))
        self.assertEqual(payload.token_lengths, [2, 2])
        np.testing.assert_array_equal(payload.codes[:5], first)
        np.testing.assert_array_equal(payload.codes[5:8], np.repeat(first[-1:], 3, axis=0))
        np.testing.assert_array_equal(payload.codes[8:14], second)
        np.testing.assert_array_equal(payload.codes[14:16], np.repeat(second[-1:], 2, axis=0))

    def test_build_payload_splits_batched_time_major_codes(self):
        first = np.full((5, 20), 1, dtype=np.int32)
        second = np.full((5, 20), 2, dtype=np.int32)

        payload = MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.stack([first, second], axis=0),
            audio_token_id=self.audio_token_id,
            source="unit",
        )

        self.assertEqual(payload.codes.shape, (16, 20))
        self.assertEqual(payload.token_lengths, [2, 2])
        np.testing.assert_array_equal(payload.codes[:5], first)
        np.testing.assert_array_equal(payload.codes[8:13], second)

    def test_build_payload_splits_batched_channel_major_codes(self):
        first = np.full((20, 5), 1, dtype=np.int32)
        second = np.full((20, 5), 2, dtype=np.int32)

        payload = MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.stack([first, second], axis=0),
            audio_token_id=self.audio_token_id,
            source="unit",
        )

        self.assertEqual(payload.codes.shape, (16, 20))
        self.assertEqual(payload.token_lengths, [2, 2])
        np.testing.assert_array_equal(payload.codes[:5], first.T)
        np.testing.assert_array_equal(payload.codes[8:13], second.T)

    def test_build_payload_rejects_out_of_range_codes(self):
        codes = np.zeros((5, 20), dtype=np.int32)
        codes[0, 0] = 1280

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            MiMoV25AudioCodecProcessor.build_payload_from_codes(
                codes,
                audio_token_id=self.audio_token_id,
                source="unit",
            )

    def test_normalize_codes_rejects_empty_time_major_codes(self):
        with self.assertRaisesRegex(ValueError, "audio_codes cannot be empty"):
            MiMoV25AudioCodecProcessor.normalize_codes(np.zeros((0, 20), dtype=np.int32))

    def test_normalize_codes_rejects_empty_channel_major_codes(self):
        with self.assertRaisesRegex(ValueError, "audio_codes cannot be empty"):
            MiMoV25AudioCodecProcessor.normalize_codes(np.zeros((20, 0), dtype=np.int32))

    def test_normalize_payload_rejects_empty_codes(self):
        payload = MiMoV25AudioPayload(
            codes=np.zeros((0, 20), dtype=np.int32),
            token_lengths=[1],
            audio_token_id=self.audio_token_id,
        )

        with self.assertRaisesRegex(ValueError, "audio_codes cannot be empty"):
            MiMoV25AudioCodecProcessor.normalize_payload(payload)

    def test_normalize_payload_pads_single_unpadded_payload(self):
        payload = MiMoV25AudioPayload(
            codes=np.ones((5, 20), dtype=np.int32),
            token_lengths=[2],
            audio_token_id=self.audio_token_id,
            offsets=[(3, 5)],
            source="direct",
            is_tokenized=False,
        )

        normalized = MiMoV25AudioCodecProcessor.normalize_payload(payload)

        self.assertEqual(normalized.codes.shape, (8, 20))
        self.assertEqual(normalized.token_lengths, [2])
        self.assertEqual(normalized.offsets, [(3, 5)])
        self.assertEqual(normalized.source, "direct")
        self.assertFalse(normalized.is_tokenized)

    def test_normalize_payload_rejects_single_payload_token_length_mismatch(self):
        payload = MiMoV25AudioPayload(
            codes=np.ones((5, 20), dtype=np.int32),
            token_lengths=[3],
            audio_token_id=self.audio_token_id,
        )

        with self.assertRaisesRegex(ValueError, "token_lengths mismatch"):
            MiMoV25AudioCodecProcessor.normalize_payload(payload)

    def test_normalize_payload_rejects_multi_segment_raw_concat(self):
        payload = MiMoV25AudioPayload(
            codes=np.ones((11, 20), dtype=np.int32),
            token_lengths=[2, 2],
            audio_token_id=self.audio_token_id,
        )

        with self.assertRaisesRegex(ValueError, "stage0-ready"):
            MiMoV25AudioCodecProcessor.normalize_payload(payload)

    def test_normalize_payload_rejects_offset_count_mismatch(self):
        payload = MiMoV25AudioPayload(
            codes=np.ones((12, 20), dtype=np.int32),
            token_lengths=[1, 2],
            offsets=[(0, 1)],
            audio_token_id=self.audio_token_id,
        )

        with self.assertRaisesRegex(ValueError, "offset count mismatch"):
            MiMoV25AudioCodecProcessor.normalize_payload(payload)

    def test_normalize_payload_rejects_offset_length_mismatch(self):
        payload = MiMoV25AudioPayload(
            codes=np.ones((8, 20), dtype=np.int32),
            token_lengths=[2],
            offsets=[(0, 1)],
            audio_token_id=self.audio_token_id,
        )

        with self.assertRaisesRegex(ValueError, "offset length mismatch"):
            MiMoV25AudioCodecProcessor.normalize_payload(payload)

    def test_payload_from_obj_normalizes_json_transport_shape(self):
        payload = MiMoV25AudioPayload.from_obj(
            {
                "codes": np.ones((8, 20), dtype=np.int32).tolist(),
                "token_lengths": ["2"],
                "offsets": [[4, 6]],
                "audio_token_id": str(self.audio_token_id),
                "num_channels": "20",
                "codebook_size": "1280",
                "group_size": "4",
                "source": "json",
                "is_tokenized": "false",
            }
        )

        normalized = MiMoV25AudioCodecProcessor.normalize_payload(payload)

        self.assertIsInstance(normalized.codes, np.ndarray)
        self.assertEqual(normalized.codes.dtype, np.int32)
        self.assertEqual(normalized.token_lengths, [2])
        self.assertEqual(normalized.offsets, [(4, 6)])
        self.assertEqual(normalized.audio_token_id, self.audio_token_id)
        self.assertFalse(normalized.is_tokenized)

    def test_payload_from_obj_rejects_invalid_offset_shape(self):
        with self.assertRaisesRegex(ValueError, "offsets must be"):
            MiMoV25AudioPayload.from_obj(
                {
                    "codes": np.ones((8, 20), dtype=np.int32).tolist(),
                    "token_lengths": [2],
                    "offsets": [[1, 2, 3]],
                    "audio_token_id": self.audio_token_id,
                }
            )

    def test_payload_transport_dict_roundtrip(self):
        payload = MiMoV25AudioPayload(
            codes=np.arange(160, dtype=np.int32).reshape(8, 20),
            token_lengths=[2],
            offsets=[(5, 7)],
            audio_token_id=self.audio_token_id,
            source="unit",
            is_tokenized=False,
        )

        restored = MiMoV25AudioPayload.from_obj(payload.to_transport_dict())
        normalized = MiMoV25AudioCodecProcessor.normalize_payload(restored)

        np.testing.assert_array_equal(normalized.codes, payload.codes)
        self.assertEqual(normalized.token_lengths, [2])
        self.assertEqual(normalized.offsets, [(5, 7)])
        self.assertEqual(normalized.audio_token_id, self.audio_token_id)
        self.assertFalse(normalized.is_tokenized)

    def test_expand_single_audio_placeholder_and_attach_offset(self):
        payload = MiMoV25AudioPayload(
            codes=np.zeros((8, 20), dtype=np.int32),
            token_lengths=[2],
            audio_token_id=self.audio_token_id,
        )

        input_ids = MiMoV25AudioCodecProcessor.expand_single_audio_placeholders(
            [1, self.audio_token_id, 2],
            payload,
        )
        MiMoV25AudioCodecProcessor.validate_placeholder_count(input_ids, payload)

        self.assertEqual(input_ids, [1, self.audio_token_id, self.audio_token_id, 2])
        self.assertEqual(payload.offsets, [(1, 3)])

    def test_expand_multiple_audio_placeholders_and_attach_offsets(self):
        payload = MiMoV25AudioPayload(
            codes=np.zeros((12, 20), dtype=np.int32),
            token_lengths=[1, 2],
            audio_token_id=self.audio_token_id,
        )

        input_ids = MiMoV25AudioCodecProcessor.expand_single_audio_placeholders(
            [self.audio_token_id, 9, self.audio_token_id],
            payload,
        )
        MiMoV25AudioCodecProcessor.validate_placeholder_count(input_ids, payload)

        self.assertEqual(
            input_ids, [self.audio_token_id, 9, self.audio_token_id, self.audio_token_id]
        )
        self.assertEqual(payload.offsets, [(0, 1), (2, 4)])

    def test_validate_rejects_merged_audio_spans(self):
        payload = MiMoV25AudioPayload(
            codes=np.zeros((12, 20), dtype=np.int32),
            token_lengths=[1, 2],
            audio_token_id=self.audio_token_id,
        )

        with self.assertRaisesRegex(ValueError, "span count mismatch"):
            MiMoV25AudioCodecProcessor.validate_placeholder_count(
                [self.audio_token_id, self.audio_token_id, self.audio_token_id],
                payload,
            )

    def test_ensure_mel_time_major_rejects_empty_time_major_mel(self):
        processor = MiMoV25AudioCodecProcessor(model_path="/unused")

        with self.assertRaisesRegex(ValueError, "time dimension cannot be empty"):
            processor._ensure_mel_time_major(np.zeros((0, 128), dtype=np.float32))

    def test_ensure_mel_time_major_transposes_numpy_channel_major_mel(self):
        processor = MiMoV25AudioCodecProcessor(model_path="/unused")
        mel = np.arange(128 * 3, dtype=np.float32).reshape(128, 3)

        normalized = processor._ensure_mel_time_major(mel)

        self.assertEqual(normalized.shape, (3, 128))
        np.testing.assert_array_equal(normalized, np.swapaxes(mel, 0, 1))

    def test_ensure_mel_time_major_rejects_empty_channel_major_mel(self):
        processor = MiMoV25AudioCodecProcessor(model_path="/unused")

        with self.assertRaisesRegex(ValueError, "time dimension cannot be empty"):
            processor._ensure_mel_time_major(np.zeros((128, 0), dtype=np.float32))

    def test_load_audio_tokenizer_rejects_missing_weight_keys(self):
        class _FakeTorch:
            bfloat16 = "bfloat16"

            @staticmethod
            def load(path, map_location=None):
                return {"present.weight": np.zeros((1,), dtype=np.float32)}

        class _FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class _FakeTokenizer:
            def __init__(self, config):
                self.config = config

            def load_state_dict(self, state_dict, strict=False):
                return SimpleNamespace(missing_keys=["encoder.required.weight"], unexpected_keys=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_dir = f"{tmpdir}/audio_tokenizer"
            import os

            os.mkdir(tokenizer_dir)
            with open(f"{tokenizer_dir}/config.json", "w") as f:
                f.write("{}")
            with open(f"{tokenizer_dir}/pytorch_model.bin", "wb") as f:
                f.write(b"placeholder")

            processor = MiMoV25AudioCodecProcessor(model_path=tmpdir)
            processor._require_torch = lambda: _FakeTorch
            processor._load_remote_symbol = lambda symbol: (
                _FakeConfig if symbol == "MiMoAudioTokenizerConfig" else _FakeTokenizer
            )

            with self.assertRaisesRegex(ValueError, "weights are incomplete"):
                processor._load_audio_tokenizer()

    def test_load_audio_tokenizer_caches_successful_tokenizer(self):
        class _FakeTorch:
            bfloat16 = "bfloat16"

            @staticmethod
            def load(path, map_location=None):
                return {"present.weight": np.zeros((1,), dtype=np.float32)}

        class _FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class _FakeTokenizer:
            load_calls = 0

            def __init__(self, config):
                self.config = config
                self.to_kwargs = None
                self.eval_called = False
                self.requires_grad = None

            def load_state_dict(self, state_dict, strict=False):
                type(self).load_calls += 1
                self.loaded_state_dict = state_dict
                self.strict = strict
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            def to(self, **kwargs):
                self.to_kwargs = kwargs
                return self

            def eval(self):
                self.eval_called = True
                return self

            def requires_grad_(self, value):
                self.requires_grad = value
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_dir = f"{tmpdir}/audio_tokenizer"
            import os

            os.mkdir(tokenizer_dir)
            with open(f"{tokenizer_dir}/config.json", "w") as f:
                f.write("{}")
            with open(f"{tokenizer_dir}/pytorch_model.bin", "wb") as f:
                f.write(b"placeholder")

            processor = MiMoV25AudioCodecProcessor(model_path=tmpdir, device="cpu")
            processor._require_torch = lambda: _FakeTorch
            processor._load_remote_symbol = lambda symbol: (
                _FakeConfig if symbol == "MiMoAudioTokenizerConfig" else _FakeTokenizer
            )

            tokenizer = processor._load_audio_tokenizer()
            cached = processor._load_audio_tokenizer()

        self.assertIs(tokenizer, cached)
        self.assertEqual(_FakeTokenizer.load_calls, 1)
        self.assertFalse(tokenizer.strict)
        self.assertEqual(tokenizer.to_kwargs, {"device": "cpu", "dtype": "bfloat16"})
        self.assertTrue(tokenizer.eval_called)
        self.assertFalse(tokenizer.requires_grad)


if __name__ == "__main__":
    unittest.main()
