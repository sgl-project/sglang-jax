"""V1 remediation tests for the MiMo-V2.5 host-side audio codec.

These cover the fixes from the step-2 review:
- D4-5: per-quantizer codebook range validation (vs the loose scalar bound).
- D3-7 / D4-4: explicit time-major codes_layout contract + square-matrix layout.
All assertions run on numpy alone (no jax/torch needed).
"""

import json
import os
import tempfile
import unittest

import numpy as np

from sgl_jax.srt.multimodal.models.mimo_v2_5.audio_codec_processor import (
    MiMoV25AudioCodecProcessor,
    MiMoV25AudioPayload,
)


class TestMiMoV25CodecPerQuantizerValidation(unittest.TestCase):
    audio_token_id = 151669

    def test_per_quantizer_sizes_reject_high_channel_overflow(self):
        # channel 2 has a real codebook of 256, so id 300 must be rejected even
        # though it is < the scalar 1280 bound.
        codes = np.zeros((4, 20), dtype=np.int32)
        codes[0, 2] = 300
        codebook_sizes = [1024, 1024, 256, 128] + [1280] * 16
        with self.assertRaises(ValueError) as ctx:
            MiMoV25AudioCodecProcessor.normalize_codes(codes, codebook_sizes=codebook_sizes)
        self.assertIn("channel 2", str(ctx.exception))

    def test_per_quantizer_sizes_accept_in_range_codes(self):
        codes = np.zeros((4, 20), dtype=np.int32)
        codes[0, 2] = 255  # in range for a 256 codebook
        codebook_sizes = [1024, 1024, 256, 128] + [1280] * 16
        out = MiMoV25AudioCodecProcessor.normalize_codes(codes, codebook_sizes=codebook_sizes)
        self.assertEqual(out.shape, (4, 20))

    def test_scalar_bound_used_when_no_per_quantizer_sizes(self):
        codes = np.zeros((4, 20), dtype=np.int32)
        codes[0, 2] = 300  # accepted under scalar 1280 bound
        out = MiMoV25AudioCodecProcessor.normalize_codes(codes)
        self.assertEqual(int(out.max()), 300)

    def test_codebook_sizes_length_mismatch_raises(self):
        codes = np.zeros((4, 20), dtype=np.int32)
        with self.assertRaises(ValueError):
            MiMoV25AudioCodecProcessor.normalize_codes(codes, codebook_sizes=[256, 256])

    def test_build_payload_threads_codebook_sizes(self):
        codebook_sizes = [1024, 1024, 256, 128] + [1280] * 16
        payload = MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.zeros((5, 20), dtype=np.int32),
            audio_token_id=self.audio_token_id,
            codebook_sizes=codebook_sizes,
            source="unit",
        )
        self.assertEqual(payload.codebook_sizes, codebook_sizes)


class TestMiMoV25CodecLayoutContract(unittest.TestCase):
    def test_payload_records_time_major_layout(self):
        payload = MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.zeros((20, 5), dtype=np.int32),  # channel-major input
            audio_token_id=151669,
            source="unit",
        )
        # normalized to time-major [T, C]
        self.assertEqual(payload.codes.shape, (8, 20))
        self.assertEqual(payload.codes_layout, "time_major")

    def test_square_codes_are_treated_time_major(self):
        # A [20, 20] block is layout-ambiguous; the contract resolves the last
        # axis as channels (time-major), so distinct per-timestep values survive.
        codes = np.arange(20 * 20, dtype=np.int32).reshape(20, 20) % 1280
        out = MiMoV25AudioCodecProcessor.normalize_codes(codes)
        self.assertEqual(out.shape, (20, 20))
        np.testing.assert_array_equal(out, codes)

    def test_transport_roundtrip_preserves_new_fields(self):
        codebook_sizes = [1024, 1024, 256, 128] + [1280] * 16
        payload = MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.zeros((5, 20), dtype=np.int32),
            audio_token_id=151669,
            codebook_sizes=codebook_sizes,
            source="unit",
        )
        restored = MiMoV25AudioPayload.from_obj(payload.to_transport_dict())
        self.assertEqual(restored.codes_layout, "time_major")
        self.assertEqual(restored.codebook_sizes, codebook_sizes)


class TestMiMoV25CodecGuards(unittest.TestCase):
    """R2-8 / R2-15: no silent downgrade / silent audio drop."""

    def test_build_payload_without_audio_token_id_raises(self):
        # R2-15: a payload implies audio is present; without a scatter target the
        # downstream expand/validate/scatter all no-op silently — fail at build time.
        with self.assertRaisesRegex(ValueError, "audio_token_id"):
            MiMoV25AudioCodecProcessor.build_payload_from_codes(
                np.zeros((5, 20), dtype=np.int32),
                audio_token_id=None,
                source="unit",
            )

    def test_codebook_sizes_missing_config_returns_none(self):
        # No audio_tokenizer/config.json at all -> None (loose scalar check applies).
        with tempfile.TemporaryDirectory() as d:
            codec = MiMoV25AudioCodecProcessor(d, audio_token_id=151669)
            self.assertIsNone(codec.get_codebook_sizes())

    def test_codebook_sizes_malformed_config_raises(self):
        # R2-8: config present but unreadable/missing list -> raise, not silent 1280.
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "audio_tokenizer"))
            with open(os.path.join(d, "audio_tokenizer", "config.json"), "w") as f:
                f.write("{ this is not valid json")
            codec = MiMoV25AudioCodecProcessor(d, audio_token_id=151669)
            with self.assertRaisesRegex(ValueError, "unreadable"):
                codec.get_codebook_sizes()

    def test_codebook_sizes_no_list_raises(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "audio_tokenizer"))
            with open(os.path.join(d, "audio_tokenizer", "config.json"), "w") as f:
                json.dump({"d_model": 1024}, f)  # no codebook_size list
            codec = MiMoV25AudioCodecProcessor(d, audio_token_id=151669)
            with self.assertRaisesRegex(ValueError, "codebook_size"):
                codec.get_codebook_sizes()

    def test_codebook_sizes_valid_list_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "audio_tokenizer"))
            sizes = [1024, 1024, 256] + [128] * 17
            with open(os.path.join(d, "audio_tokenizer", "config.json"), "w") as f:
                json.dump({"codebook_size": sizes}, f)
            codec = MiMoV25AudioCodecProcessor(d, audio_token_id=151669)
            self.assertEqual(codec.get_codebook_sizes(), sizes)


if __name__ == "__main__":
    unittest.main()
