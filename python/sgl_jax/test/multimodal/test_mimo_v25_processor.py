"""Tests for MiMoV25Processor's audio merge logic (numpy-only).

The Qwen/transformers vision call is injected as a fake; the audio merge + <audio_pad>
expansion + host validation (the part that matters) runs on real numpy codec helpers.
"""

import unittest

import numpy as np

from sgl_jax.srt.multimodal.models.mimo_v2_5.audio_codec_processor import (
    MiMoV25AudioCodecProcessor,
)
from sgl_jax.srt.multimodal.models.mimo_v2_5.processor import MiMoV25Processor

AUDIO_TOKEN_ID = 151669


class _FakeHF:
    """Stand-in for Qwen2_5_VLProcessor: returns input_ids with one <audio_pad>."""

    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.calls = []
        self.template_calls = []

    def __call__(self, images=None, videos=None, text="", return_tensors=None, **kwargs):
        self.calls.append({"images": images, "videos": videos, "text": text})
        return {"input_ids": np.asarray([self.input_ids], dtype=np.int64)}

    def apply_chat_template(self, *args, **kwargs):
        self.template_calls.append((args, kwargs))
        return "<audio>"


class _FakeCodec:
    def __init__(self):
        self.encode_calls = []

    def encode(self, audio):
        self.encode_calls.append(audio)
        return MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.zeros((5, 20), dtype=np.int32),
            audio_token_id=AUDIO_TOKEN_ID,
            source="fake",
        )


class TestMiMoV25Processor(unittest.TestCase):
    def _proc(self, hf, codec):
        return MiMoV25Processor(
            "local/MiMo-V2.5", audio_token_id=AUDIO_TOKEN_ID, hf_processor=hf, codec=codec
        )

    def test_audio_expands_placeholders_and_attaches_codes(self):
        hf = _FakeHF([1, AUDIO_TOKEN_ID, 2])  # one <audio_pad>
        codec = _FakeCodec()
        out = self._proc(hf, codec)(audio=["raw-audio"], text="<audio>")

        self.assertEqual(codec.encode_calls, [["raw-audio"]])
        # 5 codes / group_size 4 -> ceil = 2 audio tokens; one pad expanded to two
        self.assertEqual(out["input_ids"].tolist(), [[1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 2]])
        self.assertEqual(out["audio_codes"].shape, (8, 20))  # padded to group multiple
        self.assertEqual(out["audio_token_lengths"], [2])
        self.assertEqual(out["audio_group_size"], 4)

    def test_no_audio_passes_through_vision_only(self):
        hf = _FakeHF([1, 2, 3])
        codec = _FakeCodec()
        out = self._proc(hf, codec)(text="hi")
        self.assertNotIn("audio_codes", out)
        self.assertEqual(codec.encode_calls, [])
        self.assertEqual(out["input_ids"].tolist(), [[1, 2, 3]])

    def test_placeholder_count_mismatch_raises(self):
        # prompt has no <audio_pad> but audio provided -> validate should fail
        hf = _FakeHF([1, 2, 3])
        codec = _FakeCodec()
        with self.assertRaises(ValueError):
            self._proc(hf, codec)(audio=["raw-audio"], text="no placeholder")

    def test_apply_chat_template_delegates_to_wrapped_hf_processor(self):
        hf = _FakeHF([1, AUDIO_TOKEN_ID, 2])
        proc = self._proc(hf, _FakeCodec())

        rendered = proc.apply_chat_template([{"role": "user", "content": [{"type": "audio"}]}])

        self.assertEqual(rendered, "<audio>")
        self.assertEqual(len(hf.template_calls), 1)


if __name__ == "__main__":
    unittest.main()
