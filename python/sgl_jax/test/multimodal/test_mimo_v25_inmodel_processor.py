"""CPU unit tests for the IN-MODEL MiMoV2_5Processor (refactor M4 / review P3-1).

Exercises the in-model `multimodal.processors.mimo_v2_5.MiMoV2_5Processor`: the
Qwen2.5-VL-derived vision/text path plus the RVQ-codes audio merge, the mRoPE capability gate
(uses_mrope=False), the is_codes AUDIO item, and the cache_input_ids rebuild. Fakes are injected
for the HF processor + codec so no checkpoint is needed on disk. (The old staged MiMoV25Processor
and its test were removed in M6-S5.)
"""

import unittest

import numpy as np

from sgl_jax.srt.models.mimo_v2_5.audio_codec_processor import (
    MiMoV25AudioCodecProcessor,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.processors.mimo_v2_5 import MiMoV2_5Processor

AUDIO_TOKEN_ID = 151669
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeHF:
    """Stand-in for the HF AutoProcessor: returns input_ids (with placeholders) only."""

    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.calls = []

    def __call__(self, text=None, images=None, videos=None, return_tensors=None, **kwargs):
        self.calls.append({"text": text, "images": images, "videos": videos})
        return {"input_ids": np.asarray([self.input_ids], dtype=np.int64)}

    def apply_chat_template(self, *args, **kwargs):
        return "<audio>"


class _FakeCodec:
    """5 RVQ timesteps over 20 channels -> ceil(5/4) = 2 audio tokens."""

    def __init__(self):
        self.encode_calls = []

    def encode(self, audios):
        self.encode_calls.append(audios)
        return MiMoV25AudioCodecProcessor.build_payload_from_codes(
            np.zeros((5, 20), dtype=np.int32), audio_token_id=AUDIO_TOKEN_ID
        )


def _fake_config():
    return _Obj(
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        audio_token_id=AUDIO_TOKEN_ID,
        vision_config=_Obj(spatial_merge_size=2, tokens_per_second=2),
    )


class TestMiMoV25InModelProcessor(unittest.TestCase):
    def _proc(self, hf, codec=None):
        return MiMoV2_5Processor(
            "local/MiMo-V2.5", hf_processor=hf, hf_config=_fake_config(), codec=codec
        )

    def test_uses_mrope_is_false(self):
        # MiMo's AR reads forward_batch.positions, not mRoPE -> the capability gate must be off.
        self.assertFalse(MiMoV2_5Processor.uses_mrope)

    def test_audio_expands_placeholders_and_attaches_codes_item(self):
        hf = _FakeHF([1, AUDIO_TOKEN_ID, 2])  # one <audio_pad>
        codec = _FakeCodec()
        out = self._proc(hf, codec).process(audios=["raw-audio"], text="<audio>")

        self.assertEqual(codec.encode_calls, [["raw-audio"]])
        # one pad expanded to two (2 audio tokens)
        self.assertEqual(list(out["input_ids"]), [1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 2])

        items = out["mm_inputs"]["mm_items"]
        audio_items = [it for it in items if it.modality == Modality.AUDIO]
        self.assertEqual(len(audio_items), 1)
        msd = audio_items[0].model_specific_data
        self.assertTrue(msd["is_codes"])
        self.assertEqual(msd["token_lengths"], [2])
        self.assertEqual(msd["group_size"], 4)
        self.assertEqual(out["mm_inputs"]["audio_token_id"], AUDIO_TOKEN_ID)

    def test_mrope_is_none(self):
        hf = _FakeHF([1, AUDIO_TOKEN_ID, 2])
        out = self._proc(hf, _FakeCodec()).process(audios=["raw-audio"], text="<audio>")
        self.assertIsNone(out["mm_inputs"]["mrope_positions"])
        self.assertIsNone(out["mm_inputs"]["mrope_position_delta"])

    def test_cache_input_ids_rebuilt_to_expanded_length(self):
        hf = _FakeHF([1, AUDIO_TOKEN_ID, 2])
        out = self._proc(hf, _FakeCodec()).process(audios=["raw-audio"], text="<audio>")
        # cache copy (Scheme B) reflects the EXPANDED ids (4 tokens), not the 3-token template.
        self.assertEqual(len(out["mm_inputs"]["cache_input_ids"]), len(out["input_ids"]))
        self.assertEqual(len(out["input_ids"]), 4)

    def test_no_audio_passes_through_and_clears_mrope(self):
        hf = _FakeHF([1, 2, 3])
        codec = _FakeCodec()
        out = self._proc(hf, codec).process(text="hi")
        self.assertEqual(codec.encode_calls, [])
        self.assertEqual(list(out["input_ids"]), [1, 2, 3])
        self.assertIsNone(out["mm_inputs"]["mrope_positions"])
        audio_items = [it for it in out["mm_inputs"]["mm_items"] if it.modality == Modality.AUDIO]
        self.assertEqual(audio_items, [])


if __name__ == "__main__":
    unittest.main()
