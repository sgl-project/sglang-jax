import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.processors.gemma4 import Gemma4Processor

IMAGE_TOKEN = 258880


class _FakeProcessor:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def _config():
    return SimpleNamespace(
        image_token_id=IMAGE_TOKEN,
        boi_token_id=255999,
        eoi_token_id=258882,
        vision_config=SimpleNamespace(pooling_kernel_size=3),
    )


def _positions_with_padding():
    y, x = np.indices((3, 3))
    positions = np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.int32)
    return np.concatenate((positions, np.full((3, 2), -1, dtype=np.int32)))


def test_gemma4_processor_builds_dynamic_image_item(monkeypatch):
    position_ids = _positions_with_padding()
    hf_processor = _FakeProcessor(
        {
            "input_ids": np.asarray([[7, IMAGE_TOKEN, 8]], dtype=np.int64),
            "pixel_values": np.ones((1, 12, 3), dtype=np.float32),
            "image_position_ids": position_ids[None],
        }
    )
    processor = Gemma4Processor(_config(), SimpleNamespace(), hf_processor)
    monkeypatch.setattr(processor, "load_image", lambda _: object())
    request = SimpleNamespace(video_data=None, audio_data=None)

    result = asyncio.run(
        processor.process_mm_data_async(
            np.zeros((4, 4, 3), dtype=np.uint8),
            "<image> describe this",
            request,
        )
    )

    assert result.input_ids == [7, IMAGE_TOKEN, 8]
    assert result.im_start_id == 255999
    assert result.im_end_id == 258882
    assert result.im_token_id == IMAGE_TOKEN
    assert len(result.mm_items) == 1
    item = result.mm_items[0]
    assert item.modality == Modality.IMAGE
    assert item.feature.shape == (9, 3)
    assert item.placeholder_ranges == [(1, 2)]
    np.testing.assert_array_equal(item.get("pixel_position_ids"), position_ids[:9])
    assert item.pad_value is not None
    assert hf_processor.calls[0]["return_tensors"] == "pt"


def test_gemma4_processor_accepts_vllm_position_name(monkeypatch):
    hf_processor = _FakeProcessor(
        {
            "input_ids": np.asarray([[IMAGE_TOKEN]], dtype=np.int64),
            "pixel_values": np.ones((1, 9, 3), dtype=np.float32),
            "pixel_position_ids": _positions_with_padding()[None, :9],
        }
    )
    processor = Gemma4Processor(_config(), SimpleNamespace(), hf_processor)
    monkeypatch.setattr(processor, "load_image", lambda _: object())

    result = asyncio.run(
        processor.process_mm_data_async(
            np.zeros((4, 4, 3), dtype=np.uint8),
            "<image>",
            SimpleNamespace(video_data=None, audio_data=None),
        )
    )

    assert result.mm_items[0].feature.shape[0] == 9


@pytest.mark.parametrize("field", ["video_data", "audio_data"])
def test_gemma4_processor_rejects_unsupported_modalities(field):
    processor = Gemma4Processor(_config(), SimpleNamespace(), object())
    request = SimpleNamespace(video_data=None, audio_data=None)
    setattr(request, field, ["asset"])

    with pytest.raises(ValueError, match="not supported"):
        asyncio.run(processor.process_mm_data_async(None, "prompt", request))
