import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    build_radix_input_ids,
)
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor

IMAGE_TOKEN = 151655


def _make_qwen_processor(workers=2):
    hf_config = SimpleNamespace(
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        vision_config=SimpleNamespace(patch_size=14, spatial_merge_size=2),
    )
    return QwenVLProcessor(hf_config, SimpleNamespace(mm_processor_worker_num=workers), object())


def test_build_radix_input_ids():
    input_ids = [1, IMAGE_TOKEN, IMAGE_TOKEN, 2]
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        pad_value=123456,
        placeholder_ranges=[(1, 3)],
    )

    assert build_radix_input_ids(input_ids, MultimodalInputs([item])) == [
        1,
        123456,
        123456,
        2,
    ]
    assert input_ids == [1, IMAGE_TOKEN, IMAGE_TOKEN, 2]
    assert build_radix_input_ids([1, 2], None) == [1, 2]


def test_qwen_vl_rejects_audio_inputs():
    processor = QwenVLProcessor(SimpleNamespace(), SimpleNamespace(), object())
    request = SimpleNamespace(audio_data=["audio.wav"])

    try:
        with pytest.raises(ValueError, match="does not support audio"):
            asyncio.run(processor.process_mm_data_async(None, "prompt", request))
    finally:
        processor.shutdown()


def test_qwen_process_and_combine_runs_in_hf_processor_worker():
    worker_thread_id = None
    processor = _make_qwen_processor()

    def process_and_combine(*_, processor, **__):
        nonlocal worker_thread_id
        worker_thread_id = threading.get_ident()
        return MultimodalInputs(mm_items=[], input_ids=[10, 11])

    processor.process_and_combine_mm_data = process_and_combine

    async def run_processor():
        event_loop_thread_id = threading.get_ident()
        output = await processor.process_mm_data_async(None, "prompt", SimpleNamespace())
        return event_loop_thread_id, output

    try:
        event_loop_thread_id, output = asyncio.run(run_processor())
    finally:
        processor.shutdown()

    assert worker_thread_id is not None and worker_thread_id != event_loop_thread_id
    assert output.input_ids == [10, 11]


@pytest.mark.parametrize("workers", [1, 2])
def test_qwen_loads_and_processes_images_and_videos_in_one_worker(monkeypatch, workers):
    processor = _make_qwen_processor(workers)
    stages = []
    event_loop_thread_id = threading.get_ident()

    def load_image(source):
        assert source == "image-source"
        stages.append(("image", threading.get_ident()))
        return "loaded-image"

    def load_video(source, config):
        assert source == "video-source"
        assert config["factor"] == 28
        stages.append(("video", threading.get_ident()))
        return "loaded-video"

    def combine(input_text, images=None, videos=None, *, processor, **kwargs):
        assert input_text == "prompt"
        assert images == ["loaded-image"]
        assert videos == ["loaded-video"]
        assert kwargs["videos_kwargs"]["do_sample_frames"] is False
        stages.append(("processor", threading.get_ident()))
        return MultimodalInputs(mm_items=[], input_ids=[10, 11])

    monkeypatch.setattr(processor, "load_image", load_image)
    monkeypatch.setattr("sgl_jax.srt.multimodal.processors.qwen_vl.preprocess_video", load_video)
    monkeypatch.setattr(processor, "process_and_combine_mm_data", combine)

    async def run_processor():
        return await asyncio.wait_for(
            processor.process_mm_data_async(
                "image-source",
                "prompt",
                SimpleNamespace(audio_data=None, video_data="video-source"),
            ),
            timeout=1,
        )

    try:
        output = asyncio.run(run_processor())
    finally:
        processor.shutdown()

    assert output.input_ids == [10, 11]
    assert [stage for stage, _ in stages] == ["image", "video", "processor"]
    assert len({thread_id for _, thread_id in stages}) == 1
    assert stages[0][1] != event_loop_thread_id


def test_placeholder_ranges_are_half_open():
    input_ids = [1, 2, IMAGE_TOKEN, IMAGE_TOKEN, 3, *([IMAGE_TOKEN] * 4), 4]
    ranges = QwenVLProcessor._compute_image_placeholder_ranges(
        input_ids,
        [(1, 2, 4), (1, 4, 4)],
        IMAGE_TOKEN,
        spatial_merge_size=2,
    )
    assert ranges == [(2, 4), (5, 9)]


@pytest.mark.parametrize(
    ("input_ids", "token_id", "match"),
    [
        ([1, 2], IMAGE_TOKEN, "Missing IMAGE placeholder"),
        ([IMAGE_TOKEN, 1], IMAGE_TOKEN, "span does not match"),
        ([IMAGE_TOKEN, IMAGE_TOKEN], None, "token id is not configured"),
    ],
)
def test_placeholder_ranges_reject_invalid_spans(input_ids, token_id, match):
    with pytest.raises(ValueError, match=match):
        QwenVLProcessor._compute_image_placeholder_ranges(
            input_ids,
            [(1, 2, 4)],
            token_id,
            spatial_merge_size=2,
        )


def test_build_items_splits_features_and_metadata():
    items = QwenVLProcessor._build_items(
        np.arange(24).reshape(24, 1),
        [(1, 2, 4), (1, 4, 4)],
        [(2, 4), (5, 9)],
        Modality.IMAGE,
        "image_grid_thw",
    )
    assert [item.feature.shape for item in items] == [(8, 1), (16, 1)]
    assert [item.placeholder_ranges for item in items] == [[(2, 4)], [(5, 9)]]
    np.testing.assert_array_equal(items[1].get("image_grid_thw"), [[1, 4, 4]])


def test_video_timing_changes_item_identity():
    def make_item(seconds):
        item = QwenVLProcessor._build_items(
            np.ones((8, 1)),
            [(1, 2, 4)],
            [(0, 2)],
            Modality.VIDEO,
            "video_grid_thw",
        )[0]
        QwenVLProcessor._set_video_timing([item], [seconds])
        item.set_pad_value()
        return item

    assert make_item(0.5).hash != make_item(1.0).hash


@pytest.mark.parametrize(
    ("features", "grids", "ranges", "match"),
    [
        (np.ones((8, 1)), [], [], "Missing image_grid_thw"),
        (np.ones((8, 1)), [(1, 2, 4)], [], "range count"),
        (np.ones((7, 1)), [(1, 2, 4)], [(0, 2)], "feature count"),
    ],
)
def test_build_items_validates_shapes(features, grids, ranges, match):
    with pytest.raises(ValueError, match=match):
        QwenVLProcessor._build_items(
            features,
            grids,
            ranges,
            Modality.IMAGE,
            "image_grid_thw",
        )
