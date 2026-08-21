import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    build_cache_input_ids,
)
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor

IMAGE_TOKEN = 151655


def test_build_cache_input_ids_uses_item_hash():
    input_ids = [1, IMAGE_TOKEN, IMAGE_TOKEN, 2]
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        pad_value=123456,
        placeholder_ranges=[(1, 3)],
    )

    assert build_cache_input_ids(input_ids, MultimodalInputs([item])) == [
        1,
        123456,
        123456,
        2,
    ]
    assert input_ids == [1, IMAGE_TOKEN, IMAGE_TOKEN, 2]


def test_qwen_vl_rejects_audio_inputs():
    processor = QwenVLProcessor(SimpleNamespace(), SimpleNamespace(), object())
    request = SimpleNamespace(audio_data=["audio.wav"])

    with pytest.raises(ValueError, match="does not support audio"):
        asyncio.run(processor.process_mm_data_async(None, "prompt", request))


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
