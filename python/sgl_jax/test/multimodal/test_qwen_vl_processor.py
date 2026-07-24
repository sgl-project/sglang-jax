import numpy as np
import pytest

from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor

IMAGE_TOKEN = 151655


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
