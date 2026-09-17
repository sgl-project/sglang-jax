"""Tests for the Kimi-K2.5 placeholder expansion in the multimodal tokenizer.

The subtle part is video: Kimi's chat template emits one media placeholder per
media item, where a video contributes one item *per chunk* of frames. The vision
tower average-pools a chunk's frames, so the number of visual tokens a chunk
produces depends only on its spatial grid. Expanding by ``t * h * w`` instead
would emit four times too many placeholders for a 4-frame chunk and shift every
subsequent embedding.
"""

from types import SimpleNamespace

import pytest

from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import MultimodalTokenizer

MEDIA_TOKEN = 163605
MERGE_KERNEL = [2, 2]


class _Harness:
    """Minimal stand-in carrying just the state the helpers touch.

    The real manager opens sockets and loads a processor in ``__init__``, so the
    helpers are re-bound onto this object instead. They are bound under their
    real names because they call one another through ``self``.
    """

    _expand_kimi_media_placeholders = MultimodalTokenizer._expand_kimi_media_placeholders
    _kimi_merge_kernel_area = MultimodalTokenizer._kimi_merge_kernel_area
    _ensure_kimi_video_placeholders = MultimodalTokenizer._ensure_kimi_video_placeholders
    _kimi_video_placeholder = MultimodalTokenizer._kimi_video_placeholder
    _is_kimi_processor = MultimodalTokenizer._is_kimi_processor

    def __init__(self, processor_name="KimiK25Processor", merge_kernel_size=MERGE_KERNEL):
        # The code under test dispatches on the processor's class *name*, and
        # SimpleNamespace's type is immutable, so build a named type instead.
        processor_cls = type(
            processor_name,
            (),
            {"video_placeholder": "<|kimi_k25_video_placeholder|>"},
        )
        self.mm_processor = processor_cls()
        self.mm_config = SimpleNamespace(
            media_placeholder_token_id=MEDIA_TOKEN,
            vision_config=SimpleNamespace(merge_kernel_size=merge_kernel_size),
        )


def test_processor_detection():
    assert _Harness()._is_kimi_processor() is True
    assert _Harness(processor_name="Qwen2_5_VLProcessor")._is_kimi_processor() is False


def test_merge_area_reads_vision_config():
    assert _Harness()._kimi_merge_kernel_area() == 4
    assert _Harness(merge_kernel_size=2)._kimi_merge_kernel_area() == 4
    assert _Harness(merge_kernel_size=[2, 4])._kimi_merge_kernel_area() == 8


def test_image_placeholder_expands_to_spatial_tokens():
    harness = _Harness()

    out = harness._expand_kimi_media_placeholders([1, MEDIA_TOKEN, 2], [(1, 4, 6)])

    # 4 * 6 patches / (2 * 2) = 6 visual tokens.
    assert out == [1] + [MEDIA_TOKEN] * 6 + [2]


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_video_chunk_expansion_is_frame_independent(frames):
    """A chunk's token count must not grow with its frame count."""
    harness = _Harness()

    out = harness._expand_kimi_media_placeholders([MEDIA_TOKEN], [(frames, 4, 6)])

    assert out == [MEDIA_TOKEN] * 6


def test_multiple_media_consume_grids_in_order():
    harness = _Harness()

    # An image, then one 4-frame video chunk with a different resolution.
    out = harness._expand_kimi_media_placeholders(
        [MEDIA_TOKEN, 7, MEDIA_TOKEN], [(1, 2, 2), (4, 4, 4)]
    )

    assert out == [MEDIA_TOKEN] * 1 + [7] + [MEDIA_TOKEN] * 4


def test_expansion_without_media_token_id_is_a_noop():
    harness = _Harness()
    harness.mm_config = SimpleNamespace(media_placeholder_token_id=None)

    assert harness._expand_kimi_media_placeholders([1, 2, 3], [(1, 4, 4)]) == [1, 2, 3]


def test_video_placeholders_are_added_when_missing():
    harness = _Harness()

    text = harness._ensure_kimi_video_placeholders("describe this", 1)

    assert text.count("<|kimi_k25_video_placeholder|>") == 1
    assert text.endswith("describe this")


def test_existing_video_placeholders_are_left_alone():
    harness = _Harness()
    templated = "a<|kimi_k25_video_placeholder|>b"

    assert harness._ensure_kimi_video_placeholders(templated, 1) == templated


def test_no_placeholder_added_without_video():
    harness = _Harness()

    assert harness._ensure_kimi_video_placeholders("plain prompt", 0) == "plain prompt"
