import asyncio
from types import SimpleNamespace

import numpy as np
from PIL import Image

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor

IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653


class _FakeHFProcessor:
    def __init__(self, pixel_base: float = 0.0):
        self.pixel_base = pixel_base

    def __call__(self, text, images, padding, return_tensors):
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].mode == "RGB"
        return {
            "input_ids": np.array(
                [
                    [
                        1,
                        VISION_START_TOKEN_ID,
                        IMAGE_TOKEN_ID,
                        IMAGE_TOKEN_ID,
                        IMAGE_TOKEN_ID,
                        IMAGE_TOKEN_ID,
                        VISION_END_TOKEN_ID,
                        2,
                    ]
                ],
                dtype=np.int32,
            ),
            "pixel_values": self.pixel_base + np.arange(16 * 3, dtype=np.float32).reshape(16, 3),
            "image_grid_thw": np.array([[1, 4, 4]], dtype=np.int32),
        }


def _qwen_processor(pixel_base: float = 0.0):
    hf_config = SimpleNamespace(
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            patch_size=1,
            window_size=4,
            hidden_size=32,
            num_heads=4,
        ),
    )
    return QwenVLProcessor(
        hf_config=hf_config,
        server_args=SimpleNamespace(),
        processor=_FakeHFProcessor(pixel_base=pixel_base),
    )


def _process_mm_inputs(pixel_base: float = 0.0):
    image = Image.fromarray(np.full((4, 4, 3), int(pixel_base) % 255, dtype=np.uint8))
    return asyncio.run(
        _qwen_processor(pixel_base=pixel_base).process_mm_data_async(
            image_data=[image],
            input_text="describe the image",
            request_obj=SimpleNamespace(),
        )
    )


def _fake_schedule_batch(mm_inputs_by_rank):
    reqs_info = []
    for mm_inputs in mm_inputs_by_rank:
        if mm_inputs is None:
            reqs_info.append(SimpleNamespace(reqs=[], seq_lens=None, prefix_lens=None))
            continue
        req = SimpleNamespace(mm_inputs=mm_inputs)
        reqs_info.append(
            SimpleNamespace(
                reqs=[req],
                seq_lens=np.array([len(mm_inputs.input_ids)], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
            )
        )

    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        dp_size=2,
        reqs_info=reqs_info,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen2_5_VLForConditionalGeneration"],
                vision_config=_qwen_processor().hf_config.vision_config,
            )
        ),
    )


def test_processor_output_is_consumed_by_schedule_embed_plan():
    mm_inputs = _process_mm_inputs()

    batch = _fake_schedule_batch([mm_inputs, None])
    plan = ScheduleBatch._build_mm_embed_plan(batch, per_dp_token_size=16, total_token_size=32)

    assert plan is not None
    assert len(plan.image_rounds) == 1
    round0 = plan.image_rounds[0]
    np.testing.assert_array_equal(round0.encode_inputs.valid, np.array([16, 0]))

    expected_mask = np.zeros(32, dtype=bool)
    expected_mask[2:6] = True
    np.testing.assert_array_equal(round0.mask, expected_mask)

    expected_src_idx = np.zeros(32, dtype=np.int32)
    expected_src_idx[2:6] = np.arange(4, dtype=np.int32)
    np.testing.assert_array_equal(round0.src_idx, expected_src_idx)

    assert round0.encode_inputs.meta.window_index.shape == (2, 4)
    assert round0.encode_inputs.meta.rotary_pos_emb.shape == (2, 16, 4)


def test_processor_outputs_are_consumed_per_dp_rank():
    rank0_mm_inputs = _process_mm_inputs(pixel_base=10.0)
    rank1_mm_inputs = _process_mm_inputs(pixel_base=100.0)

    batch = _fake_schedule_batch([rank0_mm_inputs, rank1_mm_inputs])
    plan = ScheduleBatch._build_mm_embed_plan(batch, per_dp_token_size=16, total_token_size=32)

    assert plan is not None
    assert len(plan.image_rounds) == 1
    round0 = plan.image_rounds[0]
    np.testing.assert_array_equal(round0.encode_inputs.valid, np.array([16, 16]))

    expected_mask = np.zeros(32, dtype=bool)
    expected_mask[2:6] = True
    expected_mask[18:22] = True
    np.testing.assert_array_equal(round0.mask, expected_mask)

    expected_src_idx = np.zeros(32, dtype=np.int32)
    expected_src_idx[2:6] = np.arange(4, dtype=np.int32)
    expected_src_idx[18:22] = np.arange(4, dtype=np.int32)
    np.testing.assert_array_equal(round0.src_idx, expected_src_idx)

    np.testing.assert_array_equal(
        round0.encode_inputs.pixels[0], rank0_mm_inputs.mm_items[0].feature
    )
    np.testing.assert_array_equal(
        round0.encode_inputs.pixels[1], rank1_mm_inputs.mm_items[0].feature
    )
    assert round0.encode_inputs.pixels[0, 0, 0] == 10.0
    assert round0.encode_inputs.pixels[1, 0, 0] == 100.0
