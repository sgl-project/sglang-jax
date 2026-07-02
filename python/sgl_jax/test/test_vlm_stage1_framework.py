from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ScheduleBatch,
    ScheduleReqsInfo,
    build_mm_embed_plan,
)
from sgl_jax.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    _device_put_embed_plan,
)
from sgl_jax.srt.models.vision_metadata import (  # noqa: F401
    qwen2_5_vl as _qwen25vl_vision_metadata,
)
from sgl_jax.srt.models.vision_metadata.qwen2_5_vl import Qwen25VLVisionMetadata
from sgl_jax.srt.multimodal.common.mm_plan import (
    EmbedRound,
    MultimodalEmbedPlan,
    VisionEncodeInputs,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor


def test_generate_req_getitem_preserves_media_fields():
    req = GenerateReqInput(
        text=["a", "b"],
        sampling_params=[{}, {}],
        rid=["r0", "r1"],
        return_logprob=[False, False],
        logprob_start_len=[-1, -1],
        top_logprobs_num=[0, 0],
        token_ids_logprob=[None, None],
        return_routed_experts=[False, False],
        image_data=[["image0"], ["image1"]],
        video_data=[["video0"], ["video1"]],
        audio_data=[["audio0"], ["audio1"]],
    )
    req.input_embeds = [["emb0"], ["emb1"]]

    item = req[1]

    assert item.image_data == ["image1"]
    assert item.video_data == ["video1"]
    assert item.audio_data == ["audio1"]
    assert item.input_embeds == ["emb1"]


def test_forward_batch_input_embedding_uses_data_axis_sharding():
    devices = np.array(jax.devices()[:1])
    mesh = Mesh(devices, ("data",))
    batch = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.EXTEND,
        input_ids=np.array([1], dtype=np.int32),
        real_input_ids_len=1,
        seq_lens=np.array([1], dtype=np.int32),
        out_cache_loc=np.array([1], dtype=np.int32),
        req_pool_indices=np.array([0], dtype=np.int32),
        sampling_info=None,
        positions=np.array([0], dtype=np.int32),
        cache_loc=np.array([1], dtype=np.int32),
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=np.array([1], dtype=np.int32),
        extend_prefix_lens=np.array([0], dtype=np.int32),
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.array([0], dtype=np.int32),
        real_bs=1,
        real_bs_per_dp=[1],
        input_embedding=np.ones((1, 4), dtype=np.float32),
    )
    runner = SimpleNamespace(
        mesh=mesh,
        attn_backend=None,
        model_config=SimpleNamespace(
            is_embedding=False,
            hf_config=SimpleNamespace(architectures=[]),
        ),
    )
    captured_specs = []

    def fake_device_array(values, sharding):
        captured_specs.append(sharding.spec)
        return values

    with patch(
        "sgl_jax.srt.model_executor.forward_batch_info.device_array",
        side_effect=fake_device_array,
    ):
        ForwardBatch.init_new(batch, runner)

    assert PartitionSpec("data", None) in captured_specs


def test_mm_embed_plan_device_put_uses_data_leading_sharding():
    devices = np.array(jax.devices()[:1])
    mesh = Mesh(devices, ("data",))
    plan = MultimodalEmbedPlan(
        rounds_by_modality={
            Modality.IMAGE: [
                EmbedRound(
                    encode_inputs=VisionEncodeInputs(
                        pixels=np.ones((1, 4, 3), dtype=np.float32),
                        valid=np.array([4], dtype=np.int32),
                        meta=Qwen25VLVisionMetadata(
                            window_index=np.zeros((1, 1), dtype=np.int32),
                            cu_window_seqlens=np.ones((1, 1), dtype=np.int32),
                            rotary_pos_emb=np.ones((1, 4, 2), dtype=np.float32),
                        ),
                    ),
                    src_idx=np.zeros((4,), dtype=np.int32),
                    mask=np.zeros((4,), dtype=np.bool_),
                )
            ]
        }
    )
    captured_specs = []

    def fake_device_array(values, sharding):
        captured_specs.append(sharding.spec)
        return values

    with patch(
        "sgl_jax.srt.model_executor.forward_batch_info.device_array",
        side_effect=fake_device_array,
    ):
        _device_put_embed_plan(plan, mesh)

    assert captured_specs == [
        PartitionSpec("data", None, None),
        PartitionSpec("data"),
        PartitionSpec("data", None),
        PartitionSpec("data", None),
        PartitionSpec("data", None, None),
        PartitionSpec("data"),
        PartitionSpec("data"),
    ]


def test_mrope_positions_propagate_through_model_worker_batch():
    mrope_positions = np.array(
        [
            [0, 10, 2],
            [0, 11, 2],
            [0, 12, 2],
        ],
        dtype=np.int32,
    )
    batch = ScheduleBatch(
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[
                    SimpleNamespace(
                        mm_inputs={
                            "mrope_positions": mrope_positions,
                        },
                        lora_id="0",
                    )
                ],
                input_ids=np.array([1, 151655, 2], dtype=np.int32),
                seq_lens=np.array([3], dtype=np.int32),
                out_cache_loc=np.array([1, 2, 3], dtype=np.int32),
                req_pool_indices=np.array([0], dtype=np.int32),
                prefix_lens=np.array([0], dtype=np.int32),
                extend_lens=np.array([3], dtype=np.int32),
                extend_logprob_start_lens=np.array([0], dtype=np.int32),
            )
        ],
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
    )
    batch._merge_sampling_info = lambda per_dp_bs_size, total_bs: None
    batch._merge_cache_loc = lambda *args: np.array([1, 2, 3], dtype=np.int32)

    mwb = batch.get_model_worker_batch(
        token_paddings=[3],
        bs_paddings=[1],
        cache_loc_paddings=[3],
        page_size=1,
    )

    np.testing.assert_array_equal(mwb.mrope_positions[:, :3], mrope_positions)


def test_multimodal_data_item_get_reads_common_and_model_specific_fields():
    item = MultimodalDataItem.from_dict(
        {
            "modality": "image",
            "feature": np.ones((2, 1), dtype=np.float32),
            "offsets": [(1, 2)],
            "image_grid_thw": np.array([[1, 2, 4]], dtype=np.int32),
        }
    )

    assert item.is_image()
    np.testing.assert_array_equal(item.get("feature"), np.ones((2, 1), dtype=np.float32))
    assert item.get("offsets") == [(1, 2)]
    np.testing.assert_array_equal(
        item.get("image_grid_thw"),
        np.array([[1, 2, 4]], dtype=np.int32),
    )
    assert item.get("missing", "fallback") == "fallback"


def test_mm_embed_plan_keeps_placeholder_count_separate_from_encode_rows():
    features = np.arange(24, dtype=np.float32).reshape(24, 1)
    grids = [(1, 2, 4), (1, 4, 4)]
    offsets = [(2, 3), (5, 8)]
    items = QwenVLProcessor._build_items(features, grids, offsets)
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=10,
    )
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
        dp_size=1,
        model_config=model_config,
        per_dp_token=10,
    )

    rounds = plan.rounds_by_modality[items[0].modality]
    assert len(rounds) == 2

    np.testing.assert_array_equal(rounds[0].encode_inputs.valid, np.array([8], dtype=np.int32))
    np.testing.assert_array_equal(rounds[1].encode_inputs.valid, np.array([16], dtype=np.int32))

    np.testing.assert_array_equal(np.flatnonzero(rounds[0].mask), np.array([2, 3]))
    np.testing.assert_array_equal(rounds[0].src_idx[2:4], np.array([0, 1], dtype=np.int32))

    np.testing.assert_array_equal(np.flatnonzero(rounds[1].mask), np.array([5, 6, 7, 8]))
    np.testing.assert_array_equal(rounds[1].src_idx[5:9], np.array([0, 1, 2, 3], dtype=np.int32))


def test_mm_embed_plan_normalizes_dict_mm_items():
    feature = np.arange(8, dtype=np.float32).reshape(8, 1)
    req = SimpleNamespace(
        mm_inputs={
            "mm_items": [
                {
                    "modality": "image",
                    "feature": feature,
                    "offsets": [(0, 1)],
                    "image_grid_thw": np.array([[1, 2, 4]], dtype=np.int32),
                }
            ]
        },
        extend_input_len=2,
    )
    vision_config = SimpleNamespace(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        num_heads=16,
        hidden_size=1280,
        rope_theta=10000.0,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=vision_config,
        ),
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
        dp_size=1,
        model_config=model_config,
        per_dp_token=2,
    )

    rounds = plan.rounds_by_modality[Modality.IMAGE]
    assert len(rounds) == 1
    np.testing.assert_array_equal(rounds[0].encode_inputs.valid, np.array([8], dtype=np.int32))
    np.testing.assert_array_equal(np.flatnonzero(rounds[0].mask), np.array([0, 1]))


def test_mm_embed_plan_returns_none_before_resolving_builder_without_images():
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.AUDIO,
                    feature=np.ones((4, 2), dtype=np.float32),
                )
            ]
        ),
        extend_input_len=4,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["NoVisionBuilderForAudioOnly"],
            vision_config=SimpleNamespace(),
        ),
    )

    plan = build_mm_embed_plan(
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
        dp_size=1,
        model_config=model_config,
        per_dp_token=4,
    )

    assert plan is None


def test_mm_embed_plan_fails_fast_when_qwen_vision_config_missing():
    features = np.arange(8, dtype=np.float32).reshape(8, 1)
    items = QwenVLProcessor._build_items(features, [(1, 2, 4)], [(0, 1)])
    req = SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=2,
    )
    model_config = SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
        ),
    )

    with pytest.raises(ValueError, match="vision_config"):
        build_mm_embed_plan(
            reqs_info=[ScheduleReqsInfo(reqs=[req])],
            dp_size=1,
            model_config=model_config,
            per_dp_token=2,
        )
