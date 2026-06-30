from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ScheduleBatch,
    ScheduleReqsInfo,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


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


def test_mrope_positions_propagate_through_model_worker_batch():
    item = SimpleNamespace(modality="image", offsets=[(1, 1)])
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
                            "mm_items": [item],
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
