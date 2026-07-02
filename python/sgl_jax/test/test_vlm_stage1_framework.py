from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.mm_utils import merge_jit
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
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
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
from sgl_jax.srt.server_args import apply_multimodal_model_defaults


def _two_data_devices():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two devices for real data-axis sharding")
    return np.array(devices[:2])


def test_vision_transformer_uses_default_norm_eps_when_hf_vision_config_omits_it():
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=3,
        hidden_size=4,
        depth=1,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    with jax.set_mesh(mesh):
        Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=None,
            norm_eps=1e-6,
        )


def test_vision_transformer_encode_jit_accepts_unhashable_vision_config():
    class UnhashableVisionConfig(SimpleNamespace):
        __hash__ = None

    vision_config = UnhashableVisionConfig(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=0,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    meta = Qwen25VLVisionMetadata(
        window_index=jnp.zeros((1, 2), dtype=jnp.int32),
        cu_window_seqlens=jnp.array([[2]], dtype=jnp.int32),
        rotary_pos_emb=jnp.zeros((1, 2, 2), dtype=jnp.float32),
    )

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        features = visual.encode_jit(
            jnp.ones((1, 2, 1), dtype=jnp.float32),
            meta,
            jnp.array([2], dtype=jnp.int32),
        )

    assert features.shape == (1, 2, 4)


def test_vision_transformer_encode_jit_uses_reshard_for_explicit_mesh(monkeypatch):
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=0,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    meta = Qwen25VLVisionMetadata(
        window_index=jnp.zeros((1, 2), dtype=jnp.int32),
        cu_window_seqlens=jnp.array([[2]], dtype=jnp.int32),
        rotary_pos_emb=jnp.zeros((1, 2, 2), dtype=jnp.float32),
    )

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )

        def fail_with_sharding_constraint(*args, **kwargs):
            raise AssertionError("with_sharding_constraint must not be used in Qwen vision encode")

        reshard_specs = []
        original_reshard = jax.sharding.reshard

        def record_reshard(x, out_sharding):
            reshard_specs.append(tuple(out_sharding.spec))
            return original_reshard(x, out_sharding)

        monkeypatch.setattr(jax.lax, "with_sharding_constraint", fail_with_sharding_constraint)
        monkeypatch.setattr(jax.sharding, "reshard", record_reshard)

        features = visual.encode_jit(
            jnp.ones((1, 2, 1), dtype=jnp.float32),
            meta,
            jnp.array([2], dtype=jnp.int32),
        )

    assert features.shape == (1, 2, 4)
    assert reshard_specs == [
        ("data", None, None, None, None, None),
        ("data", None, None),
        ("data", None, None),
    ]


def test_vision_transformer_encode_binds_mesh_for_sharded_inputs_without_callsite_context(
    monkeypatch,
):
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=2,
        intermediate_size=16,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=2,
        fullatt_block_indexes=[1],
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        for block in visual.blocks:
            block.attn.attn_backend = None

    def fake_vision_attention(backend, q, k, v, seg):
        return jnp.zeros_like(q)

    monkeypatch.setattr(
        "sgl_jax.srt.models.qwen2_5_vl._vision_attention",
        fake_vision_attention,
    )

    plan = MultimodalEmbedPlan(
        rounds_by_modality={
            Modality.IMAGE: [
                EmbedRound(
                    encode_inputs=VisionEncodeInputs(
                        pixels=np.ones((1, 4, 1), dtype=np.float32),
                        valid=np.array([4], dtype=np.int32),
                        meta=Qwen25VLVisionMetadata(
                            window_index=np.array([[0]], dtype=np.int32),
                            cu_window_seqlens=np.array([[4]], dtype=np.int32),
                            rotary_pos_emb=np.zeros((1, 4, 2), dtype=np.float32),
                        ),
                    ),
                    src_idx=np.zeros((1,), dtype=np.int32),
                    mask=np.zeros((1,), dtype=np.bool_),
                )
            ]
        }
    )
    _device_put_embed_plan(plan, mesh)
    enc = plan.rounds_by_modality[Modality.IMAGE][0].encode_inputs

    features = visual.encode(enc.pixels, enc.meta, enc.valid)

    assert features.shape == (1, 1, 4)


def test_vision_patch_embed_calls_conv_with_single_batch_dim(monkeypatch):
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=0,
        intermediate_size=8,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=1,
        fullatt_block_indexes=[],
    )
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    meta = Qwen25VLVisionMetadata(
        window_index=jnp.tile(jnp.arange(3, dtype=jnp.int32)[None, :], (2, 1)),
        cu_window_seqlens=jnp.array([[3], [3]], dtype=jnp.int32),
        rotary_pos_emb=jnp.zeros((2, 3, 2), dtype=jnp.float32),
    )

    seen_input_shapes = []
    original_call = nnx.Conv.__call__

    def record_conv_input_shape(self, inputs, *args, **kwargs):
        seen_input_shapes.append(inputs.shape)
        return original_call(self, inputs, *args, **kwargs)

    monkeypatch.setattr(nnx.Conv, "__call__", record_conv_input_shape)

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        features = visual.encode_jit(
            jnp.ones((2, 3, 1), dtype=jnp.float32),
            meta,
            jnp.array([3, 3], dtype=jnp.int32),
        )

    assert features.shape == (2, 3, 4)
    assert seen_input_shapes == [(6, 1, 1, 1, 1)]


def test_vision_encode_runs_on_real_dp2_data_mesh(monkeypatch):
    mesh = Mesh(_two_data_devices(), ("data",), axis_types=(AxisType.Explicit,))
    vision_config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=4,
        depth=2,
        intermediate_size=16,
        hidden_act="silu",
        num_heads=1,
        out_hidden_size=4,
        spatial_merge_size=2,
        fullatt_block_indexes=[1],
    )

    def fake_vision_attention(backend, q, k, v, seg):
        return jnp.zeros_like(q)

    monkeypatch.setattr(
        "sgl_jax.srt.models.qwen2_5_vl._vision_attention",
        fake_vision_attention,
    )

    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config=vision_config,
            dtype=jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
        )
        for block in visual.blocks:
            block.attn.attn_backend = None

    plan = MultimodalEmbedPlan(
        rounds_by_modality={
            Modality.IMAGE: [
                EmbedRound(
                    encode_inputs=VisionEncodeInputs(
                        pixels=np.ones((2, 8, 1), dtype=np.float32),
                        valid=np.array([8, 4], dtype=np.int32),
                        meta=Qwen25VLVisionMetadata(
                            window_index=np.array([[1, 0], [0, 1]], dtype=np.int32),
                            cu_window_seqlens=np.array([[8], [4]], dtype=np.int32),
                            rotary_pos_emb=np.zeros((2, 8, 2), dtype=np.float32),
                        ),
                    ),
                    src_idx=np.zeros((8,), dtype=np.int32),
                    mask=np.zeros((8,), dtype=np.bool_),
                )
            ]
        }
    )
    _device_put_embed_plan(plan, mesh)
    enc = plan.rounds_by_modality[Modality.IMAGE][0].encode_inputs

    features = visual.encode(enc.pixels, enc.meta, enc.valid)

    assert features.shape == (2, 2, 4)
    assert tuple(features.sharding.spec) == ("data", None, None)


def test_merge_jit_consumes_dp_leading_features():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    running = jnp.zeros((2, 3), dtype=jnp.float32)
    features = jnp.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=jnp.float32)
    src_idx = jnp.array([0, 1], dtype=jnp.int32)
    mask = jnp.array([True, True])

    out = merge_jit(mesh, running, features, src_idx, mask)

    np.testing.assert_array_equal(np.asarray(out), np.asarray(features[0]))


def test_merge_jit_uses_rank_local_features_on_real_dp2_mesh():
    mesh = Mesh(_two_data_devices(), ("data",), axis_types=(AxisType.Explicit,))
    running = np.arange(12, dtype=np.float32).reshape(6, 2)
    features = np.array(
        [
            [[10.0, 11.0], [20.0, 21.0]],
            [[100.0, 101.0], [200.0, 201.0]],
        ],
        dtype=np.float32,
    )
    src_idx = np.array([1, 0, 0, 0, 0, 1], dtype=np.int32)
    mask = np.array([True, True, False, True, False, True])

    running_d = jax.device_put(running, NamedSharding(mesh, PartitionSpec("data", None)))
    features_d = jax.device_put(features, NamedSharding(mesh, PartitionSpec("data", None, None)))
    src_idx_d = jax.device_put(src_idx, NamedSharding(mesh, PartitionSpec("data")))
    mask_d = jax.device_put(mask, NamedSharding(mesh, PartitionSpec("data")))

    out = merge_jit(mesh, running_d, features_d, src_idx_d, mask_d)

    expected = running.copy()
    expected[0] = features[0, 1]
    expected[1] = features[0, 0]
    expected[3] = features[1, 0]
    expected[5] = features[1, 1]
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_device_put_embed_plan_places_qwen_metadata_data_leading():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",), axis_types=(AxisType.Explicit,))
    plan = MultimodalEmbedPlan(
        rounds_by_modality={
            Modality.IMAGE: [
                EmbedRound(
                    encode_inputs=VisionEncodeInputs(
                        pixels=np.ones((1, 4, 1), dtype=np.float32),
                        valid=np.array([4], dtype=np.int32),
                        meta=Qwen25VLVisionMetadata(
                            window_index=np.array([[0]], dtype=np.int32),
                            cu_window_seqlens=np.array([[4]], dtype=np.int32),
                            rotary_pos_emb=np.zeros((1, 4, 2), dtype=np.float32),
                        ),
                    ),
                    src_idx=np.zeros((2,), dtype=np.int32),
                    mask=np.zeros((2,), dtype=np.bool_),
                )
            ]
        }
    )

    _device_put_embed_plan(plan, mesh)
    rnd = plan.rounds_by_modality[Modality.IMAGE][0]
    enc = rnd.encode_inputs

    assert tuple(enc.pixels.sharding.spec) == ("data", None, None)
    assert tuple(enc.valid.sharding.spec) == ("data",)
    assert tuple(enc.meta.window_index.sharding.spec) == ("data", None)
    assert tuple(enc.meta.cu_window_seqlens.sharding.spec) == ("data", None)
    assert tuple(enc.meta.rotary_pos_emb.sharding.spec) == ("data", None, None)
    assert tuple(rnd.src_idx.sharding.spec) == ("data",)
    assert tuple(rnd.mask.sharding.spec) == ("data",)


def test_multimodal_model_defaults_disable_unsupported_scheduler_features():
    server_args = SimpleNamespace(
        disable_radix_cache=False,
        disable_overlap_schedule=False,
        chunked_prefill_size=4096,
        enable_mixed_chunk=True,
        limit_mm_data_per_request=None,
    )
    model_config = SimpleNamespace(is_multimodal=True)

    apply_multimodal_model_defaults(server_args, model_config)

    assert server_args.disable_radix_cache is True
    assert server_args.disable_overlap_schedule is True
    assert server_args.chunked_prefill_size == -1
    assert server_args.enable_mixed_chunk is False
    assert server_args.limit_mm_data_per_request == {"image": 16}


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


def test_schedule_batch_translates_image_offsets_to_extend_window():
    features = np.arange(16, dtype=np.float32).reshape(16, 1)
    items = QwenVLProcessor._build_items(features, [(1, 4, 4)], [(5, 8)])
    req = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=items))
    batch = SimpleNamespace(
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[req],
                seq_lens=np.array([10], dtype=np.int32),
                prefix_lens=np.array([3], dtype=np.int32),
            )
        ],
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["Qwen2_5_VLForConditionalGeneration"],
                vision_config=SimpleNamespace(
                    patch_size=14,
                    window_size=112,
                    spatial_merge_size=2,
                    fullatt_block_indexes=[],
                    num_heads=16,
                    hidden_size=1280,
                    rope_theta=10000.0,
                ),
            )
        ),
    )
    batch.contains_mm_inputs = lambda: True

    plan = ScheduleBatch._build_extend_mm_embed_plan(batch, per_dp_token_size=8)

    rnd = plan.rounds_by_modality[Modality.IMAGE][0]
    np.testing.assert_array_equal(np.flatnonzero(rnd.mask), np.array([2, 3, 4, 5]))
    np.testing.assert_array_equal(rnd.src_idx[2:6], np.array([0, 1, 2, 3], dtype=np.int32))


def test_schedule_batch_rejects_image_span_crossing_extend_window():
    features = np.arange(16, dtype=np.float32).reshape(16, 1)
    items = QwenVLProcessor._build_items(features, [(1, 4, 4)], [(5, 8)])
    req = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=items))
    batch = SimpleNamespace(
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[req],
                seq_lens=np.array([10], dtype=np.int32),
                prefix_lens=np.array([6], dtype=np.int32),
            )
        ],
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    batch.contains_mm_inputs = lambda: True

    with pytest.raises(ValueError, match="chunked prefill for image spans is not supported"):
        ScheduleBatch._build_extend_mm_embed_plan(batch, per_dp_token_size=8)


def test_schedule_batch_rejects_image_plan_for_mixed_mode():
    features = np.arange(16, dtype=np.float32).reshape(16, 1)
    items = QwenVLProcessor._build_items(features, [(1, 4, 4)], [(0, 3)])
    req = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=items))
    batch = SimpleNamespace(
        dp_size=1,
        forward_mode=ForwardMode.MIXED,
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
    )
    batch.contains_mm_inputs = lambda: True

    with pytest.raises(ValueError, match="regular EXTEND mode"):
        ScheduleBatch._build_extend_mm_embed_plan(batch, per_dp_token_size=8)


def test_schedule_batch_skips_image_plan_for_decode_mode():
    features = np.arange(16, dtype=np.float32).reshape(16, 1)
    items = QwenVLProcessor._build_items(features, [(1, 4, 4)], [(0, 3)])
    req = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=items))
    batch = SimpleNamespace(
        dp_size=1,
        forward_mode=ForwardMode.DECODE,
        reqs_info=[ScheduleReqsInfo(reqs=[req])],
    )
    batch.contains_mm_inputs = lambda: True

    assert ScheduleBatch._build_extend_mm_embed_plan(batch, per_dp_token_size=8) is None


def test_schedule_batch_rejects_image_plan_without_extend_window_metadata():
    features = np.arange(16, dtype=np.float32).reshape(16, 1)
    items = QwenVLProcessor._build_items(features, [(1, 4, 4)], [(0, 3)])
    req = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=items))
    batch = SimpleNamespace(
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        reqs_info=[ScheduleReqsInfo(reqs=[req], seq_lens=None, prefix_lens=None)],
    )
    batch.contains_mm_inputs = lambda: True

    with pytest.raises(ValueError, match="requires seq_lens and prefix_lens"):
        ScheduleBatch._build_extend_mm_embed_plan(batch, per_dp_token_size=8)


def test_schedule_batch_rejects_image_plan_with_mismatched_window_metadata():
    features = np.arange(16, dtype=np.float32).reshape(16, 1)
    items = QwenVLProcessor._build_items(features, [(1, 4, 4)], [(0, 3)])
    req0 = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=items))
    req1 = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=[]))
    batch = SimpleNamespace(
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        reqs_info=[
            ScheduleReqsInfo(
                reqs=[req0, req1],
                seq_lens=np.array([4], dtype=np.int32),
                prefix_lens=np.array([0, 0], dtype=np.int32),
            )
        ],
    )
    batch.contains_mm_inputs = lambda: True

    with pytest.raises(ValueError, match="one seq_len and prefix_len per request"):
        ScheduleBatch._build_extend_mm_embed_plan(batch, per_dp_token_size=8)


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
