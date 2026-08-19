from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ScheduleBatch,
    ScheduleReqsInfo,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sgl_jax.srt.models.qwen3_vl import Qwen3VLVisionModel
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model import host_orchestration
from sgl_jax.srt.multimodal.in_model.embedding_pool import EmbeddingPool
from sgl_jax.srt.multimodal.in_model.host_orchestration import (
    _MergeMapping,
    build_multimodal_batch,
)
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    balance_lanes,
    encoder_num_lanes,
    pack_vision_inputs,
    replicate_across_mesh,
    run_mrope_vision_model,
)
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    vision_segment_ids_from_cu_seqlens,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import VisionShardSpecs
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor
from sgl_jax.srt.server_args import apply_multimodal_model_defaults

ARCH = "Qwen2_5_VLForConditionalGeneration"


class _TestInModelModel(InModelMultimodalContract):
    def __init__(self, input_embeddings=None):
        self.input_embeddings = input_embeddings

    def get_input_embeddings(self):
        if self.input_embeddings is None:
            return lambda input_ids: input_ids
        return lambda _: self.input_embeddings


def _vision_config(**overrides):
    values = {
        "patch_size": 1,
        "temporal_patch_size": 1,
        "in_channels": 1,
        "hidden_size": 4,
        "depth": 0,
        "intermediate_size": 8,
        "hidden_act": "silu",
        "num_heads": 1,
        "out_hidden_size": 4,
        "spatial_merge_size": 1,
        "fullatt_block_indexes": [],
        "window_size": 1,
        "rope_theta": 10000.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _qwen_config(**overrides):
    values = {
        "patch_size": 14,
        "window_size": 112,
        "spatial_merge_size": 2,
        "num_heads": 16,
        "hidden_size": 1280,
        "out_hidden_size": 1280,
    }
    values.update(overrides)
    return _vision_config(**values)


def _model_config(vision_config=None, arch=ARCH):
    return SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=[arch],
            vision_config=vision_config or _qwen_config(),
        ),
    )


def _visual(config=None, mesh=None, encoder_tp=False, input_buckets=(32,)):
    mesh = mesh or _mesh()
    with jax.set_mesh(mesh):
        return Qwen2_5_VisionTransformer(
            config or _vision_config(),
            jnp.float32,
            mesh=mesh,
            vision_tp=encoder_tp,
            input_buckets=input_buckets,
        )


def _build_items(features, grids, ranges, modality=Modality.IMAGE):
    key = "image_grid_thw" if modality == Modality.IMAGE else "video_grid_thw"
    return QwenVLProcessor._build_items(features, grids, ranges, modality, key)


def _items(grids, ranges, modality=Modality.IMAGE):
    rows = sum(int(np.prod(grid)) for grid in grids)
    features = np.arange(rows, dtype=np.float32).reshape(rows, 1)
    return _build_items(features, grids, ranges, modality)


def _pack_qwen2(visual, items):
    patches, grid_thw, output_indices = pack_vision_inputs(
        items,
        num_lanes=encoder_num_lanes(visual.mesh, visual.vision_tp),
        buckets=visual.input_buckets,
        merge_unit=visual.spatial_merge_unit,
    )
    batch_sharding = visual.specs.sharding(visual.specs.batch_axis)
    patches = jax.device_put(patches, batch_sharding)
    return patches, grid_thw, output_indices


def _qwen2_metadata(visual, grid_thw, capacity):
    return jax.device_put(
        visual._build_metadata(grid_thw, capacity),
        visual.specs.sharding(visual.specs.batch_axis),
    )


def _run_grid_vision(visual, items):
    return run_mrope_vision_model(
        visual,
        items,
        mesh=visual.mesh,
        num_lanes=encoder_num_lanes(visual.mesh, visual.vision_tp),
        buckets=visual.input_buckets,
        merge_unit=visual.spatial_merge_unit,
        rope_type="rope_3d",
    )


def _req(items, extend_len):
    return SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=extend_len,
        lora_id="0",
    )


def _batch(items, *, config=None, prefix=0, extend=None, per_dp_token=None):
    ends = [end for item in items for _, end in (item.placeholder_ranges or [])]
    max_end = max(ends, default=extend or 1)
    extend = max_end - prefix if extend is None else extend
    per_dp_token = extend if per_dp_token is None else per_dp_token
    info = ScheduleReqsInfo(
        reqs=[_req(items, extend)],
        prefix_lens=[prefix],
        extend_lens=[extend],
        seq_lens=np.array([prefix + extend], dtype=np.int32),
    )
    return build_multimodal_batch(
        [info],
        1,
        _model_config(config),
        per_dp_token,
    )


def _batch_dp(items_by_dp, *, config=None, per_dp_token):
    infos = []
    for items in items_by_dp:
        ends = [end for item in items for _, end in (item.placeholder_ranges or [])]
        extend = max(ends, default=1)
        infos.append(
            ScheduleReqsInfo(
                reqs=[_req(items, extend)] if items else [],
                prefix_lens=[0] if items else [],
                extend_lens=[extend] if items else [],
                seq_lens=np.asarray([extend] if items else [], dtype=np.int32),
            )
        )
    return build_multimodal_batch(
        infos,
        len(items_by_dp),
        _model_config(config),
        per_dp_token,
    )


def _mesh(dp=1, tp=1):
    count = dp * tp
    if len(jax.devices()) < count:
        pytest.skip(f"requires {count} devices")
    return Mesh(
        np.asarray(jax.devices()[:count]).reshape(dp, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _schedule_batch(req, model_config=None):
    input_ids = np.arange(req.extend_input_len, dtype=np.int32)
    info = ScheduleReqsInfo(
        reqs=[req],
        input_ids=input_ids,
        seq_lens=np.array([len(input_ids)], dtype=np.int32),
        out_cache_loc=np.arange(1, len(input_ids) + 1, dtype=np.int32),
        req_pool_indices=np.array([0], dtype=np.int32),
        prefix_lens=np.array([0], dtype=np.int32),
        extend_lens=np.array([len(input_ids)], dtype=np.int32),
        extend_logprob_start_lens=np.array([0], dtype=np.int32),
    )
    batch = ScheduleBatch(
        reqs_info=[info],
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        model_config=model_config,
    )
    batch._merge_sampling_info = lambda *_: None
    batch._merge_cache_loc = lambda *_: info.out_cache_loc
    return batch


@pytest.mark.parametrize(
    ("vision_tp", "expected_lanes"),
    [
        (False, [[0], [3], [4], [1, 2]]),
        (True, [[0, 1, 2], [3, 4]]),
    ],
)
def test_vision_batch_layout_uses_all_encoder_lanes(vision_tp, expected_lanes):
    lengths = (8, 4, 2, 7, 6)
    data_size, tensor_size = 2, 2
    num_lanes = data_size * (1 if vision_tp else tensor_size)
    lanes = balance_lanes(lengths, num_lanes)
    assert lanes == expected_lanes

    fake_mesh = SimpleNamespace(axis_names=("data", "tensor"))
    expected_axis = "data" if vision_tp else ("data", "tensor")
    assert PartitionSpec(VisionShardSpecs(fake_mesh, vision_tp).batch_axis, None) == PartitionSpec(
        expected_axis,
        None,
    )


def _assert_vision_precompile(visual):
    calls = []

    def encode(patches, grid_thw):
        calls.append((patches.shape, np.asarray(grid_thw).tolist()))
        return jnp.zeros(
            (
                patches.shape[0],
                patches.shape[1] // visual.spatial_merge_unit,
                1,
            )
        )

    with patch.object(type(visual), "encode", side_effect=encode):
        visual.precompile()

    return calls


def test_qwen2_vision_precompile_warms_configured_buckets():
    config = _vision_config(
        spatial_merge_size=2,
        window_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    assert _assert_vision_precompile(_visual(config=config, input_buckets=(4, 8))) == [
        ((1, 4, 1), [[[1, 2, 2]]]),
        ((1, 8, 1), [[[1, 2, 4]]]),
    ]


def test_qwen2_vision_rejects_unaligned_buckets():
    with pytest.raises(ValueError, match="positive multiples of 4"):
        _visual(config=_vision_config(spatial_merge_size=2), input_buckets=(3,))


@pytest.mark.parametrize("encoder_tp", [False, True])
def test_qwen2_global_batch_spmd(encoder_tp):
    mesh = _mesh(dp=2, tp=2)
    visual = _visual(
        mesh=mesh,
        encoder_tp=encoder_tp,
        input_buckets=(8,),
    )
    items = _items(
        [(1, 1, length) for length in (8, 4, 2, 7, 6)],
        [(0, length) for length in (8, 4, 2, 7, 6)],
    )
    patches, grid_thw, output_indices = _pack_qwen2(visual, items)
    metadata = _qwen2_metadata(visual, grid_thw, patches.shape[1])
    _, _, _, full_attn = metadata
    valid = full_attn.cu_seqlens[:, -1]

    if encoder_tp:
        np.testing.assert_array_equal(
            output_indices[:27],
            np.concatenate((np.arange(14), np.arange(16, 29))),
        )
        expected_valid = {
            mesh.devices[0, 0]: (14,),
            mesh.devices[0, 1]: (14,),
            mesh.devices[1, 0]: (13,),
            mesh.devices[1, 1]: (13,),
        }
        expected_patches = {
            mesh.devices[0, 0]: tuple(range(14)) + (0, 0),
            mesh.devices[0, 1]: tuple(range(14)) + (0, 0),
            mesh.devices[1, 0]: tuple(range(14, 27)) + (0, 0, 0),
            mesh.devices[1, 1]: tuple(range(14, 27)) + (0, 0, 0),
        }
        expected_spec = PartitionSpec("data")
    else:
        np.testing.assert_array_equal(
            output_indices[:27],
            np.concatenate(
                (
                    np.arange(8),
                    np.arange(24, 30),
                    np.arange(8, 15),
                    np.arange(16, 22),
                )
            ),
        )
        expected_valid = {
            mesh.devices[0, 0]: (8,),
            mesh.devices[0, 1]: (7,),
            mesh.devices[1, 0]: (6,),
            mesh.devices[1, 1]: (6,),
        }
        expected_patches = {
            mesh.devices[0, 0]: tuple(range(8)),
            mesh.devices[0, 1]: tuple(range(14, 21)) + (0,),
            mesh.devices[1, 0]: tuple(range(21, 27)) + (0, 0),
            mesh.devices[1, 1]: tuple(range(8, 14)) + (0, 0),
        }
        expected_spec = PartitionSpec(("data", "tensor"))

    assert patches.sharding.spec[0] == expected_spec[0]
    valid_shards = {
        shard.device: tuple(int(value) for value in np.asarray(shard.data).reshape(-1))
        for shard in valid.addressable_shards
    }
    patch_shards = {
        shard.device: tuple(int(value) for value in np.asarray(shard.data).reshape(-1))
        for shard in patches.addressable_shards
    }
    assert valid_shards == expected_valid
    assert patch_shards == expected_patches


@pytest.mark.parametrize("encoder_tp", [False, True])
def test_qwen2_get_image_feature_spmd(encoder_tp):
    mesh = _mesh(dp=2, tp=2)
    visual = _visual(
        mesh=mesh,
        encoder_tp=encoder_tp,
        input_buckets=(4,),
    )
    items = _items([(1, 1, 4), (1, 1, 2)], [(0, 4), (4, 6)])
    patches, grid_thw, output_indices = _pack_qwen2(visual, items)
    encoded = visual.encode(patches, grid_thw)
    assert encoded.sharding.is_fully_replicated
    assert encoded.sharding.spec == PartitionSpec(None, None, None)
    packed = _run_grid_vision(visual, items)
    assert packed.sharding.is_fully_replicated
    assert packed.sharding.device_set == set(mesh.devices.flat)
    expected_rows = encoder_num_lanes(visual.mesh, visual.vision_tp)
    assert packed.shape[0] == expected_rows * visual.input_buckets[0]
    expected = encoded.reshape(-1, encoded.shape[-1])[output_indices[output_indices >= 0]]
    np.testing.assert_allclose(packed[: len(expected)], expected)
    np.testing.assert_array_equal(packed[len(expected) :], 0)
    calls = 0

    class Model(_TestInModelModel):
        mesh = visual.mesh

        def get_multimodal_encode_funcs(self):
            return {Modality.IMAGE: self.encode}

        @staticmethod
        def encode(values):
            nonlocal calls
            calls += 1
            return _run_grid_vision(visual, values)

    running = jax.device_put(
        jnp.zeros((8, 4)),
        NamedSharding(mesh, PartitionSpec("data", None)),
    )
    runtime_items = _items([(1, 1, 4), (1, 1, 2)], [(0, 4), (0, 2)])
    args = (
        _batch_dp(([runtime_items[0]], [runtime_items[1]]), per_dp_token=4),
        jnp.zeros(8, dtype=jnp.int32),
        Model(running),
    )
    output, _ = host_orchestration.embed_multimodal_inputs(*args)
    assert output.sharding.spec == PartitionSpec("data", None)
    assert calls == 1


def test_replicate_across_mesh_reuses_rank_explicit_replication():
    mesh = _mesh(dp=2, tp=2)
    value = jax.device_put(
        jnp.zeros((4, 8, 16)),
        NamedSharding(mesh, PartitionSpec(None, None, None)),
    )

    assert replicate_across_mesh(value, mesh) is value


def test_qwen3_vision_precompile_warms_configured_buckets():
    config = _vision_config(
        spatial_merge_size=2,
        window_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        visual = Qwen3VLVisionModel(
            config,
            jnp.float32,
            mesh=mesh,
            input_buckets=(4, 8),
        )
    assert _assert_vision_precompile(visual) == [
        ((1, 4, 1), [[[1, 2, 2]]]),
        ((1, 8, 1), [[[1, 2, 4]]]),
    ]


def test_qwen3_vision_rejects_unaligned_buckets():
    with pytest.raises(ValueError, match="positive multiples of 4"):
        Qwen3VLVisionModel(
            _vision_config(spatial_merge_size=2),
            jnp.float32,
            mesh=_mesh(),
            input_buckets=(3,),
        )


@pytest.mark.skipif(
    "TPU" not in jax.devices()[0].device_kind,
    reason="Tiny test dims (head_dim=2) exercise the block-sparse kernel, which "
    "only lowers on TPU; the CPU interpret path rejects the degenerate shape.",
)
@pytest.mark.parametrize("encoder_tp", [False, True])
def test_qwen3_get_image_feature_spmd(encoder_tp):
    mesh = _mesh(dp=2, tp=2)
    config = _vision_config(
        num_position_embeddings=16,
        depth=1,
        deepstack_visual_indexes=[0],
        num_heads=2,
    )
    with jax.set_mesh(mesh):
        visual = Qwen3VLVisionModel(
            config,
            jnp.float32,
            mesh=mesh,
            tp=encoder_tp,
            input_buckets=(4,),
        )
    items = _items([(1, 1, 2), (1, 1, 4)], [(0, 2), (2, 6)])
    packed = _run_grid_vision(visual, items)
    assert packed.sharding.is_fully_replicated
    assert packed.sharding.device_set == set(mesh.devices.flat)
    expected_rows = encoder_num_lanes(visual.mesh, visual.vision_tp)
    assert packed.shape[0] == expected_rows * visual.input_buckets[0]
    assert packed.shape[1] == config.out_hidden_size * 2


def test_batch_separates_patch_and_placeholder_counts():
    items = _items([(1, 2, 4), (1, 4, 4)], [(2, 4), (5, 9)])
    tasks = _batch(items, extend=10, per_dp_token=10)[Modality.IMAGE]
    assert [task.item for task in tasks] == items
    assert [task.output_len for task in tasks] == [2, 4]
    assert [task.merge_mappings for task in tasks] == [
        (_MergeMapping(0, 2, 2),),
        (_MergeMapping(0, 5, 4),),
    ]


@pytest.mark.parametrize(
    ("prefix", "extend", "destination", "source"),
    [
        (0, 4, [2, 3], [0, 1]),
        (4, 4, [0, 1], [2, 3]),
        (6, 2, None, None),
    ],
)
def test_batch_clips_to_chunk_boundaries(prefix, extend, destination, source):
    items = _items([(1, 4, 4)], [(2, 6)])
    batch = _batch(items, prefix=prefix, extend=extend, per_dp_token=extend)
    if destination is None:
        assert batch is None
    else:
        mapping = batch[Modality.IMAGE][0].merge_mappings[0]
        np.testing.assert_array_equal(
            range(
                mapping.destination_start,
                mapping.destination_start + mapping.length,
            ),
            destination,
        )
        np.testing.assert_array_equal(
            range(mapping.source_start, mapping.source_start + mapping.length),
            source,
        )


def test_batch_preserves_encoder_offsets_across_chunks():
    items = _items([(1, 4, 4), (1, 4, 4)], [(2, 6), (6, 10)])
    tasks = _batch(items, prefix=4, extend=4)[Modality.IMAGE]
    assert tasks[0].merge_mappings == (_MergeMapping(2, 0, 2),)
    assert tasks[1].merge_mappings == (_MergeMapping(0, 2, 2),)


def test_batch_uses_global_token_indices_for_dp_ranks():
    rank0 = _req(_items([(1, 2, 4), (1, 4, 4)], [(0, 2), (3, 7)]), 8)
    rank1 = _req(
        _items([(1, 2, 4), (1, 2, 4), (1, 4, 4)], [(1, 3), (4, 6), (7, 11)]),
        12,
    )
    batch = build_multimodal_batch(
        [ScheduleReqsInfo(reqs=[rank0]), ScheduleReqsInfo(reqs=[rank1])],
        2,
        _model_config(),
        12,
    )
    tasks = batch[Modality.IMAGE]
    destinations = [
        [
            token
            for mapping in task.merge_mappings
            for token in range(
                mapping.destination_start,
                mapping.destination_start + mapping.length,
            )
        ]
        for task in tasks
    ]
    assert destinations == [[0, 1], [3, 4, 5, 6], [13, 14], [16, 17], [19, 20, 21, 22]]


def test_batch_routes_video_modality():
    video = _items([(1, 2, 4)], [(0, 2)], Modality.VIDEO)
    batch = _batch(video)
    assert tuple(batch) == (Modality.VIDEO,)
    assert batch[Modality.VIDEO][0].item is video[0]


@pytest.mark.parametrize("search_method", ["compare_all", "scan"])
def test_vision_backend_expands_cu_seqlens_to_segment_ids(search_method):
    cu_seqlens = jnp.asarray(
        [[0, 2, 5, 5], [0, 0, 0, 0], [0, 3, 3, 3]],
        dtype=jnp.int32,
    )
    segment_ids = vision_segment_ids_from_cu_seqlens(
        cu_seqlens,
        7,
        search_method=search_method,
    )
    expected = np.asarray(
        [[0, 0, 1, 1, 1, -1, -1], [-1] * 7, [0, 0, 0, -1, -1, -1, -1]],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(segment_ids.q, expected)
    np.testing.assert_array_equal(segment_ids.kv, expected)


def test_vision_weight_tp_specs():
    mesh = _mesh(tp=4)
    config = _vision_config(
        hidden_size=8, out_hidden_size=8, intermediate_size=16, num_heads=4, depth=1
    )
    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(
            config,
            jnp.float32,
            mesh=mesh,
            norm_eps=1e-6,
            vision_tp=True,
        )
    block = visual.blocks[0]
    assert block.attn.q_proj.weight.value.sharding.spec == PartitionSpec(None, "tensor")
    assert block.attn.proj.weight.value.sharding.spec == PartitionSpec("tensor", None)
    assert visual.merger.mlp_fc2.weight.value.sharding.spec == PartitionSpec("tensor", None)


def test_merge_preserves_unmasked_tokens():
    item = MultimodalDataItem(
        Modality.AUDIO,
        feature=np.ones((2, 1)),
        placeholder_ranges=[(0, 1), (2, 3)],
    )
    batch = build_multimodal_batch(
        [ScheduleReqsInfo(reqs=[_req([item], 3)])],
        1,
        _model_config(),
        3,
    )

    class Model(_TestInModelModel):
        def get_multimodal_encode_funcs(self):
            return {Modality.AUDIO: lambda _: jnp.array([[10.0, 11.0], [20.0, 21.0]])}

    running = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float32)
    output, _ = host_orchestration.embed_multimodal_inputs(
        batch,
        jnp.zeros(3, dtype=jnp.int32),
        Model(running),
    )
    np.testing.assert_array_equal(output, [[10, 11], [3, 4], [20, 21]])


def test_packed_gather_merge_preserves_data_sharding():
    """The fast gather path merges the encoder's packed output + deepstack."""
    mesh = _mesh(dp=2)
    rank0 = MultimodalDataItem(
        Modality.IMAGE, hash=0, feature=np.ones((2, 1)), placeholder_ranges=[(0, 2)]
    )
    rank1 = MultimodalDataItem(
        Modality.IMAGE, hash=1, feature=np.ones((2, 1)), placeholder_ranges=[(0, 2)]
    )
    batch = build_multimodal_batch(
        [
            ScheduleReqsInfo(reqs=[_req([rank0], 2)]),
            ScheduleReqsInfo(reqs=[_req([rank1], 2)]),
        ],
        2,
        _model_config(),
        2,
    )

    output = jnp.asarray([[10.0, 30.0], [11.0, 31.0], [20.0, 40.0], [21.0, 41.0]])

    class Model(_TestInModelModel):
        deepstack_visual_layers = 1

        def get_multimodal_encode_funcs(self):
            def encode(items):
                assert items == [rank0, rank1]
                return jax.device_put(output, NamedSharding(mesh, PartitionSpec(None, None)))

            return {Modality.IMAGE: encode}

    Model.mesh = mesh
    running = jax.device_put(jnp.zeros((4, 1)), NamedSharding(mesh, PartitionSpec("data", None)))
    out, ds = host_orchestration.embed_multimodal_inputs(
        batch, jnp.zeros(4, dtype=jnp.int32), Model(running)
    )
    np.testing.assert_array_equal(out[:, 0], [10, 11, 20, 21])
    np.testing.assert_array_equal(ds[0, :, 0], [30, 31, 40, 41])
    assert out.sharding.spec == PartitionSpec("data", None)
    assert ds.sharding.spec == PartitionSpec(None, "data", None)


def test_precompile_multimodal_inputs_matches_runtime_layout():
    mesh = _mesh()
    data = NamedSharding(mesh, PartitionSpec("data"))
    tokens = NamedSharding(mesh, PartitionSpec("data", None))

    class Model(_TestInModelModel):
        deepstack_visual_layers = 2

        def get_input_embeddings(self):
            return lambda _: expected

    Model.mesh = mesh
    expected = jax.device_put(jnp.ones((4, 8), jnp.float32), tokens)
    with patch.object(
        host_orchestration,
        "_gather_merge",
        wraps=host_orchestration._gather_merge,
    ) as merge:
        output, deepstack = host_orchestration.precompile_multimodal_inputs(
            jax.device_put(jnp.arange(4), data),
            Model(),
        )

    merge.assert_called_once()
    np.testing.assert_array_equal(output, 0)
    np.testing.assert_array_equal(deepstack, 0)
    assert output.dtype == jnp.float32
    assert deepstack.dtype == jnp.float32
    assert output.sharding.spec == PartitionSpec("data", None)
    assert deepstack.sharding.spec == PartitionSpec(None, "data", None)


def test_precompile_multimodal_inputs_covers_packed_and_pool_shapes():
    mesh = _mesh()
    data = NamedSharding(mesh, PartitionSpec("data"))
    tokens = NamedSharding(mesh, PartitionSpec("data", None))

    class Model(_TestInModelModel):
        def get_multimodal_embedding_packed_capacities(self):
            return (6, 10)

        def get_input_embeddings(self):
            return lambda _: jax.device_put(jnp.ones((4, 2), jnp.float32), tokens)

    Model.mesh = mesh
    pool = EmbeddingPool(
        num_pages=2,
        page_size=2,
        hidden=2,
        dtype=jnp.float32,
        mesh=mesh,
    )
    with (
        patch.object(
            host_orchestration,
            "_gather_merge",
            wraps=host_orchestration._gather_merge,
        ) as fresh_merge,
        patch.object(
            host_orchestration,
            "_gather_from_pool",
            wraps=host_orchestration._gather_from_pool,
        ) as pool_merge,
    ):
        host_orchestration.precompile_multimodal_inputs(
            jax.device_put(jnp.arange(4), data),
            Model(),
            pool,
        )

    assert [call.args[1].shape for call in fresh_merge.call_args_list] == [(6, 2), (10, 2)]
    pool_merge.assert_called_once()


def test_packed_gather_merge_handles_chunk_split():
    """A placeholder only partly visible in the chunk uses source_start > 0."""
    # Item spans tokens [0, 4); chunk covers [1, 4) so only 3 of its 4 tokens show.
    item = MultimodalDataItem(
        Modality.IMAGE, hash=7, feature=np.ones((4, 1)), placeholder_ranges=[(0, 4)]
    )
    batch = build_multimodal_batch(
        [ScheduleReqsInfo(reqs=[_req([item], 3)], prefix_lens=[1], extend_lens=[3])],
        1,
        _model_config(),
        3,
    )
    # Full item output is 4 rows; the chunk should pull rows 1..4 into dest 0..3.
    full = jnp.asarray([[10.0], [11.0], [12.0], [13.0]])  # [cap=4, H=1] on row 0

    class Model(_TestInModelModel):
        def get_multimodal_encode_funcs(self):
            return {Modality.IMAGE: lambda items: full}

    running = jnp.zeros((3, 1))
    out, _ = host_orchestration.embed_multimodal_inputs(
        batch, jnp.zeros(3, dtype=jnp.int32), Model(running)
    )
    np.testing.assert_array_equal(out[:, 0], [11, 12, 13])


def test_embedding_pool_skips_write_after_final_merge():
    item = MultimodalDataItem(
        Modality.IMAGE, hash=5, feature=np.ones((2, 1)), placeholder_ranges=[(0, 2)]
    )
    calls = 0
    output = jnp.asarray([[10.0], [11.0]])

    class Model(_TestInModelModel):
        def get_multimodal_encode_funcs(self):
            def encode(items):
                nonlocal calls
                calls += 1
                return output

            return {Modality.IMAGE: encode}

    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    args = (
        _batch([item]),
        jnp.zeros(2, dtype=jnp.int32),
        Model(jnp.zeros((2, 1), dtype=jnp.float32)),
        pool,
    )
    first, _ = host_orchestration.embed_multimodal_inputs(*args)
    second, _ = host_orchestration.embed_multimodal_inputs(*args)
    np.testing.assert_array_equal(first[:, 0], [10, 11])
    np.testing.assert_array_equal(second[:, 0], [10, 11])
    assert calls == 2
    assert pool.lookup(item.hash) is None


def test_embedding_pool_hit_matches_miss_with_deepstack():
    item = MultimodalDataItem(
        Modality.IMAGE, hash=6, feature=np.ones((2, 1)), placeholder_ranges=[(0, 2)]
    )
    output = jnp.asarray([[10.0, 30.0], [11.0, 31.0]])

    class Model(_TestInModelModel):
        deepstack_visual_layers = 1

        def get_multimodal_encode_funcs(self):
            return {Modality.IMAGE: lambda items: output}

    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=2, dtype=jnp.float32)
    model = Model(jnp.zeros((1, 1), dtype=jnp.float32))
    first, first_ds = host_orchestration.embed_multimodal_inputs(
        _batch([item], extend=1, per_dp_token=1),
        jnp.zeros(1, dtype=jnp.int32),
        model,
        pool,
    )
    second, second_ds = host_orchestration.embed_multimodal_inputs(
        _batch([item], prefix=1, extend=1, per_dp_token=1),
        jnp.zeros(1, dtype=jnp.int32),
        model,
        pool,
    )
    np.testing.assert_array_equal(first[:, 0], [10])
    np.testing.assert_array_equal(second[:, 0], [11])
    np.testing.assert_array_equal(first_ds[0, :, 0], [30])
    np.testing.assert_array_equal(second_ds[0, :, 0], [31])


def test_embedding_pool_reads_hit_before_miss_can_evict_it():
    hit = MultimodalDataItem(
        Modality.IMAGE, hash=10, feature=np.ones((1, 1)), placeholder_ranges=[(0, 1)]
    )
    miss = MultimodalDataItem(
        Modality.IMAGE, hash=20, feature=np.ones((1, 1)), placeholder_ranges=[(1, 2)]
    )
    partial_miss = MultimodalDataItem(
        Modality.IMAGE, hash=30, feature=np.ones((2, 1)), placeholder_ranges=[(2, 4)]
    )

    class Model(_TestInModelModel):
        def get_multimodal_encode_funcs(self):
            def encode(items):
                assert items == [miss, partial_miss]
                return jnp.asarray([[20.0], [30.0], [31.0]])

            return {Modality.IMAGE: encode}

    pool = EmbeddingPool(num_pages=1, page_size=2, hidden=1, dtype=jnp.float32)
    pool.write_packed((hit.hash,), jnp.asarray([[10.0]]), (1,))
    out, _ = host_orchestration.embed_multimodal_inputs(
        _batch([hit, miss, partial_miss], extend=3, per_dp_token=3),
        jnp.zeros(3, dtype=jnp.int32),
        Model(jnp.zeros((3, 1), dtype=jnp.float32)),
        pool,
    )

    np.testing.assert_array_equal(out[:, 0], [10, 20, 30])
    assert pool.lookup(hit.hash) is None
    assert pool.lookup(miss.hash) is None
    partial_entry = pool.lookup(partial_miss.hash)
    assert partial_entry is not None
    np.testing.assert_array_equal(
        np.asarray(pool.pages[int(partial_entry.page_ids[0]), :, 0]),
        [30, 31],
    )


def test_embedding_pool_reuses_full_item_across_chunks():
    """The encoder produces the full item once; a later chunk hits the pool."""
    item = MultimodalDataItem(
        Modality.IMAGE, hash=7, feature=np.ones((4, 1)), placeholder_ranges=[(0, 4)]
    )
    calls = 0
    output = jnp.asarray([[10.0], [11.0], [12.0], [13.0]])

    class Model(_TestInModelModel):
        def get_multimodal_encode_funcs(self):
            def encode(items):
                nonlocal calls
                calls += 1
                return output

            return {Modality.IMAGE: encode}

    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    model = Model(jnp.zeros((2, 1), dtype=jnp.float32))
    first, _ = host_orchestration.embed_multimodal_inputs(
        _batch([item], prefix=0, extend=2, per_dp_token=2),
        jnp.zeros(2, dtype=jnp.int32),
        model,
        pool,
    )
    second, _ = host_orchestration.embed_multimodal_inputs(
        _batch([item], prefix=2, extend=2, per_dp_token=2),
        jnp.zeros(2, dtype=jnp.int32),
        model,
        pool,
    )
    np.testing.assert_array_equal(first[:, 0], [10, 11])
    np.testing.assert_array_equal(second[:, 0], [12, 13])  # tail served from the pool
    assert calls == 1


@pytest.mark.parametrize(
    ("arch", "chunked", "radix", "mixed_chunk"),
    [
        (ARCH, 4096, False, True),
        ("Qwen3VLForConditionalGeneration", 4096, False, True),
        ("UnsupportedVLM", -1, True, False),
    ],
)
def test_multimodal_defaults_follow_capabilities(arch, chunked, radix, mixed_chunk):
    args = SimpleNamespace(
        disable_radix_cache=False,
        disable_overlap_schedule=False,
        chunked_prefill_size=4096,
        enable_mixed_chunk=True,
        limit_mm_data_per_request=None,
    )
    apply_multimodal_model_defaults(args, _model_config(arch=arch))
    assert (args.chunked_prefill_size, args.disable_radix_cache) == (chunked, radix)
    assert args.disable_overlap_schedule is False
    assert args.enable_mixed_chunk is mixed_chunk
    assert args.limit_mm_data_per_request == {"image": 16}


def test_model_runner_forward_embeds_multimodal_inputs():
    from sgl_jax.srt.model_executor.model_runner import ModelRunner

    input_ids = jnp.asarray([1], dtype=jnp.int32)
    multimodal_batch = {Modality.IMAGE: ()}
    model = object()
    embedding_pool = object()
    forward_batch = SimpleNamespace(
        bid=1,
        input_ids=input_ids,
        multimodal_batch=multimodal_batch,
        input_embedding=None,
        deepstack_visual_embedding=None,
        apply_for_deepstack=False,
    )
    expected = ("forwarded", 0)
    runner = SimpleNamespace(
        forward_pass_id=0,
        model=model,
        embedding_pool=embedding_pool,
        _forward_raw=lambda batch, metadata: expected,
    )

    with (
        patch(
            "sgl_jax.srt.model_executor.model_runner.embed_multimodal_inputs",
            autospec=True,
            return_value=("embedded", "deepstack"),
        ) as embed,
        patch("sgl_jax.srt.model_executor.model_runner.precision_tracer.start_batch_trace"),
        patch(
            "sgl_jax.srt.model_executor.model_runner.precision_tracer.set_current_forward_pass_id"
        ),
    ):
        result = ModelRunner.forward(runner, forward_batch, object())

    embed.assert_called_once()
    assert embed.call_args.args == ()
    assert embed.call_args.kwargs["multimodal_batch"] is multimodal_batch
    assert embed.call_args.kwargs["input_ids"] is input_ids
    assert embed.call_args.kwargs["multimodal_model"] is model
    assert embed.call_args.kwargs["embedding_pool"] is embedding_pool
    assert forward_batch.input_embedding == "embedded"
    assert forward_batch.deepstack_visual_embedding == "deepstack"
    assert forward_batch.apply_for_deepstack is True
    assert result == expected


def test_forward_batch_shards_input_embeddings():
    batch = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.EXTEND,
        input_ids=np.array([1]),
        real_input_ids_len=1,
        seq_lens=np.array([1]),
        out_cache_loc=np.array([1]),
        req_pool_indices=np.array([0]),
        sampling_info=None,
        positions=np.array([0]),
        cache_loc=np.array([1]),
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_seq_lens=np.array([1]),
        extend_prefix_lens=np.array([0]),
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        logits_indices=np.array([0]),
        real_bs=1,
        real_bs_per_dp=[1],
        input_embedding=np.ones((1, 4)),
    )
    runner = SimpleNamespace(
        mesh=Mesh(np.asarray(jax.devices()[:1]), ("data",)),
        attn_backend=None,
        model_config=SimpleNamespace(
            is_embedding=False,
            hf_config=SimpleNamespace(architectures=[]),
        ),
    )
    specs = []
    with patch(
        "sgl_jax.srt.model_executor.forward_batch_info.device_array",
        side_effect=lambda values, sharding: specs.append(sharding.spec) or values,
    ):
        ForwardBatch.init_new(batch, runner)
    assert PartitionSpec("data", None) in specs


def test_mrope_positions_reach_worker_batch():
    positions = np.array([[0, 10, 2], [0, 11, 2], [0, 12, 2]], dtype=np.int32)
    req = SimpleNamespace(mm_inputs={"mrope_positions": positions}, extend_input_len=3, lora_id="0")
    worker_batch = _schedule_batch(req).get_model_worker_batch(
        token_paddings=[3],
        bs_paddings=[1],
        cache_loc_paddings=[3],
        page_size=1,
    )
    np.testing.assert_array_equal(worker_batch.mrope_positions[:, :3], positions)


def test_mrope_positions_continue_past_prompt_after_retraction():
    positions = np.array(
        [[0, 1, 2, 30, 31], [0, 1, 2, 40, 41], [0, 1, 2, 50, 51]],
        dtype=np.int32,
    )
    req = SimpleNamespace(
        mm_inputs={"mrope_positions": positions, "mrope_position_delta": -2},
        extend_input_len=5,
        lora_id="0",
    )
    info = ScheduleReqsInfo(
        reqs=[req],
        input_ids=np.arange(5, dtype=np.int32),
        seq_lens=np.array([8], dtype=np.int32),
        out_cache_loc=np.arange(1, 6, dtype=np.int32),
        req_pool_indices=np.array([0], dtype=np.int32),
        prefix_lens=np.array([3], dtype=np.int32),
        extend_lens=np.array([5], dtype=np.int32),
        extend_logprob_start_lens=np.array([0], dtype=np.int32),
    )
    batch = ScheduleBatch(
        reqs_info=[info],
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        model_config=None,
    )

    merged = batch._merge_multimodal(per_dp_token_size=5, total_token_size=5)

    np.testing.assert_array_equal(
        merged["mrope_positions"],
        np.array(
            [[30, 31, 3, 4, 5], [40, 41, 3, 4, 5], [50, 51, 3, 4, 5]],
            dtype=np.int32,
        ),
    )


def test_overlap_copy_rebuilds_multimodal_batch_from_requests():
    items = _items([(1, 2, 4)], [(1, 3)])
    batch = _schedule_batch(_req(items, 3), _model_config())
    worker_batch = batch.get_model_worker_batch(
        token_paddings=[3],
        bs_paddings=[1],
        cache_loc_paddings=[3],
        page_size=1,
    )
    copied = batch.copy()
    rebuilt = build_multimodal_batch(copied.reqs_info, 1, _model_config(), 3)
    assert Modality.IMAGE in worker_batch.multimodal_batch
    assert getattr(copied, "multimodal_batch", None) is None
    assert Modality.IMAGE in rebuilt


def test_mixed_chunk_keeps_multimodal_items():
    item = _items([(1, 4, 4)], [(1, 5)])[0]
    batch = _schedule_batch(_req([item], 2), _model_config())
    batch.forward_mode = ForwardMode.MIXED
    worker_batch = batch.get_model_worker_batch(
        token_paddings=[2],
        bs_paddings=[1],
        cache_loc_paddings=[2],
        page_size=1,
    )

    task = worker_batch.multimodal_batch[Modality.IMAGE][0]
    assert task.item is item
    assert task.merge_mappings == (_MergeMapping(0, 1, 1),)


def _assert_no_grid_layout_planning(jaxpr):
    text = str(jaxpr)
    for primitive in ("cumsum", "repeat", "scatter", "sort"):
        assert f"= {primitive}[" not in text


def test_qwen2_metadata_is_host_planned_and_bucket_stable():
    config = _vision_config(
        spatial_merge_size=2,
        window_size=4,
        depth=2,
        fullatt_block_indexes=[1],
    )
    visual = _visual(config=config, input_buckets=(32,))
    first = _items([(1, 4, 6)], [(0, 6)])
    patches, grid_thw, output_indices = _pack_qwen2(visual, first)
    metadata = _qwen2_metadata(visual, grid_thw, patches.shape[1])
    indices, position_ids, window_attn, full_attn = metadata
    indices = np.asarray(indices)
    position_ids = np.asarray(position_ids)

    assert window_attn.max_seq_len == 16
    assert full_attn.max_seq_len == 32
    np.testing.assert_array_equal(output_indices[:6], np.arange(6))
    assert position_ids.shape == (1, 32, 2)
    np.testing.assert_array_equal(indices[0, :, 0], [0, 1, 3, 4, 2, 5, 6, 7])
    np.testing.assert_array_equal(indices[0, :, 1], [0, 1, 4, 2, 3, 5, 6, 7])
    np.testing.assert_array_equal(
        position_ids[0, [0, 4, 8, 12, 16, 20]],
        [[0, 0], [0, 2], [2, 0], [2, 2], [0, 4], [2, 4]],
    )
    # window layout at [:, 0], full-frame at [:, 1]; tails repeat the final end.
    np.testing.assert_array_equal(
        np.asarray(window_attn.cu_seqlens)[0], [0, 16, 24, 24, 24, 24, 24, 24, 24]
    )
    np.testing.assert_array_equal(
        np.asarray(full_attn.cu_seqlens)[0], [0, 24, 24, 24, 24, 24, 24, 24, 24]
    )
    np.testing.assert_array_equal(
        visual._build_metadata(np.zeros((1, 1, 3), dtype=np.int32), 32)[2].cu_seqlens,
        np.zeros((1, 9), dtype=np.int32),
    )
    _assert_no_grid_layout_planning(
        jax.make_jaxpr(lambda p, *m: visual._forward(p, *m))(patches, *metadata)
    )

    jax.block_until_ready(visual.encode(patches, grid_thw))
    cache_size = visual._encode_jit._cache_size()
    second = _items([(1, 4, 4), (2, 2, 2)], [(0, 4), (4, 6)])
    second_patches, second_grid_thw, _ = _pack_qwen2(visual, second)
    second_metadata = _qwen2_metadata(visual, second_grid_thw, second_patches.shape[1])
    np.testing.assert_array_equal(
        np.asarray(second_metadata[3].cu_seqlens)[0],
        [0, 16, 20, 24, 24, 24, 24, 24, 24],
    )
    jax.block_until_ready(visual.encode(second_patches, second_grid_thw))
    assert visual._encode_jit._cache_size() == cache_size


def test_qwen3_metadata_is_host_planned_and_bucket_stable():
    config = _vision_config(
        spatial_merge_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        visual = Qwen3VLVisionModel(config, jnp.float32, mesh=mesh, input_buckets=(32,))
    first = _items([(2, 2, 2)], [(0, 2)])
    features, grid_thw, output_indices = _pack_qwen2(visual, first)
    inputs = visual._build_metadata(grid_thw, features.shape[1])
    pos_indices, pos_weights, position_ids, metadata = inputs

    np.testing.assert_array_equal(output_indices[:2], np.arange(2))
    assert [
        np.asarray(features).shape,
        np.asarray(pos_indices).shape,
        np.asarray(pos_weights).shape,
        np.asarray(position_ids).shape,
        np.asarray(metadata.cu_seqlens).shape,
    ] == [
        (1, 32, 1),
        (1, 4, 32),
        (1, 4, 32),
        (1, 32, 2),
        (1, 9),
    ]
    np.testing.assert_array_equal(
        np.asarray(position_ids)[0, :8],
        [[0, 0], [0, 1], [1, 0], [1, 1]] * 2,
    )
    np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens)[0], [0, 4, 8, 8, 8, 8, 8, 8, 8])
    np.testing.assert_array_equal(
        visual._lane_metadata([], 32)[-1],
        np.zeros(9, dtype=np.int32),
    )
    np.testing.assert_allclose(np.asarray(pos_weights)[0, :, :8].sum(axis=0), 1.0)
    np.testing.assert_array_equal(np.asarray(pos_indices)[0, 0, :8], [0, 3, 12, 15] * 2)
    _assert_no_grid_layout_planning(jax.make_jaxpr(visual._forward)(features, *inputs))

    jax.block_until_ready(visual.encode(features, grid_thw))
    cache_size = visual._encode_jit._cache_size()
    second_features, second_grid_thw, _ = _pack_qwen2(visual, _items([(1, 2, 4)], [(0, 2)]))
    second_inputs = visual._build_metadata(second_grid_thw, second_features.shape[1])
    assert not np.array_equal(
        np.asarray(metadata.cu_seqlens),
        np.asarray(second_inputs[-1].cu_seqlens),
    )
    jax.block_until_ready(visual.encode(second_features, second_grid_thw))
    assert visual._encode_jit._cache_size() == cache_size
