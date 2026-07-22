import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers import mm_utils
from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.mm_utils import build_mm_embed_plan, merge_jit
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    ScheduleBatch,
    ScheduleReqsInfo,
)
from sgl_jax.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    _device_put_embed_plan,
)
from sgl_jax.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionTransformer,
    _segment_ids_from_cu_seqlens,
    _vision_attention,
)
from sgl_jax.srt.models.vision_metadata.qwen2_5_vl import (
    Qwen25VLVisionEncoderPlugin,
    Qwen25VLVisionMetadata,
)
from sgl_jax.srt.multimodal.common import mm_plan
from sgl_jax.srt.multimodal.common.in_model_plan_builder import (
    register_in_model_plan_builder,
    resolve_in_model_plan_builder,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.common.vision_plan_builder import (
    InModelVisionPlanBuilder,
    MergeSlice,
    VisionEncodeInputs,
    _ceil_to_bucket,
)
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    VisionFlashAttentionBackend,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import VisionShardSpecs
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor
from sgl_jax.srt.server_args import apply_multimodal_model_defaults

ARCH = "Qwen2_5_VLForConditionalGeneration"


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


def _plugin(config=None):
    return Qwen25VLVisionEncoderPlugin(_model_config(config))


def _build_items(features, grids, ranges, modality=Modality.IMAGE):
    key = "image_grid_thw" if modality == Modality.IMAGE else "video_grid_thw"
    return QwenVLProcessor._build_items(features, grids, ranges, modality, key)


def _items(grids, ranges, modality=Modality.IMAGE):
    rows = sum(int(np.prod(grid)) for grid in grids)
    features = np.arange(rows, dtype=np.float32).reshape(rows, 1)
    return _build_items(features, grids, ranges, modality)


def _req(items, extend_len):
    return SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=extend_len,
        lora_id="0",
    )


def _plan(items, *, config=None, prefix=0, extend=None, per_dp_token=None, tp_size=1):
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
    return build_mm_embed_plan(
        [info],
        1,
        _model_config(config),
        per_dp_token,
        tp_size=tp_size,
    )


def _assert_lane(merge, dp_rank, tp_rank, dst, src):
    mask = np.asarray(merge.mask[dp_rank, tp_rank])
    np.testing.assert_array_equal(np.flatnonzero(mask), dst)
    np.testing.assert_array_equal(np.asarray(merge.src_idx[dp_rank, tp_rank])[mask], src)


def _mesh(dp=1, tp=1):
    count = dp * tp
    if len(jax.devices()) < count:
        pytest.skip(f"requires {count} devices")
    return Mesh(
        np.asarray(jax.devices()[:count]).reshape(dp, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _flat_metadata(lanes, patches):
    return Qwen25VLVisionMetadata(
        window_index=jnp.tile(jnp.arange(patches)[None], (lanes, 1)),
        cu_window_seqlens=jnp.full((lanes, 1), patches, dtype=jnp.int32),
        rotary_pos_emb=jnp.zeros((lanes, patches, 2), dtype=jnp.float32),
        cu_image_seqlens=jnp.full((lanes, 1), patches, dtype=jnp.int32),
    )


def _host_plan(dp=1, tp=1, patches=4, tokens=2, mask=False):
    metadata = _flat_metadata(dp * tp, patches)
    metadata = jax.tree.map(lambda x: np.asarray(x).reshape(dp, tp, *x.shape[1:]), metadata)
    return {
        Modality.IMAGE: mm_plan.ModalityEmbedBatch(
            encode_inputs=VisionEncodeInputs(
                patches=np.ones((dp, tp, patches, 1), dtype=np.float32),
                valid=np.full((dp, tp), patches, dtype=np.int32),
                meta=metadata,
            ),
            merge=mm_plan.DeviceMergePlan(
                src_idx=np.zeros((dp, tp, tokens), dtype=np.int32),
                mask=np.full((dp, tp, tokens), mask, dtype=np.bool_),
            ),
        )
    }


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


def test_mm_plan_contract():
    assert dataclasses.astuple(MergeSlice(4, 9, 2)) == (4, 9, 2)
    plan = _host_plan()
    batch = plan[Modality.IMAGE]
    assert isinstance(batch, mm_plan.ModalityEmbedBatch)
    assert batch.merge.src_idx.shape == (1, 1, 2)


def test_plan_builder_registry():
    class Builder:
        def __init__(self, config):
            self.config = config

    register_in_model_plan_builder("TestArchitecture", Builder)
    config = SimpleNamespace(hf_config=SimpleNamespace(architectures=["TestArchitecture"]))
    assert isinstance(resolve_in_model_plan_builder(config), Builder)
    assert resolve_in_model_plan_builder(_model_config(arch="MissingArchitecture")) is None
    assert isinstance(resolve_in_model_plan_builder(_model_config()), InModelVisionPlanBuilder)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _FakeInputs:
    values: object
    lengths: object

    def tree_flatten(self):
        return (self.values, self.lengths), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)


def test_embed_mm_inputs_accepts_opaque_encoder_inputs(monkeypatch):
    inputs = _FakeInputs(
        values=jnp.arange(24, dtype=jnp.float32).reshape(1, 2, 4, 3),
        lengths=jnp.full((1, 2), 4, dtype=jnp.int32),
    )
    batch = mm_plan.ModalityEmbedBatch(
        encode_inputs=inputs,
        merge=mm_plan.DeviceMergePlan(
            src_idx=jnp.zeros((1, 2, 1), dtype=jnp.int32),
            mask=jnp.zeros((1, 2, 1), dtype=jnp.bool_),
        ),
    )

    class Model:
        mesh = _mesh()

        def get_multimodal_encoder(self, modality):
            assert modality is Modality.AUDIO
            return self._encode_audio

        @staticmethod
        def _encode_audio(value):
            assert value.values.shape == (2, 4, 3)
            return value.values[..., :2]

    def merge(_, running, features, *args, **kwargs):
        assert features.shape == (1, 2, 4, 2)
        return running

    monkeypatch.setattr(mm_utils, "merge_jit", merge)
    running = jnp.zeros((1, 2), dtype=jnp.float32)
    result = mm_utils.embed_mm_inputs(
        {Modality.AUDIO: batch},
        jnp.array([1]),
        lambda _: running,
        Model(),
    )
    np.testing.assert_array_equal(result, running)


@pytest.mark.parametrize(
    ("logical_tp", "input_spec", "output_spec"),
    [
        (
            2,
            PartitionSpec("data", "tensor", None, None),
            PartitionSpec(("data", "tensor"), None, None),
        ),
        (
            1,
            PartitionSpec("data", None, None, None),
            PartitionSpec("data", None, None),
        ),
    ],
)
def test_flatten_device_batch_preserves_explicit_sharding(logical_tp, input_spec, output_spec):
    mesh = _mesh(tp=2)
    values = np.arange(logical_tp * 12, dtype=np.float32).reshape(1, logical_tp, 4, 3)
    values = jax.device_put(
        values,
        NamedSharding(mesh, input_spec),
    )
    out_sharding = NamedSharding(mesh, output_spec)

    with jax.set_mesh(mesh):
        output = mm_utils._flatten_device_batch(
            values,
            out_sharding=out_sharding,
        )
        jax.block_until_ready(output)

    assert output.shape == (logical_tp, 4, 3)
    assert output.sharding.spec == output_spec


def test_plan_balances_images_across_tp_lanes():
    items = _items(
        [(1, 1, 8), (1, 1, 6), (1, 1, 4), (1, 1, 2)],
        [(0, 8), (8, 14), (14, 18), (18, 20)],
    )
    batch = _plan(items, config=_vision_config(), tp_size=2)[Modality.IMAGE]
    np.testing.assert_array_equal(batch.encode_inputs.valid, [[10, 10]])
    np.testing.assert_array_equal(batch.encode_inputs.patches[0, 0, :, 0], [*range(8), 18, 19])
    np.testing.assert_array_equal(batch.encode_inputs.patches[0, 1, :, 0], range(8, 18))
    _assert_lane(batch.merge, 0, 0, [*range(8), 18, 19], range(10))
    _assert_lane(batch.merge, 0, 1, range(8, 18), range(10))


def test_plan_separates_patch_and_placeholder_counts():
    items = _items([(1, 2, 4), (1, 4, 4)], [(2, 4), (5, 9)])
    batch = _plan(items, extend=10, per_dp_token=10)[Modality.IMAGE]
    np.testing.assert_array_equal(batch.encode_inputs.valid, [[24]])
    _assert_lane(batch.merge, 0, 0, [2, 3, 5, 6, 7, 8], range(6))


@pytest.mark.parametrize(
    ("prefix", "extend", "dst", "src"),
    [
        (0, 4, [2, 3], [0, 1]),
        (4, 4, [0, 1], [2, 3]),
        (6, 2, None, None),
    ],
)
def test_plan_clips_to_chunk_boundaries(prefix, extend, dst, src):
    items = _items([(1, 4, 4)], [(2, 6)])
    plan = _plan(items, prefix=prefix, extend=extend, per_dp_token=extend)
    if dst is None:
        assert plan is None
    else:
        batch = plan[Modality.IMAGE]
        np.testing.assert_array_equal(batch.encode_inputs.valid, [[16]])
        _assert_lane(batch.merge, 0, 0, dst, src)


def test_plan_preserves_encoder_offsets_across_chunks():
    items = _items([(1, 4, 4), (1, 4, 4)], [(2, 6), (6, 10)])
    batch = _plan(items, prefix=4, extend=4)[Modality.IMAGE]
    np.testing.assert_array_equal(batch.encode_inputs.valid, [[32]])
    _assert_lane(batch.merge, 0, 0, range(4), [2, 3, 4, 5])


def test_plan_pads_uneven_dp_ranks():
    rank0 = _req(_items([(1, 2, 4), (1, 4, 4)], [(0, 2), (3, 7)]), 8)
    rank1 = _req(
        _items([(1, 2, 4), (1, 2, 4), (1, 4, 4)], [(1, 3), (4, 6), (7, 11)]),
        12,
    )
    plan = build_mm_embed_plan(
        [ScheduleReqsInfo(reqs=[rank0]), ScheduleReqsInfo(reqs=[rank1])],
        2,
        _model_config(),
        12,
    )
    batch = plan[Modality.IMAGE]
    assert batch.encode_inputs.patches.shape == (2, 1, 32, 1)
    np.testing.assert_array_equal(batch.encode_inputs.valid, [[24], [32]])
    _assert_lane(batch.merge, 0, 0, [0, 1, 3, 4, 5, 6], range(6))
    _assert_lane(batch.merge, 1, 0, [1, 2, 4, 5, 7, 8, 9, 10], range(8))


@pytest.mark.parametrize(
    ("feature", "ranges", "match"),
    [
        (np.ones((3, 1)), [(0, 2), (1, 3)], "assigned more than once"),
        (np.ones((3, 1)), [(0, 0)], "non-empty"),
        (np.ones((3, 1)), None, "no placeholder"),
        (np.empty((0, 1)), [(0, 1)], "non-empty 2D"),
    ],
)
def test_plan_rejects_invalid_items(feature, ranges, match):
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=feature,
        placeholder_ranges=ranges,
        model_specific_data={"image_grid_thw": np.array([[1, 1, 3]])},
    )
    with pytest.raises(ValueError, match=match):
        _plan([item], config=_vision_config(), extend=3, per_dp_token=3)


def test_plan_rejects_dict_inputs():
    req = SimpleNamespace(mm_inputs={"mm_items": [{}]}, extend_input_len=1)
    with pytest.raises(TypeError, match="MultimodalInputs"):
        build_mm_embed_plan([ScheduleReqsInfo(reqs=[req])], 1, _model_config(), 1)


def test_plan_handles_non_image_modalities():
    audio = MultimodalDataItem(Modality.AUDIO, feature=np.ones((4, 2)))
    req = _req([audio], 4)
    assert build_mm_embed_plan([ScheduleReqsInfo(reqs=[req])], 1, _model_config(), 4) is None

    video = _items([(1, 2, 4)], [(0, 2)], Modality.VIDEO)
    plan = _plan(video)
    assert tuple(plan) == (Modality.IMAGE,)
    np.testing.assert_array_equal(plan[Modality.IMAGE].encode_inputs.valid, [[8]])


def test_plan_requires_qwen_vision_config():
    config = SimpleNamespace(is_multimodal=True, hf_config=SimpleNamespace(architectures=[ARCH]))
    req = _req(_items([(1, 2, 4)], [(0, 2)]), 2)
    with pytest.raises(ValueError, match="vision_config"):
        build_mm_embed_plan([ScheduleReqsInfo(reqs=[req])], 1, config, 2)


def test_metadata_packs_image_boundaries():
    builder = _plugin()
    items = _items([(1, 16, 16), (1, 4, 4)], [(0, 64), (64, 68)])
    metadata = builder.get_metadata(items)
    np.testing.assert_array_equal(metadata.cu_window_seqlens, [64, 128, 192, 256, 272])
    np.testing.assert_array_equal(metadata.cu_image_seqlens, [256, 272])
    np.testing.assert_array_equal(np.sort(metadata.window_index), np.arange(68))
    assert metadata.rotary_pos_emb.shape == (272, 40)


def test_metadata_stacks_real_and_dummy_lanes():
    builder = _plugin()
    metadata = builder.get_metadata(_items([(1, 2, 4), (1, 4, 4)], [(0, 2), (2, 6)]))
    stacked = builder.stack_metadata([metadata, None], patch_k=24)
    assert jax.tree.map(np.shape, stacked) == Qwen25VLVisionMetadata(
        (2, 6), (2, 6), (2, 24, 40), (2, 6)
    )
    np.testing.assert_array_equal(stacked.cu_image_seqlens[0], [8, 24, 24, 24, 24, 24])
    np.testing.assert_array_equal(stacked.cu_image_seqlens[1], [24] * 6)
    np.testing.assert_array_equal(stacked.window_index[1], range(6))


@pytest.mark.parametrize(
    ("feature_rows", "ranges", "match"),
    [(7, [(0, 2)], "feature rows"), (8, [(0, 0)], "placeholder rows")],
)
def test_metadata_validates_grid_counts(feature_rows, ranges, match):
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=np.ones((feature_rows, 1)),
        placeholder_ranges=ranges,
        model_specific_data={"image_grid_thw": np.array([[1, 2, 4]])},
    )
    with pytest.raises(ValueError, match=match):
        _plugin().get_metadata([item])


def test_metadata_validates_stack_inputs():
    builder = _plugin()
    metadata = builder.get_metadata(_items([(1, 2, 4)], [(0, 2)]))
    with pytest.raises(ValueError, match="at least one real"):
        builder.stack_metadata([None], patch_k=0)
    with pytest.raises(ValueError, match="divisible"):
        builder.stack_metadata([metadata], patch_k=10)


class _NaiveSegmentAttention:
    def __call__(self, q, k, v, segment_ids):
        scores = jnp.einsum("dnth,dnsh->dnts", q, k)
        mask = (segment_ids.q[:, None, :, None] == segment_ids.kv[:, None, None, :]) & (
            segment_ids.q[:, None, :, None] >= 0
        )
        probs = jax.nn.softmax(jnp.where(mask, scores, -1e9), axis=-1)
        return jnp.einsum("dnts,dnsh->dnth", probs, v)


def test_packed_attention_is_block_diagonal():
    config = _vision_config(depth=1, fullatt_block_indexes=[0])
    mesh = _mesh()
    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(config, jnp.float32, mesh=mesh, norm_eps=1e-6)
    visual.blocks[0].attn.attn_backend = _NaiveSegmentAttention()
    builder = _plugin(config)
    features = np.arange(1, 8, dtype=np.float32).reshape(7, 1)
    packed_items = _build_items(features, [(1, 2, 2), (1, 1, 3)], [(0, 4), (4, 7)])
    single_items = _build_items(features[:4], [(1, 2, 2)], [(0, 4)])
    packed_meta = builder.stack_metadata([builder.get_metadata(packed_items)], 7)
    single_meta = builder.stack_metadata([builder.get_metadata(single_items)], 4)

    def compute(value, metadata):
        return visual._compute(
            jnp.asarray(value[None]),
            *jax.tree.leaves(jax.tree.map(jnp.asarray, metadata)),
            jnp.array([len(value)]),
        )

    packed = compute(features, packed_meta)
    single = compute(features[:4], single_meta)
    np.testing.assert_allclose(packed[:, :4], single, rtol=1e-5, atol=1e-5)


def test_vision_encode_accepts_unhashable_config():
    class Config(SimpleNamespace):
        __hash__ = None

    config = Config(**vars(_vision_config()))
    mesh = _mesh()
    with jax.set_mesh(mesh):
        visual = Qwen2_5_VisionTransformer(config, jnp.float32, mesh=mesh, norm_eps=1e-6)
        output = visual.encode(
            jnp.ones((1, 2, 1), dtype=jnp.float32),
            _flat_metadata(1, 2),
            jnp.array([2]),
        )
    assert output.shape == (1, 2, 4)
    assert output.sharding.spec == PartitionSpec(("data", "tensor"), None, None)


@pytest.mark.parametrize("tp", [False, True])
def test_vision_shard_specs(tp):
    specs = VisionShardSpecs(_mesh(), tp)
    assert specs.col_kernel_axes == ((None, "tensor") if tp else (None, None))
    assert specs.row_kernel_axes == (("tensor", None) if tp else (None, None))
    assert specs.head_axis == ("tensor" if tp else None)


@pytest.mark.parametrize(
    ("head_tp", "qkv", "segment"),
    [
        (
            False,
            PartitionSpec(("data", "tensor"), None, None, None),
            PartitionSpec(("data", "tensor"), None),
        ),
        (True, PartitionSpec("data", "tensor", None, None), PartitionSpec("data", None)),
    ],
)
def test_flash_attention_sharding(head_tp, qkv, segment):
    captured = {}

    def shard_map(fn, *, in_specs, out_specs, **kwargs):
        captured.update(in_specs=in_specs, out_specs=out_specs)
        return fn

    with patch(
        "sgl_jax.srt.multimodal.layers.attention.flash_attention_backend.jax.shard_map",
        side_effect=shard_map,
    ):
        VisionFlashAttentionBackend(_mesh(), head_tp=head_tp)
    assert captured == {"in_specs": (qkv, qkv, qkv, segment), "out_specs": qkv}


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
    col = [
        block.attn.q_proj,
        block.attn.k_proj,
        block.attn.v_proj,
        block.mlp.gate_proj,
        block.mlp.up_proj,
        visual.merger.mlp_fc1,
    ]
    row = [block.attn.proj, block.mlp.down_proj, visual.merger.mlp_fc2]
    assert all(layer.weight.value.sharding.spec == PartitionSpec(None, "tensor") for layer in col)
    assert all(layer.weight.value.sharding.spec == PartitionSpec("tensor", None) for layer in row)


def test_attention_padding_and_segment_boundaries():
    captured = {}

    def backend(q, k, v, segment_ids):
        captured.update(q=q.shape, segment=segment_ids.q.shape)
        return q

    q = jnp.zeros((1, 128, 1, 4))
    assert (
        _vision_attention(backend, q, q, q, jnp.zeros((1, 128), dtype=jnp.int32)).shape == q.shape
    )
    assert captured == {"q": (1, 1, 256, 4), "segment": (1, 256)}
    cu = jnp.array([[2, 5, 8, 8], [4, 8, 8, 8]])
    np.testing.assert_array_equal(
        _segment_ids_from_cu_seqlens(cu, 8),
        [[0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1]],
    )


def test_merge_preserves_unmasked_tokens():
    running = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
    features = jnp.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=jnp.float32)
    output = merge_jit(
        _mesh(),
        running,
        features,
        jnp.array([[[1, 0, 0]]]),
        jnp.array([[[True, False, True]]]),
    )
    np.testing.assert_array_equal(output, [[4, 5, 6], [3, 4, 5], [1, 2, 3]])


@pytest.mark.parametrize("encoder_tp", [False, True])
def test_merge_parallel_modes(encoder_tp):
    mesh = _mesh(tp=2)
    running = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    if encoder_tp:
        features = np.array([[[[10, 11], [20, 21]]]], dtype=np.float32)
        src = np.array([[[0, 0, 1]]])
        mask = np.array([[[True, False, True]]])
        feature_spec = PartitionSpec("data", None, None, None)
        route_spec = PartitionSpec("data", None, None)
    else:
        features = np.array([[[[10, 11]], [[20, 21]]]], dtype=np.float32)
        src = np.zeros((1, 2, 3), dtype=np.int32)
        mask = np.array([[[True, False, False], [False, False, True]]])
        feature_spec = PartitionSpec("data", "tensor", None, None)
        route_spec = PartitionSpec("data", "tensor", None)
    output = merge_jit(
        mesh,
        jax.device_put(running, NamedSharding(mesh, PartitionSpec("data", None))),
        jax.device_put(features, NamedSharding(mesh, feature_spec)),
        jax.device_put(src, NamedSharding(mesh, route_spec)),
        jax.device_put(mask, NamedSharding(mesh, route_spec)),
        encoder_tp=encoder_tp,
    )
    np.testing.assert_array_equal(output, [[10, 11], [3, 4], [20, 21]])


def test_merge_uses_rank_local_features():
    mesh = _mesh(dp=2)
    running = np.arange(12, dtype=np.float32).reshape(6, 2)
    features = np.array([[[[10, 11], [20, 21]]], [[[100, 101], [200, 201]]]])
    src = np.array([[[1, 0, 0]], [[0, 0, 1]]])
    mask = np.array([[[True, True, False]], [[True, False, True]]])
    output = merge_jit(
        mesh,
        jax.device_put(running, NamedSharding(mesh, PartitionSpec("data", None))),
        jax.device_put(features, NamedSharding(mesh, PartitionSpec("data", "tensor", None, None))),
        jax.device_put(src, NamedSharding(mesh, PartitionSpec("data", "tensor", None))),
        jax.device_put(mask, NamedSharding(mesh, PartitionSpec("data", "tensor", None))),
    )
    expected = running.copy()
    expected[[0, 1, 3, 5]] = features[[0, 0, 1, 1], 0, [1, 0, 0, 1]]
    np.testing.assert_array_equal(output, expected)


def test_device_put_plan_shards_lane_axes():
    plan = _host_plan()
    specs = []

    def device_array(values, sharding):
        specs.append(sharding.spec)
        return values

    with patch(
        "sgl_jax.srt.model_executor.forward_batch_info.device_array",
        side_effect=device_array,
    ):
        _device_put_embed_plan(plan, _mesh())
    assert specs == [
        PartitionSpec("data", "tensor", None, None),
        PartitionSpec("data", "tensor"),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None, None),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None),
        PartitionSpec("data", "tensor", None),
    ]


@pytest.mark.parametrize(
    ("arch", "chunked", "radix"),
    [(ARCH, 4096, False), ("UnsupportedVLM", -1, True)],
)
def test_multimodal_defaults_follow_capabilities(arch, chunked, radix):
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
    assert args.enable_mixed_chunk is False
    assert args.limit_mm_data_per_request == {"image": 16}


def test_generate_request_preserves_media_fields():
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
    assert (item.image_data, item.video_data, item.audio_data, item.input_embeds) == (
        ["image1"],
        ["video1"],
        ["audio1"],
        ["emb1"],
    )


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


def test_overlap_copy_rebuilds_plan_from_requests():
    items = _items([(1, 2, 4)], [(1, 3)])
    batch = _schedule_batch(_req(items, 3), _model_config())
    worker_batch = batch.get_model_worker_batch(
        token_paddings=[3],
        bs_paddings=[1],
        cache_loc_paddings=[3],
        page_size=1,
    )
    copied = batch.copy()
    rebuilt = build_mm_embed_plan(copied.reqs_info, 1, _model_config(), 3)
    assert Modality.IMAGE in worker_batch.mm_embed_plan
    assert getattr(copied, "mm_embed_plan", None) is None
    assert Modality.IMAGE in rebuilt


def test_multimodal_item_reads_common_and_model_fields():
    item = MultimodalDataItem.from_dict(
        {
            "modality": "image",
            "feature": np.ones((2, 1)),
            "placeholder_ranges": [(1, 2)],
            "image_grid_thw": np.array([[1, 2, 4]]),
        }
    )
    assert item.is_image()
    assert item.placeholder_ranges == [(1, 2)]
    np.testing.assert_array_equal(item.get("image_grid_thw"), [[1, 2, 4]])
    assert item.get("missing", "fallback") == "fallback"


def _dummy_builder():
    config = _qwen_config(in_channels=3, temporal_patch_size=2)
    return InModelVisionPlanBuilder(
        _plugin(config),
        patch_buckets=[256, 1024],
        merge_buckets=[64, 256],
    )


@pytest.mark.parametrize(("tp", "patches", "tokens"), [(1, 256, 64), (2, 1024, 128)])
def test_dummy_plan_uses_bucketed_shapes(tp, patches, tokens):
    builder = _dummy_builder()
    batch = builder.dummy_plan(1, tp, patches, tokens)[Modality.IMAGE]
    assert batch.encode_inputs.patches.shape == (1, tp, patches, builder.plugin.feature_dim)
    assert batch.merge.mask.shape == (1, tp, tokens)
    assert batch.source_capacity == 256
    assert not batch.merge.mask.any()


@pytest.mark.parametrize(
    ("value", "buckets", "expected"),
    [
        (10, [64, 256], 64),
        (64, [64, 256], 64),
        (65, [64, 256], 256),
        (300, [64, 256], 300),
        (10, None, 10),
    ],
)
def test_ceil_to_bucket_boundaries(value, buckets, expected):
    assert _ceil_to_bucket(value, buckets) == expected
