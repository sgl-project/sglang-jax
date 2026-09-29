"""Essential chunk-prefill, encoder-layout and sharding regressions.

Run multi-device cases locally with XLA_FLAGS=--xla_force_host_platform_device_count=4.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sgl_jax.srt.models.qwen3_vl import Qwen3VLVisionModel
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model import host_orchestration as orchestration
from sgl_jax.srt.multimodal.in_model.embedding_pool import EmbeddingPool
from sgl_jax.srt.multimodal.in_model.interface import (
    InModelMultimodalContract,
    VisionInputSpec,
)
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    _bucket_capacity,
    encoder_num_lanes,
    mrope_vision_dummy_inputs,
    pack_vision_inputs,
    run_mrope_vision_model,
)


def _mesh(dp=1, tp=1):
    if jax.device_count() < dp * tp:
        pytest.skip(f"requires {dp * tp} devices")
    return Mesh(
        np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _item(key=1, ranges=((1, 6),), width=1):
    length = sum(end - start for start, end in ranges)
    return MultimodalDataItem(
        Modality.IMAGE,
        hash=key,
        placeholder_ranges=list(ranges),
        feature=np.arange(10, 10 + length * width, dtype=np.float32).reshape(length, width),
    )


def _info(items, prefix, length):
    req = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=items), extend_input_len=length)
    return ScheduleReqsInfo(reqs=[req], prefix_lens=[prefix], extend_lens=[length])


def _batch(infos, tokens, pool=None, lanes=1):
    config = SimpleNamespace(is_in_model_multimodal=True)
    return orchestration.build_multimodal_batch(
        infos, len(infos), config, tokens, embedding_pool=pool, num_encoder_lanes=lanes
    )


class _EchoModel(InModelMultimodalContract):
    """Deterministic encoder oracle; feature rows are already the expected output."""

    def __init__(self, mesh=None, deepstack=0):
        self.mesh = mesh
        self.deepstack_visual_layers = deepstack
        self.calls = 0

    def get_input_embeddings(self):
        def embed(ids):
            result = jnp.full((len(ids), 1), -1.0)
            return (
                jax.device_put(result, NamedSharding(self.mesh, PartitionSpec("data", None)))
                if self.mesh is not None
                else result
            )

        return embed

    def get_multimodal_encode_funcs(self):
        def encode(lanes):
            self.calls += 1
            result = jnp.asarray(np.concatenate([item.feature for lane in lanes for item in lane]))
            return (
                jax.device_put(result, NamedSharding(self.mesh, PartitionSpec()))
                if self.mesh is not None
                else result
            )

        return {Modality.IMAGE: encode}


@pytest.mark.parametrize(
    "prefix, length, expected",
    [
        (0, 2, [-1, -1]),
        (2, 2, [10, 11]),
        (4, 2, [-1, -1]),
        (6, 2, [12, 13]),
        (8, 2, [-1, -1]),
        (3, 4, [11, -1, -1, 12]),
    ],
)
def test_chunk_intersections_preserve_text_and_skip_empty_overlap(prefix, length, expected):
    item = _item(ranges=[(2, 4), (6, 8)])
    batch = _batch([_info([item], prefix, length)], length)
    model = _EchoModel()
    result, _, _ = orchestration.embed_multimodal_inputs(batch, jnp.zeros(length, jnp.int32), model)
    np.testing.assert_array_equal(result[:, 0], expected)
    assert model.calls == int(any(value != -1 for value in expected))


@pytest.mark.parametrize("dp", [1, 2])
def test_merge_keeps_request_offsets_and_deepstack_sharding(dp):
    mesh = _mesh(dp=dp)
    a, b = _item(1, [(1, 2)], 2), _item(2, [(0, 1)], 2)
    b.feature[:] = [20, 21]
    first = _info([a], 0, 2)
    second = _info([b], 0, 2)
    # Two requests in rank 0; rank 1 is empty when present.
    first.reqs += second.reqs
    first.prefix_lens += second.prefix_lens
    first.extend_lens += second.extend_lens
    infos = [first] + [ScheduleReqsInfo(reqs=[])] * (dp - 1)
    model = _EchoModel(mesh, deepstack=1)
    result, deepstack, enabled = orchestration.embed_multimodal_inputs(
        _batch(infos, 4, lanes=dp), jnp.zeros(4 * dp, jnp.int32), model
    )
    expected = np.full(4 * dp, -1.0)
    expected[[1, 2]] = [10, 20]
    np.testing.assert_array_equal(result[:, 0], expected)
    expected_deepstack = np.zeros(4 * dp)
    expected_deepstack[[1, 2]] = [11, 21]
    np.testing.assert_array_equal(deepstack[0, :, 0], expected_deepstack)
    assert result.sharding.spec == PartitionSpec("data", None)
    assert deepstack.sharding.spec == PartitionSpec(None, "data", None)
    assert enabled and model.calls == 1


@pytest.mark.parametrize("cached", [False, True])
def test_chunk_prefill_reuses_full_image_until_its_last_token(cached):
    pool = EmbeddingPool(3, 2, 1, jnp.float32) if cached else None
    item, model = _item(), _EchoModel()
    outputs = []
    for prefix in (0, 2, 4, 6):
        batch = _batch([_info([item], prefix, 2)], 2, pool)
        result, _, _ = orchestration.embed_multimodal_inputs(
            batch, jnp.zeros(2, jnp.int32), model, pool
        )
        outputs.extend(np.asarray(result[:, 0]))
    np.testing.assert_array_equal(outputs, [-1, 10, 11, 12, 13, 14, -1, -1])
    assert model.calls == (1 if cached else 3)
    if cached:
        pool.clear()


def test_forward_reencodes_cache_hit_evicted_after_scheduling():
    pool = EmbeddingPool(1, 2, 1, jnp.float32)
    cached = _item(1, [(0, 2)])
    miss = _item(2, [(2, 3)])
    pool.write_packed([1], cached.feature, [2])
    batch = _batch([_info([cached, miss], 0, 3)], 3, pool)
    assert len(batch.cached_tasks) == 1
    pool.write_packed([3], np.full((2, 1), 99), [2])
    model = _EchoModel()
    result, _, _ = orchestration.embed_multimodal_inputs(
        batch, jnp.zeros(3, jnp.int32), model, pool
    )
    np.testing.assert_array_equal(result[:, 0], [10, 11, 10])
    assert model.calls == 1


@pytest.mark.parametrize("destination", [-1, 4])
def test_merge_rejects_mappings_outside_chunk(destination):
    task = orchestration.ItemTask(_item(), 1, [orchestration.MergeMapping(0, destination, 1)])
    with pytest.raises(ValueError, match="exceeds"):
        orchestration._gather_merge(jnp.zeros((4, 1)), jnp.ones((1, 1)), [task], None)


def _vision(model_type, mesh, tensor_parallel):
    config = SimpleNamespace(
        patch_size=1,
        temporal_patch_size=1,
        in_channels=1,
        hidden_size=8,
        depth=1,
        intermediate_size=16,
        hidden_act="silu",
        num_heads=2,
        out_hidden_size=8,
        spatial_merge_size=2,
        fullatt_block_indexes=[],
        window_size=4,
        rope_theta=10000.0,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0] if model_type == "qwen3" else [],
    )
    with jax.set_mesh(mesh):
        if model_type == "qwen3":
            return Qwen3VLVisionModel(config, jnp.float32, mesh=mesh, tp=tensor_parallel)
        return Qwen2_5_VisionTransformer(config, jnp.float32, mesh=mesh, vision_tp=tensor_parallel)


def _vision_item(grid, key):
    patches = int(np.prod(grid))
    item = _item(key, [(0, patches // 4)])
    item.feature = np.arange(patches, dtype=np.float32).reshape(-1, 1) / 10
    item.model_specific_data = {"image_grid_thw": np.asarray(grid, np.int32)}
    return item


def _encode(visual, lanes):
    return run_mrope_vision_model(
        visual,
        lanes,
        mesh=visual.mesh,
        num_lanes=encoder_num_lanes(visual.mesh, visual.vision_tp),
        merge_unit=visual.spatial_merge_unit,
        rope_type="rope_3d",
        input_sharding=visual.specs.sharding(visual.specs.batch_axis),
        output_sharding=visual.specs.sharding(),
    )


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
@pytest.mark.parametrize("parallel", ["single", "dp", "tp"])
def test_packed_encoder_matches_individual_images_with_padding_and_empty_lanes(
    model_type, parallel
):
    mesh = _mesh() if parallel == "single" else _mesh(dp=2, tp=2)
    visual = _vision(model_type, mesh, parallel == "tp")
    lanes = encoder_num_lanes(mesh, visual.vision_tp)
    items = [_vision_item((1, 2, 2), 1), _vision_item((2, 2, 4), 2)]
    # Different input shapes can select different default matmul precision on
    # TPU. Compare packing semantics with consistent FP32 multiplication.
    with jax.default_matmul_precision("highest"):
        # Keep empty lanes even though other lanes have multiple images.
        packed = np.asarray(_encode(visual, [items] + [[] for _ in range(lanes - 1)]))
        expected = np.concatenate(
            [
                np.asarray(_encode(visual, [[item]] + [[] for _ in range(lanes - 1)]))[
                    : len(item.feature) // 4
                ]
                for item in items
            ]
        )
    np.testing.assert_allclose(packed[: len(expected)], expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(packed[len(expected) :], 0)
    assert packed.shape[1] == (16 if model_type == "qwen3" else 8)


@pytest.mark.parametrize("invalid", ["empty", "features", "placeholders"])
def test_packing_rejects_inconsistent_image_lengths(invalid):
    item = _vision_item((1, 2, 2), 1)
    if invalid == "features":
        item.feature = item.feature[:-1]
    elif invalid == "placeholders":
        item.placeholder_ranges = [(0, 2)]
    with pytest.raises(ValueError):
        pack_vision_inputs(
            [[]] if invalid == "empty" else [[item]],
            merge_unit=4,
            input_sharding=NamedSharding(_mesh(), PartitionSpec("data")),
        )


@pytest.mark.parametrize(
    "length, unit, expected",
    [
        (0, 9, 9),
        (9, 9, 9),
        (10, 9, 18),
        (18, 9, 18),
        (19, 9, 36),
        (1025, 4, 2048),
        (-1, 4, None),
        (1, 0, None),
    ],
)
def test_vision_bucket_edges(length, unit, expected):
    if expected is None:
        with pytest.raises(ValueError):
            _bucket_capacity(length, unit)
    else:
        assert _bucket_capacity(length, unit) == expected


def test_warmup_deduplicates_shapes_and_does_not_evict_resident_cache():
    spec = VisionInputSpec(1, 3)
    dummy = list(mrope_vision_dummy_inputs(spec, [10, 17, 19]))
    assert [lanes[0][0].feature.shape[0] for _, lanes in dummy] == [18, 36]
    seen = []

    class Model(_EchoModel):
        vision_input_spec = spec

        def get_multimodal_encode_funcs(self):
            def encode(lanes):
                assert len(lanes) == 2 and not lanes[1]
                item = lanes[0][0]
                assert np.prod(item.get("image_grid_thw")) == len(item.feature)
                seen.append(len(item.feature))
                return jnp.zeros((len(item.feature) // 9 * 2, 1))

            return {Modality.IMAGE: encode}

    pool = EmbeddingPool(1, 2, 1, jnp.float32)
    (entry,) = pool.write_packed([1], np.full((2, 1), 7), [2])
    capacities = orchestration.precompile_multimodal_encoder(
        Model(), pool, token_buckets=[2, 4], num_lanes=2, patch_paddings=[10, 17, 19]
    )
    assert seen == [18, 36] and capacities == (4, 8)
    assert pool.lookup(1) is entry
    np.testing.assert_array_equal(np.asarray(pool.pages)[entry.page_ids].reshape(-1), [7, 7])


def test_retracted_prefill_continues_mrope_positions_past_prompt():
    positions = np.asarray([[0, 1, 2, 30, 31], [0, 1, 2, 40, 41], [0, 1, 2, 50, 51]], np.int32)
    req = SimpleNamespace(
        mm_inputs={"mrope_positions": positions, "mrope_position_delta": -2},
        extend_input_len=5,
        lora_id="0",
    )
    info = ScheduleReqsInfo(
        reqs=[req],
        input_ids=np.arange(5, dtype=np.int32),
        seq_lens=np.array([8], np.int32),
        out_cache_loc=np.arange(1, 6, dtype=np.int32),
        req_pool_indices=np.array([0], np.int32),
        prefix_lens=np.array([3], np.int32),
        extend_lens=np.array([5], np.int32),
        extend_logprob_start_lens=np.array([0], np.int32),
    )
    batch = ScheduleBatch(
        reqs_info=[info],
        dp_size=1,
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        model_config=None,
    )
    result = batch._merge_multimodal(per_dp_token_size=5, total_token_size=5)
    np.testing.assert_array_equal(
        result["mrope_positions"],
        [[30, 31, 3, 4, 5], [40, 41, 3, 4, 5], [50, 51, 3, 4, 5]],
    )


@pytest.mark.parametrize("num_lanes", [1, 2, 4])
@pytest.mark.parametrize("grids", [[(4, 2, 4), (1, 2, 2)], [(1, 2, 2), (3, 2, 4), (1, 4, 4)]])
def test_temporal_pooling_restores_item_order(grids, num_lanes):
    # An independent, per-item mean encoder isolates packing from attention.
    class MeanEncoder:
        def prepare_metadata(self, grid_thw, capacity, *, sharding):
            return dict(grids=grid_thw, capacity=capacity)

        def __call__(self, patches, *, grids, capacity):
            patches = np.asarray(patches).reshape(num_lanes, capacity, 1)
            output = np.zeros((num_lanes, capacity // 4, 1), dtype=np.float32)
            for lane, lane_grids in enumerate(grids):
                src = dst = 0
                for t, h, w in lane_grids:
                    if not t:
                        continue
                    count = t * h * w
                    values = patches[lane, src : src + count].reshape(t, h // 2, 2, w // 2, 2)
                    values = values.mean(axis=(0, 2, 4)).reshape(-1, 1)
                    output[lane, dst : dst + len(values)] = values
                    src += count
                    dst += len(values)
            return jax.device_put(output, NamedSharding(mesh, PartitionSpec("data")))

    mesh = _mesh(dp=num_lanes)
    ranges = [(0, h * w // 4) for _, h, w in grids]
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=(np.arange(t * h * w, dtype=np.float32) + index * 100).reshape(-1, 1),
            placeholder_ranges=[span],
            model_specific_data={"image_grid_thw": np.array([t, h, w])},
        )
        for index, ((t, h, w), span) in enumerate(zip(grids, ranges))
    ]
    items_per_lane = [items[lane::num_lanes] for lane in range(num_lanes)]
    ordered_items = [item for lane in items_per_lane for item in lane]
    expected = np.concatenate(
        [
            item.feature.reshape(t, h // 2, 2, w // 2, 2).mean(axis=(0, 2, 4)).reshape(-1, 1)
            for item in ordered_items
            for t, h, w in [item.image_grid_thw]
        ]
    )
    kwargs = dict(
        mesh=mesh,
        num_lanes=num_lanes,
        merge_unit=4,
        rope_type="rope_2d",
        input_sharding=NamedSharding(mesh, PartitionSpec("data")),
        output_sharding=NamedSharding(mesh, PartitionSpec()),
    )
    with pytest.raises(ValueError, match="placeholder tokens"):
        run_mrope_vision_model(MeanEncoder(), items_per_lane, **kwargs)
    actual = np.asarray(
        run_mrope_vision_model(
            MeanEncoder(), items_per_lane, pool_temporal_dimension=True, **kwargs
        )
    )
    np.testing.assert_allclose(actual[: len(expected)], expected)
    np.testing.assert_array_equal(actual[len(expected) :], 0)
