from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
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
    encoder_num_lanes,
    pack_vision_inputs,
    run_mrope_vision_model,
)
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


def _qwen_config():
    return _vision_config(
        patch_size=14,
        window_size=112,
        spatial_merge_size=2,
        num_heads=16,
        hidden_size=1280,
        out_hidden_size=1280,
    )


def _model_config(arch=ARCH):
    return SimpleNamespace(
        is_multimodal=True,
        hf_config=SimpleNamespace(
            architectures=[arch],
            vision_config=_qwen_config(),
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


def _items(grids, ranges):
    rows = sum(int(np.prod(grid)) for grid in grids)
    features = np.arange(rows, dtype=np.float32).reshape(rows, 1)
    return QwenVLProcessor._build_items(
        features,
        grids,
        ranges,
        Modality.IMAGE,
        "image_grid_thw",
    )


def _pack_qwen2(visual, items):
    patches, grid_thw, output_indices = pack_vision_inputs(
        items,
        num_lanes=encoder_num_lanes(visual.mesh, visual.vision_tp),
        buckets=visual.input_buckets,
        merge_unit=visual.spatial_merge_unit,
        dtype=visual.dtype,
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
        dtype=visual.dtype,
    )


def _req(items, extend_len):
    return SimpleNamespace(
        mm_inputs=MultimodalInputs(mm_items=items),
        extend_input_len=extend_len,
        lora_id="0",
    )


def _batch(items, *, prefix=0, extend=None, per_dp_token=None):
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
        _model_config(),
        per_dp_token,
    )


def _batch_dp(items_by_dp, *, per_dp_token):
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
        _model_config(),
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


@pytest.mark.parametrize("encoder_tp", [False, True])
def test_qwen2_vision_encode_and_merge_spmd(encoder_tp):
    mesh = _mesh(dp=2, tp=2)
    visual = _visual(
        mesh=mesh,
        encoder_tp=encoder_tp,
        input_buckets=(4,),
    )
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
    output, _, _ = host_orchestration.embed_multimodal_inputs(*args)
    assert output.sharding.spec == PartitionSpec("data", None)
    assert calls == 1


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
    output, deepstack, apply_for_deepstack = host_orchestration.embed_multimodal_inputs(
        batch,
        jnp.zeros(3, dtype=jnp.int32),
        Model(running),
    )
    np.testing.assert_array_equal(output, [[10, 11], [3, 4], [20, 21]])
    assert deepstack is None
    assert apply_for_deepstack is False


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
    out, ds, apply_for_deepstack = host_orchestration.embed_multimodal_inputs(
        batch, jnp.zeros(4, dtype=jnp.int32), Model(running)
    )
    np.testing.assert_array_equal(out[:, 0], [10, 11, 20, 21])
    np.testing.assert_array_equal(ds[0, :, 0], [30, 31, 40, 41])
    assert out.sharding.spec == PartitionSpec("data", None)
    assert ds.sharding.spec == PartitionSpec(None, "data", None)
    assert apply_for_deepstack is True

    _, text_ds, text_apply_for_deepstack = host_orchestration.embed_multimodal_inputs(
        None, jnp.zeros(4, dtype=jnp.int32), Model(running)
    )
    np.testing.assert_array_equal(text_ds, 0)
    assert text_apply_for_deepstack is False


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
        output, deepstack, apply_for_deepstack = host_orchestration.precompile_multimodal_inputs(
            jax.device_put(jnp.arange(4), data),
            Model(),
        )

    merge.assert_called_once()
    np.testing.assert_array_equal(output, 0)
    np.testing.assert_array_equal(deepstack, 0)
    assert output.dtype == jnp.float32
    assert deepstack.dtype == jnp.float32
    assert apply_for_deepstack is True
    assert output.sharding.spec == PartitionSpec("data", None)
    assert deepstack.sharding.spec == PartitionSpec(None, "data", None)


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
    first, _, _ = host_orchestration.embed_multimodal_inputs(
        _batch([item], prefix=0, extend=2, per_dp_token=2),
        jnp.zeros(2, dtype=jnp.int32),
        model,
        pool,
    )
    second, _, _ = host_orchestration.embed_multimodal_inputs(
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
    model = _TestInModelModel()
    embedding_pool = object()
    forward_batch = SimpleNamespace(
        bid=1,
        forward_mode=ForwardMode.EXTEND,
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
            return_value=("embedded", "deepstack", True),
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


def test_mrope_positions_reach_worker_batch():
    positions = np.array([[0, 10, 2], [0, 11, 2], [0, 12, 2]], dtype=np.int32)
    req = SimpleNamespace(mm_inputs={"mrope_positions": positions}, extend_input_len=3, lora_id="0")
    worker_batch = _schedule_batch(req, _model_config()).get_model_worker_batch(
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


def test_qwen2_vision_metadata_is_bucket_stable():
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
    _, position_ids, window_attn, full_attn = metadata

    assert window_attn.max_seq_len == 16
    assert full_attn.max_seq_len == 32
    np.testing.assert_array_equal(output_indices[:6], np.arange(6))
    assert position_ids.shape == (1, 32, 2)

    jax.block_until_ready(visual.encode(patches, grid_thw))
    cache_size = visual._encode_jit._cache_size()
    second = _items([(1, 4, 4), (2, 2, 2)], [(0, 4), (4, 6)])
    second_patches, second_grid_thw, _ = _pack_qwen2(visual, second)
    jax.block_until_ready(visual.encode(second_patches, second_grid_thw))
    assert visual._encode_jit._cache_size() == cache_size
