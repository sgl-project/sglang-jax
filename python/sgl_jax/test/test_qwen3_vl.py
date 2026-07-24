from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding

from sgl_jax.srt.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLVisionModel,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model import host_orchestration
from sgl_jax.srt.multimodal.in_model.encoder_planning import (
    EncodeInputs,
    _stack_metadata,
)
from sgl_jax.srt.multimodal.in_model.encoders.qwen3_vl import Qwen3VLVisionEncoderPlugin
from sgl_jax.srt.multimodal.in_model.plan import DeviceMergePlan, ModalityEmbedBatch
from sgl_jax.srt.multimodal.layers.vision_sharding import VisionShardSpecs
from sgl_jax.srt.multimodal.manager.mrope_utils import compute_qwen3vl_mrope_positions
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor


def _config():
    return SimpleNamespace(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        hidden_act="gelu_pytorch_tanh",
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )


def _plugin():
    return Qwen3VLVisionEncoderPlugin(
        SimpleNamespace(hf_config=SimpleNamespace(vision_config=_config()))
    )


def _item(grid, ranges):
    rows = int(np.prod(grid))
    item = MultimodalDataItem(
        Modality.IMAGE,
        feature=np.arange(rows * 12, dtype=np.float32).reshape(rows, 12),
        placeholder_ranges=ranges,
    )
    item.set("image_grid_thw", np.asarray([grid], dtype=np.int32))
    return item


def test_qwen3vl_metadata_uses_spatial_merge_order():
    meta = _plugin().get_metadata([_item((1, 2, 4), [(0, 2)])])
    np.testing.assert_array_equal(
        meta.rotary_pos_emb[:, [0, 1]],
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]],
    )
    np.testing.assert_array_equal(meta.cu_seqlens, [8])
    np.testing.assert_allclose(meta.pos_weights.sum(axis=0), 1)


def test_qwen3vl_metadata_packs_multiple_items():
    meta = _plugin().get_metadata([_item((1, 2, 2), [(0, 1)]), _item((2, 2, 2), [(2, 3), (4, 5)])])
    assert meta.pos_indices.shape == (4, 12)
    assert meta.rotary_pos_emb.shape == (12, 2)
    np.testing.assert_array_equal(meta.cu_seqlens, [4, 8, 12])


def test_qwen3vl_grouped_video_ranges_and_mrope():
    token_types = np.asarray([0, 2, 0, 2, 0], dtype=np.int32)
    ranges = QwenVLProcessor._grouped_placeholder_ranges(
        token_types, 2, [(2, 2, 2)], 2, split_temporal=True
    )
    assert ranges == [[(1, 2), (3, 4)]]
    positions, delta = compute_qwen3vl_mrope_positions(
        mm_token_type_ids=token_types,
        image_grid_thw=None,
        video_grid_thw=[(2, 2, 2)],
        spatial_merge_size=2,
    )
    np.testing.assert_array_equal(positions, [[0, 1, 2, 3, 4]] * 3)
    assert delta == 0


def test_qwen3vl_vision_forward_explicit_mesh():
    plugin = _plugin()
    item = _item((1, 2, 2), [(0, 1)])
    meta = _stack_metadata(plugin, [plugin.get_metadata([item])], 4)
    inputs = EncodeInputs(
        jnp.asarray(item.feature)[None],
        jnp.asarray([4], dtype=jnp.int32),
        jax.tree.map(jnp.asarray, meta),
    )
    tp = 4 if len(jax.devices()) >= 4 else 1
    mesh = Mesh(
        np.asarray(jax.devices()[:tp]).reshape(1, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    with jax.set_mesh(mesh):
        specs = VisionShardSpecs(mesh, tp=tp > 1)
        inputs = jax.tree.map(
            lambda value: jax.device_put(
                value,
                NamedSharding(mesh, specs.batch_spec(*([None] * (value.ndim - 1)))),
            ),
            inputs,
        )
        model = Qwen3VLVisionModel(_config(), jnp.float32, nnx.Rngs(0), mesh, tp=tp > 1)
        output, deepstack = model.encode(inputs)
    assert output.shape == (1, 1, 8)
    assert deepstack.shape == (1, 1, 1, 8)
    assert jnp.isfinite(output).all() and jnp.isfinite(deepstack).all()


def test_in_model_merge_populates_deepstack(monkeypatch):
    inputs = EncodeInputs(
        jnp.zeros((1, 1, 1, 1)),
        jnp.ones((1, 1), dtype=jnp.int32),
        jnp.zeros((1, 1, 1)),
    )
    batch = ModalityEmbedBatch(
        inputs,
        DeviceMergePlan(
            jnp.zeros((1, 1, 1), dtype=jnp.int32),
            jnp.ones((1, 1, 1), dtype=jnp.bool_),
        ),
    )

    class Model:
        mesh = Mesh(
            np.asarray(jax.devices()[:1]).reshape(1, 1),
            ("data", "tensor"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )

        def get_image_feature(self, _):
            return jnp.asarray([[[1.0, 2.0]]]), jnp.asarray([[[[3.0, 4.0]]]])

        def get_multimodal_encoder(self, modality):
            assert modality is Modality.IMAGE
            return self.get_image_feature

    def merge(_, running, features, *args, **kwargs):
        return features[0, 0]

    monkeypatch.setattr(host_orchestration, "merge_jit", merge)
    embedding, deepstack = host_orchestration.embed_mm_inputs(
        {Modality.IMAGE: batch},
        jnp.asarray([0]),
        lambda _: jnp.zeros((1, 2)),
        Model(),
        return_deepstack=True,
    )
    np.testing.assert_array_equal(embedding, [[1, 2]])
    np.testing.assert_array_equal(deepstack, [[[3, 4]]])


def test_qwen3vl_exposes_generic_image_encoder():
    class Model:
        def get_image_feature(self, inputs):
            return inputs

    model = Model()
    encoder = Qwen3VLForConditionalGeneration.get_multimodal_encoder(model, Modality.IMAGE)
    assert encoder("inputs") == "inputs"
    with pytest.raises(ValueError, match="does not support AUDIO"):
        Qwen3VLForConditionalGeneration.get_multimodal_encoder(model, Modality.AUDIO)
