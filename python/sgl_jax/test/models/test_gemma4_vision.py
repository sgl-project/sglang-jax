from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.configs.gemma4 import Gemma4Config, Gemma4VisionConfig
from sgl_jax.srt.models.gemma4 import Gemma4ForConditionalGeneration
from sgl_jax.srt.models.gemma4_vision import (
    Gemma4VisionModel,
    apply_multidimensional_rope,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    pack_lanes,
    restore_encoder_output,
)


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _vision_config(**overrides) -> Gemma4VisionConfig:
    values = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "patch_size": 1,
        "pooling_kernel_size": 3,
        "position_embedding_size": 16,
        "default_output_length": 2,
        "standardize": True,
    }
    values.update(overrides)
    return Gemma4VisionConfig(**values)


def _grid(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack((x, y), axis=-1).reshape(-1, 2).astype(np.int32)


def _item(width: int, height: int) -> MultimodalDataItem:
    positions = _grid(width, height)
    return MultimodalDataItem(
        Modality.IMAGE,
        feature=np.full((len(positions), 3), 0.5, dtype=np.float32),
        model_specific_data={"pixel_position_ids": positions},
    )


def _pack(model, items):
    num_lanes = encoder_num_lanes(model.mesh, model.vision_tp)
    packed = pack_lanes(
        items,
        num_lanes,
        buckets=model.input_buckets,
        merge_unit=model.pooling_unit,
    )
    position_ids = np.full((num_lanes, packed.cap, 2), -1, dtype=np.int32)
    patch_counts = np.zeros((num_lanes, max(map(len, packed.lanes))), dtype=np.int32)
    for lane_index, lane in enumerate(packed.lanes):
        offset = 0
        for item_offset, item_index in enumerate(lane):
            positions = items[item_index].get("pixel_position_ids")
            position_ids[lane_index, offset : offset + len(positions)] = positions
            patch_counts[lane_index, item_offset] = len(positions)
            offset += len(positions)
    return packed.features, position_ids, patch_counts, packed.output_indices


def test_gemma4_config_builds_typed_vision_config():
    config = Gemma4Config(
        text_config={"head_dim": 8, "num_key_value_heads": 1},
        vision_config={"hidden_size": 32, "patch_size": 8},
    )

    assert isinstance(config.vision_config, Gemma4VisionConfig)
    assert config.vision_config.hidden_size == 32
    assert config.vision_config.patch_size == 8
    assert config.vision_config.pooling_kernel_size == 3


def test_multidimensional_rope_rotates_each_axis_independently():
    inputs = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    positions = jnp.asarray([[[0, 0], [1, 0]]], dtype=jnp.int32)

    output = apply_multidimensional_rope(inputs, positions, base_frequency=100.0)

    np.testing.assert_allclose(output[0, 0], inputs[0, 0])
    assert not np.allclose(output[0, 1, 0, :2], inputs[0, 1, 0, :2])
    np.testing.assert_allclose(output[0, 1, 0, 2:], inputs[0, 1, 0, 2:])


def test_pool_indices_follow_two_dimensional_windows():
    indices = Gemma4VisionModel._pool_indices(_grid(6, 3), kernel_size=3)

    np.testing.assert_array_equal(np.bincount(indices), [9, 9])
    np.testing.assert_array_equal(indices.reshape(3, 6)[:, :3], 0)
    np.testing.assert_array_equal(indices.reshape(3, 6)[:, 3:], 1)


def test_patch_position_embedding_supports_jitted_dynamic_indices():
    mesh = _mesh()
    patches = jnp.full((1, 4, 3), 0.5, dtype=jnp.float32)
    positions = jnp.asarray([[[0, 0], [1, 0], [0, 1], [-1, -1]]], dtype=jnp.int32)
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(4,),
        )
        eager = model.patch_embedder(patches, positions)
        compiled = jax.jit(lambda x, pos: model.patch_embedder(x, pos))(patches, positions)

    np.testing.assert_allclose(compiled, eager, rtol=1e-5, atol=1e-5)


def test_lane_metadata_keeps_packed_images_as_separate_attention_segments():
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(18,),
        )

    _, position_ids, patch_counts, output_indices = _pack(model, [_item(3, 3), _item(3, 3)])
    metadata = model._build_metadata(position_ids, patch_counts)

    np.testing.assert_array_equal(np.asarray(metadata.attention.cu_seqlens), [[0, 9, 18]])
    np.testing.assert_array_equal(np.asarray(metadata.pool_indices), [[0] * 9 + [1] * 9])
    np.testing.assert_array_equal(output_indices, [0, 1])


def test_vision_tower_uses_varlen_backend_and_returns_item_ordered_array(monkeypatch):
    backend_options = {}

    class IdentityAttention:
        def __call__(self, query, key, value, metadata):
            del key, value, metadata
            return query

    def fake_backend(mesh, **kwargs):
        del mesh
        backend_options.update(kwargs)
        return IdentityAttention()

    monkeypatch.setattr(
        "sgl_jax.srt.models.gemma4_vision.make_vision_attention_backend",
        fake_backend,
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(num_hidden_layers=1),
            text_hidden_size=12,
            dtype=jnp.float32,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(9,),
        )

    item = _item(3, 3)
    patches, position_ids, patch_counts, output_indices = _pack(model, [item])
    output = model.encode(patches, position_ids, patch_counts)
    packed = restore_encoder_output(output, output_indices, mesh)

    assert backend_options["use_varlen"] is True
    assert packed.shape == (1, 12)
    assert bool(jnp.all(jnp.isfinite(packed)))


def test_vision_attention_casts_projection_outputs_to_model_dtype(monkeypatch):
    attention_dtypes = {}

    class Float32Projection(nnx.Module):
        def __call__(self, inputs, *, out_sharding=None):
            del out_sharding
            return inputs.astype(jnp.float32), None

    class IdentityAttention:
        def __call__(self, query, key, value, metadata):
            del metadata
            attention_dtypes.update(q=query.dtype, k=key.dtype, v=value.dtype)
            return query

    monkeypatch.setattr(
        "sgl_jax.srt.models.gemma4_vision.make_vision_attention_backend",
        lambda mesh, **kwargs: IdentityAttention(),
    )
    mesh = _mesh()
    with jax.set_mesh(mesh):
        model = Gemma4VisionModel(
            _vision_config(num_hidden_layers=1),
            text_hidden_size=12,
            dtype=jnp.bfloat16,
            rngs=None,
            mesh=mesh,
            vision_tp=False,
            input_buckets=(9,),
        )

    attention = model.layers[0].self_attn
    attention.q_proj = Float32Projection()
    attention.k_proj = Float32Projection()
    attention.v_proj = Float32Projection()

    item = _item(3, 3)
    patches, position_ids, patch_counts, output_indices = _pack(model, [item])
    output = model.encode(patches, position_ids, patch_counts)
    packed = restore_encoder_output(output, output_indices, mesh)

    assert attention_dtypes == {"q": jnp.bfloat16, "k": jnp.bfloat16, "v": jnp.bfloat16}
    assert bool(jnp.all(jnp.isfinite(packed)))


def test_vision_weight_mappings_match_gemma4_checkpoint_layout():
    fake_model = SimpleNamespace(
        visual=SimpleNamespace(
            standardize=True,
            specs=SimpleNamespace(
                col_kernel_axes=(None, "tensor"),
                row_kernel_axes=("tensor", None),
                tensor_axis="tensor",
            ),
        ),
        root_config=SimpleNamespace(vision_config=SimpleNamespace(num_hidden_layers=1)),
    )

    mappings = Gemma4ForConditionalGeneration._create_vision_weight_mappings(fake_model)

    assert (
        mappings["model.vision_tower.patch_embedder.position_embedding_table"].target_path
        == "visual.patch_embedder.position_embedding_table"
    )
    assert (
        mappings["model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight"].target_path
        == "visual.layers.0.self_attn.q_proj.weight"
    )
    assert (
        mappings["model.embed_vision.embedding_projection.weight"].target_path
        == "visual.projector.embedding_projection.weight"
    )
    assert "model.vision_tower.std_scale" in mappings
