"""E-owned V4 loading using a small synthetic safetensors checkpoint.

Reference: epic/dsv4 8f4103184 (strict converter) and 55ae1f742 (loader).
CPU validation uses BF16 activations, E4M3FN expert weights, a two-device
EP mesh, and a 1% relative output tolerance for BF16 rounding. The generated
checkpoint is a format fixture, not a released DeepSeek V4 checkpoint.
"""

import json
import struct
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.deepseek_v4_moe import DeepseekV4MoE
from sgl_jax.srt.layers.deepseek_v4_moe_loader import (
    STATIC_EXPERT_FORMAT,
    _physical_to_logical_for_layer,
    expected_moe_keys,
)
from sgl_jax.srt.model_loader.weights.source import LocalSource

save_file = pytest.importorskip("safetensors.numpy").save_file


def test_nonaddressable_replicated_expert_map_uses_local_replica():
    class _GlobalMap:
        is_fully_addressable = False
        sharding = SimpleNamespace(is_fully_replicated=True)

        def __getitem__(self, index):
            raise AssertionError("global map must not be indexed on a single host")

        def addressable_data(self, index):
            assert index == 0
            return np.asarray([[1, 0, 1, 0]], np.int32)

    mapping = _GlobalMap()
    np.testing.assert_array_equal(
        _physical_to_logical_for_layer(SimpleNamespace(physical_to_logical_map=mapping), 0),
        [1, 0, 1, 0],
    )
    mapping.sharding = SimpleNamespace(is_fully_replicated=False)
    with pytest.raises(ValueError, match="replicated across hosts"):
        _physical_to_logical_for_layer(SimpleNamespace(physical_to_logical_map=mapping), 0)


def _inventory(path):
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_size))
    start = 8 + header_size
    return {
        key: [
            {
                "file": str(path),
                "shape": tuple(value["shape"]),
                "dtype": value["dtype"],
                "byte_offset": start + value["data_offsets"][0],
                "byte_size": value["data_offsets"][1] - value["data_offsets"][0],
            }
        ]
        for key, value in header.items()
        if key != "__metadata__"
    }


def _checkpoint(path, *, layer_id=0, hash_layer=True, distinct_experts=False):
    prefix = f"layers.{layer_id}.ffn."
    values = {
        prefix + "gate.weight": np.arange(64, dtype=np.float32).reshape(2, 32) / 128,
    }
    if hash_layer:
        values[prefix + "gate.tid2eid"] = np.tile(np.asarray([[0], [1]], np.int32), (8, 1))
    else:
        values[prefix + "gate.bias"] = np.asarray([0.25, -0.25], np.float32)
    for source, out_size, in_size in (("w1", 64, 32), ("w3", 64, 32), ("w2", 32, 64)):
        stem = prefix + f"shared_experts.{source}"
        values[stem + ".weight"] = np.full((out_size, in_size), 0.125, ml_dtypes.float8_e4m3fn)
        values[stem + ".scale"] = np.full(
            ((out_size + 127) // 128, (in_size + 127) // 128),
            127,
            np.uint8,
        ).view(ml_dtypes.float8_e8m0fnu)
        for expert_id in range(2):
            stem = prefix + f"experts.{expert_id}.{source}"
            code = 0x44 if distinct_experts and expert_id == 1 else 0x22
            values[stem + ".weight"] = np.full((out_size, in_size // 2), code, np.uint8).view(
                np.int8
            )
            values[stem + ".scale"] = np.full((out_size, in_size // 32), 127, np.uint8).view(
                ml_dtypes.float8_e8m0fnu
            )
    save_file(values, str(path))
    return _inventory(path)


def _static_checkpoint(path, *, size=128, shared_scale_codes=None, routed_weight=None, layer_id=0):
    prefix = f"layers.{layer_id}.ffn."
    values = {
        prefix + "gate.weight": np.ones((2, size), np.float32),
    }
    if layer_id == 0:
        values[prefix + "gate.tid2eid"] = np.tile(np.asarray([[0], [1]], np.int32), (8, 1))
    else:
        values[prefix + "gate.bias"] = np.asarray([0.25, -0.25], np.float32)
    if shared_scale_codes is None:
        shared_scale_codes = np.full((size // 128, size // 128), 127, np.uint8)
    for source in ("w1", "w3", "w2"):
        stem = prefix + f"shared_experts.{source}"
        values[stem + ".weight"] = np.full((size, size), 0.125, ml_dtypes.float8_e4m3fn)
        values[stem + ".scale"] = np.asarray(shared_scale_codes, np.uint8).view(
            ml_dtypes.float8_e8m0fnu
        )
        for expert_id in range(2):
            stem = prefix + f"experts.{expert_id}.{source}"
            expert_value = expert_id + 1 if routed_weight is None else routed_weight
            values[stem + ".weight"] = np.full((size, size), expert_value, ml_dtypes.float8_e4m3fn)
            values[stem + ".scale"] = np.ones((size,), np.float32)
    save_file(values, str(path))
    return _inventory(path)


def _layer(layer_id=0, *, data=1, tensor=1, ep_size=1, abstract=False):
    mesh = Mesh(
        np.asarray(jax.devices()[: data * tensor]).reshape(data, tensor),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    config = DeepseekV4Config(
        hidden_size=32,
        num_hidden_layers=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        moe_intermediate_size=64,
        vocab_size=16,
        num_hash_layers=1,
        expert_dtype="fp4",
        ep_size=ep_size,
    )
    with jax.set_mesh(mesh):
        initialize = lambda: DeepseekV4MoE(config, mesh, layer_id)
        return nnx.eval_shape(initialize) if abstract else initialize()


def _static_layer(
    *, ep_size=1, tensor=1, backend="reference", size=128, layer_id=0, abstract=False
):
    if len(jax.devices()) < ep_size * tensor:
        pytest.skip("requires enough devices for expert parallel loading")
    mesh = Mesh(
        np.asarray(jax.devices()[: ep_size * tensor]).reshape(ep_size, tensor),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    config = DeepseekV4Config(
        hidden_size=size,
        num_hidden_layers=2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        moe_intermediate_size=size,
        vocab_size=16,
        num_hash_layers=1,
        expert_dtype="fp4",
        ep_size=ep_size,
        quantization_config=QuantizationConfig(is_static_checkpoint=True),
    )
    with jax.set_mesh(mesh):
        initialize = lambda: DeepseekV4MoE(config, mesh, layer_id, backend=backend)
        return nnx.eval_shape(initialize) if abstract else initialize()


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("layer_id", [0, 1])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_serving_abstract_parameters_load_on_concrete_mesh(tmp_path, static, layer_id, ep_size):
    # JAXModelLoader constructs nnx.eval_shape models, unlike direct layer tests.
    # Every ordinary parameter starts with AbstractMesh sharding at this boundary.
    if static:
        layer = _static_layer(ep_size=ep_size, layer_id=layer_id, abstract=True)
        assigned = _static_checkpoint(tmp_path / "static.safetensors", layer_id=layer_id)
        expert_format = STATIC_EXPERT_FORMAT
    else:
        layer = _layer(layer_id, data=ep_size, ep_size=ep_size, abstract=True)
        assigned = _checkpoint(
            tmp_path / "online.safetensors", layer_id=layer_id, hash_layer=layer_id == 0
        )
        expert_format = None
    assert isinstance(layer.gate.kernel.value, jax.ShapeDtypeStruct)
    assert isinstance(layer.gate.kernel.value.sharding.mesh, jax.sharding.AbstractMesh)
    report = layer.load_owned_weights(assigned, expert_format=expert_format)
    assert report.consumed_keys == set(assigned)
    prefix = f"layers.{layer_id}.ffn."
    expected_gate = (
        np.ones((128, 2), np.float32)
        if static
        else np.arange(64, dtype=np.float32).reshape(2, 32).T / 128
    )
    np.testing.assert_array_equal(np.asarray(layer.gate.kernel.value), expected_gate)
    if layer_id:
        np.testing.assert_array_equal(np.asarray(layer.gate.bias.value), [0.25, -0.25])
    else:
        assert layer._hash_table_loaded
        np.testing.assert_array_equal(
            np.asarray(layer.gate.tid2eid.value), np.tile([[0], [1]], (8, 1))
        )
    for value in jax.tree.leaves(nnx.state(layer, nnx.Param)):
        assert isinstance(value, jax.Array), f"unmaterialized serving parameter in {prefix}"
        assert value.is_fully_addressable
        assert not isinstance(value.sharding.mesh, jax.sharding.AbstractMesh)


@pytest.mark.parametrize("static", [False, True])
def test_shared_weight_source_inventory_hands_off_to_e(tmp_path, static):
    path = tmp_path / "model.safetensors"
    if static:
        _static_checkpoint(path)
        layer = _static_layer()
        expert_format = STATIC_EXPERT_FORMAT
    else:
        _checkpoint(path)
        layer = _layer()
        expert_format = None

    with LocalSource(SimpleNamespace(model_path=str(tmp_path))) as source:
        inventory = source.metadata
        assert set(inventory) == expected_moe_keys(layer)
        scale_key = "layers.0.ffn.shared_experts.w1.scale"
        scale = source.read_tensor(inventory[scale_key][0]["file"], scale_key, slice(None))
        assert scale.dtype == ml_dtypes.float8_e8m0fnu
        report = layer.load_owned_weights(inventory, expert_format=expert_format)

    assert report.consumed_keys == set(inventory)
    assert report.local_payload_keys == report.consumed_keys


@pytest.mark.parametrize("ep_size", [1, 2])
def test_static_expert_fp8_loads_directly_with_quantized_shared_expert(
    tmp_path, monkeypatch, ep_size
):
    from sgl_jax.srt.layers import deepseek_v4_moe_loader as loader

    layer = _static_layer(ep_size=ep_size)
    assigned = _static_checkpoint(tmp_path / "static.safetensors")

    def no_online_conversion(*args, **kwargs):
        raise AssertionError("static loading must not convert MXFP4")

    monkeypatch.setattr(loader, "convert_mxfp4_pair_from_reader", no_online_conversion)
    report = layer.load_owned_weights(assigned, expert_format=STATIC_EXPERT_FORMAT)
    assert report.consumed_keys == expected_moe_keys(layer)
    assert report.local_payload_keys == report.consumed_keys
    assert report.converted_pairs == 0
    for name in ("wi_0", "wi_1", "wo"):
        weight = getattr(layer.experts, name).value
        scale = getattr(layer.experts, name + "_scale").value
        assert weight.dtype == jnp.float8_e4m3fn
        np.testing.assert_array_equal(np.asarray(weight, np.float32)[0], 1)
        np.testing.assert_array_equal(np.asarray(weight, np.float32)[1], 2)
        np.testing.assert_array_equal(np.asarray(scale), 1)
    for name in ("gate_proj", "up_proj", "down_proj"):
        linear = getattr(layer.shared_experts, name)
        assert linear.weight_q.value.dtype == jnp.float8_e4m3fn
        np.testing.assert_array_equal(np.asarray(linear.weight_q.value, np.float32), 0.125)
        np.testing.assert_array_equal(np.asarray(linear.weight_scale.value), 1)


@pytest.mark.parametrize("ep_size,tensor", [(1, 1), (2, 1), (1, 2)])
def test_reference_static_expert_fp8_complete_moe_output(tmp_path, ep_size, tensor):
    layer = _static_layer(ep_size=ep_size, tensor=tensor)
    assigned = _static_checkpoint(tmp_path / "static.safetensors")
    layer.load_owned_weights(assigned, expert_format=STATIC_EXPERT_FORMAT)
    with jax.set_mesh(layer.mesh):
        output, ids = layer(
            jnp.ones((ep_size, 128), jnp.bfloat16),
            jnp.zeros((ep_size,), jnp.int32),
            return_expert_ids=True,
        )
    np.testing.assert_array_equal(np.asarray(ids), np.zeros((ep_size, 1), np.int32))
    expected = 1.5 * 128 * (10 / (1 + np.exp(-10))) * 10
    expected += 128 * 0.125 * (10 / (1 + np.exp(-10))) * 10
    np.testing.assert_allclose(np.asarray(output, np.float32), expected, rtol=0.03)


def test_reference_static_shared_block_scales_match_numpy(tmp_path):
    size = 256
    codes = np.asarray([[127, 128], [129, 127]], np.uint8)
    layer = _static_layer(size=size)
    assigned = _static_checkpoint(
        tmp_path / "block_scales.safetensors",
        size=size,
        shared_scale_codes=codes,
        routed_weight=0,
    )
    layer.load_owned_weights(assigned, expert_format=STATIC_EXPERT_FORMAT)
    hidden = np.full((1, size), 1 / size, np.float32)
    with jax.set_mesh(layer.mesh):
        actual = layer(jnp.asarray(hidden, jnp.bfloat16), jnp.asarray([0], jnp.int32))

    scale = np.exp2(codes.astype(np.int32) - 127)
    weight = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1) * 0.125
    gate = hidden @ weight.T
    activated = (gate / (1 + np.exp(-gate))) * gate
    expected = activated @ weight.T
    np.testing.assert_allclose(np.asarray(actual, np.float32), expected, rtol=0.01)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="static FP8 TPU kernels required")
def test_tpu_static_expert_fp8_complete_moe_output(tmp_path):
    layer = _static_layer(backend="epmoe")
    assigned = _static_checkpoint(tmp_path / "static.safetensors")
    layer.load_owned_weights(assigned, expert_format=STATIC_EXPERT_FORMAT)
    with jax.set_mesh(layer.mesh):
        output, ids = layer(
            jnp.ones((1, 128), jnp.bfloat16),
            jnp.asarray([0], jnp.int32),
            return_expert_ids=True,
        )
    np.testing.assert_array_equal(np.asarray(ids), [[0]])
    expected = 1.5 * 128 * (10 / (1 + np.exp(-10))) * 10
    expected += 128 * 0.125 * (10 / (1 + np.exp(-10))) * 10
    np.testing.assert_allclose(np.asarray(output, np.float32), expected, rtol=0.03)


def test_static_expert_fp8_rejects_wrong_mode_and_metadata(tmp_path):
    layer = _static_layer()
    assigned = _static_checkpoint(tmp_path / "static.safetensors")
    with pytest.raises(ValueError):
        layer.load_owned_weights(assigned)
    with pytest.raises(ValueError, match="unsupported V4 expert format"):
        layer.load_owned_weights(assigned, expert_format="unknown")
    with pytest.raises(ValueError, match="static FP8 shared-expert parameters"):
        _layer().load_owned_weights(assigned, expert_format=STATIC_EXPERT_FORMAT)
    key = "layers.0.ffn.experts.0.w1.scale"
    broken = {**assigned, key: [{**assigned[key][0], "dtype": "F8_E8M0"}]}
    with pytest.raises(ValueError, match=key):
        layer.load_owned_weights(broken, expert_format=STATIC_EXPERT_FORMAT)
    for broken in (
        {name: entries for name, entries in assigned.items() if name != key},
        {**assigned, key: assigned[key] * 2},
        {**assigned, key: [{**assigned[key][0], "shape": (1,)}]},
        {**assigned, key: [{**assigned[key][0], "byte_size": 1}]},
        {**assigned, "layers.1.ffn.gate.weight": assigned["layers.0.ffn.gate.weight"]},
    ):
        with pytest.raises(ValueError):
            layer.load_owned_weights(broken, expert_format=STATIC_EXPERT_FORMAT)


@pytest.mark.parametrize("dispatch_algorithm", ["static", "dynamic"])
def test_static_experts_follow_physical_to_logical_placement(tmp_path, dispatch_algorithm):
    from sgl_jax.srt.eplb.expert_location import (
        get_global_expert_location_metadata,
        set_global_expert_location_metadata,
    )

    if len(jax.devices()) < 2:
        pytest.skip("requires two devices for expert parallel placement")
    previous = get_global_expert_location_metadata()
    locations = SimpleNamespace(
        num_physical_experts=4,
        physical_to_logical_map=np.asarray([[1, 0, 1, 0]], np.int32),
        ep_dispatch_algorithm=dispatch_algorithm,
        logical_to_rank_dispatch_physical_map=None,
        logical_to_all_physical_map=None,
        logical_to_all_physical_map_num_valid=None,
    )
    try:
        set_global_expert_location_metadata(locations)
        layer = _static_layer(ep_size=2)
        locations.logical_to_rank_dispatch_physical_map = jax.device_put(
            np.asarray([[1, 0]], np.int32), NamedSharding(layer.mesh, P(None, None))
        )
        locations.logical_to_all_physical_map = jax.device_put(
            np.asarray([[[1, 3], [0, 2]]], np.int32),
            NamedSharding(layer.mesh, P(None, None, None)),
        )
        locations.logical_to_all_physical_map_num_valid = jax.device_put(
            np.asarray([[2, 2]], np.int32), NamedSharding(layer.mesh, P(None, None))
        )
        report = layer.load_owned_weights(
            _static_checkpoint(tmp_path / "static.safetensors"),
            expert_format=STATIC_EXPERT_FORMAT,
        )
        assert report.consumed_keys == expected_moe_keys(layer)
        np.testing.assert_array_equal(
            np.asarray(layer.experts.wi_0.value, np.float32)[:, 0, 0], [2, 1, 2, 1]
        )
        with jax.set_mesh(layer.mesh):
            _, physical_ids, logical_ids = layer.route(
                jnp.ones((2, 128), jnp.bfloat16),
                jnp.asarray([0, 1], jnp.int32),
                dispatch_info=locations,
                return_logical_ids=True,
            )
        np.testing.assert_array_equal(np.asarray(logical_ids), [[0], [1]])
        if dispatch_algorithm == "static":
            np.testing.assert_array_equal(np.asarray(physical_ids), [[1], [0]])
        else:
            physical_ids = np.asarray(physical_ids)
            assert int(physical_ids[0, 0]) in (1, 3)
            assert int(physical_ids[1, 0]) in (0, 2)
    finally:
        set_global_expert_location_metadata(previous)


def test_online_experts_follow_physical_to_logical_placement(tmp_path):
    from sgl_jax.srt.eplb.expert_location import (
        get_global_expert_location_metadata,
        set_global_expert_location_metadata,
    )

    if len(jax.devices()) < 2:
        pytest.skip("requires two devices for expert parallel placement")
    previous = get_global_expert_location_metadata()
    try:
        locations = SimpleNamespace(
            num_physical_experts=4,
            physical_to_logical_map=np.asarray([[1, 0, 1, 0]], np.int32),
            ep_dispatch_algorithm="static",
        )
        set_global_expert_location_metadata(locations)
        layer = _layer(data=2, ep_size=2)
        locations.logical_to_rank_dispatch_physical_map = jax.device_put(
            np.asarray([[1, 0]], np.int32), NamedSharding(layer.mesh, P(None, None))
        )
        assigned = _checkpoint(tmp_path / "online.safetensors", distinct_experts=True)
        report = layer.load_owned_weights(assigned, row_chunk_size=7)
        assert report.consumed_keys == set(assigned)
        for name in ("wi_0", "wi_1", "wo"):
            weight = np.asarray(getattr(layer.experts, name).value, np.float32)
            scale = np.asarray(getattr(layer.experts, name + "_scale").value)
            np.testing.assert_array_equal(weight[:, 0, 0] * scale[:, 0, 0, 0], [2, 1, 2, 1])
        set_global_expert_location_metadata(None)
        identity_layer = _layer(data=2, ep_size=2)
        identity_layer.load_owned_weights(assigned, row_chunk_size=7)
        hidden = jnp.ones((2, 32), jnp.bfloat16)
        token_ids = jnp.asarray([0, 1], jnp.int32)
        with jax.set_mesh(layer.mesh):
            placed_output, placed_ids = layer(
                hidden, token_ids, dispatch_info=locations, return_expert_ids=True
            )
            identity_output, identity_ids = identity_layer(
                hidden, token_ids, return_expert_ids=True
            )
        np.testing.assert_array_equal(np.asarray(placed_ids), np.asarray(identity_ids))
        np.testing.assert_array_equal(np.asarray(placed_output), np.asarray(identity_output))
    finally:
        set_global_expert_location_metadata(previous)


@pytest.mark.parametrize("layer_id", [0, 1])
def test_real_pair_loading_covers_exact_e_partition(tmp_path, layer_id):
    layer = _layer(layer_id)
    assigned = _checkpoint(
        tmp_path / "model.safetensors", layer_id=layer_id, hash_layer=layer_id == 0
    )
    report = layer.load_owned_weights(assigned, row_chunk_size=17)
    assert report.consumed_keys == expected_moe_keys(layer) == set(assigned)
    assert report.local_payload_keys == report.consumed_keys
    assert report.converted_pairs == 6
    assert report.max_conversion_error == 0
    assert report.peak_host_bytes_calculated > 0
    np.testing.assert_array_equal(
        np.asarray(layer.gate.kernel.value),
        np.asarray(np.arange(64, dtype=np.float32).reshape(2, 32) / 128).T,
    )
    for source, target, shape in (
        ("w1", "wi_0", (2, 32, 64)),
        ("w3", "wi_1", (2, 32, 64)),
        ("w2", "wo", (2, 64, 32)),
    ):
        weight = getattr(layer.experts, target).value
        scale = getattr(layer.experts, target + "_scale").value
        assert weight.shape == shape
        np.testing.assert_array_equal(np.asarray(weight, np.float32) * np.asarray(scale), 1)
    if layer_id == 0:
        assert layer._hash_table_loaded
        weights, ids = layer.route(
            jnp.zeros((2, 32), jnp.bfloat16),
            jnp.asarray([0, 1], jnp.int32),
        )
        np.testing.assert_array_equal(np.asarray(ids), [[0], [1]])
        np.testing.assert_allclose(np.asarray(weights), 1.5)


def test_online_loader_uses_assigned_inventory_without_rescanning(tmp_path, monkeypatch):
    from sgl_jax.srt.utils.quantization import mxfp4_fp8_loader

    layer = _layer()
    assigned = _checkpoint(tmp_path / "model.safetensors")
    monkeypatch.setattr(
        mxfp4_fp8_loader,
        "_read_safetensors_header",
        lambda path: (_ for _ in ()).throw(AssertionError("E must use M's assigned entries")),
    )
    report = layer.load_owned_weights(assigned, row_chunk_size=7)
    assert report.consumed_keys == set(assigned)


def test_inventory_rejects_missing_duplicate_wrong_layer_and_shape(tmp_path):
    layer = _layer()
    assigned = _checkpoint(tmp_path / "model.safetensors")
    stem = "layers.0.ffn.experts.0.w1.weight"
    for broken in (
        {key: value for key, value in assigned.items() if key != stem},
        {**assigned, stem: assigned[stem] * 2},
        {**assigned, stem: [{**assigned[stem][0], "shape": (1, 16)}]},
        {**assigned, "layers.1.ffn.gate.weight": assigned["layers.0.ffn.gate.weight"]},
    ):
        with pytest.raises(ValueError):
            layer.load_owned_weights(broken)


def test_reserved_shared_scale_is_rejected(tmp_path):
    layer = _layer()
    assigned = _checkpoint(tmp_path / "model.safetensors")
    key = "layers.0.ffn.shared_experts.w1.scale"
    entry = assigned[key][0]
    with open(entry["file"], "r+b") as target:
        target.seek(entry["byte_offset"])
        target.write(b"\xff")
    with pytest.raises(ValueError, match="reserved"):
        layer.load_owned_weights(assigned)


def test_complete_output_after_loading_and_nonfinite_padding(tmp_path):
    layer = _layer()
    layer.load_owned_weights(_checkpoint(tmp_path / "model.safetensors"))
    x = jnp.asarray(
        np.concatenate((np.ones((1, 32), np.float32), np.full((2, 32), np.nan, np.float32))),
        jnp.bfloat16,
    )
    token_ids = jnp.asarray([0, -1, 100], jnp.int32)
    mask = jnp.asarray([True, False, True])
    with jax.set_mesh(layer.mesh):
        output = jax.jit(lambda h, ids, valid: layer(h, ids, token_valid_mask=valid))(
            x, token_ids, mask
        )
    assert output.shape == (3, 32)
    assert output.dtype == jnp.bfloat16
    # All routed projections contain 1 and all shared projections contain 1/8.
    routed = 1.5 * 64 * (10 / (1 + np.exp(-10))) * 10
    shared = 64 * 0.125 * (4 / (1 + np.exp(-4))) * 4
    np.testing.assert_allclose(np.asarray(output[0], np.float32), routed + shared, rtol=0.01)
    np.testing.assert_array_equal(np.asarray(output[1:], np.float32), 0)


def test_local_expert_assembly_and_independent_output_sharding(tmp_path):
    if len(jax.devices()) < 2:
        pytest.skip("requires two devices for expert parallel assembly")
    layer = _layer(data=2, ep_size=2)
    report = layer.load_owned_weights(_checkpoint(tmp_path / "model.safetensors"), row_chunk_size=7)
    assert report.converted_pairs == 6
    assert layer.experts.wi_0.value.sharding.spec == P("expert", None, "tensor")
    np.testing.assert_array_equal(
        np.asarray(layer.experts.wi_0.value, np.float32)
        * np.asarray(layer.experts.wi_0_scale.value),
        1,
    )
    x = jax.device_put(jnp.ones((4, 32), jnp.bfloat16), NamedSharding(layer.mesh, P("data", None)))
    token_ids = jax.device_put(
        jnp.asarray([0, 1, 0, 1], jnp.int32), NamedSharding(layer.mesh, P("data"))
    )
    with jax.set_mesh(layer.mesh):
        output = jax.jit(
            lambda h, ids: layer(
                h,
                ids,
                out_sharding=NamedSharding(layer.mesh, P(None, None)),
                output_sharding=NamedSharding(layer.mesh, P("data", None)),
            )
        )(x, token_ids)
    assert output.sharding.spec == P("data", None)
    output_host = np.asarray(output, np.float32)
    np.testing.assert_allclose(output_host, np.broadcast_to(output_host[0], output_host.shape))
