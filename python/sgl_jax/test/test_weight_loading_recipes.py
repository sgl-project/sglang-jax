"""Real safetensors -> NNX/JAX loads, with independent numerical references."""

import json
import struct
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors.numpy import save_file

from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.mimo_weight_loading import load_fused_kv, load_fused_qkv


@pytest.fixture
def mesh():
    return Mesh(np.array(jax.devices()), ("tensor",), axis_types=(AxisType.Explicit,))


def save_weights(path, weights):
    # Portable writer, including safetensors' BF16/FP8 byte representation.
    names = {
        np.dtype(np.float32): "F32",
        np.dtype(ml_dtypes.bfloat16): "BF16",
        np.dtype(ml_dtypes.float8_e4m3fn): "F8_E4M3",
    }
    offset, header, data = 0, {}, []
    for key, value in weights.items():
        raw = value.tobytes()
        header[key] = {
            "dtype": names[value.dtype],
            "shape": value.shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        data.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(data))


def load_group(tmp_path, mesh, weights, expected, recipe):
    save_weights(tmp_path / "model.safetensors", weights)
    model = nnx.Module()
    targets = []
    for i, value in enumerate(expected):
        path = f"w{i}"
        targets.append(path)
        setattr(model, path, nnx.Param(jax.ShapeDtypeStruct(value.shape, value.dtype)))
    config = SimpleNamespace(model_path=str(tmp_path), quantization_config=None)
    with jax.set_mesh(mesh):
        WeightLoader(model, config, mesh).load(
            {"group": WeightSpec(targets, sources=tuple(weights), recipe=recipe)}
        )
    for path, value in zip(targets, expected):
        actual = getattr(model, path).value
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint16), value.view(np.uint16))
        assert actual.sharding.spec == P(None, "tensor")


def test_mimo_pro_shard_interleaving_and_cross_block_scale(tmp_path, mesh):
    rng = np.random.default_rng(4)
    sizes = (4 * 192, 192, 128)
    shard_rows, block, cols = sum(sizes), 128, 256
    padded = (shard_rows + block - 1) // block * block
    weight = rng.normal(size=(2 * shard_rows, cols)).astype(ml_dtypes.float8_e4m3fn)
    scale = rng.uniform(0.01, 0.2, size=(2 * padded // block, cols // block)).astype(np.float32)
    expected = [[], [], []]
    for shard in range(2):
        w = weight[shard * shard_rows : (shard + 1) * shard_rows].astype(np.float32)
        s = scale[shard * (padded // block) : (shard + 1) * (padded // block)]
        expanded = np.repeat(np.repeat(s, block, axis=0), block, axis=1)[:shard_rows]
        parts = np.split(w * expanded, np.cumsum(sizes)[:-1], axis=0)
        for group, part in zip(expected, parts):
            group.append(part.T)
    expected = [np.concatenate(parts, axis=1).astype(ml_dtypes.bfloat16) for parts in expected]
    load_group(
        tmp_path,
        mesh,
        {"w": weight, "s": scale},
        expected,
        partial(
            load_fused_qkv,
            head_dim=192,
            v_head_dim=128,
            num_heads=8,
            kv_heads=2,
            block_size=128,
            mesh=mesh,
        ),
    )


@pytest.mark.parametrize("per_head", [False, True])
def test_mimo_flash_kv_and_head_replication(tmp_path, mesh, per_head):
    rng = np.random.default_rng(8)
    heads, kh, vh, cols, block = 2, 192, 128, 256, 128
    k = rng.normal(size=(heads * kh, cols)).astype(ml_dtypes.float8_e4m3fn)
    v = rng.normal(size=(heads * vh, cols)).astype(ml_dtypes.float8_e4m3fn)
    ks = rng.uniform(0.01, 0.2, size=(4 if per_head else 3, 2)).astype(np.float32)
    vs = rng.uniform(0.01, 0.2, size=(2, 2)).astype(np.float32)
    if per_head:
        fused = np.concatenate((k.reshape(heads, kh, cols), v.reshape(heads, vh, cols)), axis=1)
        scales = np.concatenate((ks.reshape(heads, 2, 2), vs.reshape(heads, 1, 2)), axis=1)
        values = (
            fused.astype(np.float32)
            * np.repeat(np.repeat(scales, block, 1), block, 2)[:, : kh + vh]
        )
        values = values.astype(ml_dtypes.bfloat16)
        expected = (values[:, :kh].reshape(-1, cols), values[:, kh:].reshape(-1, cols))
    else:
        expected = tuple(
            (w.astype(np.float32) * np.repeat(np.repeat(s, block, 0), block, 1)).astype(
                ml_dtypes.bfloat16
            )
            for w, s in ((k, ks), (v, vs))
        )
    expected = [
        np.repeat(w.reshape(heads, hd, cols), 2, axis=0).reshape(-1, cols).T.copy()
        for w, hd in zip(expected, (kh, vh))
    ]
    load_group(
        tmp_path,
        mesh,
        {"k": k, "ks": ks, "v": v, "vs": vs},
        expected,
        partial(
            load_fused_kv,
            head_dim=kh,
            v_head_dim=vh,
            block_size=block,
            kv_heads=4,
            mesh=mesh,
        ),
    )


def test_split_file_dense_and_explicit_alias(tmp_path, mesh):
    from sgl_jax.srt.model_loader.weights import JaxShardReader, LocalSource

    original = np.arange(128, dtype=np.float32).reshape(16, 8)
    for i, part in enumerate(np.split(original, 2)):
        save_file({"weight": part}, tmp_path / f"{i}.safetensors")
    model = nnx.Module()
    model.weight = nnx.Param(
        jax.ShapeDtypeStruct((8, 16), jnp.float32, sharding=NamedSharding(mesh, P(None, "tensor")))
    )
    model.tied = model.weight
    config = SimpleNamespace(model_path=str(tmp_path))
    with LocalSource(config) as source, jax.set_mesh(mesh):
        WeightLoader(model, config, mesh, source=source, reader=JaxShardReader(mesh)).load(
            {"weight": WeightSpec("weight", transpose=True, concat_axis=0)}
        )
    assert model.tied is model.weight
    np.testing.assert_array_equal(model.weight.value, original.T)


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_invalid_plan_fails_before_assignment(tmp_path, mesh, kind):
    save_file(
        {"a": np.ones((4, 4), np.float32), "b": np.ones((4, 4), np.float32)},
        tmp_path / "m.safetensors",
    )
    model = nnx.Module()
    original = jax.ShapeDtypeStruct((4, 4), jnp.float32)
    model.weight = nnx.Param(original)
    specs = {"a": WeightSpec("weight")}
    specs["absent" if kind == "missing" else "b"] = WeightSpec("weight")
    with pytest.raises(ValueError, match="Missing|Duplicate"):
        WeightLoader(model, SimpleNamespace(model_path=str(tmp_path)), mesh).load(specs)
    assert model.weight.value is original


@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_recipe_output_rejected_before_parameter_write(tmp_path, mesh, invalid):
    save_file({"w": np.ones((8, 8), np.float32)}, tmp_path / "model.safetensors")
    model = nnx.Module()
    model.weight = nnx.Param(jax.ShapeDtypeStruct((8, 8), jnp.float32))
    parameter = model.weight

    def recipe(inputs):
        value = jnp.asarray(inputs[0])
        return (value[:4] if invalid == "shape" else value.astype(jnp.bfloat16),)

    with jax.set_mesh(mesh), pytest.raises(ValueError, match="Loaded target weight"):
        WeightLoader(model, SimpleNamespace(model_path=str(tmp_path)), mesh).load(
            {"group": WeightSpec("weight", sources=("w",), recipe=recipe)}
        )
    assert model.weight is parameter
    assert isinstance(parameter.value, jax.ShapeDtypeStruct)
    assert parameter.value.shape == (8, 8)
    assert parameter.value.dtype == jnp.float32


def test_pd_cache_tracks_checkpoint_and_expert_placement(tmp_path, mesh, monkeypatch):
    from sgl_jax.srt.model_loader.weights.loader import _PD_WEIGHT_CACHE

    monkeypatch.setenv("SGLANG_PD_WEIGHT_CACHE", "1")
    _PD_WEIGHT_CACHE.clear()
    original = np.arange(128, dtype=np.float32).reshape(16, 8)
    filename = tmp_path / "model.safetensors"
    save_file({"e0": original, "e1": original + 1}, filename)
    for offset, placement in ((0, [0, 1]), (0, [1, 0]), (10, [0, 1])):
        if offset:
            replacement = tmp_path / "replacement"
            save_file({"e0": original + offset, "e1": original + offset + 1}, replacement)
            replacement.replace(filename)
        model = nnx.Module()
        model.w = nnx.Param(jax.ShapeDtypeStruct((2, 8, 16), np.float32))
        with jax.set_mesh(mesh):
            WeightLoader(model, SimpleNamespace(model_path=str(tmp_path)), mesh).load(
                {
                    "experts": WeightSpec(
                        "w",
                        sources=("e0", "e1"),
                        transpose=True,
                        sharding=(None, None, "tensor"),
                        physical_to_logical_map=np.asarray(placement),
                    )
                }
            )
        np.testing.assert_array_equal(
            model.w.value, np.stack([(original + offset + i).T for i in placement])
        )
    assert len(_PD_WEIGHT_CACHE) == 3
    _PD_WEIGHT_CACHE.clear()


def test_mimo_prepare_binds_final_bf16_parameters(tmp_path, mesh):
    from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
    from sgl_jax.srt.models.mimo_weight_loading import prepare_mimo

    class Model(nnx.Module):
        def __init__(self):
            self.layer = nnx.Module()
            attn = self.layer.self_attn = nnx.Module()
            attn.head_dim = attn.v_head_dim = 128
            attn.k_head_num = 2
            for name in ("q_proj", "k_proj", "v_proj"):
                setattr(
                    attn,
                    name,
                    QuantizedLinear(
                        jax.ShapeDtypeStruct((256, 256), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((2, 1, 256), jnp.float32),
                        None,
                        None,
                        mesh,
                        (None, "tensor"),
                        weight_block_size=(128, 128),
                    ),
                )

        def prepare_weight_loading(self, loader, mappings):
            return prepare_mimo(self, loader, mappings, [("layer", self.layer, False)])

    rng = np.random.default_rng(13)
    weights, mappings, expected = {}, {}, {}
    for name in ("q_proj", "k_proj", "v_proj"):
        weight = rng.normal(size=(256, 256)).astype(ml_dtypes.float8_e4m3fn)
        scale = rng.uniform(0.01, 0.2, size=(2, 2)).astype(np.float32)
        weights[name + ".weight"], weights[name + ".scale"] = weight, scale
        mappings[name + ".weight"] = WeightSpec(f"layer.self_attn.{name}.weight_q", transpose=True)
        mappings[name + ".scale"] = WeightSpec(f"layer.self_attn.{name}.weight_scale")
        expected[name] = (
            weight.astype(np.float32) * np.repeat(np.repeat(scale, 128, 0), 128, 1)
        ).T.astype(ml_dtypes.bfloat16)
    save_weights(tmp_path / "model.safetensors", weights)
    config = SimpleNamespace(
        model_path=str(tmp_path),
        quantization_config=SimpleNamespace(
            is_static_checkpoint=True, weight_block_size=(128, 128)
        ),
    )
    with jax.set_mesh(mesh):
        model = Model()
        WeightLoader(model, config, mesh).load(mappings)
    for name, value in expected.items():
        proj = getattr(model.layer.self_attn, name)
        assert isinstance(proj, LinearBase)
        np.testing.assert_array_equal(
            np.asarray(proj.weight.value).view(np.uint16), value.view(np.uint16)
        )


def test_glm52_shared_expert_prepare_and_requantization(tmp_path, mesh):
    from sgl_jax.srt.layers.fused_moe import FusedEPMoEV2
    from sgl_jax.srt.models.glm5_moe import Glm5ForCausalLM
    from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

    mesh = Mesh(
        mesh.devices.reshape(1, -1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )

    class Model(nnx.Module):
        prepare_weight_loading = Glm5ForCausalLM.prepare_weight_loading

        def __init__(self):
            self.model = nnx.Module()
            layer = nnx.Module()
            layer.mlp = nnx.eval_shape(
                lambda: FusedEPMoEV2(
                    hidden_size=256,
                    num_experts=8,
                    num_experts_per_tok=2,
                    ep_size=1,
                    mesh=mesh,
                    intermediate_dim=256,
                    num_shared_experts=1,
                )
            )
            layer.mlp.quantized_dtype = jnp.float8_e4m3fn
            for name in ("w1_shared", "w3_shared", "w2_shared"):
                setattr(
                    layer.mlp,
                    name,
                    nnx.Param(jax.ShapeDtypeStruct((256, 256), jnp.float8_e4m3fn)),
                )
                delattr(layer.mlp, name + "_scale")
                setattr(
                    layer.mlp,
                    name + "_scale",
                    nnx.Param(jax.ShapeDtypeStruct((1, 1, 256), jnp.float32)),
                )
                setattr(
                    layer.mlp,
                    name + "_block_scale",
                    nnx.Param(jax.ShapeDtypeStruct((2, 2), jnp.float32)),
                )
            self.model.layers = nnx.List([layer])

    rng = np.random.default_rng(31)
    weights, specs, expected = {}, {}, {}
    with jax.set_mesh(mesh):
        for name in ("w1_shared", "w3_shared", "w2_shared"):
            w = rng.normal(size=(256, 256)).astype(ml_dtypes.float8_e4m3fn)
            scales = rng.uniform(0.01, 0.2, size=(2, 2)).astype(np.float32)
            weights[name], weights[name + "_scale"] = w, scales
            target = f"model.layers.0.mlp.{name}"
            specs[name] = WeightSpec(target, transpose=True)
            specs[name + "_scale"] = WeightSpec(target + "_block_scale")
            expanded = np.repeat(np.repeat(scales, 128, 0), 128, 1)
            real = jnp.asarray((w.astype(np.float32) * expanded).T)
            expected[name] = quantize_tensor(jnp.float8_e4m3fn, real, axis=0)
        save_weights(tmp_path / "model.safetensors", weights)
        model = Model()
        WeightLoader(model, SimpleNamespace(model_path=str(tmp_path)), mesh).load(specs)
    mlp = model.model.layers[0].mlp
    for name, (weight, scale) in expected.items():
        assert not hasattr(mlp, name + "_block_scale")
        np.testing.assert_array_equal(np.asarray(getattr(mlp, name).value), np.asarray(weight))
        np.testing.assert_array_equal(
            np.asarray(getattr(mlp, name + "_scale").value).reshape(-1),
            np.asarray(scale).reshape(-1),
        )


@pytest.mark.parametrize("split", [False, True])
def test_experts_tp_slices_transpose_and_redundant_placement(tmp_path, split):
    if jax.device_count() < 4:
        pytest.skip("requires four CPU devices")
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("expert", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    weights = {
        f"e.{i}": (np.arange(128, dtype=np.float32).reshape(16, 8) + i * 1000) for i in range(3)
    }
    if split:
        for part in range(2):
            save_file(
                {key: np.split(value, 2, axis=1)[part] for key, value in weights.items()},
                tmp_path / f"part-{part}.safetensors",
            )
    else:
        save_file(weights, tmp_path / "model.safetensors")
    placement = np.array([2, 0, 2, 1])
    model = nnx.Module()
    sharding = NamedSharding(mesh, P("expert", None, "tensor"))
    model.w = nnx.Param(jax.ShapeDtypeStruct((4, 8, 16), np.float32, sharding=sharding))
    with jax.set_mesh(mesh):
        WeightLoader(model, SimpleNamespace(model_path=str(tmp_path)), mesh).load(
            {
                "experts": WeightSpec(
                    "w",
                    sources=tuple(weights),
                    transpose=True,
                    physical_to_logical_map=placement,
                    concat_axis=1 if split else None,
                )
            }
        )
    expected = np.stack([weights[f"e.{i}"].T for i in placement])
    np.testing.assert_array_equal(model.w.value, expected)
    assert model.w.value.sharding == sharding


def test_bulk_experts_reads_real_offsets_and_bounds_merged_ranges(tmp_path, monkeypatch):
    from sgl_jax.srt.model_loader.weights import LocalSource
    from sgl_jax.srt.model_loader.weights.reader import JaxShardReader

    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        ("expert", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    weights = {
        "e.0": np.arange(1024 * 256, dtype=np.float32).reshape(1024, 256),
        "gap": np.zeros((4096, 256), np.float32),
        "e.1": np.ones((1024, 256), np.float32),
    }
    # Insertion order makes a large intervening tensor. The bulk reader must
    # still select correct byte ranges, including a duplicated physical slot.
    save_weights(tmp_path / "model.safetensors", weights)
    cfg = SimpleNamespace(model_path=str(tmp_path))
    monkeypatch.setenv("SGLANG_MOE_BULK_READ", "1")
    source = LocalSource(cfg)
    reader = JaxShardReader(mesh)
    value = reader.read(
        source,
        "experts",
        WeightSpec(
            "w", sources=("e.0", "e.1"), transpose=True, physical_to_logical_map=np.array([1, 0, 1])
        ),
        NamedSharding(mesh, P("expert", None, "tensor")),
    )
    np.testing.assert_array_equal(value, np.stack([weights[k].T for k in ("e.1", "e.0", "e.1")]))
    source.close()


@pytest.mark.parametrize("is_moe", [False, True])
def test_qwen35_complete_small_checkpoint(tmp_path, is_moe):
    from sgl_jax.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from sgl_jax.test.models.test_qwen3_5 import _make_config

    tp = min(2, jax.device_count())
    mesh = Mesh(
        np.asarray(jax.devices()[:tp]).reshape(1, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    config = _make_config(num_layers=4, is_moe=is_moe)
    # Deliberately different from the text tower (hidden=256, GQA heads=4/2).
    config.vision_config = SimpleNamespace(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        in_channels=3,
        out_hidden_size=256,
        spatial_merge_size=2,
        num_position_embeddings=16,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=[],
    )
    shapes = {
        "model.language_model.embed_tokens.weight": (256, 256),
        "model.language_model.norm.weight": (256,),
        "lm_head.weight": (256, 256),
        "model.visual.patch_embed.proj.weight": (32, 3, 2, 2, 2),
        "model.visual.patch_embed.proj.bias": (32,),
        "model.visual.pos_embed.weight": (16, 32),
    }
    visual = {
        "attn.qkv": (96, 32),
        "attn.proj": (32, 32),
        "mlp.linear_fc1": (64, 32),
        "mlp.linear_fc2": (32, 64),
    }
    for name, shape in visual.items():
        shapes[f"model.visual.blocks.0.{name}.weight"] = shape
        shapes[f"model.visual.blocks.0.{name}.bias"] = (shape[0],)
    for name in ("blocks.0.norm1", "blocks.0.norm2", "merger.norm"):
        for kind in ("weight", "bias"):
            shapes[f"model.visual.{name}.{kind}"] = (32,)
    for name, shape in {"linear_fc1": (128, 128), "linear_fc2": (256, 128)}.items():
        shapes[f"model.visual.merger.{name}.weight"] = shape
        shapes[f"model.visual.merger.{name}.bias"] = (shape[0],)
    for i in range(4):
        prefix = f"model.language_model.layers.{i}."
        layer = {
            "input_layernorm.weight": (256,),
            "post_attention_layernorm.weight": (256,),
        }
        if i == 3:
            layer.update(
                {
                    f"self_attn.{name}.weight": shape
                    for name, shape in {
                        "q_proj": (512, 256),
                        "k_proj": (128, 256),
                        "v_proj": (128, 256),
                        "o_proj": (256, 256),
                        "q_norm": (64,),
                        "k_norm": (64,),
                    }.items()
                }
            )
        else:
            layer.update(
                {
                    f"linear_attn.{name}": shape
                    for name, shape in {
                        "in_proj_qkv.weight": (256, 256),
                        "in_proj_z.weight": (128, 256),
                        "in_proj_b.weight": (4, 256),
                        "in_proj_a.weight": (4, 256),
                        "conv1d.weight": (256, 1, 4),
                        "A_log": (4,),
                        "dt_bias": (4,),
                        "norm.weight": (32,),
                        "out_proj.weight": (256, 128),
                    }.items()
                }
            )
        if is_moe:
            layer.update(
                {
                    "mlp.gate.weight": (8, 256),
                    "mlp.experts.gate_up_proj": (8, 256, 256),
                    "mlp.experts.down_proj": (8, 256, 128),
                    "mlp.shared_expert_gate.weight": (1, 256),
                }
            )
        mlp = "mlp.shared_expert" if is_moe else "mlp"
        layer.update(
            {
                f"{mlp}.{name}.weight": shape
                for name, shape in {
                    "gate_proj": (128, 256),
                    "up_proj": (128, 256),
                    "down_proj": (256, 128),
                }.items()
            }
        )
        shapes.update({prefix + key: shape for key, shape in layer.items()})
    rng = np.random.default_rng(19)
    weights = {key: rng.normal(size=shape).astype(np.float32) for key, shape in shapes.items()}
    save_file(weights, tmp_path / "model.safetensors")
    mc = SimpleNamespace(
        hf_config=config,
        model_path=str(tmp_path),
        num_hidden_layers=4,
        quantization_config=None,
        needs_kv_head_replication=lambda _: False,
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: Qwen3_5ForConditionalGeneration(config, mesh))
        model.load_weights(mc)
    params = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    assert params and all(isinstance(param, jax.Array) for param in params)
    for name, expected in zip("qkv", np.split(weights["model.visual.blocks.0.attn.qkv.weight"], 3)):
        projection = getattr(model.visual.blocks[0].attn, name + "_proj")
        np.testing.assert_array_equal(
            projection.weight.value, expected.T.astype(ml_dtypes.bfloat16)
        )
    for name, expected in zip("qkv", np.split(weights["model.visual.blocks.0.attn.qkv.bias"], 3)):
        np.testing.assert_array_equal(
            getattr(model.visual.blocks[0].attn, name + "_proj").bias.value,
            expected.astype(ml_dtypes.bfloat16),
        )
    for i, layer in enumerate(model.language_model.model.layers):
        prefix = f"model.language_model.layers.{i}."
        if i != 3:
            source = prefix + "linear_attn."
            expected = np.concatenate(
                [weights[source + name + ".weight"] for name in ("in_proj_qkv", "in_proj_z")],
                axis=0,
            ).T
            np.testing.assert_array_equal(
                layer.self_attn.in_proj_qkvz.weight.value,
                expected.astype(ml_dtypes.bfloat16),
            )
            conv = weights[source + "conv1d.weight"].reshape(256, 4)
            q, k, v = np.split(conv, [64, 128])
            stripes = [
                piece
                for rank in range(tp)
                for piece in (
                    np.split(q, tp)[rank],
                    np.split(k, tp)[rank],
                    np.split(v, tp)[rank],
                )
            ]
            np.testing.assert_array_equal(
                layer.self_attn.conv1d.weight.value,
                np.concatenate(stripes).astype(ml_dtypes.bfloat16),
            )
        if is_moe:
            gate, up = np.split(weights[prefix + "mlp.experts.gate_up_proj"], 2, axis=1)
            for name, value in (("w1", gate), ("w3", up)):
                np.testing.assert_array_equal(
                    getattr(layer.mlp.experts, name).value,
                    value.transpose(0, 2, 1).astype(ml_dtypes.bfloat16),
                )


@pytest.mark.parametrize("quantization", [None, "channel", "block"])
def test_absorbed_mla_final_targets(tmp_path, quantization):
    from sgl_jax.srt.layers.linear import QuantizedLinear
    from sgl_jax.srt.models.deepseek_v3 import DeepseekV3Attention

    tp = min(4, jax.device_count())
    mesh = Mesh(
        np.asarray(jax.devices()[:tp]).reshape(1, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rng = np.random.default_rng(5)
    weight = rng.normal(size=(1024, 128)).astype(
        ml_dtypes.float8_e4m3fn if quantization else ml_dtypes.bfloat16
    )
    values = {"kv.weight": weight}
    expected = weight.astype(np.float32)
    block = (128, 128) if quantization == "block" else None
    if quantization:
        scale = rng.uniform(0.01, 0.2, (8, 1) if block else (1024, 1)).astype(np.float32)
        values["kv.scale"] = scale
        expected *= np.repeat(np.repeat(scale, 128, 0), 128, 1) if block else scale
    save_weights(tmp_path / "model.safetensors", values)
    config = SimpleNamespace(
        model_path=str(tmp_path),
        quantization_config=SimpleNamespace(
            is_static_checkpoint=bool(quantization), weight_block_size=block
        ),
    )
    with jax.set_mesh(mesh):
        model = nnx.Module()
        model.attn = nnx.eval_shape(
            lambda: DeepseekV3Attention(
                hidden_size=256,
                num_heads=4,
                q_lora_rank=None,
                kv_lora_rank=128,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                mesh=mesh,
                use_absorbed=True,
                skip_rope=True,
            )
        )
        mappings = {"kv.weight": WeightSpec("attn.kv_b_proj.weight", transpose=True)}
        if quantization:
            model.attn.kv_b_proj = QuantizedLinear(
                jax.ShapeDtypeStruct(weight.shape, jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((1, 1, 1024) if block else (1024,), jnp.float32),
                None,
                None,
                mesh,
                (None, "tensor"),
                weight_block_size=block,
            )
            mappings = {
                "kv.weight": WeightSpec("attn.kv_b_proj.weight_q"),
                "kv.scale": WeightSpec("attn.kv_b_proj.weight_scale"),
            }
        WeightLoader(model, config, mesh).load(mappings)
    expected = expected.astype(ml_dtypes.bfloat16).T.reshape(128, 4, 256)
    assert model.attn.kv_b_proj is None
    for name, part in (("w_uk", expected[:, :, :128]), ("w_uv", expected[:, :, 128:])):
        value = getattr(model.attn, name).value
        np.testing.assert_array_equal(value, part)
        assert value.sharding.spec == P(None, "tensor", None)


def test_glm_fused_mlp_packing(tmp_path):
    from sgl_jax.srt.models.glm5_moe import Glm5MLP

    tp = min(4, jax.device_count())
    mesh = Mesh(
        np.asarray(jax.devices()[:tp]).reshape(1, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rng = np.random.default_rng(61)
    weights = {
        name: rng.normal(size=(256, 256)).astype(ml_dtypes.bfloat16)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    save_weights(tmp_path / "model.safetensors", weights)
    with jax.set_mesh(mesh):
        model = nnx.Module()
        model.mlp = nnx.eval_shape(lambda: Glm5MLP(256, 256, mesh))
        block = model.mlp.b_inter
        WeightLoader(model, SimpleNamespace(model_path=str(tmp_path)), mesh).load(
            {name: WeightSpec(f"mlp.{name}.weight", transpose=True) for name in weights}
        )
    chunks = [
        part
        for i in range(0, 256, block)
        for part in (
            weights["gate_proj"].T[:, i : i + block],
            weights["up_proj"].T[:, i : i + block],
        )
    ]
    np.testing.assert_array_equal(model.mlp.w_gu.value, np.concatenate(chunks, axis=1))
    np.testing.assert_array_equal(model.mlp.w_d.value, weights["down_proj"].T)
    assert model.mlp.gate_proj is model.mlp.up_proj is model.mlp.down_proj is None


def test_gemma4_prefused_experts_and_missing_head_alias(tmp_path):
    from sgl_jax.srt.models.gemma4 import Gemma4ForCausalLM

    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    config = SimpleNamespace(
        hidden_size=128,
        vocab_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        layer_types=["sliding_attention"],
        sliding_window=128,
        max_position_embeddings=256,
        intermediate_size=128,
        enable_moe_block=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        ep_size=2,
        tie_word_embeddings=False,
    )
    shapes = {"model.embed_tokens.weight": (256, 128), "model.norm.weight": (128,)}
    prefix = "model.layers.0."
    shapes[prefix + "layer_scalar"] = (1,)
    for name in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
        "pre_feedforward_layernorm_2",
    ):
        shapes[prefix + name + ".weight"] = (128,)
    for name, shape in {
        "q_proj": (256, 128),
        "k_proj": (256, 128),
        "v_proj": (256, 128),
        "o_proj": (128, 256),
        "q_norm": (128,),
        "k_norm": (128,),
    }.items():
        shapes[prefix + "self_attn." + name + ".weight"] = shape
    for name in ("gate_proj", "up_proj", "down_proj"):
        shapes[prefix + "mlp." + name + ".weight"] = (128, 128)
    for name, shape in {
        "router.scale": (128,),
        "router.per_expert_scale": (8,),
        "router.proj.weight": (8, 128),
        "experts.gate_up_proj": (8, 256, 128),
        "experts.down_proj": (8, 128, 128),
    }.items():
        shapes[prefix + name] = shape
    rng = np.random.default_rng(31)
    weights = {key: rng.normal(size=shape).astype(np.float32) for key, shape in shapes.items()}
    save_file(weights, tmp_path / "model.safetensors")
    mc = SimpleNamespace(model_path=str(tmp_path), hf_config=config, quantization_config=None)
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: Gemma4ForCausalLM(config, mesh))
        model.load_weights(mc)
    assert all(
        isinstance(value, jax.Array)
        for value in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    assert model.lm_head.embedding is model.model.embed_tokens.embedding
    expected = [
        *np.split(weights[prefix + "experts.gate_up_proj"], 2, axis=1),
        weights[prefix + "experts.down_proj"],
    ]
    for name, value in zip(("wi_0", "wi_1", "wo"), expected):
        actual = getattr(model.model.layers[0].experts, name).value
        np.testing.assert_array_equal(actual, value.transpose(0, 2, 1).astype(ml_dtypes.bfloat16))
        assert actual.sharding.mesh.shape["expert"] == 2
