"""CPU tests for the int4 per-channel routed-expert path (quantization math,
offline converter, quantization config, EPMoE placeholders, loader dtype)."""

import json
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from safetensors.numpy import save_file

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.model_loader.weights import JaxShardReader, LocalSource, WeightSpec
from sgl_jax.srt.model_loader.weights.recipes import TensorLayout
from sgl_jax.srt.utils.quantization.int4_experts import (
    dequantize_fp8_block,
    dequantize_int4_per_channel,
    match_routed_expert_weight,
    quantize_int4_per_channel,
    routed_expert_scale_suffix,
)
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor
from sgl_jax.tools import convert_moe_experts_int4 as converter

INT4_YAML = "fp8_linear_int4_moe_channel.yaml"


# --------------------------------------------------------------------------- math
def test_quantize_int4_per_channel_matches_runtime_quantize_tensor():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 256)).astype(np.float32) * rng.uniform(0.01, 2.0, (64, 1))
    w[3] = 0.0  # all-zero output channel
    w[5, 17] = 100.0  # outlier drives the channel scale
    codes, scale = quantize_int4_per_channel(w)
    assert codes.dtype == np.int8 and scale.dtype == np.float32
    assert codes.shape == (64, 256) and scale.shape == (64, 1)
    assert codes.min() >= -7 and codes.max() <= 7
    assert np.all(codes[3] == 0) and scale[3, 0] == 0.0

    ref_codes, ref_scale = quantize_tensor(jnp.int4, jnp.asarray(w), axis=1)
    np.testing.assert_array_equal(codes, np.asarray(ref_codes).astype(np.int8))
    # XLA folds ``x / 7`` into a reciprocal multiply: scales agree to 1 ulp.
    np.testing.assert_allclose(scale[:, 0], np.asarray(ref_scale), rtol=2e-7, atol=0)

    deq = dequantize_int4_per_channel(codes, scale)
    assert np.all(np.abs(deq - w) <= 0.5 * scale + 1e-6)


def test_dequantize_fp8_block_matches_naive_expansion():
    rng = np.random.default_rng(1)
    n, k, bn, bk = 200, 384, 128, 128  # narrow last out-block
    w8 = rng.standard_normal((n, k)).astype(ml_dtypes.float8_e4m3fn)
    s = rng.uniform(0.5, 2.0, (2, 3)).astype(np.float32)
    got = dequantize_fp8_block(w8, s, (bn, bk))
    ref = np.empty((n, k), np.float32)
    for i in range(n):
        for j in range(k):
            ref[i, j] = np.float32(w8[i, j]) * s[i // bn, j // bk]
    np.testing.assert_array_equal(got, ref)
    with pytest.raises(ValueError):
        dequantize_fp8_block(w8, s[:1], (bn, bk))


def test_match_routed_expert_weight():
    assert match_routed_expert_weight("model.layers.7.mlp.experts.31.up_proj.weight") == (
        "model.layers.7.mlp.experts.31.up_proj",
        7,
    )
    assert (
        match_routed_expert_weight("model.layers.7.mlp.experts.31.up_proj.weight_scale_inv") is None
    )
    assert match_routed_expert_weight("model.layers.7.mlp.shared_experts.up_proj.weight") is None
    assert match_routed_expert_weight("model.layers.7.mlp.gate.weight") is None


# ---------------------------------------------------------------------- converter
def _write_fp8_checkpoint(root, *, num_layers=3, first_dense=1, experts=2, n=256, k=128):
    rng = np.random.default_rng(2)
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    names = list(shards)
    weight_map = {}
    # layers 0..num_layers-1 are decoder layers, layer num_layers is an MTP draft layer
    for layer in range(num_layers + 1):
        shard = names[layer % 2]
        norm = f"model.layers.{layer}.input_layernorm.weight"
        shards[shard][norm] = np.ones((k,), ml_dtypes.bfloat16)
        weight_map[norm] = shard
        if layer < first_dense:
            continue
        for e in range(experts):
            for proj, (out, inp) in (
                ("gate_proj", (n, k)),
                ("up_proj", (n, k)),
                ("down_proj", (k, n)),
            ):
                prefix = f"model.layers.{layer}.mlp.experts.{e}.{proj}"
                w = (rng.standard_normal((out, inp)) * 8).astype(ml_dtypes.float8_e4m3fn)
                s = rng.uniform(0.001, 0.01, (out // 128, inp // 128)).astype(np.float32)
                shards[shard][prefix + ".weight"] = w
                shards[shard][prefix + ".weight_scale_inv"] = s
                weight_map[prefix + ".weight"] = shard
                weight_map[prefix + ".weight_scale_inv"] = shard
    total = 0
    for shard, tensors in shards.items():
        save_file(tensors, os.path.join(root, shard), metadata={"format": "pt"})
        total += sum(t.nbytes for t in tensors.values())
    with open(os.path.join(root, converter.INDEX_NAME), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f)
    config = {
        "num_hidden_layers": num_layers,
        "first_k_dense_replace": first_dense,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(root, "tokenizer.json"), "w") as f:
        f.write("{}")
    return shards


def test_converter_end_to_end(tmp_path):
    src = tmp_path / "fp8"
    dst = tmp_path / "int4"
    src.mkdir()
    shards = _write_fp8_checkpoint(str(src))

    common = ["--input-dir", str(src), "--output-dir", str(dst), "--workers", "2"]
    # Worker-style selection for the first shard, explicit file for the second.
    assert converter.main(common + ["--shard-index", "0", "--num-shards", "2"]) == 0
    assert converter.main(common + ["--files", "model-00002-of-00002.safetensors"]) == 0
    with pytest.raises(ValueError):
        converter.main(common + ["--files", "model-00009-of-00002.safetensors"])
    assert converter.main(["--input-dir", str(src), "--output-dir", str(dst), "--finalize"]) == 0

    with open(dst / converter.INDEX_NAME) as f:
        index = json.load(f)
    with open(dst / "config.json") as f:
        config = json.load(f)
    assert (dst / "tokenizer.json").exists()
    assert config["quantization_config"]["quant_method"] == "fp8"
    assert config["quantization_config"]["routed_experts"] == {
        "quant_method": "int4",
        "scheme": "per_channel_symmetric",
        "storage_dtype": "int8",
        "scale_key": "weight_scale",
        "layers": [1, 2],
    }

    seen = 0
    total = 0
    for shard, src_tensors in shards.items():
        with safe_open(str(dst / shard), framework="np") as f:
            assert f.metadata() == {"format": "pt"}
            keys = set(f.keys())
            for key, value in src_tensors.items():
                matched = match_routed_expert_weight(key)
                layer = int(key.split(".")[2])
                if matched is not None and 1 <= layer <= 2:
                    prefix = matched[0]
                    assert prefix + ".weight_scale_inv" not in keys
                    codes = f.get_tensor(prefix + ".weight")
                    scale = f.get_tensor(prefix + ".weight_scale")
                    assert codes.dtype == np.int8 and codes.shape == value.shape
                    assert scale.dtype == np.float32 and scale.shape == (value.shape[0], 1)
                    ref = dequantize_fp8_block(value, src_tensors[prefix + ".weight_scale_inv"])
                    exp_codes, exp_scale = quantize_int4_per_channel(ref)
                    np.testing.assert_array_equal(codes, exp_codes)
                    np.testing.assert_array_equal(scale, exp_scale)
                    assert np.all(
                        np.abs(dequantize_int4_per_channel(codes, scale) - ref)
                        <= 0.5 * scale + 1e-7
                    )
                    seen += 1
                elif key.endswith(".weight_scale_inv") and 1 <= layer <= 2 and ".experts." in key:
                    continue
                else:
                    # untouched: norms and the draft layer's FP8 experts
                    got = f.get_tensor(key)
                    assert got.dtype == value.dtype and got.shape == value.shape
                    np.testing.assert_array_equal(got.view(np.uint8), value.view(np.uint8))
            for key in keys:
                assert index["weight_map"][key] == shard
                total += f.get_tensor(key).nbytes
    assert seen == 2 * 2 * 3  # layers 1,2 x 2 experts x 3 projections
    assert index["metadata"]["total_size"] == total
    draft_keys = [k for k in index["weight_map"] if k.startswith("model.layers.3.mlp.experts.")]
    assert draft_keys and all(k.endswith((".weight", ".weight_scale_inv")) for k in draft_keys)


# ------------------------------------------------------------------------- config
def test_int4_yaml_and_moe_block_size_override(tmp_path):
    cfg = QuantizationConfig.from_yaml(INT4_YAML)
    assert cfg.is_static_checkpoint
    assert jnp.dtype(cfg.get_moe_weight_dtype()) == jnp.dtype(jnp.int4)
    assert jnp.dtype(cfg.get_moe_activation_dtype()) == jnp.dtype(jnp.float8_e4m3fn)
    assert cfg.weight_block_size == (128, 128)
    assert cfg.moe_weight_block_size_override and cfg.get_moe_weight_block_size() is None
    assert cfg.to_dict()["moe_weight_block_size"] is None
    assert routed_expert_scale_suffix(cfg) == ".weight_scale"

    # No MoE override: experts inherit the global block size (existing FP8 configs).
    fp8 = QuantizationConfig.from_yaml("fp8_deepseek_v3.yaml")
    assert not fp8.moe_weight_block_size_override
    assert fp8.get_moe_weight_block_size() == (128, 128)
    assert routed_expert_scale_suffix(fp8) == ".weight_scale_inv"
    assert routed_expert_scale_suffix(None) == ".weight_scale_inv"

    # Explicit MoE block size different from the linear one.
    path = tmp_path / "mixed.yaml"
    path.write_text(
        "quantization:\n  is_static_checkpoint: true\n  weight_block_size: [128, 128]\n"
        "  linear:\n    rules:\n      - {module_path: '.*', weight_dtype: 'float8_e4m3fn'}\n"
        "  moe:\n    weight_dtype: 'int8'\n    weight_block_size: [64, 64]\n"
    )
    mixed = QuantizationConfig.from_yaml(str(path))
    assert mixed.get_moe_weight_block_size() == (64, 64) and mixed.weight_block_size == (128, 128)


def _cpu_mesh():
    return Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def test_epmoe_int4_static_placeholders_are_per_channel():
    cfg = QuantizationConfig.from_yaml(INT4_YAML)
    mesh = _cpu_mesh()
    with jax.set_mesh(mesh):
        layer = EPMoE(
            hidden_size=256,
            num_experts=2,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=128,
            quantization_config=cfg,
        )
        assert layer.weight_block_size is None
        assert jnp.dtype(layer.quantized_dtype) == jnp.dtype(jnp.int4)
        assert jnp.dtype(layer.activation_quantized_dtype) == jnp.dtype(jnp.float8_e4m3fn)
        for name in ("wi_0", "wi_1", "wo"):
            param = getattr(layer, name)
            param.value = jax.ShapeDtypeStruct(
                param.value.shape, jnp.bfloat16, sharding=param.value.sharding
            )
        layer.quantize_weights(is_static=True)
    assert layer.wi_0.value.dtype == jnp.int4 and layer.wo.value.dtype == jnp.int4
    assert layer.wi_0_scale.value.shape == (2, 1, 1, 128)
    assert layer.wi_1_scale.value.shape == (2, 1, 1, 128)
    assert layer.wo_scale.value.shape == (2, 1, 1, 256)


# ------------------------------------------------------------------------- loader
def test_loader_stacks_int8_expert_codes_and_promotes_channel_scale(tmp_path):
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1),
        axis_names=("expert", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    keys = [f"model.layers.1.mlp.experts.{e}.gate_proj.weight" for e in range(2)]
    codes = {k: (np.arange(6, dtype=np.int8).reshape(2, 3) - 3 + i) for i, k in enumerate(keys)}
    save_file(codes, str(tmp_path / "model.safetensors"))
    spec = WeightSpec("model.layers.1.mlp.wi_0", sources=tuple(keys), transpose=True)
    with LocalSource(SimpleNamespace(model_path=str(tmp_path))) as source:
        result = JaxShardReader(mesh).read(
            source,
            "model.layers.1.mlp.wi_0",
            spec,
            NamedSharding(mesh, P("expert", None, "tensor")),
        )
    assert result.dtype == jnp.int8 and result.shape == (2, 3, 2)
    np.testing.assert_array_equal(np.asarray(result)[1], codes[keys[1]].T)
    # The assignment path casts the codes to the int4 placeholder dtype.
    as_int4 = result.astype(jnp.int4)
    np.testing.assert_array_equal(np.asarray(as_int4).astype(np.int8), np.asarray(result))

    # Stacked per-channel scale [E, out, 1] -> GMM layout [E, 1, 1, out].
    layout = TensorLayout(SimpleNamespace(quantization_config=None), mesh)
    scale = jnp.arange(2 * 4, dtype=jnp.float32).reshape(2, 4, 1)
    target = jax.ShapeDtypeStruct((2, 1, 1, 4), jnp.float32)
    out = layout._maybe_convert_epmoe_scale_for_kernel(
        scale, target, "model.layers.1.mlp.wi_0_scale"
    )
    assert out.shape == (2, 1, 1, 4)
    np.testing.assert_array_equal(np.asarray(out)[:, 0, 0, :], np.asarray(scale)[:, :, 0])


def test_transform_experts_unpacks_only_packed_4bit_codes():
    # One int8 code per element (convert_moe_experts_int4 output): no nibble unpacking,
    # only the transpose the int4 read path deferred. Half-width int8 (compressed-tensors
    # packing) is unpacked to the target shape.
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1),
        axis_names=("expert", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    layout = TensorLayout(SimpleNamespace(quantization_config=None), mesh)
    spec = WeightSpec("model.layers.1.mlp.wi_0", sources=("a", "b"), transpose=True)
    target = jax.ShapeDtypeStruct((2, 6, 4), jnp.int4)  # [E, K, N]
    codes = (np.arange(2 * 4 * 6, dtype=np.int8).reshape(2, 4, 6) % 15) - 7  # [E, N, K]
    out = layout.transform_experts(jnp.asarray(codes), spec, target)
    assert out.shape == (2, 6, 4) and out.dtype == jnp.int4
    np.testing.assert_array_equal(np.asarray(out).astype(np.int8), np.transpose(codes, (0, 2, 1)))

    packed = np.zeros((2, 4, 3), dtype=np.uint8)  # two nibbles per byte along K
    nib = (codes + 8).astype(np.uint8)
    packed[...] = nib[:, :, 0::2] | (nib[:, :, 1::2] << 4)
    out_p = layout.transform_experts(jnp.asarray(packed), spec, target)
    assert out_p.shape == (2, 6, 4)
    np.testing.assert_array_equal(np.asarray(out_p).astype(np.int8), np.transpose(codes, (0, 2, 1)))


def test_default_max_layer_includes_nextn_draft_layers():
    # #1810 review: the serving int4 YAML applies to every routed-expert layer, so the
    # converter's default range must cover the NextN (MTP) draft layers too.
    assert (
        converter.default_max_layer({"num_hidden_layers": 78, "num_nextn_predict_layers": 1}) == 79
    )
    assert converter.default_max_layer({"num_hidden_layers": 78}) == 78
    assert (
        converter.default_max_layer({"num_hidden_layers": 4, "num_nextn_predict_layers": None}) == 4
    )


def test_epmoe_scale_expansion_uses_moe_block_size(tmp_path):
    # #1810 review: global [128, 128] with a MoE override of [64, 64] stores the fused 2D
    # scale as [E, k_blocks, out/64]; expanding it with the linear block size (128) left it
    # compact and failed the load-time shape check.
    path = tmp_path / "mixed.yaml"
    path.write_text(
        "quantization:\n  is_static_checkpoint: true\n  weight_block_size: [128, 128]\n"
        "  linear:\n    rules:\n      - {module_path: '.*', weight_dtype: 'float8_e4m3fn'}\n"
        "  moe:\n    weight_dtype: 'int8'\n    weight_block_size: [64, 64]\n"
    )
    cfg = QuantizationConfig.from_yaml(str(path))
    assert cfg.get_moe_weight_block_size() == (64, 64) and cfg.weight_block_size == (128, 128)
    E, k_blocks, out_dim = 2, 3, 256
    target = jax.ShapeDtypeStruct((E, k_blocks, 1, out_dim), jnp.float32)
    compact = jnp.arange(E * k_blocks * (out_dim // 64), dtype=jnp.float32).reshape(
        E, k_blocks, out_dim // 64
    )
    loader = SimpleNamespace(model_config=SimpleNamespace(quantization_config=cfg))
    out = TensorLayout._maybe_convert_epmoe_scale_for_kernel(
        loader, compact, target, "model.layers.3.mlp.experts.w1_scale"
    )
    assert out.shape == target.shape
    np.testing.assert_array_equal(
        np.asarray(out)[:, :, 0, :], np.repeat(np.asarray(compact), 64, axis=2)
    )
