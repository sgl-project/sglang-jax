# SPDX-License-Identifier: Apache-2.0
import importlib
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh

import sgl_jax.srt.kernels.quantized_matmul.kernel as quant_kernel
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.kernels.quantized_matmul.kernel import xla_quantized_matmul_local
from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.utils.quantization.quantization_utils import (
    apply_linear_quantization,
    quantize_tensor,
)

third_party_quant_util = importlib.import_module(
    "sgl_jax.srt.kernels.quantized_matmul.3rd_quantized_matmul.util"
)


def _create_single_device_mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("data", "tensor"))


def _make_linear_test_inputs():
    batch, in_dim, out_dim = 2, 256, 512
    compute_dtype = jnp.bfloat16
    key = jax.random.PRNGKey(42)
    k_x, k_w = jax.random.split(key)
    x = jax.random.normal(k_x, (batch, in_dim), dtype=compute_dtype)
    w_fp = jax.random.normal(k_w, (out_dim, in_dim), dtype=compute_dtype)
    return x, w_fp, compute_dtype


def _quantize_linear_weight(weight, weight_dtype, scale_format):
    weight_f32 = weight.astype(jnp.float32)

    if scale_format == "per_channel":
        weight_q, weight_scale = quantize_tensor(
            dtype=weight_dtype,
            tensor=weight_f32,
            axis=1,
        )
        return weight_q, weight_scale, None

    if scale_format == "block_channel":
        weight_q, weight_scale = quantize_tensor(
            dtype=weight_dtype,
            tensor=weight_f32,
            axis=1,
            block_size=128,
        )
        return weight_q, weight_scale, (1, 128)

    if scale_format == "block_quant":
        weight_q, weight_scale = quantize_tensor(
            dtype=weight_dtype,
            tensor=weight_f32,
            axis=(0, 1),
            block_size=(128, 128),
        )
        return weight_q, weight_scale, (128, 128)

    raise ValueError(f"Unsupported scale_format={scale_format}")


def _assert_close(name, out, ref_out):
    diff = jnp.abs(out - ref_out)
    mae = jnp.mean(diff)
    max_diff = jnp.max(diff)
    rel_error = mae / jnp.mean(jnp.abs(ref_out))

    print(f"\n>>> {name} <<<")
    print(f"  MAE (Mean Absolute Error): {mae.item():.6f}")
    print(f"  Max Absolute Difference:  {max_diff.item():.6f}")
    print(f"  Relative Error:           {rel_error.item():.6%}")

    assert rel_error.item() < 0.05, f"Relative error too high: {rel_error.item():.6%}"


@pytest.mark.parametrize(
    "scale_format",
    ["per_channel", "block_channel", "block_quant"],
    ids=["per-channel", "block-channel", "block-quant"],
)
def test_quantized_linear_offline_scale_formats(scale_format):
    x, w_fp, compute_dtype = _make_linear_test_inputs()
    mesh = _create_single_device_mesh()
    weight_q, weight_scale, weight_block_size = _quantize_linear_weight(
        w_fp,
        jnp.int8,
        scale_format,
    )

    quant_linear = QuantizedLinear(
        weight_q=weight_q,
        weight_scale=weight_scale,
        bias=None,
        activation_dtype=None,
        mesh=mesh,
        kernel_axes=(None, None),
        params_dtype=compute_dtype,
        compute_dtype=compute_dtype,
        weight_block_size=weight_block_size,
    )

    ref_out = jnp.dot(x, w_fp.T)
    out, bias = quant_linear(x)
    assert bias is None

    _assert_close(f"Offline QuantizedLinear ({scale_format})", out, ref_out)


def _run_block_quant_kernel_test(weight_dtype, dtype_name):
    x, w_fp, compute_dtype = _make_linear_test_inputs()
    weight_q, weight_scale, weight_block_size = _quantize_linear_weight(
        w_fp,
        weight_dtype,
        "block_quant",
    )

    ref_out = jnp.dot(x, w_fp.T)
    out = xla_quantized_matmul_local(
        x=x,
        w_q=weight_q,
        w_scale=weight_scale,
        quantize_activation=False,
        compute_dtype=compute_dtype,
        weight_block_size=weight_block_size,
    )

    _assert_close(f"Block Quant Kernel ({dtype_name})", out, ref_out)


def test_xla_quantized_matmul_block_quant_all():
    _run_block_quant_kernel_test(jnp.int8, "INT8")

    if hasattr(jnp, "float8_e4m3fn"):
        _run_block_quant_kernel_test(jnp.float8_e4m3fn, "FP8_E4M3")


def _assert_blockwise_tuning_fallback_uses_compatible_seed():
    key_cls = namedtuple(
        "FakeTunedKey",
        ["tpu_version", "n_batch", "n_out", "n_in", "x_q_dtype", "w_q_dtype"],
    )
    tuned_value_cls = namedtuple(
        "FakeTunedValue",
        ["batch_block_size", "out_block_size", "in_block_size", "n_lane_multiplier"],
    )

    fake_tuned_table = {
        key_cls(7, 1, 256, 4096, "int8", "int8"): tuned_value_cls(1, 4096, 4096, 1),
        key_cls(6, 16, 1024, 4096, "int8", "int8"): tuned_value_cls(16, 1024, 2048, 1),
        key_cls(6, 16, 4096, 4096, "int8", "int8"): tuned_value_cls(16, 1024, 4096, 1),
    }

    saved_state = {
        "_TRIED_LOADING_BLOCKWISE_3RD_TUNING": quant_kernel._TRIED_LOADING_BLOCKWISE_3RD_TUNING,
        "_BLOCKWISE_3RD_TUNED_VALUE_CLS": quant_kernel._BLOCKWISE_3RD_TUNED_VALUE_CLS,
        "_BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES": quant_kernel._BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES,
        "_BLOCKWISE_3RD_TUNED_BLOCK_SIZES": quant_kernel._BLOCKWISE_3RD_TUNED_BLOCK_SIZES,
        "_get_current_tpu_version": quant_kernel._get_current_tpu_version,
    }
    try:
        quant_kernel._TRIED_LOADING_BLOCKWISE_3RD_TUNING = True
        quant_kernel._BLOCKWISE_3RD_TUNED_VALUE_CLS = tuned_value_cls
        quant_kernel._BLOCKWISE_3RD_GET_TUNED_BLOCK_SIZES = None
        quant_kernel._BLOCKWISE_3RD_TUNED_BLOCK_SIZES = fake_tuned_table
        quant_kernel._get_current_tpu_version = lambda: 6

        tuned = quant_kernel._get_safe_blockwise_tuned_value(
            n_batch=1,
            n_out=256,
            n_in=4096,
            x_q_dtype=jnp.bfloat16,
            w_q_dtype=jnp.int8,
            block_size_in=128,
        )
    finally:
        for name, value in saved_state.items():
            setattr(quant_kernel, name, value)

    assert tuned.batch_block_size == 1
    assert tuned.out_block_size == 256
    assert tuned.in_block_size in (2048, 4096)
    assert tuned.in_block_size != 128


def test_blockwise_tuning_fallback_uses_compatible_seed(monkeypatch):
    del monkeypatch
    _assert_blockwise_tuning_fallback_uses_compatible_seed()


def test_linear_rule_weight_block_size_override():
    class DummyModel(nnx.Module):
        def __init__(self, mesh):
            self.proj = LinearBase(
                input_size=256,
                output_size=512,
                use_bias=False,
                mesh=mesh,
                kernel_axes=(None, None),
                params_dtype=jnp.bfloat16,
                scope_name="proj",
            )

    mesh = _create_single_device_mesh()
    with jax.set_mesh(mesh):
        model = DummyModel(mesh)

    class FakeModelConfig:
        pass

    model_config = FakeModelConfig()
    model_config.quantization_config = type(
        "FakeQuantConfig",
        (),
        {
            "get_linear_rules": staticmethod(
                lambda: [
                    {
                        "module_path": ".*",
                        "weight_dtype": "int8",
                        "activation_dtype": None,
                        "weight_block_size": None,
                    }
                ]
            ),
            "ignored_layers": None,
            "weight_block_size": [128, 128],
        },
    )()

    apply_linear_quantization(model_config, model, is_static_input=False)

    assert isinstance(model.proj, QuantizedLinear)
    assert model.proj.weight_block_size is None
    assert model.proj.weight_scale.value.ndim == 1


def test_linear_return_contract_with_bias():
    mesh = _create_single_device_mesh()
    x = jnp.ones((2, 64), dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        linear = LinearBase(
            input_size=64,
            output_size=32,
            use_bias=True,
            mesh=mesh,
            kernel_axes=(None, None),
            params_dtype=jnp.bfloat16,
            scope_name="biased_proj",
        )
        out, bias = linear(x)

    assert out.shape == (2, 32)
    assert bias is None

    with jax.set_mesh(mesh):
        quant_linear = QuantizedLinear.from_linear(
            linear,
            weight_dtype=jnp.int8,
            activation_dtype=None,
            is_static_input=False,
        )
        q_out, q_bias = quant_linear(x)

    assert q_out.shape == (2, 32)
    assert q_bias is None


def test_linear_skip_bias_add_returns_param_bias():
    mesh = _create_single_device_mesh()
    x = jnp.ones((2, 64), dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        linear = LinearBase(
            input_size=64,
            output_size=32,
            use_bias=True,
            skip_bias_add=True,
            mesh=mesh,
            kernel_axes=(None, None),
            params_dtype=jnp.bfloat16,
            scope_name="biased_skip_proj",
        )
        out, bias = linear(x)

    assert out.shape == (2, 32)
    assert isinstance(bias, nnx.Param)

    with jax.set_mesh(mesh):
        quant_linear = QuantizedLinear.from_linear(
            linear,
            weight_dtype=jnp.int8,
            activation_dtype=None,
            is_static_input=False,
        )
        q_out, q_bias = quant_linear(x)

    assert q_out.shape == (2, 32)
    assert isinstance(q_bias, nnx.Param)


def test_linear_accepts_legacy_positional_mesh_argument():
    mesh = _create_single_device_mesh()
    x = jnp.ones((2, 64), dtype=jnp.bfloat16)

    with jax.set_mesh(mesh):
        linear = LinearBase(
            64,
            32,
            mesh,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=jnp.bfloat16,
            scope_name="legacy_mesh_positional",
        )
        out, bias = linear(x)

    assert out.shape == (2, 32)
    assert bias is None


def test_static_linear_rejects_non_prequantized_concrete_weights():
    mesh = _create_single_device_mesh()

    with jax.set_mesh(mesh):
        linear = LinearBase(
            input_size=64,
            output_size=32,
            use_bias=False,
            mesh=mesh,
            kernel_axes=(None, None),
            params_dtype=jnp.bfloat16,
            scope_name="static_bad_input",
        )

    with pytest.raises(ValueError, match="pre-quantized concrete weights or abstract shapes"):
        QuantizedLinear.from_linear(
            linear,
            weight_dtype=jnp.int8,
            activation_dtype=None,
            is_static_input=True,
        )


def test_validate_inputs_rejects_bad_2d_block_scale():
    x = jnp.ones((4, 256), dtype=jnp.bfloat16)
    w_q = jnp.ones((128, 256), dtype=jnp.int8)
    bad_scale = jnp.ones((2, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="w_q.shape\\[0\\].*w_scale.shape\\[1\\]"):
        third_party_quant_util.validate_inputs(
            x=x,
            w_q=w_q,
            w_scale=bad_scale,
            x_abs_max=None,
            x_q_dtype=jnp.int8,
            batch_block_size=4,
            out_block_size=128,
            in_block_size=256,
        )


@pytest.mark.parametrize(
    "weight_block_size",
    ([128], "foo", [-1, 128]),
    ids=["wrong-len", "wrong-type", "negative"],
)
def test_quantization_config_rejects_invalid_weight_block_size(tmp_path, weight_block_size):
    config_path = tmp_path / "invalid_quant.yaml"
    config_path.write_text(
        f"""
quantization:
  weight_block_size: {weight_block_size!r}
  linear:
    rules:
      - module_path: '.*'
        weight_dtype: 'int8'
        activation_dtype: null
  moe:
    weight_dtype: 'int8'
    activation_dtype: null
""".strip()
    )

    with pytest.raises(ValueError, match="weight_block_size"):
        QuantizationConfig.from_yaml(str(config_path))


def test_ignored_layers_only_skips_requested_paths():
    class SelfAttn(nnx.Module):
        def __init__(self, mesh):
            self.q_proj = LinearBase(
                input_size=64,
                output_size=32,
                use_bias=False,
                mesh=mesh,
                kernel_axes=(None, None),
                params_dtype=jnp.bfloat16,
                scope_name="q_proj",
            )
            self.o_proj = LinearBase(
                input_size=64,
                output_size=32,
                use_bias=False,
                mesh=mesh,
                kernel_axes=(None, None),
                params_dtype=jnp.bfloat16,
                scope_name="o_proj",
            )

    class DummyBlock(nnx.Module):
        def __init__(self, mesh):
            self.self_attn = SelfAttn(mesh)

    class FakeModelConfig:
        pass

    def _make_config(ignored_layers):
        model_config = FakeModelConfig()
        model_config.quantization_config = type(
            "FakeQuantConfig",
            (),
            {
                "get_linear_rules": staticmethod(
                    lambda: [
                        {
                            "module_path": ".*",
                            "weight_dtype": "int8",
                            "activation_dtype": None,
                            "weight_block_size": None,
                        }
                    ]
                ),
                "ignored_layers": ignored_layers,
                "weight_block_size": [128, 128],
            },
        )()
        return model_config

    mesh = _create_single_device_mesh()
    with jax.set_mesh(mesh):
        model = DummyBlock(mesh)
    apply_linear_quantization(_make_config(["some_other_layer"]), model, is_static_input=False)
    assert isinstance(model.self_attn.q_proj, QuantizedLinear)
    assert isinstance(model.self_attn.o_proj, QuantizedLinear)

    with jax.set_mesh(mesh):
        model = DummyBlock(mesh)
    apply_linear_quantization(_make_config(["self_attn.o_proj"]), model, is_static_input=False)
    assert isinstance(model.self_attn.q_proj, QuantizedLinear)
    assert isinstance(model.self_attn.o_proj, LinearBase)


def test_ignored_layers_exact_match_does_not_overmatch():
    class SelfAttn(nnx.Module):
        def __init__(self, mesh):
            self.q_proj = LinearBase(
                input_size=64,
                output_size=32,
                use_bias=False,
                mesh=mesh,
                kernel_axes=(None, None),
                params_dtype=jnp.bfloat16,
                scope_name="q_proj",
            )
            self.o_proj = LinearBase(
                input_size=64,
                output_size=32,
                use_bias=False,
                mesh=mesh,
                kernel_axes=(None, None),
                params_dtype=jnp.bfloat16,
                scope_name="o_proj",
            )

    class DummyBlock(nnx.Module):
        def __init__(self, mesh):
            self.self_attn = SelfAttn(mesh)

    class FakeModelConfig:
        pass

    model_config = FakeModelConfig()
    model_config.quantization_config = type(
        "FakeQuantConfig",
        (),
        {
            "get_linear_rules": staticmethod(
                lambda: [
                    {
                        "module_path": ".*",
                        "weight_dtype": "int8",
                        "activation_dtype": None,
                        "weight_block_size": None,
                    }
                ]
            ),
            "ignored_layers": ["proj"],
            "weight_block_size": [128, 128],
        },
    )()

    mesh = _create_single_device_mesh()
    with jax.set_mesh(mesh):
        model = DummyBlock(mesh)
    apply_linear_quantization(model_config, model, is_static_input=False)
    assert isinstance(model.self_attn.q_proj, QuantizedLinear)
    assert isinstance(model.self_attn.o_proj, QuantizedLinear)


# ---------------------------------------------------------------------------
# Non-128-aligned output dimension tests (e.g., MiMo k_proj head_dim=192)
# ---------------------------------------------------------------------------


def _make_non_aligned_block_quant_inputs(out_dim, in_dim, weight_dtype):
    """Create inputs simulating a static checkpoint with non-aligned out_dim.

    Returns (x, w_fp, weight_q, weight_scale) where weight_q has shape
    [out_dim, in_dim] (NOT padded to block boundary) and weight_scale has
    shape [ceil(out_dim/128), ceil(in_dim/128)].
    """
    batch = 2
    compute_dtype = jnp.bfloat16
    key = jax.random.PRNGKey(42)
    k_x, k_w = jax.random.split(key)
    x = jax.random.normal(k_x, (batch, in_dim), dtype=compute_dtype)
    w_fp = jax.random.normal(k_w, (out_dim, in_dim), dtype=compute_dtype)

    # quantize_tensor with pad_tensor=True pads to block boundary,
    # then slice back to original out_dim to simulate static checkpoint layout
    w_q_padded, w_scale = quantize_tensor(
        dtype=weight_dtype,
        tensor=w_fp.astype(jnp.float32),
        axis=(0, 1),
        block_size=(128, 128),
        pad_tensor=True,
    )
    w_q = w_q_padded[:out_dim, :in_dim]
    return x, w_fp, w_q, w_scale, compute_dtype


def test_quantized_linear_block_quant_non_aligned_192():
    """QuantizedLinear with out_dim=192, block_size=(128,128) — partial second block."""
    mesh = _create_single_device_mesh()
    out_dim, in_dim = 192, 256
    x, w_fp, w_q, w_scale, compute_dtype = _make_non_aligned_block_quant_inputs(
        out_dim, in_dim, jnp.int8
    )

    # scale should have 2 output blocks: ceil(192/128) = 2
    assert w_scale.shape[0] == 2, f"Expected 2 output scale blocks, got {w_scale.shape[0]}"

    quant_linear = QuantizedLinear(
        weight_q=w_q,
        weight_scale=w_scale,
        bias=None,
        activation_dtype=None,
        mesh=mesh,
        kernel_axes=(None, None),
        params_dtype=compute_dtype,
        compute_dtype=compute_dtype,
        weight_block_size=(128, 128),
    )
    ref_out = jnp.dot(x, w_fp.T)
    out, bias = quant_linear(x)
    assert bias is None
    _assert_close("QuantizedLinear non-aligned 192 (INT8)", out, ref_out)


def test_xla_quantized_matmul_block_quant_non_aligned_192():
    """Kernel-level block quant with out_dim=192 — partial second block."""
    out_dim, in_dim = 192, 256
    x, w_fp, w_q, w_scale, compute_dtype = _make_non_aligned_block_quant_inputs(
        out_dim, in_dim, jnp.int8
    )
    ref_out = jnp.dot(x, w_fp.T)
    out = xla_quantized_matmul_local(
        x=x,
        w_q=w_q,
        w_scale=w_scale,
        quantize_activation=False,
        compute_dtype=compute_dtype,
        weight_block_size=(128, 128),
    )
    _assert_close("Kernel non-aligned 192 (INT8)", out, ref_out)

    if hasattr(jnp, "float8_e4m3fn"):
        x_fp8, w_fp_fp8, w_q_fp8, w_scale_fp8, _ = _make_non_aligned_block_quant_inputs(
            out_dim, in_dim, jnp.float8_e4m3fn
        )
        ref_out_fp8 = jnp.dot(x_fp8, w_fp_fp8.T)
        out_fp8 = xla_quantized_matmul_local(
            x=x_fp8,
            w_q=w_q_fp8,
            w_scale=w_scale_fp8,
            quantize_activation=False,
            compute_dtype=compute_dtype,
            weight_block_size=(128, 128),
        )
        _assert_close("Kernel non-aligned 192 (FP8_E4M3)", out_fp8, ref_out_fp8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
