import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.utils.quantization.quantization_utils import apply_linear_quantization
from sgl_jax.srt.utils.weight_utils import unpack_4bit_jax


def test_model_config_pack_quantized_parsing():
    hf_quant_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 32,
                }
            }
        },
        "ignore": ["lm_head", "model.layers.0.mlp"],
    }

    dummy_config = ModelConfig.__new__(ModelConfig)
    dummy_config.quantization_config = None
    dummy_config._get_hf_quant_config = lambda: hf_quant_config
    quant_cfg = dummy_config._resolve_quantization_config()

    assert quant_cfg is not None
    assert quant_cfg.is_static_checkpoint is True
    assert quant_cfg.moe_weight_dtype == jnp.int4
    assert quant_cfg.ignored_layers == ["lm_head", "model.layers.0.mlp"]
    assert quant_cfg.weight_block_size == (32, 32)
    assert quant_cfg.linear_rules == []
    assert quant_cfg.has_linear_quantization() is False
    assert quant_cfg.has_moe_quantization() is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_bits": 8},
        {"type": "float"},
        {"symmetric": False},
        {"strategy": "channel"},
        {"dynamic": True},
        {"actorder": "group"},
        {"group_size": -1},
        {"group_size": 0},
    ],
)
def test_pack_quantized_rejects_unsupported_weight_schemes(overrides):
    weights = dict(num_bits=4, type="int", symmetric=True, strategy="group", group_size=32)
    weights.update(overrides)
    config = ModelConfig.__new__(ModelConfig)
    config.quantization_config = None
    config._get_hf_quant_config = lambda: {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {"group_0": {"weights": weights}},
    }
    with pytest.raises(ValueError, match="pack-quantized"):
        config._resolve_quantization_config()


def test_pack_quantized_validates_every_group():
    weights = dict(num_bits=4, type="int", symmetric=True, strategy="group", group_size=32)
    config = ModelConfig.__new__(ModelConfig)
    config.quantization_config = None
    hf_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "first": {"weights": weights},
            "second": {"weights": {**weights, "group_size": 64}},
        },
    }
    config._get_hf_quant_config = lambda: hf_config
    with pytest.raises(ValueError, match="same group_size"):
        config._resolve_quantization_config()
    hf_config["config_groups"]["second"] = {
        "weights": weights,
        "input_activations": {"num_bits": 8},
    }
    with pytest.raises(ValueError, match="activation quantization"):
        config._resolve_quantization_config()


@pytest.mark.parametrize("transpose", [False, True])
def test_unpack_multiple_words_preserves_values_and_transpose(transpose):
    # All signed nibble values, multiple rows/experts, and packed words with
    # the int32 sign bit set. Compare values as well as the resulting layout.
    expected = np.arange(2 * 3 * 16, dtype=np.int32).reshape(2, 3, 16) % 16 - 8
    nibbles = (expected + 8).astype(np.uint32).reshape(2, 3, 2, 8)
    packed = np.bitwise_or.reduce(nibbles << (4 * np.arange(8, dtype=np.uint32)), axis=-1)
    result = unpack_4bit_jax(jnp.asarray(packed.view(np.int32)), jnp.int4, do_transpose=transpose)
    if transpose:
        expected = expected.transpose(0, 2, 1)
    np.testing.assert_array_equal(np.asarray(result, dtype=np.int8), expected)


def test_apply_linear_quantization_raises_when_no_rules():
    from types import SimpleNamespace

    from flax import nnx

    config = SimpleNamespace(quantization_config=QuantizationConfig(linear_rules=[]))
    with pytest.raises(ValueError, match="No linear rules found"):
        apply_linear_quantization(config, nnx.Module())


@pytest.mark.parametrize("dtype", [jnp.int4, jnp.float8_e4m3fn])
def test_epmoe_static_parameters(dtype):
    import jax
    from flax import nnx
    from jax.sharding import AxisType, Mesh

    from sgl_jax.srt.layers.moe import EPMoE

    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    config = QuantizationConfig(
        is_static_checkpoint=True, moe_weight_dtype=dtype, weight_block_size=(32, 32)
    )
    model = nnx.eval_shape(
        lambda: EPMoE(64, 2, 1, 1, mesh, intermediate_dim=64, quantization_config=config)
    )
    model.quantize_weights(is_static=True, abstract=True)
    expected_scale_dtype = jnp.bfloat16 if dtype == jnp.int4 else jnp.float32
    for name in ("wi_0", "wi_1", "wo"):
        weight = getattr(model, name).value
        assert isinstance(weight, jax.ShapeDtypeStruct)
        assert weight.dtype == dtype
        assert getattr(model, name + "_scale").value.dtype == expected_scale_dtype

    # The FP8 parity tool loads checkpoint weights before preparing scales.
    if dtype == jnp.float8_e4m3fn:
        loaded = {}
        for name in ("wi_0", "wi_1", "wo"):
            param = getattr(model, name)
            loaded[name] = jax.device_put(
                np.ones(param.value.shape, dtype=dtype), param.value.sharding
            )
            param.value = loaded[name]
        model.quantize_weights(is_static=True)
        for name, value in loaded.items():
            assert getattr(model, name).value is value


def test_fused_static_int4_rejected_before_scale_allocation():
    from types import SimpleNamespace

    from sgl_jax.srt.layers.fused_moe import FusedEPMoE

    # Only dtype should be read before rejecting the unsupported combination.
    with pytest.raises(ValueError, match="require moe_backend='epmoe'"):
        FusedEPMoE.quantize_weights(SimpleNamespace(quantized_dtype=jnp.int4), is_static=True)
