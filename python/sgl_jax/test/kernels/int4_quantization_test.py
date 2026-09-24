import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.configs.quantization_config import DTYPE_MAP, QuantizationConfig
from sgl_jax.srt.utils.quantization.quantization_utils import apply_linear_quantization
from sgl_jax.srt.utils.weight_utils import unpack_4bit_jax


def test_dtype_map_contains_int4():
    assert "int4" in DTYPE_MAP
    assert "uint4" in DTYPE_MAP


def test_unpack_4bit_jax_int32():
    # 8 values packed into one int32: [0, 1, 2, 3, 4, 5, 6, 7]
    # packed value = sum(v << (4*i))
    packed_val = sum(i << (4 * i) for i in range(8))
    packed_arr = jnp.array([[packed_val]], dtype=jnp.int32)  # shape (1, 1)

    int4_dtype = getattr(jnp, "int4", getattr(ml_dtypes, "int4", jnp.int8))
    unpacked = unpack_4bit_jax(packed_arr, int4_dtype)

    assert unpacked.shape == (1, 8)
    # The unpack subtracts offset 8: [0-8, 1-8, 2-8, ..., 7-8] = [-8, -7, -6, ..., -1]
    expected = np.array([[-8, -7, -6, -5, -4, -3, -2, -1]])
    np.testing.assert_array_equal(np.array(unpacked, dtype=np.int8), expected)


def test_unpack_4bit_jax_transpose():
    packed_val = sum(i << (4 * i) for i in range(8))
    packed_arr = jnp.array([[[packed_val]]], dtype=jnp.int32)  # shape (1, 1, 1)

    int4_dtype = getattr(jnp, "int4", getattr(ml_dtypes, "int4", jnp.int8))
    unpacked = unpack_4bit_jax(packed_arr, int4_dtype, do_transpose=True)

    # Original unpacked shape would be (1, 1, 8), transposed (0, 2, 1) -> (1, 8, 1)
    assert unpacked.shape == (1, 8, 1)


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
    assert quant_cfg.moe_weight_dtype in (
        getattr(jnp, "int4", None),
        getattr(ml_dtypes, "int4", None),
    )
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


def test_unpack_multiple_words_preserves_values_and_transpose():
    # All signed nibble values, multiple rows/experts, and packed words with
    # the int32 sign bit set. Compare values as well as the resulting layout.
    expected = np.arange(2 * 3 * 16, dtype=np.int32).reshape(2, 3, 16) % 16 - 8
    nibbles = (expected + 8).astype(np.uint32).reshape(2, 3, 2, 8)
    packed = np.bitwise_or.reduce(nibbles << (4 * np.arange(8, dtype=np.uint32)), axis=-1)
    result = unpack_4bit_jax(jnp.asarray(packed.view(np.int32)), jnp.int4, do_transpose=True)
    np.testing.assert_array_equal(np.asarray(result, dtype=np.int8), expected.transpose(0, 2, 1))


def test_apply_linear_quantization_raises_when_no_rules():
    import pytest
    from flax import nnx

    class DummyModel(nnx.Module):
        def __init__(self):
            pass

    dummy_model = DummyModel()
    dummy_model_config = ModelConfig.__new__(ModelConfig)
    dummy_model_config.quantization_config = QuantizationConfig(
        is_static_checkpoint=True,
        linear_rules=[],
    )

    assert dummy_model_config.quantization_config.has_linear_quantization() is False
    with pytest.raises(ValueError, match="No linear rules found"):
        apply_linear_quantization(dummy_model_config, dummy_model)


def test_epmoe_static_scale_dtype_int4_vs_fp8():
    import jax
    from jax.sharding import AxisType, Mesh

    from sgl_jax.srt.layers.moe import EPMoE

    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    int4_cfg = QuantizationConfig(
        is_static_checkpoint=True,
        moe_weight_dtype=getattr(jnp, "int4", jnp.int8),
        weight_block_size=(32, 32),
    )
    moe_int4 = EPMoE(
        hidden_size=64,
        num_experts=2,
        num_experts_per_tok=1,
        ep_size=1,
        mesh=mesh,
        intermediate_dim=64,
        dtype=jnp.bfloat16,
        quantization_config=int4_cfg,
    )
    moe_int4.quantize_weights(is_static=True)
    assert moe_int4.wi_0_scale.value.dtype == jnp.bfloat16
    assert moe_int4.wo_scale.value.dtype == jnp.bfloat16

    fp8_cfg = QuantizationConfig(
        is_static_checkpoint=True,
        moe_weight_dtype=jnp.float8_e4m3fn,
        weight_block_size=(32, 32),
    )
    moe_fp8 = EPMoE(
        hidden_size=64,
        num_experts=2,
        num_experts_per_tok=1,
        ep_size=1,
        mesh=mesh,
        intermediate_dim=64,
        dtype=jnp.bfloat16,
        quantization_config=fp8_cfg,
    )
    # Match the parity tool: checkpoint weights are present before scale prep.
    for name in ("wi_0", "wi_1", "wo"):
        param = getattr(moe_fp8, name)
        param.value = jnp.ones_like(param.value, dtype=jnp.float8_e4m3fn)
    checkpoint_weights = {name: getattr(moe_fp8, name).value for name in ("wi_0", "wi_1", "wo")}
    moe_fp8.quantize_weights(is_static=True)
    assert moe_fp8.wi_0_scale.value.dtype == jnp.float32
    assert moe_fp8.wo_scale.value.dtype == jnp.float32
    for name, weight in checkpoint_weights.items():
        assert getattr(moe_fp8, name).value is weight

    from flax import nnx

    abstract_moe = nnx.eval_shape(
        lambda: EPMoE(
            hidden_size=64,
            num_experts=2,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=64,
            quantization_config=int4_cfg,
        )
    )
    abstract_moe.quantize_weights(is_static=True)
    for name in ("wi_0", "wi_1", "wo"):
        weight = getattr(abstract_moe, name).value
        assert isinstance(weight, jax.ShapeDtypeStruct)
        assert weight.dtype == jnp.int4


def test_fused_static_int4_rejected_before_scale_allocation():
    from types import SimpleNamespace

    from sgl_jax.srt.layers.fused_moe import FusedEPMoE

    # Only dtype should be read before rejecting the unsupported combination.
    with pytest.raises(ValueError, match="require moe_backend='epmoe'"):
        FusedEPMoE.quantize_weights(SimpleNamespace(quantized_dtype=jnp.int4), is_static=True)
