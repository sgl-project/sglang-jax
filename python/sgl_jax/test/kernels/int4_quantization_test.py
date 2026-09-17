import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.configs.quantization_config import DTYPE_MAP, QuantizationConfig
from sgl_jax.srt.utils.quantization.quantization_utils import (
    apply_linear_quantization,
)
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
    assert quant_cfg.moe_weight_dtype in (getattr(jnp, "int4", None), getattr(ml_dtypes, "int4", None))
    assert quant_cfg.ignored_layers == ["lm_head", "model.layers.0.mlp"]
    assert quant_cfg.weight_block_size == (32, 32)
    assert quant_cfg.linear_rules == []


def test_apply_linear_quantization_noop_when_no_rules():
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

    result = apply_linear_quantization(dummy_model_config, dummy_model)
    assert result is dummy_model
