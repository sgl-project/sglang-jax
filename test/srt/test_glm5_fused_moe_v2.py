from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.configs.model_config import MoEBackend, _assert_fused_moe_v2_supported
from sgl_jax.srt.layers.fused_moe import _pad_fused_moe_tokens_to_ep
from sgl_jax.srt.models.glm5_moe import (
    Glm5ForCausalLM,
    _requantize_blockwise_shared_weight,
)
from sgl_jax.srt.utils.quantization.quantization_utils import (
    dequantize_tensor,
    quantize_tensor,
)


def test_fused_v2_pads_small_decode_batch_to_ep_size():
    hidden_states = jnp.ones((8, 4), dtype=jnp.bfloat16)
    topk_weights = jnp.ones((8, 2), dtype=jnp.float32)
    topk_ids = jnp.zeros((8, 2), dtype=jnp.int32)

    padded_states, padded_weights, padded_ids, original_num_tokens = _pad_fused_moe_tokens_to_ep(
        hidden_states, topk_weights, topk_ids, ep_size=16
    )

    assert original_num_tokens == 8
    assert padded_states.shape == (16, 4)
    assert padded_weights.shape == (16, 2)
    assert padded_ids.shape == (16, 2)
    np.testing.assert_array_equal(padded_states[8:], 0)
    np.testing.assert_array_equal(padded_weights[8:], 0)
    np.testing.assert_array_equal(padded_ids[8:], -1)


def test_glm_moe_dsa_allows_fused_moe_v2():
    _assert_fused_moe_v2_supported(
        MoEBackend.FUSED_V2,
        ["GlmMoeDsaForCausalLM"],
    )


def test_glm5_fused_v2_shared_expert_mappings():
    model = SimpleNamespace(
        config=SimpleNamespace(
            n_routed_experts=2,
            n_shared_experts=1,
            moe_backend="fused_v2",
        )
    )

    mappings = Glm5ForCausalLM._create_moe_layer_mappings(
        model,
        layer_idx=3,
        target_idx=3,
        is_mlp_layer=False,
        is_static_quant=True,
        has_indexer=False,
    )

    prefix = "model.layers.3.mlp.shared_experts"
    target = "model.layers.3.mlp"
    assert mappings[f"{prefix}.gate_proj.weight"].target_path == f"{target}.w1_shared"
    assert mappings[f"{prefix}.up_proj.weight"].target_path == f"{target}.w3_shared"
    assert mappings[f"{prefix}.down_proj.weight"].target_path == f"{target}.w2_shared"
    assert (
        mappings[f"{prefix}.gate_proj.weight_scale_inv"].target_path
        == f"{target}.w1_shared_block_scale"
    )
    assert (
        mappings[f"{prefix}.down_proj.weight_scale_inv"].target_path
        == f"{target}.w2_shared_block_scale"
    )


def test_requantize_blockwise_shared_weight_to_per_channel():
    hf_weight = jnp.asarray(
        [
            [0.25, -0.5, 1.0, -2.0],
            [0.75, 1.5, -1.25, 0.5],
            [-0.125, 0.375, 2.5, -1.75],
            [1.125, -0.875, 0.625, 1.875],
        ],
        dtype=jnp.float32,
    )
    block_weight, block_scale = quantize_tensor(
        jnp.float8_e4m3fn,
        hf_weight,
        axis=(0, 1),
        block_size=(2, 2),
    )

    # The model mapping transposes HF [out, in] weights to [in, out], while the
    # block scale keeps its checkpoint [out_blocks, in_blocks] orientation.
    per_channel_weight, per_channel_scale = _requantize_blockwise_shared_weight(
        block_weight.T,
        block_scale,
        quantized_dtype=jnp.float8_e4m3fn,
    )

    expected = dequantize_tensor(
        block_weight,
        block_scale,
        axis=(0, 1),
        out_dtype=jnp.float32,
    ).T
    actual = per_channel_weight.astype(jnp.float32) * per_channel_scale[None, :]
    np.testing.assert_allclose(actual, expected, rtol=0.08, atol=0.02)
