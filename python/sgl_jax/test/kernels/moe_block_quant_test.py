# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.models.mimo_v2_flash import create_moe_weights_mapping_quantized


@pytest.mark.parametrize(
    ("weight_block_size", "expected_k_blocks_wi", "expected_k_blocks_wo"),
    [
        (None, 1, 1),
        ([1, 128], 4, 8),
        ([128, 128], 4, 8),
    ],
    ids=["per-channel", "block-channel", "block-quant"],
)
def test_epmoe_block_quant_logic(weight_block_size, expected_k_blocks_wi, expected_k_blocks_wo):
    """
    Test the block quantization logic of EPMoE on CPU.
    Focuses on weight/scale shapes and placeholder generation.
    """
    print("\n>>> Testing EP MoE Block Quantization Logic (CPU) <<<")

    # 1. Setup a minimal mesh
    devices = jax.devices()
    # Use names that are safe for CPU/Standard JAX
    mesh = Mesh(np.array(devices[:1]).reshape(1, 1), axis_names=("data", "tensor"))

    # 2. Configuration
    hidden_size = 512
    intermediate_dim = 1024
    num_experts = 4
    num_experts_per_tok = 1

    # Mock QuantizationConfig
    class MockQuantConfig:
        def get_moe_weight_dtype(self):
            return jnp.int8

        def get_moe_activation_dtype(self):
            return None

        @property
        def weight_block_size(self):
            return weight_block_size

    quant_config = MockQuantConfig()

    # 3. Initialize EPMoE with Mocked Mesh to bypass sharding checks on CPU
    # We monkeypatch the sharding in EPMoE to be CPU-friendly for this test
    import sgl_jax.srt.layers.moe as moe_module

    with mock.patch.object(moe_module, "P", lambda *args: None):
        moe = EPMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=intermediate_dim,
            quantization_config=quant_config,
        )

    # 4. Run Quantization Prep
    moe.quantize_weights(is_static=True)

    # 5. Assert Scale Shapes
    # EPMoE static placeholders only depend on block_size_k. block_size_out is
    # consumed later when compact offline block scales are expanded for GMM.
    k_blocks_wi = expected_k_blocks_wi
    k_blocks_wo = expected_k_blocks_wo

    print(f"  Expert Count: {num_experts}")
    print(f"  K Blocks (WI): {k_blocks_wi}")
    print(f"  K Blocks (WO): {k_blocks_wo}")

    expected_wi_shape = (num_experts, k_blocks_wi, 1, intermediate_dim)
    expected_wo_shape = (num_experts, k_blocks_wo, 1, hidden_size)

    print(f"  WI_0 Scale Shape: {moe.wi_0_scale.value.shape}")
    print(f"  WO Scale Shape:   {moe.wo_scale.value.shape}")

    assert (
        moe.wi_0_scale.value.shape == expected_wi_shape
    ), f"WI shape mismatch: {moe.wi_0_scale.value.shape} vs {expected_wi_shape}"
    assert (
        moe.wo_scale.value.shape == expected_wo_shape
    ), f"WO shape mismatch: {moe.wo_scale.value.shape} vs {expected_wo_shape}"

    print("  Shape Verification: PASSED")

    # 6. Verify Content (Should be zeros as initialized in is_static=True)
    assert jnp.all(moe.wi_0_scale.value == 0)
    print("  Content Verification: PASSED")


def _make_epmoe_for_scale_tests(weight_block_size):
    devices = jax.devices()
    mesh = Mesh(np.array(devices[:1]).reshape(1, 1), axis_names=("data", "tensor"))

    class MockQuantConfig:
        def get_moe_weight_dtype(self):
            return jnp.int8

        def get_moe_activation_dtype(self):
            return None

        @property
        def weight_block_size(self):
            return weight_block_size

    import sgl_jax.srt.layers.moe as moe_module

    with mock.patch.object(moe_module, "P", lambda *args: None):
        return EPMoE(
            hidden_size=512,
            num_experts=2,
            num_experts_per_tok=1,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=1024,
            quantization_config=MockQuantConfig(),
        )


def test_epmoe_rejects_invalid_4d_scale_layout():
    moe = _make_epmoe_for_scale_tests([128, 128])
    # Weight layout is [E, k, n]. For wi_0: k=hidden_size=512, n=intermediate_dim=1024
    invalid_scale = jnp.ones((2, 4, 2, 1024), dtype=jnp.float32)
    weight = jnp.zeros((2, 512, 1024), dtype=jnp.int8)

    with pytest.raises(ValueError, match="Expected 4D GMM scale layout"):
        moe._normalize_scale_for_gmm(invalid_scale, weight, scale_name="wi_0_scale")


def test_epmoe_rejects_per_channel_4d_scale_with_non_unit_k_blocks():
    moe = _make_epmoe_for_scale_tests(None)
    # Weight [E, k, n]: k=512, n=1024 → out_dim=1024
    invalid_scale = jnp.ones((2, 4, 1, 1024), dtype=jnp.float32)
    weight = jnp.zeros((2, 512, 1024), dtype=jnp.int8)

    with pytest.raises(ValueError, match="Per-channel 4D GMM scales must have k_blocks=1"):
        moe._normalize_scale_for_gmm(invalid_scale, weight, scale_name="wi_0_scale")


def test_epmoe_accepts_4d_scale_when_k_blocks_divides_input_dim():
    moe = _make_epmoe_for_scale_tests([128, 128])
    # Weight [E, k, n]: k=512, n=1024. Even though config implies k_blocks=4,
    # runtime should accept any checkpoint-provided k_blocks that evenly divide k.
    valid_scale = jnp.ones((2, 8, 1, 1024), dtype=jnp.float32)
    weight = jnp.zeros((2, 512, 1024), dtype=jnp.int8)

    normalized = moe._normalize_scale_for_gmm(valid_scale, weight, scale_name="wi_0_scale")

    assert normalized.shape == valid_scale.shape


def test_epmoe_offline_block_scale_expansion_uses_block_size_out(monkeypatch):
    moe = _make_epmoe_for_scale_tests([128, 128])
    monkeypatch.setattr(jax.sharding, "reshard", lambda x, _: x)

    # Weight [E, k, n]: in_dim(k)=512, out_dim(n)=1024
    num_experts, out_dim, in_dim = 2, 1024, 512
    out_blocks, k_blocks = 8, 4
    weight = jnp.zeros((num_experts, in_dim, out_dim), dtype=jnp.int8)
    compact_scale = jnp.arange(num_experts * out_blocks * k_blocks, dtype=jnp.float32).reshape(
        num_experts, out_blocks, k_blocks
    )

    with jax.set_mesh(moe.moe_mesh):
        expanded = moe._normalize_scale_for_gmm(compact_scale, weight, scale_name="wi_0_scale")

    assert expanded.shape == (num_experts, k_blocks, 1, out_dim)
    # block_size_out=128 means channels [0:128] and [128:256] should come from different out-blocks.
    np.testing.assert_array_equal(
        np.asarray(expanded[:, :, 0, 0]),
        np.asarray(compact_scale[:, 0, :]),
    )
    np.testing.assert_array_equal(
        np.asarray(expanded[:, :, 0, 127]),
        np.asarray(compact_scale[:, 0, :]),
    )
    np.testing.assert_array_equal(
        np.asarray(expanded[:, :, 0, 128]),
        np.asarray(compact_scale[:, 1, :]),
    )
    np.testing.assert_array_equal(
        np.asarray(expanded[:, :, 0, 255]),
        np.asarray(compact_scale[:, 1, :]),
    )


def test_mimo_quantized_epmoe_mapping_uses_kernel_native_weight_layout():
    mappings = create_moe_weights_mapping_quantized(
        prefix="model.layers.0",
        target_prefix="model.layers.0",
        num_experts=8,
        moe_backend="epmoe",
        moe_path="mlp.experts",
        source_expert_pattern="{i}",
        is_quantized=True,
        hidden_size=4096,
        intermediate_size=2048,
        weight_block_size=128,
    )

    wi0 = mappings["__MOE_EXPERTS__model.layers.0.mlp.experts.wi_0"]
    wi1 = mappings["__MOE_EXPERTS__model.layers.0.mlp.experts.wi_1"]
    wo = mappings["__MOE_EXPERTS__model.layers.0.mlp.experts.wo"]

    assert wi0.transpose is True
    assert wi1.transpose is True
    assert wo.transpose is True
    assert wi0.sharding == ("expert", None, "tensor")
    assert wi1.sharding == ("expert", None, "tensor")
    assert wo.sharding == ("expert", "tensor", None)


def test_mimo_quantized_epmoe_scale_mapping_matches_gmm_layout():
    mappings = create_moe_weights_mapping_quantized(
        prefix="model.layers.0",
        target_prefix="model.layers.0",
        num_experts=8,
        moe_backend="epmoe",
        moe_path="mlp.experts",
        source_expert_pattern="{i}",
        is_quantized=True,
        hidden_size=4096,
        intermediate_size=2048,
        weight_block_size=128,
    )

    wi0_scale = mappings["__MOE_EXPERTS__model.layers.0.mlp.experts.wi_0_scale"]
    wi1_scale = mappings["__MOE_EXPERTS__model.layers.0.mlp.experts.wi_1_scale"]
    wo_scale = mappings["__MOE_EXPERTS__model.layers.0.mlp.experts.wo_scale"]

    assert wi0_scale.reshape == (8, 32, 1, 2048)
    assert wi1_scale.reshape == (8, 32, 1, 2048)
    assert wo_scale.reshape == (8, 16, 1, 4096)
    assert wi0_scale.sharding == ("expert", None, None, "tensor")
    assert wi1_scale.sharding == ("expert", None, None, "tensor")
    assert wo_scale.sharding == ("expert", None, None, None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
