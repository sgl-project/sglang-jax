# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from sgl_jax.srt.layers.moe import EPMoE


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
    block_size = 128
    
    # Mock QuantizationConfig
    class MockQuantConfig:
        def get_moe_weight_dtype(self): return jnp.int8
        def get_moe_activation_dtype(self): return None
        @property
        def weight_block_size(self): return weight_block_size
        
    quant_config = MockQuantConfig()
    
    # 3. Initialize EPMoE with Mocked Mesh to bypass sharding checks on CPU
    from unittest.mock import MagicMock
    mock_mesh = MagicMock(spec=Mesh)
    mock_mesh.shape = {"expert": 1, "tensor": 1}
    mock_mesh.axis_names = ("expert", "tensor")
    mock_mesh.devices = np.array(jax.devices()[:1]).reshape(1, 1)
    
    # We monkeypatch the sharding in EPMoE to be CPU-friendly for this test
    original_p = P
    import sgl_jax.srt.layers.moe as moe_module
    moe_module.P = lambda *args: None # Disable sharding for CPU UT
    
    try:
        moe = EPMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=intermediate_dim,
            quantization_config=quant_config,
        )
    finally:
        moe_module.P = original_p # Restore

    
    # 4. Run Quantization Prep
    moe.quantize_weights(is_static=True)
    
    # 5. Assert Scale Shapes
    # EPMoE logic: k_blocks = hidden_size // block_size
    k_blocks_wi = expected_k_blocks_wi
    k_blocks_wo = expected_k_blocks_wo
    
    print(f"  Expert Count: {num_experts}")
    print(f"  K Blocks (WI): {k_blocks_wi}")
    print(f"  K Blocks (WO): {k_blocks_wo}")
    
    expected_wi_shape = (num_experts, k_blocks_wi, 1, intermediate_dim)
    expected_wo_shape = (num_experts, k_blocks_wo, 1, hidden_size)
    
    print(f"  WI_0 Scale Shape: {moe.wi_0_scale.value.shape}")
    print(f"  WO Scale Shape:   {moe.wo_scale.value.shape}")
    
    assert moe.wi_0_scale.value.shape == expected_wi_shape, f"WI shape mismatch: {moe.wi_0_scale.value.shape} vs {expected_wi_shape}"
    assert moe.wo_scale.value.shape == expected_wo_shape, f"WO shape mismatch: {moe.wo_scale.value.shape} vs {expected_wo_shape}"
    
    print("  Shape Verification: PASSED")
    
    # 6. Verify Content (Should be zeros as initialized in is_static=True)
    assert jnp.all(moe.wi_0_scale.value == 0)
    print("  Content Verification: PASSED")

if __name__ == "__main__":
    try:
        test_epmoe_block_quant_logic(None, 1, 1)
        test_epmoe_block_quant_logic([1, 128], 4, 8)
        test_epmoe_block_quant_logic([128, 128], 4, 8)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
