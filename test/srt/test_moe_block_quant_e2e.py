# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

def get_cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    # Use float32 for stable calculation
    a_flat = a_flat.astype(jnp.float32)
    b_flat = b_flat.astype(jnp.float32)
    return jnp.dot(a_flat, b_flat) / (jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat))

def test_epmoe_block_quant_accuracy():
    """
    E2E Accuracy Test for EP MoE Block Quantization.
    Compares Block Quantized EPMoE output against BF16 Reference.
    """
    print("\n>>> Running E2E EP MoE Block Quantization Accuracy Alignment <<<")
    
    # 1. Setup Mesh (Support Multi-Device)
    num_devices = len(jax.devices())
    ep_size = num_devices
    tp_size = 1
    devices = np.array(jax.devices()).reshape(ep_size, tp_size)
    mesh = Mesh(devices, axis_names=("expert", "tensor"))
    
    # 2. Config
    hidden_size = 512
    intermediate_dim = 1024
    num_experts = 4 * ep_size
    num_experts_per_tok = 1
    block_size = 128
    compute_dtype = jnp.bfloat16
    weight_dtype = jnp.int8
    
    class BlockQuantConfig:
        def get_moe_weight_dtype(self): return weight_dtype
        def get_moe_activation_dtype(self): return None
        @property
        def weight_block_size(self): return [1, block_size]
        
    # 3. Create Reference (BF16) and Quantized Models
    print(f"  Initializing models with {num_experts} experts...")
    
    # Reference Model (BF16)
    moe_ref = EPMoE(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        ep_size=ep_size,
        mesh=mesh,
        intermediate_dim=intermediate_dim,
        quantization_config=None, # Pure BF16
    )
    
    # Quantized Model
    moe_quant = EPMoE(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        ep_size=ep_size,
        mesh=mesh,
        intermediate_dim=intermediate_dim,
        quantization_config=BlockQuantConfig(),
    )
    
    # 4. Generate Reference Weights and Quantize
    key = jax.random.PRNGKey(42)
    k_wi0, k_wi1, k_wo, k_in = jax.random.split(key, 4)
    
    # Generate distinct random weights for each projection
    w_wi0_fp = jax.random.normal(k_wi0, (num_experts, intermediate_dim, hidden_size), dtype=compute_dtype)
    w_wi1_fp = jax.random.normal(k_wi1, (num_experts, intermediate_dim, hidden_size), dtype=compute_dtype)
    w_wo_fp = jax.random.normal(k_wo, (num_experts, hidden_size, intermediate_dim), dtype=compute_dtype)
    
    # Set weights for Ref
    moe_ref.wi_0 = jax.nnx.Param(w_wi0_fp, out_sharding=P("expert", "tensor", None))
    moe_ref.wi_1 = jax.nnx.Param(w_wi1_fp, out_sharding=P("expert", "tensor", None))
    moe_ref.wo = jax.nnx.Param(w_wo_fp, out_sharding=P("expert", None, "tensor"))
    
    # Perform Block Quantization
    def block_quant(w, axis=2):
        w_q, w_scale = quantize_tensor(weight_dtype, w, axis=axis, block_size=block_size)
        k_blocks = w.shape[axis] // block_size
        # For WI, out_dim is inter_dim; for WO, out_dim is hidden_size
        out_dim = w.shape[1]
        w_scale = w_scale.reshape(num_experts, k_blocks, 1, out_dim)
        return w_q, w_scale

    wi0_q, wi0_scale = block_quant(w_wi0_fp, axis=2)
    wi1_q, wi1_scale = block_quant(w_wi1_fp, axis=2)
    wo_q, wo_scale = block_quant(w_wo_fp, axis=2)
    
    moe_quant.wi_0 = jax.nnx.Param(wi0_q, out_sharding=P("expert", "tensor", None))
    moe_quant.wi_1 = jax.nnx.Param(wi1_q, out_sharding=P("expert", "tensor", None))
    moe_quant.wo = jax.nnx.Param(wo_q, out_sharding=P("expert", None, "tensor"))
    
    # IMPORTANT: Sharding must match EPMoE's shard_map in_specs
    moe_quant.wi_0_scale = jax.nnx.Param(wi0_scale, out_sharding=P("expert", None, None, "tensor"))
    moe_quant.wi_1_scale = jax.nnx.Param(wi1_scale, out_sharding=P("expert", None, None, "tensor"))
    moe_quant.wo_scale = jax.nnx.Param(wo_scale, out_sharding=P("expert", None, None, None))
    
    # 5. Run Inference
    batch_size = 16
    x = jax.random.normal(k3, (batch_size, hidden_size), dtype=compute_dtype)
    topk_weights = jnp.ones((batch_size, num_experts_per_tok), dtype=compute_dtype)
    topk_ids = jax.random.randint(k4, (batch_size, num_experts_per_tok), 0, num_experts)
    
    print("  Running BF16 Reference...")
    out_ref = moe_ref(x, topk_weights, topk_ids)
    
    print("  Running Block Quantized MoE...")
    out_quant = moe_quant(x, topk_weights, topk_ids)
    
    # 6. Evaluation
    cos_sim = get_cosine_similarity(out_ref, out_quant)
    mae = jnp.mean(jnp.abs(out_ref - out_quant))
    rel_error = mae / jnp.mean(jnp.abs(out_ref))
    
    print(f"\n--- Accuracy Results ---")
    print(f"  Cosine Similarity: {cos_sim.item():.6f}")
    print(f"  Relative Error:    {rel_error.item():.6%}")
    print(f"  MAE:               {mae.item():.6f}")
    
    # Assertions
    assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim:.6f}"
    assert rel_error < 0.05, f"Relative error too high: {rel_error:.6%}"
    
    print("\n  Accuracy Alignment PASSED!")

if __name__ == "__main__":
    test_epmoe_block_quant_accuracy()
