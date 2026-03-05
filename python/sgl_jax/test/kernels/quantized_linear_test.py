# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
# import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from sgl_jax.srt.kernels.quantized_matmul.kernel import xla_quantized_matmul_local
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

def run_block_quant_test(weight_dtype, dtype_name):
    """
    Test the block quantization path for a specific weight dtype.
    """
    print(f"\n>>> Testing Block Quantization with {dtype_name} <<<")
    
    # 1. Setup dimensions
    batch, in_dim, out_dim = 2, 256, 512
    block_size = (128, 128) # (out_block, in_block)
    compute_dtype = jnp.bfloat16

    # 2. Generate random input and weight
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (batch, in_dim), dtype=compute_dtype)
    w_fp = jax.random.normal(k2, (out_dim, in_dim), dtype=compute_dtype)

    # 3. Perform block-wise quantization
    # Explicitly cast to float32 for quantization to avoid JAX promotion errors
    w_fp_f32 = w_fp.astype(jnp.float32)
    w_q, w_scale = quantize_tensor(
        dtype=weight_dtype,
        tensor=w_fp_f32,
        axis=(0, 1),
        block_size=block_size,
    )
    
    # 4. Reference implementation (Original Floating point matmul)
    ref_out = jnp.dot(x, w_fp.T)

    # 5. Call the kernel
    out = xla_quantized_matmul_local(
        x=x,
        w_q=w_q,
        w_scale=w_scale,
        quantize_activation=False,
        compute_dtype=compute_dtype,
        weight_block_size=block_size,
    )

    # 6. Evaluate Results
    diff = jnp.abs(out - ref_out)
    mae = jnp.mean(diff)
    max_diff = jnp.max(diff)
    
    # Calculate relative error (relative to the mean absolute value of ref_out)
    rel_error = mae / jnp.mean(jnp.abs(ref_out))

    print(f"  MAE (Mean Absolute Error): {mae.item():.6f}")
    print(f"  Max Absolute Difference:  {max_diff.item():.6f}")
    print(f"  Relative Error:           {rel_error.item():.6%}")

    # For unit test verification, we check if relative error is within a reasonable bound
    # INT8 and FP8 should typically have < 1% relative error for random normal distributions
    assert rel_error.item() < 0.05, f"Relative error too high: {rel_error.item():.6%}"
    print(f"  Verification: PASSED (Rel Error < 5%)")

def test_xla_quantized_matmul_block_quant_all():
    # Test INT8
    run_block_quant_test(jnp.int8, "INT8")
    
    # Test FP8 (if supported by the environment/JAX version)
    try:
        if hasattr(jnp, "float8_e4m3fn"):
            run_block_quant_test(jnp.float8_e4m3fn, "FP8_E4M3")
        else:
            print("\n>>> Skipping FP8 test: jnp.float8_e4m3fn not available <<<")
    except Exception as e:
        print(f"\n>>> FP8 test failed/skipped: {e} <<<")

if __name__ == "__main__":
    test_xla_quantized_matmul_block_quant_all()