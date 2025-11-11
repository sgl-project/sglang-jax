#!/usr/bin/env python3
"""
Verify gate matrix multiplication manually.

This script loads the dumped arrays and performs manual matrix multiplication
to verify if the computation is correct.
"""

import os

import numpy as np


def verify_gate_computation(layer_id: int = 0):
    """Verify gate computation for a specific layer."""
    dump_dir = "debug_dumps"

    # Load arrays
    hidden_file = os.path.join(dump_dir, f"layer{layer_id}_hidden_states_fp32.npy")
    weight_file = os.path.join(dump_dir, f"layer{layer_id}_gate_weight_fp32.npy")
    output_file = os.path.join(dump_dir, f"layer{layer_id}_router_logits_output.npy")

    if not os.path.exists(hidden_file):
        print(f"❌ File not found: {hidden_file}")
        print("Please run your JAX model first to generate the dump files.")
        return

    print("📂 Loading dumped arrays...")
    hidden_states = np.load(hidden_file)  # (batch, hidden_dim)
    gate_weight = np.load(
        weight_file
    )  # Should be (hidden_dim, num_experts) OR (num_experts, hidden_dim)
    router_logits_jax = np.load(output_file)  # (batch, num_experts)

    print(f"✅ hidden_states shape: {hidden_states.shape}")
    print(f"✅ gate_weight shape: {gate_weight.shape}")
    print(f"✅ router_logits (JAX output) shape: {router_logits_jax.shape}")

    # Determine the correct computation based on shapes
    batch_size, hidden_dim = hidden_states.shape
    expected_num_experts = router_logits_jax.shape[1]

    print(f"\n🔍 Expected output shape: ({batch_size}, {expected_num_experts})")

    # Try different interpretations of weight matrix
    print("\n" + "=" * 60)
    print("Testing different weight matrix layouts...")
    print("=" * 60)

    # Case 1: weight shape is (hidden_dim, num_experts) - NO transpose needed
    if gate_weight.shape == (hidden_dim, expected_num_experts):
        print("\n✅ Case 1: weight shape = (hidden_dim, num_experts)")
        print(f"   Computing: hidden_states @ gate_weight")
        router_logits_manual_1 = hidden_states @ gate_weight
        print(f"   Result shape: {router_logits_manual_1.shape}")

        diff_1 = np.abs(router_logits_manual_1 - router_logits_jax)
        max_diff_1 = np.max(diff_1)
        mean_diff_1 = np.mean(diff_1)

        print(f"   Max absolute difference: {max_diff_1:.6e}")
        print(f"   Mean absolute difference: {mean_diff_1:.6e}")

        if max_diff_1 < 1e-4:
            print("   ✅ MATCH! This is the correct interpretation.")
        else:
            print("   ❌ NO MATCH. Large difference detected.")

    # Case 2: weight shape is (num_experts, hidden_dim) - NEEDS transpose
    elif gate_weight.shape == (expected_num_experts, hidden_dim):
        print("\n✅ Case 2: weight shape = (num_experts, hidden_dim)")
        print(f"   Computing: hidden_states @ gate_weight.T")
        router_logits_manual_2 = hidden_states @ gate_weight.T
        print(f"   Result shape: {router_logits_manual_2.shape}")

        diff_2 = np.abs(router_logits_manual_2 - router_logits_jax)
        max_diff_2 = np.max(diff_2)
        mean_diff_2 = np.mean(diff_2)

        print(f"   Max absolute difference: {max_diff_2:.6e}")
        print(f"   Mean absolute difference: {mean_diff_2:.6e}")

        if max_diff_2 < 1e-4:
            print("   ✅ MATCH! This is the correct interpretation.")
        else:
            print("   ❌ NO MATCH. Large difference detected.")
    else:
        print(f"\n❌ Unexpected weight shape: {gate_weight.shape}")
        print(f"   Cannot match expected dimensions.")

    # Show sample values for debugging
    print("\n" + "=" * 60)
    print("Sample values (first token, first 5 experts):")
    print("=" * 60)
    print(f"JAX output:    {router_logits_jax[0, :5]}")
    if gate_weight.shape == (hidden_dim, expected_num_experts):
        print(f"Manual calc 1: {router_logits_manual_1[0, :5]}")
    elif gate_weight.shape == (expected_num_experts, hidden_dim):
        print(f"Manual calc 2: {router_logits_manual_2[0, :5]}")

    # Check weight statistics
    print("\n" + "=" * 60)
    print("Weight matrix statistics:")
    print("=" * 60)
    print(f"Min:  {np.min(gate_weight):.6f}")
    print(f"Max:  {np.max(gate_weight):.6f}")
    print(f"Mean: {np.mean(gate_weight):.6f}")
    print(f"Std:  {np.std(gate_weight):.6f}")

    print("\n" + "=" * 60)
    print("Hidden states statistics:")
    print("=" * 60)
    print(f"Min:  {np.min(hidden_states):.6f}")
    print(f"Max:  {np.max(hidden_states):.6f}")
    print(f"Mean: {np.mean(hidden_states):.6f}")
    print(f"Std:  {np.std(hidden_states):.6f}")


if __name__ == "__main__":
    print("🔬 Gate Matrix Multiplication Verification Tool")
    print("=" * 60)
    verify_gate_computation(layer_id=0)
