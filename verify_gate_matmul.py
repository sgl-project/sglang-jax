#!/usr/bin/env python3
"""
Verify gate matrix multiplication manually.

This script loads the dumped arrays and performs manual matrix multiplication
to verify if the computation is correct.
"""

import glob
import os

import numpy as np


def find_latest_dump_files(dump_dir: str, layer_id: int = 0, suffix: str = None):
    """Find the latest dump files for a given layer.

    Supports both formats:
    - layer_0_hidden_states_fp32.npy (without suffix)
    - layer_0_hidden_states_fp32_run_001.npy (with suffix)

    Args:
        dump_dir: Directory containing dump files
        layer_id: Layer ID to search for
        suffix: Optional suffix to match (e.g., "run_001"). If None, uses latest.

    Returns:
        tuple: (hidden_file, weight_file, output_file) or None if not found
    """
    if suffix:
        # Use specific suffix
        hidden_file = os.path.join(dump_dir, f"layer_{layer_id}_hidden_states_fp32_{suffix}.npy")
        weight_file = os.path.join(dump_dir, f"layer_{layer_id}_gate_weight_fp32_{suffix}.npy")
        output_file = os.path.join(dump_dir, f"layer_{layer_id}_router_logits_output_{suffix}.npy")

        if not os.path.exists(hidden_file):
            print(f"❌ No files found with suffix '{suffix}'")
            return None

        return hidden_file, weight_file, output_file
    else:
        # Auto-detect: try to find files with run suffix first (newest format)
        hidden_pattern = os.path.join(dump_dir, f"layer_{layer_id}_hidden_states_fp32_run_*.npy")
        weight_pattern = os.path.join(dump_dir, f"layer_{layer_id}_gate_weight_fp32_run_*.npy")
        output_pattern = os.path.join(dump_dir, f"layer_{layer_id}_router_logits_output_run_*.npy")

        hidden_files = sorted(glob.glob(hidden_pattern))
        weight_files = sorted(glob.glob(weight_pattern))
        output_files = sorted(glob.glob(output_pattern))

        # If no files with suffix found, try without suffix
        if not hidden_files:
            hidden_files = [os.path.join(dump_dir, f"layer_{layer_id}_hidden_states_fp32.npy")]
            weight_files = [os.path.join(dump_dir, f"layer_{layer_id}_gate_weight_fp32.npy")]
            output_files = [os.path.join(dump_dir, f"layer_{layer_id}_router_logits_output.npy")]

        # Check if files exist
        if not os.path.exists(hidden_files[-1]):
            return None

        return hidden_files[-1], weight_files[-1], output_files[-1]


def print_matrix_sample(name: str, matrix: np.ndarray, num_rows: int = 3, num_cols: int = 8):
    """Print a sample of the matrix values."""
    print(f"\n{'='*60}")
    print(f"{name} - Sample values (first {num_rows} rows, first {num_cols} cols):")
    print(f"Shape: {matrix.shape}, dtype: {matrix.dtype}")
    print("=" * 60)

    rows_to_show = min(num_rows, matrix.shape[0])
    cols_to_show = min(num_cols, matrix.shape[1] if matrix.ndim > 1 else matrix.shape[0])

    if matrix.ndim == 1:
        print(f"[{', '.join([f'{x:.6f}' for x in matrix[:cols_to_show]])}...]")
    else:
        for i in range(rows_to_show):
            row = matrix[i, :cols_to_show]
            print(f"Row {i}: [{', '.join([f'{x:.6f}' for x in row])}...]")
        if rows_to_show < matrix.shape[0]:
            print(f"... ({matrix.shape[0] - rows_to_show} more rows)")


def verify_gate_computation(layer_id: int = 0, dump_dir: str = "debug_dumps", suffix: str = None):
    """Verify gate computation for a specific layer."""

    if suffix:
        print(f"📂 Searching for dump files with suffix '{suffix}' in {dump_dir}...")
    else:
        print(f"📂 Searching for dump files in {dump_dir}...")
    result = find_latest_dump_files(dump_dir, layer_id, suffix)

    if result is None:
        print(f"❌ No dump files found for layer {layer_id}")
        print("Please run your JAX model first to generate the dump files.")
        return

    hidden_file, weight_file, output_file = result
    print(f"✅ Found dump files:")
    print(f"   Hidden states: {os.path.basename(hidden_file)}")
    print(f"   Gate weight:   {os.path.basename(weight_file)}")
    print(f"   Router logits: {os.path.basename(output_file)}")

    print("\n📂 Loading dumped arrays...")
    hidden_states = np.load(hidden_file)  # (batch, hidden_dim)
    gate_weight = np.load(
        weight_file
    )  # Should be (hidden_dim, num_experts) OR (num_experts, hidden_dim)
    router_logits_jax = np.load(output_file)  # (batch, num_experts)

    print(f"✅ hidden_states shape: {hidden_states.shape}")
    print(f"✅ gate_weight shape: {gate_weight.shape}")
    print(f"✅ router_logits (JAX output) shape: {router_logits_jax.shape}")

    # Print original matrix samples
    print("\n" + "=" * 60)
    print("📊 ORIGINAL MATRIX VALUES")
    print("=" * 60)

    print_matrix_sample("Hidden States", hidden_states, num_rows=2, num_cols=8)
    print_matrix_sample("Gate Weight", gate_weight, num_rows=5, num_cols=8)
    print_matrix_sample("Router Logits (JAX Output)", router_logits_jax, num_rows=2, num_cols=8)

    # Determine the correct computation based on shapes
    batch_size, hidden_dim = hidden_states.shape
    expected_num_experts = router_logits_jax.shape[1]

    print(f"\n🔍 Expected output shape: ({batch_size}, {expected_num_experts})")

    # Try different interpretations of weight matrix
    print("\n" + "=" * 60)
    print("🔬 VERIFICATION: Testing different weight matrix layouts...")
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

        print_matrix_sample(
            "Manual Calculation Result (Case 1)", router_logits_manual_1, num_rows=2, num_cols=8
        )

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

        print_matrix_sample(
            "Manual Calculation Result (Case 2)", router_logits_manual_2, num_rows=2, num_cols=8
        )
    else:
        print(f"\n❌ Unexpected weight shape: {gate_weight.shape}")
        print(f"   Cannot match expected dimensions.")

    # Show sample values for debugging
    print("\n" + "=" * 60)
    print("📊 COMPARISON: Sample values (first token, first 5 experts):")
    print("=" * 60)
    print(f"JAX output:    {router_logits_jax[0, :5]}")
    if gate_weight.shape == (hidden_dim, expected_num_experts):
        router_logits_manual_1 = hidden_states @ gate_weight
        print(f"Manual calc 1: {router_logits_manual_1[0, :5]}")
        diff = router_logits_jax[0, :5] - router_logits_manual_1[0, :5]
        print(f"Difference:    {diff}")
    elif gate_weight.shape == (expected_num_experts, hidden_dim):
        router_logits_manual_2 = hidden_states @ gate_weight.T
        print(f"Manual calc 2: {router_logits_manual_2[0, :5]}")
        diff = router_logits_jax[0, :5] - router_logits_manual_2[0, :5]
        print(f"Difference:    {diff}")

    # Check weight statistics
    print("\n" + "=" * 60)
    print("📈 STATISTICS")
    print("=" * 60)

    print("\nWeight matrix statistics:")
    print(f"  Min:  {np.min(gate_weight):.6f}")
    print(f"  Max:  {np.max(gate_weight):.6f}")
    print(f"  Mean: {np.mean(gate_weight):.6f}")
    print(f"  Std:  {np.std(gate_weight):.6f}")

    print("\nHidden states statistics:")
    print(f"  Min:  {np.min(hidden_states):.6f}")
    print(f"  Max:  {np.max(hidden_states):.6f}")
    print(f"  Mean: {np.mean(hidden_states):.6f}")
    print(f"  Std:  {np.std(hidden_states):.6f}")

    print("\nRouter logits (JAX) statistics:")
    print(f"  Min:  {np.min(router_logits_jax):.6f}")
    print(f"  Max:  {np.max(router_logits_jax):.6f}")
    print(f"  Mean: {np.mean(router_logits_jax):.6f}")
    print(f"  Std:  {np.std(router_logits_jax):.6f}")


def list_all_dumps(dump_dir: str = "debug_dumps"):
    """List all available dump files."""
    if not os.path.exists(dump_dir):
        print(f"❌ Directory {dump_dir} does not exist")
        return

    print("\n" + "=" * 60)
    print("📋 All available dump files:")
    print("=" * 60)

    files = sorted(glob.glob(os.path.join(dump_dir, "*.npy")))
    if not files:
        print("No dump files found")
        return

    for f in files:
        basename = os.path.basename(f)
        size = os.path.getsize(f)
        print(f"  {basename:60s} ({size:>10,} bytes)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify gate matrix multiplication",
        epilog="""
Examples:
  # Verify latest dump files for layer 0
  python verify_gate_matmul.py

  # List all available dump files
  python verify_gate_matmul.py --list

  # Verify specific run (e.g., prefill phase)
  python verify_gate_matmul.py --suffix run_001

  # Verify layer 1 with specific suffix
  python verify_gate_matmul.py --layer 1 --suffix run_002
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer ID to verify (default: 0)")
    parser.add_argument(
        "--dump-dir", type=str, default="debug_dumps", help="Dump directory (default: debug_dumps)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Specific suffix to match (e.g., 'run_001_t123456'). If not specified, uses latest.",
    )
    parser.add_argument("--list", action="store_true", help="List all dump files")

    args = parser.parse_args()

    print("🔬 Gate Matrix Multiplication Verification Tool")
    print("=" * 60)

    if args.list:
        list_all_dumps(args.dump_dir)
        print()

    verify_gate_computation(layer_id=args.layer, dump_dir=args.dump_dir, suffix=args.suffix)
