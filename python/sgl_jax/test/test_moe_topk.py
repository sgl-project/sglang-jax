"""Unit tests for MoE TopK class."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.moe import TopK


def test_standard_topk():
    """Test standard topk without grouping or bias."""
    router_logits = jnp.array(
        [
            [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5],
            [0.0, 0.0, 0.0, 2.0, 1.5, 1.0, 0.5, 0.0],
        ],
        dtype=jnp.float32,
    )

    topk = TopK(topk=3, renormalize=True, num_expert_group=0, topk_group=0)
    topk_weights, topk_ids = topk(router_logits)  # Changed order!

    expected_ids = np.array([[0, 1, 2], [3, 4, 5]])
    np.testing.assert_array_equal(np.array(topk_ids), expected_ids)
    np.testing.assert_allclose(np.array(topk_weights).sum(axis=1), 1.0, atol=1e-6)


def test_biased_topk():
    """Test biased topk shifts expert selection."""
    router_logits = jnp.array([[0.1] * 8], dtype=jnp.float32)
    correction_bias = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0], dtype=jnp.float32)

    topk = TopK(topk=3, renormalize=True, num_expert_group=0, topk_group=0)
    _, topk_ids = topk(router_logits, correction_bias)  # Changed order!

    assert 5 in np.array(topk_ids)[0], f"Biased expert 5 not selected: {np.array(topk_ids)[0]}"


def test_grouped_topk():
    """Test grouped topk with multiple groups (including single group case)."""
    # Case 1: Single group selection (topk_group=1)
    router_logits_1 = jnp.array(
        [
            [0.1, 0.1, 0.1, 0.1, 2.0, 1.5, 1.0, 0.5],  # Group 1 has higher scores
        ],
        dtype=jnp.float32,
    )

    topk_1 = TopK(topk=2, renormalize=True, num_expert_group=2, topk_group=1)
    _, topk_ids_1 = topk_1(router_logits_1)

    selected_1 = np.array(topk_ids_1)[0]
    assert all(
        4 <= eid <= 7 for eid in selected_1
    ), f"Experts {selected_1} not all from group 1 (4-7)"

    # Case 2: Multiple groups selection (topk_group=2)
    router_logits_2 = jnp.array(
        [
            [
                2.0,
                1.9,
                0.1,
                0.1,  # Group 0: high
                0.1,
                0.1,
                0.1,
                0.1,  # Group 1: low
                0.1,
                0.1,
                0.1,
                0.1,  # Group 2: low
                1.8,
                1.7,
                0.1,
                0.1,
            ],  # Group 3: high
        ],
        dtype=jnp.float32,
    )

    topk_2 = TopK(topk=3, renormalize=True, num_expert_group=4, topk_group=2)
    _, topk_ids_2 = topk_2(router_logits_2)

    selected_2 = np.array(topk_ids_2)[0]
    valid = all((0 <= eid <= 3) or (12 <= eid <= 15) for eid in selected_2)
    assert valid, f"Experts {selected_2} not from groups 0 or 3"


def test_biased_grouped_topk():
    """Test biased grouped topk."""
    router_logits = jnp.array([[0.5] * 8], dtype=jnp.float32)
    correction_bias = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0], dtype=jnp.float32)

    topk = TopK(
        topk=2,
        renormalize=True,
        num_expert_group=2,
        topk_group=1,
        routed_scaling_factor=1.0,
    )
    _, topk_ids = topk(router_logits, correction_bias)

    assert 6 in np.array(topk_ids)[0], f"Biased expert 6 not selected: {np.array(topk_ids)[0]}"


def test_weight_normalization():
    """Test weight normalization and scaling."""
    router_logits = jnp.array(
        [
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4],
        ],
        dtype=jnp.float32,
    )

    # With normalization
    topk_norm = TopK(topk=3, renormalize=True)
    weights_norm, _ = topk_norm(router_logits)

    # With scaling
    topk_scaled = TopK(topk=3, renormalize=True, routed_scaling_factor=2.0)
    weights_scaled, _ = topk_scaled(router_logits)

    np.testing.assert_allclose(np.array(weights_norm).sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.array(weights_scaled).sum(), 2.0, atol=1e-5)


def test_biased_grouped_topk_batch_consistency():
    """Test biased grouped topk consistency across different batch processing strategies.

    This test verifies that processing 128 requests in different ways produces identical results:
    - Strategy A: Process all 128 at once (bsz=128)
    - Strategy B: Process in 4 batches of 32 (bsz=32, 4 iterations)
    - Strategy C: Process in 2 batches of 64 (bsz=64, 2 iterations)

    This ensures numerical stability and correctness regardless of batch size.
    """
    # Setup: 256 experts, 32 groups, select top 4 groups, top 8 experts
    num_experts = 256
    num_expert_group = 32
    topk_group = 4
    topk = 8
    hidden_dim = 2048
    total_requests = 128

    print(f"\n  Generating {total_requests} fixed test requests...")

    # Generate deterministic test data using JAX PRNG
    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)

    correction_bias = jax.random.normal(key1, (num_experts,), dtype=jnp.float32) * 0.1
    gate_weight = jax.random.normal(key2, (hidden_dim, num_experts), dtype=jnp.float32) * 0.02

    # Generate 128 fixed inputs
    inputs_128 = jax.random.normal(key3, (total_requests, hidden_dim), dtype=jnp.float32) * 0.5

    # Compute router logits for all 128 requests
    logits_128 = jnp.dot(inputs_128, gate_weight)
    logits_128 = 1.0 / (1.0 + jnp.exp(-logits_128))  # Apply sigmoid

    # Create TopK instance
    topk_module = TopK(
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        routed_scaling_factor=1.0,
    )

    # Strategy A: Process all 128 at once
    print(f"\n  Strategy A: Processing all {total_requests} requests at once (bsz=128)")
    weights_128, ids_128 = topk_module(logits_128, correction_bias)

    weight_sums = jnp.sum(weights_128, axis=1)
    print(f"    - Shape: {ids_128.shape}, Avg weight sum: {float(jnp.mean(weight_sums)):.6f}")

    # Verify basic properties
    assert ids_128.shape == (total_requests, topk), f"Unexpected shape: {ids_128.shape}"
    np.testing.assert_allclose(
        np.array(weight_sums), 1.0, atol=1e-5, err_msg="Weights not normalized"
    )

    # Strategy B: Process in 4 batches of 32
    print("\n  Strategy B: Processing in 4 batches of 32")
    weights_32_batches = []
    ids_32_batches = []

    for i in range(4):
        start_idx = i * 32
        end_idx = start_idx + 32
        logits_batch = logits_128[start_idx:end_idx]

        weights_batch, ids_batch = topk_module(logits_batch, correction_bias)
        weights_32_batches.append(weights_batch)
        ids_32_batches.append(ids_batch)

        print(f"    - Batch {i+1}/4 [{start_idx}:{end_idx}]: Shape {ids_batch.shape}")

    # Concatenate results
    weights_32_concat = jnp.concatenate(weights_32_batches, axis=0)
    ids_32_concat = jnp.concatenate(ids_32_batches, axis=0)

    # Strategy C: Process in 2 batches of 64
    print("\n  Strategy C: Processing in 2 batches of 64")
    weights_64_batches = []
    ids_64_batches = []

    for i in range(2):
        start_idx = i * 64
        end_idx = start_idx + 64
        logits_batch = logits_128[start_idx:end_idx]

        weights_batch, ids_batch = topk_module(logits_batch, correction_bias)
        weights_64_batches.append(weights_batch)
        ids_64_batches.append(ids_batch)

        print(f"    - Batch {i+1}/2 [{start_idx}:{end_idx}]: Shape {ids_batch.shape}")

    # Concatenate results
    weights_64_concat = jnp.concatenate(weights_64_batches, axis=0)
    ids_64_concat = jnp.concatenate(ids_64_batches, axis=0)

    # Verify all strategies produce identical results
    print("\n  Verifying output alignment:")

    # Compare IDs
    ids_32_match = bool(jnp.array_equal(ids_128, ids_32_concat))
    ids_64_match = bool(jnp.array_equal(ids_128, ids_64_concat))

    print("    - Expert IDs alignment:")
    print(f"      bsz=128 vs 4×bsz=32: {ids_32_match}")
    print(f"      bsz=128 vs 2×bsz=64: {ids_64_match}")

    # Compare weights
    weights_32_match = bool(jnp.allclose(weights_128, weights_32_concat, atol=1e-6))
    weights_64_match = bool(jnp.allclose(weights_128, weights_64_concat, atol=1e-6))

    print("    - Routing weights alignment:")
    print(f"      bsz=128 vs 4×bsz=32: {weights_32_match}")
    print(f"      bsz=128 vs 2×bsz=64: {weights_64_match}")

    # Show detailed mismatch if any
    if not ids_32_match or not weights_32_match:
        print("\n    Detailed mismatch (bsz=128 vs 4×bsz=32):")
        mismatch_indices = jnp.where(jnp.any(ids_128 != ids_32_concat, axis=1))[0]
        for idx in mismatch_indices[:3]:
            print(f"      Request {idx}:")
            print(f"        IDs (bsz=128): {ids_128[idx]}")
            print(f"        IDs (bsz=32):  {ids_32_concat[idx]}")
            print(f"        Weights (bsz=128): {weights_128[idx]}")
            print(f"        Weights (bsz=32):  {weights_32_concat[idx]}")

    if not ids_64_match or not weights_64_match:
        print("\n    Detailed mismatch (bsz=128 vs 2×bsz=64):")
        mismatch_indices = jnp.where(jnp.any(ids_128 != ids_64_concat, axis=1))[0]
        for idx in mismatch_indices[:3]:
            print(f"      Request {idx}:")
            print(f"        IDs (bsz=128): {ids_128[idx]}")
            print(f"        IDs (bsz=64):  {ids_64_concat[idx]}")

    # Assert all alignments
    assert ids_32_match, "Expert IDs differ between bsz=128 and 4×bsz=32"
    assert weights_32_match, "Weights differ between bsz=128 and 4×bsz=32"
    assert ids_64_match, "Expert IDs differ between bsz=128 and 2×bsz=64"
    assert weights_64_match, "Weights differ between bsz=128 and 2×bsz=64"

    # Compute and display statistics
    print("\n  Distribution statistics:")

    def compute_stats(ids, weights):
        """Compute expert usage statistics."""
        # Expert frequency
        ids_flat = ids.flatten()
        unique_experts = jnp.unique(ids_flat)

        # Weight distribution
        max_weight_per_req = jnp.max(weights, axis=1)
        min_weight_per_req = jnp.min(weights, axis=1)

        return {
            "unique_experts": len(unique_experts),
            "total_selections": len(ids_flat),
            "avg_max_weight": float(jnp.mean(max_weight_per_req)),
            "avg_min_weight": float(jnp.mean(min_weight_per_req)),
        }

    stats_128 = compute_stats(ids_128, weights_128)
    stats_32 = compute_stats(ids_32_concat, weights_32_concat)
    stats_64 = compute_stats(ids_64_concat, weights_64_concat)

    print(f"    - Unique experts used: {stats_128['unique_experts']}/{num_experts}")
    print(f"    - Total expert selections: {stats_128['total_selections']}")
    print(f"    - Avg max weight per request: {stats_128['avg_max_weight']:.4f}")
    print(f"    - Avg min weight per request: {stats_128['avg_min_weight']:.4f}")

    # Verify stats match across strategies
    assert stats_128 == stats_32 == stats_64, "Statistics differ across batch sizes"

    print(f"\n  ✓ All {total_requests} requests produce identical outputs across batch sizes")


def main():
    tests = [
        ("Standard TopK", test_standard_topk),
        ("Biased TopK", test_biased_topk),
        ("Grouped TopK", test_grouped_topk),
        ("Biased Grouped TopK", test_biased_grouped_topk),
        ("Weight Normalization", test_weight_normalization),
        (
            "Biased Grouped TopK Batch Consistency",
            test_biased_grouped_topk_batch_consistency,
        ),
    ]

    print("Running TopK tests...")
    for name, test_func in tests:
        try:
            test_func()
            print(f"  ✓ {name}")
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            raise
        except Exception as e:
            print(f"  ✗ {name}: Unexpected error: {e}")
            raise

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
