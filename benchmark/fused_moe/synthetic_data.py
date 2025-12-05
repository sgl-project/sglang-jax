"""Synthetic data generation for MoE benchmark."""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from benchmark.fused_moe.config_utils import MoEBenchmarkConfig


def create_synthetic_weights(
    config: MoEBenchmarkConfig,
    mesh: Mesh,
    seed: int = 42,
) -> Tuple[Dict[str, jax.Array], Dict[str, jax.Array]]:
    """
    Create synthetic weights for both FusedEPMoE and EPMoE.

    Ensures mathematical equivalence between the two implementations.

    Args:
        config: Benchmark configuration
        mesh: JAX mesh for sharding
        seed: Random seed for reproducibility

    Returns:
        fused_weights: Dictionary with keys "w1", "w2" for FusedEPMoE
        epmoe_weights: Dictionary with keys "wi_0", "wi_1", "wo" for EPMoE
    """
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)

    dtype = jnp.bfloat16 if config.weight_dtype == "bfloat16" else jnp.float32

    # Generate base weights in EPMoE format
    # wi_0: gate projection (num_experts, hidden_size, intermediate_size)
    wi_0 = (
        jax.random.normal(
            key1,
            (config.num_experts, config.hidden_size, config.intermediate_size),
            dtype=dtype,
        )
        * 0.02
    )

    # wi_1: up projection (num_experts, hidden_size, intermediate_size)
    wi_1 = (
        jax.random.normal(
            key2,
            (config.num_experts, config.hidden_size, config.intermediate_size),
            dtype=dtype,
        )
        * 0.02
    )

    # wo: down projection (num_experts, intermediate_size, hidden_size)
    wo = (
        jax.random.normal(
            key3,
            (config.num_experts, config.intermediate_size, config.hidden_size),
            dtype=dtype,
        )
        * 0.02
    )

    # Create fused format for FusedEPMoE
    # IMPORTANT: FusedEPMoE expects transposed weights!
    # w1[:, 0, :, :] = wi_0.transpose(0, 2, 1)  # gate
    # w1[:, 1, :, :] = wi_1.transpose(0, 2, 1)  # up
    w1_gate = jnp.transpose(wi_0, (0, 2, 1))  # (num_experts, intermediate_size, hidden_size)
    w1_up = jnp.transpose(wi_1, (0, 2, 1))  # (num_experts, intermediate_size, hidden_size)
    w1 = jnp.stack([w1_gate, w1_up], axis=1)  # (num_experts, 2, intermediate_size, hidden_size)

    w2 = wo  # Same format for both

    fused_weights = {
        "w1": w1,
        "w2": w2,
    }

    epmoe_weights = {
        "wi_0": wi_0,
        "wi_1": wi_1,
        "wo": wo,
    }

    return fused_weights, epmoe_weights


def generate_router_logits(
    num_tokens: int,
    num_experts: int,
    scenario: str,
    num_experts_per_tok: int = 2,
    imbalance_factor: float = 3.0,
    seed: int = 42,
) -> jax.Array:
    """
    Generate router logits with different distribution patterns.

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        scenario: One of "random", "balanced", "imbalanced"
        num_experts_per_tok: Top-k value (for balanced scenario)
        imbalance_factor: Target max_load / avg_load for imbalanced scenario
        seed: Random seed

    Returns:
        router_logits: (num_tokens, num_experts) array of logits

    Scenarios:
        - random: Uniform random N(0, 1) logits, natural imbalance ~1.2-1.5x
        - balanced: Engineered to achieve ~1.0x imbalance (perfect balance)
        - imbalanced: Skewed to achieve target imbalance_factor (default 3.0x)
    """
    key = jax.random.PRNGKey(seed)

    if scenario == "random":
        # Uniform random logits
        router_logits = jax.random.normal(key, (num_tokens, num_experts), dtype=jnp.float32)

    elif scenario == "balanced":
        # Round-robin assignment to ensure equal distribution
        router_logits = jnp.ones((num_tokens, num_experts), dtype=jnp.float32) * -10.0

        # Assign each token to experts in round-robin fashion
        for token_idx in range(num_tokens):
            # Calculate which experts this token should prefer
            start_expert = (token_idx * num_experts_per_tok) % num_experts
            for k in range(num_experts_per_tok):
                expert_idx = (start_expert + k) % num_experts
                router_logits = router_logits.at[token_idx, expert_idx].set(10.0)

        # Add small random noise for diversity (but keep assignment clear)
        noise = jax.random.normal(key, router_logits.shape, dtype=jnp.float32) * 0.1
        router_logits = router_logits + noise

    elif scenario == "imbalanced":
        # Create exponential distribution favoring first few experts
        # Adjust temperature to achieve target imbalance_factor

        # Start with exponential decay
        temperature = num_experts / (imbalance_factor * 2)  # Heuristic
        expert_base_logits = jnp.arange(num_experts, dtype=jnp.float32)
        expert_base_logits = 10.0 * jnp.exp(-expert_base_logits / temperature)

        # Broadcast to all tokens with random variation
        router_logits = jnp.tile(expert_base_logits, (num_tokens, 1))

        # Add random noise to create variation
        noise = jax.random.normal(key, router_logits.shape, dtype=jnp.float32) * 2.0
        router_logits = router_logits + noise

    else:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Must be one of: random, balanced, imbalanced"
        )

    return router_logits


def compute_imbalance_metrics(
    topk_ids: jax.Array,
    num_experts: int,
) -> Dict[str, float]:
    """
    Compute load imbalance metrics from expert assignments.

    Args:
        topk_ids: (num_tokens, num_experts_per_tok) expert indices
        num_experts: Total number of experts

    Returns:
        Dictionary containing:
            max_load: Maximum tokens assigned to any expert
            min_load: Minimum tokens assigned to any expert
            avg_load: Average tokens per expert
            std_load: Standard deviation of load
            max_imbalance: max_load / avg_load
            min_imbalance: min_load / avg_load
            load_distribution: Per-expert load counts (list)

    Example:
        If avg_load = 100 and max_load = 300, then max_imbalance = 3.0
        This means the busiest expert received 3x more tokens than average.
    """
    # Flatten topk_ids and count occurrences per expert
    flat_ids = topk_ids.flatten()
    expert_counts = jnp.bincount(flat_ids, length=num_experts)

    max_load = int(jnp.max(expert_counts))
    min_load = int(jnp.min(expert_counts))
    avg_load = float(jnp.mean(expert_counts))
    std_load = float(jnp.std(expert_counts))

    # Compute imbalance ratios
    max_imbalance = float(max_load / avg_load) if avg_load > 0 else float("inf")
    min_imbalance = float(min_load / avg_load) if avg_load > 0 and min_load > 0 else 0.0

    return {
        "max_load": max_load,
        "min_load": min_load,
        "avg_load": avg_load,
        "std_load": std_load,
        "max_imbalance": max_imbalance,
        "min_imbalance": min_imbalance,
        "load_distribution": expert_counts.tolist(),
    }


def create_hidden_states(
    num_tokens: int,
    hidden_size: int,
    dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 42,
) -> jax.Array:
    """
    Create synthetic input hidden states.

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension
        dtype: Data type
        seed: Random seed

    Returns:
        hidden_states: (num_tokens, hidden_size) array
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (num_tokens, hidden_size), dtype=dtype) * 0.02
