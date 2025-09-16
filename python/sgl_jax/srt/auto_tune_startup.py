#!/usr/bin/env python3
"""
Auto-tuning system for SGL-JAX startup.

This module provides functionality to automatically tune tiling parameters
for MoE layers during service startup, optimizing performance for the
specific hardware and model configuration.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add the benchmark directory to sys.path to import auto_tune_tiling
benchmark_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "benchmark", "kernels", "megablox_gmm"
)
sys.path.insert(0, benchmark_dir)

from auto_tune_tiling import TilingAutoTuner

from sgl_jax.srt.layers.gmm.tiling_manager import get_tiling_manager


def get_model_specific_shapes(
    model_name: str, batch_sizes: List[int] = None, seq_lengths: List[int] = None
) -> List[Tuple[int, int, int, int]]:
    """Get typical shapes for specific model configurations."""

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]  # Common batch sizes

    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096]  # Common sequence lengths

    # Model-specific configurations
    model_configs = {
        "llama-7b": {"hidden_dim": 4096, "intermediate_dim": 11008, "num_experts": 8},
        "llama-13b": {"hidden_dim": 5120, "intermediate_dim": 13824, "num_experts": 8},
        "llama-30b": {"hidden_dim": 6656, "intermediate_dim": 17920, "num_experts": 8},
        "llama-65b": {"hidden_dim": 8192, "intermediate_dim": 22016, "num_experts": 8},
        "deepseek-v3": {
            "hidden_dim": 7168,
            "intermediate_dim": 18432,
            "num_experts": 160,
        },
        "mixtral-8x7b": {
            "hidden_dim": 4096,
            "intermediate_dim": 14336,
            "num_experts": 8,
        },
        "mixtral-8x22b": {
            "hidden_dim": 6144,
            "intermediate_dim": 16384,
            "num_experts": 8,
        },
    }

    if model_name not in model_configs:
        print(f"Warning: Unknown model '{model_name}', using default configuration")
        config = {"hidden_dim": 4096, "intermediate_dim": 11008, "num_experts": 8}
    else:
        config = model_configs[model_name]

    shapes = []
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            m = batch_size * seq_len
            k = config["hidden_dim"]
            n_gate = config["intermediate_dim"]
            n_down = config["hidden_dim"]  # down projection back to hidden_dim
            num_groups = config["num_experts"]

            # Add shapes for gate/up projections (k -> intermediate_dim)
            shapes.append((m, k, n_gate, num_groups))

            # Add shapes for down projections (intermediate_dim -> hidden_dim)
            shapes.append((m, n_gate, n_down, num_groups))

    return shapes


def auto_tune_for_startup(
    model_name: str = None,
    custom_shapes: List[Tuple[int, int, int, int]] = None,
    cache_dir: str = "tuning_cache",
    force_retune: bool = False,
    verbose: bool = True,
) -> Dict[str, Tuple[int, int, int]]:
    """
    Auto-tune tiling parameters for startup.

    Args:
        model_name: Name of the model to tune for (e.g., 'llama-7b', 'mixtral-8x7b')
        custom_shapes: Custom list of (m, k, n, num_groups) shapes to tune for
        cache_dir: Directory to store tuning cache
        force_retune: Whether to force re-tuning even if cache exists
        verbose: Whether to print detailed progress

    Returns:
        Dictionary mapping shape keys to optimal tiling parameters
    """

    tuner = TilingAutoTuner(cache_dir=cache_dir)

    if custom_shapes:
        shapes = custom_shapes
    elif model_name:
        shapes = get_model_specific_shapes(model_name)
    else:
        # Default shapes if no model specified
        shapes = get_model_specific_shapes("llama-7b")

    if verbose:
        print(f"Auto-tuning for {len(shapes)} shape configurations...")
        if model_name:
            print(f"Model: {model_name}")
        print(f"Cache directory: {cache_dir}")
        print("=" * 60)

    results = {}

    for i, (m, k, n, num_groups) in enumerate(shapes):
        if verbose:
            print(
                f"\nTuning {i+1}/{len(shapes)}: m={m}, k={k}, n={n}, groups={num_groups}"
            )

        try:
            # Skip if cache exists and not forcing retune
            cache_key = tuner._get_cache_key(m, k, n, num_groups)
            if not force_retune and not force_retune:
                cached_result = tuner._load_cached_result(cache_key)
                if cached_result is not None:
                    if verbose:
                        print(f"  Using cached result: {cached_result}")
                    results[cache_key] = cached_result
                    continue

            # Tune for this shape
            optimal_tiling = tuner.tune_for_problem_size(
                m, k, n, num_groups, use_cache=not force_retune, verbose=verbose
            )
            results[cache_key] = optimal_tiling

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            continue

    if verbose:
        print("\n" + "=" * 60)
        print("TUNING COMPLETED")
        print("-" * 60)
        for shape_key, optimal_tiling in results.items():
            print(f"{shape_key:30} -> {optimal_tiling}")
        print(f"\nResults cached in: {cache_dir}/")

    return results


def initialize_tiling_manager(cache_dir: str = "tuning_cache", verbose: bool = True):
    """Initialize the global tiling manager with tuning cache."""
    manager = get_tiling_manager(cache_dir)

    if verbose:
        cache_count = len(manager.tiling_cache)
        print(f"Initialized tiling manager with {cache_count} cached configurations")
        if cache_count > 0:
            print("Sample cached tilings:")
            for i, (key, tiling) in enumerate(list(manager.tiling_cache.items())[:5]):
                print(f"  {key:30} -> {tiling}")
            if cache_count > 5:
                print(f"  ... and {cache_count - 5} more")


def main():
    """Main CLI for auto-tuning."""
    parser = argparse.ArgumentParser(
        description="Auto-tune MoE tiling parameters for SGL-JAX"
    )

    parser.add_argument(
        "--model", type=str, help="Model name (e.g., llama-7b, mixtral-8x7b)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="tuning_cache",
        help="Directory to store tuning cache",
    )
    parser.add_argument(
        "--force-retune",
        action="store_true",
        help="Force re-tuning even if cache exists",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to tune for",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Sequence lengths to tune for",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    verbose = not args.quiet

    if verbose:
        print("SGL-JAX MoE Auto-Tuner")
        print("=" * 40)
        print(f"Model: {args.model or 'default'}")
        print(f"Cache directory: {args.cache_dir}")
        print(f"Force retune: {args.force_retune}")
        print()

    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)

    # Generate shapes for the specified model
    if args.model:
        shapes = get_model_specific_shapes(
            args.model, args.batch_sizes, args.seq_lengths
        )
    else:
        shapes = get_model_specific_shapes(
            "llama-7b", args.batch_sizes, args.seq_lengths
        )

    # Run auto-tuning
    start_time = time.time()
    results = auto_tune_for_startup(
        model_name=args.model,
        custom_shapes=shapes,
        cache_dir=args.cache_dir,
        force_retune=args.force_retune,
        verbose=verbose,
    )

    # Initialize tiling manager to verify everything works
    initialize_tiling_manager(args.cache_dir, verbose)

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"\nAuto-tuning completed in {elapsed_time:.1f} seconds")
        print(f"Tuned {len(results)} configurations")
        print(
            "\nYou can now start your SGL-JAX service with optimized tiling parameters!"
        )


if __name__ == "__main__":
    main()
