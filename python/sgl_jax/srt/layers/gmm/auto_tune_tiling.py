import functools
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.gmm.megablox_gmm_backend import gmm


class TilingAutoTuner:
    """Auto-tuner for megablox GMM tiling parameters."""

    def __init__(self, cache_dir: str = "/tmp/tune_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, m: int, k: int, n: int, num_groups: int) -> str:
        """Generate cache key for given problem size."""
        return f"m{m}_k{k}_n{n}_g{num_groups}"

    def _get_cache_file(self, cache_key: str) -> str:
        """Get cache file path for given cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_cached_result(self, cache_key: str) -> Optional[Tuple[int, int, int]]:
        """Load cached optimal tiling if available."""
        cache_file = self._get_cache_file(cache_key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    return tuple(data["optimal_tiling"])
            except Exception:
                pass
        return None

    def _save_cached_result(
        self, cache_key: str, optimal_tiling: Tuple[int, int, int], best_time: float
    ):
        """Save optimal tiling to cache."""
        cache_file = self._get_cache_file(cache_key)
        data = {
            "optimal_tiling": list(optimal_tiling),
            "best_time_ms": best_time * 1000,
            "timestamp": time.time(),
        }
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _create_test_data(
        self,
        m: int,
        k: int,
        n: int,
        num_groups: int,
        dtype: jnp.dtype = jnp.bfloat16,
        seed: int = 42,
    ):
        """Create test data for benchmarking."""
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 2)

        lhs = jax.random.normal(keys[0], (m, k), dtype=dtype)
        rhs = jax.random.normal(keys[1], (num_groups, k, n), dtype=dtype)
        group_sizes = jnp.array([m // num_groups] * num_groups, dtype=jnp.int32)

        return lhs, rhs, group_sizes

    def _benchmark_tiling(
        self,
        lhs,
        rhs,
        group_sizes,
        tiling: Tuple[int, int, int],
        num_warmup: int = 1,
        num_trials: int = 3,
    ) -> float:
        """Benchmark a specific tiling configuration."""

        @functools.partial(jax.jit, static_argnames=["tiling"])
        def jitted_gmm(lhs, rhs, group_sizes, tiling):
            return gmm(
                lhs, rhs, group_sizes, preferred_element_type=jnp.float32, tiling=tiling
            )

        # Warmup
        for _ in range(num_warmup):
            out = jitted_gmm(lhs, rhs, group_sizes, tiling)
            jax.block_until_ready(out)

        # Benchmark
        times = []
        for _ in range(num_trials):
            start = time.perf_counter()
            out = jitted_gmm(lhs, rhs, group_sizes, tiling)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - start)

        return np.mean(times)

    def _generate_tiling_candidates(
        self, m: int, k: int, n: int
    ) -> List[Tuple[int, int, int]]:
        """Generate candidate tiling configurations."""
        # Common tile sizes that work well on TPU
        tile_sizes = [64, 128, 256, 512, 1024, 2048]

        candidates = []

        # Add some heuristic-based candidates
        for tm in tile_sizes:
            if tm > m:
                continue
            for tk in tile_sizes:
                if tk > k:
                    continue
                for tn in tile_sizes:
                    if tn > n:
                        continue
                    candidates.append((tm, tk, tn))

        # Add default configuration
        default_tiling = (512, 1024, 1024)
        if default_tiling not in candidates:
            candidates.append(default_tiling)

        # Sort by preference (smaller tiles first for better memory usage)
        candidates.sort(key=lambda x: (x[0] * x[1] * x[2], x[0], x[1], x[2]))

        return candidates

    def tune_for_problem_size(
        self,
        m: int,
        k: int,
        n: int,
        num_groups: int,
        use_cache: bool = True,
        verbose: bool = True,
    ) -> Tuple[int, int, int]:
        """Tune tiling parameters for a specific problem size."""

        # Ensure problem size is valid
        if m % num_groups != 0:
            raise ValueError(f"m ({m}) must be divisible by num_groups ({num_groups})")

        cache_key = self._get_cache_key(m, k, n, num_groups)

        # Check cache first
        if use_cache:
            cached_result = self._load_cached_result(cache_key)
            if cached_result is not None:
                if verbose:
                    print(f"Using cached tiling for {cache_key}: {cached_result}")
                return cached_result

        if verbose:
            print(
                f"Tuning tiling for problem size: m={m}, k={k}, n={n}, groups={num_groups}"
            )

        # Create test data
        lhs, rhs, group_sizes = self._create_test_data(m, k, n, num_groups)

        # Generate candidate tilings
        candidates = self._generate_tiling_candidates(m, k, n)

        if verbose:
            print(f"Testing {len(candidates)} tiling candidates...")

        best_tiling = None
        best_time = float("inf")

        for i, tiling in enumerate(candidates):
            try:
                avg_time = self._benchmark_tiling(lhs, rhs, group_sizes, tiling)

                if verbose:
                    print(
                        f"  {i+1:2d}/{len(candidates)}: {tiling} -> {avg_time*1000:.2f} ms"
                    )

                if avg_time < best_time:
                    best_time = avg_time
                    best_tiling = tiling

            except Exception as e:
                if verbose:
                    print(f"  {i+1:2d}/{len(candidates)}: {tiling} -> ERROR: {e}")
                continue

        if best_tiling is None:
            # Fallback to default
            best_tiling = (512, 1024, 1024)
            if verbose:
                print(f"All tilings failed, using default: {best_tiling}")
        else:
            if verbose:
                print(f"Best tiling: {best_tiling} ({best_time*1000:.2f} ms)")

            # Save to cache
            if use_cache:
                self._save_cached_result(cache_key, best_tiling, best_time)

        return best_tiling

    def tune_for_moe_layer(
        self,
        typical_shapes: List[Tuple[int, int, int, int]],
        use_cache: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Tuple[int, int, int]]:
        """Tune tiling for typical MoE layer shapes."""
        results = {}

        for m, k, n, num_groups in typical_shapes:
            cache_key = self._get_cache_key(m, k, n, num_groups)
            optimal_tiling = self.tune_for_problem_size(
                m, k, n, num_groups, use_cache, verbose
            )
            results[cache_key] = optimal_tiling

        return results


def get_typical_moe_shapes() -> List[Tuple[int, int, int, int]]:
    """Get typical MoE shapes based on common model configurations."""
    # These are example shapes, you should adjust based on your actual usage
    shapes = [
        # (m, k, n, num_groups) - batch_size * seq_len, hidden_dim, intermediate_dim, num_experts
        (1024, 4096, 11008, 8),  # Llama-like model, batch=1, seq=1024
        (2048, 4096, 11008, 8),  # Llama-like model, batch=2, seq=1024
        (4096, 4096, 11008, 8),  # Llama-like model, batch=4, seq=1024
        (512, 5120, 13824, 8),  # Larger model
        (1024, 5120, 13824, 8),  # Larger model
        (2048, 5120, 13824, 8),  # Larger model
    ]
    return shapes


def main():
    """Main auto-tuning function."""
    tuner = TilingAutoTuner()

    # Get typical shapes for your use case
    typical_shapes = get_typical_moe_shapes()

    print("Starting auto-tuning for typical MoE shapes...")
    print("=" * 60)

    # Tune for all typical shapes
    results = tuner.tune_for_moe_layer(typical_shapes, use_cache=True, verbose=True)

    print("\n" + "=" * 60)
    print("TUNING RESULTS SUMMARY:")
    print("-" * 60)

    for shape_key, optimal_tiling in results.items():
        print(f"{shape_key:25} -> {optimal_tiling}")

    print(f"\nResults cached in: {tuner.cache_dir}/")
    print("You can now integrate these optimal tilings into your MoE layer!")


if __name__ == "__main__":
    main()
