import functools
import json
import logging
import os
import time
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.gmm.megablox_gmm_backend import gmm

logger = logging.getLogger(__name__)


class TilingAutoTuner:
    def __init__(self, cache_dir: str = "/tmp/tune_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, m: int, k: int, n: int, num_groups: int) -> str:
        return f"m{m}_k{k}_n{n}_g{num_groups}"

    def _get_cache_file(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_cached_result(self, cache_key: str) -> Optional[Tuple[int, int, int]]:
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
        @functools.partial(jax.jit, static_argnames=["tiling"])
        def jitted_gmm(lhs, rhs, group_sizes, tiling):
            return gmm(
                lhs, rhs, group_sizes, preferred_element_type=jnp.float32, tiling=tiling
            )

        # Warmup
        for _ in range(num_warmup):
            out = jitted_gmm(lhs, rhs, group_sizes, tiling)
            jax.block_until_ready(out)

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
        # Use conservative tile sizes that work well with TPU constraints
        tile_sizes_m = [
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
        ]  # m can be small for decode
        tile_sizes_k = [128, 256, 512, 1024, 2048]  # k should be >= 128 for TPU
        tile_sizes_n = [128, 256, 512, 1024, 2048]  # n should be >= 128 for TPU

        candidates = []

        for tm in tile_sizes_m:
            if tm > m:
                continue
            for tk in tile_sizes_k:
                if tk > k:
                    continue
                for tn in tile_sizes_n:
                    if tn > n:
                        continue

                    # GMM constraint: dimensions must be divisible by tile sizes
                    if m % tm != 0 or k % tk != 0 or n % tn != 0:
                        continue

                    # TPU constraints: check effective dimensions (min of tile_size and actual dimension)
                    effective_tk = min(tk, k)
                    effective_tn = min(tn, n)

                    # TPU requires: k dimension divisible by 8, n dimension divisible by 128
                    if effective_tk % 8 != 0 or effective_tn % 128 != 0:
                        continue

                    candidates.append((tm, tk, tn))

        # Generate valid default tiling that satisfies both GMM and TPU constraints
        default_tm = 8  # Start with small value for decode compatibility
        default_tk = 128  # Start with TPU-safe minimum
        default_tn = 128  # Start with TPU-safe minimum

        # Find the largest tm that divides m (including smaller values for decode)
        for tm in tile_sizes_m:
            if tm <= m and m % tm == 0:
                default_tm = tm

        # Find the largest tk that divides k and meets TPU constraints
        for tk in reversed(tile_sizes_k):  # Try larger values first
            if tk <= k and k % tk == 0:
                default_tk = tk
                break

        # Find the largest tn that divides n and meets TPU constraints
        for tn in reversed(tile_sizes_n):  # Try larger values first
            if tn <= n and n % tn == 0:
                default_tn = tn
                break

        default_tiling = (default_tm, default_tk, default_tn)
        if default_tiling not in candidates and all(d > 0 for d in default_tiling):
            candidates.append(default_tiling)

        candidates.sort(key=lambda x: (x[0] * x[1] * x[2], x[0], x[1], x[2]))

        return candidates

    def _format_failure_summary(self, failure_reasons: dict) -> str:
        """Format failure reasons into a readable summary."""
        if not failure_reasons:
            return "None"

        summary_parts = []
        for error_type, details in failure_reasons.items():
            count = details["count"]
            examples = details["examples"]
            if count == 1 and examples:
                summary_parts.append(f"{error_type}(1): {examples[0]}")
            else:
                example_str = f" e.g. {examples[0]}" if examples else ""
                summary_parts.append(f"{error_type}({count}){example_str}")

        return "; ".join(summary_parts)

    def tune_for_target_size(
        self,
        m: int,
        k: int,
        n: int,
        num_groups: int,
        use_cache: bool = True,
    ) -> Tuple[int, int, int]:
        # Note: GMM requires m % tile_m == 0, k % tile_k == 0, n % tile_n == 0
        # These constraints will be checked during candidate generation

        cache_key = self._get_cache_key(m, k, n, num_groups)

        if use_cache:
            cached_result = self._load_cached_result(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached tiling for {cache_key}: {cached_result}")
                return cached_result

        logger.debug(
            f"Tuning tiling for problem size: m={m}, k={k}, n={n}, groups={num_groups}"
        )

        lhs, rhs, group_sizes = self._create_test_data(m, k, n, num_groups)

        candidates = self._generate_tiling_candidates(m, k, n)

        best_tiling = None
        best_time = float("inf")
        failed_count = 0
        failure_reasons = {}  # Track failure reasons

        for i, tiling in enumerate(candidates):
            try:
                avg_time = self._benchmark_tiling(lhs, rhs, group_sizes, tiling)
                if avg_time < best_time:
                    best_time = avg_time
                    best_tiling = tiling

            except Exception as e:
                failed_count += 1
                error_type = type(e).__name__
                error_msg = str(e)
                # Group similar errors
                if error_type not in failure_reasons:
                    failure_reasons[error_type] = {"count": 0, "examples": []}
                failure_reasons[error_type]["count"] += 1
                if (
                    len(failure_reasons[error_type]["examples"]) < 3
                ):  # Keep max 3 examples
                    failure_reasons[error_type]["examples"].append(
                        f"{tiling}: {error_msg}"
                    )
                logger.debug(f"Tiling {tiling} failed: {error_type}: {error_msg}")
                continue

        if best_tiling is None:
            # Generate valid fallback tiling that satisfies both GMM and TPU constraints
            fallback_tm = 8  # Start with small value for decode compatibility
            fallback_tk = 128  # Start with TPU-safe minimum
            fallback_tn = 128  # Start with TPU-safe minimum

            # Define tile sizes (same as in _generate_tiling_candidates)
            tile_sizes_m = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            tile_sizes_k = [128, 256, 512, 1024, 2048]
            tile_sizes_n = [128, 256, 512, 1024, 2048]

            # Find the largest tm that divides m (including smaller values for decode)
            for tm in tile_sizes_m:
                if tm <= m and m % tm == 0:
                    fallback_tm = tm

            # Find the largest tk that divides k and meets TPU constraints
            for tk in reversed(tile_sizes_k):  # Try larger values first
                if tk <= k and k % tk == 0:
                    fallback_tk = tk
                    break

            # Find the largest tn that divides n and meets TPU constraints
            for tn in reversed(tile_sizes_n):  # Try larger values first
                if tn <= n and n % tn == 0:
                    fallback_tn = tn
                    break

            best_tiling = (fallback_tm, fallback_tk, fallback_tn)
            failure_summary = self._format_failure_summary(failure_reasons)
            logger.warning(
                f"[GMM AUTO-TUNE] All {len(candidates)} tiling candidates failed for problem (m={m}, k={k}, n={n}, groups={num_groups}), using default {best_tiling}. "
                f"Failure reasons: {failure_summary}"
            )
        else:
            if failed_count > 0:
                failure_summary = self._format_failure_summary(failure_reasons)
                logger.warning(
                    f"[GMM AUTO-TUNE] {failed_count}/{len(candidates)} tiling candidates failed for problem (m={m}, k={k}, n={n}, groups={num_groups}). "
                    f"Failure reasons: {failure_summary}"
                )

            if use_cache:
                self._save_cached_result(cache_key, best_tiling, best_time)

        return best_tiling
