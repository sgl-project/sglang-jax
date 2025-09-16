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
        tile_sizes = [64, 128, 256, 512, 1024, 2048]

        candidates = []

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

        default_tiling = (min(512, m), min(1024, k), min(1024, n))
        if default_tiling not in candidates:
            candidates.append(default_tiling)

        candidates.sort(key=lambda x: (x[0] * x[1] * x[2], x[0], x[1], x[2]))

        return candidates

    def tune_for_target_size(
        self,
        m: int,
        k: int,
        n: int,
        num_groups: int,
        use_cache: bool = True,
    ) -> Tuple[int, int, int]:
        if m % num_groups != 0:
            raise ValueError(f"m ({m}) must be divisible by num_groups ({num_groups})")

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

        for i, tiling in enumerate(candidates):
            try:
                avg_time = self._benchmark_tiling(lhs, rhs, group_sizes, tiling)
                if avg_time < best_time:
                    best_time = avg_time
                    best_tiling = tiling

            except Exception as e:
                failed_count += 1
                logger.debug(f"Tiling {tiling} failed: {type(e).__name__}: {e}")
                continue

        if best_tiling is None:
            best_tiling = (min(512, m), min(1024, k), min(1024, n))
            logger.warning(
                f"[GMM AUTO-TUNE] All {len(candidates)} tiling candidates failed for problem (m={m}, k={k}, n={n}, groups={num_groups}), using default {best_tiling}"
            )
        else:
            if failed_count > 0:
                logger.warning(
                    f"[GMM AUTO-TUNE] {failed_count}/{len(candidates)} tiling candidates failed for problem (m={m}, k={k}, n={n}, groups={num_groups})"
                )

            if use_cache:
                self._save_cached_result(cache_key, best_tiling, best_time)

        return best_tiling
