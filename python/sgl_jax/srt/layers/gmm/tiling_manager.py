import json
import os
import threading
from typing import Dict, Optional, Tuple

import jax


def get_default_cache_dir() -> str:
    return os.environ.get("GMM_TUNE_CACHE_DIR", "/tmp/tune_cache")


class TilingManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, cache_dir: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_dir: Optional[str] = None):
        if self._initialized:
            return

        self.cache_dir = cache_dir or get_default_cache_dir()
        self.tiling_cache: Dict[str, Tuple[int, int, int]] = {}
        self.default_tiling = (512, 1024, 1024)
        self._load_all_cached_tilings()
        self._initialized = True

    def _get_cache_key(self, m: int, k: int, n: int, num_groups: int) -> str:
        return f"m{m}_k{k}_n{n}_g{num_groups}"

    def _load_all_cached_tilings(self):
        if not os.path.exists(self.cache_dir):
            return

        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                cache_key = filename[:-5]  # Remove .json extension
                cache_file = os.path.join(self.cache_dir, filename)

                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                        if "optimal_tiling" in data:
                            self.tiling_cache[cache_key] = tuple(data["optimal_tiling"])
                except Exception:
                    continue

    def get_optimal_tiling(
        self, m: int, k: int, n: int, num_groups: int
    ) -> Tuple[int, int, int]:
        cache_key = self._get_cache_key(m, k, n, num_groups)

        jax.debug.print(
            "[TilingManager] Looking for cache_key: {cache_key}",
            cache_key=cache_key,
        )
        jax.debug.print(
            "[TilingManager] Cache dir: {cache_dir}",
            cache_dir=self.cache_dir,
        )

        if cache_key in self.tiling_cache:
            jax.debug.print(
                "[TilingManager] Found exact match: {tiling}",
                tiling=self.tiling_cache[cache_key],
            )
            return self.tiling_cache[cache_key]

        # Try to find a close match with same k, n, num_groups but different m
        # This is common when batch size varies but model dimensions stay the same
        for cached_key, tiling in self.tiling_cache.items():
            parts = cached_key.split("_")
            if len(parts) == 4:
                try:
                    cached_m = int(parts[0][1:])  # Remove 'm' prefix
                    cached_k = int(parts[1][1:])  # Remove 'k' prefix
                    cached_n = int(parts[2][1:])  # Remove 'n' prefix
                    cached_groups = int(parts[3][1:])  # Remove 'g' prefix

                    if (
                        cached_k == k
                        and cached_n == n
                        and cached_groups == num_groups
                        and (
                            abs(cached_m - m) / max(cached_m, m) < 0.5
                            or min(cached_m, m) <= 256
                        )
                    ):  # Within 50% of cached m, or for small sizes use any available
                        jax.debug.print(
                            "[TilingManager] Found close match {cached_key}: {tiling}",
                            cached_key=cached_key,
                            tiling=tiling,
                        )
                        return tiling
                except ValueError:
                    continue

        # fallback to default
        jax.debug.print(
            "[TilingManager] No match found, using default: {default_tiling}",
            default_tiling=self.default_tiling,
        )
        jax.debug.print(
            "[TilingManager] Available cache keys: {cache_keys}",
            cache_keys=list(self.tiling_cache.keys()),
        )
        return self.default_tiling


_global_tiling_manager = None


def get_tiling_manager(cache_dir: Optional[str] = None) -> TilingManager:
    global _global_tiling_manager
    if _global_tiling_manager is None:
        _global_tiling_manager = TilingManager(cache_dir)
    return _global_tiling_manager


def get_optimal_tiling_for_gmm(
    m: int, k: int, n: int, num_groups: int = 1
) -> Tuple[int, int, int]:
    manager = get_tiling_manager()
    return manager.get_optimal_tiling(m, k, n, num_groups)
