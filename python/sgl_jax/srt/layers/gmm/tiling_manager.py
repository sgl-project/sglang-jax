import json
import os
import threading
from typing import Dict, Optional, Tuple


def get_default_cache_dir() -> str:
    """Get the default cache directory from environment variable or fallback."""
    return os.environ.get("GMM_TUNE_CACHE_DIR", "tuning_cache")


class TilingManager:
    """Manages optimal tiling parameters for GMM operations."""

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
        """Generate cache key for given problem size."""
        return f"m{m}_k{k}_n{n}_g{num_groups}"

    def _get_cache_file(self, cache_key: str) -> str:
        """Get cache file path for given cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_all_cached_tilings(self):
        """Load all cached tiling results."""
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
        """Get optimal tiling for given problem size."""
        cache_key = self._get_cache_key(m, k, n, num_groups)

        # Check exact match first
        if cache_key in self.tiling_cache:
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
                        and abs(cached_m - m) / max(cached_m, m) < 0.5
                    ):  # Within 50% of cached m
                        return tiling
                except ValueError:
                    continue

        # Fallback to default
        return self.default_tiling

    def set_default_tiling(self, tiling: Tuple[int, int, int]):
        """Set the default tiling to use when no optimal tiling is found."""
        self.default_tiling = tiling

    def add_optimal_tiling(
        self, m: int, k: int, n: int, num_groups: int, tiling: Tuple[int, int, int]
    ):
        """Manually add an optimal tiling for a problem size."""
        cache_key = self._get_cache_key(m, k, n, num_groups)
        self.tiling_cache[cache_key] = tiling

    def get_adaptive_tiling(
        self, m: int, k: int, n: int, max_tile_size: Tuple[int, int, int] = None
    ) -> Tuple[int, int, int]:
        """Get adaptive tiling that doesn't exceed problem dimensions."""
        if max_tile_size is None:
            max_tile_size = self.default_tiling

        tm = min(max_tile_size[0], m)
        tk = min(max_tile_size[1], k)
        tn = min(max_tile_size[2], n)

        return (tm, tk, tn)


# Global tiling manager instance
_global_tiling_manager = None


def get_tiling_manager(cache_dir: Optional[str] = None) -> TilingManager:
    """Get the global tiling manager instance."""
    global _global_tiling_manager
    if _global_tiling_manager is None:
        _global_tiling_manager = TilingManager(cache_dir)
    return _global_tiling_manager


def get_optimal_tiling_for_gmm(
    m: int, k: int, n: int, num_groups: int = 1
) -> Tuple[int, int, int]:
    """Convenience function to get optimal tiling for GMM operation."""
    manager = get_tiling_manager()
    return manager.get_optimal_tiling(m, k, n, num_groups)
