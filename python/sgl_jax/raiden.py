"""Early loader for the optional tpu-raiden runtime."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Any

# New wheels use tpu_sync; older wheels retain the tpu_raiden namespace.
_RAIDEN_NAMESPACES = ("tpu_sync", "tpu_raiden")
_EXTENSION_SUFFIX = ".frameworks.jax._tpu_raiden_jax"


def _preloaded_namespace() -> str | None:
    loaded = [name for name in _RAIDEN_NAMESPACES if name + _EXTENSION_SUFFIX in sys.modules]
    if len(loaded) > 1:
        raise RuntimeError("Both tpu_sync and tpu_raiden native extensions are loaded")
    return loaded[0] if loaded else None


def raiden_requested(argv: Sequence[str] | None = None) -> bool:
    requested = False
    for arg in sys.argv[1:] if argv is None else argv:
        if arg == "--disaggregation-use-raiden":
            requested = True
        elif arg == "--no-disaggregation-use-raiden":
            requested = False
    return requested


def preload_raiden() -> None:
    if _preloaded_namespace() is not None:
        return
    if "jax" in sys.modules or "jaxlib" in sys.modules:
        raise RuntimeError("tpu-raiden must be preloaded before jax/jaxlib")
    for namespace in _RAIDEN_NAMESPACES:
        try:
            importlib.import_module(namespace + _EXTENSION_SUFFIX)
            return
        except ModuleNotFoundError as exc:
            # Only an absent top-level package permits fallback. Missing internal
            # modules/dependencies and native ABI failures must remain visible.
            if exc.name != namespace:
                raise
    raise ModuleNotFoundError(
        "Neither tpu_sync nor tpu_raiden is installed; install a wheel matching JAX and libtpu"
    )


def preload_raiden_if_requested(argv: Sequence[str] | None = None) -> None:
    if raiden_requested(argv):
        preload_raiden()


def require_raiden_preloaded() -> None:
    if _preloaded_namespace() is None:
        raise RuntimeError(
            "tpu-raiden was not preloaded. Use sgl_jax.launch_server or call "
            "sgl_jax.raiden.preload_raiden() before importing JAX."
        )


def get_raiden_kv_cache_manager() -> Any:
    """Load the public API from the same namespace as the preloaded extension."""
    require_raiden_preloaded()
    namespace = _preloaded_namespace()
    return importlib.import_module(f"{namespace}.api.jax.kv_cache_manager").KVCacheManager
