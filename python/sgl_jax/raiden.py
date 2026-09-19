"""Early loader for the optional tpu-raiden runtime."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

_RAIDEN_NAMESPACES = ("tpu_sync", "tpu_raiden")


def raiden_requested(argv: Sequence[str] | None = None) -> bool:
    requested = False
    hicache_backend = "jax"
    args = list(sys.argv[1:] if argv is None else argv)
    for i, arg in enumerate(args):
        if arg == "--disaggregation-use-raiden":
            requested = True
        elif arg == "--no-disaggregation-use-raiden":
            requested = False
        elif arg.startswith("--hicache-transfer-backend="):
            hicache_backend = arg.split("=", 1)[1]
        elif arg == "--hicache-transfer-backend" and i + 1 < len(args):
            hicache_backend = args[i + 1]
    return requested or hicache_backend == "raiden"


def _loaded_namespace() -> str | None:
    for namespace in _RAIDEN_NAMESPACES:
        if f"{namespace}.frameworks.jax._tpu_raiden_jax" in sys.modules:
            return namespace
    return None


def get_raiden_kv_cache_manager():
    """Resolve the Python API from the same namespace as the loaded extension."""
    loaded = _loaded_namespace()
    namespaces = (loaded,) if loaded else _RAIDEN_NAMESPACES
    for namespace in namespaces:
        try:
            return importlib.import_module(f"{namespace}.api.jax.kv_cache_manager").KVCacheManager
        except ModuleNotFoundError as exc:
            # Only an absent top-level package permits fallback. Missing native
            # dependencies or an incomplete installed package must be reported.
            if exc.name != namespace:
                raise
    raise ModuleNotFoundError("Install a tpu-sync/tpu-raiden wheel matching JAX and libtpu")


def preload_raiden() -> None:
    """Preload the tpu-raiden native extension module.

    Must be called before importing JAX or jaxlib to ensure the native C++
    runtime extensions link and initialize properly before libtpu loads.
    """
    if _loaded_namespace():
        return
    if "jax" in sys.modules or "jaxlib" in sys.modules:
        raise RuntimeError("tpu-raiden must be preloaded before jax/jaxlib")
    try:
        for namespace in _RAIDEN_NAMESPACES:
            try:
                importlib.import_module(f"{namespace}.frameworks.jax._tpu_raiden_jax")
                return
            except ModuleNotFoundError as exc:
                if exc.name != namespace:
                    raise
        raise ModuleNotFoundError(
            "No tpu-sync/tpu-raiden package installed; install a wheel matching JAX and libtpu"
        )
    except ModuleNotFoundError:
        # Preserve the missing module name. An installed but incomplete wheel
        # (or a missing dependency) must not be reported as an absent package.
        raise
    except Exception as exc:  # pragma: no cover - native loader failure
        raise RuntimeError(
            "tpu-raiden failed to load; verify that its wheel matches JAX and libtpu"
        ) from exc


def preload_raiden_if_requested(argv: Sequence[str] | None = None) -> None:
    if raiden_requested(argv):
        preload_raiden()


def require_raiden_preloaded() -> None:
    if not _loaded_namespace():
        raise RuntimeError(
            "tpu-raiden was not preloaded. Use sgl_jax.launch_server or call "
            "sgl_jax.raiden.preload_raiden() before importing JAX."
        )
