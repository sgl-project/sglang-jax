"""Early loader for the optional tpu-raiden runtime."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

_RAIDEN_EXTENSION = "tpu_raiden.frameworks.jax._tpu_raiden_jax"


def raiden_requested(argv: Sequence[str] | None = None) -> bool:
    requested = False
    for arg in sys.argv[1:] if argv is None else argv:
        if arg == "--disaggregation-use-raiden":
            requested = True
        elif arg == "--no-disaggregation-use-raiden":
            requested = False
    return requested


def preload_raiden() -> None:
    if _RAIDEN_EXTENSION in sys.modules:
        return
    if "jax" in sys.modules or "jaxlib" in sys.modules:
        raise RuntimeError("tpu-raiden must be preloaded before jax/jaxlib")
    try:
        importlib.import_module(_RAIDEN_EXTENSION)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tpu-raiden is not installed; install a wheel matching JAX and libtpu"
        ) from exc
    except Exception as exc:  # pragma: no cover - native loader failure
        raise RuntimeError(
            "tpu-raiden failed to load; verify that its wheel matches JAX and libtpu"
        ) from exc


def preload_raiden_if_requested(argv: Sequence[str] | None = None) -> None:
    if raiden_requested(argv):
        preload_raiden()


def require_raiden_preloaded() -> None:
    if _RAIDEN_EXTENSION not in sys.modules:
        raise RuntimeError(
            "tpu-raiden was not preloaded. Use sgl_jax.launch_server or call "
            "sgl_jax.raiden.preload_raiden() before importing JAX."
        )
