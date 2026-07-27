import os
import sys


def _raiden_requested() -> bool:
    requested = os.environ.get("SGLANG_JAX_USE_RAIDEN", "").lower() in {
        "1",
        "true",
        "yes",
    }
    for arg in sys.argv[1:]:
        if arg == "--disaggregation-use-raiden":
            requested = True
        elif arg == "--no-disaggregation-use-raiden":
            requested = False
    return requested


# Raiden embeds XLA and must load before jaxlib to avoid static-initializer
# collisions. Programmatic launchers opt in through SGLANG_JAX_USE_RAIDEN.
if _raiden_requested():
    if (
        "jax" in sys.modules or "jaxlib" in sys.modules
    ) and "tpu_raiden.frameworks.jax._tpu_raiden_jax" not in sys.modules:
        raise RuntimeError(
            "Raiden must be loaded before jax/jaxlib. Set "
            "SGLANG_JAX_USE_RAIDEN=1 before importing either package."
        )
    try:
        import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Raiden was selected but tpu_raiden is not installed. Install a "
            "tpu_raiden_jax wheel built for this JAX/libtpu version, or remove "
            "--disaggregation-use-raiden."
        ) from exc
    except Exception as exc:  # pragma: no cover - native loader failure
        raise RuntimeError(
            "tpu_raiden failed to load before jaxlib; verify that the Raiden "
            "wheel matches this JAX/libtpu build"
        ) from exc

from sgl_jax.version import __version__  # noqa: F401,E402
