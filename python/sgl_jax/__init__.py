import os

# Engine-first XLA load: when the raiden data plane is enabled, import raiden's
# compiled JAX extension before anything imports jax/jaxlib. The extension
# embeds its own XLA runtime; loading it after jaxlib's libjax_common.so makes
# the two XLA copies collide in static initializers. This mirrors
# tpu-inference/tpu_inference/__init__.py. Gated so normal (path-A / non-PD)
# runs are unaffected, and guarded so a missing extension never breaks import.
if os.environ.get("SGLANG_JAX_USE_RAIDEN") == "1":
    try:
        import tpu_raiden.frameworks.jax._tpu_raiden_jax  # noqa: F401
    except ModuleNotFoundError:
        pass
    except Exception as _raiden_exc:  # pragma: no cover - best-effort preload
        import sys as _sys

        print(f"[tpu_raiden] engine preload failed: {_raiden_exc}", file=_sys.stderr)

from sgl_jax.version import __version__  # noqa: F401,E402
