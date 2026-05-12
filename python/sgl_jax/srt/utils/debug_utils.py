import functools
import itertools
import os
import re
from enum import IntEnum

import jax
import numpy as np


class FrameworkLogLevel(IntEnum):
    ERROR = 0
    WARN = 1
    INFO = 2
    DEBUG = 3
    TRACE = 4


FRAMEWORK_LOG_LEVEL = FrameworkLogLevel(int(os.environ.get("SGLANG_FRAMEWORK_LOG_LEVEL", "0")))

_DUMP_COUNTER = itertools.count()


def _sanitize_filename_part(value) -> str:
    text = str(value).replace(".", "_")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_") or "unknown"


def save_jax_array(jax_arr: jax.Array, filename: str) -> None:
    """Save a staged JAX array through the host-side debug callback path."""

    dump_dir = os.environ.get("SGLANG_JAX_DEBUG_DUMP_DIR", "debug_dumps")
    process_id = jax.process_index()
    filename = _sanitize_filename_part(filename)
    stem, ext = os.path.splitext(filename)
    ext = ext or ".npy"

    def _save_to_file(arr):
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"p{process_id:05d}_{stem}_{next(_DUMP_COUNTER):06d}{ext}")
        np.save(path, np.asarray(arr))

    jax.debug.callback(_save_to_file, jax_arr)


def maybe_dump_jax_array(
    jax_arr: jax.Array,
    *,
    component: str,
    name: str,
    layer_id: int | None = None,
    forward_mode=None,
) -> None:
    """Dump a JAX array for Ling/Bailing debug instrumentation.

    Enabled by default on this debug branch. Set ``SGLANG_JAX_DEBUG_DUMP=0`` to
    skip dumps, or use ``SGLANG_JAX_DEBUG_DUMP_COMPONENTS`` /
    ``SGLANG_JAX_DEBUG_DUMP_LAYERS`` to narrow the output.
    """

    if os.environ.get("SGLANG_JAX_DEBUG_DUMP", "1").lower() in {"0", "false", "off", "no"}:
        return

    components = os.environ.get("SGLANG_JAX_DEBUG_DUMP_COMPONENTS")
    if components:
        allowed_components = {_sanitize_filename_part(item) for item in components.split(",")}
        if _sanitize_filename_part(component) not in allowed_components:
            return

    layers = os.environ.get("SGLANG_JAX_DEBUG_DUMP_LAYERS")
    if layers and layer_id is not None:
        allowed_layers = {int(item) for item in layers.split(",") if item.strip()}
        if layer_id not in allowed_layers:
            return

    mode_name = getattr(forward_mode, "name", forward_mode)
    parts = [_sanitize_filename_part(component)]
    if layer_id is not None:
        parts.append(f"layer{layer_id:03d}")
    if mode_name is not None:
        parts.append(_sanitize_filename_part(mode_name).lower())
    parts.append(_sanitize_filename_part(name))
    save_jax_array(jax_arr, f"{'_'.join(parts)}.npy")


def print_parameter_shardings(model):
    if FRAMEWORK_LOG_LEVEL < FrameworkLogLevel.DEBUG:
        return
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.value.shape} sharding={param.value.sharding}")


def log_shardings(name):
    def decorator(fn):
        if FRAMEWORK_LOG_LEVEL < FrameworkLogLevel.DEBUG:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for i, a in enumerate(args):
                if hasattr(a, "aval") and hasattr(a.aval, "sharding"):
                    print(f"{name} input[{i}]: {a.aval.shape} {a.aval.sharding}")
            result = fn(*args, **kwargs)
            if hasattr(result, "aval") and hasattr(result.aval, "sharding"):
                print(f"{name} output: {result.aval.shape} {result.aval.sharding}")
            elif isinstance(result, tuple):
                for i, r in enumerate(result):
                    if hasattr(r, "aval") and hasattr(r.aval, "sharding"):
                        print(f"{name} output[{i}]: {r.aval.shape} {r.aval.sharding}")
            return result

        return wrapper

    return decorator
