import functools
import itertools
import json
import os
import re
import threading
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
_DUMP_LOCK = threading.Lock()


def _sanitize_filename_part(value) -> str:
    text = str(value).replace(".", "_")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_") or "unknown"


def jax_debug_dump_enabled(component: str, layer_id: int | None = None) -> bool:
    """Return whether an opt-in JAX tensor dump is enabled for this site.

    The check intentionally happens while Python traces the model. With the
    default environment no callback is inserted into serving executables.
    """

    if os.environ.get("SGLANG_JAX_DEBUG_DUMP", "0").lower() not in {
        "1",
        "true",
        "on",
        "yes",
    }:
        return False

    components = os.environ.get("SGLANG_JAX_DEBUG_DUMP_COMPONENTS")
    if components:
        allowed = {_sanitize_filename_part(item) for item in components.split(",")}
        if _sanitize_filename_part(component) not in allowed:
            return False

    layers = os.environ.get("SGLANG_JAX_DEBUG_DUMP_LAYERS")
    if layers and layer_id is not None:
        allowed_layers = {int(item) for item in layers.split(",") if item.strip()}
        if layer_id not in allowed_layers:
            return False
    return True


def maybe_dump_jax_array(
    array: jax.Array,
    *,
    component: str,
    name: str,
    layer_id: int | None = None,
    forward_mode=None,
) -> None:
    """Save a staged JAX array and a JSONL index through a debug callback.

    This follows the proven callback approach from PR #1062, but is disabled by
    default and writes a machine-readable manifest for golden comparisons.
    """

    if not jax_debug_dump_enabled(component, layer_id):
        return

    dump_dir = os.environ.get("SGLANG_JAX_DEBUG_DUMP_DIR", "debug_dumps")
    process_id = jax.process_index()
    mode_name = getattr(forward_mode, "name", forward_mode)

    def _save_to_file(host_array):
        host_array = np.asarray(host_array)
        with _DUMP_LOCK:
            index = next(_DUMP_COUNTER)
            parts = [f"p{process_id:05d}", f"{index:06d}", _sanitize_filename_part(component)]
            if layer_id is not None:
                parts.append(f"layer{layer_id:03d}")
            if mode_name is not None:
                parts.append(_sanitize_filename_part(mode_name).lower())
            parts.append(_sanitize_filename_part(name))
            filename = "_".join(parts) + ".npy"

            os.makedirs(dump_dir, exist_ok=True)
            np.save(os.path.join(dump_dir, filename), host_array)
            record = {
                "index": index,
                "process_id": process_id,
                "component": component,
                "name": name,
                "layer_id": layer_id,
                "forward_mode": None if mode_name is None else str(mode_name).lower(),
                "shape": list(host_array.shape),
                "dtype": str(host_array.dtype),
                "filename": filename,
            }
            manifest = os.path.join(dump_dir, f"manifest-p{process_id:05d}.jsonl")
            with open(manifest, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")

    jax.debug.callback(_save_to_file, array, ordered=True)


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
