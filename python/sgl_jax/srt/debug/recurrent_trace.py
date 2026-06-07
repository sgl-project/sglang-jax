"""Env-gated recurrent-state trace helpers for overlap debugging."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_LOCK = threading.Lock()
_RID_INDEX_RE = re.compile(r"(?:^|[-_])(\d+)$")


def enabled() -> bool:
    value = os.environ.get("SGLANG_DBG_RECUR_TRACE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def trace_path() -> Path:
    return Path(os.environ.get("SGLANG_DBG_RECUR_TRACE_OUT", "/tmp/recur_trace.jsonl"))


def rid_prefixes() -> tuple[str, ...]:
    raw = os.environ.get("SGLANG_DBG_RECUR_TRACE_RID_PREFIX", "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def should_trace_rid(rid: Any) -> bool:
    if not enabled() or rid is None:
        return False
    rid_str = str(rid)
    prefixes = rid_prefixes()
    return not prefixes or any(rid_str.startswith(prefix) for prefix in prefixes)


def request_index_from_rid(rid: Any) -> int | None:
    match = _RID_INDEX_RE.search(str(rid))
    return int(match.group(1)) if match else None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def layer_indices(num_layers: int) -> list[int]:
    raw = os.environ.get("SGLANG_DBG_RECUR_TRACE_LAYERS", "0")
    if raw.strip().lower() in {"all", "*"}:
        return list(range(num_layers))
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            continue
        if 0 <= idx < num_layers:
            out.append(idx)
    return out or [0]


def safe_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list | tuple):
        return [safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    return value


def write_event(event: str, **payload: Any) -> None:
    if not enabled():
        return
    record = {
        "ts": time.time(),
        "event": event,
        "pid": os.getpid(),
        "node_rank": os.environ.get("NODE_RANK") or os.environ.get("RANK"),
        **payload,
    }
    path = trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(safe_json_value(record), sort_keys=True, ensure_ascii=False)
    with _LOCK, path.open("a") as f:
        f.write(line + "\n")


def digest_arrays_for_indices(buffers: list[Any], indices: list[int], layers: list[int]) -> Any:
    """Return a small JAX array [layer, row, metric] without host synchronization."""
    if not buffers or not indices or not layers:
        return None
    idx = jnp.asarray(indices, dtype=jnp.int32)
    layer_digests = []
    for layer in layers:
        selected = buffers[layer][idx].astype(jnp.float32)
        axes = tuple(range(1, selected.ndim))
        abs_selected = jnp.abs(selected)
        layer_digests.append(
            jnp.stack(
                [
                    jnp.sum(selected, axis=axes),
                    jnp.sum(abs_selected, axis=axes),
                    jnp.max(abs_selected, axis=axes),
                    jnp.mean(selected, axis=axes),
                ],
                axis=-1,
            )
        )
    return jnp.stack(layer_digests, axis=0)


def materialize(value: Any) -> Any:
    if value is None:
        return None
    return jax.device_get(value)
