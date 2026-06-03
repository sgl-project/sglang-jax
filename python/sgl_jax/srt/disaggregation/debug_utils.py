"""Debug-only KV diagnostics for PD disaggregation.

These helpers are intentionally environment-gated and side-effect free
so we can turn them on in manual TPU investigations without perturbing
the steady-state PD path by default.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import jax
import numpy as np


_ENV_ENABLE = "SGL_JAX_PD_DEBUG_KV"
_ENV_REQ_FILTER = "SGL_JAX_PD_DEBUG_REQ_ID"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def kv_debug_enabled(req_id: str | None = None) -> bool:
    if os.environ.get(_ENV_ENABLE, "").strip().lower() not in _TRUTHY:
        return False
    req_filter = os.environ.get(_ENV_REQ_FILTER, "").strip()
    if not req_filter:
        return True
    if req_id is None:
        return False
    return req_filter in req_id


def safe_sharding_repr(value) -> str:
    sharding = getattr(value, "sharding", value)
    if sharding is None:
        return "None"
    try:
        return str(sharding)
    except Exception as exc:  # noqa: BLE001
        return f"<unrepr {type(sharding).__name__}: {exc}>"


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:16]


def _host_array(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    shards = list(getattr(value, "addressable_shards", []))
    if shards:
        try:
            host = np.empty(tuple(int(d) for d in value.shape), dtype=value.dtype)
            for shard in shards:
                host[shard.index] = np.asarray(jax.device_get(shard.data))
            return host
        except Exception:  # noqa: BLE001
            pass

    return np.asarray(jax.device_get(value))


@dataclass(frozen=True)
class KVDebugSnapshot:
    shape: tuple[int, ...]
    dtype: str
    sharding: str
    global_digest: str
    page_digests: tuple[tuple[str, ...], ...]

    def sample_page_digests(
        self, *, max_layers: int = 2, max_pages: int = 4
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(
            row[:max_pages] for row in self.page_digests[:max_layers]
        )


def build_kv_debug_snapshot(value) -> KVDebugSnapshot:
    host = _host_array(value)
    if host.ndim < 2:
        raise ValueError(
            "build_kv_debug_snapshot expects an array with at least 2 "
            "dims (layer, page, ...)"
        )

    page_digests = []
    for layer_idx in range(host.shape[0]):
        row = []
        for page_idx in range(host.shape[1]):
            row.append(_digest_bytes(host[layer_idx, page_idx].tobytes()))
        page_digests.append(tuple(row))

    return KVDebugSnapshot(
        shape=tuple(int(d) for d in host.shape),
        dtype=str(host.dtype),
        sharding=safe_sharding_repr(value),
        global_digest=_digest_bytes(host.tobytes()),
        page_digests=tuple(page_digests),
    )


def count_kv_debug_mismatches(
    left: KVDebugSnapshot, right: KVDebugSnapshot
) -> int:
    _validate_snapshot_shapes(left, right)
    mismatches = 0
    for left_row, right_row in zip(left.page_digests, right.page_digests):
        for left_digest, right_digest in zip(left_row, right_row):
            mismatches += int(left_digest != right_digest)
    return mismatches


def find_first_kv_debug_mismatch(
    left: KVDebugSnapshot, right: KVDebugSnapshot
) -> tuple[int, int] | None:
    _validate_snapshot_shapes(left, right)
    for layer_idx, (left_row, right_row) in enumerate(
        zip(left.page_digests, right.page_digests)
    ):
        for page_idx, (left_digest, right_digest) in enumerate(
            zip(left_row, right_row)
        ):
            if left_digest != right_digest:
                return (layer_idx, page_idx)
    return None


def _validate_snapshot_shapes(
    left: KVDebugSnapshot, right: KVDebugSnapshot
) -> None:
    if len(left.page_digests) != len(right.page_digests):
        raise ValueError(
            "snapshot layer counts differ: "
            f"{len(left.page_digests)} != {len(right.page_digests)}"
        )
    for left_row, right_row in zip(left.page_digests, right.page_digests):
        if len(left_row) != len(right_row):
            raise ValueError(
                "snapshot page counts differ within a layer: "
                f"{len(left_row)} != {len(right_row)}"
            )
