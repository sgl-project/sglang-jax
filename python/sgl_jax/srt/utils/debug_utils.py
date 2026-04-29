"""Debug helpers for dumping intermediate arrays from inside jit.

Set ``SGL_DUMP_DIR`` to enable dumping; unset/empty disables it at trace time
so there is zero runtime overhead in production.

Each forward writes into a per-mode subdir (``prefill`` / ``decode_1`` /
``decode_2``...) auto-derived from a host-side per-tag counter. No explicit
"begin forward" call is required — each ``dump_array`` invocation carries
``forward_mode`` (or a mode-string) so the bucket can be computed independently.

Multi-device caveat: ``jax.debug.callback`` does not support ``ordered=True``
when the jit runs on more than one device, so callbacks fire **once per local
device** with no inter-callback ordering guarantee. The host computes
``forward_idx`` and ``shard_idx`` from the per-tag counter divided by the
local device count, which is robust to interleaving::

    SGL_DUMP_DIR/
        prefill/
            embed_tokens_p0_s0.npy        # shard 0
            embed_tokens_p0_s1.npy        # shard 1, ...
            layer_00_attn_out_p0_s0.npy
            ...
        decode_1/
            ...
        decode_2/
            ...

Usage inside a jitted model::

    from sgl_jax.srt.utils.debug_utils import dump_array

    def __call__(self, forward_batch, ...):
        x = self.embed(forward_batch.input_ids)
        dump_array("embed_tokens", x, forward_batch.forward_mode)
        ...
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_DUMP_DIR_ENV = "SGL_DUMP_DIR"
_lock = threading.Lock()
_tag_counter: dict[tuple[str, str], int] = {}
_local_dev_n: int | None = None
_warmup_complete: bool = False


def _dump_dir() -> Path | None:
    raw = os.environ.get(_DUMP_DIR_ENV, "").strip()
    return Path(raw) if raw else None


def _local_device_count() -> int:
    global _local_dev_n
    if _local_dev_n is None:
        try:
            _local_dev_n = max(1, jax.local_device_count())
        except Exception:
            _local_dev_n = 1
    return _local_dev_n


def _resolve_mode_kind(forward_mode) -> str:
    if forward_mode is None:
        return "default"
    if isinstance(forward_mode, str):
        return forward_mode
    if hasattr(forward_mode, "is_prefill"):
        return "prefill" if forward_mode.is_prefill() else "decode"
    return str(forward_mode)


def _format_subdir(mode_kind: str, forward_idx: int) -> str:
    # Match user-requested layout: "prefill" (no suffix on first occurrence)
    # and "decode_1", "decode_2", ...
    if mode_kind == "prefill" and forward_idx == 1:
        return "prefill"
    return f"{mode_kind}_{forward_idx}"


def _host_save(tag: str, mode_kind: str, summary: bool, array: np.ndarray) -> None:
    out_dir = _dump_dir()
    if out_dir is None:
        return
    if not _warmup_complete:
        # Precompile / warmup pass: callback fires but we drop the write so
        # the real first forward starts from idx 0 (subdir "prefill" /
        # "decode_1") instead of being shifted by the precompile count.
        return
    n = _local_device_count()
    with _lock:
        key = (tag, mode_kind)
        idx = _tag_counter.get(key, 0)
        _tag_counter[key] = idx + 1
    forward_idx = idx // n + 1
    shard_idx = idx % n
    sub = _format_subdir(mode_kind, forward_idx)
    full_dir = out_dir / sub
    full_dir.mkdir(parents=True, exist_ok=True)
    proc = jax.process_index()
    path = full_dir / f"{tag}_p{proc}_s{shard_idx}.npy"
    np.save(path, array)
    if summary:
        is_float = np.issubdtype(array.dtype, np.floating)
        n_nan = int(np.isnan(array).sum()) if is_float else 0
        n_inf = int(np.isinf(array).sum()) if is_float else 0
        finite = np.isfinite(array) if is_float else None
        if finite is not None and finite.any():
            vals = array[finite].astype(np.float32)
            stats = (
                f"min={vals.min():.6g} max={vals.max():.6g} "
                f"mean={vals.mean():.6g} absmax={np.abs(vals).max():.6g}"
            )
        elif finite is None:
            stats = f"int min={int(array.min())} max={int(array.max())}"
        else:
            stats = "all non-finite"
        print(
            f"[dump:{sub}] {path.name} shape={array.shape} dtype={array.dtype} "
            f"nan={n_nan} inf={n_inf} {stats}",
            flush=True,
        )


def dump_array(
    tag: str,
    array,
    forward_mode=None,
    *,
    summary: bool = True,
    gather: bool = False,
) -> None:
    """Dump a (possibly sharded) array to disk from inside jit.

    Files land under ``$SGL_DUMP_DIR/<subdir>/<tag>_p<process>_s<shard>.npy``.
    The subdir is derived from ``forward_mode`` plus a host counter that
    advances every time the same tag is dumped; once the counter wraps past
    the local device count, ``forward_idx`` increments and a new subdir is
    used.

    Args:
        tag: Logical name; used in the filename and to sequence dumps.
        array: jax array. Sharded arrays dump per-shard local data by default;
            set ``gather=True`` to first replicate (single content per shard).
        forward_mode: A ``ForwardMode`` enum (must expose ``is_prefill()``),
            a string like ``"prefill"`` / ``"decode"``, or ``None`` (bucket
            is named ``"default"``). Pass ``forward_batch.forward_mode`` from
            inside the model for automatic ``prefill`` / ``decode_N`` buckets.
        summary: Print a one-line min/max/nan summary per file.
        gather: Replicate to a fully-replicated layout before dumping.
            Requires being inside a mesh context.
    """
    if _dump_dir() is None:
        return

    mode_kind = _resolve_mode_kind(forward_mode)

    if gather and hasattr(array, "sharding") and hasattr(array.sharding, "mesh"):
        replicated = jax.sharding.NamedSharding(
            array.sharding.mesh, jax.sharding.PartitionSpec()
        )
        array = jax.lax.with_sharding_constraint(array, replicated)

    array = jnp.asarray(array)
    # ordered=False is required: ordered debug effects raise
    # `ValueError: ordered effects are not supported for more than 1 device`.
    jax.debug.callback(_host_save, tag, mode_kind, summary, array, ordered=False)


def reset_dump_state() -> None:
    """Reset all host-side counters. For tests."""
    global _warmup_complete
    with _lock:
        _tag_counter.clear()
        _warmup_complete = False


def mark_warmup_complete() -> None:
    """Signal that precompile / warmup is done. Until this is called,
    ``dump_array`` callbacks still fire but the host drops every write, so
    precompile forwards do not consume the ``prefill`` / ``decode_N`` slots.
    On call, per-tag counters are cleared so the first real forward starts
    at ``forward_idx = 1``.

    Hook point: call once at the end of ``tp_worker.run_precompile()``.
    Idempotent.
    """
    global _warmup_complete
    with _lock:
        _tag_counter.clear()
        _warmup_complete = True
