"""Debug helpers for dumping intermediate arrays from inside jit.

Set ``SGL_DUMP_DIR`` to enable dumping; unset/empty disables it at trace time
so there is zero runtime overhead in production.

Each forward writes into a per-mode subdir, automatically rolled by a host
counter every time ``begin_forward`` is called::

    SGL_DUMP_DIR/
        prefill/
            embed_tokens_p0_i0000.npy
            layer_00_input_layernorm_p0_i0000.npy
            ...
        decode_1/
            ...
        decode_2/
            ...

Usage inside a jitted model::

    from sgl_jax.srt.utils.debug_utils import begin_forward, dump_array

    def __call__(self, forward_batch, ...):
        begin_forward("prefill" if forward_batch.forward_mode.is_prefill()
                      else "decode")
        x = self.embed(forward_batch.input_ids)
        dump_array("embed_tokens", x)
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
_tag_counter: dict[str, int] = {}
_mode_counter: dict[str, int] = {}
_current_subdir: str | None = None


def _dump_dir() -> Path | None:
    raw = os.environ.get(_DUMP_DIR_ENV, "").strip()
    return Path(raw) if raw else None


def _format_subdir(mode_kind: str, idx: int) -> str:
    # idx is 1-based. Match user-requested layout: "prefill" (no suffix on
    # first occurrence) and "decode_1", "decode_2", ...
    if mode_kind == "prefill" and idx == 1:
        return "prefill"
    return f"{mode_kind}_{idx}"


def _host_begin_forward(mode_kind: str) -> None:
    global _current_subdir
    with _lock:
        idx = _mode_counter.get(mode_kind, 0) + 1
        _mode_counter[mode_kind] = idx
        _tag_counter.clear()
        _current_subdir = _format_subdir(mode_kind, idx)


def _host_save(tag: str, summary: bool, array: np.ndarray) -> None:
    out_dir = _dump_dir()
    if out_dir is None:
        return
    sub = _current_subdir or "default"
    full_dir = out_dir / sub
    full_dir.mkdir(parents=True, exist_ok=True)
    with _lock:
        idx = _tag_counter.get(tag, 0)
        _tag_counter[tag] = idx + 1
    proc = jax.process_index()
    path = full_dir / f"{tag}_p{proc}_i{idx:04d}.npy"
    np.save(path, array)
    if summary:
        n_nan = int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0
        n_inf = int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0
        finite = np.isfinite(array) if np.issubdtype(array.dtype, np.floating) else None
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


def begin_forward(mode_kind: str, *, ordered: bool = True) -> None:
    """Roll the per-mode counter and switch to a fresh subdir for this forward.

    Call once at the top of the model's ``__call__``. ``mode_kind`` should be
    a short string like ``"prefill"`` or ``"decode"``. The host counter
    derives the actual subdir name (``prefill``, ``decode_1``, ``decode_2`` ...).

    No-op if ``SGL_DUMP_DIR`` is unset (decided at trace time).
    """
    if _dump_dir() is None:
        return
    jax.debug.callback(_host_begin_forward, mode_kind, ordered=ordered)


def dump_array(
    tag: str,
    array,
    *,
    summary: bool = True,
    gather: bool = False,
    ordered: bool = True,
) -> None:
    """Dump a (possibly sharded) array to disk from inside jit.

    Files land under ``$SGL_DUMP_DIR/<subdir>/<tag>_p<process>_i<NNNN>.npy``,
    where ``<subdir>`` is set by the most recent :func:`begin_forward` call.

    Args:
        tag: Logical name; used in the filename and to sequence repeated dumps.
        array: jax array. Sharded arrays dump per-process local shards by
            default; set ``gather=True`` to first replicate.
        summary: Print a one-line min/max/nan summary alongside the file write.
        gather: Replicate to a fully-replicated layout before dumping. Requires
            being inside a mesh context.
        ordered: Pass through to ``jax.debug.callback`` to preserve order.
    """
    if _dump_dir() is None:
        return

    if gather and hasattr(array, "sharding") and hasattr(array.sharding, "mesh"):
        replicated = jax.sharding.NamedSharding(
            array.sharding.mesh, jax.sharding.PartitionSpec()
        )
        array = jax.lax.with_sharding_constraint(array, replicated)

    array = jnp.asarray(array)
    jax.debug.callback(_host_save, tag, summary, array, ordered=ordered)


def reset_dump_state() -> None:
    """Reset all host-side counters and current subdir. For tests."""
    global _current_subdir
    with _lock:
        _tag_counter.clear()
        _mode_counter.clear()
        _current_subdir = None
