"""Pure index math for decode context parallel (DCP).

Port of CUDA SGLang ``sglang.srt.layers.dcp.layout`` owner rule. Host code uses
numpy; device code can call the same formulas with ``jax.numpy``.

Virtual token ``v`` (scheduler / ``out_cache_loc`` when ``dcp_size > 1``) is
**block-interleaved** with block width ``I = interleave``:

    owner(v)     = (v // I) % dcp_size
    physical(v)  = (v // (I * dcp_size)) * I + (v % I)

``I=1`` is the token-striped CUDA rule (``v % dcp_size`` / ``v // dcp_size``).
Production uses ``I = page_size``, which makes one **DSA page** (``page_size``
consecutive virtual tokens, the indexer's max-pool unit) land entirely on rank
``d % dcp_size`` as one whole physical page ``d // dcp_size``. That is what lets
the existing page-granular Pallas kernels read a selected page unchanged; at
``I=1`` a selected page is split ``page_size/dcp_size`` ways and the resulting
read is sub-16 and illegal. Measured 2.1-3.1x faster than ``I=1``.

``I`` does **not** change the allocator: one virtual page is ``page_size *
dcp_size`` tokens and still maps to one physical page of ``page_size`` on every
rank. Only the permutation inside that virtual page differs, so
``virtual_page_size`` and ``physical_page_indices`` are ``I``-invariant.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.integer]


def _check_interleave(interleave: int) -> None:
    if interleave < 1:
        raise ValueError(f"interleave must be >= 1, got {interleave}")


def owner(virtual_pos: int | IntArray, dcp_size: int, interleave: int = 1) -> int | IntArray:
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    _check_interleave(interleave)
    if interleave == 1:
        return virtual_pos % dcp_size
    return (virtual_pos // interleave) % dcp_size


def physical_index(
    virtual_pos: int | IntArray, dcp_size: int, interleave: int = 1
) -> int | IntArray:
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    _check_interleave(interleave)
    if interleave == 1:
        return virtual_pos // dcp_size
    return (virtual_pos // (interleave * dcp_size)) * interleave + (virtual_pos % interleave)


def virtual_index(
    physical_pos: int | IntArray,
    dcp_size: int,
    dcp_rank: int,
    interleave: int = 1,
) -> int | IntArray:
    """Inverse of ``(owner, physical_index)`` for the rank that owns the token."""
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank must be in [0, {dcp_size}), got {dcp_rank}")
    _check_interleave(interleave)
    if interleave == 1:
        return physical_pos * dcp_size + dcp_rank
    block, off = physical_pos // interleave, physical_pos % interleave
    return (block * dcp_size + dcp_rank) * interleave + off


def physical_write_loc(
    virtual_loc: int | IntArray,
    dcp_size: int,
    dcp_rank: int,
    interleave: int = 1,
) -> int | IntArray:
    """Physical scatter index for this rank, or -1 if padded / not owned.

    Matches the prefill kernels' drop rule (``wrap_negative_indices=False``):
    padding (``virtual_loc < 0``) and tokens owned by another rank become -1.
    ``dcp_size=1`` is identity (only padding stays -1) for any ``interleave``.
    """
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank must be in [0, {dcp_size}), got {dcp_rank}")
    _check_interleave(interleave)
    loc = np.asarray(virtual_loc)
    padded = loc < 0
    safe = np.where(padded, 0, loc)
    drop = padded | (owner(safe, dcp_size, interleave) != dcp_rank)
    out = np.where(drop, -1, physical_index(safe, dcp_size, interleave))
    if np.ndim(virtual_loc) == 0:
        return int(np.asarray(out).reshape(()))
    return out.astype(np.int32, copy=False)


def physical_page_indices(
    virtual_cache_loc: IntArray,
    page_size: int,
    dcp_size: int,
) -> IntArray:
    """Physical page ids from virtual ``cache_loc`` (MLA metadata table).

    Strides by the virtual page (``page_size * dcp_size``) then maps the first
    token of each virtual page to ``physical_index // page_size``.
    ``dcp_size=1`` is ``cache_loc[::page_size] // page_size``.
    """
    vpage = virtual_page_size(page_size, dcp_size)
    loc = np.asarray(virtual_cache_loc)
    if loc.ndim == 1:
        strided = loc[::vpage]
    elif loc.ndim == 2:
        strided = loc[:, ::vpage]
    else:
        raise ValueError(f"virtual_cache_loc must be 1-D or 2-D, got shape {loc.shape}")
    return (physical_index(strided, dcp_size) // page_size).astype(np.int32)


def virtual_page_size(page_size: int, dcp_size: int) -> int:
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    return page_size * dcp_size


def _owned_count(upto: int | IntArray, dcp_size: int, dcp_rank: int, interleave: int) -> IntArray:
    """Number of virtual tokens in ``[0, upto)`` owned by ``dcp_rank``.

    Blocks of ``interleave`` tokens round-robin over ranks, so the rank owns
    whole blocks plus (only if it owns the final partial block) its remainder.
    """
    x = np.asarray(upto)
    full, rem = x // interleave, x % interleave
    whole = (full - dcp_rank + dcp_size - 1) // dcp_size  # >= 0: dcp_rank <= dcp_size-1
    tail = np.where((full % dcp_size) == dcp_rank, rem, 0)
    return whole * interleave + tail


def get_dcp_lens(
    lens: int | IntArray,
    dcp_size: int,
    dcp_rank: int,
    start: int | IntArray | None = None,
    interleave: int = 1,
) -> int | IntArray:
    """Per-rank visible KV length under ``owner(pos) == dcp_rank``.

    ``start is None`` is the CUDA ``update_local_kv_lens_for_dcp`` case; at
    ``interleave=1`` that is ``lens // N + (rank < lens % N)``.
    """
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank must be in [0, {dcp_size}), got {dcp_rank}")
    _check_interleave(interleave)
    if dcp_size == 1:
        return lens
    if interleave > 1:
        # count over the window, as a difference of prefix counts
        lo = 0 if start is None else start
        return _owned_count(lo + lens, dcp_size, dcp_rank, interleave) - _owned_count(
            lo, dcp_size, dcp_rank, interleave
        )
    if start is None:
        return lens // dcp_size + (dcp_rank < (lens % dcp_size))
    first = start + np.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return np.maximum((remaining + dcp_size - 1) // dcp_size, 0)


def attention_tp_size(tp_size: int, dp_size: int) -> int:
    if dp_size < 1 or tp_size < 1:
        raise ValueError(f"tp_size and dp_size must be >= 1, got tp={tp_size} dp={dp_size}")
    if tp_size % dp_size != 0:
        raise ValueError(f"tp_size={tp_size} must be divisible by dp_size={dp_size}")
    return tp_size // dp_size


def validate_dcp_mesh(tp_size: int, dp_size: int, dcp_size: int) -> None:
    """DCP groups nest inside one attention-TP group (CUDA DCP docs)."""
    if dcp_size < 1:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    attn_tp = attention_tp_size(tp_size, dp_size)
    if attn_tp % dcp_size != 0:
        raise ValueError(
            f"dcp_size={dcp_size} must divide attention_tp={attn_tp} "
            f"(tp_size={tp_size} / dp_size={dp_size}). "
            "A DCP group cannot cross an attention-DP replica."
        )
