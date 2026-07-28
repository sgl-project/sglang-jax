"""Capacity helpers shared by PD transfer backends and admission."""

from __future__ import annotations


def per_rank_inflight_limit(max_inflight: int, dp_size: int) -> int:
    """Return the Raiden slot/admission limit for one DP rank.

    ``max_inflight`` is the server-wide limit. Raiden uses one manager per DP
    rank, so giving every manager the global value over-allocates transient
    buffers by ``dp_size``. Divide the capacity evenly and round up so the
    aggregate remains at least the configured global limit.

    A non-positive global value retains the existing "unlimited" admission
    sentinel. Raiden factory validation rejects that sentinel before manager
    construction.
    """

    max_inflight = int(max_inflight)
    dp_size = int(dp_size)
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if max_inflight <= 0:
        return max_inflight
    return max(1, (max_inflight + dp_size - 1) // dp_size)
