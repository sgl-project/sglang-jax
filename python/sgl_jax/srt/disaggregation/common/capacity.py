"""PD transfer capacity helpers."""

from __future__ import annotations


def per_rank_inflight_limit(max_inflight: int, dp_size: int) -> int:
    """Split the global transfer budget across rank-local managers."""

    max_inflight = int(max_inflight)
    dp_size = int(dp_size)
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if max_inflight <= 0:
        return max_inflight
    return max(1, (max_inflight + dp_size - 1) // dp_size)
