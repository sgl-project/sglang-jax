"""Pure token-extent helpers for fused-MoE wrappers.

This module intentionally has no JAX dependency so the launch-shape contract can
be validated on a host that does not have, import, or initialize JAX.
"""

from __future__ import annotations

import math


def align_fused_moe_v1_local_tokens(local_num_tokens: int, t_packing: int) -> int:
    """Round a logical local token count up to a legal v1 launch extent.

    The v1 kernel permits the small decode tiles 2 and 4; larger token tiles
    must be multiples of 8. The launch extent must also be divisible by the
    dtype packing factor. The logical token count is not changed by this
    helper: callers pad each device-local shard and slice the result back.
    """

    if local_num_tokens <= 0:
        raise ValueError(f"Expected {local_num_tokens=} to be > 0.")
    if t_packing <= 0:
        raise ValueError(f"Expected {t_packing=} to be > 0.")

    for small_extent in (2, 4):
        if local_num_tokens <= small_extent and small_extent % t_packing == 0:
            return small_extent

    alignment = math.lcm(8, t_packing)
    return ((local_num_tokens + alignment - 1) // alignment) * alignment


def align_fused_moe_v1_num_tokens(
    num_tokens: int,
    ep_size: int,
    t_packing: int,
) -> int:
    """Return the global-equivalent extent for rank-local v1 padding.

    This value sizes v1 tuning and launch internals only. It is not a new
    global tensor shape or scheduler bucket.
    """

    if num_tokens <= 0:
        raise ValueError(f"Expected {num_tokens=} to be > 0.")
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")

    local_num_tokens = num_tokens // ep_size
    return align_fused_moe_v1_local_tokens(local_num_tokens, t_packing) * ep_size
