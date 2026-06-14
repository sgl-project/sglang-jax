"""Fixed-shape cross-process agreement on per-NP receiver terminal state.

Decode-side only. The scatter into the KV pool is a cross-host jit, so
every NP must drain the same set of receivers in the same loop iteration.
Each NP's receiver reaches SUCCESS/FAILED independently (it pulls from a
different P host), so we allgather (bootstrap_room, state) with a fixed
buffer shape — safe even when the in-flight set differs across NPs.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll

_SYNC_MAX_INFLIGHT = 256


def synced_terminal_rooms(
    entries: Iterable,
    poll_fn: Callable[[object], KVPoll],
    room_fn: Callable[[object], int | None],
) -> tuple[set[int], set[int]]:
    """Return ``(success_rooms, failed_rooms)`` agreed across all processes.

    ``poll_fn`` is wrapped in a try/except so a per-NP receiver exception
    becomes FAILED instead of skipping the allgather (which would desync
    the SPMD program counter).
    """

    from jax.experimental import multihost_utils

    local = np.full((_SYNC_MAX_INFLIGHT, 2), -1, dtype=np.int64)
    for i, e in enumerate(entries):
        if i >= _SYNC_MAX_INFLIGHT:
            break
        room = room_fn(e)
        if room is None:
            continue
        try:
            st = poll_fn(e)
        except Exception:
            st = KVPoll.FAILED
        local[i, 0] = int(room)
        local[i, 1] = 1 if st == KVPoll.SUCCESS else (-2 if st == KVPoll.FAILED else 0)
    gathered = multihost_utils.process_allgather(local)
    nproc = int(gathered.shape[0])
    per_room: dict[int, list[int]] = {}
    for p in range(nproc):
        for i in range(_SYNC_MAX_INFLIGHT):
            room = int(gathered[p, i, 0])
            if room < 0:
                continue
            per_room.setdefault(room, []).append(int(gathered[p, i, 1]))
    success: set[int] = set()
    failed: set[int] = set()
    for room, sts in per_room.items():
        if -2 in sts:
            failed.add(room)
        elif len(sts) >= nproc and all(s == 1 for s in sts):
            success.add(room)
    return success, failed
