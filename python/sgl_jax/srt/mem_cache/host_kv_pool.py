"""HiCache L2 pinned-host KV slot pool (RFC-1.0).

Manages slot ID allocation and jax.Array reference storage for the
HiCache host tier. No pre-allocation — each slot holds a reference
to a pinned-host jax.Array produced by jax.device_put with
memory_kind='pinned_host'.
"""

from __future__ import annotations

import logging
from typing import Sequence

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

logger = logging.getLogger(__name__)


def make_host_sharding(mesh: Mesh, partition_spec: PartitionSpec) -> NamedSharding:
    """Build a host-side NamedSharding with pinned_host memory kind.

    Falls back to default memory kind on platforms without pinned_host
    support (e.g. CPU-only JAX) so unit tests can run anywhere.
    """
    try:
        return NamedSharding(mesh, partition_spec, memory_kind="pinned_host")
    except (TypeError, ValueError):
        logger.warning(
            "pinned_host memory_kind unavailable; falling back to default."
        )
        return NamedSharding(mesh, partition_spec)


class HostKVPool:
    """Pinned-host KV slot pool for HiCache L2.

    Provides slot ID allocation/free and jax.Array reference storage.
    Eviction is NOT handled here — it's driven by the tree cache layer
    (RFC-1.1) which atomically calls pool.free() + clears node.host_value.
    """

    def __init__(self, capacity: int, host_sharding: NamedSharding):
        self.capacity = capacity
        self.host_sharding = host_sharding
        self.slot_table: dict[int, jax.Array] = {}
        self.free_list: list[int] = list(range(capacity))

    def alloc(self, n: int = 1) -> list[int] | None:
        """Allocate n slot IDs. Returns None if insufficient space."""
        if len(self.free_list) < n:
            return None
        allocated = self.free_list[:n]
        self.free_list = self.free_list[n:]
        return allocated

    def free(self, indices: list[int] | Sequence[int]) -> None:
        """Free slots: delete array references, return IDs to free_list."""
        for idx in indices:
            self.slot_table.pop(idx, None)
            self.free_list.append(idx)

    def put(self, slot_id: int, data: jax.Array) -> None:
        """Store a pinned-host jax.Array reference at slot_id."""
        self.slot_table[slot_id] = data

    def get(self, slot_id: int) -> jax.Array:
        """Retrieve the pinned-host jax.Array at slot_id."""
        return self.slot_table[slot_id]

    def available_size(self) -> int:
        """Number of free slots."""
        return len(self.free_list)
