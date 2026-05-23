"""HiCache ``TreeComponent`` ABC + supporting enums (PR-1-3 stub).

Surface only — no behavior yet. Concrete subclasses
(``FullComponent`` / ``SWAComponent`` / ``RecurrentStateComponent``)
land in PR-2/PR-3 per the roadmap §6.2/§6.3 task lists.

References:
- RFC-1 §4.1 (`/Users/jiongxuan/workspace/wiki/.../rfc_1_hicache.md`)
- roadmap §6.1 PR-1-3

This module deliberately has no runtime side effects: it only declares
the interface so downstream PRs and tests can import the symbols and
type-check against them.
"""

from __future__ import annotations

import abc
import enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax


class ComponentType(enum.Enum):
    """Attention shape a :class:`TreeComponent` handles.

    Mirrors RFC-1 §4.1's three concrete component subclasses; the
    scheduler / cache builder will dispatch on this enum to wire the
    right component when a model is loaded.
    """

    FULL = "full"
    SWA = "swa"
    RECURRENT_STATE = "recurrent_state"


class CacheTransferPhase(enum.Enum):
    """Lifecycle state of an L1↔L2 (and later L2↔L3) KV transfer.

    Tracked by :class:`HiCacheController` (PR-2) to keep the scheduler
    in sync with async ``device_put`` / Pallas ``copy_to_host`` futures.
    Names match the verbs used in RFC-1 §4.1 hooks so the state machine
    reads top-to-bottom in the controller log.
    """

    IDLE = "idle"
    SCHEDULED = "scheduled"
    IN_FLIGHT = "in_flight"
    DONE = "done"
    FAILED = "failed"


class TreeComponent(abc.ABC):
    """Abstract base for attention-shape-specific HiCache hooks.

    Each subclass owns the per-attention-type policy for moving KV pages
    between L1 (device) and L2 (host pinned) — and later L3 (storage).
    Three hooks per the RFC:

    - ``write_through`` — fired on ``UnifiedRadixCache.insert``; enqueue
      a non-blocking D2H copy for the new page so it is mirrored to L2
      while the scheduler keeps running forward.
    - ``write_back`` — fired on ``UnifiedRadixCache.evict``; if L3 is
      configured, hand the page off for persistence; otherwise a no-op
      that just releases the L2 slot.
    - ``prefetch`` — fired on prefix lookup hit when the page lives in
      L2/L3 only; pull the page back to L1 and block the consumer
      until the H2D copy is ready.

    Concrete subclasses must declare which :class:`ComponentType` they
    handle (the scheduler uses this for dispatch).
    """

    @property
    @abc.abstractmethod
    def component_type(self) -> ComponentType:
        """The attention shape this component owns."""

    @abc.abstractmethod
    def write_through(
        self,
        *,
        node: Any,
        device_indices: "jax.Array",
        host_indices: "jax.Array",
    ) -> CacheTransferPhase:
        """Schedule a non-blocking D2H copy for a newly-inserted node."""

    @abc.abstractmethod
    def write_back(
        self,
        *,
        node: Any,
        host_indices: "jax.Array",
    ) -> CacheTransferPhase:
        """Persist (or release) host pages of a node being evicted."""

    @abc.abstractmethod
    def prefetch(
        self,
        *,
        node: Any,
        host_indices: "jax.Array",
        device_indices: "jax.Array",
    ) -> CacheTransferPhase:
        """Pull pages from L2/L3 back to L1; blocks the consumer."""


__all__ = [
    "CacheTransferPhase",
    "ComponentType",
    "TreeComponent",
]
