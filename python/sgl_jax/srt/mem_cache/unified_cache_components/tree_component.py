"""Component layer for UnifiedRadixCache.

Stage 1 ships only the FULL (full-attention) component. The seam surface is
deliberately wider than what stage 1 exercises: CacheTransferPhase /
LRURefreshPhase / next_component_uuid / eviction_priority /
recover_after_unevict / value_len / the ComponentType.is_* helpers and the
unused ``params`` ctor arg exist so SWA / Recurrent / HiCache components can land
against a stable contract without re-touching this module.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import IntEnum, IntFlag, StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from sgl_jax.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
    from sgl_jax.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class ComponentType(IntEnum):
    """Integer enum so that per-node list/tuple storage can be indexed directly."""

    FULL = 0
    SWA = 1
    RECURRENT = 2

    def __str__(self) -> str:  # keep human-readable logging
        return self.name.lower()

    @property
    def is_full(self) -> bool:
        return self == ComponentType.FULL

    @property
    def is_swa(self) -> bool:
        return self == ComponentType.SWA

    @property
    def is_recurrent(self) -> bool:
        return self == ComponentType.RECURRENT


BASE_COMPONENT_TYPE = ComponentType.FULL
_NUM_COMPONENT_TYPES = len(ComponentType)

_LAST_ACCESS_TIME_COUNTER_FLOAT = np.float64(1.0)
_COMPONENT_UUID_COUNTER = 1


@dataclasses.dataclass
class ComponentData:
    value: np.ndarray | None = None
    lock_ref: int = 0
    host_value: np.ndarray | None = None
    host_lock_ref: int = 0


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation.

    Lean stage-1 version (upstream keeps this in base_prefix_cache);
    used only in component seam annotations."""

    prefix_len: int = 0
    # Recurrent: True iff the target node already held a recurrent value
    # (duplicate insert). recurrent_committed: True iff this insert attached
    # the request's recurrent slot to a fresh leaf (tree took ownership);
    # cleanup_after_caching_req keys its donate-vs-free decision on it.
    recurrent_exist: bool = False
    recurrent_committed: bool = False


class EvictLayer(IntFlag):
    """Which storage layer(s) to evict.  Combinable via bitwise OR."""

    DEVICE = 1
    HOST = 2
    ALL = DEVICE | HOST


class CacheTransferPhase(StrEnum):

    BACKUP_HOST = "backup_host"  # D→H
    LOAD_BACK = "load_back"  # H→D
    BACKUP_STORAGE = "backup_storage"  # H→Storage
    PREFETCH = "prefetch"  # Storage→H


class LRURefreshPhase(StrEnum):

    WALKDOWN = "walkdown"  # touching a node while walking through the tree
    MATCH_END = "match_end"  # end of a successful prefix match
    INSERT_END = "insert_end"  # after a new/updated leaf is committed


def get_and_increase_time_counter() -> np.float64:
    global _LAST_ACCESS_TIME_COUNTER_FLOAT
    ret = _LAST_ACCESS_TIME_COUNTER_FLOAT
    _LAST_ACCESS_TIME_COUNTER_FLOAT += 1.0
    return ret


def next_component_uuid() -> int:
    global _COMPONENT_UUID_COUNTER
    _COMPONENT_UUID_COUNTER += 1
    return _COMPONENT_UUID_COUNTER


class TreeComponent(ABC):
    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams | None = None):
        self.cache = cache

    # Subclasses MUST set this as a class attribute (not @property)
    component_type: ComponentType

    def node_has_component_data(
        self, node: UnifiedTreeNode, target: EvictLayer = EvictLayer.DEVICE
    ) -> bool:
        cd = node.component_data[self.component_type]
        # EvictLayer is combinable: require data on every requested layer.
        device_ok = EvictLayer.DEVICE not in target or cd.value is not None
        host_ok = EvictLayer.HOST not in target or cd.host_value is not None
        return device_ok and host_ok

    def value_len(self, node: UnifiedTreeNode) -> int:
        value = node.component_data[self.component_type].value
        return len(value) if value is not None else 0

    @abstractmethod
    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        """Return a per-match stateful predicate that decides whether a node
        is a valid match boundary for this component.
        Called once per match_prefix; the returned closure may carry state.
        When match_device_only is true, host-backed nodes must not be accepted
        as valid match boundaries.
        - Full: returns True if the node has full component data.
        - SWA: tracks accumulated length since last gap; returns True only
          when the contiguous window reaches swa_sliding_window_size.
        - Recurrent: returns True iff the node has recurrent component data."""
        ...

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[np.ndarray],
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the match result after prefix matching completes.
        - Full & SWA: pass through unchanged.
        - Recurrent: records the matched node's recurrent slot as the request's
          copy-on-write source and sets branching_seqlen in result; the state copy
          itself happens lazily in the forward pass, not here."""
        return result

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: np.ndarray,
        params: InsertParams,
    ) -> int:
        """Called per-node when an insert's key overlaps an existing node.
        Returns the index within value_slice from which this component
        consumed (took ownership of) the underlying KV pool slots.
        Returns prefix_len if nothing was consumed (default).
        In stage 1 the core discards the return value (request-caching
        callers free the duplicate overlap instead); once aux components
        land, _insert_helper will use it to free only the non-consumed
        duplicate portion: value_slice[dup_start:consumed_from]."""
        return prefix_len

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        """Return True to veto leaf creation when the entire new leaf would
        be a tombstone for this component."""
        return False

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        """Later-stage hook (no-op in stage 1, which has no tombstones):
        called after the core restores the base (Full) value on an evicted
        node during insert. Aux components (e.g. SWA) override this to
        rebuild their own data from the freshly assigned base value when
        their entry is still tombstoned. Default no-op."""
        return None

    def on_parent_gains_child(self, node: UnifiedTreeNode) -> None:
        """Hook: ``node`` just gained its first child (leaf→internal transition
        in _add_new_node). Leaf-only components (Recurrent) drop their per-leaf
        data here so it is not stranded on an unevictable internal node. Default
        no-op (Full keeps internal-node data)."""
        return None

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        """Finalize component data on the target (leaf) node after the insert
        walk completes. Called once per insert.
        - Full: no-op (full data is handled by _add_new_node).
        - SWA: for new leaves, checks whether the node straddles the SWA
          eviction boundary (swa_evicted_seqlen). If so, splits the node
          via _split_node — the parent becomes a tombstone (no SWA) and the
          child (the deeper portion) receives SWA data. If the entire node
          is within the window, sets SWA directly. If entirely outside,
          leaves SWA as None (tombstone).
        - Recurrent: sets the recurrent component value from params and increments
          evictable size."""
        return None

    @abstractmethod
    def redistribute_on_node_split(self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode):
        """Redistribute component data between new_parent and child when a
        node is split. new_parent is the newly created prefix node.
        - Full: copies child's lock_ref to new_parent.
        - SWA: slices (or copies) the swa value for new_parent, copies
          lock_ref and the swa component_uuid, then syncs child's swa
          value with its (now-trimmed) full_value.
        - Recurrent: sets new_parent's recurrent value to None and lock_ref to 0
          (recurrent data stays on the original leaf, not on prefix nodes)."""
        ...

    @abstractmethod
    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        """Free this component's KV resources on a node being evicted.

        *target* controls which layer(s) to evict:
          - DEVICE: free device memory. Implementations may defer the
                    tombstone (value = None) to the caller's cascade step
                    (see FullComponent) — the caller owns setting
                    value = None after all components have been driven.
                    Host data is untouched.
          - HOST:   free host memory (host_value = None).
                    Device data is untouched.
          - ALL:    free both device and host memory.
                    No tombstone — caller will delete the node.

        Returns (device_freed, host_freed) token counts."""
        ...

    def eviction_priority(self, is_leaf: bool) -> int:
        """Eviction priority on this node type. Higher = evicted later.
        When a component is evicted, all other components with equal or
        lower priority on the same node are also cascade-evicted.

        Leaf: all components equal (0) — evicting any cascades to all,
        because the node will be deleted.

        Internal: full=2 > swa=1 > recurrent=0.
        Why swa > recurrent: SWA data on internal nodes is *path data* —
        the sliding window needs continuous SWA coverage along the path
        from root to the match boundary. E.g. A->B->C->D->E where C
        and E both have recurrent and the window covers C->E: if C's recurrent
        is evicted, C's SWA must stay so E remains reachable.
        Recurrent data, by contrast, is only meaningful at the match
        boundary node; on internal nodes it
        contributes nothing to the path. So SWA is more valuable to
        keep and should be evicted later.

        Cascade consequences:
        - Recurrent evict internal: no cascade.
        - SWA evict internal: cascades to Recurrent. SWA gone -> SWA
          validator fails -> recurrent data is useless (match requires all
          validators to pass).
        - Full evict internal: cascades to SWA + Recurrent."""
        return 0

    @abstractmethod
    def drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int]) -> None:
        """Drive eviction for this component.
        Each component extracts its own request from params, walks its own
        candidate set (LRU order via last_access_time), evicts, and calls
        cache._cascade_evict for priority cascade.
        Updates the shared tracker with freed amounts for all components.
        - Full: walks evictable device leaves, evicts full then cascades
          the entire leaf.
        - Recurrent: walks internal/leaf candidates; tombstones internal nodes
          (with cascade to equal-priority components like swa), cascades
          leaves to all."""
        ...

    @abstractmethod
    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        """Increment component lock refs, protecting nodes from
        eviction. Updates evictable → protected size on first lock.
        - Full: path-lock — walks from node up to root, incrementing
          lock_ref on every ancestor.
        - SWA: path-lock — walks upward collecting swa values until the
          sliding window is filled; records a component_uuid at the
          boundary for release_component_lock to know where to stop.
        - Recurrent: single-node lock — only increments lock_ref on the
          node itself (recurrent state is per-leaf, not per-path).

        When ``lock_host`` is True, the lock applies to host-side state:
        - Full: single-node host lock.
        - SWA: host window-lock with a dedicated host UUID boundary.
        - Recurrent: single-node host lock."""
        ...

    @abstractmethod
    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams | None,
        lock_host: bool = False,
    ) -> None:
        """Decrement component lock refs, un-protecting nodes.
        Updates protected → evictable size when lock_ref drops to 0.
        - Full: path-unlock — walks from node up to root, decrementing
          lock_ref on every ancestor.
        - SWA: path-unlock — walks upward, stopping at the node whose
          component_uuid matches the one recorded during acquire.
        - Recurrent: single-node unlock — only decrements lock_ref on the
          node itself.

        When ``lock_host`` is True, the inverse host-side semantics apply."""
        ...

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> int | None:
        """Prepare component-specific data before insert, fill component
        fields in insert_params, return effective cache_len.
        Return None for no truncation opinion (use full length);
        return int >= 0 for effective cache length.
        - Full: no-op, returns None.
        - SWA: sets insert_params.swa_evicted_seqlen on finished; returns None.
        - Recurrent: prepares recurrent_value (finished from ping-pong buffer,
          unfinished fork from req); returns recurrent_last_track_seqlen."""
        return None

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: InsertResult | None = None,
        insert_params: InsertParams | None = None,
    ) -> None:
        """Post-cache cleanup for component-specific resources.

        ``is_finished`` — whether the request has finished generation.
        True means the request is complete and its resources can be released;
        ``insert_result`` is None when insert was skipped (cache disabled
        or the retract path); treat as "no insert happened".
        ``insert_params`` is None when no insert flow ran (cache disabled,
        or the finished-retract path); on other early-return paths it is
        still provided so components can free their resources."""
        return None

    # ---- HiCache Hooks ----

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        req: Req | None = None,
        token_ids: Sequence[int] | None = None,
        prefetch_tokens: int = 0,
        last_hash: str | None = None,
    ) -> list | None:
        """Build transfer descriptors for this component in the given phase.
        Returns None if the component has nothing to transfer."""
        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: Sequence[Any] = (),
        *,
        insert_result: InsertResult | None = None,
        pool_storage_result: Any = None,
    ) -> None:
        """Post-transfer bookkeeping: store host indices, update LRU, etc."""
        return None

    def drive_host_eviction(self, num_tokens: int, tracker: dict[ComponentType, int]) -> None:
        """Evict from this component's host-side resources.
        Called by the host pool owner when the host pool is full.
        Default no-op for components without host storage."""
        return None
