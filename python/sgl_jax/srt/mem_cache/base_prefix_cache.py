from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import numpy as np

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.radix_cache import RadixKey, TreeNode
else:
    TreeNode = Any


@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across cache types."""

    key: RadixKey
    # Record the deepest recurrent-bearing match on ``req`` as the CoW clone source.
    cow_recurrent: bool = False
    req: Any = None
    # Match with the base FULL validator only: a request's own prefix re-match
    # must not be gated on aux components (its recurrent state lives in the
    # running slot, not the tree).
    full_only: bool = False


@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across cache types."""

    key: RadixKey | None = None
    value: Any = None
    # Length of ``value`` already owned by the tree. UnifiedRadixCache uses it
    # to free only request-owned overlap; SWARadixCache also uses it for SWA
    # overlap/healing. RadixCache ignores it.
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0
    # Length-1 int32 array (a RecurrentStatePool slot index); ownership passes
    # to the tree at commit.
    recurrent_value: Any = None


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation."""

    prefix_len: int = 0
    # recurrent_committed: the tree took ownership of the request's slot;
    # cleanup_after_caching_req keys donate-vs-free on it.
    recurrent_exist: bool = False
    recurrent_committed: bool = False


@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across cache types."""

    num_tokens: int = 0
    swa_num_tokens: int = 0
    dp_rank: int | None = None
    recurrent_num: int = 0


@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation."""

    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    recurrent_num_evicted: int = 0


@dataclasses.dataclass
class DecLockRefParams:
    """Parameters for dec_lock_ref."""

    swa_uuid_for_lock: int | None = None
    # Node ids where inc_lock_ref acquired nothing (no value); release skips them.
    skip_lock_node_ids: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class IncLockRefResult:
    """Result of inc_lock_ref."""

    delta: int | None = None
    swa_uuid_for_lock: int | None = None
    skip_lock_node_ids: dict = dataclasses.field(default_factory=dict)

    def to_dec_params(self) -> DecLockRefParams:
        return DecLockRefParams(
            swa_uuid_for_lock=self.swa_uuid_for_lock,
            skip_lock_node_ids=self.skip_lock_node_ids,
        )


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
        best_match_node :   Deepest node accepted by the match;
                            equals last_device_node when HiCache is off.
        host_hit_length :   Length of the KV cache hit on the host, if applicable.
                            0 if HiCache is not enabled.
    """

    device_indices: jax.Array
    last_device_node: TreeNode | None
    last_host_node: TreeNode | None
    best_match_node: TreeNode | None
    host_hit_length: int = 0
    # Always None in the base path (clones the full match); branch truncation is
    # a follow-up.
    recurrent_branching_seqlen: int | None = None


class BasePrefixCache(abc.ABC):
    """Cache can be indexed by either rid or key."""

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    @abc.abstractmethod
    def cache_finished_req(self, req: Any, **kwargs):
        pass

    @abc.abstractmethod
    def cache_unfinished_req(self, req: Any, **kwargs):
        pass

    @abc.abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    @abc.abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        pass

    @abc.abstractmethod
    def dec_lock_ref(self, node: Any, params: DecLockRefParams | None = None):
        pass

    def evictable_size(self, dp_rank: int = 0):
        return 0

    def supports_swa(self) -> bool:
        """Whether this cache owns a sliding-window KV pool."""
        return False

    def cache_ledger_snapshot(self, dp_rank: int, live_reqs):
        """Debug-only ownership ledger; subclasses must provide real data."""
        raise NotImplementedError

    def supports_recurrent(self) -> bool:
        return False

    def recurrent_extra_buffer_active(self) -> bool:
        """True when this cache materializes page-boundary recurrent snapshots
        (extra-buffer recurrent path). Drives scheduler boundary splitting and
        track-entry computation; False keeps the path byte-identical to today."""
        return False

    def full_evictable_size(self, dp_rank: int = 0):
        return 0

    def swa_evictable_size(self, dp_rank: int = 0):
        return 0

    def protected_size(self, dp_rank: int = 0):
        return 0

    def full_protected_size(self, dp_rank: int = 0):
        return 0

    def swa_protected_size(self, dp_rank: int = 0):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        last_host_node: Any,
        host_hit_length: int,
    ) -> tuple[jax.Array, Any]:
        """
        Preparing KV cache loading from host to device.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def check_hicache_events(self) -> Any:
        raise NotImplementedError()

    def take_events(self):
        return []


def build_swa_cache_ledger_snapshot(
    *,
    dp_rank: int,
    allocator: Any,
    full_tree_evictable: set[int],
    full_tree_protected: set[int],
    swa_tree_evictable: set[int],
    swa_tree_protected: set[int],
    full_request_occurrences: list[int],
    swa_request_occurrences: list[int],
    event_totals: dict[str, int],
) -> dict[str, int]:
    """Build the common debug-only SWA ownership schema from real owners."""

    def usable_capacity(pool: Any) -> int:
        if hasattr(pool, "pages_per_rank"):
            return pool.pages_per_rank * pool.page_size
        return pool.size_per_rank

    def reserved_page_slack(*owners: set[int]) -> int:
        if allocator.page_size == 1:
            return 0
        owned = set().union(*owners)
        reserved_pages = {index // allocator.page_size for index in owned}
        return len(reserved_pages) * allocator.page_size - len(owned)

    full_request = set(full_request_occurrences)
    swa_request = set(swa_request_occurrences)
    full_capacity = usable_capacity(allocator.full_attn_allocator)
    swa_capacity = usable_capacity(allocator.swa_attn_allocator)

    page_size = allocator.page_size
    first_usable = page_size if page_size > 1 else 1
    mapping = (
        allocator.full_to_swa_index_mapping
        if allocator.dp_size == 1
        else allocator.full_to_swa_index_mapping[dp_rank]
    )
    mapping_array = np.asarray(mapping)
    nonzero_sources = np.flatnonzero(mapping_array)
    nonzero = mapping_array[nonzero_sources]
    last_usable_full = first_usable + full_capacity - 1
    last_usable_swa = first_usable + swa_capacity - 1

    snapshot = {
        "dp_rank": dp_rank,
        "full_capacity": full_capacity,
        "full_available": allocator.full_available_size(dp_rank),
        "full_tree_evictable": len(full_tree_evictable),
        "full_tree_protected": len(full_tree_protected),
        "full_request_owned": len(full_request),
        "full_reserved_page_slack": reserved_page_slack(
            full_tree_evictable, full_tree_protected, full_request
        ),
        "swa_capacity": swa_capacity,
        "swa_available": allocator.swa_available_size(dp_rank),
        "swa_tree_evictable": len(swa_tree_evictable),
        "swa_tree_protected": len(swa_tree_protected),
        "swa_request_owned": len(swa_request),
        "swa_reserved_page_slack": reserved_page_slack(
            swa_tree_evictable, swa_tree_protected, swa_request
        ),
        "mapping_nonzero_count": int(len(nonzero)),
        "mapping_invalid_count": int(
            (
                (nonzero_sources < first_usable)
                | (nonzero_sources > last_usable_full)
                | (nonzero < first_usable)
                | (nonzero > last_usable_swa)
            ).sum()
        ),
        "mapping_duplicate_count": int(len(nonzero) - len(set(map(int, nonzero)))),
        "full_duplicate_request_owner_count": len(full_request_occurrences) - len(full_request),
        "swa_duplicate_request_owner_count": len(swa_request_occurrences) - len(swa_request),
        "full_request_tree_overlap_count": len(
            full_request & (full_tree_evictable | full_tree_protected)
        ),
        "swa_request_tree_overlap_count": len(
            swa_request & (swa_tree_evictable | swa_tree_protected)
        ),
    }
    for field in (
        "full_evicted_total",
        "swa_evicted_total",
        "tombstone_created_total",
        "tombstone_healed_total",
    ):
        snapshot[field] = int(event_totals.get(field, 0))
    return snapshot


def validate_swa_cache_ledger(snapshot: dict[str, int], *, require_idle: bool) -> None:
    """Reject leaks and ambiguous ownership before or after an idle flush."""
    for pool in ("full", "swa"):
        actual = sum(
            snapshot[f"{pool}_{owner}"]
            for owner in (
                "available",
                "tree_evictable",
                "tree_protected",
                "request_owned",
                "reserved_page_slack",
            )
        )
        expected = snapshot[f"{pool}_capacity"]
        if actual != expected:
            raise ValueError(
                f"[{pool}] balance mismatch for dp_rank={snapshot['dp_rank']}: "
                f"expected={expected}, actual={actual}"
            )

    for field in (
        "mapping_invalid_count",
        "mapping_duplicate_count",
        "full_duplicate_request_owner_count",
        "swa_duplicate_request_owner_count",
        "full_request_tree_overlap_count",
        "swa_request_tree_overlap_count",
    ):
        if snapshot[field] != 0:
            raise ValueError(
                f"SWA ledger {field}={snapshot[field]} for dp_rank={snapshot['dp_rank']}"
            )

    mapped_owners = sum(
        snapshot[f"swa_{owner}"] for owner in ("tree_evictable", "tree_protected", "request_owned")
    )
    if snapshot["mapping_nonzero_count"] != mapped_owners:
        raise ValueError(
            f"SWA mapping ownership mismatch for dp_rank={snapshot['dp_rank']}: "
            f"mapping_nonzero={snapshot['mapping_nonzero_count']}, owners={mapped_owners}"
        )
    if require_idle and (snapshot["full_request_owned"] or snapshot["swa_request_owned"]):
        raise ValueError(f"Idle SWA ledger still has request owners: {snapshot}")
