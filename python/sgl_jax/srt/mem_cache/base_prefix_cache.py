from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Any, NamedTuple

import jax

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.radix_cache import RadixKey, TreeNode
else:
    TreeNode = Any


@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across cache types."""

    key: RadixKey
    # Recurrent CoW: when True and the deepest match holds a recurrent value,
    # finalize_match_result records the src slot on ``req`` for a copy-on-write
    # clone into the request's running slot during the next forward.
    cow_recurrent: bool = False
    req: Any = None
    # Match only the base (FULL) component's boundary, ignoring aux-component
    # validators. Used by cache_unfinished_req's internal re-match: a request's
    # own FULL-prefix bookkeeping must not be gated on recurrent state (which
    # lives in the request's running slot, not the tree).
    full_only: bool = False


@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across cache types."""

    key: RadixKey | None = None
    value: Any = None
    # SWA-specific: consumed by SWARadixCache, ignored by RadixCache.
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0
    # Recurrent: length-1 np.int32 array (a RecurrentStatePool slot index)
    # donated to the tree node at commit; set by RecurrentComponent.
    recurrent_value: Any = None


@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across cache types."""

    num_tokens: int = 0
    swa_num_tokens: int = 0
    dp_rank: int | None = None
    # Recurrent: number of recurrent slots to free from the tree.
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
    # Per-component node ids that were tombstones when inc_lock_ref ran, so
    # release_component_lock must skip them (no lock was acquired there).
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
    # Recurrent: aligned branching length for partial-prefix CoW. The base path
    # always clones the full match (None); branch truncation is a follow-up.
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

    def supports_recurrent(self) -> bool:
        """True for caches that manage recurrent (linear-recurrent) state."""
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

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
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
