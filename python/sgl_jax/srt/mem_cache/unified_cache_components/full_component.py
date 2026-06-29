from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import TYPE_CHECKING

from sgl_jax.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
)
from sgl_jax.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
    from sgl_jax.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams | None = None):
        super().__init__(cache, params)
        self._free_full = cache.token_to_kv_pool_allocator.free
        # HiCache state: set to host KV pool when HiCache enabled (not in stage 1)
        self._full_kv_pool_host = None

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        # HiCache: the default (host-aware) validator also accepts a node whose
        # FULL KV lives only on host (value is None, host_value set), so the
        # match can walk into the L2 tier. ``match_device_only`` keeps the strict
        # device check for callers that need a device-resident boundary.
        ct = self.component_type
        if self.cache.hicache_enabled and not match_device_only:
            return lambda node: (
                node.component_data[ct].value is not None
                or node.component_data[ct].host_value is not None
            )
        return lambda node: node.component_data[ct].value is not None

    def redistribute_on_node_split(self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode):
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        child_cd = child.component_data[ct]
        split_len = len(new_parent.key)
        if child_cd.value is not None:
            new_parent.component_data[ct].value = child_cd.value[:split_len].copy()
            child_cd.value = child_cd.value[split_len:].copy()
        # HiCache: a backed-up node carries one host buffer_id per PAGE, so the
        # host_value array splits at the PAGE boundary -- not at split_len, which
        # is a TOKEN count. split_len is page-aligned (radix invariant), so
        # split_pages = split_len // page_size folds it to the page unit. Using
        # the token split_len here would slice past the (much shorter) page array
        # and hand the whole host buffer to one side (the partial-match split bug).
        if child_cd.host_value is not None:
            PS = self.cache.page_size
            assert (
                split_len % PS == 0
            ), f"node split at non-page-aligned len {split_len} (page_size={PS})"
            split_pages = split_len // PS
            new_parent.component_data[ct].host_value = child_cd.host_value[:split_pages].copy()
            child_cd.host_value = child_cd.host_value[split_pages:].copy()

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0

        if EvictLayer.DEVICE in target and cd.value is not None:
            node_dp_rank = node.key.dp_rank if node.key and node.key.dp_rank is not None else 0
            freed = len(cd.value)
            self._free_full(cd.value, dp_rank=node_dp_rank)
            self.cache.component_evictable_size_[self.component_type][node_dp_rank] -= freed
            # NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)
            # because SWA's free_swa still needs to read Full.value.
        return freed, 0

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int]) -> None:
        request = params.num_tokens
        dp_rank = params.dp_rank
        # UnifiedTreeNode.__lt__ compares last_access_time, so heapifying the
        # node objects directly yields LRU order.
        heap = list(self.cache.evictable_device_leaves)
        heapq.heapify(heap)
        ct = self.component_type
        while tracker[ct] < request and heap:
            x = heapq.heappop(heap)
            if x not in self.cache.evictable_device_leaves:
                continue
            node_dp_rank = x.key.dp_rank if x.key and x.key.dp_rank is not None else 0
            if dp_rank is not None and node_dp_rank != dp_rank:
                continue
            self.cache._evict_device_leaf(x, tracker)
            if x.parent is not None and x.parent in self.cache.evictable_device_leaves:
                heapq.heappush(heap, x.parent)

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        if lock_host:
            # No host tier in stage 1.
            return result

        ct = self.component_type
        root = self.cache.root_node
        cur = node

        # A lock path may cross tombstones (evicted-but-backuped nodes): with
        # HiCache, partial load_back / re-insert can leave a live node below a
        # still-demoted ancestor, and write_backup locks node->root through it.
        # That mirrors sglang (inc_lock_ref tolerates evicted nodes on the path).
        # Device-token accounting only applies to LIVE nodes: a tombstone was
        # already removed from component_evictable_size_ at eviction and holds no
        # device slots, so we just bump its lock_ref to protect it (eviction
        # skips locked nodes, so a node's live/tombstone state cannot flip while
        # locked -- acquire and release stay symmetric).
        delta = 0
        while cur is not root:
            cd = cur.component_data[ct]
            if cd.value is not None:
                if cd.lock_ref == 0:
                    key_len = len(cd.value)
                    cur_dp_rank = cur.key.dp_rank if cur.key and cur.key.dp_rank is not None else 0
                    self.cache.component_evictable_size_[ct][cur_dp_rank] -= key_len
                    self.cache.component_protected_size_[ct][cur_dp_rank] += key_len
                    # This repo's convention (RadixCache.inc_lock_ref): delta is the
                    # evictable-size change, negative on acquire.
                    delta -= key_len
                self.cache.evictable_device_leaves.discard(cur)
            cd.lock_ref += 1
            cur = cur.parent
        result.delta = delta
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams | None,
        lock_host: bool = False,
    ) -> None:
        if lock_host:
            # No host tier in stage 1.
            return

        ct = self.component_type
        root = self.cache.root_node
        cur = node
        while cur is not root:
            cd = cur.component_data[ct]
            assert cd.lock_ref > 0

            # Mirror acquire: device-token accounting only for live nodes; a
            # tombstone on the path just gets its lock_ref decremented. State
            # cannot have flipped while locked, so this stays symmetric.
            if cd.value is not None and cd.lock_ref == 1:
                key_len = len(cd.value)
                cur_dp_rank = cur.key.dp_rank if cur.key and cur.key.dp_rank is not None else 0
                self.cache.component_evictable_size_[ct][cur_dp_rank] += key_len
                self.cache.component_protected_size_[ct][cur_dp_rank] -= key_len
            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                self.cache._update_evictable_leaf_sets(cur)
            cur = cur.parent
