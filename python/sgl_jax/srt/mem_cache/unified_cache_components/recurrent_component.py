"""Recurrent (linear-recurrent state) component for UnifiedRadixCache.

Brings KDA / GDN / GLA recurrent state into the radix tree so it is reused
across requests like full KV. Recurrent state is per-leaf (a single
RecurrentStatePool slot index), so this component is leaf-only: it rides the
core's ``evictable_device_leaves`` and skips upstream's LRU-list machinery.

PR#1 scope (page_size=1): finished-request commit + prefix-hit copy-on-write
clone. Unfinished/fork donation and page>=128 (extra buffer) are deferred.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import TYPE_CHECKING

from sgl_jax.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sgl_jax.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    EvictLayer,
    InsertResult,
    TreeComponent,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
    from sgl_jax.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


def _node_dp_rank(node: UnifiedTreeNode) -> int:
    return node.key.dp_rank if node.key and node.key.dp_rank is not None else 0


class RecurrentComponent(TreeComponent):
    component_type = ComponentType.RECURRENT

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams | None = None):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool

        assert isinstance(cache.req_to_token_pool, HybridReqToTokenPool), (
            f"RecurrentComponent requires HybridReqToTokenPool, "
            f"got {type(cache.req_to_token_pool)}"
        )
        if not getattr(cache, "enable_mamba_extra_buffer", False):
            assert cache.page_size == 1, (
                f"RecurrentComponent requires page_size=1 when the extra buffer "
                f"is off (PR#1), got {cache.page_size}"
            )
        super().__init__(cache, params)
        self.req_to_token_pool = cache.req_to_token_pool
        self.recurrent_state_pool = cache.req_to_token_pool.recurrent_state_pool

    # ---- matching -----------------------------------------------------------

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        ct = self.component_type
        return lambda node: node.component_data[ct].value is not None

    def finalize_match_result(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list,
        best_value_len: int,
    ) -> MatchResult:
        # CoW decision only: record the src slot, do NOT allocate. The dst is the
        # request's running slot (allocated post-admission in alloc_req_slots);
        # the copy runs in the next forward's one-shot pre-pass. PR#1 clones the
        # full match (recurrent_branching_seqlen stays None).
        if not params.cow_recurrent:
            return result
        req = params.req
        node = result.best_match_node
        if req is None or node is None:
            return result
        cd = node.component_data[self.component_type]
        if cd.value is not None:
            req.recurrent_cow_src_index = int(cd.value[0])
        return result

    # ---- insert / commit ----------------------------------------------------

    def redistribute_on_node_split(self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode):
        # Recurrent data is leaf-only; the new prefix (internal) parent gets none.
        ct = self.component_type
        new_parent.component_data[ct].value = None
        new_parent.component_data[ct].lock_ref = 0

    def on_parent_gains_child(self, node: UnifiedTreeNode) -> None:
        # Recurrent is leaf-only: a node that gains a child becomes internal and
        # must drop its recurrent value (else the slot is stranded — internal
        # nodes are never device leaves, so drive_eviction can't reclaim it). If
        # still locked (a CoW src this round), defer the free to the final unlock.
        cd = node.component_data[self.component_type]
        if cd.value is None or cd.lock_ref > 0:
            return
        self.req_to_token_pool.free_recurrent_slot(int(cd.value[0]), _node_dp_rank(node))
        self.cache.component_evictable_size_[self.component_type][_node_dp_rank(node)] -= 1
        cd.value = None

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        ct = self.component_type
        if params.recurrent_value is None:
            # No recurrent donation (unfinished/fork in PR#1, or the request had
            # no running slot): leave the node's recurrent value untouched.
            return
        cd = node.component_data[ct]
        if cd.value is not None:
            # Re-cache of a prefix the tree already holds: keep the existing
            # value, leave the request slot for the caller to free.
            result.recurrent_exist = True
            result.recurrent_committed = False
            return
        if node.children:
            # Internal node (shorter-prefix-after-longer split target). Recurrent
            # is leaf-only, so attaching here would orphan the slot. Skip.
            result.recurrent_committed = False
            return
        cd.value = params.recurrent_value
        self.cache.component_evictable_size_[ct][_node_dp_rank(node)] += 1
        self.cache._update_evictable_leaf_sets(node)
        result.recurrent_committed = True

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> int | None:
        # PR#1 page_size=1: only finished requests donate. The running slot is
        # already the final, materialized state, so the tree value is the slot
        # itself (no copy). Unfinished/fork donation is deferred (PR#2) to avoid
        # the publish-before-materialize race.
        if not is_finished or req.recurrent_pool_idx is None:
            return None
        insert_params.recurrent_value = self.req_to_token_pool.recurrent_value_from_slot(
            req.recurrent_pool_idx
        )
        return None

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: InsertResult | None = None,
        insert_params: InsertParams | None = None,
    ) -> None:
        # Sole owner of the finished donate-vs-free decision. Unfinished requests
        # donate nothing in PR#1 and keep their running slot (the live state);
        # ownership-based release frees it later if needed.
        if not is_finished:
            return
        committed = insert_result.recurrent_committed if insert_result is not None else False
        if committed:
            # Tree owns the running slot now (its content is the final state).
            self.req_to_token_pool.commit_to_tree(req)
        # else: internal-target / duplicate / no-insert → leave
        # req.recurrent_pool_idx set so the ownership-based release frees it.

    # ---- eviction -----------------------------------------------------------

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        if EvictLayer.DEVICE in target and cd.value is not None:
            self.req_to_token_pool.free_recurrent_slot(int(cd.value[0]), _node_dp_rank(node))
            self.cache.component_evictable_size_[ct][_node_dp_rank(node)] -= 1
            cd.value = None
            freed = 1
        return freed, 0

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0

    def drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int]) -> None:
        request = params.recurrent_num
        if request <= 0:
            return
        ct = self.component_type
        dp_rank = params.dp_rank
        # Leaf-only: walk recurrent-bearing device leaves LRU-first (UnifiedTreeNode
        # __lt__ on last_access_time). Evicting a leaf frees its FULL KV too —
        # atomic, matching upstream's leaf-cascade branch at page=1. Parents of
        # evicted leaves are internal (no recurrent value), so none are re-pushed.
        heap = [
            x
            for x in self.cache.evictable_device_leaves
            if self.node_has_component_data(x, EvictLayer.DEVICE)
        ]
        heapq.heapify(heap)
        while tracker[ct] < request and heap:
            x = heapq.heappop(heap)
            if x not in self.cache.evictable_device_leaves:
                continue
            if not self.node_has_component_data(x, EvictLayer.DEVICE):
                continue
            if dp_rank is not None and _node_dp_rank(x) != dp_rank:
                continue
            self.cache._evict_device_leaf(x, tracker)

    # ---- locking (single-node; recurrent state is per-leaf, not per-path) ----

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type
        if node is self.cache.root_node:
            return result
        cd = node.component_data[ct]
        if cd.value is None:
            # Tombstone when locked → release must skip it (no lock acquired).
            result.skip_lock_node_ids.setdefault(ct, set()).add(node.id)
            return result
        if cd.lock_ref == 0:
            node_dp_rank = _node_dp_rank(node)
            self.cache.component_evictable_size_[ct][node_dp_rank] -= 1
            self.cache.component_protected_size_[ct][node_dp_rank] += 1
        cd.lock_ref += 1
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams | None,
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if node is self.cache.root_node:
            return
        skip = params.skip_lock_node_ids.get(ct, ()) if params else ()
        if node.id in skip:
            return
        cd = node.component_data[ct]
        if cd.lock_ref > 0:
            if cd.lock_ref == 1:
                node_dp_rank = _node_dp_rank(node)
                if cd.value is not None and node.children:
                    # Became internal while locked (a child was inserted during
                    # this request's caching): free instead of re-exposing, since
                    # recurrent is leaf-only. Mirrors on_parent_gains_child.
                    self.req_to_token_pool.free_recurrent_slot(int(cd.value[0]), node_dp_rank)
                    self.cache.component_protected_size_[ct][node_dp_rank] -= 1
                    cd.value = None
                    cd.lock_ref = 0
                    return
                self.cache.component_evictable_size_[ct][node_dp_rank] += 1
                self.cache.component_protected_size_[ct][node_dp_rank] -= 1
            cd.lock_ref -= 1
