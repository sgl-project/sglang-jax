"""Recurrent-state (KDA/GDN/GLA) component for UnifiedRadixCache.

State is one RecurrentStatePool slot per leaf, so the component is leaf-only:
it rides the core's ``evictable_device_leaves`` rather than an LRU list.

The base path (page_size=1): finished-request commit + prefix-hit
copy-on-write clone. The extra-buffer path (page_size>=128) adds a
page-aligned ping-pong extra buffer and unfinished/fork donation."""

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
        if not getattr(cache, "enable_recurrent_extra_buffer", False):
            assert cache.page_size == 1, (
                f"RecurrentComponent requires page_size=1 when the extra buffer "
                f"is off, got {cache.page_size}"
            )
        super().__init__(cache, params)
        self.req_to_token_pool = cache.req_to_token_pool
        self.recurrent_state_pool = cache.req_to_token_pool.recurrent_state_pool
        self.enable_recurrent_extra_buffer = getattr(cache, "enable_recurrent_extra_buffer", False)

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
        ct = self.component_type
        new_parent.component_data[ct].value = None
        new_parent.component_data[ct].lock_ref = 0

    def on_parent_gains_child(self, node: UnifiedTreeNode) -> None:
        # Leaf-only invariant: a node turning internal must drop its slot or it
        # strands unevictable; if locked (a CoW src), defer the free to final unlock.
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
            return
        cd = node.component_data[ct]
        if cd.value is not None:
            result.recurrent_exist = True
            result.recurrent_committed = False
            return
        if node.children:
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
        if not self.enable_recurrent_extra_buffer:
            # Base path (page_size=1), finished-only donation: the running slot
            # already holds the final state (no copy); donating unfinished would
            # publish state not yet materialized.
            if not is_finished or req.recurrent_pool_idx is None:
                return None
            insert_params.recurrent_value = self.req_to_token_pool.recurrent_value_from_slot(
                req.recurrent_pool_idx
            )
            return None

        # Extra-buffer (page>=128): donate the materialized page-boundary snapshot
        # held by the KEEP ping-pong slot, capping the tree key at that boundary.
        cache_len = req.recurrent_last_track_seqlen
        if cache_len is None:
            # No materialized boundary: nothing to donate; cleanup frees owned slots.
            return 0

        keep_idx = self.req_to_token_pool.get_recurrent_ping_pong_keep_idx(req)
        keep_slot = req.recurrent_ping_pong_track_buffer[keep_idx]
        if is_finished:
            # The keep slot IS the final boundary state; donate it directly. Cleanup
            # transfers ownership (do NOT remove it from the buffer here).
            insert_params.recurrent_value = self.req_to_token_pool.recurrent_value_from_slot(
                keep_slot
            )
            return cache_len

        # Unfinished: secure a fresh REPLACEMENT track slot FIRST, then donate the
        # keep slot. A mid-flight request must always retain two owned track slots,
        # so never donate without a secured replacement.
        dp = req.dp_rank if req.dp_rank is not None else 0
        replacement_slot = self.req_to_token_pool.alloc_recurrent_slot(dp)
        if replacement_slot is None:
            self.cache.evict(EvictParams(recurrent_num=1, dp_rank=dp))
            replacement_slot = self.req_to_token_pool.alloc_recurrent_slot(dp)
        if replacement_slot is None:
            # All candidates locked: skip donation (buffer + watermark intact).
            insert_params.recurrent_value = None
            return 0
        insert_params.recurrent_value = self.req_to_token_pool.donate_recurrent_ping_pong_slot(
            req, replacement_slot
        )
        return cache_len

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: InsertResult | None = None,
        insert_params: InsertParams | None = None,
    ) -> None:
        committed = insert_result.recurrent_committed if insert_result is not None else False
        if not self.enable_recurrent_extra_buffer:
            # Base path: sole owner of the finished donate-vs-free decision.
            if not is_finished:
                return
            if committed:
                # Tree owns the running slot now (its content is the final state).
                self.req_to_token_pool.commit_to_tree(req)
            # else: leave req.recurrent_pool_idx set so release frees it.
            return

        # Extra-buffer: the donated value is the KEEP ping-pong slot, not the
        # running slot. Do NOT use commit_to_tree (page=1 running-slot donation).
        dp = req.dp_rank if req.dp_rank is not None else 0
        if is_finished:
            if committed:
                # Tree now owns the keep slot. Zero the keep position so
                # free_recurrent_cache skips it, then free running + the non-keep
                # track slot. Net: 3 req-owned → 1 tree-owned + 2 free.
                keep_idx = self.req_to_token_pool.get_recurrent_ping_pong_keep_idx(req)
                self.req_to_token_pool.set_recurrent_ping_pong_slot(req, keep_idx, 0)
                self.req_to_token_pool.free_recurrent_cache(req)
            else:
                # Nothing donated to the tree (duplicate/internal/no-insert): free
                # running + BOTH track slots.
                self.req_to_token_pool.free_recurrent_cache(req)
            return

        # Unfinished: the request stays live, keeping running + the other track
        # slot in the buffer (free nothing here).
        if committed:
            # Tree owns the donated slot; the buffer already holds the replacement.
            # No request-owned track slot represents that boundary anymore.
            req.recurrent_last_track_seqlen = None
        elif insert_params is not None and insert_params.recurrent_value is not None:
            # A donation was attempted but the tree rejected it: the donated slot
            # was swapped OUT of the buffer (replacement took its place) and is now
            # orphaned. Free it and invalidate the watermark.
            self.req_to_token_pool.free_recurrent_slot(int(insert_params.recurrent_value[0]), dp)
            req.recurrent_last_track_seqlen = None
        # else: donation SKIPPED (replacement alloc failed) → buffer + watermark
        # intact; the request keeps both track slots and may publish later.

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
        # Evicting a leaf frees its FULL KV too (atomic at page=1); parents are
        # internal (no recurrent value), so nothing is re-pushed after a pop.
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

    # ---- locking ------------------------------------------------------------

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
                    # Node went internal while locked: deferred free from
                    # on_parent_gains_child.
                    self.req_to_token_pool.free_recurrent_slot(int(cd.value[0]), node_dp_rank)
                    self.cache.component_protected_size_[ct][node_dp_rank] -= 1
                    cd.value = None
                    cd.lock_ref = 0
                    return
                self.cache.component_evictable_size_[ct][node_dp_rank] += 1
                self.cache.component_protected_size_[ct][node_dp_rank] -= 1
            cd.lock_ref -= 1
