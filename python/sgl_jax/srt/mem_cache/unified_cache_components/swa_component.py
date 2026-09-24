"""Sliding-window-attention device and host component for UnifiedRadixCache."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from sgl_jax.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
)
from sgl_jax.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    TreeComponent,
    get_and_increase_time_counter,
    next_component_uuid,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
    from sgl_jax.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


def _node_dp_rank(cache: UnifiedRadixCache, node: UnifiedTreeNode) -> int:
    if node.key is not None and node.key.dp_rank is not None:
        return node.key.dp_rank
    if getattr(cache.token_to_kv_pool_allocator, "dp_size", 1) > 1:
        raise ValueError("SWA component requires node.key.dp_rank when dp_size > 1")
    return 0


class SWAComponent(TreeComponent):
    """Own the device SWA view while FULL remains the tree's base ownership."""

    component_type = ComponentType.SWA

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams | None = None):
        super().__init__(cache, params)
        assert params is not None and params.sliding_window_size is not None
        self.sliding_window_size = params.sliding_window_size
        self.allocator = cache.token_to_kv_pool_allocator

    def _set_swa_value(self, node: UnifiedTreeNode) -> None:
        """Rebuild the component's derived SWA indices from its FULL owner."""
        full_indices = node.component_data[ComponentType.FULL].value
        if full_indices is None:
            return
        dp_rank = _node_dp_rank(self.cache, node)
        swa_indices = self.allocator.translate_full_to_swa(full_indices, dp_rank=dp_rank)
        node.component_data[self.component_type].value = swa_indices
        self.cache.component_evictable_size_[self.component_type][dp_rank] += len(swa_indices)

    def _clear_swa_value(self, node: UnifiedTreeNode) -> int:
        cd = node.component_data[self.component_type]
        if cd.value is None:
            return 0
        full_indices = node.component_data[ComponentType.FULL].value
        assert full_indices is not None
        dp_rank = _node_dp_rank(self.cache, node)
        freed = self.allocator.count_swa_mapped(full_indices, dp_rank=dp_rank)
        self.allocator.free_swa(full_indices, dp_rank=dp_rank)
        cd.value = None
        self.cache.component_evictable_size_[self.component_type][dp_rank] -= freed
        return freed

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        live_since_gap: float = float("inf")
        ct = self.component_type

        def validate(node: UnifiedTreeNode) -> bool:
            nonlocal live_since_gap
            value = node.component_data[ct].value
            cd = node.component_data[ct]
            if value is None and (match_device_only or cd.host_value is None):
                live_since_gap = 0
                return False
            live_since_gap += len(node.key)
            return live_since_gap >= self.sliding_window_size

        return validate

    def finalize_match_result(self, result, **kwargs):
        if self.cache._hybrid_hicache:
            result = result._replace(
                swa_host_hit_length=self.cache.get_load_back_sizes(result.last_host_node)[1]
            )
        return result

    def build_hicache_transfers(self, node, phase, *, device_indices=None, **kwargs):
        # Both components use the same page descriptor, but tokens are raw
        # indices into their own device pool; host handles stay pool-scoped.
        from sgl_jax.srt.mem_cache.unified_cache_components.full_component import (
            FullComponent,
        )

        return FullComponent.build_hicache_transfers(
            self, node, phase, device_indices=device_indices, **kwargs
        )

    def commit_hicache_transfer(self, node, phase, transfers=(), **kwargs):
        from sgl_jax.srt.mem_cache.unified_cache_components.full_component import (
            FullComponent,
        )

        FullComponent.commit_hicache_transfer(self, node, phase, transfers=transfers, **kwargs)

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        if phase == LRURefreshPhase.WALKDOWN:
            return
        ct = self.component_type
        remaining = self.sliding_window_size + self.cache.page_size
        cur_time = get_and_increase_time_counter()
        current: UnifiedTreeNode | None = node
        while current is not root_node and remaining > 0:
            cd = current.component_data[ct]
            if cd.value is not None:
                cd.metadata["last_access_time"] = cur_time
            remaining -= len(current.key)
            cur_time -= 0.00001
            current = current.parent

    def _heal_from_fresh_full(
        self,
        node: UnifiedTreeNode,
        fresh_full: np.ndarray,
        *,
        dp_rank: int,
    ) -> None:
        full_cd = node.component_data[ComponentType.FULL]
        old_full = full_cd.value
        assert old_full is not None
        assert not self.allocator.translate_full_to_swa(
            old_full, dp_rank=dp_rank, require_mapped=False
        ).any()
        if full_cd.lock_ref > 0:
            # Another request may still reference the tree's FULL slots. Keep
            # those slots stable and move only the freshly computed SWA view.
            self.allocator.transfer_swa_mapping(fresh_full, old_full, dp_rank=dp_rank)
        else:
            self.allocator.free_full(old_full, dp_rank=dp_rank)
            full_cd.value = fresh_full.copy()
        self._set_swa_value(node)

    def update_component_on_insert_overlap(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        value_slice: np.ndarray,
        params: InsertParams,
    ) -> int:
        cd = node.component_data[self.component_type]
        if cd.value is not None:
            return prefix_len

        node_end = total_prefix_len + prefix_len
        if node_end <= params.prev_prefix_len:
            # This node was already tree-owned when the request acquired its
            # lock; a tombstone here has no fresh request mapping to donate.
            return prefix_len
        boundary = max(params.swa_evicted_seqlen, params.prev_prefix_len)
        dp_rank = _node_dp_rank(self.cache, node)

        if boundary <= total_prefix_len:
            # The request owns a fresh full slice for the entire tombstone.
            self._heal_from_fresh_full(node, value_slice, dp_rank=dp_rank)
            self.cache._update_aux_evictable_node_sets(node)
            return 0

        if boundary < node_end:
            assert boundary % self.cache.page_size == 0
            start_idx = boundary - total_prefix_len
            new_parent = self.cache._split_node(node.key, node, start_idx)
            suffix = next(iter(new_parent.children.values()))
            self._heal_from_fresh_full(suffix, value_slice[start_idx:], dp_rank=dp_rank)
            self.cache._update_aux_evictable_node_sets(suffix)
            return start_idx

        # The entire node lies in the request's already released SWA prefix.
        return prefix_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        """Adopt the still-live SWA suffix of recomputed FULL tombstones."""
        if node.component_data[self.component_type].value is not None:
            return
        boundary = max(params.swa_evicted_seqlen, params.prev_prefix_len)
        if boundary >= total_prefix_len + prefix_len:
            return
        if boundary > total_prefix_len:
            split_at = boundary - total_prefix_len
            assert split_at % self.cache.page_size == 0
            # The original node becomes the suffix and already owns its fresh
            # FULL indices. Do not heal by allocating/replacing them a second time.
            self.cache._split_node(node.key, node, split_at)
        self._set_swa_value(node)
        self.cache._update_aux_evictable_node_sets(node)

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        return params.swa_evicted_seqlen >= total_prefix_len + key_len

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
    ) -> None:
        if not is_new_leaf:
            return

        start = result.prefix_len
        full_value = node.component_data[ComponentType.FULL].value
        assert full_value is not None
        boundary = params.swa_evicted_seqlen
        if boundary >= start + len(full_value):
            return
        if boundary > start:
            assert boundary % self.cache.page_size == 0
            new_parent = self.cache._split_node(node.key, node, boundary - start)
            node = next(iter(new_parent.children.values()))

        self._set_swa_value(node)
        tail_size = (
            (self.sliding_window_size + self.cache.page_size - 1) // self.cache.page_size
        ) * self.cache.page_size
        full_value = node.component_data[ComponentType.FULL].value
        assert full_value is not None
        while len(full_value) > tail_size:
            split_at = len(full_value) - tail_size
            assert split_at % self.cache.page_size == 0
            node = self.cache._split_node(node.key, node, split_at)
            full_value = node.component_data[ComponentType.FULL].value
            assert full_value is not None

    def redistribute_on_node_split(self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode):
        parent_cd = new_parent.component_data[self.component_type]
        child_cd = child.component_data[self.component_type]
        parent_cd.lock_ref = child_cd.lock_ref
        parent_cd.host_lock_ref = child_cd.host_lock_ref
        # Only receipts acquired before this split inherit the new prefix.
        # The lists are shared with IncLockRefResult/to_dec_params and Req;
        # static split ancestry would incorrectly skip locks acquired later.
        receipts = child_cd.metadata.get("skip_lock_receipts", {})
        if receipts:
            parent_cd.metadata["skip_lock_receipts"] = receipts.copy()
            for skipped_ids in receipts.values():
                skipped_ids.append(new_parent.id)
        if child_cd.host_value is not None:
            split_pages = len(new_parent.key) // self.cache.page_size
            parent_cd.host_value = child_cd.host_value[:split_pages].copy()
            child_cd.host_value = child_cd.host_value[split_pages:].copy()
        if "component_uuid" in child_cd.metadata:
            parent_cd.metadata["component_uuid"] = child_cd.metadata.pop("component_uuid")
        if child_cd.value is None:
            return
        # FULL has already been split by the earlier component hook.
        child_dp_rank = _node_dp_rank(self.cache, child)
        parent_full = new_parent.component_data[ComponentType.FULL].value
        child_full = child.component_data[ComponentType.FULL].value
        assert parent_full is not None and child_full is not None
        parent_cd.value = self.allocator.translate_full_to_swa(parent_full, dp_rank=child_dp_rank)
        child_cd.value = self.allocator.translate_full_to_swa(child_full, dp_rank=child_dp_rank)

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 1

    def evict_component(
        self,
        node: UnifiedTreeNode,
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        if EvictLayer.DEVICE not in target:
            return 0, 0
        if self.cache._hybrid_hicache and self.cache.write_policy == "write_back":
            self.cache._hybrid_coordinator.backup_component(node, self.component_type)
        freed = self._clear_swa_value(node)
        self.cache._update_aux_evictable_node_sets(node)
        return freed, 0

    def drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int]) -> None:
        request = params.swa_num_tokens
        if request <= 0:
            return
        ct = self.component_type
        heap = [
            (
                node.component_data[ct].metadata.get("last_access_time", node.last_access_time),
                node.id,
                node,
            )
            for node in self.cache.aux_evictable_device_nodes[ct]
        ]
        heapq.heapify(heap)
        while tracker[ct] < request and heap:
            _, _, node = heapq.heappop(heap)
            if node not in self.cache.aux_evictable_device_nodes[ct]:
                continue
            if params.dp_rank is not None and _node_dp_rank(self.cache, node) != params.dp_rank:
                continue
            if node.children:
                freed, _ = self.evict_component(node)
                tracker[ct] += freed
            else:
                self.cache._evict_device_leaf(node, tracker)

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        if lock_host:
            return result
        ct = self.component_type
        remaining = self.sliding_window_size
        current = node
        skips = result.skip_lock_node_ids.setdefault(ct, [])

        def skip_fragment(fragment):
            skips.append(fragment.id)
            receipts = fragment.component_data[ct].metadata.setdefault("skip_lock_receipts", {})
            receipts[id(skips)] = skips

        while current is not self.cache.root_node and remaining > 0:
            cd = current.component_data[ct]
            if cd.value is None:
                skip_fragment(current)
                if self.cache._hybrid_hicache:
                    remaining -= len(current.key)
                current = current.parent
                continue
            dp_rank = _node_dp_rank(self.cache, current)
            if cd.lock_ref == 0:
                self.cache.component_evictable_size_[ct][dp_rank] -= len(cd.value)
                self.cache.component_protected_size_[ct][dp_rank] += len(cd.value)
            cd.lock_ref += 1
            remaining -= len(cd.value)
            if remaining <= 0:
                if "component_uuid" not in cd.metadata:
                    cd.metadata["component_uuid"] = next_component_uuid()
                result.swa_uuid_for_lock = cd.metadata["component_uuid"]
                return result
            current = current.parent
        if self.cache._hybrid_hicache:
            # A missing boundary has no component UUID. Preserve an exact
            # receipt so release never visits live nodes above this window.
            while current is not self.cache.root_node:
                skip_fragment(current)
                current = current.parent
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: DecLockRefParams | None,
        lock_host: bool = False,
    ) -> None:
        if lock_host:
            return
        ct = self.component_type
        skipped_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        skip = set(skipped_ids)
        uuid = params.swa_uuid_for_lock if params else None
        current = node
        while current is not self.cache.root_node:
            cd = current.component_data[ct]
            receipts = cd.metadata.get("skip_lock_receipts")
            if receipts is not None:
                receipts.pop(id(skipped_ids), None)
                if not receipts:
                    cd.metadata.pop("skip_lock_receipts")
            if current.id not in skip and cd.value is not None:
                assert cd.lock_ref > 0
                dp_rank = _node_dp_rank(self.cache, current)
                if cd.lock_ref == 1:
                    self.cache.component_evictable_size_[ct][dp_rank] += len(cd.value)
                    self.cache.component_protected_size_[ct][dp_rank] -= len(cd.value)
                cd.lock_ref -= 1
                if uuid is not None and cd.metadata.get("component_uuid") == uuid:
                    return
            current = current.parent

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> int | None:
        del token_ids_len, is_finished
        insert_params.swa_evicted_seqlen = req.swa_evicted_seqlen
        return None
