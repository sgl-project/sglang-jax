"""Device-only sliding-window-attention component for ``UnifiedRadixCache``."""

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
        del match_device_only  # The first device-only implementation has no host tier.
        live_since_gap: float = float("inf")
        ct = self.component_type

        def validate(node: UnifiedTreeNode) -> bool:
            nonlocal live_since_gap
            value = node.component_data[ct].value
            if value is None:
                live_since_gap = 0
                return False
            live_since_gap += len(value)
            return live_since_gap >= self.sliding_window_size

        return validate

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
        old_full = node.component_data[ComponentType.FULL].value
        assert old_full is not None
        dp_rank = _node_dp_rank(self.cache, node)

        if boundary <= total_prefix_len:
            # The request owns a fresh full slice for the entire tombstone.
            assert not self.allocator.translate_full_to_swa(
                old_full, dp_rank=dp_rank, require_mapped=False
            ).any()
            self.allocator.free_full(old_full, dp_rank=dp_rank)
            node.component_data[ComponentType.FULL].value = value_slice.copy()
            self._set_swa_value(node)
            self.cache.record_swa_tombstone_healed(dp_rank)
            self.cache._update_aux_evictable_node_sets(node)
            return 0

        if boundary < node_end:
            assert boundary % self.cache.page_size == 0
            start_idx = boundary - total_prefix_len
            new_parent = self.cache._split_node(node.key, node, start_idx)
            suffix = next(iter(new_parent.children.values()))
            old_suffix = suffix.component_data[ComponentType.FULL].value
            assert old_suffix is not None
            assert not self.allocator.translate_full_to_swa(
                old_suffix, dp_rank=dp_rank, require_mapped=False
            ).any()
            self.allocator.free_full(old_suffix, dp_rank=dp_rank)
            suffix.component_data[ComponentType.FULL].value = value_slice[start_idx:].copy()
            self._set_swa_value(suffix)
            self.cache.record_swa_tombstone_healed(dp_rank)
            self.cache._update_aux_evictable_node_sets(suffix)
            return start_idx

        # The entire node lies in the request's already released SWA prefix.
        return prefix_len

    def should_skip_leaf_creation(
        self, total_prefix_len: int, key_len: int, params: InsertParams
    ) -> bool:
        return params.swa_evicted_seqlen >= total_prefix_len + key_len

    def recover_after_unevict(
        self,
        node: UnifiedTreeNode,
        prefix_len: int,
        total_prefix_len: int,
        params: InsertParams,
    ) -> None:
        del prefix_len, total_prefix_len, params
        if node.component_data[self.component_type].value is None:
            self._set_swa_value(node)
            self.cache.record_swa_tombstone_healed(_node_dp_rank(self.cache, node))

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
        freed = self._clear_swa_value(node)
        if freed and node.children:
            self.cache._ledger_event_totals[_node_dp_rank(self.cache, node)][
                "tombstone_created_total"
            ] += 1
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
        while current is not self.cache.root_node and remaining > 0:
            cd = current.component_data[ct]
            if cd.value is None:
                skips.append(current.id)
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
        skip = set(params.skip_lock_node_ids.get(ct, ())) if params else set()
        uuid = params.swa_uuid_for_lock if params else None
        current = node
        while current is not self.cache.root_node:
            if current.id not in skip:
                cd = current.component_data[ct]
                if cd.value is not None:
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
