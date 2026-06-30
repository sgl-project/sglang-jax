"""Unified radix cache: a component-based radix tree over the device KV cache.

Stage 1 ports the FULL-attention subset of upstream sglang's UnifiedRadixCache.
All per-component behavior (matching, locking, eviction, split redistribution)
is delegated to TreeComponent implementations so that later stages (SWA, Mamba,
HiCache) plug in without touching the core walk.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.mem_cache.radix_cache import (
    RadixKey,
    _convert_to_bigram_key,
    _key_match_page_size1,
    _key_match_paged,
    get_child_key,
)
from sgl_jax.srt.mem_cache.unified_cache_components import (
    _NUM_COMPONENT_TYPES,
    BASE_COMPONENT_TYPE,
    ComponentData,
    ComponentType,
    EvictLayer,
    FullComponent,
    InsertResult,
    TreeComponent,
    get_and_increase_time_counter,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req


class UnifiedTreeNode:
    counter = 0

    def __init__(self, tree_components: tuple[ComponentType, ...]):
        self.children: dict = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: RadixKey | None = None
        self.tree_components = tree_components
        # List indexed by ComponentType (int enum 0..N-1).
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1
        # HiCache: prefix-reuse counter; backs up to host when >= write_through_threshold.
        self.hit_count = 0

    @property
    def evicted(self) -> bool:
        """Tree-level: FULL KV not on device (non-root with value=None)."""
        return self.parent is not None and self.component_data[BASE_COMPONENT_TYPE].value is None

    @property
    def backuped(self) -> bool:
        """Tree-level: FULL KV present on host."""
        return self.component_data[BASE_COMPONENT_TYPE].host_value is not None

    def __lt__(self, other: UnifiedTreeNode) -> bool:
        return self.last_access_time < other.last_access_time


COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
}


class UnifiedRadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int = 1,
        disable: bool = False,
        kv_head_num: int = 32,
        head_dim: int = 128,
        layer_num: int = 32,
        max_seq_len: int = 4096,
        dtype: jnp.dtype = jnp.bfloat16,
        enable_kv_cache_events: bool = False,
        is_eagle: bool = False,
        tree_components: tuple[ComponentType, ...] = (ComponentType.FULL,),
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list = []

        self.process_id = jax.process_index()

        self.is_eagle = is_eagle
        if is_eagle:
            self.key_convert_fn = _convert_to_bigram_key
        else:
            self.key_convert_fn = lambda key: key

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=page_size)

        self.tree_components = tree_components
        self.components: dict[ComponentType, TreeComponent] = {
            ct: COMPONENT_REGISTRY[ct](self, None) for ct in tree_components
        }
        self._components_tuple: tuple[TreeComponent, ...] = tuple(self.components.values())

        # HiCache (L1<->L2) wiring — populated by init_hicache() when enabled.
        self.hicache_enabled: bool = False
        self.hicache_controller = None
        self.host_pool = None
        self.write_through_threshold: int = 1
        self.write_policy: str = "write_through"
        # Injected by the overlap scheduler so write_back's eviction-time D2H
        # gather waits for the prior forward's replace_all (donation-safe).
        self._donation_barrier = None

        self.reset()

    def reset(self):
        self.root_node = UnifiedTreeNode(self.tree_components)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None, dp_rank=None)
        self.root_node.component_data[BASE_COMPONENT_TYPE].value = []
        for ct in self.tree_components:
            self.root_node.component_data[ct].lock_ref = 1
        # Per-component, per-dp_rank token counts.
        self.component_evictable_size_: dict[ComponentType, defaultdict] = {
            ct: defaultdict(int) for ct in self.tree_components
        }
        self.component_protected_size_: dict[ComponentType, defaultdict] = {
            ct: defaultdict(int) for ct in self.tree_components
        }
        self.evictable_device_leaves: set[UnifiedTreeNode] = set()
        # HiCache: host-tier leaves (evicted-but-backuped tombstones) and
        # in-flight async D2H writes (future -> (node, buffer_ids)).
        self.evictable_host_leaves: set[UnifiedTreeNode] = set()
        self.ongoing_write: dict = {}

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key

        if self.disable or len(key) == 0:
            return self._empty_match_result()

        converted_key = RadixKey(self.key_convert_fn(key.token_ids), key.extra_key, key.dp_rank)
        if self.page_size != 1:
            page_aligned_len = len(converted_key) // self.page_size * self.page_size
            converted_key = converted_key[:page_aligned_len]
        if len(converted_key) == 0:
            return self._empty_match_result()

        value, best_device_node, best_value_len, best_host_node, host_hit_length = (
            self._match_prefix_helper(converted_key)
        )
        return self._match_post_processor(
            params, value, best_device_node, best_value_len, best_host_node, host_hit_length
        )

    def insert(self, params: InsertParams) -> int:
        if self.disable:
            return 0

        key = params.key
        value = params.value

        # Support both RadixKey and plain list for backward compatibility.
        if not isinstance(key, RadixKey):
            key = RadixKey(key, None, None)

        converted_key = RadixKey(self.key_convert_fn(key.token_ids), key.extra_key, key.dp_rank)

        if value is None:
            value = np.array(converted_key.token_ids, dtype=np.int32)
        elif isinstance(value, list):
            value = np.array(value, dtype=np.int32)

        if self.is_eagle:
            # Make sure the value len equals the EAGLE bigram key len.
            value = value[: len(converted_key)]

        # Page-align defensively (callers in this repo pre-align; upstream
        # aligns here).
        if self.page_size != 1:
            page_aligned_len = len(converted_key) // self.page_size * self.page_size
            converted_key = converted_key[:page_aligned_len]
            value = value[:page_aligned_len]

        return self._insert_helper(self.root_node, converted_key, value, params)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        tracker = {ct: 0 for ct in self.tree_components}
        for component in self._components_tuple:
            component.drive_eviction(params=params, tracker=tracker)

        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
        )

    def inc_lock_ref(self, node: UnifiedTreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)

        self._update_evictable_leaf_sets(node)
        return result

    def dec_lock_ref(self, node: UnifiedTreeNode, params: DecLockRefParams | None = None):
        if self.disable:
            return 0

        for component in self._components_tuple:
            component.release_component_lock(node=node, params=params)

        self._update_evictable_leaf_sets(node)
        return 0

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        """Cache completed requests. ``is_insert=False`` skips the radix
        insert (retract path) and frees the would-be-cached range directly."""
        committed_kv_len = req.pop_committed_kv_cache()
        dp_rank = req.dp_rank if req.dp_rank is not None else 0
        if self.disable:
            kv_indices = self.req_to_token_pool.read(req.req_pool_idx, committed_kv_len)
            kv_indices = kv_indices[kv_indices != 0]
            self.token_to_kv_pool_allocator.free(kv_indices, dp_rank=dp_rank)
            for component in self._components_tuple:
                component.cleanup_after_caching_req(
                    req, is_finished=True, insert_result=None, insert_params=None
                )
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:committed_kv_len]
        # For the EAGLE bigram key the key length is one less than the token
        # length, so the corresponding kv length is reduced by one as well.
        actual_kv_len = committed_kv_len - 1 if self.is_eagle else committed_kv_len
        kv_indices = self.req_to_token_pool.read(req.req_pool_idx, committed_kv_len)
        kv_indices = kv_indices[kv_indices != 0]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:], dp_rank=dp_rank)
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()

        page_aligned_token_len = page_aligned_len + 1 if self.is_eagle else page_aligned_len
        # cache_protected_len, not len(prefix_indices): the latter may include
        # an unaligned tail owned by the req but not by the tree.
        old_prefix_len = req.cache_protected_len
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In the EAGLE chunked prefill case the prefix indices include one
            # unmatched token; -1 so its kv can be freed without leaking.
            old_prefix_len -= 1

        insert_params = None
        insert_result = None
        if is_insert:
            insert_params = InsertParams(
                key=RadixKey(token_ids[:page_aligned_token_len], req.extra_key, req.dp_rank),
                value=page_aligned_kv_indices,
            )
            for component in self._components_tuple:
                component.prepare_for_caching_req(
                    req, insert_params, len(token_ids), is_finished=True
                )
            # Radix cache takes over one reference from the memory pool.
            new_prefix_len = self.insert(insert_params)
            insert_result = InsertResult(prefix_len=new_prefix_len)
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:new_prefix_len], dp_rank=dp_rank
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:page_aligned_len], dp_rank=dp_rank
            )

        for component in self._components_tuple:
            component.cleanup_after_caching_req(
                req, is_finished=True, insert_result=insert_result, insert_params=insert_params
            )

        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, **kwargs):
        """Cache incomplete requests."""
        if self.disable:
            for component in self._components_tuple:
                component.cleanup_after_caching_req(
                    req, is_finished=False, insert_result=None, insert_params=None
                )
            return

        dp_rank = req.dp_rank if req.dp_rank is not None else 0
        token_ids = req.fill_ids
        all_token_len = len(token_ids)
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        kv_indices = self.req_to_token_pool.read(req.req_pool_idx, all_token_len)

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices

        # For EAGLE, page_aligned_len is for the bigram key; the token len is +1.
        page_aligned_token_len = page_aligned_len + 1 if self.is_eagle else page_aligned_len
        page_aligned_token_ids = token_ids[:page_aligned_token_len]

        # cache_protected_len, not len(prefix_indices): see cache_finished_req.
        old_prefix_len = req.cache_protected_len
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            old_prefix_len -= 1

        radix_key = RadixKey(page_aligned_token_ids, req.extra_key, req.dp_rank)
        insert_params = InsertParams(key=radix_key, value=page_aligned_kv_indices)
        for component in self._components_tuple:
            component.prepare_for_caching_req(req, insert_params, all_token_len, is_finished=False)
        # Radix cache takes over one reference from the memory pool.
        new_prefix_len = self.insert(insert_params)
        insert_result = InsertResult(prefix_len=new_prefix_len)
        self.token_to_kv_pool_allocator.free(
            kv_indices[old_prefix_len:new_prefix_len], dp_rank=dp_rank
        )

        # Prefix indices may have been updated, reuse them.
        new_match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices = new_match_result.device_indices
        new_last_node = new_match_result.last_device_node

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
            new_indices[old_prefix_len:],
        )
        req.last_matched_prefix_len = len(new_indices)
        req.cache_protected_len = len(new_indices)
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` is used later in `PrefillAdder::add_chunked_req`.
        if self.page_size != 1:
            req.prefix_indices = np.concatenate([new_indices, kv_indices[len(new_indices) :]])
        elif self.is_eagle:
            # Attach the kv index of the last token for EAGLE chunked prefill.
            req.prefix_indices = np.concatenate([new_indices, kv_indices[actual_kv_len:]])
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

        for component in self._components_tuple:
            component.cleanup_after_caching_req(
                req, is_finished=False, insert_result=insert_result, insert_params=insert_params
            )

    ##### Size Accessors #####

    def evictable_size(self, dp_rank: int = 0):
        return self.component_evictable_size_[BASE_COMPONENT_TYPE][dp_rank]

    def full_evictable_size(self, dp_rank: int = 0):
        return self.evictable_size(dp_rank)

    def protected_size(self, dp_rank: int = 0):
        return self.component_protected_size_[BASE_COMPONENT_TYPE][dp_rank]

    def full_protected_size(self, dp_rank: int = 0):
        return self.protected_size(dp_rank)

    def swa_evictable_size(self, dp_rank: int = 0):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            full_value = node.component_data[BASE_COMPONENT_TYPE].value
            if full_value is not None:
                total_size += len(full_value)
            stack.extend(node.children.values())
        return total_size

    def pretty_print(self):
        print(f"\n[process {self.process_id}] Unified Radix Tree structure:")
        self._print_helper(self.root_node, 0)
        print(f"total tokens: {self.total_size()}")
        evictable = dict(self.component_evictable_size_[BASE_COMPONENT_TYPE])
        protected = dict(self.component_protected_size_[BASE_COMPONENT_TYPE])
        print(f"evictable size per dp_rank: {evictable}")
        print(f"protected size per dp_rank: {protected}")

    def take_events(self):
        """Atomically takes all events and clears the queue."""
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

    ##### Internal Helper Functions #####

    def _empty_match_result(self) -> MatchResult:
        return MatchResult(
            device_indices=np.empty((0,), dtype=np.int32),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
            host_hit_length=0,
        )

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> tuple[list[np.ndarray], UnifiedTreeNode, int, UnifiedTreeNode, int]:
        node = self.root_node
        node.last_access_time = get_and_increase_time_counter()
        child_key = self.get_child_key_fn(key)

        value: list[np.ndarray] = []
        best_device_node = node
        best_value_len = 0
        best_device_tokens = 0
        best_host_node = node
        best_host_tokens = 0
        cur_tokens = 0
        # Device coverage must stay a contiguous prefix from root. Once a
        # tombstone breaks the chain, device-best freezes — otherwise a deeper
        # still-resident node under an evicted ancestor breaks inc_lock_ref.
        device_broken = False

        device_validators = tuple(
            comp.create_match_validator(match_device_only=True) for comp in self._components_tuple
        )
        host_validators = tuple(
            comp.create_match_validator(match_device_only=False) for comp in self._components_tuple
        )

        def _update_best_if_valid(candidate: UnifiedTreeNode):
            nonlocal best_device_node, best_value_len, best_device_tokens
            nonlocal best_host_node, best_host_tokens, device_broken
            if not device_broken and all(v(candidate) for v in device_validators):
                best_device_node = candidate
                best_value_len = len(value)
                best_device_tokens = cur_tokens
            else:
                device_broken = True
            if all(v(candidate) for v in host_validators):
                best_host_node = candidate
                best_host_tokens = cur_tokens

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # Dead node (evicted and not backed up) ends the walk.
            if child.evicted and not child.backuped:
                break

            child.last_access_time = get_and_increase_time_counter()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                cur_tokens += prefix_len
                if not new_node.evicted:
                    value.append(new_node.component_data[BASE_COMPONENT_TYPE].value)
                _update_best_if_valid(new_node)
                node = new_node
                break

            cur_tokens += prefix_len
            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        host_hit_length = best_host_tokens - best_device_tokens
        return value, best_device_node, best_value_len, best_host_node, host_hit_length

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[np.ndarray],
        best_device_node: UnifiedTreeNode,
        best_value_len: int,
        best_host_node: UnifiedTreeNode,
        host_hit_length: int,
    ) -> MatchResult:
        # Refresh the matched path so deeper nodes look more recently used.
        refresh_from = best_host_node if self.hicache_enabled else best_device_node
        cur_time = get_and_increase_time_counter()
        node_update: UnifiedTreeNode | None = refresh_from
        while node_update is not None:
            node_update.last_access_time = cur_time
            cur_time -= 0.00001
            node_update = node_update.parent

        if best_value_len > 0:
            device_indices = np.concatenate(value[:best_value_len])
        else:
            device_indices = np.empty((0,), dtype=np.int32)

        result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_device_node,
            last_host_node=best_host_node if self.hicache_enabled else best_device_node,
            best_match_node=best_host_node if self.hicache_enabled else best_device_node,
            host_hit_length=host_hit_length if self.hicache_enabled else 0,
        )
        for component in self._components_tuple:
            result = component.finalize_match_result(
                result=result,
                params=params,
                value_chunks=value,
                best_value_len=best_value_len,
            )
        return result

    def _split_node(self, key: RadixKey, child: UnifiedTreeNode, split_len: int) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.key = child.key[:split_len]

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.hit_count = child.hit_count
        for component in self._components_tuple:
            component.redistribute_on_node_split(new_parent=new_node, child=child)
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        child.last_access_time = get_and_increase_time_counter()

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(child)
        return new_node

    def _add_new_node(
        self, parent: UnifiedTreeNode, key: RadixKey, value: np.ndarray
    ) -> UnifiedTreeNode:
        new_node = UnifiedTreeNode(self.tree_components)
        new_node.parent = parent
        new_node.key = key
        new_node.component_data[BASE_COMPONENT_TYPE].value = value.copy()
        parent.children[self.get_child_key_fn(key)] = new_node
        node_dp_rank = key.dp_rank if key.dp_rank is not None else 0
        self.component_evictable_size_[BASE_COMPONENT_TYPE][node_dp_rank] += len(value)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        return new_node

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: np.ndarray,
        params: InsertParams,
    ) -> int:
        node.last_access_time = get_and_increase_time_counter()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = get_and_increase_time_counter()
            prefix_len = self.key_match_fn(node.key, key)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            cd = node.component_data[BASE_COMPONENT_TYPE]
            if self.hicache_enabled and cd.value is None:
                # Revive a tombstone: adopt the recomputed KV as device value
                # while keeping the host copy. Must NOT add to total_prefix_length
                # so cache_*_req does not free slots the tree now owns.
                assert prefix_len % self.page_size == 0, (
                    f"tombstone revive at non-page-aligned len {prefix_len} "
                    f"(page_size={self.page_size})"
                )
                cd.value = value[:prefix_len].copy()
                node_dp_rank = node.key.dp_rank if node.key and node.key.dp_rank is not None else 0
                self.component_evictable_size_[BASE_COMPONENT_TYPE][node_dp_rank] += prefix_len
                self._update_evictable_leaf_sets(node)
                self._update_evictable_leaf_sets(node.parent)
            else:
                value_slice = value[:prefix_len]
                for component in self._components_tuple:
                    component.update_component_on_insert_overlap(
                        node=node,
                        prefix_len=prefix_len,
                        total_prefix_len=total_prefix_length,
                        value_slice=value_slice,
                        params=params,
                    )

                total_prefix_length += prefix_len
                if self.hicache_enabled:
                    self._inc_hit_count(node)

            key = key[prefix_len:]
            value = value[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        is_new_leaf = False
        target_node = node
        if len(key):
            if any(
                comp.should_skip_leaf_creation(
                    total_prefix_len=total_prefix_length,
                    key_len=len(key),
                    params=params,
                )
                for comp in self._components_tuple
            ):
                node_dp_rank = key.dp_rank if key.dp_rank is not None else 0
                self.token_to_kv_pool_allocator.free(value, dp_rank=node_dp_rank)
                return total_prefix_length
            target_node = self._add_new_node(node, key, value)
            is_new_leaf = True

        # Finalize: let each component attach its data to the target node.
        result = InsertResult(prefix_len=total_prefix_length)
        for component in self._components_tuple:
            component.commit_insert_component_data(
                node=target_node,
                is_new_leaf=is_new_leaf,
                params=params,
                result=result,
            )

        return total_prefix_length

    ##### HiCache (L1<->L2) Write Path #####

    def _inc_hit_count(self, node: UnifiedTreeNode) -> None:
        """Bump hit counter; back up to host on threshold (write_through only)."""
        if self.write_policy == "write_back":
            return
        if node is self.root_node or node.backuped:
            return
        node.hit_count += 1
        if node.hit_count >= self.write_through_threshold:
            self.write_backup(node)

    def _reserve_host_slots(self, num_pages: int) -> list[int] | None:
        """Alloc host pages, evicting host LRU leaves if short. None if still short."""
        pages = self.host_pool.alloc(num_pages)
        if pages is None:
            shortfall = num_pages - self.host_pool.available_size()
            if shortfall > 0:
                self.evict_host(shortfall)
            pages = self.host_pool.alloc(num_pages)
        if pages is None:
            return None
        return [int(p) for p in pages]

    def _to_global_device_pages(self, local_pages, dp_rank: int) -> list[int]:
        """Map per-rank local device page ids to global page ids.

        The allocator hands out per-rank local views; HiCache gather/scatter runs
        outside shard_map on the global buffer, so convert via
        ``global = dp_rank * pages_per_shard + local``."""
        dp_size = getattr(self.token_to_kv_pool_allocator, "dp_size", 1)
        if dp_size <= 1 or dp_rank == 0:
            return [int(x) for x in local_pages]
        device_pool = self.token_to_kv_pool_allocator.get_kvcache()
        pages_per_shard = device_pool.kv_buffer[0].shape[0] // dp_size
        return [int(x) + dp_rank * pages_per_shard for x in local_pages]

    def write_backup(self, node: UnifiedTreeNode, write_back: bool = False) -> int:
        """Async D2H backup of a node's device KV to host.

        write_back=True is the eviction-time path: leaf-up eviction guarantees
        ancestors are already backed up, so parent recursion is skipped. The device
        lock is held only around the synchronous gather (stage_backup); the slow
        host transfer (flush_backup) is async and never touches the device buffer,
        so releasing the lock early lets the node be demoted under memory pressure.

        Token->page folding: cd.value is token-level; take every page_size-th
        element and divide by page_size to get local device page ids."""
        if node is self.root_node or node.backuped:
            return 0

        if not write_back and node.parent is not self.root_node and not node.parent.backuped:
            self.write_backup(node.parent)
            if not node.parent.backuped:
                return 0

        cd = node.component_data[BASE_COMPONENT_TYPE]
        device_indices = cd.value
        if device_indices is None or len(device_indices) == 0:
            return 0

        PS = self.page_size
        device_tokens = np.asarray(device_indices)
        assert (
            len(device_tokens) % PS == 0
        ), f"node.value len {len(device_tokens)} not page-aligned (page_size={PS})"
        device_pages = device_tokens[::PS] // PS
        num_pages = len(device_pages)
        host_pages = self._reserve_host_slots(num_pages)
        if host_pages is None:
            return 0

        self.inc_lock_ref(node)
        node_dp_rank = node.key.dp_rank if node.key and node.key.dp_rank is not None else 0
        global_pages = self._to_global_device_pages(device_pages, node_dp_rank)
        future = self.hicache_controller.write(global_pages, host_pages)
        cd.host_value = np.array(host_pages, dtype=np.int64)
        self.dec_lock_ref(node)
        self.ongoing_write[future] = (node, host_pages)
        return num_pages

    def precompile_hicache_transfers(self) -> None:
        """Precompile host<->device transfer kernels at startup (no-op if disabled)."""
        if not self.hicache_enabled or self.host_pool is None:
            return
        self.host_pool.precompile_transfers()

    def writing_check(self) -> None:
        """Settle completed async D2H writes (non-blocking)."""
        if not self.ongoing_write:
            return
        done = [f for f in self.ongoing_write if f.done()]
        for f in done:
            self.ongoing_write.pop(f)
        self.hicache_controller.check_write_status()

    def check_hicache_events(self) -> None:
        """Non-blocking poll of in-flight D2H writes (scheduler hook)."""
        if self.hicache_enabled:
            self.writing_check()

    def flush_write_through_acks(self) -> None:
        """Settle finished D2H writes mid-step to free device locks early."""
        if self.hicache_enabled:
            self.writing_check()

    def ready_to_load_host_cache(self) -> int:
        """No-op: H2D load is synchronous, nothing to pre-arm."""
        return 0

    ##### Evict Helpers #####

    def _cascade_evict(self, node: UnifiedTreeNode) -> None:
        """Tombstone the base value after all components have been driven.

        Stage 1 collapses the upstream priority cascade: FULL is both the
        trigger and the only component. The deferral contract puts
        value = None here, not in FullComponent.evict_component."""
        node.component_data[BASE_COMPONENT_TYPE].value = None
        self._update_evictable_leaf_sets(node)

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode) -> None:
        child_key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(child_key, None)
        assert v is node

    def _evict_device_leaf(self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]) -> None:
        """Evict a device leaf. Backed-up nodes are demoted to tombstones
        instead of deleted, so later matches can reload them from host."""
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"

        # write_back: do the D2H backup BEFORE freeing device pages, otherwise
        # the gather would read reclaimed pages. _donation_barrier is only set
        # by the overlap scheduler; in non-overlap mode it's None, which is safe
        # because no forward is in flight during get_next_batch_to_run.
        if self.hicache_enabled and self.write_policy == "write_back" and not node.backuped:
            if self._donation_barrier is not None:
                self._donation_barrier()
            self.write_backup(node, write_back=True)

        for component in self._components_tuple:
            if component.node_has_component_data(node, EvictLayer.DEVICE):
                freed, _ = component.evict_component(node, EvictLayer.DEVICE)
                tracker[component.component_type] += freed

        self._cascade_evict(node)
        self.evictable_device_leaves.discard(node)

        if self.hicache_enabled and node.backuped:
            # Demote to host tier: keep the tombstone in-tree.
            self._update_evictable_leaf_sets(node)
            self._update_evictable_leaf_sets(node.parent)
            return

        parent = node.parent
        self._remove_leaf_from_parent(node)
        self._update_evictable_leaf_sets(parent)

    def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        """D-leaf: FULL device value present, no child with FULL KV on device,
        unlocked, not root. Auxiliary components are not required."""
        ct = BASE_COMPONENT_TYPE
        if node is self.root_node or node.evicted:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        return not any(
            child.component_data[ct].value is not None for child in node.children.values()
        )

    def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        """H-leaf: evicted-but-backuped tombstone with no backed-up child."""
        if node is self.root_node:
            return False
        if not (node.evicted and node.backuped):
            return False
        return not any(child.backuped for child in node.children.values())

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)
        if self.hicache_enabled:
            if self._is_host_leaf(node):
                self.evictable_host_leaves.add(node)
            else:
                self.evictable_host_leaves.discard(node)

    ##### HiCache (L1<->L2) Host Eviction #####

    def evict_host(self, num_pages: int) -> int:
        """Free at least num_pages host slots by evicting host-tier LRU leaves."""
        if not self.hicache_enabled:
            return 0
        num_freed = 0
        heap = list(self.evictable_host_leaves)
        heapq.heapify(heap)
        while num_freed < num_pages and heap:
            node = heapq.heappop(heap)
            if node not in self.evictable_host_leaves:
                continue
            parent = node.parent
            num_freed += self._evict_host_leaf(node)
            if parent is not None and self._is_host_leaf(parent):
                heapq.heappush(heap, parent)
        return num_freed

    def _evict_host_leaf(self, node: UnifiedTreeNode) -> int:
        """Release a host leaf's page ids and delete it from the tree."""
        cd = node.component_data[BASE_COMPONENT_TYPE]
        buffer_ids = cd.host_value
        num = len(buffer_ids)
        self.hicache_controller.evict_callback([int(b) for b in buffer_ids])
        cd.host_value = None
        self.evictable_host_leaves.discard(node)
        parent = node.parent
        self._remove_leaf_from_parent(node)
        self._update_evictable_leaf_sets(parent)
        return num

    ##### HiCache (L1<->L2) Load Path #####

    def init_load_back(
        self,
        last_host_node: UnifiedTreeNode,
        host_hit_length: int,
        mem_quota: int | None = None,
    ) -> tuple[np.ndarray, UnifiedTreeNode, list[tuple[list[int], list[int]]]]:
        """Start async reload of a host-only prefix onto device (H2D).

        Walks the tombstone chain from last_host_node up to the device boundary,
        allocates device slots, and submits async stage_load for each node.
        Returns (device_indices, deepest_reloaded_node, flush_plan) — the caller
        must hand flush_plan to finish_load_back in a donation-safe window to
        complete the kernel scatter into kv_buffer.
        """
        if not self.hicache_enabled or host_hit_length <= 0:
            return np.empty((0,), dtype=np.int32), last_host_node, []

        chain: list[UnifiedTreeNode] = []
        node = last_host_node
        while node is not self.root_node and node.evicted and node.backuped:
            chain.append(node)
            node = node.parent
        chain.reverse()
        if not chain:
            return np.empty((0,), dtype=np.int32), last_host_node, []
        attach_boundary = node

        # All-or-nothing: reload the entire tombstone chain or none of it.
        # Partial reload breaks the contiguous-prefix invariant (live node below
        # a tombstone), which inc_lock_ref and evictable accounting rely on.
        PS = self.page_size
        selected = list(chain)
        total = sum(len(n.component_data[BASE_COMPONENT_TYPE].host_value) * PS for n in selected)
        if mem_quota is not None and total > mem_quota:
            return np.empty((0,), dtype=np.int32), last_host_node, []

        dp_rank = (
            last_host_node.key.dp_rank
            if last_host_node.key and last_host_node.key.dp_rank is not None
            else 0
        )

        # Lock attach_boundary across eviction so it isn't demoted (it's the
        # device leaf whose sole child is the tombstone we're reloading).
        lock_res = self.inc_lock_ref(attach_boundary)
        try:
            avail = self.token_to_kv_pool_allocator.available_size(dp_rank)
            if avail < total:
                self.evict(EvictParams(num_tokens=total - avail, dp_rank=dp_rank))
            device_indices_all = self.token_to_kv_pool_allocator.alloc(total, dp_rank=dp_rank)
        finally:
            self.dec_lock_ref(attach_boundary, lock_res.to_dec_params())
        if device_indices_all is None:
            return np.empty((0,), dtype=np.int32), last_host_node, []

        offset = 0
        flush_plan: list[tuple[list[int], list[int]]] = []
        for n in selected:
            cd = n.component_data[BASE_COMPONENT_TYPE]
            host_pages = [int(b) for b in cd.host_value]
            n_tokens = len(host_pages) * PS
            dev_tokens = device_indices_all[offset : offset + n_tokens]
            dev_pages = np.asarray(dev_tokens)[::PS] // PS
            global_pages = self._to_global_device_pages(dev_pages, dp_rank)
            self.hicache_controller.stage_load(host_pages)
            flush_plan.append((host_pages, [int(p) for p in global_pages]))
            # Un-tombstone: restore device value, retain host copy (write-through).
            cd.value = np.array(dev_tokens, dtype=np.int32)
            node_dp_rank = n.key.dp_rank if n.key and n.key.dp_rank is not None else 0
            self.component_evictable_size_[BASE_COMPONENT_TYPE][node_dp_rank] += n_tokens
            offset += n_tokens

        for n in selected:
            self._update_evictable_leaf_sets(n)
        self._update_evictable_leaf_sets(selected[0].parent)

        return device_indices_all, selected[-1], flush_plan

    def finish_load_back(self, flush_plan: list[tuple[list[int], list[int]]]) -> None:
        """Complete the H2D scatter into kv_buffer. Must run donation-safe."""
        if not self.hicache_enabled or not flush_plan:
            return
        self.hicache_controller.drain_loads()
        for host_pages, global_pages in flush_plan:
            self.hicache_controller.flush_load(host_pages, list(global_pages))

    def _print_helper(self, node: UnifiedTreeNode, indent: int) -> None:
        cd = node.component_data[BASE_COMPONENT_TYPE]
        value_info = f"len={len(cd.value)}" if cd.value is not None else "evicted"
        print(
            " " * indent,
            len(node.key) if node.key else 0,
            node.key.token_ids[:10] if node.key else [],
            f"r={cd.lock_ref}",
            f"extra_key={node.key.extra_key if node.key else None}",
            value_info,
        )
        for child in node.children.values():
            self._print_helper(child, indent + 2)
