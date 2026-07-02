"""Unified radix cache: a component-based radix tree over the device KV cache.

Ports upstream sglang's UnifiedRadixCache. Per-component behavior (matching,
locking, eviction, split redistribution) is delegated to TreeComponent
implementations: a FULL-attention component plus a leaf-only recurrent
component (KDA / GDN). SWA / HiCache components plug into the same contract
without touching the core walk.
"""

from __future__ import annotations

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
    RecurrentComponent,
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
    ComponentType.RECURRENT: RecurrentComponent,
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
        enable_recurrent_extra_buffer: bool = False,
        recurrent_track_interval: int | None = None,
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
        self.enable_recurrent_extra_buffer = enable_recurrent_extra_buffer
        self.recurrent_track_interval = recurrent_track_interval
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

        value, best_match_node, best_value_len = self._match_prefix_helper(
            converted_key, full_only=params.full_only
        )
        return self._match_post_processor(params, value, best_match_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

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
            recurrent_num_evicted=tracker.get(ComponentType.RECURRENT, 0),
        )

    def supports_recurrent(self) -> bool:
        return ComponentType.RECURRENT in self.components

    def recurrent_extra_buffer_active(self) -> bool:
        return (
            self.supports_recurrent()
            and self.enable_recurrent_extra_buffer
            and self.recurrent_track_interval is not None
        )

    def recurrent_evictable_size(self, dp_rank: int = 0) -> int:
        """Unlocked tree-owned recurrent slots on ``dp_rank`` — what ``evict``
        can reclaim (protected/locked snapshots excluded)."""
        return self.component_evictable_size_[ComponentType.RECURRENT][dp_rank]

    def assert_recurrent_slot_ledger(self, dp_rank: int = 0, live_reqs: list | None = None) -> int:
        """Per-rank invariant ``active + tree_owned + free == slots_per_rank``;
        returns the derived ``active`` (request-owned) count. Tree-owned =
        recurrent evictable + protected; free = recurrent free-list length.

        When ``live_reqs`` is given, the structurally-derived ``active`` is
        cross-checked against the slots those requests actually hold (running +
        ping-pong track), so a leaked track slot is caught rather than silently
        absorbed into the tautological subtraction."""
        ct = ComponentType.RECURRENT
        rtp = self.req_to_token_pool
        free = len(rtp.recurrent_free_slots[dp_rank])
        tree_owned = (
            self.component_evictable_size_[ct][dp_rank]
            + self.component_protected_size_[ct][dp_rank]
        )
        slots = rtp.slots_per_rank
        active = slots - tree_owned - free
        assert active >= 0, (
            f"recurrent slot ledger broken (dp={dp_rank}): free={free} "
            f"tree_owned={tree_owned} > slots_per_rank={slots}"
        )
        if live_reqs is not None:
            owned = rtp.count_request_owned_recurrent_slots(live_reqs, dp_rank)
            assert owned == active, (
                f"recurrent slot leak (dp={dp_rank}): request-owned={owned} != "
                f"active={active} (free={free} tree_owned={tree_owned} "
                f"slots_per_rank={slots})"
            )
        return active

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

        # cache_protected_len, not len(prefix_indices): the latter may include
        # an unaligned tail owned by the req but not by the tree.
        old_prefix_len = req.cache_protected_len
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In the EAGLE chunked prefill case the prefix indices include one
            # unmatched token; -1 so its kv can be freed without leaking.
            old_prefix_len -= 1

        insert_params = InsertParams() if is_insert else None
        effective_cache_len = len(token_ids)
        if is_insert:
            for component in self._components_tuple:
                cl = component.prepare_for_caching_req(
                    req, insert_params, len(token_ids), is_finished=True
                )
                if cl is not None:
                    effective_cache_len = min(effective_cache_len, cl)

        capped_kv_len = min(actual_kv_len, effective_cache_len)
        if self.page_size != 1:
            page_aligned_len = capped_kv_len // self.page_size * self.page_size
        else:
            page_aligned_len = capped_kv_len
        page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()
        page_aligned_token_len = page_aligned_len + 1 if self.is_eagle else page_aligned_len

        # Tail free is clamped to old_prefix_len so the tree-protected prefix is
        # never freed; it stops at actual_kv_len so the EAGLE +1 token stays
        # request-owned.
        tail_start = max(page_aligned_len, old_prefix_len)
        if self.page_size != 1:
            self.token_to_kv_pool_allocator.free(kv_indices[tail_start:], dp_rank=dp_rank)
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[tail_start:actual_kv_len], dp_rank=dp_rank
            )

        insert_result = None
        if is_insert and effective_cache_len > 0:
            insert_params.key = RadixKey(
                token_ids[:page_aligned_token_len], req.extra_key, req.dp_rank
            )
            insert_params.value = page_aligned_kv_indices
            # Radix cache takes over one reference from the memory pool.
            insert_result = self.insert(insert_params)
            new_prefix_len = insert_result.prefix_len
            self.token_to_kv_pool_allocator.free(
                kv_indices[old_prefix_len:new_prefix_len], dp_rank=dp_rank
            )
        elif not is_insert:
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

        # cache_protected_len, not len(prefix_indices): see cache_finished_req.
        old_prefix_len = req.cache_protected_len
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            old_prefix_len -= 1

        insert_params = InsertParams()
        effective_cache_len = all_token_len
        for component in self._components_tuple:
            cl = component.prepare_for_caching_req(
                req, insert_params, all_token_len, is_finished=False
            )
            if cl is not None:
                effective_cache_len = min(effective_cache_len, cl)

        if effective_cache_len <= 0:
            # Nothing entered the tree, but the chunk's KV is committed: advance
            # prefix_indices so the next chunked round (which does not re-match)
            # extends from it instead of re-allocating over it and leaking the pages.
            req.prefix_indices = kv_indices.copy()
            for component in self._components_tuple:
                component.cleanup_after_caching_req(
                    req, is_finished=False, insert_result=None, insert_params=insert_params
                )
            return

        capped_kv_len = min(actual_kv_len, effective_cache_len)
        if self.page_size != 1:
            page_aligned_len = capped_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()
        else:
            page_aligned_len = capped_kv_len
            page_aligned_kv_indices = kv_indices[:page_aligned_len]

        # For EAGLE, page_aligned_len is for the bigram key; the token len is +1.
        page_aligned_token_len = page_aligned_len + 1 if self.is_eagle else page_aligned_len
        page_aligned_token_ids = token_ids[:page_aligned_token_len]

        radix_key = RadixKey(page_aligned_token_ids, req.extra_key, req.dp_rank)
        insert_params.key = radix_key
        insert_params.value = page_aligned_kv_indices
        # Radix cache takes over one reference from the memory pool.
        insert_result = self.insert(insert_params)
        new_prefix_len = insert_result.prefix_len
        self.token_to_kv_pool_allocator.free(
            kv_indices[old_prefix_len:new_prefix_len], dp_rank=dp_rank
        )

        # Prefix indices may have been updated, reuse them.
        new_match_result = self.match_prefix(MatchPrefixParams(key=radix_key, full_only=True))
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
        # A capped (still-running) request keeps its committed KV tail beyond the
        # cached prefix.
        if self.page_size != 1:
            req.prefix_indices = np.concatenate([new_indices, kv_indices[len(new_indices) :]])
        elif self.is_eagle:
            # Attach the kv index of the last token for EAGLE chunked prefill.
            req.prefix_indices = np.concatenate([new_indices, kv_indices[actual_kv_len:]])
        elif len(new_indices) < all_token_len:
            req.prefix_indices = np.concatenate([new_indices, kv_indices[len(new_indices) :]])
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
        self, key: RadixKey, full_only: bool = False
    ) -> tuple[list[np.ndarray], UnifiedTreeNode, int]:
        node = self.root_node
        node.last_access_time = get_and_increase_time_counter()
        child_key = self.get_child_key_fn(key)

        value: list[np.ndarray] = []
        best_match_node = node
        # Number of value CHUNKS accepted at the best match, not a token
        # count (upstream seam name kept for port parity).
        best_value_len = 0
        # Stage 1 has no host tier: device-only matching is the only mode, so
        # the best match and the best device match coincide.
        if full_only:
            validators = (
                self.components[BASE_COMPONENT_TYPE].create_match_validator(match_device_only=True),
            )
        else:
            validators = tuple(
                comp.create_match_validator(match_device_only=True)
                for comp in self._components_tuple
            )

        def _update_best_if_valid(candidate: UnifiedTreeNode):
            nonlocal best_match_node, best_value_len
            if all(v(candidate) for v in validators):
                best_match_node = candidate
                best_value_len = len(value)

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]

            # Dead node (evicted and not backed up) ends the walk.
            if child.evicted and not child.backuped:
                break

            child.last_access_time = get_and_increase_time_counter()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                if not new_node.evicted:
                    value.append(new_node.component_data[BASE_COMPONENT_TYPE].value)
                _update_best_if_valid(new_node)
                node = new_node
                break

            if not child.evicted:
                value.append(child.component_data[BASE_COMPONENT_TYPE].value)
            node = child
            _update_best_if_valid(node)
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        return value, best_match_node, best_value_len

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: list[np.ndarray],
        best_match_node: UnifiedTreeNode,
        best_value_len: int,
    ) -> MatchResult:
        # Refresh the matched path so deeper nodes look more recently used.
        cur_time = get_and_increase_time_counter()
        node_update: UnifiedTreeNode | None = best_match_node
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
            last_device_node=best_match_node,
            last_host_node=best_match_node,
            best_match_node=best_match_node,
            host_hit_length=0,
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

        for component in self._components_tuple:
            component.on_parent_gains_child(parent)

        self._update_evictable_leaf_sets(new_node)
        self._update_evictable_leaf_sets(parent)
        return new_node

    def _insert_helper(
        self,
        node: UnifiedTreeNode,
        key: RadixKey,
        value: np.ndarray,
        params: InsertParams,
    ) -> InsertResult:
        node.last_access_time = get_and_increase_time_counter()
        if len(key) == 0:
            return InsertResult(prefix_len=0)

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = get_and_increase_time_counter()
            prefix_len = self.key_match_fn(node.key, key)
            if prefix_len < len(node.key):
                node = self._split_node(node.key, node, prefix_len)

            # Let each component claim ownership of overlapping KV slots.
            # FULL never consumes and recurrent does not override this overlap
            # hook; duplicate frees stay in cache_*_req (this repo's
            # convention), so the returned index is currently unused.
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
                return InsertResult(prefix_len=total_prefix_length)
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

        return result

    ##### Evict Helpers #####

    def _cascade_evict(self, node: UnifiedTreeNode) -> None:
        """Tombstone the base value after all components have been driven.

        FULL is the eviction trigger (device-leaf status keys off its value);
        the base-value tombstone is deferred to here -- after every component's
        evict_component has run -- rather than inside FullComponent, so an aux
        component (e.g. recurrent) can still read FULL.value while evicting."""
        node.component_data[BASE_COMPONENT_TYPE].value = None
        self._update_evictable_leaf_sets(node)

    def _remove_leaf_from_parent(self, node: UnifiedTreeNode) -> None:
        child_key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(child_key, None)
        assert v is node

    def _evict_device_leaf(self, node: UnifiedTreeNode, tracker: dict[ComponentType, int]) -> None:
        """Evict a device leaf: free all component device data, tombstone the
        base value, and delete the node from the tree (no host tier yet)."""
        assert self._is_device_leaf(node), f"node {node.id} is not a D-leaf"

        for component in self._components_tuple:
            if component.node_has_component_data(node, EvictLayer.DEVICE):
                freed, _ = component.evict_component(node, EvictLayer.DEVICE)
                tracker[component.component_type] += freed

        self._cascade_evict(node)
        self.evictable_device_leaves.discard(node)
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

    def _update_evictable_leaf_sets(self, node: UnifiedTreeNode) -> None:
        if self._is_device_leaf(node):
            self.evictable_device_leaves.add(node)
        else:
            self.evictable_device_leaves.discard(node)

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
