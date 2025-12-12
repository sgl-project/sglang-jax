from __future__ import annotations

import heapq
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    pass


class TreeNode:
    counter = 0

    def __init__(self, id: int | None = None):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is loading KV cache from host
        self.loading = False
        # store the host indices of KV cache
        self.host_value = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


def _key_match_page_size1(key0: list, key1: list):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: list, key1: list, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


def _convert_to_bigram_key(tokens: list[int]) -> list[tuple[int, int]]:
    # EAGLE uses bigram keys in the radix tree since draft sequence is the one-token-shifted version of target
    # [1, 2, 3, 4] -> [(1,2), (2,3), (3,4)]
    if len(tokens) < 2:
        return []
    if isinstance(tokens[0], tuple):
        return tokens
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def get_child_key(key: list, page_size: int = 1):
    if page_size == 1:
        val = key[0]
        if hasattr(val, "item"):
            val = int(val)
        if isinstance(val, list):
            return tuple(val)
        return val

    res = []
    for x in key[:page_size]:
        if hasattr(x, "item"):
            x = int(x)
        if isinstance(x, list):
            x = tuple(x)
        res.append(x)
    return tuple(res)


class RadixCache(BasePrefixCache):
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
        self.kv_event_queue = []

        self.process_id = jax.process_index()
        self.num_processes = jax.process_count()
        self.local_devices = jax.local_device_count()

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
            # Ensure returning hashable types, convert numpy arrays to Python native types
            self.get_child_key_fn = partial(get_child_key, page_size=page_size)
        self.reset()

    def _create_tokens_data(self, tokens: list[int]) -> np.ndarray:
        if self.disable:
            return np.array(tokens, dtype=np.int32)

        return np.array(tokens, dtype=np.int32)

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: list[int], **kwargs) -> MatchResult:
        key = self.key_convert_fn(key)
        if self.disable or len(key) == 0:
            empty_array = np.empty((0,), dtype=np.int32)

            return MatchResult(
                device_indices=empty_array,
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                host_hit_length=0,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        token_sequences, last_node = self._match_prefix_helper(self.root_node, key)

        if token_sequences:
            valid_tokens = []
            for tokens in token_sequences:
                if tokens is not None and len(tokens) > 0:
                    if isinstance(tokens, (list, tuple)):
                        valid_tokens.extend(tokens)
                    elif isinstance(tokens, np.ndarray):
                        valid_tokens.extend(tokens.tolist())

            if valid_tokens:
                matched_tokens = np.array(valid_tokens, dtype=np.int32)
            else:
                matched_tokens = np.empty((0,), dtype=np.int32)
        else:
            matched_tokens = np.empty((0,), dtype=np.int32)

        return MatchResult(
            device_indices=matched_tokens,
            last_device_node=last_node,
            last_host_node=last_node,
            host_hit_length=0,
        )

    def insert(self, key: list, value=None):
        if self.disable:
            return 0
        key = self.key_convert_fn(key)
        if value is None:
            value = self._create_tokens_data(key)
        elif isinstance(value, list):
            value = self._create_tokens_data(value)

        if self.is_eagle:
            # Make sure the value len equal to the EAGLE bigram key len
            value = value[: len(key)]

        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req):
        """Cache completed requests"""
        all_token_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
        if self.disable:
            kv_indices = self.req_to_token_pool.read(
                req.req_pool_idx,
                all_token_len,
            )
            kv_indices = kv_indices[kv_indices != 0]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:all_token_len]
        # For EAGLE radix cache, we will convert the key to bigram key, e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)], the length will -1. ((len([(1,2), (2,3), (3,4)]) = len([1,2,3,4]) - 1))
        # So for the corresponding kv length should also -1. Then we get the actual_kv_len, and use it to do later calculation and slicing.
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        kv_indices = self.req_to_token_pool.read(req.req_pool_idx, all_token_len)
        kv_indices = kv_indices[kv_indices != 0]

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()

        page_aligned_token_len = page_aligned_len + 1 if self.is_eagle else page_aligned_len
        old_prefix_len = len(req.prefix_indices)
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes over one reference from memory pool
        new_prefix_len = self.insert(token_ids[:page_aligned_token_len], page_aligned_kv_indices)

        self.token_to_kv_pool_allocator.free(kv_indices[old_prefix_len:new_prefix_len])
        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])

        # Remove request slot and release cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req):
        """Cache incomplete requests"""
        if self.disable:
            return

        token_ids = req.fill_ids
        all_token_len = len(token_ids)
        # For EAGLE radix cache, we will convert the key to bigram key, e.g. [1,2,3,4] -> [(1,2), (2,3), (3,4)], the length will -1. ((len([(1,2), (2,3), (3,4)]) = len([1,2,3,4]) - 1))
        # So for the corresponding kv length should also -1. Then we get the actual_kv_len, and use it to do later calculation and slicing.
        actual_kv_len = all_token_len - 1 if self.is_eagle else all_token_len
        kv_indices = self.req_to_token_pool.read(req.req_pool_idx, all_token_len)

        if self.page_size != 1:
            page_aligned_len = actual_kv_len // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].copy()
        else:
            page_aligned_len = actual_kv_len
            page_aligned_kv_indices = kv_indices

        # For EAGLE, the page_aligned_len is for the bigram key, the normal key len should +1
        page_aligned_token_len = page_aligned_len + 1 if self.is_eagle else page_aligned_len
        page_aligned_token_ids = token_ids[:page_aligned_token_len]

        old_prefix_len = len(req.prefix_indices)
        if self.is_eagle and old_prefix_len > req.last_matched_prefix_len:
            # In EAGLE chunked prefill case, the prefix_indices included one unmatched token (kv_indices[actual_kv_len:])
            # Here we -1 to make sure the kv of the unmatched token can be freed correctly to avoid memory leak
            old_prefix_len -= 1

        # Radix Cache takes over one reference from memory pool
        new_prefix_len = self.insert(page_aligned_token_ids, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(kv_indices[old_prefix_len:new_prefix_len])

        # Prefix indices may have been updated, reuse them
        new_match_result = self.match_prefix(page_aligned_token_ids)
        new_indices = new_match_result.device_indices  # cpu
        new_last_node = new_match_result.last_device_node

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_prefix_len, len(new_indices))),
            new_indices[old_prefix_len:],
        )
        req.last_matched_prefix_len = len(new_indices)
        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used later in `PrefillAdder::add_chunked_req`
        if self.page_size != 1:
            # create array on CPU
            req.prefix_indices = np.concat([new_indices, kv_indices[len(new_indices) :]])
        else:
            req.prefix_indices = new_indices
            if self.is_eagle:
                # Attach the kv index of the last token for EAGLE, it can be used in chunked prefill
                req.prefix_indices = np.concatenate([new_indices, kv_indices[actual_kv_len:]])
            else:
                req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        print(f"\n[process {self.process_id}] Radix Tree structure:")
        self._print_helper(self.root_node, 0)
        print(f"total tokens: {self.total_size()}")
        print(f"evictable size: {self.evictable_size_}")
        print(f"protected size: {self.protected_size_}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode, swa_uuid_for_lock: str | None = None):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        return self.protected_size_

    def take_events(self):
        """Atomically takes all events and clears the queue."""
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: list):
        node.last_access_time = time.monotonic()

        child_key = self.get_child_key_fn(key)

        token_sequences = []
        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                token_sequences.append(new_node.value)
                node = new_node
                break
            else:
                token_sequences.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return token_sequences, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].copy()

        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].copy()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    def _insert_helper(self, node: TreeNode, key: list, value):
        if isinstance(value, jnp.ndarray):
            assert value.ndim == 1, "value must be a 1D array"

        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        value_info = ""
        if isinstance(node.value, np.ndarray):
            value_info = f"JAX{node.value.shape}"
        elif node.value:
            value_info = f"len={len(node.value)}"

        print(
            " " * indent,
            len(node.key),
            node.key[:10] if node.key else [],
            f"r={node.lock_ref}",
            value_info,
        )
        for key, child in node.children.items():
            self._print_helper(child, indent + 2)

    def _delete_leaf_no_size_update(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if not child.evicted:
                    stack.append(child)
        return total_size

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list
