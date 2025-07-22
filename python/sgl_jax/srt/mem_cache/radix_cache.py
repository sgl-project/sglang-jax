import heapq
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    pass


class TreeNode:
    counter = 0

    def __init__(self, id: Optional[int] = None):
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

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


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

        self.cpu_device = jax.local_devices(backend="cpu")[0]

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: int(key[0]) if hasattr(key[0], "item") else key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            # Ensure returning hashable types, convert numpy arrays to Python native types
            self.get_child_key_fn = lambda key: tuple(
                int(x) if hasattr(x, "item") else x for x in key[:page_size]
            )

        self._init_cache_strategy()
        self.reset()

    def _init_cache_strategy(self):
        print(f"[process {self.process_id}] init JAX Radix Cache")
        print(f"  cpu_device: {self.cpu_device}")
        print(f"  kv_head_num: {self.kv_head_num}")
        print(f"  head_dim: {self.head_dim}")
        print(f"  layer_num: {self.layer_num}")

    def _create_tokens_data(self, tokens: List[int]) -> jnp.ndarray:
        if self.disable:
            return jnp.array(tokens, dtype=jnp.int32)

        with jax.default_device(self.cpu_device):
            token_array = jnp.array(tokens, dtype=jnp.int32)
            cpu_tokens = jax.device_put(token_array, self.cpu_device)

        return cpu_tokens

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        if self.disable or len(key) == 0:
            # create empty array on CPU
            with jax.default_device(self.cpu_device):
                empty_array = jnp.empty((0,), dtype=jnp.int32)
                empty_array = jax.device_put(empty_array, self.cpu_device)

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
                    elif isinstance(tokens, jnp.ndarray):
                        valid_tokens.extend(tokens.tolist())

            if valid_tokens:
                # create array on CPU
                with jax.default_device(self.cpu_device):
                    matched_tokens = jnp.array(valid_tokens, dtype=jnp.int32)
                    matched_tokens = jax.device_put(matched_tokens, self.cpu_device)
            else:
                with jax.default_device(self.cpu_device):
                    matched_tokens = jnp.empty((0,), dtype=jnp.int32)
                    matched_tokens = jax.device_put(matched_tokens, self.cpu_device)
        else:
            with jax.default_device(self.cpu_device):
                matched_tokens = jnp.empty((0,), dtype=jnp.int32)
                matched_tokens = jax.device_put(matched_tokens, self.cpu_device)

        return MatchResult(
            device_indices=matched_tokens,
            last_device_node=last_node,
            last_host_node=last_node,
            host_hit_length=0,
        )

    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = self._create_tokens_data(key)
        elif isinstance(value, list):
            value = self._create_tokens_data(value)

        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req):
        """Cache completed requests"""
        if self.disable:
            kv_indices = self.req_to_token_pool.read(
                req.req_pool_idx,
                len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0),
            )
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.read(req.req_pool_idx, len(token_ids))

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len]
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices

        # Radix Cache takes over one reference from memory pool
        new_prefix_len = self.insert(token_ids[:page_aligned_len], page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # Remove request slot and release cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req):
        """Cache incomplete requests"""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.read(req.req_pool_idx, len(token_ids))

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len]
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices
        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Radix Cache takes over one reference from memory pool
        new_prefix_len = self.insert(page_aligned_token_ids, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # Prefix indices may have been updated, reuse them
        new_match_result = self.match_prefix(page_aligned_token_ids)
        new_indices = new_match_result.device_indices  # cpu
        new_last_node = new_match_result.last_device_node

        new_indices_device = device_array(self.req_to_token_pool.mesh, np.asarray(new_indices))

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices_device[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used later in `PrefillAdder::add_chunked_req`
        if self.page_size != 1:
            # create array on CPU
            with jax.default_device(self.cpu_device):
                req.prefix_indices = jnp.concatenate([new_indices, kv_indices[len(new_indices) :]])
                req.prefix_indices = jax.device_put(req.prefix_indices, self.cpu_device)
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def get_cached_kv(self, token_ids: List[int]) -> Tuple[jnp.ndarray, int]:
        if self.disable:
            # create empty array on CPU
            with jax.default_device(self.cpu_device):
                empty_kv = jnp.empty(
                    (self.layer_num, 0, self.kv_head_num, self.head_dim),
                    dtype=self.dtype,
                )
                empty_kv = jax.device_put(empty_kv, self.cpu_device)
            return (empty_kv, 0)

        match_result = self.match_prefix(token_ids)
        matched_tokens = match_result.device_indices
        last_node = match_result.last_device_node
        matched_len = len(matched_tokens)

        if matched_len == 0:
            # create empty array on CPU
            with jax.default_device(self.cpu_device):
                empty_kv = jnp.empty(
                    (self.layer_num, 0, self.kv_head_num, self.head_dim),
                    dtype=self.dtype,
                )
                empty_kv = jax.device_put(empty_kv, self.cpu_device)
            return (empty_kv, 0)

        # RadixCache stores token indices, not KV data directly
        # We need to get the actual KV data from the KV pool using matched token indices
        if matched_len == 0:
            # No matched tokens, return empty KV data
            with jax.default_device(self.cpu_device):
                kv_data = jnp.empty(
                    (self.layer_num, 0, self.kv_head_num, self.head_dim), dtype=self.dtype
                )
                kv_data = jax.device_put(kv_data, self.cpu_device)
        else:
            # get CPU copy of KV cache
            kv_cache = self.token_to_kv_pool_allocator.get_kvcache()

            # convert matched_tokens to numpy array for indexing
            matched_tokens_np = jax.device_get(matched_tokens)

            # get CPU copy
            kv_cache_cpu = kv_cache.get_cpu_copy(matched_tokens_np)

            # build result on CPU
            k_data_list = []
            v_data_list = []

            for layer_id in range(self.layer_num):
                k_host, v_host = kv_cache_cpu[layer_id]
                k_data_list.append(k_host)
                v_data_list.append(v_host)

            # stack data on CPU
            with jax.default_device(self.cpu_device):
                k_data = jnp.stack(
                    k_data_list, axis=0
                )  # (layer_num, matched_len, head_num, head_dim)
                v_data = jnp.stack(
                    v_data_list, axis=0
                )  # (layer_num, matched_len, head_num, head_dim)

                # For this implementation, we return K data (could also return concatenated K,V)
                kv_data = k_data
                kv_data = jax.device_put(kv_data, self.cpu_device)

        return kv_data, matched_len

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

            # Release KV cache memory
            if x.value is not None and hasattr(x.value, "__len__"):
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
                node_size = self._get_node_size(node)
                self.evictable_size_ -= node_size
                self.protected_size_ += node_size
                delta -= node_size
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode, swa_uuid_for_lock: Optional[str] = None):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                node_size = self._get_node_size(node)
                self.evictable_size_ += node_size
                self.protected_size_ -= node_size
                delta += node_size
            node.lock_ref -= 1
            node = node.parent
        return delta

    def _get_node_size(self, node: TreeNode) -> int:
        if node.value is None:
            return 0
        if isinstance(node.value, jnp.ndarray):
            return node.value.shape[1] if node.value.ndim >= 2 else node.value.shape[0]
        else:
            return len(node.value) if node.value else 0

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

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()

        child_key = self.get_child_key_fn(key)

        token_sequences = []
        while len(key) > 0 and child_key in node.children.keys():
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

        if isinstance(child.value, jnp.ndarray) and child.value.ndim >= 2:
            new_node.value = (
                child.value[:, :split_len, :, :]
                if child.value.ndim == 4
                else child.value[:split_len]
            )
            child.value = (
                child.value[:, split_len:, :, :]
                if child.value.ndim == 4
                else child.value[split_len:]
            )
        else:
            # Handle non-ndarray values (lists, None, etc.)
            if child.value is not None and len(child.value) > 0:
                new_node.value = child.value[:split_len]
                child.value = child.value[split_len:]
            else:
                new_node.value = []
                child.value = []

        child.parent = new_node
        child.key = child.key[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]

            if isinstance(value, jnp.ndarray) and value.ndim >= 2:
                value = value[:, prefix_len:, :, :] if value.ndim == 4 else value[prefix_len:]
            else:
                # Handle non-ndarray values (lists, None, etc.)
                if value is not None and len(value) > 0:
                    value = value[prefix_len:]
                else:
                    value = []

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

            node_size = self._get_node_size(new_node)
            self.evictable_size_ += node_size

        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        value_info = ""
        if isinstance(node.value, jnp.ndarray):
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
        node_size = self._get_node_size(node)
        self.evictable_size_ -= node_size

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += self._get_node_size(current_node)
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
