import heapq
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache, MatchResult
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

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
        mesh: Mesh,
        kv_partition_axis_name: str = "tensor",
        token_partition_axis_name: str = "data",
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
        self.mesh = mesh
        self.page_size = page_size
        self.disable = disable
        self.kv_head_num = kv_head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []

        devices = jax.devices()
        self.kv_partition_spec = P(None, None, None, None) # (layer_num, max_tokens, kv_head_num, head_dim)
        self.token_partition_spec = P()  # (max_tokens)
        if len(devices) > 1:
            self.kv_partition_spec = P(None, None, kv_partition_axis_name, None)
            self.token_partition_spec = P(token_partition_axis_name)

        self.process_id = jax.process_index()
        self.num_processes = jax.process_count()
        self.local_devices = jax.local_device_count()

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])

        self._init_sharding_strategy()
        self.reset()

    def _init_sharding_strategy(self):
        print(f"[process {self.process_id}] init JAX Radix Cache")
        print(f"  mesh: {self.mesh}")
        print(f"  kv_head_num: {self.kv_head_num}")
        print(f"  head_dim: {self.head_dim}")
        print(f"  layer_num: {self.layer_num}")
        
        mesh_axis_names = self.mesh.axis_names
        print(f"  mesh_axis_names: {mesh_axis_names}")
            
        self.kv_cache_sharding = NamedSharding(self.mesh, self.kv_partition_spec)
        self.token_sharding = NamedSharding(self.mesh, self.token_partition_spec)
        
        print(f"  kv_cache_sharding: {self.kv_cache_sharding}")
        print(f"  token_sharding: {self.token_sharding}")

    def _create_sharded_kv_cache(self, shape: Tuple[int, ...]) -> jnp.ndarray:
        if self.disable:
            return jnp.zeros(shape, dtype=self.dtype)
        
        with self.mesh:
            data = jnp.zeros(shape, dtype=self.dtype)
            sharded_data = jax.device_put(data, self.kv_cache_sharding)
            
        return sharded_data

    def _create_sharded_tokens(self, tokens: List[int]) -> jnp.ndarray:
        if self.disable:
            return jnp.array(tokens, dtype=jnp.int32)
        
        token_array = jnp.array(tokens, dtype=jnp.int32)
        
        with self.mesh:
            sharded_tokens = jax.device_put(token_array, NamedSharding(self.mesh, P()))
            
        return sharded_tokens

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        if self.disable or len(key) == 0:
            return MatchResult(
                device_indices=jnp.empty((0,), dtype=jnp.int32),
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
                matched_tokens = jnp.array(valid_tokens, dtype=jnp.int32)
            else:
                matched_tokens = jnp.empty((0,), dtype=jnp.int32)
        else:
            matched_tokens = jnp.empty((0,), dtype=jnp.int32)
            
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
            value = self._create_sharded_tokens(key)
        elif isinstance(value, list):
            value = self._create_sharded_tokens(value)
        
        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req):
        """缓存已完成的请求"""
        if self.disable:
            kv_indices = self.req_to_token_pool.read(
                req.req_pool_idx, 
                len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
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

        # Radix Cache 接管内存池中的一个引用
        new_prefix_len = self.insert(
            token_ids[:page_aligned_len], page_aligned_kv_indices
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices):new_prefix_len]
        )

        # 移除请求槽位并释放缓存锁
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req):
        """缓存未完成的请求"""
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

        # Radix Cache 接管内存池中的一个引用
        new_prefix_len = self.insert(page_aligned_token_ids, page_aligned_kv_indices)
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices):new_prefix_len]
        )

        # 前缀索引可能已更新，重新使用它
        new_match_result = self.match_prefix(page_aligned_token_ids)
        new_indices = new_match_result.device_indices
        new_last_node = new_match_result.last_device_node
        
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices):],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` 将在稍后的 `PrefillAdder::add_chunked_req` 中使用
        if self.page_size != 1:
            req.prefix_indices = jnp.concatenate(
                [new_indices, kv_indices[len(new_indices):]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def get_cached_kv(self, token_ids: List[int]) -> Tuple[jnp.ndarray, int]:
        if self.disable:
            return jnp.empty((self.layer_num, 0, self.kv_head_num, self.head_dim), dtype=self.dtype), 0

        matched_tokens, last_node = self.match_prefix(token_ids)
        matched_len = len(matched_tokens)

        if matched_len == 0:
            return jnp.empty((self.layer_num, 0, self.kv_head_num, self.head_dim), dtype=self.dtype), 0

        kv_data_list = []
        node = last_node
        while node != self.root_node and node.value is not None:
            if isinstance(node.value, jnp.ndarray) and node.value.ndim == 4:
                kv_data_list.append(node.value)
            node = node.parent

        if kv_data_list:
            kv_data_list.reverse()
            kv_data = jnp.concatenate(kv_data_list, axis=1) 
        else:
            kv_data = jnp.empty((self.layer_num, 0, self.kv_head_num, self.head_dim), dtype=self.dtype)

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

            # 释放 KV 缓存内存
            if x.value is not None and hasattr(x.value, '__len__'):
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
                token_sequences.append(new_node.key)
                node = new_node
                break
            else:
                token_sequences.append(child.key)
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
            new_node.value = child.value[:, :split_len, :, :] if child.value.ndim == 4 else child.value[:split_len]
            child.value = child.value[:, split_len:, :, :] if child.value.ndim == 4 else child.value[split_len:]
        else:
            new_node.value = child.value[:split_len] if child.value else []
            child.value = child.value[split_len:] if child.value else []
            
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
                value = value[prefix_len:] if value else []

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
            value_info
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

# if __name__ == "__main__":
#     devices = jax.devices()
#     if len(devices) >= 4:
#         mesh = Mesh(devices[:4], axis_names=('data', 'tensor'))
#     else:
#         mesh = Mesh(devices, axis_names=('data', 'tensor'))
    
#     print(f"mesh: {mesh}")
    
#     # 创建内存池和分配器
#     req_pool = ReqToTokenPool(
#         size=1024, 
#         max_context_len=2048, 
#         mesh=mesh
#     )
    
#     kv_cache = MHATokenToKVPool(
#         size=8192,
#         page_size=1,
#         dtype=jnp.bfloat16,
#         head_num=32,
#         head_dim=128,
#         layer_num=24,
#         mesh=mesh
#     )
    
#     from .allocator import TokenToKVPoolAllocator
#     allocator = TokenToKVPoolAllocator(
#         size=8192,
#         dtype=jnp.bfloat16,
#         kvcache=kv_cache
#     )
    
#     cache = RadixCache(
#         req_to_token_pool=req_pool,
#         token_to_kv_pool_allocator=allocator,
#         mesh=mesh,
#         page_size=1,
#         kv_head_num=32,
#         head_dim=128,
#         layer_num=24,
#         max_seq_len=2048,
#         dtype=jnp.bfloat16
#     )
    
#     print("\n=== JAX RadixCache 测试 ===")
    
#     # 基础功能测试
#     tokens1 = [1, 2, 3, 4, 5]
#     result1 = cache.match_prefix(tokens1)
#     print(f"匹配结果: {result1.device_indices.shape}")
    
#     cache.pretty_print()