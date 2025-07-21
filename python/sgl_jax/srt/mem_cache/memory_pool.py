import abc
import logging
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


class ReqToTokenPool:
    def __init__(
        self,
        size: int,
        max_context_len: int,
        mesh: Mesh,
        dtype: jnp.dtype = jnp.int32,
        token_partition_axis: str = "data",
    ):
        self.size = size
        self.max_context_len = max_context_len
        self.mesh = mesh
        self.dtype = dtype

        # Create sharded request to token mapping table
        self.req_to_token = jnp.zeros(
            (size, max_context_len), dtype=dtype
        )
        
        # Use data sharding strategy
        self.token_sharding = NamedSharding(mesh, P(token_partition_axis, None))
        self.req_to_token = jax.device_put(self.req_to_token, self.token_sharding)

        # Use simple list to manage free slots (non-JAX array)
        self.free_slots = list(range(size))

    def write(self, indices, values):
        """Write token indices to specified request slots"""
        if isinstance(indices, tuple) and len(indices) == 2:
            # Handle (req_idx, slice) case
            req_idx, slice_obj = indices
            self.req_to_token = self.req_to_token.at[req_idx, slice_obj].set(values)
        else:
            # Handle direct indexing case
            self.req_to_token = self.req_to_token.at[indices].set(values)

    def read(self, req_idx: int, length: int) -> jnp.ndarray:
        """Read token indices from specified request slot"""
        return self.req_to_token[req_idx, :length]

    def available_size(self) -> int:
        """Return number of available request slots"""
        return len(self.free_slots)

    def alloc(self, need_size: int = 1) -> Optional[Union[int, List[int]]]:
        """Allocate request slots"""
        if need_size > len(self.free_slots):
            return None

        if need_size == 1:
            return self.free_slots.pop(0)
        else:
            select_indices = self.free_slots[:need_size]
            self.free_slots = self.free_slots[need_size:]
            return select_indices

    def free(self, free_index: Union[int, List[int]]):
        """Free request slots"""
        if isinstance(free_index, int):
            self.free_slots.append(free_index)
            # Clear corresponding memory region
            self.req_to_token = self.req_to_token.at[free_index].set(0)
        else:
            self.free_slots.extend(free_index)
            # Batch clear
            for idx in free_index:
                self.req_to_token = self.req_to_token.at[idx].set(0)

    def clear(self):
        """Clear all allocation states"""
        self.free_slots = list(range(self.size))
        self.req_to_token = jnp.zeros((self.size, self.max_context_len), dtype=self.dtype)
        self.req_to_token = jax.device_put(self.req_to_token, self.token_sharding)


class KVCache(abc.ABC):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        layer_num: int,
        mesh: Mesh,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.layer_num = layer_num
        self.mesh = mesh
        self.start_layer = start_layer or 0
        self.end_layer = end_layer or layer_num - 1
        self.mem_usage = 0

    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> jnp.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> jnp.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jnp.ndarray,
        cache_k: jnp.ndarray,
        cache_v: jnp.ndarray,
    ) -> None:
        raise NotImplementedError()

    def get_kv_size_bytes(self):
        """Calculate KV cache size in bytes"""
        raise NotImplementedError()

    def get_cpu_copy(self, indices):
        """Get CPU copy of KV cache for specified indices"""
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices):
        """Load CPU copy back to device"""
        raise NotImplementedError()


class MHATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        mesh: Mesh,
        kv_partition_axis: str = "tensor",
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size, page_size, dtype, layer_num, mesh, start_layer, end_layer
        )
        self.head_num = head_num
        self.head_dim = head_dim
        self.kv_partition_axis = kv_partition_axis

        self._create_buffers()
        self._calculate_memory_usage()

    def _create_buffers(self):
        """Create sharded KV cache buffers"""
        # KV sharding strategy: shard along head dimension
        self.kv_sharding = NamedSharding(
            self.mesh, P(None, None, self.kv_partition_axis, None)
        )

        # Create K and V buffers [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens
        with self.mesh:
            self.k_buffer = []
            self.v_buffer = []
            
            for _ in range(self.layer_num):
                k_buf = jnp.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.dtype
                )
                v_buf = jnp.zeros(
                    (self.size + self.page_size, self.head_num, self.head_dim),
                    dtype=self.dtype
                )
                
                # Apply sharding
                k_buf = jax.device_put(k_buf, self.kv_sharding)
                v_buf = jax.device_put(v_buf, self.kv_sharding)
                
                self.k_buffer.append(k_buf)
                self.v_buffer.append(v_buf)

    def _calculate_memory_usage(self):
        """Calculate memory usage"""
        bytes_per_element = 2 if self.dtype == jnp.bfloat16 else 4
        k_size = self.size * self.head_num * self.head_dim * bytes_per_element * self.layer_num
        v_size = k_size  # K and V have same size
        self.mem_usage = (k_size + v_size) / GB
        
        logger.info(
            f"JAX KV Cache allocated. #tokens: {self.size}, "
            f"K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )

    def get_kv_size_bytes(self):
        """Calculate KV cache size in bytes"""
        bytes_per_element = 2 if self.dtype == jnp.bfloat16 else 4
        k_size = self.size * self.head_num * self.head_dim * bytes_per_element * self.layer_num
        v_size = k_size
        return k_size, v_size

    def get_key_buffer(self, layer_id: int) -> jnp.ndarray:
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int) -> jnp.ndarray:
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jnp.ndarray,
        cache_k: jnp.ndarray,
        cache_v: jnp.ndarray,
    ) -> None:
        """Set KV cache data"""
        layer_idx = layer_id - self.start_layer
        self.k_buffer[layer_idx] = self.k_buffer[layer_idx].at[loc].set(cache_k)
        self.v_buffer[layer_idx] = self.v_buffer[layer_idx].at[loc].set(cache_v)

    def get_kv_data(self, layer_id: int, indices: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get KV data at specified positions"""
        layer_idx = layer_id - self.start_layer
        k_data = self.k_buffer[layer_idx][indices]
        v_data = self.v_buffer[layer_idx][indices]
        return k_data, v_data

    def get_cpu_copy(self, indices):
        """Get CPU copy of KV cache for specified indices"""
        # JAX equivalent would be transferring to host
        kv_cache_host = []
        for layer_id in range(self.layer_num):
            k_host = jax.device_get(self.k_buffer[layer_id][indices])
            v_host = jax.device_get(self.v_buffer[layer_id][indices])
            kv_cache_host.append([k_host, v_host])
        return kv_cache_host

    def load_cpu_copy(self, kv_cache_host, indices):
        """Load host copy back to device"""
        for layer_id in range(self.layer_num):
            k_host, v_host = kv_cache_host[layer_id]
            k_device = jax.device_put(k_host, self.kv_sharding)
            v_device = jax.device_put(v_host, self.kv_sharding)
            self.k_buffer[layer_id] = self.k_buffer[layer_id].at[indices].set(k_device)
            self.v_buffer[layer_id] = self.v_buffer[layer_id].at[indices].set(v_device)

    def move_kv_cache(self, tgt_loc: jnp.ndarray, src_loc: jnp.ndarray):
        """Move KV cache from source locations to target locations"""
        for layer_id in range(self.layer_num):
            # Get data from source locations
            k_data = self.k_buffer[layer_id][src_loc]
            v_data = self.v_buffer[layer_id][src_loc]
            
            # Set data to target locations
            self.k_buffer[layer_id] = self.k_buffer[layer_id].at[tgt_loc].set(k_data)
            self.v_buffer[layer_id] = self.v_buffer[layer_id].at[tgt_loc].set(v_data)

    def clear_cache(self, indices: jnp.ndarray):
        """Clear cache at specified indices"""
        for layer_id in range(self.layer_num):
            self.k_buffer[layer_id] = self.k_buffer[layer_id].at[indices].set(0)
            self.v_buffer[layer_id] = self.v_buffer[layer_id].at[indices].set(0)


class MLATokenToKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        mesh: Mesh,
        kv_partition_axis: str = "data",
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size, page_size, dtype, layer_num, mesh, start_layer, end_layer
        )
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_partition_axis = kv_partition_axis

        self._create_buffers()
        self._calculate_memory_usage()

    def _create_buffers(self):
        """Create KV buffers for MLA"""
        # MLA sharding strategy
        self.kv_sharding = NamedSharding(self.mesh, P(self.kv_partition_axis, None, None))
        
        with self.mesh:
            # The padded slot 0 is used for writing dummy outputs from padded tokens
            self.kv_buffer = []
            for _ in range(self.layer_num):
                kv_buf = jnp.zeros(
                    (self.size + self.page_size, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                    dtype=self.dtype
                )
                kv_buf = jax.device_put(kv_buf, self.kv_sharding)
                self.kv_buffer.append(kv_buf)

    def _calculate_memory_usage(self):
        """Calculate memory usage"""
        bytes_per_element = 2 if self.dtype == jnp.bfloat16 else 4
        kv_size = (self.size * (self.kv_lora_rank + self.qk_rope_head_dim) * 
                   bytes_per_element * self.layer_num)
        self.mem_usage = kv_size / GB
        
        logger.info(
            f"JAX MLA KV Cache allocated. #tokens: {self.size}, "
            f"KV size: {kv_size / GB:.2f} GB"
        )

    def get_kv_size_bytes(self):
        """Calculate KV cache size in bytes"""
        bytes_per_element = 2 if self.dtype == jnp.bfloat16 else 4
        kv_size = (self.size * (self.kv_lora_rank + self.qk_rope_head_dim) * 
                   bytes_per_element * self.layer_num)
        return kv_size

    def get_key_buffer(self, layer_id: int) -> jnp.ndarray:
        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int) -> jnp.ndarray:
        # For MLA, value is part of the combined buffer
        return self.kv_buffer[layer_id - self.start_layer][..., :self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jnp.ndarray,
        cache_k: jnp.ndarray,
        cache_v: jnp.ndarray,
    ) -> None:
        """Set KV cache data for MLA"""
        layer_idx = layer_id - self.start_layer
        self.kv_buffer[layer_idx] = self.kv_buffer[layer_idx].at[loc].set(cache_k)

    def set_mla_kv_buffer(
        self,
        layer_id: int,
        loc: jnp.ndarray,
        cache_k_nope: jnp.ndarray,
        cache_k_rope: jnp.ndarray,
    ):
        """Set MLA KV buffer with separate nope and rope components"""
        layer_idx = layer_id - self.start_layer
        # Concatenate nope and rope components
        cache_k_combined = jnp.concatenate([cache_k_nope, cache_k_rope], axis=-1)
        self.kv_buffer[layer_idx] = self.kv_buffer[layer_idx].at[loc].set(cache_k_combined)

    def get_cpu_copy(self, indices):
        """Get CPU copy of KV cache for specified indices"""
        kv_cache_host = []
        for layer_id in range(self.layer_num):
            kv_host = jax.device_get(self.kv_buffer[layer_id][indices])
            kv_cache_host.append(kv_host)
        return kv_cache_host

    def load_cpu_copy(self, kv_cache_host, indices):
        """Load host copy back to device"""
        for layer_id in range(self.layer_num):
            kv_host = kv_cache_host[layer_id]
            kv_device = jax.device_put(kv_host, self.kv_sharding)
            self.kv_buffer[layer_id] = self.kv_buffer[layer_id].at[indices].set(kv_device) 
