import abc
import logging
import os
import time
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


@register_pytree_node_class
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
        self.req_to_token = jnp.zeros((size, max_context_len), dtype=dtype)

        # Use data sharding strategy
        self.token_sharding = NamedSharding(mesh, P(None, None))
        self.req_to_token = jax.device_put(self.req_to_token, self.token_sharding)

        # Use simple list to manage free slots (non-JAX array)
        self.free_slots = list(range(size))

    def tree_flatten(self):
        children = (self.req_to_token,)
        aux_data = {
            "size": self.size,
            "max_context_len": self.max_context_len,
            "mesh": self.mesh,
            "dtype": self.dtype,
            "token_sharding": self.token_sharding,
            "free_slots": self.free_slots,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)

        obj.size = aux_data["size"]
        obj.max_context_len = aux_data["max_context_len"]
        obj.mesh = aux_data["mesh"]
        obj.dtype = aux_data["dtype"]
        obj.token_sharding = aux_data["token_sharding"]
        obj.free_slots = aux_data["free_slots"]

        obj.req_to_token = children[0]

        return obj

    def write(self, indices, values):
        """Write token indices to specified request slots"""
        if isinstance(indices, tuple) and len(indices) == 2:
            # Handle (req_idx, slice) case
            req_idx, slice_obj = indices
            self.req_to_token = self.req_to_token.at[req_idx, slice_obj].set(values)
        else:
            # Handle direct indexing case
            print(f"{indices=} {values=}")
            self.req_to_token = self.req_to_token.at[indices].set(values)

    def read(self, req_idx: int, length: int) -> jnp.ndarray:
        """Read token indices from specified request slot"""
        return self.req_to_token[req_idx, :length]

    def available_size(self) -> int:
        """Return number of available request slots"""
        return len(self.free_slots)

    def alloc(self, need_size: int = 1) -> List[int]:
        """Allocate request slots"""
        if need_size > len(self.free_slots):
            return None

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
        self.req_to_token = jnp.zeros(
            (self.size, self.max_context_len), dtype=self.dtype
        )
        self.req_to_token = jax.device_put(self.req_to_token, self.token_sharding)


@register_pytree_node_class
class KVCache(abc.ABC):
    @abc.abstractmethod
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

    def tree_flatten(self):
        children = ()
        aux_data = {
            "size": self.size,
            "page_size": self.page_size,
            "dtype": self.dtype,
            "layer_num": self.layer_num,
            "mesh": self.mesh,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "mem_usage": self.mem_usage,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.size = aux_data["size"]
        obj.page_size = aux_data["page_size"]
        obj.dtype = aux_data["dtype"]
        obj.layer_num = aux_data["layer_num"]
        obj.mesh = aux_data["mesh"]
        obj.start_layer = aux_data["start_layer"]
        obj.end_layer = aux_data["end_layer"]
        obj.mem_usage = aux_data["mem_usage"]
        return obj

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
        is_decode: bool,
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


@register_pytree_node_class
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

    def tree_flatten(self):
        parent_children, parent_aux_data = super().tree_flatten()

        children = (self.k_buffer, self.v_buffer) + parent_children
        aux_data = {
            **parent_aux_data,
            "head_num": self.head_num,
            "head_dim": self.head_dim,
            "kv_partition_axis": self.kv_partition_axis,
            "kv_sharding": self.kv_sharding,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        k_buffer, v_buffer = children[0], children[1]
        parent_children = children[2:] if len(children) > 2 else ()

        obj = object.__new__(cls)

        parent_obj = super().tree_unflatten(aux_data, parent_children)
        for attr in [
            "size",
            "page_size",
            "dtype",
            "layer_num",
            "mesh",
            "start_layer",
            "end_layer",
            "mem_usage",
        ]:
            setattr(obj, attr, getattr(parent_obj, attr))

        obj.head_num = aux_data["head_num"]
        obj.head_dim = aux_data["head_dim"]
        obj.kv_partition_axis = aux_data["kv_partition_axis"]
        obj.kv_sharding = aux_data["kv_sharding"]

        obj.k_buffer = k_buffer
        obj.v_buffer = v_buffer

        return obj

    def _create_buffers(self):
        """Create sharded KV cache buffers with proper distributed allocation"""
        self.kv_sharding = NamedSharding(self.mesh, P(None, self.kv_partition_axis))

        print(f"Creating buffers for {self.layer_num} layers")
        start_time = time.time()

        buffer_shape = (self.size + self.page_size, self.head_num, self.head_dim)
        print(
            f"Total KV cachememory per layer: {buffer_shape[0] * buffer_shape[1] * buffer_shape[2] * 2 / 1024**3:.2f} GB, dtype: {self.dtype}"
        )
        with self.mesh:
            self.k_buffer = []
            self.v_buffer = []

            for _ in range(self.layer_num):
                tensor_size = self.mesh.shape.get(self.kv_partition_axis, 1)

                def create_kv_callback(index):
                    local_shape = (
                        buffer_shape[0],
                        buffer_shape[1] // tensor_size,
                        buffer_shape[2],
                    )
                    return jnp.zeros(local_shape, dtype=self.dtype)

                k_buf = jax.make_array_from_callback(
                    buffer_shape, self.kv_sharding, create_kv_callback
                )
                v_buf = jax.make_array_from_callback(
                    buffer_shape, self.kv_sharding, create_kv_callback
                )
                self.k_buffer.append(k_buf)
                self.v_buffer.append(v_buf)

        end_time = time.time()
        print(
            f"Total time to create {self.layer_num} buffers: {end_time - start_time:.2f} seconds"
        )

    def _calculate_memory_usage(self):
        """Calculate memory usage"""
        k_size = (
            (self.size + self.page_size)
            * self.head_num
            * self.head_dim
            * jnp.dtype(self.dtype).itemsize
            * self.layer_num
        )
        v_size = k_size  # K and V have same size
        self.mem_usage = (k_size + v_size) / GB

        logger.info(
            f"JAX KV Cache allocated. #tokens: {self.size}, "
            f"K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )

    def get_kv_size_bytes(self):
        """Calculate KV cache size in bytes"""
        k_size = (
            (self.size + self.page_size)
            * self.head_num
            * self.head_dim
            * jnp.dtype(self.dtype).itemsize
            * self.layer_num
        )
        v_size = k_size
        return k_size, v_size

    def get_key_buffer(self, layer_id: int) -> jnp.ndarray:
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int) -> jnp.ndarray:
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return _get_kv_buffer(layer_id, self.k_buffer, self.v_buffer)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jax.Array,
        k: jax.Array,  # [total_tokens, num_heads, head_dim]
        v: jax.Array,  # [total_tokens, num_heads, head_dim]
        is_decode: bool = False,
    ) -> None:
        """
        Set KV cache data using JAX-style interface with padding support.
        This method uses the token-by-token update approach to avoid contiguity issues.

        Args:
            layer_id: Which layer to update
            k: Key tensor [total_tokens, num_heads, head_dim]
            v: Value tensor [total_tokens, num_heads, head_dim]
            seq_lens: Sequence lengths [batch_size]
            kv_start_loc: Start positions in k,v tensors [batch_size]
            kv_cache_start_loc: Start positions in cache [batch_size]
        """
        layer_idx = layer_id - self.start_layer

        page_size = 1 if is_decode else self.page_size
        # Use the token-by-token update implementation
        self.k_buffer[layer_idx], self.v_buffer[layer_idx] = _set_kv_buffer(
            k=k,
            v=v,
            loc=loc,
            k_cache=self.k_buffer[layer_idx],
            v_cache=self.v_buffer[layer_idx],
            page_size=page_size,
            kv_partition_axis=self.kv_partition_axis,
        )

    def get_kv_data(
        self, layer_id: int, indices: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    def set_kv_buffer_legacy(
        self,
        layer_id: int,
        loc: jnp.ndarray,
        cache_k: jnp.ndarray,
        cache_v: jnp.ndarray,
    ) -> None:
        """
        Legacy interface for backward compatibility.
        This assumes contiguous cache locations and uses simple JAX operations.
        """
        layer_idx = layer_id - self.start_layer
        self.k_buffer[layer_idx] = self.k_buffer[layer_idx].at[loc].set(cache_k)
        self.v_buffer[layer_idx] = self.v_buffer[layer_idx].at[loc].set(cache_v)


def _set_kv_buffer(
    k: jax.Array,
    v: jax.Array,
    loc: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    page_size: int,
    kv_partition_axis: str = "tensor",
):
    """
    k: jax.Array,          # [total_tokens, num_heads, head_dim]
    v: jax.Array,          # [total_tokens, num_heads, head_dim]
    loc: jax.Array,        # [total_tokens] total_tokens is the padding tokens, if the value is -1, it means the token is padding
    k_cache: jax.Array,
    v_cache: jax.Array,
    """
    k_cache, v_cache = update_kv_cache(
        k,
        v,
        loc,
        k_cache,
        v_cache,
        page_size=page_size,
        kv_partition_axis=kv_partition_axis,
    )

    return k_cache, v_cache


def update_kv_cache(
    k: jax.Array,  # [total_tokens, num_heads, head_dim]
    v: jax.Array,  # [total_tokens, num_heads, head_dim]
    loc: jax.Array,  # [total_tokens], -1 for padding
    k_cache: jax.Array,
    v_cache: jax.Array,
    page_size: int = 1,
    kv_partition_axis: str = "tensor",
):
    """
    Main KV cache update function that chooses between vectorized and token-by-token approaches.

    Args:
        k: Key tensor [total_tokens, num_heads, head_dim]
        v: Value tensor [total_tokens, num_heads, head_dim]
        loc: Location indices [total_tokens], -1 for padding tokens
        k_cache: Key cache buffer
        v_cache: Value cache buffer
        use_vectorized: Whether to use vectorized (True) or token-by-token (False) approach

    Returns:
        Updated k_cache and v_cache
    """
    return update_kv_cache_vectorized(
        k,
        v,
        loc,
        k_cache,
        v_cache,
        page_size=page_size,
        kv_partition_axis=kv_partition_axis,
    )


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def _optimize_contiguous_updates(
    loc: jax.Array,  # [total_tokens], -1 for padding
    page_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """
    Optimize KV cache updates by grouping contiguous locations into page_size chunks.

    Args:
        loc: Location index array [total_tokens], -1 for padding tokens
        page_size: Page size (must be > 1)

    Returns:
        kv_cache_locs: Start locations in cache for each slice
        new_kv_locs: Start locations in new_kv for each slice
        slice_lens: Length of each slice (0 means skip this position)
        num_slices: Total number of slices (equals total_tokens for array size consistency)
    """
    total_tokens = loc.shape[0]
    valid_mask = loc != -1
    indices = jnp.arange(total_tokens)

    # Detect contiguous segments
    is_continuous_to_next = valid_mask[:-1] & valid_mask[1:] & (loc[1:] == loc[:-1] + 1)
    is_continuous_to_next = jnp.concatenate([is_continuous_to_next, jnp.array([False])])

    # Mark segment starts
    is_segment_start = valid_mask & jnp.concatenate(
        [
            jnp.array([True]),
            ~is_continuous_to_next[:-1],
        ]
    )

    # Simple approach: compute remaining tokens from each position by looking ahead
    # For each position i, count how many contiguous valid tokens exist starting from i
    remaining_in_segment = jnp.zeros(total_tokens, dtype=jnp.int32)

    # Use scan to compute remaining tokens efficiently
    def compute_remaining(carry, x):
        i, valid, is_cont_next = x
        next_remaining = carry

        # Current remaining = 1 (for self) + next_remaining (if continuous)
        current_remaining = jnp.where(
            valid, 1 + jnp.where(is_cont_next, next_remaining, 0), 0
        )
        return current_remaining, current_remaining

    # Scan backwards to compute remaining tokens
    # We manually reverse inputs and use reverse=False, then reverse the output
    _, remaining_in_segment = jax.lax.scan(
        compute_remaining,
        0,  # initial carry
        (indices[::-1], valid_mask[::-1], is_continuous_to_next[::-1]),
        reverse=False,
    )
    remaining_in_segment = remaining_in_segment[::-1]

    # Now generate slices
    def compute_slice_info(carry, x):
        i, valid, is_seg_start, remaining = x
        tokens_since_segment_start = carry

        # Reset counter at segment start
        new_tokens_since_start = jnp.where(
            valid & is_seg_start,
            0,
            jnp.where(
                valid, tokens_since_segment_start + 1, tokens_since_segment_start
            ),
        )

        # Start slice at segment start or every page_size tokens within segment
        is_slice_start = valid & (
            is_seg_start | (new_tokens_since_start % page_size == 0)
        )

        # Slice length is min(page_size, remaining tokens in segment)
        slice_len = jnp.where(is_slice_start, jnp.minimum(page_size, remaining), 0)

        return new_tokens_since_start, (is_slice_start, slice_len)

    _, (slice_starts, slice_lengths) = jax.lax.scan(
        compute_slice_info,
        -1,  # tokens_since_segment_start
        (indices, valid_mask, is_segment_start, remaining_in_segment),
    )

    kv_cache_locs = jnp.where(slice_starts, loc, 0)
    new_kv_locs = jnp.where(slice_starts, indices, 0)
    slice_lens = slice_lengths

    return kv_cache_locs, new_kv_locs, slice_lens, total_tokens


def kv_cache_update_kernel(
    # Prefetch
    slices_ref,  # [3, padded_num_slices], list of (kv_cache_start, new_kv_start, slice_len)
    # Input
    new_kv_hbm_ref,  # [num_tokens, num_combined_kv_heads, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages * page_size, num_combined_kv_heads,
    # head_dim]
    # Output
    _,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    # Scratch
    scratch,  # [num_slices_per_block, page_size, num_combined_kv_heads,
    # head_dim]
    sem,
):
    async_copies = []
    block_idx = pl.program_id(0)
    num_slices_per_block = scratch.shape[0]
    # Copy from new_kv_hbm_ref to scratch
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        new_kv_start = slices_ref[1, offset_i]
        length = slices_ref[2, offset_i]
        async_copy = pltpu.make_async_copy(
            new_kv_hbm_ref.at[pl.ds(new_kv_start, length), ...],
            scratch.at[jnp.uint32(i), pl.ds(0, length), ...],
            sem,
        )
        async_copy.start()
        async_copies.append(async_copy)

    for async_copy in async_copies:
        async_copy.wait()

    # Copy from scratch to kv_cache_hbm_ref
    async_copies.clear()
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        kv_cache_start = slices_ref[0, offset_i]
        length = slices_ref[2, offset_i]
        async_copy = pltpu.make_async_copy(
            scratch.at[jnp.uint32(i), pl.ds(0, length), ...],
            kv_cache_hbm_ref.at[pl.ds(kv_cache_start, length), ...],
            sem,
        )
        async_copy.start()
        async_copies.append(async_copy)
    for async_copy in async_copies:
        async_copy.wait()


def get_num_slices_per_block(new_kv: jax.Array, kv_cache: jax.Array, page_size=128):
    """
    new_kv: [total_num_token, num_combined_kv_heads, head_dim]
    kv_cache: [max_num_tokens, num_combined_kv_heads, head_dim]
    """
    assert (
        new_kv.dtype == kv_cache.dtype
    ), f"new_kv.dtype={new_kv.dtype} is not equal to kv_cache.dtype={kv_cache.dtype}"
    assert new_kv.dtype != jnp.float16, f"new_kv.dtype={new_kv.dtype} is not supported"

    bits = dtypes.bit_width(kv_cache.dtype)
    assert bits % 8 == 0, f"bits={bits} is not divisible by 8"

    bytes_per_element = bits // 8

    total_num_token = new_kv.shape[0]
    kv_head_num = new_kv.shape[1]
    head_dim = new_kv.shape[2]

    max_num_slices_per_block = VMEM_SIZE // (
        bytes_per_element * page_size * kv_head_num * head_dim
    )
    assert (
        max_num_slices_per_block > 0
    ), f"max_num_slices_per_block={max_num_slices_per_block} is not greater than 0"

    return (
        total_num_token
        if total_num_token < max_num_slices_per_block
        else max_num_slices_per_block
    )


# @partial(
#     jax.jit,
#     static_argnames=["page_size", "num_slices_per_block"],
# )
def kv_cache_update(
    new_kv: jax.Array,  # [total_num_token, num_kv_heads, head_dim]
    # [3, slices], list of (kv_cache_start, new_kv_start, slice_len)
    slices: jax.Array,
    # [max_num_tokens, num_kv_heads, head_dim]
    kv_cache: jax.Array,
    num_kv_update_slices: jax.Array,  # [1]
    *,
    page_size: int = 1,  # because we treat each token as an independent query
    num_slices_per_block: int = 8,
    kv_partition_axis: str = "tensor",
):
    mesh = jax.sharding.get_abstract_mesh()

    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P(
                None, kv_partition_axis, None
            ),  # new_kv - consistent with KV cache sharding
            P(None, None),  # slices
            P(
                None, kv_partition_axis, None
            ),  # kv_cache - consistent with KV cache sharding
            P(None),  # num_kv_update_slices
        ),
        out_specs=P(
            None, kv_partition_axis, None
        ),  # output also maintains KV cache sharding consistency
        check_vma=False,
    )
    def _kv_cache_update_wrapper(new_kv, slices, kv_cache, num_kv_update_slices):
        assert (
            slices.shape[1] % num_slices_per_block == 0
        ), f"slices.shape[1]={slices.shape[1]} is not divisible by num_slices_per_block={num_slices_per_block}"
        _, num_combined_kv_heads, head_dim = new_kv.shape

        assert num_combined_kv_heads % 2 == 0, (
            f"num_combined_kv_heads={num_combined_kv_heads} should be even after pre-padding. "
            "This indicates a configuration issue with kv heads padding."
        )

        assert (
            kv_cache.shape[1] == num_combined_kv_heads
        ), f"kv_cache.shape[1]={kv_cache.shape[1]} is not equal to num_combined_kv_heads={num_combined_kv_heads}"
        assert (
            kv_cache.shape[2] == head_dim
        ), f"kv_cache.shape[2]={kv_cache.shape[2]} is not equal to head_dim={head_dim}"
        assert head_dim % 128 == 0, f"head_dim={head_dim} is not divisible by 128"
        # smaller or equal to page_size

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ]

        out_specs = [pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY)]
        out_shape = [jax.ShapeDtypeStruct(kv_cache.shape, dtype=kv_cache.dtype)]

        scalar_prefetches = [slices]
        scratch = pltpu.VMEM(
            (num_slices_per_block, page_size, num_combined_kv_heads, head_dim),
            new_kv.dtype,
        )

        scratch_shapes = [
            scratch,
            pltpu.SemaphoreType.DMA,
        ]

        kernel = pl.pallas_call(
            kv_cache_update_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(cdiv(num_kv_update_slices[0], num_slices_per_block),),
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shape,
            input_output_aliases={len(scalar_prefetches) + 1: 0},
        )

        result = kernel(*scalar_prefetches, new_kv, kv_cache)[0]

        return result

    return _kv_cache_update_wrapper(new_kv, slices, kv_cache, num_kv_update_slices)


def get_slot_mapping(
    num_slices_per_block: int,
    kv_cache_start_loc: jax.Array,
    new_kv_start_loc: jax.Array,
    slice_lens: jax.Array,
):
    slot_mapping = jnp.stack([kv_cache_start_loc, new_kv_start_loc, slice_lens], axis=1)
    padded_size = (
        (slot_mapping.shape[0] + num_slices_per_block - 1)
        // num_slices_per_block
        * num_slices_per_block
    )
    slot_mapping = jnp.pad(
        slot_mapping,
        [[0, padded_size - slot_mapping.shape[0]], [0, 0]],
        constant_values=0,
    )
    slot_mapping = jnp.transpose(slot_mapping)
    return slot_mapping.astype(jnp.int32)


VMEM_SIZE = 32 * 1024 * 1024  # 32MB
PAGE_SIZE = 1


# @jax.jit
def update_kv_cache_vectorized(
    k: jax.Array,  # [total_tokens, num_heads, head_dim]
    v: jax.Array,  # [total_tokens, num_heads, head_dim]
    loc: jax.Array,  # [total_tokens], -1 for padding
    k_cache: jax.Array,
    v_cache: jax.Array,
    page_size: int,
    kv_partition_axis: str = "tensor",
):
    """
    Vectorized KV cache update that handles padding and supports page_size > 1
    by grouping contiguous tokens into page-sized chunks for efficient updates.
    """
    total_tokens = loc.shape[0]
    loc = loc.astype(jnp.int32)

    # Choose strategy based on page_size
    if page_size > 1:
        # Use optimized contiguous grouping for page_size > 1
        kv_cache_locs, new_kv_locs, slice_lens, num_slices = (
            _optimize_contiguous_updates(loc, page_size)
        )
    else:
        # Use original logic for page_size = 1: one slice per token
        kv_cache_locs = jnp.where(loc == -1, 0, loc).astype(jnp.int32)
        new_kv_locs = jnp.arange(total_tokens, dtype=jnp.int32)
        slice_lens = jnp.where(loc == -1, 0, 1).astype(jnp.int32)
        num_slices = total_tokens

    # num_slices_per_block = get_num_slices_per_block(k, k_cache)
    num_slices_per_block = 4

    slot_mapping = get_slot_mapping(
        num_slices_per_block=num_slices_per_block,
        kv_cache_start_loc=kv_cache_locs,
        new_kv_start_loc=new_kv_locs,
        slice_lens=slice_lens,
    )

    num_kv_update_slices = jnp.array([num_slices], dtype=jnp.int32)

    k_cache = kv_cache_update(
        new_kv=k,
        slices=slot_mapping,
        kv_cache=k_cache,
        num_kv_update_slices=num_kv_update_slices,
        page_size=page_size,
        num_slices_per_block=num_slices_per_block,
        kv_partition_axis=kv_partition_axis,
    )

    v_cache = kv_cache_update(
        new_kv=v,
        slices=slot_mapping,
        kv_cache=v_cache,
        num_kv_update_slices=num_kv_update_slices,
        page_size=page_size,
        num_slices_per_block=num_slices_per_block,
        kv_partition_axis=kv_partition_axis,
    )

    return k_cache, v_cache


# @partial(jax.jit, static_argnames=["layer_id"])
def _get_kv_buffer(
    layer_id: int, k_cache: jax.Array, v_cache: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    return k_cache[layer_id], v_cache[layer_id]


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
        kv_partition_axis: str = "data",  # Note: ignored in MLA, no sharding applied
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
        # MLA sharding strategy - no sharding for MLA KV cache even with TP
        self.kv_sharding = NamedSharding(self.mesh, P(None, None, None))

        with self.mesh:
            # The padded slot 0 is used for writing dummy outputs from padded tokens
            self.kv_buffer = []
            for _ in range(self.layer_num):
                kv_buf = jnp.zeros(
                    (
                        self.size + self.page_size,
                        1,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                    ),
                    dtype=self.dtype,
                )
                kv_buf = jax.device_put(kv_buf, self.kv_sharding)
                self.kv_buffer.append(kv_buf)

    def _calculate_memory_usage(self):
        """Calculate memory usage"""
        kv_size = (
            self.size
            * (self.kv_lora_rank + self.qk_rope_head_dim)
            * jnp.dtype(self.dtype).itemsize
            * self.layer_num
        )
        self.mem_usage = kv_size / GB

        logger.info(
            f"JAX MLA KV Cache allocated. #tokens: {self.size}, "
            f"KV size: {kv_size / GB:.2f} GB"
        )

    def get_kv_size_bytes(self):
        """Calculate KV cache size in bytes"""
        kv_size = (
            self.size
            * (self.kv_lora_rank + self.qk_rope_head_dim)
            * jnp.dtype(self.dtype).itemsize
            * self.layer_num
        )
        return kv_size

    def get_key_buffer(self, layer_id: int) -> jnp.ndarray:
        return self.kv_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int) -> jnp.ndarray:
        # For MLA, value is part of the combined buffer
        return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jnp.ndarray,
        cache_k: jnp.ndarray,
        cache_v: jnp.ndarray,
        is_decode: bool = False,
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
        self.kv_buffer[layer_idx] = (
            self.kv_buffer[layer_idx].at[loc].set(cache_k_combined)
        )

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
            self.kv_buffer[layer_id] = (
                self.kv_buffer[layer_id].at[indices].set(kv_device)
            )
