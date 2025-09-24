import abc
import logging
import time
from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.utils import cdiv


def merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
    assert (
        k.shape == v.shape
    ), f"k and v must have same shape, got {k.shape} vs {v.shape}"

    num_tokens, num_kv_heads, head_dim = k.shape

    kv_concat = jnp.concatenate([k, v], axis=-1)  # [tokens, heads, head_dim*2]
    kv_fused = kv_concat.reshape(num_tokens, num_kv_heads * 2, head_dim)

    return kv_fused


logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


@register_pytree_node_class
class ReqToTokenPool:
    def __init__(
        self,
        size: int,
        max_context_len: int,
        dtype: np.dtype = np.int32,
    ):
        self.size = size
        self.max_context_len = max_context_len
        self.dtype = dtype

        # Create sharded request to token mapping table
        self.req_to_token = np.zeros((size, max_context_len), dtype=dtype)

        # Use simple list to manage free slots
        self.free_slots = list(range(size))

    def tree_flatten(self):
        children = (self.req_to_token,)
        aux_data = {
            "size": self.size,
            "max_context_len": self.max_context_len,
            "dtype": self.dtype,
            "free_slots": self.free_slots,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)

        obj.size = aux_data["size"]
        obj.max_context_len = aux_data["max_context_len"]
        obj.dtype = aux_data["dtype"]
        obj.free_slots = aux_data["free_slots"]

        obj.req_to_token = children[0]

        return obj

    def write(self, indices, values):
        """Write token indices to specified request slots"""
        if isinstance(indices, tuple) and len(indices) == 2:
            # Handle (req_idx, slice) case
            req_idx, slice_obj = indices
            self.req_to_token[req_idx, slice_obj] = values
        else:
            # Handle direct indexing case
            print(f"{indices=} {values=}")
            self.req_to_token[indices] = values

    def read(self, req_idx: int, length: int) -> np.ndarray:
        """Read token indices from specified request slot"""
        return self.req_to_token[req_idx, :length].copy()

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
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        """Clear all allocation states"""
        self.free_slots = list(range(self.size))


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
    def get_fused_kv_buffer(self, layer_id: int) -> jnp.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get separate K and V buffers for native attention.

        Returns:
            Tuple of (k_buffer, v_buffer)
        """
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
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size, page_size, dtype, layer_num, mesh, start_layer, end_layer
        )
        self.head_num = head_num
        self.head_dim = head_dim
        self.kv_partition_axis = "tensor"

        self._create_buffers()
        self._calculate_memory_usage()

    def tree_flatten(self):
        parent_children, parent_aux_data = super().tree_flatten()

        children = (self.kv_buffer,) + parent_children
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
        kv_buffer = children[0]
        parent_children = children[1:] if len(children) > 1 else ()

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

        obj.kv_buffer = kv_buffer

        return obj

    def _create_buffers(self):
        """Create sharded fused KV cache buffers with proper distributed allocation"""
        self.kv_sharding = NamedSharding(self.mesh, P(None, self.kv_partition_axis))

        logger.info(f"Creating fused KV buffers for {self.layer_num} layers")
        start_time = time.time()

        fused_buffer_shape = (
            self.size + self.page_size,
            self.head_num * 2,  # [K0,V0,K1,V1,...]
            self.head_dim,
        )
        total_memory_per_layer = (
            fused_buffer_shape[0]
            * fused_buffer_shape[1]
            * fused_buffer_shape[2]
            * jnp.dtype(self.dtype).itemsize
        )
        logger.info(
            f"Total fused KV cache memory per layer: {total_memory_per_layer / 1024**3:.2f} GB, dtype: {self.dtype}"
        )
        with self.mesh:
            self.kv_buffer = []
            for _ in range(self.layer_num):
                kv_buf = jax.jit(
                    lambda: jnp.zeros(
                        shape=fused_buffer_shape,
                        dtype=self.dtype,
                    ),
                    out_shardings=self.kv_sharding,
                )()

                self.kv_buffer.append(kv_buf)

        end_time = time.time()
        logger.info(
            f"Total time to create {self.layer_num} buffers: {end_time - start_time:.2f} seconds"
        )

    def _calculate_memory_usage(self):
        """Calculate memory usage for fused KV cache"""
        fused_kv_size = (
            (self.size + self.page_size)
            * self.head_num  # num_kv_heads
            * self.head_dim
            * 2  # num_heads * 2 (head interleaving)
            * jnp.dtype(self.dtype).itemsize
            * self.layer_num
        )
        self.mem_usage = fused_kv_size / GB

        logger.info(
            f"JAX Fused KV Cache allocated. #tokens: {self.size}, "
            f"Fused KV size: {fused_kv_size / GB:.2f} GB"
        )

    def get_kv_size_bytes(self):
        """Calculate KV cache size in bytes for fused format"""
        fused_kv_size = (
            (self.size + self.page_size)
            * self.head_num  # num_kv_heads
            * self.head_dim
            * 2  # num_heads * 2 (head interleaving)
            * jnp.dtype(self.dtype).itemsize
            * self.layer_num
        )
        # For backward compatibility, return as separate k and v sizes
        k_size = fused_kv_size // 2
        v_size = fused_kv_size // 2
        return k_size, v_size

    def get_fused_kv_buffer(self, layer_id: int) -> jnp.ndarray:
        return self.kv_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        layer_idx = layer_id - self.start_layer
        fused_kv = self.kv_buffer[layer_idx]  # [cache_size, num_kv_heads * 2, head_dim]

        # Extract K and V from head interleaving format [K1,V1,K2,V2,...]
        k_buffer = fused_kv[:, ::2, :]  # Even indices: K heads (0, 2, 4, ...)
        v_buffer = fused_kv[:, 1::2, :]  # Odd indices: V heads (1, 3, 5, ...)

        return k_buffer, v_buffer

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jax.Array,
        k: jax.Array,  # [total_tokens, num_heads, head_dim]
        v: jax.Array,  # [total_tokens, num_heads, head_dim]
        is_decode: bool = False,
    ) -> None:
        """
        Set KV cache data using fused KV cache format.

        Args:
            layer_id: Which layer to update
            k: Key tensor [total_tokens, num_heads, head_dim]
            v: Value tensor [total_tokens, num_heads, head_dim]
            loc: Location indices [total_tokens], -1 for padding tokens
            is_decode: Whether this is decode mode
        """
        layer_idx = layer_id - self.start_layer

        page_size = 1 if is_decode else self.page_size

        # Merge k and v into fused format
        fused_kv = merge_kv(k, v)  # [total_tokens, num_heads * 2, head_dim]

        # Update the fused KV cache
        self.kv_buffer[layer_idx] = _set_fused_kv_buffer(
            fused_kv=fused_kv,
            loc=loc,
            kv_cache=self.kv_buffer[layer_idx],
            page_size=page_size,
            kv_partition_axis=self.kv_partition_axis,
        )

    def get_kv_data(
        self, layer_id: int, indices: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get KV data at specified positions"""
        layer_idx = layer_id - self.start_layer
        fused_kv_data = self.kv_buffer[layer_idx][indices]
        k_data = fused_kv_data[:, ::2, :]  # Head interleaving: K at even indices
        v_data = fused_kv_data[:, 1::2, :]  # Head interleaving: V at odd indices
        return k_data, v_data

    def get_cpu_copy(self, indices):
        """Get CPU copy of fused KV cache for specified indices"""
        kv_cache_host = []
        for layer_id in range(self.layer_num):
            fused_kv_host = jax.device_get(self.kv_buffer[layer_id][indices])
            # Extract k and v from fused format using head interleaving
            k_host = fused_kv_host[:, ::2, :]  # Head interleaving: K at even indices
            v_host = fused_kv_host[:, 1::2, :]  # Head interleaving: V at odd indices
            kv_cache_host.append([k_host, v_host])
        return kv_cache_host

    def load_cpu_copy(self, kv_cache_host, indices):
        """Load host copy back to device"""
        for layer_id in range(self.layer_num):
            k_host, v_host = kv_cache_host[layer_id]
            # Merge k and v into fused format
            fused_kv_host = merge_kv(k_host, v_host)
            fused_kv_device = jax.device_put(fused_kv_host, self.kv_sharding)
            self.kv_buffer[layer_id] = (
                self.kv_buffer[layer_id].at[indices].set(fused_kv_device)
            )

    def move_kv_cache(self, tgt_loc: jnp.ndarray, src_loc: jnp.ndarray):
        """Move fused KV cache from source locations to target locations"""
        for layer_id in range(self.layer_num):
            # Get fused KV data from source locations
            fused_kv_data = self.kv_buffer[layer_id][src_loc]
            # Set data to target locations
            self.kv_buffer[layer_id] = (
                self.kv_buffer[layer_id].at[tgt_loc].set(fused_kv_data)
            )

    def clear_cache(self, indices: jnp.ndarray):
        """Clear fused KV cache at specified indices"""
        for layer_id in range(self.layer_num):
            self.kv_buffer[layer_id] = self.kv_buffer[layer_id].at[indices].set(0)

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
        # Merge k and v into fused format
        fused_kv = merge_kv(cache_k, cache_v)
        self.kv_buffer[layer_idx] = self.kv_buffer[layer_idx].at[loc].set(fused_kv)


def _set_fused_kv_buffer(
    fused_kv: jax.Array,
    loc: jax.Array,
    kv_cache: jax.Array,
    page_size: int,
    kv_partition_axis: str = "tensor",
) -> jax.Array:
    """
    Update fused KV cache with new fused KV data.

    Args:
        fused_kv: Fused KV tensor [total_tokens, num_kv_heads * 2, head_dim]
        loc: Location indices [total_tokens], -1 for padding tokens
        kv_cache: Fused KV cache buffer [cache_size, num_kv_heads * 2, head_dim]
        page_size: Page size for vectorized updates
        kv_partition_axis: Partition axis for sharding

    Returns:
        Updated fused KV cache
    """
    return update_fused_kv_cache(
        fused_kv,
        loc,
        kv_cache,
        page_size=page_size,
        kv_partition_axis=kv_partition_axis,
    )


def update_fused_kv_cache(
    fused_kv: jax.Array,  # [total_tokens, num_kv_heads * 2, head_dim]
    loc: jax.Array,  # [total_tokens], -1 for padding
    kv_cache: jax.Array,  # [cache_size, num_kv_heads * 2, head_dim]
    page_size: int = 1,
    kv_partition_axis: str = "tensor",
) -> jax.Array:
    """
    Main fused KV cache update function.

    Args:
        fused_kv: Fused KV tensor [total_tokens, num_kv_heads * 2, head_dim]
        loc: Location indices [total_tokens], -1 for padding tokens
        kv_cache: Fused KV cache buffer
        page_size: Page size for vectorized updates
        kv_partition_axis: Partition axis for sharding

    Returns:
        Updated kv_cache
    """
    return update_fused_kv_cache_vectorized(
        fused_kv,
        loc,
        kv_cache,
        page_size=page_size,
        kv_partition_axis=kv_partition_axis,
    )


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


@partial(
    jax.jit,
    static_argnames=["page_size", "num_slices_per_block", "kv_partition_axis"],
)
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
    @jax.shard_map(
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
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY),
        ]

        out_specs = [pl.BlockSpec(memory_space=pltpu.TPUMemorySpace.ANY)]
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


def update_kv_cache_vectorized(
    k: jax.Array,  # [total_tokens, num_heads, head_dim]
    v: jax.Array,  # [total_tokens, num_heads, head_dim]
    loc: jax.Array,  # [total_tokens], -1 for padding
    k_cache: jax.Array,
    v_cache: jax.Array,
    page_size: int,
    kv_partition_axis: str = "tensor",
    mesh: jax.sharding.Mesh = None,
):
    """
    Vectorized KV cache update that handles padding and supports page_size > 1
    by grouping contiguous tokens into page-sized chunks for efficient updates.
    """
    total_tokens = loc.shape[0]
    loc = loc.astype(jnp.int32)

    # # Choose strategy based on page_size
    # if page_size > 1:
    #     # Use optimized contiguous grouping for page_size > 1
    #     kv_cache_locs, new_kv_locs, slice_lens, num_slices = (
    #         _optimize_contiguous_updates(loc, page_size)
    #     )
    # else:
    # Use original logic for page_size = 1: one slice per token
    kv_cache_locs = jnp.where(loc == -1, 0, loc).astype(jnp.int32)
    new_kv_locs = jnp.arange(total_tokens, dtype=jnp.int32)
    slice_lens = jnp.where(loc == -1, 0, 1).astype(jnp.int32)
    num_slices = total_tokens

    # head_num, cache_len, new_kv_len, head_dim, page_size
    num_slices_per_block = get_best_num_slices_per_block(
        k.shape[1],
        k_cache.shape[0],
        k.shape[0],
        k.shape[2],
        page_size,
    )

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


def update_fused_kv_cache_vectorized(
    fused_kv: jax.Array,  # [total_tokens, num_kv_heads * 2, head_dim]
    loc: jax.Array,  # [total_tokens], -1 for padding
    kv_cache: jax.Array,  # [cache_size, num_kv_heads * 2, head_dim]
    page_size: int,
    kv_partition_axis: str = "tensor",
) -> jax.Array:
    """
    Vectorized fused KV cache update that handles padding and supports page_size > 1
    by grouping contiguous tokens into page-sized chunks for efficient updates.
    """
    total_tokens = loc.shape[0]
    loc = loc.astype(jnp.int32)

    # Use original logic for page_size = 1: one slice per token
    kv_cache_locs = jnp.where(loc == -1, 0, loc).astype(jnp.int32)
    new_kv_locs = jnp.arange(total_tokens, dtype=jnp.int32)
    slice_lens = jnp.where(loc == -1, 0, 1).astype(jnp.int32)
    num_slices = total_tokens

    # head_num, cache_len, new_kv_len, head_dim (fused), page_size
    num_slices_per_block = get_best_num_slices_per_block(
        fused_kv.shape[1],  # num_kv_heads
        kv_cache.shape[0],
        fused_kv.shape[0],
        fused_kv.shape[2],  # head_dim (after interleaving)
        page_size,
    )

    slot_mapping = get_slot_mapping(
        num_slices_per_block=num_slices_per_block,
        kv_cache_start_loc=kv_cache_locs,
        new_kv_start_loc=new_kv_locs,
        slice_lens=slice_lens,
    )

    num_kv_update_slices = jnp.array([num_slices], dtype=jnp.int32)

    kv_cache = kv_cache_update(
        new_kv=fused_kv,
        slices=slot_mapping,
        kv_cache=kv_cache,
        num_kv_update_slices=num_kv_update_slices,
        page_size=page_size,
        num_slices_per_block=num_slices_per_block,
        kv_partition_axis=kv_partition_axis,
    )

    return kv_cache


def get_best_num_slices_per_block(head_num, cache_len, new_kv_len, head_dim, page_size):
    # keep same to original implementation
    if page_size == 1:
        num_slices_per_block = 4
    else:
        num_slices_per_block = page_size

    return num_slices_per_block

    # note: the following logic will be supported in the future, the best num_slices_per_block is right tested well currently.
    # search domain, ensure list is sorted
    head_num_config = [8, 16, 32]
    max_cache_len_config = [80000, 160000, 320000, 640000, 1280000]
    new_kv_len_config = [1024, 2048, 4096, 9182, 16384]
    head_dim_config = [128]
    page_size_config = [64, 128, 256]

    def find_value(lst, target_num) -> int:
        left, right = 0, len(lst) - 1

        if not lst or target_num < lst[0] or target_num > lst[-1]:
            return -1

        while left <= right:
            mid = (left + right) // 2
            if lst[mid] == target_num:
                return lst[mid]
            elif lst[mid] < target_num:
                left = mid + 1
            else:
                right = mid - 1

        if left < len(lst):
            return lst[left]
        else:
            return -1

    hn_val = find_value(head_num_config, head_num)
    mcl_val = find_value(max_cache_len_config, cache_len)
    nkl_val = find_value(new_kv_len_config, new_kv_len)
    hd_val = find_value(head_dim_config, head_dim)
    ps_val = find_value(page_size_config, page_size)

    if (
        hn_val != -1
        and mcl_val != -1
        and nkl_val != -1
        and hd_val != -1
        and ps_val != -1
    ):
        return best_num_slices_per_block_config[
            f"hn_{hn_val}_mcl_{mcl_val}_nvl_{nkl_val}_hd_{hd_val}_ps_{ps_val}"
        ]


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

    def get_fused_kv_buffer(self, layer_id: int) -> jnp.ndarray:
        """Get fused buffer for MLA architecture.

        Note: MLA has different architecture than standard MHA,
        but we provide this interface for compatibility.
        """
        return self.kv_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get separate K and V buffers for native attention from MLA KV cache.

        Note: MLA architecture differs from standard MHA. For native attention compatibility,
        we split the combined kv_lora_rank + qk_rope_head_dim into separate K and V components.

        Returns:
            Tuple of (k_buffer, v_buffer) where:
            - k_buffer contains the kv_lora_rank portion
            - v_buffer contains the qk_rope_head_dim portion
        """
        layer_idx = layer_id - self.start_layer
        mla_kv = self.kv_buffer[
            layer_idx
        ]  # [cache_size, 1, kv_lora_rank + qk_rope_head_dim]

        # Split MLA KV buffer into K and V components for native attention
        k_buffer = mla_kv[:, :, : self.kv_lora_rank]  # [cache_size, 1, kv_lora_rank]
        v_buffer = mla_kv[
            :, :, self.kv_lora_rank :
        ]  # [cache_size, 1, qk_rope_head_dim]

        return k_buffer, v_buffer

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


best_num_slices_per_block_config = {
    "hn_8_mcl_80000_nvl_1024_hd_128_ps_64": 16,
    "hn_8_mcl_80000_nvl_2048_hd_128_ps_64": 32,
    "hn_8_mcl_80000_nvl_4096_hd_128_ps_64": 64,
    "hn_8_mcl_80000_nvl_9182_hd_128_ps_64": 32,
    "hn_8_mcl_80000_nvl_16384_hd_128_ps_64": 512,
    "hn_8_mcl_160000_nvl_1024_hd_128_ps_64": 16,
    "hn_8_mcl_160000_nvl_2048_hd_128_ps_64": 8,
    "hn_8_mcl_160000_nvl_4096_hd_128_ps_64": 64,
    "hn_8_mcl_160000_nvl_9182_hd_128_ps_64": 32,
    "hn_8_mcl_160000_nvl_16384_hd_128_ps_64": 128,
    "hn_8_mcl_320000_nvl_1024_hd_128_ps_64": 16,
    "hn_8_mcl_320000_nvl_2048_hd_128_ps_64": 16,
    "hn_8_mcl_320000_nvl_4096_hd_128_ps_64": 32,
    "hn_8_mcl_320000_nvl_9182_hd_128_ps_64": 16,
    "hn_8_mcl_320000_nvl_16384_hd_128_ps_64": 256,
    "hn_8_mcl_640000_nvl_1024_hd_128_ps_64": 4,
    "hn_8_mcl_640000_nvl_2048_hd_128_ps_64": 64,
    "hn_8_mcl_640000_nvl_4096_hd_128_ps_64": 64,
    "hn_8_mcl_640000_nvl_9182_hd_128_ps_64": 32,
    "hn_8_mcl_640000_nvl_16384_hd_128_ps_64": 64,
    "hn_8_mcl_1280000_nvl_1024_hd_128_ps_64": 1024,
    "hn_8_mcl_1280000_nvl_2048_hd_128_ps_64": 4,
    "hn_8_mcl_1280000_nvl_4096_hd_128_ps_64": 16,
    "hn_8_mcl_1280000_nvl_9182_hd_128_ps_64": 256,
    "hn_8_mcl_1280000_nvl_16384_hd_128_ps_64": 1024,
    "hn_16_mcl_80000_nvl_1024_hd_128_ps_64": 16,
    "hn_16_mcl_80000_nvl_2048_hd_128_ps_64": 32,
    "hn_16_mcl_80000_nvl_4096_hd_128_ps_64": 64,
    "hn_16_mcl_80000_nvl_9182_hd_128_ps_64": 32,
    "hn_16_mcl_80000_nvl_16384_hd_128_ps_64": 2048,
    "hn_16_mcl_160000_nvl_1024_hd_128_ps_64": 4,
    "hn_16_mcl_160000_nvl_2048_hd_128_ps_64": 16,
    "hn_16_mcl_160000_nvl_4096_hd_128_ps_64": 512,
    "hn_16_mcl_160000_nvl_9182_hd_128_ps_64": 128,
    "hn_16_mcl_160000_nvl_16384_hd_128_ps_64": 64,
    "hn_16_mcl_320000_nvl_1024_hd_128_ps_64": 8,
    "hn_16_mcl_320000_nvl_2048_hd_128_ps_64": 16,
    "hn_16_mcl_320000_nvl_4096_hd_128_ps_64": 16,
    "hn_16_mcl_320000_nvl_9182_hd_128_ps_64": 8,
    "hn_16_mcl_320000_nvl_16384_hd_128_ps_64": 256,
    "hn_16_mcl_640000_nvl_1024_hd_128_ps_64": 128,
    "hn_16_mcl_640000_nvl_2048_hd_128_ps_64": 4,
    "hn_16_mcl_640000_nvl_4096_hd_128_ps_64": 32,
    "hn_16_mcl_640000_nvl_9182_hd_128_ps_64": 64,
    "hn_16_mcl_640000_nvl_16384_hd_128_ps_64": 512,
    "hn_16_mcl_1280000_nvl_1024_hd_128_ps_64": 128,
    "hn_16_mcl_1280000_nvl_2048_hd_128_ps_64": 2,
    "hn_16_mcl_1280000_nvl_4096_hd_128_ps_64": 2048,
    "hn_16_mcl_1280000_nvl_9182_hd_128_ps_64": 8,
    "hn_16_mcl_1280000_nvl_16384_hd_128_ps_64": 8,
    "hn_32_mcl_80000_nvl_1024_hd_128_ps_64": 256,
    "hn_32_mcl_80000_nvl_2048_hd_128_ps_64": 32,
    "hn_32_mcl_80000_nvl_4096_hd_128_ps_64": 16,
    "hn_32_mcl_80000_nvl_9182_hd_128_ps_64": 512,
    "hn_32_mcl_80000_nvl_16384_hd_128_ps_64": 4096,
    "hn_32_mcl_160000_nvl_1024_hd_128_ps_64": 8,
    "hn_32_mcl_160000_nvl_2048_hd_128_ps_64": 8,
    "hn_32_mcl_160000_nvl_4096_hd_128_ps_64": 512,
    "hn_32_mcl_160000_nvl_9182_hd_128_ps_64": 64,
    "hn_32_mcl_160000_nvl_16384_hd_128_ps_64": 1024,
    "hn_32_mcl_320000_nvl_1024_hd_128_ps_64": 4,
    "hn_32_mcl_320000_nvl_2048_hd_128_ps_64": 512,
    "hn_32_mcl_320000_nvl_4096_hd_128_ps_64": 256,
    "hn_32_mcl_320000_nvl_9182_hd_128_ps_64": 8,
    "hn_32_mcl_320000_nvl_16384_hd_128_ps_64": 64,
    "hn_32_mcl_640000_nvl_1024_hd_128_ps_64": 2048,
    "hn_32_mcl_640000_nvl_2048_hd_128_ps_64": 8,
    "hn_32_mcl_640000_nvl_4096_hd_128_ps_64": 256,
    "hn_32_mcl_640000_nvl_9182_hd_128_ps_64": 256,
    "hn_32_mcl_640000_nvl_16384_hd_128_ps_64": 16,
    "hn_32_mcl_1280000_nvl_1024_hd_128_ps_64": 256,
    "hn_32_mcl_1280000_nvl_2048_hd_128_ps_64": 512,
    "hn_32_mcl_1280000_nvl_4096_hd_128_ps_64": 4096,
    "hn_32_mcl_1280000_nvl_9182_hd_128_ps_64": 128,
    "hn_32_mcl_1280000_nvl_16384_hd_128_ps_64": 1024,
    "hn_8_mcl_80000_nvl_1024_hd_128_ps_128": 8,
    "hn_8_mcl_80000_nvl_2048_hd_128_ps_128": 32,
    "hn_8_mcl_80000_nvl_4096_hd_128_ps_128": 64,
    "hn_8_mcl_80000_nvl_9182_hd_128_ps_128": 1024,
    "hn_8_mcl_80000_nvl_16384_hd_128_ps_128": 16,
    "hn_8_mcl_160000_nvl_1024_hd_128_ps_128": 16,
    "hn_8_mcl_160000_nvl_2048_hd_128_ps_128": 4,
    "hn_8_mcl_160000_nvl_4096_hd_128_ps_128": 64,
    "hn_8_mcl_160000_nvl_9182_hd_128_ps_128": 32,
    "hn_8_mcl_160000_nvl_16384_hd_128_ps_128": 4096,
    "hn_8_mcl_320000_nvl_1024_hd_128_ps_128": 32,
    "hn_8_mcl_320000_nvl_2048_hd_128_ps_128": 8,
    "hn_8_mcl_320000_nvl_4096_hd_128_ps_128": 16,
    "hn_8_mcl_320000_nvl_9182_hd_128_ps_128": 32,
    "hn_8_mcl_320000_nvl_16384_hd_128_ps_128": 16,
    "hn_8_mcl_640000_nvl_1024_hd_128_ps_128": 8,
    "hn_8_mcl_640000_nvl_2048_hd_128_ps_128": 16,
    "hn_8_mcl_640000_nvl_4096_hd_128_ps_128": 16,
    "hn_8_mcl_640000_nvl_9182_hd_128_ps_128": 16,
    "hn_8_mcl_640000_nvl_16384_hd_128_ps_128": 4096,
    "hn_8_mcl_1280000_nvl_1024_hd_128_ps_128": 128,
    "hn_8_mcl_1280000_nvl_2048_hd_128_ps_128": 32,
    "hn_8_mcl_1280000_nvl_4096_hd_128_ps_128": 256,
    "hn_8_mcl_1280000_nvl_9182_hd_128_ps_128": 1024,
    "hn_8_mcl_1280000_nvl_16384_hd_128_ps_128": 32,
    "hn_16_mcl_80000_nvl_1024_hd_128_ps_128": 128,
    "hn_16_mcl_80000_nvl_2048_hd_128_ps_128": 8,
    "hn_16_mcl_80000_nvl_4096_hd_128_ps_128": 2048,
    "hn_16_mcl_80000_nvl_9182_hd_128_ps_128": 1024,
    "hn_16_mcl_80000_nvl_16384_hd_128_ps_128": 256,
    "hn_16_mcl_160000_nvl_1024_hd_128_ps_128": 8,
    "hn_16_mcl_160000_nvl_2048_hd_128_ps_128": 1024,
    "hn_16_mcl_160000_nvl_4096_hd_128_ps_128": 2048,
    "hn_16_mcl_160000_nvl_9182_hd_128_ps_128": 4096,
    "hn_16_mcl_160000_nvl_16384_hd_128_ps_128": 16,
    "hn_16_mcl_320000_nvl_1024_hd_128_ps_128": 8,
    "hn_16_mcl_320000_nvl_2048_hd_128_ps_128": 128,
    "hn_16_mcl_320000_nvl_4096_hd_128_ps_128": 256,
    "hn_16_mcl_320000_nvl_9182_hd_128_ps_128": 1024,
    "hn_16_mcl_320000_nvl_16384_hd_128_ps_128": 32,
    "hn_16_mcl_640000_nvl_1024_hd_128_ps_128": 2,
    "hn_16_mcl_640000_nvl_2048_hd_128_ps_128": 8,
    "hn_16_mcl_640000_nvl_4096_hd_128_ps_128": 8,
    "hn_16_mcl_640000_nvl_9182_hd_128_ps_128": 32,
    "hn_16_mcl_640000_nvl_16384_hd_128_ps_128": 4,
    "hn_16_mcl_1280000_nvl_1024_hd_128_ps_128": 32,
    "hn_16_mcl_1280000_nvl_2048_hd_128_ps_128": 16,
    "hn_16_mcl_1280000_nvl_4096_hd_128_ps_128": 128,
    "hn_16_mcl_1280000_nvl_9182_hd_128_ps_128": 64,
    "hn_16_mcl_1280000_nvl_16384_hd_128_ps_128": 2048,
    "hn_32_mcl_80000_nvl_1024_hd_128_ps_128": 2,
    "hn_32_mcl_80000_nvl_2048_hd_128_ps_128": 4,
    "hn_32_mcl_80000_nvl_4096_hd_128_ps_128": 256,
    "hn_32_mcl_80000_nvl_9182_hd_128_ps_128": 8,
    "hn_32_mcl_80000_nvl_16384_hd_128_ps_128": 512,
    "hn_32_mcl_160000_nvl_1024_hd_128_ps_128": 16,
    "hn_32_mcl_160000_nvl_2048_hd_128_ps_128": 1024,
    "hn_32_mcl_160000_nvl_4096_hd_128_ps_128": 512,
    "hn_32_mcl_160000_nvl_9182_hd_128_ps_128": 8,
    "hn_32_mcl_160000_nvl_16384_hd_128_ps_128": 4096,
    "hn_32_mcl_320000_nvl_1024_hd_128_ps_128": 256,
    "hn_32_mcl_320000_nvl_2048_hd_128_ps_128": 256,
    "hn_32_mcl_320000_nvl_4096_hd_128_ps_128": 256,
    "hn_32_mcl_320000_nvl_9182_hd_128_ps_128": 1024,
    "hn_32_mcl_320000_nvl_16384_hd_128_ps_128": 4,
    "hn_32_mcl_640000_nvl_1024_hd_128_ps_128": 512,
    "hn_32_mcl_640000_nvl_2048_hd_128_ps_128": 4096,
    "hn_32_mcl_640000_nvl_4096_hd_128_ps_128": 4,
    "hn_32_mcl_640000_nvl_9182_hd_128_ps_128": 2,
    "hn_32_mcl_640000_nvl_16384_hd_128_ps_128": 1024,
    "hn_32_mcl_1280000_nvl_1024_hd_128_ps_128": 32,
    "hn_32_mcl_1280000_nvl_2048_hd_128_ps_128": 2048,
    "hn_32_mcl_1280000_nvl_4096_hd_128_ps_128": 128,
    "hn_32_mcl_1280000_nvl_9182_hd_128_ps_128": 1024,
    "hn_32_mcl_1280000_nvl_16384_hd_128_ps_128": 1024,
    "hn_8_mcl_80000_nvl_1024_hd_128_ps_256": 2,
    "hn_8_mcl_80000_nvl_2048_hd_128_ps_256": 4,
    "hn_8_mcl_80000_nvl_4096_hd_128_ps_256": 2,
    "hn_8_mcl_80000_nvl_9182_hd_128_ps_256": 512,
    "hn_8_mcl_80000_nvl_16384_hd_128_ps_256": 2048,
    "hn_8_mcl_160000_nvl_1024_hd_128_ps_256": 4,
    "hn_8_mcl_160000_nvl_2048_hd_128_ps_256": 256,
    "hn_8_mcl_160000_nvl_4096_hd_128_ps_256": 128,
    "hn_8_mcl_160000_nvl_9182_hd_128_ps_256": 2048,
    "hn_8_mcl_160000_nvl_16384_hd_128_ps_256": 2048,
    "hn_8_mcl_320000_nvl_1024_hd_128_ps_256": 4,
    "hn_8_mcl_320000_nvl_2048_hd_128_ps_256": 1024,
    "hn_8_mcl_320000_nvl_4096_hd_128_ps_256": 64,
    "hn_8_mcl_320000_nvl_9182_hd_128_ps_256": 16,
    "hn_8_mcl_320000_nvl_16384_hd_128_ps_256": 512,
    "hn_8_mcl_640000_nvl_1024_hd_128_ps_256": 1024,
    "hn_8_mcl_640000_nvl_2048_hd_128_ps_256": 8,
    "hn_8_mcl_640000_nvl_4096_hd_128_ps_256": 16,
    "hn_8_mcl_640000_nvl_9182_hd_128_ps_256": 16,
    "hn_8_mcl_640000_nvl_16384_hd_128_ps_256": 4096,
    "hn_8_mcl_1280000_nvl_1024_hd_128_ps_256": 64,
    "hn_8_mcl_1280000_nvl_2048_hd_128_ps_256": 2,
    "hn_8_mcl_1280000_nvl_4096_hd_128_ps_256": 2048,
    "hn_8_mcl_1280000_nvl_9182_hd_128_ps_256": 1024,
    "hn_8_mcl_1280000_nvl_16384_hd_128_ps_256": 128,
    "hn_16_mcl_80000_nvl_1024_hd_128_ps_256": 2,
    "hn_16_mcl_80000_nvl_2048_hd_128_ps_256": 16,
    "hn_16_mcl_80000_nvl_4096_hd_128_ps_256": 64,
    "hn_16_mcl_80000_nvl_9182_hd_128_ps_256": 256,
    "hn_16_mcl_80000_nvl_16384_hd_128_ps_256": 16,
    "hn_16_mcl_160000_nvl_1024_hd_128_ps_256": 4,
    "hn_16_mcl_160000_nvl_2048_hd_128_ps_256": 2,
    "hn_16_mcl_160000_nvl_4096_hd_128_ps_256": 128,
    "hn_16_mcl_160000_nvl_9182_hd_128_ps_256": 16,
    "hn_16_mcl_160000_nvl_16384_hd_128_ps_256": 8,
    "hn_16_mcl_320000_nvl_1024_hd_128_ps_256": 16,
    "hn_16_mcl_320000_nvl_2048_hd_128_ps_256": 8,
    "hn_16_mcl_320000_nvl_4096_hd_128_ps_256": 4,
    "hn_16_mcl_320000_nvl_9182_hd_128_ps_256": 8,
    "hn_16_mcl_320000_nvl_16384_hd_128_ps_256": 8,
    "hn_16_mcl_640000_nvl_1024_hd_128_ps_256": 512,
    "hn_16_mcl_640000_nvl_2048_hd_128_ps_256": 1024,
    "hn_16_mcl_640000_nvl_4096_hd_128_ps_256": 2048,
    "hn_16_mcl_640000_nvl_9182_hd_128_ps_256": 4096,
    "hn_16_mcl_640000_nvl_16384_hd_128_ps_256": 32,
    "hn_16_mcl_1280000_nvl_1024_hd_128_ps_256": 4,
    "hn_16_mcl_1280000_nvl_2048_hd_128_ps_256": 2,
    "hn_16_mcl_1280000_nvl_4096_hd_128_ps_256": 1024,
    "hn_16_mcl_1280000_nvl_9182_hd_128_ps_256": 2048,
    "hn_16_mcl_1280000_nvl_16384_hd_128_ps_256": 16,
    "hn_32_mcl_80000_nvl_1024_hd_128_ps_256": 4,
    "hn_32_mcl_80000_nvl_2048_hd_128_ps_256": 256,
    "hn_32_mcl_80000_nvl_4096_hd_128_ps_256": 4096,
    "hn_32_mcl_80000_nvl_9182_hd_128_ps_256": 128,
    "hn_32_mcl_80000_nvl_16384_hd_128_ps_256": 512,
    "hn_32_mcl_160000_nvl_1024_hd_128_ps_256": 64,
    "hn_32_mcl_160000_nvl_2048_hd_128_ps_256": 4096,
    "hn_32_mcl_160000_nvl_4096_hd_128_ps_256": 4096,
    "hn_32_mcl_160000_nvl_9182_hd_128_ps_256": 256,
    "hn_32_mcl_160000_nvl_16384_hd_128_ps_256": 128,
    "hn_32_mcl_320000_nvl_1024_hd_128_ps_256": 4,
    "hn_32_mcl_320000_nvl_2048_hd_128_ps_256": 64,
    "hn_32_mcl_320000_nvl_4096_hd_128_ps_256": 1024,
    "hn_32_mcl_320000_nvl_9182_hd_128_ps_256": 256,
    "hn_32_mcl_320000_nvl_16384_hd_128_ps_256": 32,
    "hn_32_mcl_640000_nvl_1024_hd_128_ps_256": 256,
    "hn_32_mcl_640000_nvl_2048_hd_128_ps_256": 8,
    "hn_32_mcl_640000_nvl_4096_hd_128_ps_256": 64,
    "hn_32_mcl_640000_nvl_9182_hd_128_ps_256": 32,
    "hn_32_mcl_640000_nvl_16384_hd_128_ps_256": 32,
    "hn_32_mcl_1280000_nvl_1024_hd_128_ps_256": 256,
    "hn_32_mcl_1280000_nvl_2048_hd_128_ps_256": 2048,
    "hn_32_mcl_1280000_nvl_4096_hd_128_ps_256": 8,
    "hn_32_mcl_1280000_nvl_9182_hd_128_ps_256": 4,
    "hn_32_mcl_1280000_nvl_16384_hd_128_ps_256": 2,
}
