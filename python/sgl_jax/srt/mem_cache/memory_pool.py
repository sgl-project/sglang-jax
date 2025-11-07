import abc
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    get_num_slices_per_block,
    get_slot_mapping,
    kv_cache_update,
)


def merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
    assert k.shape == v.shape, f"k and v must have same shape, got {k.shape} vs {v.shape}"

    num_tokens, num_kv_heads, head_dim = k.shape

    kv_stacked = jnp.stack([k, v], axis=2)  # [tokens, heads, 2, head_dim]
    kv_fused = kv_stacked.reshape(num_tokens, num_kv_heads * 2, head_dim)

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

    def alloc(self, need_size: int = 1) -> list[int]:
        """Allocate request slots"""
        if need_size > len(self.free_slots):
            return None

        select_indices = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select_indices

    def free(self, free_index: int | list[int]):
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
        start_layer: int | None = None,
        end_layer: int | None = None,
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
    def get_kv_buffer(self, layer_id: int) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    @abc.abstractmethod
    def replace_kv_buffer(self, kv_buffer: list[jnp.ndarray]) -> None:
        """Replace the internal KV buffer with a new one.

        This method is essential for JAX jit compatibility since JAX functions
        are pure and cannot perform in-place mutations. After running forward
        passes that update KV cache through functional operations (like .at[].set()),
        we need to replace the original buffer references with the updated ones
        returned from the jitted computation.

        Args:
            kv_buffer: The updated KV buffer returned from jitted forward pass

        Note:
            This enables the functional programming paradigm required by JAX
            while maintaining the illusion of in-place updates for the user API.
        """
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
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(size, page_size, dtype, layer_num, mesh, start_layer, end_layer)
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
        self.kv_sharding = NamedSharding(self.mesh, P(None, self.kv_partition_axis, None))

        logger.info("Creating fused KV buffers for %s layers", self.layer_num)
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
            "Total fused KV cache memory per layer: %.2f GB, dtype: %s",
            total_memory_per_layer / 1024**3,
            self.dtype,
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
            "Total time to create %s buffers: %.2f seconds",
            self.layer_num,
            end_time - start_time,
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
            "JAX Fused KV Cache allocated. #tokens: %s, Fused KV size: %.2f GB",
            self.size,
            fused_kv_size / GB,
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

    def get_kv_buffer(self, layer_id: int) -> tuple[jnp.ndarray, jnp.ndarray]:
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

    def replace_kv_buffer(self, fused_kv_buffer: list[jnp.ndarray]) -> None:
        self.kv_buffer[self.start_layer : self.start_layer + len(fused_kv_buffer)] = fused_kv_buffer

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
            self.kv_buffer[layer_id] = self.kv_buffer[layer_id].at[indices].set(fused_kv_device)

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
    ) -> jax.Array:
        """
        Legacy interface for backward compatibility.
        This assumes contiguous cache locations and uses simple JAX operations.
        """
        layer_idx = layer_id - self.start_layer
        # Merge k and v into fused format
        fused_kv = merge_kv(cache_k, cache_v)
        N = self.kv_buffer[layer_idx].shape[0]
        safe_loc = jnp.where(loc >= 0, loc, jnp.int32(N))
        # for jax function
        updated_layer = self.kv_buffer[layer_idx].at[safe_loc].set(fused_kv, mode="drop")
        return updated_layer


@register_pytree_node_class
class SWAKVPool(KVCache):
    """KV cache with separate pools for full and SWA attention layers."""

    def __init__(
        self,
        size: int,
        size_swa: int,
        swa_attention_layer_ids: list[int],
        full_attention_layer_ids: list[int],
        token_to_kv_pool_class: KVCache = MHATokenToKVPool,
        **kwargs,
    ):
        self.size = size
        self.size_swa = size_swa
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.mesh = kwargs["mesh"]
        self.kv_partition_axis = "tensor"
        kwargs["page_size"] = 1

        self.swa_kv_pool = token_to_kv_pool_class(
            size=size_swa,
            layer_num=self.swa_layer_nums,
            **kwargs,
        )
        self.full_kv_pool = token_to_kv_pool_class(
            size=size,
            layer_num=self.full_layer_nums,
            **kwargs,
        )

        self.layers_mapping: dict[int, tuple[int, bool]] = {}
        for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
        for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_layer_id] = (swa_layer_id, True)

        # A host-side mapping array that maps indices from the full-attention
        # token space to the SWA token space. This is owned and updated by
        # SWATokenToKVPoolAllocator, and injected here via reference.
        # Shape: [size_full + size_swa + 1], dtype=int64/int32 on host.
        self.full_to_swa_index_mapping: np.array | None = None

        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def tree_flatten(self):
        children = (
            self.swa_kv_pool,
            self.full_kv_pool,
            self.full_to_swa_index_mapping,
        )
        aux_data = {
            "size": self.size,
            "size_swa": self.size_swa,
            "swa_layer_nums": self.swa_layer_nums,
            "full_layer_nums": self.full_layer_nums,
            "layers_mapping": self.layers_mapping,
            "mem_usage": self.mem_usage,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)

        obj.size = aux_data["size"]
        obj.size_swa = aux_data["size_swa"]
        obj.swa_layer_nums = aux_data["swa_layer_nums"]
        obj.full_layer_nums = aux_data["full_layer_nums"]
        obj.layers_mapping = aux_data["layers_mapping"]
        obj.mem_usage = aux_data["mem_usage"]

        obj.swa_kv_pool = children[0]
        obj.full_kv_pool = children[1]
        obj.full_to_swa_index_mapping = children[2]

        return obj

    def get_kv_size_bytes(self):
        k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
        k_size_swa, v_size_swa = self.swa_kv_pool.get_kv_size_bytes()
        return k_size + k_size_swa, v_size + v_size_swa

    def get_kv_buffer(self, layer_id: int):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_kv_buffer(layer_id_pool)
        return self.full_kv_pool.get_kv_buffer(layer_id_pool)

    def get_fused_kv_buffer(self, layer_id):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            return self.swa_kv_pool.get_fused_kv_buffer(layer_id_pool)
        return self.full_kv_pool.get_fused_kv_buffer(layer_id_pool)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jnp.ndarray,
        cache_k: jnp.ndarray,
        cache_v: jnp.ndarray,
        is_decode: bool = False,
    ):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            if self.full_to_swa_index_mapping is not None:
                loc = self.full_to_swa_index_mapping[loc].to(np.int32)
            self.swa_kv_pool.set_kv_buffer(layer_id_pool, loc, cache_k, cache_v, is_decode)
        else:
            self.full_kv_pool.set_kv_buffer(layer_id_pool, loc, cache_k, cache_v, is_decode)

    def replace_kv_buffer(self, kv_buffer: list[jnp.ndarray]):
        assert len(kv_buffer) == len(self.layers_mapping)

        full_kv_buffer = []
        swa_kv_buffer = []
        for layer_id, layer_kv_buffer in enumerate(kv_buffer):
            _, is_swa = self.layers_mapping[layer_id]
            if is_swa:
                swa_kv_buffer.append(layer_kv_buffer)
            else:
                full_kv_buffer.append(layer_kv_buffer)

        self.swa_kv_pool.replace_kv_buffer(swa_kv_buffer)
        self.full_kv_pool.replace_kv_buffer(full_kv_buffer)

    def remap_cache_loc(self, loc: jax.Array, layer_id: int) -> jax.Array:
        """
        Remap cache locations from the full-attention token space to the SWA
        token space if the given layer is an SWA layer.

        Args:
            loc: jax.Array of int indices (token-level when page_size==1)
            layer_id: global layer id

        Returns:
            jax.Array indices valid for the underlying pool of the given layer
        """
        _, is_swa = self.layers_mapping[layer_id]
        if not is_swa:
            return loc
        if self.full_to_swa_index_mapping is None:
            # No mapping available yet; return as-is to avoid crash. Caller may handle.
            return loc
        # Convert host mapping to jax array and gather
        mapping_jax = jnp.asarray(self.full_to_swa_index_mapping, dtype=jnp.int32)
        return mapping_jax[loc]


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
    num_slices_per_block = get_num_slices_per_block(
        k,
        k_cache,
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
    num_slices_per_block = get_num_slices_per_block(
        fused_kv,  # num_kv_heads
        kv_cache,
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


# @partial(jax.jit, static_argnames=["layer_id"])
def _get_kv_buffer(
    layer_id: int, k_cache: jax.Array, v_cache: jax.Array
) -> tuple[jax.Array, jax.Array]:
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
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(size, page_size, dtype, layer_num, mesh, start_layer, end_layer)
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
            "JAX MLA KV Cache allocated. #tokens: %s, KV size: %.2f GB",
            self.size,
            kv_size / GB,
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

    def get_kv_buffer(self, layer_id: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get separate K and V buffers for native attention from MLA KV cache.

        Note: MLA architecture differs from standard MHA. For native attention compatibility,
        we split the combined kv_lora_rank + qk_rope_head_dim into separate K and V components.

        Returns:
            Tuple of (k_buffer, v_buffer) where:
            - k_buffer contains the kv_lora_rank portion
            - v_buffer contains the qk_rope_head_dim portion
        """
        layer_idx = layer_id - self.start_layer
        mla_kv = self.kv_buffer[layer_idx]  # [cache_size, 1, kv_lora_rank + qk_rope_head_dim]

        # Split MLA KV buffer into K and V components for native attention
        k_buffer = mla_kv[:, :, : self.kv_lora_rank]  # [cache_size, 1, kv_lora_rank]
        v_buffer = mla_kv[:, :, self.kv_lora_rank :]  # [cache_size, 1, qk_rope_head_dim]

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
