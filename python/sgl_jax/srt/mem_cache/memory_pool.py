import abc
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing
from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    get_num_slices_per_block,
    get_slot_mapping,
    kv_cache_update,
    kv_cache_update_impl,
)


def merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
    """Merge 3D k/v into 5D fused format matching KV cache shape.

    Input:  k, v: [num_tokens, num_kv_heads, head_dim]
    Output: [num_tokens, 1, num_kv_heads * 2 // packing, packing, head_dim_aligned]
    """
    assert k.shape == v.shape, f"k and v must have same shape, got {k.shape} vs {v.shape}"

    num_tokens, num_kv_heads, head_dim = k.shape

    from sgl_jax.srt.kernels.ragged_paged_attention.util import align_to

    packing = get_dtype_packing(k.dtype)
    num_kv_heads_x2 = num_kv_heads * 2
    head_dim_aligned = align_to(head_dim, 128)

    # Interleave k and v: [tokens, heads, 2, head_dim] -> [tokens, heads*2, head_dim]
    kv_stacked = jnp.stack([k, v], axis=2)
    kv_fused = kv_stacked.reshape(num_tokens, num_kv_heads_x2, head_dim)

    # Pad head_dim to aligned size, then reshape to 5D step by step
    # (each step only splits one axis to avoid JAX ShardingTypeError)
    kv_fused = jnp.pad(
        kv_fused,
        (
            (0, 0),
            (0, 0),
            (0, head_dim_aligned - head_dim),
        ),
        constant_values=0,
    )
    # Step 1: [tokens, heads*2, hdim] -> [tokens, heads*2//packing, packing, hdim]
    kv_fused = kv_fused.reshape(num_tokens, num_kv_heads_x2 // packing, packing, head_dim_aligned)
    # Step 2: [tokens, h, packing, hdim] -> [tokens, 1, h, packing, hdim]
    kv_fused = jnp.expand_dims(kv_fused, axis=1)
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

    def alloc(self, reqs) -> list[int] | None:
        """Allocate request slots for a list of Req-like objects.

        Chunked-prefill semantics:
        - Reqs already holding a req_pool_idx are skipped (must satisfy
          is_chunked > 0 -- safety assert).
        - Reqs with no req_pool_idx are allocated from free_slots and have
          req.req_pool_idx written back.
        - If capacity is insufficient, return None and leave every req field
          untouched (atomic semantics).

        Returns: list[int] (the slot per req), or None if capacity is insufficient.
        """
        # Pass 1: count slots that need fresh allocation; assert safety on reuses.
        new_count = 0
        for req in reqs:
            if req.req_pool_idx is None:
                new_count += 1
            else:
                assert req.is_chunked > 0, (
                    "ReqToTokenPool.alloc: req with existing req_pool_idx must have "
                    f"is_chunked > 0 (chunked prefill); got is_chunked={req.is_chunked!r}"
                )

        if new_count > len(self.free_slots):
            return None

        # Pass 2: atomic allocation.
        new_slots = self.free_slots[:new_count]
        self.free_slots = self.free_slots[new_count:]

        result = []
        cursor = 0
        for req in reqs:
            if req.req_pool_idx is None:
                req.req_pool_idx = new_slots[cursor]
                cursor += 1
            result.append(req.req_pool_idx)
        return result

    def free(self, req):
        """Free request slot held by ``req``.

        Accepts a Req-like object with a ``req_pool_idx`` attribute (matches
        ``alloc(reqs)`` signature symmetry). Subclasses (HybridReqToTokenPool)
        override to release additional state slots (recurrent state) before
        delegating here.
        """
        # Defensive type check: legacy callers passing bare int / list[int]
        # hit an immediate, clear error instead of silent AttributeError
        # downstream when self.free_slots.append(req.req_pool_idx) accesses
        # a missing attribute.
        if not hasattr(req, "req_pool_idx"):
            raise TypeError(
                f"ReqToTokenPool.free expects a Req object with req_pool_idx attr; "
                f"got {type(req).__name__}. Legacy int/list[int] callers must migrate."
            )
        self.free_slots.append(req.req_pool_idx)
        # Clear the slot ref on the req so a stale value can't leak into a
        # later caller (the slot may already be re-handed to another req by
        # then). Keeps free / free_chunked / free_recurrent_cache symmetric.
        req.req_pool_idx = None

    def free_chunked(self, chunked_req) -> None:
        """Release for chunked-prefill. Default: identical to free.
        HybridReqToTokenPool overrides to preserve slots across chunks.
        """
        self.free(chunked_req)

    def clear(self):
        """Clear all allocation states"""
        self.free_slots = list(range(self.size))


class HybridReqToTokenPool(ReqToTokenPool):
    """Coordinates KV slot + recurrent state slot allocation for hybrid models.

    - Owns `recurrent_free_slots` (slot 0 dummy, valid 1..max). Allocator
      state lives here, not on RecurrentStatePool — so the buffer pool stays
      a pytree leaf safe for JIT donate.
    - Maintains `req_index_to_recurrent_index_mapping` bridging
      req_pool_idx -> recurrent_pool_idx (defaults to 0 = dummy).
    - NOT registered as a pytree node: the host-side allocator state
      must not enter JIT donate (would otherwise reset on tree_unflatten).
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        dtype: np.dtype,
        recurrent_state_pool,
    ):
        super().__init__(size=size, max_context_len=max_context_len, dtype=dtype)
        self.recurrent_state_pool = recurrent_state_pool
        # Recurrent slot allocator state. 1-indexed because slot 0 is reserved as
        # the dummy slot in the underlying RecurrentStatePool buffers; mapping
        # entries default to 0 so an unallocated req lands on the dummy slot.
        # Mirrors sglang PyTorch MambaPool free_slots semantics.
        self.recurrent_free_slots: list[int] = list(range(1, recurrent_state_pool.max_num_reqs + 1))
        # mapping[req_pool_idx] = recurrent_pool_idx; initialized to 0 (dummy slot).
        self.req_index_to_recurrent_index_mapping = np.zeros(size, dtype=np.int32)

    def alloc(self, reqs):
        """Coordinate allocation: KV slot first, then recurrent slot; atomic semantics.

        Mirrors the "all-or-nothing" coordinated allocation from sglang PyTorch
        HybridReqToTokenPool.alloc and the parent ReqToTokenPool.alloc Pass-2
        batch-slice pattern.

        Strategy:
        1. Pre-check the local recurrent_free_slots capacity for fresh allocs.
        2. super().alloc(reqs) -- KV slot path is already atomic.
        3. Batch-slice the needed recurrent slots, clear-on-alloc all of them in
           one recurrent_state_pool.clear_slot call (avoids per-slot JIT scatter
           overhead), then assign + write mapping.

        On any resource shortage we return None; req fields, recurrent_free_slots
        and mapping all stay completely untouched before the KV alloc succeeds
        (recurrent writes only happen after KV passes).
        """
        # 1. Pre-check recurrent capacity (only fresh reqs consume a slot).
        needed_recurrent = sum(1 for req in reqs if req.recurrent_pool_idx is None)
        if needed_recurrent > len(self.recurrent_free_slots):
            return None

        # 2. KV slot (parent is atomic).
        kv_indices = super().alloc(reqs)
        if kv_indices is None:
            return None

        # 3. Atomic batch slice (mirrors parent's Pass-2 pattern), then a single
        #    clear-on-alloc call covering every newly-handed-out slot.
        new_recurrent_slots = self.recurrent_free_slots[:needed_recurrent]
        self.recurrent_free_slots = self.recurrent_free_slots[needed_recurrent:]

        if new_recurrent_slots:
            self.recurrent_state_pool.clear_slot(new_recurrent_slots)

        cursor = 0
        for req in reqs:
            if req.recurrent_pool_idx is None:
                req.recurrent_pool_idx = new_recurrent_slots[cursor]
                cursor += 1
            self.req_index_to_recurrent_index_mapping[req.req_pool_idx] = req.recurrent_pool_idx

        return kv_indices

    def free(self, req) -> None:
        """Coordinate release: recurrent slot first, then KV slot.

        Idempotent when ``req.recurrent_pool_idx`` is None (free_recurrent_cache
        no-op). Mirrors the alloc(reqs) coordination pattern (KV first then
        recurrent on alloc; recurrent first then KV on free, so a partially
        constructed req can never end up with a freed KV slot but a still-held
        recurrent slot).
        """
        # Defensive type check at the override entry too: free_recurrent_cache
        # touches req.recurrent_pool_idx before super().free reaches its own
        # type check, so without this guard a legacy int caller would fail
        # with AttributeError instead of the explicit TypeError contract.
        if not hasattr(req, "req_pool_idx"):
            raise TypeError(
                f"HybridReqToTokenPool.free expects a Req object with req_pool_idx attr; "
                f"got {type(req).__name__}. Legacy int/list[int] callers must migrate."
            )
        self.free_recurrent_cache(req)
        super().free(req)

    def free_chunked(self, chunked_req) -> None:
        """Hybrid override: keep both KV slot AND recurrent slot so the next
        alloc(reqs) reuses them -- preserves the recurrent state across
        chunks without an alloc/free roundtrip.

        No-op by design: req keeps both req_pool_idx and recurrent_pool_idx;
        alloc(reqs) sees them set and skips re-allocation, returning the same
        indices.
        """
        # Intentional no-op (see docstring).

    def free_recurrent_cache(self, req) -> None:
        """Release the recurrent state slot held by `req`. Idempotent.

        Returns the slot to recurrent_free_slots, sets req.recurrent_pool_idx
        back to None and clears the corresponding mapping entry to the dummy
        slot (0). Safe to call when the req has no recurrent slot allocated.
        """
        if req.recurrent_pool_idx is None:
            return
        self.recurrent_free_slots.append(req.recurrent_pool_idx)
        self.req_index_to_recurrent_index_mapping[req.req_pool_idx] = 0
        req.recurrent_pool_idx = None

    def get_linear_recurrent_indices(self, req_pool_indices):
        """Look up linear recurrent slot indices via the CPU-side mapping table.

        Accepts an np.ndarray (or list[int]) of req_pool_idx and returns
        the same-shape np.ndarray of recurrent_pool_idx values.
        """
        return self.req_index_to_recurrent_index_mapping[req_pool_indices]

    def clear(self) -> None:
        """Full reset: parent clear + recurrent buffer clear + recurrent allocator
        reset + mapping zeroed.

        Called from flush_cache; resets KV slot allocator, recurrent slot
        allocator, recurrent + conv buffers, and the req_pool_idx ->
        recurrent_pool_idx mapping.
        """
        super().clear()
        self.recurrent_state_pool.clear()
        self.recurrent_free_slots = list(range(1, self.recurrent_state_pool.max_num_reqs + 1))
        self.req_index_to_recurrent_index_mapping[:] = 0


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
    def get_fused_kv_buffer(self, layer_id: int) -> jax.Array:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> tuple[jax.Array, jax.Array]:
        """Get separate K and V buffers for native attention.

        Returns:
            Tuple of (k_buffer, v_buffer)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jax.Array,
        cache_k: jax.Array,
        cache_v: jax.Array,
        is_decode: bool,
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def replace_buffer(self, kv_buffer: list[jax.Array]) -> None:
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
        self.kv_sharding = NamedSharding(
            self.mesh,
            P(None, None, self.kv_partition_axis, None, None),
        )

        logger.info("Creating fused KV buffers for %s layers", self.layer_num)
        start_time = time.time()

        assert (
            self.size % self.page_size == 0
        ), "Cache size must be divisible by dp_size and size must be divisible by page size"

        # Hack: this shape is more friendly to rpav3
        packing = get_dtype_packing(self.dtype)
        fused_buffer_shape = (
            (self.size + self.page_size) // self.page_size,
            self.page_size,
            self.head_num * 2 // packing,  # [K0,V0,K1,V1,...]
            packing,
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
            total_memory_per_layer / GB,
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

    def get_fused_kv_buffer(self, layer_id: int) -> jax.Array:
        return self.kv_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int) -> jax.Array:
        return self.kv_buffer[layer_id - self.start_layer]

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
            attention_data_partition_axis=None,
            mesh=self.mesh,
        )

    def replace_buffer(self, fused_kv_buffer: list[jax.Array]) -> None:
        # tp_size==1 sharding fix (issue #233): when the tp axis is degenerate
        # (size 1), JAX optimizes away the P(None, "tensor") constraint on
        # JIT output; explicit device_put with self.kv_sharding restores it
        # before the slice assign so the next JIT trace sees a stable shape.
        # Trigger condition is `mesh.shape["tensor"] == 1` (== ServerArgs
        # tp_size == 1), which is what aolemila's original fix in
        # _set_kv_cache_after_forward used. NOT `len(device_set) == 1` —
        # that's mesh-total-devices, which differs in multi-device tp_size=1
        # deployments (e.g. 8 devices with tp=1 → device_set=8 but tensor
        # axis still degenerate, fix still required).
        if self.mesh.shape.get("tensor", 1) == 1 and hasattr(self, "kv_sharding"):
            fused_kv_buffer = [jax.device_put(buf, self.kv_sharding) for buf in fused_kv_buffer]
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

    def clear_cache(self, indices: jax.Array):
        """Clear fused KV cache at specified indices"""
        for layer_id in range(self.layer_num):
            self.kv_buffer[layer_id] = self.kv_buffer[layer_id].at[indices].set(0)

    def set_kv_buffer_legacy(
        self,
        layer_id: int,
        loc: jax.Array,
        cache_k: jax.Array,
        cache_v: jax.Array,
    ) -> jax.Array:
        """
        Legacy interface for backward compatibility.
        This assumes contiguous cache locations and uses simple JAX operations.

        Returns:
            Updated 5D fused KV cache buffer.
        """
        layer_idx = layer_id - self.start_layer
        fused_kv = merge_kv(cache_k, cache_v)

        # Flatten both 5D -> 3D for token-level scatter, since loc contains flat token indices
        cache_5d = self.kv_buffer[layer_idx]
        num_pages, page_size, heads_x2_per_pack, packing, head_dim = cache_5d.shape
        total_cache_tokens = num_pages * page_size
        cache_3d = jax.lax.reshape(
            cache_5d,
            (total_cache_tokens, heads_x2_per_pack * packing, head_dim),
            out_sharding=P(None, self.kv_partition_axis, None),
        )

        num_tokens, _one, fkv_h, fkv_p, fkv_d = fused_kv.shape
        fused_kv_3d = jax.lax.reshape(
            fused_kv,
            (num_tokens, fkv_h * fkv_p, fkv_d),
            out_sharding=P(None, self.kv_partition_axis, None),
        )

        safe_loc = jnp.where(loc >= 0, loc, jnp.int32(total_cache_tokens))
        updated_3d = cache_3d.at[safe_loc].set(fused_kv_3d, mode="drop")

        # Reshape back to 5D
        return jax.lax.reshape(
            updated_3d,
            (num_pages, page_size, heads_x2_per_pack, packing, head_dim),
            out_sharding=P(None, None, self.kv_partition_axis, None, None),
        )


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
        swa_head_num: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.size_swa = size_swa
        self.swa_layer_nums = len(swa_attention_layer_ids)
        self.full_layer_nums = len(full_attention_layer_ids)
        self.mesh = kwargs["mesh"]
        self.kv_partition_axis = "tensor"

        # If SWA layers have different KV head count, create separate kwargs
        if swa_head_num is not None and swa_head_num != kwargs.get("head_num"):
            swa_kwargs = dict(kwargs)
            swa_kwargs["head_num"] = swa_head_num
        else:
            swa_kwargs = kwargs

        self.swa_kv_pool = token_to_kv_pool_class(
            size=size_swa,
            layer_num=self.swa_layer_nums,
            **swa_kwargs,
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

    def _remap_swa_loc(self, loc):
        """Remap full-pool locations to SWA-pool locations via the index mapping."""
        mapping = jnp.asarray(self.full_to_swa_index_mapping)
        return mapping[loc].astype(jnp.int32)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jax.Array,
        cache_k: jax.Array,
        cache_v: jax.Array,
        is_decode: bool = False,
    ):
        layer_id_pool, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            if self.full_to_swa_index_mapping is not None:
                loc = self._remap_swa_loc(loc)
            self.swa_kv_pool.set_kv_buffer(layer_id_pool, loc, cache_k, cache_v, is_decode)
        else:
            self.full_kv_pool.set_kv_buffer(layer_id_pool, loc, cache_k, cache_v, is_decode)

    def replace_buffer(self, kv_buffer: list[jax.Array]):
        assert len(kv_buffer) == len(self.layers_mapping)

        full_kv_buffer = []
        swa_kv_buffer = []
        for layer_id, layer_kv_buffer in enumerate(kv_buffer):
            _, is_swa = self.layers_mapping[layer_id]
            if is_swa:
                swa_kv_buffer.append(layer_kv_buffer)
            else:
                full_kv_buffer.append(layer_kv_buffer)

        self.swa_kv_pool.replace_buffer(swa_kv_buffer)
        self.full_kv_pool.replace_buffer(full_kv_buffer)

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
    attention_data_partition_axis: str = None,
    mesh: Mesh = None,
) -> jax.Array:
    """
    Update fused KV cache with new fused KV data.

    Args:
        fused_kv: Fused KV tensor, 5D [tokens, 1, heads//pack, pack, hdim]
        loc: Location indices [total_tokens], -1 for padding tokens
        kv_cache: Fused KV cache buffer, 5D [pages, page_size, heads//pack, pack, hdim]
        page_size: Page size for vectorized updates
        kv_partition_axis: Partition axis for sharding

    Returns:
        Updated fused KV cache (5D)
    """
    return update_fused_kv_cache(
        fused_kv,
        loc,
        kv_cache,
        page_size=page_size,
        kv_partition_axis=kv_partition_axis,
        data_partition_axis=attention_data_partition_axis,
        mesh=mesh,
    )


def update_fused_kv_cache(
    fused_kv: jax.Array,  # [tokens, 1, heads*2//packing, packing, head_dim]
    loc: jax.Array,  # [total_tokens], -1 for padding
    kv_cache: jax.Array,  # [num_pages, page_size, heads*2//packing, packing, head_dim]
    page_size: int = 1,
    kv_partition_axis: str = "tensor",
    data_partition_axis: str = None,
    mesh: Mesh = None,
) -> jax.Array:
    """
    Main fused KV cache update function.

    Args:
        fused_kv: Fused KV tensor, 5D [tokens, 1, heads*2//packing, packing, head_dim]
        loc: Location indices [total_tokens], -1 for padding tokens
        kv_cache: Fused KV cache buffer, 5D [num_pages, page_size, heads*2//packing, packing, head_dim]
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
        data_partition_axis=data_partition_axis,
        mesh=mesh,
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
    fused_kv: jax.Array,  # [tokens, 1, heads*2//packing, packing, head_dim]
    loc: jax.Array,  # [total_tokens], -1 for padding
    kv_cache: jax.Array,  # [num_pages, page_size, heads*2//packing, packing, head_dim]
    page_size: int,
    kv_partition_axis: str = "tensor",
    data_partition_axis: str = None,
    mesh: Mesh = None,
) -> jax.Array:
    """
    Vectorized fused KV cache update that handles padding and supports page_size > 1
    by grouping contiguous tokens into page-sized chunks for efficient updates.
    """

    @jax.shard_map(
        in_specs=(
            # fused_kv: 5D sharded by data and tensor
            P(data_partition_axis, None, kv_partition_axis, None, None),
            # loc: sharded by data
            P(data_partition_axis),
            # kv_cache: 5D sharded by data and tensor
            P(data_partition_axis, None, kv_partition_axis, None, None),
        ),
        out_specs=P(data_partition_axis, None, kv_partition_axis, None, None),
        mesh=mesh,
        check_vma=False,
    )
    def _sharded_update(local_fused_kv, local_loc, local_kv_cache):
        total_tokens = local_loc.shape[0]
        local_loc_int = local_loc.astype(jnp.int32)

        kv_cache_locs = jnp.where(local_loc_int == -1, 0, local_loc_int).astype(jnp.int32)
        new_kv_locs = jnp.arange(total_tokens, dtype=jnp.int32)
        slice_lens = jnp.where(local_loc_int == -1, 0, 1).astype(jnp.int32)
        num_slices = total_tokens

        num_slices_per_block = get_num_slices_per_block(
            local_fused_kv,
            local_kv_cache,
            page_size,
        )

        slot_mapping = get_slot_mapping(
            num_slices_per_block=num_slices_per_block,
            kv_cache_start_loc=kv_cache_locs,
            new_kv_start_loc=new_kv_locs,
            slice_lens=slice_lens,
        )

        num_kv_update_slices = jnp.array([num_slices], dtype=jnp.int32)

        return kv_cache_update_impl(
            new_kv=local_fused_kv,
            slices=slot_mapping,
            kv_cache=local_kv_cache,
            num_kv_update_slices=num_kv_update_slices,
            page_size=page_size,
            num_slices_per_block=num_slices_per_block,
        )

    return _sharded_update(fused_kv, loc, kv_cache)


@register_pytree_node_class
class MLATokenToKVPool(KVCache):
    """Paged KV cache for absorbed-MLA path.

    Layout matches the MLA v2 Pallas kernel (`get_kv_cache_shape`):
      `[num_pages, align_to(page_size, kv_packing)//kv_packing, kv_packing,
        align_to(kv_lora_rank, 128) + align_to(qk_rope_head_dim, 128)]`

    The cache is fully replicated across the TP mesh (single latent head, no head
    axis to shard). Cache writes happen inside the kernel via input/output aliasing,
    so `set_kv_buffer` is only used by non-kernel fallback paths.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        mesh: Mesh,
        kv_partition_axis: str = "data",  # ignored: MLA cache is replicated
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(size, page_size, dtype, layer_num, mesh, start_layer, end_layer)
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_partition_axis = kv_partition_axis

        from sgl_jax.srt.kernels.mla.v2.kernel import align_to

        self.nope_dim = align_to(kv_lora_rank, 128)
        self.rope_dim = align_to(qk_rope_head_dim, 128)
        self.kv_dim = self.nope_dim + self.rope_dim

        self._create_buffers()
        self._calculate_memory_usage()

    def tree_flatten(self):
        parent_children, parent_aux_data = super().tree_flatten()

        children = (self.kv_buffer,) + parent_children
        aux_data = {
            **parent_aux_data,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "kv_partition_axis": self.kv_partition_axis,
            "nope_dim": self.nope_dim,
            "rope_dim": self.rope_dim,
            "kv_dim": self.kv_dim,
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

        obj.kv_lora_rank = aux_data["kv_lora_rank"]
        obj.qk_rope_head_dim = aux_data["qk_rope_head_dim"]
        obj.kv_partition_axis = aux_data["kv_partition_axis"]
        obj.nope_dim = aux_data["nope_dim"]
        obj.rope_dim = aux_data["rope_dim"]
        obj.kv_dim = aux_data["kv_dim"]
        obj.kv_sharding = aux_data["kv_sharding"]

        obj.kv_buffer = kv_buffer

        return obj

    def _create_buffers(self):
        """Allocate replicated 4D paged KV buffers for the MLA v2 kernel.

        Layout matches the kernel ABI (`get_kv_cache_shape`):
            [num_pages, align_to(page_size, kv_packing) // kv_packing,
             kv_packing, align_to(kv_lora_rank, 128) + align_to(qk_rope_head_dim, 128)]

        The last dim is `align(lkv,128) + align(rope,128)` — each segment
        padded INDEPENDENTLY, NOT `align(lkv+rope, 128)`. The kernel slices
        the cache as two adjacent buffers (`bkvc_x2_ref` at offset 0,
        `bkpe_x2_ref` at offset `lkv_dim`); under-allocating with the
        single-align formula would cause out-of-bounds rope reads on shapes
        where the per-segment padding diverges from the combined-then-aligned
        size (e.g. lora=192, rope=64: 256+128=384 vs align(256,128)=256).
        DeepSeek-V3 (lora=512, rope=64) is a coincidental match — 512+128=640
        and align(576,128)=640.
        """
        from sgl_jax.srt.kernels.mla.v2.kernel import get_kv_cache_shape

        # MLA cache has no head axis to shard; replicate across the mesh.
        self.kv_sharding = NamedSharding(self.mesh, P(None, None, None, None))

        assert self.size % self.page_size == 0, "Cache size must be divisible by page size"

        total_num_pages = (self.size + self.page_size) // self.page_size
        buffer_shape = get_kv_cache_shape(
            total_num_pages=total_num_pages,
            page_size=self.page_size,
            kv_dim=self.kv_dim,
            kv_dtype=self.dtype,
        )

        per_layer_bytes = (
            buffer_shape[0]
            * buffer_shape[1]
            * buffer_shape[2]
            * buffer_shape[3]
            * jnp.dtype(self.dtype).itemsize
        )
        logger.info(
            "MLA KV cache shape per layer: %s, dtype: %s, %.2f GB",
            buffer_shape,
            self.dtype,
            per_layer_bytes / GB,
        )

        with self.mesh:
            self.kv_buffer = []
            for _ in range(self.layer_num):
                kv_buf = jax.jit(
                    lambda: jnp.zeros(shape=buffer_shape, dtype=self.dtype),
                    out_shardings=self.kv_sharding,
                )()
                self.kv_buffer.append(kv_buf)

    def _calculate_memory_usage(self):
        """Calculate memory usage for the 4D paged MLA cache."""
        total_bytes = self._buffer_bytes() * self.layer_num
        self.mem_usage = total_bytes / GB

        logger.info(
            "JAX MLA KV Cache allocated. #tokens: %s, KV size: %.2f GB",
            self.size,
            total_bytes / GB,
        )

    def _buffer_bytes(self) -> int:
        total_num_pages = (self.size + self.page_size) // self.page_size
        from sgl_jax.srt.kernels.mla.v2.kernel import get_kv_cache_shape

        shape = get_kv_cache_shape(
            total_num_pages=total_num_pages,
            page_size=self.page_size,
            kv_dim=self.kv_dim,
            kv_dtype=self.dtype,
        )
        return shape[0] * shape[1] * shape[2] * shape[3] * jnp.dtype(self.dtype).itemsize

    def get_kv_size_bytes(self):
        """Calculate KV cache size in bytes."""
        return self._buffer_bytes() * self.layer_num

    def get_fused_kv_buffer(self, layer_id: int) -> jax.Array:
        """Return the 4D paged buffer; consumed directly by the MLA v2 kernel."""
        return self.kv_buffer[layer_id - self.start_layer]

    def get_kv_buffer(self, layer_id: int) -> tuple[jax.Array, jax.Array]:
        """Split the latent buffer into (c_kv, k_pe) views for non-kernel fallbacks."""
        buf = self.kv_buffer[layer_id - self.start_layer]
        c_kv = buf[..., : self.kv_lora_rank]
        k_pe = buf[..., self.nope_dim : self.nope_dim + self.qk_rope_head_dim]
        return c_kv, k_pe

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: jax.Array,
        cache_k: jax.Array,
        cache_v: jax.Array = None,
        is_decode: bool = False,
    ) -> None:
        """Non-kernel write path. The MLA v2 kernel writes the cache via input/output
        aliasing, so this is only invoked from native fallback / eval paths and is
        intentionally left as a NotImplementedError for the absorbed PR scope.
        """
        raise NotImplementedError(
            "MLATokenToKVPool.set_kv_buffer is not supported in the absorbed path; "
            "the MLA v2 kernel writes the cache in-place via input_output_aliases."
        )

    def replace_buffer(self, kv_buffer: list[jax.Array]) -> None:
        # tp_size==1 sharding fix (issue #233): see MHATokenToKVPool.replace_buffer
        # for full rationale. MLA's kv_sharding is fully replicated (P(None)*4),
        # but the same JIT-output constraint loss happens on the tensor axis when
        # it's degenerate; trigger condition matches aolemila's original
        # _set_kv_cache_after_forward fix (mesh.shape["tensor"] == 1).
        if self.mesh.shape.get("tensor", 1) == 1 and hasattr(self, "kv_sharding"):
            kv_buffer = [jax.device_put(buf, self.kv_sharding) for buf in kv_buffer]
        self.kv_buffer[self.start_layer : self.start_layer + len(kv_buffer)] = kv_buffer

    def get_cpu_copy(self, indices):
        """Get CPU copy of KV cache for specified indices.

        `indices` selects pages along the leading axis (num_pages).
        """
        kv_cache_host = []
        for layer_id in range(self.layer_num):
            kv_host = jax.device_get(self.kv_buffer[layer_id][indices])
            kv_cache_host.append(kv_host)
        return kv_cache_host

    def load_cpu_copy(self, kv_cache_host, indices):
        """Load host copy back to device."""
        for layer_id in range(self.layer_num):
            kv_host = kv_cache_host[layer_id]
            kv_device = jax.device_put(kv_host, self.kv_sharding)
            self.kv_buffer[layer_id] = self.kv_buffer[layer_id].at[indices].set(kv_device)


@register_pytree_node_class
class MemoryPools:
    """Pytree container that uniformly manages multiple pool replace operations.

    Single JIT function + fixed donate_argnames; O(1) extension as new pools are added.

    - Construct: MemoryPools(token_to_kv_pool=..., recurrent_state_pool=...)
    - Each pool exposed via __getattr__ (mp.token_to_kv_pool / mp.recurrent_state_pool)
    - tree_flatten yields children sorted by pool name; aux_data is the sorted key tuple
    - replace_all(updates) does NOT unpack values; each value is forwarded as-is to
      the corresponding pool.replace_buffer(value) (each pool interprets its own type)
    """

    def __init__(self, **pools):
        self._pools = pools

    def __getattr__(self, name):
        # Block dunder/private lookups so we never confuse Python internals.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._pools[name]
        except KeyError:
            raise AttributeError(f"MemoryPools has no pool '{name}'") from None

    def tree_flatten(self):
        keys = sorted(self._pools.keys())
        return [self._pools[k] for k in keys], tuple(keys)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))

    def replace_all(self, updates) -> None:
        """Mirror _set_kv_cache_after_forward semantics for the multi-pool world.

        Single-pool legacy shape (bare list of per-layer KV arrays) is
        normalized to dict internally so the caller never needs to know
        whether the workload is hybrid; multi-pool dict shape is required
        only when the caller actually updates multiple pools (hybrid
        recurrent state).

        Each pool's replace_buffer carries its own implementation details
        (tp_size==1 device_put, sharding fix, ...); replace_all only does
        shape normalization and key dispatch.

        Constraints (after normalization):
        - updates.keys() must exactly match self._pools.keys() (count + names).
        - value is NOT unpacked: token_to_kv_pool receives list[jax.Array],
          recurrent_state_pool receives tuple[list[jax.Array], list[list[jax.Array]]],
          etc. Each pool interprets its own value shape.
        - Strict key matching is required because after JIT donate the old buffers
          are invalidated; any pool not present in `updates` would hold a dangling
          reference.
        """
        if isinstance(updates, list):
            updates = {"token_to_kv_pool": updates}
        if set(updates.keys()) != set(self._pools.keys()):
            raise ValueError(
                f"replace_all: updates keys {sorted(updates.keys())} "
                f"must exactly match MemoryPools keys {sorted(self._pools.keys())}"
            )
        for key, value in updates.items():
            self._pools[key].replace_buffer(value)
