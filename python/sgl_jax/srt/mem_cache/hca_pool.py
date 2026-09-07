"""Physical KV and recurrent-state pools for HCA."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.hca.compressor import init_hca_state_pool
from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing
from sgl_jax.srt.mem_cache.memory_pool import KVCache


def _align(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@register_pytree_node_class
class HCARecurrentStatePool:
    """Own one FP32 compressor-state array for each HCA layer."""

    def __init__(
        self,
        layer_ids: list[int] | tuple[int, ...],
        size: int,
        mesh: jax.sharding.Mesh,
        *,
        dp_size: int = 1,
        compress_ratio: int = 128,
        head_dim: int = 512,
    ):
        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("HCA layer ids must be unique")
        if size <= 0 or size % dp_size:
            raise ValueError("HCA recurrent size must be positive and divisible by dp_size")
        if compress_ratio != 128 or head_dim != 512:
            raise ValueError("production HCA recurrent state requires r128/d512")

        self.linear_recurrent_layer_ids = layer_ids
        self.layers_mapping = {layer_id: i for i, layer_id in enumerate(layer_ids)}
        self.size = size
        self.dp_size = dp_size
        self.slots_per_rank = size // dp_size
        self.total_slots = size + dp_size
        self.mesh = mesh
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.state_sharding = NamedSharding(mesh, P("data", None, None, None))
        self.state_buffers = self._create_buffers()

    def _create_buffers(self):
        allocate = jax.jit(
            lambda: init_hca_state_pool(
                self.total_slots,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
            ),
            out_shardings=self.state_sharding,
        )
        with jax.set_mesh(self.mesh):
            return [allocate() for _ in self.linear_recurrent_layer_ids]

    def _layer_index(self, layer_id: int) -> int:
        try:
            return self.layers_mapping[int(layer_id)]
        except KeyError as exc:
            raise ValueError(f"layer_id={layer_id} is not an HCA layer") from exc

    def get_hca_state(self, layer_id: int):
        return self.state_buffers[self._layer_index(layer_id)]

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        """Expose the standard recurrent-pool `(temporal, conv)` contract."""
        return self.get_hca_state(layer_id), []

    def reset_slots(self, global_slots) -> None:
        """Reset newly allocated global rows before request-slot reuse."""
        global_slots = jnp.asarray(global_slots, jnp.int32)
        if not global_slots.size:
            return
        reset = init_hca_state_pool(
            global_slots.shape[0],
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
        )
        with jax.set_mesh(self.mesh):
            for index, buffer in enumerate(self.state_buffers):
                self.state_buffers[index] = buffer.at[global_slots].set(
                    reset,
                    mode="promise_in_bounds",
                    unique_indices=True,
                    out_sharding=P("data", None, None, None),
                )

    def clear(self) -> None:
        self.state_buffers = self._create_buffers()

    def copy_slots(self, src_indices, dst_indices):
        """Clone rank-local recurrent rows for SGLang's copy-on-write path."""
        src_indices = jax.sharding.reshard(src_indices, P("data"))
        dst_indices = jax.sharding.reshard(dst_indices, P("data"))

        def _copy(buffer, src, dst):
            buffer = jax.lax.optimization_barrier(buffer)
            values = jnp.where(
                (src == 0).reshape(-1, 1, 1, 1),
                buffer[dst],
                buffer[src],
            )
            return jax.lax.optimization_barrier(buffer.at[dst].set(values))

        copy = jax.shard_map(
            _copy,
            mesh=self.mesh,
            in_specs=(
                P("data", None, None, None),
                P("data"),
                P("data"),
            ),
            out_specs=P("data", None, None, None),
            check_vma=False,
        )
        return [copy(buffer, src_indices, dst_indices) for buffer in self.state_buffers], []

    def replace_buffer(self, buffers) -> None:
        if isinstance(buffers, dict):
            buffers = buffers["state_buffers"]
        elif isinstance(buffers, tuple):
            buffers, conv = buffers
            if conv:
                raise ValueError("HCA recurrent state has no convolution buffers")
        if len(buffers) != len(self.state_buffers):
            raise ValueError("one recurrent update is required per HCA layer")
        self.state_buffers = list(buffers)

    def get_size_bytes(self) -> int:
        return sum(buffer.size * buffer.dtype.itemsize for buffer in self.state_buffers)

    def tree_flatten(self):
        children = (tuple(self.state_buffers), tuple())
        aux = (
            self.linear_recurrent_layer_ids,
            self.size,
            self.dp_size,
            self.total_slots,
            self.mesh,
            self.compress_ratio,
            self.head_dim,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        layer_ids, size, dp_size, total_slots, mesh, compress_ratio, head_dim = aux
        obj = object.__new__(cls)
        obj.linear_recurrent_layer_ids = layer_ids
        obj.layers_mapping = {layer_id: i for i, layer_id in enumerate(layer_ids)}
        obj.size = size
        obj.dp_size = dp_size
        obj.slots_per_rank = size // dp_size
        obj.total_slots = total_slots
        obj.mesh = mesh
        obj.compress_ratio = compress_ratio
        obj.head_dim = head_dim
        obj.state_sharding = NamedSharding(mesh, P("data", None, None, None))
        obj.state_buffers = list(children[0])
        return obj


@register_pytree_node_class
class HCAKVPool(KVCache):
    """Own paged BF16 sliding-window and compressed-history buffers."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        layer_num: int,
        mesh: jax.sharding.Mesh,
        *,
        max_num_requests: int,
        max_context_len: int,
        head_dim: int = 512,
        window_size: int = 128,
        compress_ratio: int = 128,
        dp_size: int = 1,
        layer_ids: list[int] | tuple[int, ...] | None = None,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            mesh,
            start_layer,
            end_layer,
        )
        if size <= 0 or size % dp_size:
            raise ValueError("HCA token capacity must be positive and divisible by dp_size")
        if max_num_requests <= 0 or max_num_requests % dp_size:
            raise ValueError("request capacity must be positive and divisible by dp_size")
        if page_size < 2 or window_size % page_size:
            raise ValueError("HCA page_size must be >=2 and divide window_size")
        if compress_ratio != 128 or window_size != 128 or head_dim != 512:
            raise ValueError("production HCA cache requires r128/window128/d512")

        self.max_num_requests = max_num_requests
        self.max_context_len = max_context_len
        self.head_dim = head_dim
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.dp_size = dp_size
        if layer_ids is None:
            layer_ids = range(self.start_layer, self.start_layer + layer_num)
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if len(self.layer_ids) != layer_num or len(set(self.layer_ids)) != layer_num:
            raise ValueError("layer_ids must contain one unique id per HCA layer")
        self.layers_mapping = {layer_id: index for index, layer_id in enumerate(self.layer_ids)}
        self.packing = get_dtype_packing(dtype)
        if page_size % self.packing:
            raise ValueError("HCA page_size must be divisible by dtype packing")
        self.cache_sharding = NamedSharding(mesh, P("data", None, None, None))
        self.window_buffer, self.compressed_buffer = self._create_buffers()
        self.mem_usage = self.get_kv_size_bytes() / 2**30

    def _pages_per_rank(self, entries_per_rank: int, *, request_padding: bool) -> int:
        pages = math.ceil(entries_per_rank / self.page_size)
        if request_padding:
            pages += self.max_num_requests // self.dp_size
        return pages

    def _cache_shape(self, pages_per_rank: int):
        return (
            self.dp_size * (pages_per_rank + 1),
            self.page_size // self.packing,
            self.packing,
            _align(self.head_dim, 128),
        )

    @property
    def window_pages_per_rank(self) -> int:
        requests = self.max_num_requests // self.dp_size
        return requests * (self.window_size // self.page_size)

    @property
    def compressed_pages_per_rank(self) -> int:
        entries = (self.size // self.dp_size) // self.compress_ratio
        return self._pages_per_rank(entries, request_padding=True)

    @property
    def compressed_pool_size(self) -> int:
        return self.compressed_pages_per_rank * self.page_size * self.dp_size

    def _create_buffers(self):
        allocate_window = jax.jit(
            lambda: jnp.zeros(self._cache_shape(self.window_pages_per_rank), self.dtype),
            out_shardings=self.cache_sharding,
        )
        allocate_compressed = jax.jit(
            lambda: jnp.zeros(self._cache_shape(self.compressed_pages_per_rank), self.dtype),
            out_shardings=self.cache_sharding,
        )
        with jax.set_mesh(self.mesh):
            return (
                [allocate_window() for _ in range(self.layer_num)],
                [allocate_compressed() for _ in range(self.layer_num)],
            )

    def create_window_allocator(self):
        from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        usable = self.window_pages_per_rank * self.page_size * self.dp_size
        return PagedTokenToKVPoolAllocator(usable, self.page_size, self, dp_size=self.dp_size)

    def create_compressed_allocator(self):
        from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        return PagedTokenToKVPoolAllocator(
            self.compressed_pool_size,
            self.page_size,
            self,
            dp_size=self.dp_size,
        )

    def get_fused_kv_buffer(self, layer_id: int):
        return self.window_buffer[self._layer_index(layer_id)]

    def _layer_index(self, layer_id: int) -> int:
        try:
            return self.layers_mapping[int(layer_id)]
        except KeyError as exc:
            raise ValueError(f"layer_id={layer_id} is not an HCA layer") from exc

    def get_kv_buffer(self, layer_id: int):
        buffer = self.get_fused_kv_buffer(layer_id)
        return buffer, buffer

    def set_kv_buffer(self, layer_id, loc, cache_k, cache_v=None, is_decode=False):
        del cache_v, is_decode
        index = self._layer_index(layer_id)
        self.window_buffer[index] = self._scatter_slots(self.window_buffer[index], loc, cache_k)

    def _scatter_slots(self, cache, locations, values):
        locations = jax.sharding.reshard(locations, P("data"))
        values = jax.sharding.reshard(values, P("data", None))

        @jax.shard_map(
            mesh=self.mesh,
            in_specs=(P("data", None, None, None), P("data"), P("data", None)),
            out_specs=P("data", None, None, None),
            check_vma=False,
        )
        def _scatter(local_cache, local_locations, local_values):
            flat = local_cache.reshape(-1, local_cache.shape[-1])
            values = jnp.pad(
                local_values.astype(local_cache.dtype),
                ((0, 0), (0, flat.shape[-1] - local_values.shape[-1])),
            )
            safe = jnp.where(local_locations >= 0, local_locations, flat.shape[0])
            return flat.at[safe].set(values, mode="drop").reshape(local_cache.shape)

        return _scatter(cache, locations.astype(jnp.int32), values)

    def materialize(self, layer_id: int, locations, *, compressed: bool = False):
        index = self._layer_index(layer_id)
        cache = self.compressed_buffer[index] if compressed else self.window_buffer[index]
        locations = jax.sharding.reshard(locations, P("data", None))

        @jax.shard_map(
            mesh=self.mesh,
            in_specs=(P("data", None, None, None), P("data", None)),
            out_specs=P("data", None, None),
            check_vma=False,
        )
        def _gather(local_cache, local_locations):
            flat = local_cache.reshape(-1, local_cache.shape[-1])
            valid = (local_locations > 0) & (local_locations < flat.shape[0])
            safe = jnp.where(valid, local_locations, 0)
            values = flat[safe, : self.head_dim]
            return jnp.where(valid[..., None], values, 0)

        return _gather(cache, locations.astype(jnp.int32))

    def replace_buffer(self, buffers) -> None:
        if isinstance(buffers, dict):
            window = buffers["window_buffer"]
            compressed = buffers["compressed_buffer"]
        else:
            window, compressed = buffers
        if len(window) != self.layer_num or len(compressed) != self.layer_num:
            raise ValueError("one window and compressed update is required per HCA layer")
        self.window_buffer = list(window)
        self.compressed_buffer = list(compressed)

    def get_kv_size_bytes(self) -> int:
        return sum(
            buffer.size * buffer.dtype.itemsize
            for buffer in self.window_buffer + self.compressed_buffer
        )

    def get_cpu_copy(self, indices):
        del indices
        return jax.device_get((self.window_buffer, self.compressed_buffer))

    def load_cpu_copy(self, kv_cache_cpu, indices):
        del indices
        window, compressed = kv_cache_cpu
        self.window_buffer = [jax.device_put(value, self.cache_sharding) for value in window]
        self.compressed_buffer = [
            jax.device_put(value, self.cache_sharding) for value in compressed
        ]

    def tree_flatten(self):
        children = (tuple(self.window_buffer), tuple(self.compressed_buffer))
        aux = {
            "size": self.size,
            "page_size": self.page_size,
            "dtype": self.dtype,
            "layer_num": self.layer_num,
            "mesh": self.mesh,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "max_num_requests": self.max_num_requests,
            "max_context_len": self.max_context_len,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "compress_ratio": self.compress_ratio,
            "dp_size": self.dp_size,
            "layer_ids": self.layer_ids,
            "mem_usage": self.mem_usage,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        for name, value in aux.items():
            setattr(obj, name, value)
        obj.packing = get_dtype_packing(obj.dtype)
        obj.layers_mapping = {layer_id: index for index, layer_id in enumerate(obj.layer_ids)}
        obj.cache_sharding = NamedSharding(obj.mesh, P("data", None, None, None))
        obj.window_buffer = list(children[0])
        obj.compressed_buffer = list(children[1])
        return obj


__all__ = ["HCAKVPool", "HCARecurrentStatePool"]
