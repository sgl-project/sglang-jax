"""BF16 V4 resources. Addresses are rank-local; page zero is always padding.

TP/EP replicate the single KV head; only the leading capacity axis is sharded
on the existing attention ``data`` mesh axis. No raw full-history KV is stored.
"""

import os
from dataclasses import dataclass
from functools import partial
from math import prod
from operator import index

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache.memory_pool import KVCache


@dataclass(frozen=True)
class DeepseekV4CacheSpec:
    compress_ratios: tuple[int, ...]
    head_dim: int = 512
    index_head_dim: int = 128

    def __post_init__(self):
        object.__setattr__(self, "compress_ratios", tuple(self.compress_ratios))
        if not self.compress_ratios or any(r not in (0, 4, 128) for r in self.compress_ratios):
            raise ValueError("V4 backbone ratios must be 0, 4 or 128")
        if self.head_dim <= 0 or self.index_head_dim <= 0:
            raise ValueError("V4 cache dimensions must be positive")

    @classmethod
    def from_config(cls, config):
        n = config.num_hidden_layers
        if n <= 0 or len(config.compress_ratios) < n:
            raise ValueError("compress_ratios must cover all backbone layers")
        return cls(tuple(config.compress_ratios[:n]), config.head_dim, config.index_head_dim)

    def layers(self, ratio):
        return tuple(i for i, r in enumerate(self.compress_ratios) if r == ratio)

    def history_bytes_per_page(self, page_size):
        # BF16, no additional value cache: the same representation serves K/V.
        return 2 * (
            len(self.layers(4)) * (self.head_dim + self.index_head_dim) * (page_size // 4)
            + len(self.layers(128)) * self.head_dim * (page_size // 128)
        )

    @property
    def swa_bytes_per_token(self):
        return 2 * len(self.compress_ratios) * self.head_dim

    @property
    def state_bytes_per_request(self):
        return 4 * (
            len(self.layers(4)) * 8 * 4 * (self.head_dim + self.index_head_dim)
            + len(self.layers(128)) * 128 * 2 * self.head_dim
        )


def native_hca_layout() -> bool:
    """``DSV4_HCA_NATIVE_LAYOUT`` (default on): keep the ratio-128 compressor state in
    the HCA kernels' physical ``[slots, 128, 2, D]`` layout and the ratio-128 KV
    cache in their ``[pages, 1, page/128, D]`` layout, instead of reshaping into those
    layouts per layer per step. On TPU the ``[..., 2, D]`` layout occupies exactly the
    same HBM as ``[..., 2*D]`` (measured on v7x), but the reshape between them is a
    relayout copy of the whole buffer on the way in and out of every HCA layer."""
    return os.environ.get("DSV4_HCA_NATIVE_LAYOUT", "1") != "0"


def allocate_buffer(shape, dtype, mesh):
    sharding = NamedSharding(mesh, P("data", *([None] * (len(shape) - 1))))
    with jax.set_mesh(mesh):
        return jax.jit(partial(jnp.zeros, tuple(shape), dtype), out_shardings=sharding)()


def scatter_sharding(mesh, ndim):
    # Explicit meshes require a result sharding for scattered indices. Auto
    # meshes infer it from the operand and reject explicit scatter specs.
    data_axis = mesh.axis_names.index("data")
    if mesh.axis_types[data_axis] == jax.sharding.AxisType.Explicit:
        return NamedSharding(mesh, P("data", *([None] * (ndim - 1))))
    return None


def _buffer_nbytes(buffers):
    return sum(a.size * a.dtype.itemsize for arrays in buffers.values() for a in arrays)


def _validate_buffer_updates(current, updates):
    if set(updates) != set(current):
        raise ValueError("V4 update must contain every buffer family")
    for family, old in current.items():
        new = updates[family]
        if len(new) != len(old) or any(
            a.shape != b.shape or a.dtype != b.dtype for a, b in zip(old, new)
        ):
            raise ValueError(f"V4 {family} update shape/dtype mismatch")


def _layer_ratio(spec, layer_id):
    layer_id = index(layer_id)
    if not 0 <= layer_id < len(spec.compress_ratios):
        raise IndexError(f"V4 layer {layer_id} is out of range")
    return spec.compress_ratios[layer_id]


def _build_buffer_updates(pool, layer_updates, *, state=False):
    """Merge semantic per-layer updates without mutating either array owner."""
    buffers = {family: list(arrays) for family, arrays in pool.buffers.items()}
    for layer_id, updates in layer_updates.items():
        ratio = _layer_ratio(pool.spec, layer_id)
        resources = {} if state else {"swa": "swa"}
        if ratio:
            resources["compressor" if state else "compressed"] = f"c{ratio}"
        if ratio == 4:
            resources["indexer"] = "indexer"
        for resource, array in updates.items():
            if resource not in resources:
                raise ValueError(f"V4 layer {layer_id} has no {resource!r} resource")
            family = resources[resource]
            buffers[family][pool.layer_to_buffer[family][layer_id]] = array
    buffers = {family: tuple(arrays) for family, arrays in buffers.items()}
    _validate_buffer_updates(pool.buffers, buffers)
    return buffers


@jax.tree_util.register_pytree_node_class
class DeepseekV4TokenToKVPool(KVCache):
    """Own SWA and compressed history; request continuation state is separate.

    The ordinary single-buffer KVCache accessors cannot represent a V4 layer.
    Backends use the resource accessors and return functional buffer updates.
    """

    def __init__(self, size, size_swa, page_size, spec, mesh, dp_size=1, dtype=jnp.bfloat16):
        if jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
            raise ValueError("V4 initial KV cache requires BF16")
        if page_size not in (128, 256):
            raise ValueError("V4 history/SWA page_size must be 128 or 256")
        if dp_size <= 0 or mesh.shape.get("data") != dp_size:
            raise ValueError("dp_size must match the attention data mesh")
        if any(s <= 0 or s % (page_size * dp_size) for s in (size, size_swa)):
            raise ValueError("V4 capacities must be positive page/DP aligned token counts")
        self._configure(size, size_swa, page_size, spec, mesh, dp_size)
        self.buffers = {
            family: tuple(allocate_buffer(shape, self.dtype, mesh) for _ in layers)
            for family, (layers, shape) in self.layout.items()
        }

    def _configure(self, size, size_swa, page_size, spec, mesh, dp_size):
        super().__init__(size, page_size, jnp.bfloat16, len(spec.compress_ratios), mesh)
        self._metadata = (size, size_swa, page_size, spec, mesh, dp_size)
        self.size_swa = size_swa
        self.spec, self.mesh, self.dp_size = spec, mesh, dp_size
        self.dtype = jnp.bfloat16
        self.layer_num = len(spec.compress_ratios)
        self.pages_per_rank = size // dp_size // page_size
        self.swa_slots_per_rank = size_swa // dp_size + page_size
        pages = (self.pages_per_rank + 1) * dp_size
        self.layout = {
            "swa": (
                tuple(range(self.layer_num)),
                (self.swa_slots_per_rank * dp_size, spec.head_dim),
            ),
            "c4": (spec.layers(4), (pages, page_size // 4, spec.head_dim)),
            "c128": (
                spec.layers(128),
                # The HCA kernels address this cache as [pages, 1, page/128, D]; allocating
                # it that way (same bytes) avoids a relayout copy per HCA layer per step.
                (
                    (pages, 1, page_size // 128, spec.head_dim)
                    if native_hca_layout()
                    else (pages, page_size // 128, spec.head_dim)
                ),
            ),
            "indexer": (spec.layers(4), (pages, page_size // 4, spec.index_head_dim)),
        }
        self.layer_to_buffer = {
            family: {layer: i for i, layer in enumerate(layers)}
            for family, (layers, _) in self.layout.items()
        }

        self.mem_usage = (
            sum(
                len(layers) * prod(shape) * jnp.dtype(self.dtype).itemsize
                for layers, shape in self.layout.values()
            )
            / 1024**3
        )

    def tree_flatten(self):
        return (self.buffers,), self._metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        obj = object.__new__(cls)
        obj._configure(*metadata)
        obj.buffers = children[0]
        return obj

    @property
    def nbytes(self):
        return _buffer_nbytes(self.buffers)

    def get_kv_size_bytes(self) -> int:
        """Global logical KV bytes, including padding, excluding compressor state."""
        return self.nbytes

    def validate_buffer_updates(self, buffers):
        _validate_buffer_updates(self.buffers, buffers)

    def replace_buffer(self, buffers):
        """Commit a complete, shape-preserving family update after forward."""
        self.validate_buffer_updates(buffers)
        self.buffers = {key: tuple(arrays) for key, arrays in buffers.items()}

    def build_buffer_updates(self, layer_updates):
        """Merge {layer_id: {swa/compressed/indexer: array}} into a full payload.

        Omitted layers/resources keep their current arrays. The returned payload
        is consumed by MemoryPools.replace_all; this method does not commit it.
        """
        return _build_buffer_updates(self, layer_updates)

    def get_swa_buffer(self, layer_id: int) -> jax.Array:
        _layer_ratio(self.spec, layer_id)
        return self.get_buffer("swa", layer_id)

    def _compressed_ratio(self, layer_id):
        ratio = _layer_ratio(self.spec, layer_id)
        if not ratio:
            raise ValueError(f"V4 layer {layer_id} has no compressed history")
        return ratio

    def get_compressed_buffer(self, layer_id: int) -> jax.Array:
        return self.get_buffer(f"c{self._compressed_ratio(layer_id)}", layer_id)

    def get_compressed_page_size(self, layer_id: int) -> int:
        return self.page_size // self._compressed_ratio(layer_id)

    def get_indexer_buffer(self, layer_id: int) -> jax.Array:
        if _layer_ratio(self.spec, layer_id) != 4:
            raise ValueError(f"V4 layer {layer_id} has no indexer history")
        return self.get_buffer("indexer", layer_id)

    def get_fused_kv_buffer(self, layer_id: int) -> jax.Array:
        raise NotImplementedError("V4 requires separate SWA/compressed/indexer accessors")

    def get_kv_buffer(self, layer_id: int) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError("V4 requires separate SWA/compressed/indexer accessors")

    def set_kv_buffer(self, layer_id, loc, cache_k, cache_v, is_decode):
        raise NotImplementedError("V4 requires resource writes or build_buffer_updates")

    def get_cpu_copy(self, indices):
        raise NotImplementedError("V4 has no multi-resource CPU snapshot protocol")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("V4 has no multi-resource CPU snapshot protocol")

    def get_buffer(self, family, layer_id):
        return self.buffers[family][self.layer_to_buffer[family][layer_id]]

    def set_buffer(self, family, layer_id, value):
        i = self.layer_to_buffer[family][layer_id]
        old = self.buffers[family][i]
        if old.shape != value.shape or old.dtype != value.dtype:
            raise ValueError("V4 buffer shape/dtype mismatch")
        arrays = list(self.buffers[family])
        arrays[i] = value
        self.buffers = {**self.buffers, family: tuple(arrays)}

    def write(self, family, layer_id, loc, values, valid_mask, dp_rank=0):
        """Masked reference scatter; compressed loc is in compressed-entry units.

        This operates on global arrays. A shard_map consumer uses the same local
        loc without the rank offset. Invalid entries are dropped, never sent to
        slot zero (duplicate padding writes must not race with real tokens).
        """
        array = self.get_buffer(family, layer_id)
        flat = array.reshape(-1, array.shape[-1])
        per_rank = flat.shape[0] // self.dp_size
        reserved = self.page_size // {"swa": 1, "c4": 4, "c128": 128, "indexer": 4}[family]
        valid = (
            valid_mask
            & (loc >= reserved)
            & (loc < per_rank)
            & (dp_rank >= 0)
            & (dp_rank < self.dp_size)
        )
        index = jnp.where(valid, loc + dp_rank * per_rank, flat.shape[0])
        self.set_buffer(
            family,
            layer_id,
            flat.at[index]
            .set(values, mode="drop", out_sharding=scatter_sharding(self.mesh, 2))
            .reshape(array.shape),
        )
