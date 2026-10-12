"""Request-owned FP32 continuation state, separate from compressed KV history."""

import jax
import jax.numpy as jnp

from sgl_jax.srt.mem_cache.deepseek_v4.pool import (  # noqa: F401 (re-exported)
    _buffer_nbytes,
    _build_buffer_updates,
    _validate_buffer_updates,
    allocate_buffer,
    native_hca_layout,
    scatter_sharding,
)


def score_slice(shape):
    """Index selecting the score half of an empty state of ``shape``.

    C4/indexer states are ``[.., 8, 4*D]`` = [two contents | two scores] on the last
    axis; the C128 state is ``[.., 128, 2*D]`` = [content | score], or in the native
    HCA layout ``[.., 128, 2, D]`` where the score is index 1 of the packing axis.
    """
    if len(shape) == 4:
        return (Ellipsis, 1, slice(None))
    return (Ellipsis, slice(shape[-1] // 2, None))


@jax.tree_util.register_pytree_node_class
class DeepseekV4CompressStatePool:
    """Each DP shard indexes global ReqToTokenPool slots directly, including 0.

    ReqToTokenPool does not partition its free list by DP. Therefore each shard
    reserves all ``size`` request positions; it must not use req_slot % (R/DP).
    Position ``size`` is padding. No independent state allocator or free list.
    The request lifecycle owns invalidation; the backend resets on its
    initialization mask before consuming a recycled slot. reset() supplies the empty state:
    content=0, score=-inf, with the last axis [two contents | two scores] for C4
    and [content | score] for C128.
    """

    def __init__(self, size, spec, mesh, dp_size=1):
        if size <= 0 or dp_size <= 0 or mesh.shape.get("data") != dp_size:
            raise ValueError("state size must be positive and DP must match the mesh")
        self._configure(size, spec, mesh, dp_size)
        with jax.set_mesh(mesh):
            self.buffers = {
                family: tuple(
                    allocate_buffer(shape, jnp.float32, mesh).at[score_slice(shape)].set(-jnp.inf)
                    for _ in layers
                )
                for family, (layers, shape) in self.layout.items()
            }

    def _configure(self, size, spec, mesh, dp_size):
        self._metadata = (size, spec, mesh, dp_size)
        self.size, self.spec, self.mesh, self.dp_size = size, spec, mesh, dp_size
        self.padding_index = size
        self.slots_per_rank = size + 1
        slots = (size + 1) * dp_size
        c128_shape = (
            (slots, 128, 2, spec.head_dim)
            if native_hca_layout()
            else (slots, 128, 2 * spec.head_dim)
        )
        self.layout = {
            "c4": (spec.layers(4), (slots, 8, 4 * spec.head_dim)),
            "c128": (spec.layers(128), c128_shape),
            "indexer": (spec.layers(4), (slots, 8, 4 * spec.index_head_dim)),
        }
        self.layer_to_buffer = {
            f: {layer: i for i, layer in enumerate(layers)}
            for f, (layers, _) in self.layout.items()
        }

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

    @property
    def mem_usage(self):
        return self.nbytes / 1024**3

    def validate_buffer_updates(self, buffers):
        _validate_buffer_updates(self.buffers, buffers)

    def replace_buffer(self, buffers):
        self.validate_buffer_updates(buffers)
        self.buffers = {key: tuple(arrays) for key, arrays in buffers.items()}

    def build_buffer_updates(self, layer_updates):
        """Merge {layer_id: {compressor/indexer: array}} without committing state."""
        return _build_buffer_updates(self, layer_updates, state=True)

    def state_indices(self, request_slots, valid_mask):
        return jnp.where(
            valid_mask & (request_slots >= 0) & (request_slots < self.size),
            request_slots,
            self.padding_index,
        )

    def get_buffer(self, family, layer_id):
        return self.buffers[family][self.layer_to_buffer[family][layer_id]]

    def write(self, family, layer_id, request_slots, values, valid_mask, dp_rank=0):
        i = self.layer_to_buffer[family][layer_id]
        arrays = list(self.buffers[family])
        array = arrays[i]
        local = self.state_indices(request_slots, valid_mask)
        indices = jnp.where(
            (local < self.size) & (dp_rank >= 0) & (dp_rank < self.dp_size),
            local + dp_rank * self.slots_per_rank,
            array.shape[0],
        )
        arrays[i] = array.at[indices].set(
            values, mode="drop", out_sharding=scatter_sharding(self.mesh, array.ndim)
        )
        self.buffers = {**self.buffers, family: tuple(arrays)}

    def reset(self, request_slots, valid_mask, dp_rank=0):
        """Initialize/reinitialize selected slots without touching padded rows."""
        for family, arrays in self.buffers.items():
            for layer, i in self.layer_to_buffer[family].items():
                shape = (request_slots.shape[0], *arrays[i].shape[1:])
                empty = jnp.zeros(shape, jnp.float32).at[score_slice(shape)].set(-jnp.inf)
                self.write(family, layer, request_slots, empty, valid_mask, dp_rank)
