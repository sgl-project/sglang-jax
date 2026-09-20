"""RecurrentStatePool -- buffer pool for linear recurrent layers (KDA/GDN)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

# Module-level cache for jitted zero-allocators (recurrent/conv buffers).
_RECURRENT_ZERO_ALLOCATOR_CACHE: dict = {}


def _get_recurrent_zero_allocator(shape, dtype, sharding):
    """Return a cached jax.jit(jnp.zeros) allocator for recurrent/conv buffers."""
    key = (
        id(sharding.mesh),
        tuple(shape),
        str(jnp.dtype(dtype)),
        repr(sharding.spec),
        getattr(sharding, "memory_kind", None),
    )
    if key not in _RECURRENT_ZERO_ALLOCATOR_CACHE:
        _RECURRENT_ZERO_ALLOCATOR_CACHE[key] = jax.jit(
            partial(jnp.zeros, shape=tuple(shape), dtype=dtype),
            out_shardings=sharding,
        )
    return _RECURRENT_ZERO_ALLOCATOR_CACHE[key]


_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _resolve_dtype(env_var: str, default):
    name = os.environ.get(env_var)
    return _DTYPE_MAP[name] if name else default


@dataclass(frozen=True)
class RecurrentStateDType:
    conv: jnp.dtype
    temporal: jnp.dtype


LINEAR_CONV = "linear"
SHORT_CONV = "short_conv"


@dataclass(frozen=True)
class ConvStateSpec:
    """[total_slots, channels, state_len]"""

    name: Literal["linear", "short_conv"]
    layers: tuple[int, ...]
    channels: int
    state_len: int


@dataclass(frozen=True)
class LinearRecurrentStateParams:
    layers: list[int]
    num_heads: int
    head_dim: int
    conv_kernel_size: int
    dtype: RecurrentStateDType
    # GDN has asymmetric K vs V projection widths (e.g.
    # Qwen3.5 GDN: num_k_heads=16/head_k_dim=128 vs num_v_heads=32/head_v_dim=128).
    # When None (KDA / Lightning / Bailing), RecurrentStatePool falls back to
    # treating K dim = V dim.
    num_k_heads: int | None = None
    head_k_dim: int | None = None


def _conv_specs(
    *,
    layers: tuple[int, ...],
    proj_size: int,
    conv_kernel_size: int,
    conv_states: tuple[ConvStateSpec, ...] | None,
) -> tuple[ConvStateSpec, ...]:
    """Every conv state the pool allocates.
    Order is not meaningful. Consumers ask by name.
    """
    if conv_states is None:
        return (ConvStateSpec(LINEAR_CONV, layers, proj_size, conv_kernel_size - 1),)
    return tuple(conv_states)


def recurrent_state_dtype() -> RecurrentStateDType:
    return RecurrentStateDType(
        conv=_resolve_dtype("SGLANG_JAX_CONV_STATE_DTYPE", jnp.bfloat16),
        temporal=_resolve_dtype("SGLANG_JAX_RECURRENT_STATE_DTYPE", jnp.float32),
    )


@register_pytree_node_class
class RecurrentStatePool:

    def __init__(
        self,
        linear_recurrent_layer_ids: list[int],
        size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        mesh: Mesh,
        dp_size: int = 1,
        recurrent_partition_axis: str = "tensor",
        conv_partition_axis: str = "tensor",
        data_partition_axis: str = "data",
        temporal_dtype=None,
        conv_dtype=None,
        num_k_heads: int | None = None,
        head_k_dim: int | None = None,
        conv_states: tuple[ConvStateSpec, ...] | None = None,
    ):
        """`size` is the **global** number of valid slots across all DP ranks
        (mirrors MHATokenToKVPool.size semantics). Internally we partition by
        DP: each rank gets `size // dp_size` valid slots + 1 dummy slot at
        index 0, so total buffer slots = size + dp_size.
        """
        state_dtype = recurrent_state_dtype()
        if temporal_dtype is None:
            temporal_dtype = state_dtype.temporal
        if conv_dtype is None:
            conv_dtype = state_dtype.conv
        self.temporal_dtype = temporal_dtype
        self.conv_dtype = conv_dtype

        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim

        assert len(set(linear_recurrent_layer_ids)) == len(linear_recurrent_layer_ids), (
            f"linear_recurrent_layer_ids must not contain duplicates, "
            f"got {linear_recurrent_layer_ids}"
        )
        self.linear_recurrent_layer_ids: list[int] = list(linear_recurrent_layer_ids)
        self.layers_mapping: dict[int, int] = {
            layer_id: idx for idx, layer_id in enumerate(self.linear_recurrent_layer_ids)
        }
        self.num_linear_recurrent_layers: int = len(self.linear_recurrent_layer_ids)

        assert (
            size % dp_size == 0
        ), f"RecurrentStatePool size ({size}) must be divisible by dp_size ({dp_size})."

        self.size = size
        self.dp_size = dp_size
        self.slots_per_rank = size // dp_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.conv_kernel_size = conv_kernel_size

        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        self.proj_size = proj_v + 2 * proj_k
        self.conv_specs: tuple[ConvStateSpec, ...] = _conv_specs(
            layers=tuple(self.linear_recurrent_layer_ids),
            proj_size=self.proj_size,
            conv_kernel_size=conv_kernel_size,
            conv_states=conv_states,
        )
        # A layer outside linear_recurrent_layer_ids has no slot to hang on.
        for spec in self.conv_specs:
            missing = sorted(set(spec.layers) - set(self.linear_recurrent_layer_ids))
            assert not missing, (
                f"conv state {spec.name!r} requested for layers {missing}, which are "
                f"not linear recurrent layers ({self.linear_recurrent_layer_ids}); "
                "a separate pool would be needed to hold them"
            )

        # Each rank reserves slot 0 as a dummy → +1 per rank.
        self.total_slots = size + dp_size

        self.mesh = mesh
        self.recurrent_partition_axis = recurrent_partition_axis
        self.conv_partition_axis = conv_partition_axis
        self.data_partition_axis = data_partition_axis

        recurrent_axis_size = mesh.shape[recurrent_partition_axis]
        conv_axis_size = mesh.shape[conv_partition_axis]
        assert num_heads % recurrent_axis_size == 0, (
            f"num_heads {num_heads} must be divisible by "
            f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
        )
        assert num_k_heads % recurrent_axis_size == 0, (
            f"num_k_heads {num_k_heads} must be divisible by "
            f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
        )
        assert self.proj_size % conv_axis_size == 0, (
            f"proj_size {self.proj_size} must be divisible by "
            f"'{conv_partition_axis}' size {conv_axis_size}"
        )
        for spec in self.conv_specs:
            assert spec.channels % conv_axis_size == 0, (
                f"{spec.name} conv channels {spec.channels} must be divisible by "
                f"'{conv_partition_axis}' size {conv_axis_size}"
            )

        self.recurrent_sharding = NamedSharding(
            mesh, P(data_partition_axis, recurrent_partition_axis, None, None)
        )
        self.conv_sharding = NamedSharding(mesh, P(data_partition_axis, conv_partition_axis, None))

        self.recurrent_buffers, self.conv_buffers = self._create_buffers()

    def _create_buffers(self) -> tuple[list, list]:
        recurrent_shape = (self.total_slots, self.num_heads, self.head_dim, self.head_dim)
        alloc_recurrent = _get_recurrent_zero_allocator(
            recurrent_shape, self.temporal_dtype, self.recurrent_sharding
        )
        alloc_conv = {
            spec.name: _get_recurrent_zero_allocator(
                (self.total_slots, spec.channels, spec.state_len),
                self.conv_dtype,
                self.conv_sharding,
            )
            for spec in self.conv_specs
        }
        with jax.set_mesh(self.mesh):
            recurrent_buffers = [alloc_recurrent() for _ in range(self.num_linear_recurrent_layers)]
            # Ragged per layer: a spec only contributes where it lists the
            # layer. clear / replace_buffer / copy_slots iterate, so only
            # get_conv_state needs to know an index.
            conv_buffers = [
                [alloc_conv[s.name]() for s in self.conv_specs if layer_id in s.layers]
                for layer_id in self.linear_recurrent_layer_ids
            ]

        return recurrent_buffers, conv_buffers

    def conv_buffer_index(self, layer_id: int, name: str) -> int:
        """Position of ``name``'s buffer within ``conv_buffers[layer]``."""
        idx = 0
        for spec in self.conv_specs:
            if layer_id not in spec.layers:
                continue
            if spec.name == name:
                return idx
            idx += 1
        raise ValueError(
            f"layer_id={layer_id} has no {name!r} conv state; it has "
            f"{[s.name for s in self.conv_specs if layer_id in s.layers]}"
        )

    def get_linear_conv_state(self, layer_id: int):
        """[total_slots, proj_size, K-1] -- the linear-attention conv."""
        return self.get_conv_state(layer_id, LINEAR_CONV)

    def get_short_conv_state(self, layer_id: int):
        """[total_slots, C, state_len] -- the N-gram short conv's state."""
        return self.get_conv_state(layer_id, SHORT_CONV)

    def get_conv_state(self, layer_id: int, name: str):
        """[total_slots, channels, state_len] for one named conv state."""
        if layer_id not in self.layers_mapping:
            raise ValueError(
                f"layer_id={layer_id} is not a registered linear recurrent layer. "
                f"Registered: {self.linear_recurrent_layer_ids}"
            )
        return self.conv_buffers[self.layers_mapping[layer_id]][
            self.conv_buffer_index(layer_id, name)
        ]

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        if layer_id not in self.layers_mapping:
            raise ValueError(
                f"layer_id={layer_id} is not a registered linear recurrent layer. "
                f"Registered: {self.linear_recurrent_layer_ids}"
            )
        idx = self.layers_mapping[layer_id]
        return self.recurrent_buffers[idx], self.conv_buffers[idx]

    def replace_buffer(self, buffers) -> None:
        new_recurrent, new_conv = buffers

        assert len(new_recurrent) == self.num_linear_recurrent_layers
        assert len(new_conv) == self.num_linear_recurrent_layers

        # tp_size==1 sharding fix: see MHATokenToKVPool.replace_buffer
        tp_degenerate = self.mesh.shape.get("tensor", 1) == 1
        for layer in range(self.num_linear_recurrent_layers):
            buf = new_recurrent[layer]
            if tp_degenerate:
                buf = jax.device_put(buf, self.recurrent_sharding)
            self.recurrent_buffers[layer] = buf

            assert len(new_conv[layer]) == len(self.conv_buffers[layer])
            for i in range(len(new_conv[layer])):
                cbuf = new_conv[layer][i]
                if tp_degenerate:
                    cbuf = jax.device_put(cbuf, self.conv_sharding)
                self.conv_buffers[layer][i] = cbuf

    def clear(self) -> None:
        for layer in range(self.num_linear_recurrent_layers):
            self.recurrent_buffers[layer] = jnp.zeros_like(self.recurrent_buffers[layer])
            for inner in range(len(self.conv_buffers[layer])):
                self.conv_buffers[layer][inner] = jnp.zeros_like(self.conv_buffers[layer][inner])

    def copy_slots(self, src_indices, dst_indices):
        """Clone src->dst slots across all layers; rows with src==0 keep dst.
        Indices are per-DP-rank local; returns new buffers for the donated pool."""
        mesh = self.mesh
        data_axis = self.data_partition_axis

        def _temporal(buf, src, dst):
            # Donated-buffer aliasing barriers: without them the scatter races the
            # gather under multi-host SPMD -> NaN. Value-preserving; do not remove.
            buf = jax.lax.optimization_barrier(buf)
            val = jnp.where((src == 0).reshape(-1, 1, 1, 1), buf[dst], buf[src])
            return jax.lax.optimization_barrier(buf.at[dst].set(val))

        def _conv(buf, src, dst):
            buf = jax.lax.optimization_barrier(buf)  # see _temporal
            val = jnp.where((src == 0).reshape(-1, 1, 1), buf[dst], buf[src])
            return jax.lax.optimization_barrier(buf.at[dst].set(val))

        copy_temporal = jax.shard_map(
            _temporal,
            mesh=mesh,
            in_specs=(
                P(data_axis, self.recurrent_partition_axis, None, None),
                P(data_axis),
                P(data_axis),
            ),
            out_specs=P(data_axis, self.recurrent_partition_axis, None, None),
            check_vma=False,
        )
        copy_conv = jax.shard_map(
            _conv,
            mesh=mesh,
            in_specs=(
                P(data_axis, self.conv_partition_axis, None),
                P(data_axis),
                P(data_axis),
            ),
            out_specs=P(data_axis, self.conv_partition_axis, None),
            check_vma=False,
        )

        new_recurrent = [
            copy_temporal(buf, src_indices, dst_indices) for buf in self.recurrent_buffers
        ]
        new_conv = [
            [copy_conv(cbuf, src_indices, dst_indices) for cbuf in inner]
            for inner in self.conv_buffers
        ]
        return new_recurrent, new_conv

    # --- pytree ---
    def tree_flatten(self):
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            tuple(self.linear_recurrent_layer_ids),
            self.size,
            self.dp_size,
            self.total_slots,
            self.num_heads,
            self.head_dim,
            self.num_k_heads,
            self.head_k_dim,
            self.conv_kernel_size,
            self.temporal_dtype,
            self.conv_dtype,
            self.mesh,
            self.recurrent_partition_axis,
            self.conv_partition_axis,
            self.data_partition_axis,
            self.recurrent_sharding,
            self.conv_sharding,
            self.conv_specs,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            linear_recurrent_layer_ids_tup,
            size,
            dp_size,
            total_slots,
            num_heads,
            head_dim,
            num_k_heads,
            head_k_dim,
            conv_kernel_size,
            temporal_dtype,
            conv_dtype,
            mesh,
            recurrent_partition_axis,
            conv_partition_axis,
            data_partition_axis,
            recurrent_sharding,
            conv_sharding,
            conv_specs,
        ) = aux_data
        obj = cls.__new__(cls)
        obj.linear_recurrent_layer_ids = list(linear_recurrent_layer_ids_tup)
        obj.layers_mapping = {
            layer_id: idx for idx, layer_id in enumerate(obj.linear_recurrent_layer_ids)
        }
        obj.num_linear_recurrent_layers = len(obj.linear_recurrent_layer_ids)
        obj.conv_specs = conv_specs
        obj.size = size
        obj.dp_size = dp_size
        obj.slots_per_rank = size // dp_size
        obj.total_slots = total_slots
        obj.num_heads = num_heads
        obj.head_dim = head_dim
        obj.num_k_heads = num_k_heads
        obj.head_k_dim = head_k_dim
        obj.conv_kernel_size = conv_kernel_size
        obj.temporal_dtype = temporal_dtype
        obj.conv_dtype = conv_dtype
        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        obj.proj_size = proj_v + 2 * proj_k
        obj.mesh = mesh
        obj.recurrent_partition_axis = recurrent_partition_axis
        obj.conv_partition_axis = conv_partition_axis
        obj.data_partition_axis = data_partition_axis
        obj.recurrent_sharding = recurrent_sharding
        obj.conv_sharding = conv_sharding
        new_recurrent, new_conv = children
        obj.recurrent_buffers = list(new_recurrent)
        obj.conv_buffers = [list(inner) for inner in new_conv]
        return obj
