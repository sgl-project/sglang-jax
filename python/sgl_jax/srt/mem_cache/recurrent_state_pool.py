"""RecurrentStatePool -- buffer pool for linear recurrent layers (KDA/Mamba/GDN)."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _resolve_dtype(env_var: str, default):
    name = os.environ.get(env_var)
    return _DTYPE_MAP[name] if name else default


def _ceil_to(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor * divisor


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
    ):
        if temporal_dtype is None:
            temporal_dtype = _resolve_dtype("SGLANG_JAX_RECURRENT_STATE_DTYPE", jnp.float32)
        if conv_dtype is None:
            conv_dtype = _resolve_dtype("SGLANG_JAX_CONV_STATE_DTYPE", jnp.bfloat16)
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

        self.size = size
        self.dp_size = dp_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.conv_kernel_size = conv_kernel_size

        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        self.proj_size = proj_v + 2 * proj_k

        # total_slots: size+1 (for dummy slot 0), ceil to dp_size
        self.total_slots = _ceil_to(size + 1, dp_size)

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

        self.recurrent_sharding = NamedSharding(
            mesh, P(data_partition_axis, recurrent_partition_axis, None, None)
        )
        self.conv_sharding = NamedSharding(mesh, P(data_partition_axis, conv_partition_axis, None))

        self.recurrent_buffers, self.conv_buffers = self._create_buffers()

    def _create_buffers(self) -> tuple[list, list]:
        recurrent_shape = (self.total_slots, self.num_heads, self.head_dim, self.head_dim)
        conv_shape = (self.total_slots, self.proj_size, self.conv_kernel_size - 1)
        temporal_dtype = self.temporal_dtype
        conv_dtype = self.conv_dtype

        with self.mesh:
            recurrent_buffers = []
            for _ in range(self.num_linear_recurrent_layers):
                buf = jax.jit(
                    lambda: jnp.zeros(shape=recurrent_shape, dtype=temporal_dtype),
                    out_shardings=self.recurrent_sharding,
                )()
                recurrent_buffers.append(buf)

            conv_buffers = []
            for _ in range(self.num_linear_recurrent_layers):
                inner = []
                buf = jax.jit(
                    lambda: jnp.zeros(shape=conv_shape, dtype=conv_dtype),
                    out_shardings=self.conv_sharding,
                )()
                inner.append(buf)
                conv_buffers.append(inner)

        return recurrent_buffers, conv_buffers

    def clear_slot(self, idx_or_indices) -> None:
        indices = [idx_or_indices] if isinstance(idx_or_indices, int) else list(idx_or_indices)
        if not indices:
            return

        idx_arr = jnp.asarray(indices, dtype=jnp.int32)
        with jax.set_mesh(self.mesh):
            for layer in range(self.num_linear_recurrent_layers):
                self.recurrent_buffers[layer] = self.recurrent_buffers[layer].at[idx_arr].set(0)
                for inner in range(len(self.conv_buffers[layer])):
                    self.conv_buffers[layer][inner] = (
                        self.conv_buffers[layer][inner].at[idx_arr].set(0)
                    )

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

    # --- pytree ---
    def tree_flatten(self):
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            tuple(self.linear_recurrent_layer_ids),
            self.size,
            self.dp_size,
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
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            linear_recurrent_layer_ids_tup,
            size,
            dp_size,
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
        ) = aux_data
        obj = cls.__new__(cls)
        obj.linear_recurrent_layer_ids = list(linear_recurrent_layer_ids_tup)
        obj.layers_mapping = {
            layer_id: idx for idx, layer_id in enumerate(obj.linear_recurrent_layer_ids)
        }
        obj.num_linear_recurrent_layers = len(obj.linear_recurrent_layer_ids)
        obj.size = size
        obj.dp_size = dp_size
        obj.total_slots = _ceil_to(size + 1, dp_size)
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
        obj.recurrent_sharding = NamedSharding(
            mesh, P(data_partition_axis, recurrent_partition_axis, None, None)
        )
        obj.conv_sharding = NamedSharding(mesh, P(data_partition_axis, conv_partition_axis, None))
        new_recurrent, new_conv = children
        obj.recurrent_buffers = list(new_recurrent)
        obj.conv_buffers = [list(inner) for inner in new_conv]
        return obj
