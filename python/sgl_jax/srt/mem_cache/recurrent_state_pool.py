"""RecurrentStatePool -- recurrent + conv state buffer pool for linear recurrent layers.

- Dual list containers:
    recurrent_buffers: list[jax.Array] of length L, each [N+1, H, D, D] (default f32)
    conv_buffers: list[list[jax.Array]] outer L, inner currently 1, each [N+1, K-1, proj] (default bf16)
- Slot 0 is reserved as dummy; valid slots start from 1 (aligned with sglang PyTorch MambaPool).
- Pure buffer pool: slot allocator state lives in HybridReqToTokenPool, mirroring
  the MHATokenToKVPool / TokenToKVPoolAllocator separation. This pool only
  exposes per-slot buffer manipulation primitives (clear_slot, replace_buffer,
  clear) that the allocator can drive.
- Does NOT inherit from KVCache (KVCache abstract methods are meaningless for recurrent state).
- Outer list is keyed by local 0..L-1 internally; the public API
  (get_linear_recurrent_layer_cache) accepts the model-global layer_id and
  translates via self.layers_mapping. Mirrors sgl-jax SWAKVPool's
  swa_attention_layer_ids / layers_mapping pattern.
- Naming is generic (linear_recurrent_*) to allow Mamba / GDN reuse beyond KDA.
- Buffer creation mirrors MHATokenToKVPool: jax.jit(... out_shardings=...)()
  inside `with self.mesh:` so the pool ships sharded arrays end-to-end and
  the tp_size==1 device_put fix uses the persisted sharding (no probing).
"""

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
    """dtype priority: env var > default.

    Constructor-arg priority is handled in __init__ (constructor arg > env var > default).
    """
    name = os.environ.get(env_var)
    return _DTYPE_MAP[name] if name else default


@register_pytree_node_class
class RecurrentStatePool:
    """Recurrent + conv state buffer pool (per-slot indexing, no slot allocator)."""

    def __init__(
        self,
        linear_recurrent_layer_ids: list[int],
        max_num_reqs: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        mesh: Mesh,
        recurrent_partition_axis: str = "tensor",
        conv_partition_axis: str = "tensor",
        temporal_dtype=None,
        conv_dtype=None,
        num_k_heads: int | None = None,
        head_k_dim: int | None = None,
    ):
        # dtype priority: constructor arg > env var > default
        # (use `is None` instead of `or` to avoid issues with falsy dtype objects)
        if temporal_dtype is None:
            temporal_dtype = _resolve_dtype("SGLANG_JAX_RECURRENT_STATE_DTYPE", jnp.float32)
        if conv_dtype is None:
            conv_dtype = _resolve_dtype("SGLANG_JAX_CONV_STATE_DTYPE", jnp.bfloat16)
        self.temporal_dtype = temporal_dtype
        self.conv_dtype = conv_dtype

        # Per-head K dims default to V dims (current Kimi-Linear convention;
        # sglang upstream KimiLinearStateShape.create() defaults the same way).
        # Models with GQA-style linear-recurrent attention (different
        # num_k_heads / head_k_dim from V) can pass these explicitly.
        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim

        # linear_recurrent_layer_ids: model-global layer ids of linear recurrent layers
        # (KDA / Mamba / GDN ...); duplicates are not allowed since they would collide
        # in layers_mapping. layers_mapping: global layer_id -> local 0..L-1 index;
        # used internally so the public get_linear_recurrent_layer_cache API can
        # accept a global layer_id. Mirrors sgl-jax SWAKVPool's
        # swa_attention_layer_ids / layers_mapping pattern.
        assert len(set(linear_recurrent_layer_ids)) == len(linear_recurrent_layer_ids), (
            f"linear_recurrent_layer_ids must not contain duplicates, "
            f"got {linear_recurrent_layer_ids}"
        )
        self.linear_recurrent_layer_ids: list[int] = list(linear_recurrent_layer_ids)
        self.layers_mapping: dict[int, int] = {
            layer_id: idx for idx, layer_id in enumerate(self.linear_recurrent_layer_ids)
        }
        # Cached derived count; kept so existing clear/clear_slot/replace_buffer
        # loops can keep referring to self.num_linear_recurrent_layers.
        self.num_linear_recurrent_layers: int = len(self.linear_recurrent_layer_ids)

        # Dimension bookkeeping
        self.max_num_reqs = max_num_reqs
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.conv_kernel_size = conv_kernel_size

        # proj_size = proj_v + 2*proj_k. For current Kimi-Linear num_k_heads ==
        # num_heads and head_k_dim == head_dim, so this collapses to
        # 3*num_heads*head_dim. GQA-capable models override the K dims via
        # the num_k_heads / head_k_dim constructor args.
        proj_v = num_heads * head_dim
        proj_k = num_k_heads * head_k_dim
        self.proj_size = proj_v + 2 * proj_k

        # Boundary asserts. NOTE: linear_recurrent_layer_ids may legitimately be empty
        # (degenerate pool with no recurrent layers); we do NOT assert num_linear_recurrent_layers > 0.
        assert max_num_reqs > 0, f"max_num_reqs must be > 0, got {max_num_reqs}"
        assert num_heads > 0, f"num_heads must be > 0, got {num_heads}"
        assert head_dim > 0, f"head_dim must be > 0, got {head_dim}"
        assert num_k_heads > 0, f"num_k_heads must be > 0, got {num_k_heads}"
        assert head_k_dim > 0, f"head_k_dim must be > 0, got {head_k_dim}"
        # K=1 would make conv_buffers[l][i] second dim (K-1) zero; min meaningful value is 2.
        assert conv_kernel_size >= 2, (
            f"conv_kernel_size must be >= 2 (got {conv_kernel_size}); "
            "K=1 produces empty conv buffers."
        )
        assert self.proj_size > 0, f"proj_size must be > 0, got {self.proj_size}"

        # Mesh + sharding specs. Partition axis names are kept on the instance
        # (so _create_buffers / replace_buffer can reach them) but tp_size is
        # NOT cached -- we read mesh.shape[axis] only here for the divisibility
        # asserts, mirroring MHATokenToKVPool's "no self.tp_size" pattern.
        self.mesh = mesh
        self.recurrent_partition_axis = recurrent_partition_axis
        self.conv_partition_axis = conv_partition_axis
        recurrent_axis_size = mesh.shape[recurrent_partition_axis]
        conv_axis_size = mesh.shape[conv_partition_axis]
        assert num_heads % recurrent_axis_size == 0, (
            f"num_heads {num_heads} must be divisible by mesh axis "
            f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
        )
        assert num_k_heads % recurrent_axis_size == 0, (
            f"num_k_heads {num_k_heads} must be divisible by mesh axis "
            f"'{recurrent_partition_axis}' size {recurrent_axis_size}"
        )
        assert self.proj_size % conv_axis_size == 0, (
            f"proj_size {self.proj_size} must be divisible by mesh axis "
            f"'{conv_partition_axis}' size {conv_axis_size}"
        )
        # recurrent_buffers shape: [N+1, H, D, D] -> partition H on tensor axis.
        self.recurrent_sharding = NamedSharding(mesh, P(None, recurrent_partition_axis, None, None))
        # conv_buffers[layer][inner] shape: [N+1, K-1, proj_size] -> partition
        # proj_size on tensor axis (matches the projection's TP split).
        self.conv_sharding = NamedSharding(mesh, P(None, None, conv_partition_axis))

        # Dual list containers; each element has +1 row reserved for dummy slot 0.
        self.recurrent_buffers, self.conv_buffers = self._create_buffers()

    # --- buffer creation (mirrors MHATokenToKVPool._create_buffers) ---
    def _create_buffers(self) -> tuple[list, list]:
        """Allocate per-layer recurrent + conv buffers under the persisted
        sharding, mirroring MHATokenToKVPool._create_buffers.

        Each buffer is created via jax.jit(..., out_shardings=sharding)() inside
        `with self.mesh:` so the array ships with the right sharding from the
        start; downstream replace_buffer / clear can rely on
        self.recurrent_sharding / self.conv_sharding instead of probing.
        """
        recurrent_shape = (
            self.max_num_reqs + 1,
            self.num_heads,
            self.head_dim,
            self.head_dim,
        )
        conv_shape = (self.max_num_reqs + 1, self.conv_kernel_size - 1, self.proj_size)
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
                # Inner list currently has length 1; reserved for future
                # multi-conv-segment expansion (mirroring PyTorch
                # KimiLinearStateShape.conv: List[tuple]).
                inner = []
                buf = jax.jit(
                    lambda: jnp.zeros(shape=conv_shape, dtype=conv_dtype),
                    out_shardings=self.conv_sharding,
                )()
                inner.append(buf)
                conv_buffers.append(inner)

        return recurrent_buffers, conv_buffers

    # --- interface methods ---
    def clear_slot(self, idx_or_indices) -> None:
        """Zero the per-slot view of recurrent + conv buffers for the given slot(s).

        - Accepts int or iterable[int] (clear-on-alloc entry point).
        - Used by HybridReqToTokenPool for clear-on-alloc when a recurrent slot
          is handed out: the allocator picks the slot, this method wipes any
          stale state left from a previous occupant.
        - Cross-layer uses list element mutation (write back to list[l]);
          intra-layer uses vectorized scatter (clears all slots in one
          ``.at[].set(0)``); the conv inner list is also iterated (does NOT
          assume fixed length 1).
        - **Critical pitfall**: do NOT use local-variable assignment
          (``new = self.recurrent_buffers[l].at[...].set(...)`` without writing
          back to the list) -- when multiple layers share the pool reference,
          updates from layers 0..N-1 would all be lost.
        - No-op for an empty index iterable.
        """
        indices = [idx_or_indices] if isinstance(idx_or_indices, int) else list(idx_or_indices)
        if not indices:
            return

        idx_arr = jnp.asarray(indices, dtype=jnp.int32)
        for layer in range(self.num_linear_recurrent_layers):
            self.recurrent_buffers[layer] = self.recurrent_buffers[layer].at[idx_arr].set(0)
            for inner in range(len(self.conv_buffers[layer])):
                self.conv_buffers[layer][inner] = self.conv_buffers[layer][inner].at[idx_arr].set(0)

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        """Read the per-layer view, keyed by model-global layer_id.

        Mirrors sgl-jax KV pool get_fused_kv_buffer + PyTorch mamba2_layer_cache.
        Returns a 2-tuple (recurrent_per_layer, conv_per_layer); both are the
        list elements themselves (no copy; `is` relation holds with
        recurrent_buffers[idx] / conv_buffers[idx]).

        Consumers (e.g., the linear recurrent attention backend) read with this
        method, then update functionally by doing
        ``new_layer = cur_layer.at[indices].set(new_state)`` and returning the
        new buffer up to the model layer (which collects per-layer outputs into
        ``(layers_recurrent, layers_conv)`` and hands them to
        ``MemoryPools.replace_all`` outside the JIT). No setter method is
        exposed: the pool's internal lists are only swapped via
        ``replace_buffer``, so backend calls have no side effect on the pool.
        """
        if layer_id not in self.layers_mapping:
            raise ValueError(
                f"layer_id={layer_id} is not a registered linear recurrent layer. "
                f"Registered: {self.linear_recurrent_layer_ids}"
            )
        idx = self.layers_mapping[layer_id]
        return self.recurrent_buffers[idx], self.conv_buffers[idx]

    def replace_buffer(self, buffers) -> None:
        """Update both buffer-list references after a JIT donate.

        - buffers: tuple[list[jax.Array], list[list[jax.Array]]]
            [0] = new_recurrent_buffers list (length num_linear_recurrent_layers)
            [1] = new_conv_buffers list-of-list (outer length num_linear_recurrent_layers; inner lengths must match)
        - Per-element device_put (single-device only) carries the persisted
          sharding back onto each new buffer. NamedSharding constraint can be
          lost on JIT output under tp_size==1; explicit device_put restores
          it before the slice assign so the next JIT trace sees a stable shape.
          Mirrors MHATokenToKVPool.set_kv_buffer's tp_size==1 fix (issue #233).
        """
        new_recurrent, new_conv = buffers

        # Length asserts (outer + inner).
        assert len(new_recurrent) == self.num_linear_recurrent_layers, (
            f"recurrent_buffers list length {len(new_recurrent)} "
            f"!= num_linear_recurrent_layers {self.num_linear_recurrent_layers}"
        )
        assert (
            len(new_conv) == self.num_linear_recurrent_layers
        ), f"conv_buffers outer list length {len(new_conv)} != num_linear_recurrent_layers {self.num_linear_recurrent_layers}"
        for layer in range(self.num_linear_recurrent_layers):
            assert len(new_conv[layer]) == len(self.conv_buffers[layer]), (
                f"conv_buffers[{layer}] inner length {len(new_conv[layer])} "
                f"!= existing {len(self.conv_buffers[layer])}"
            )

        # tp_size==1 sharding fix using persisted sharding (mirrors MHA pool's
        # `if hasattr(self, "kv_sharding") and len(...) == 1` guard).
        for layer in range(self.num_linear_recurrent_layers):
            buf = new_recurrent[layer]
            if hasattr(self, "recurrent_sharding") and len(self.recurrent_sharding.device_set) == 1:
                buf = jax.device_put(buf, self.recurrent_sharding)
            self.recurrent_buffers[layer] = buf

        for layer in range(self.num_linear_recurrent_layers):
            for i in range(len(new_conv[layer])):
                buf = new_conv[layer][i]
                if hasattr(self, "conv_sharding") and len(self.conv_sharding.device_set) == 1:
                    buf = jax.device_put(buf, self.conv_sharding)
                self.conv_buffers[layer][i] = buf

    def clear(self) -> None:
        """Full reset: zero out every layer's recurrent + conv buffer.

        MUST use list element mutation (assigning each layer in place);
        we cannot replace the list reference wholesale because downstream
        recurrent layers may hold a captured reference to self.recurrent_buffers.

        jnp.zeros_like preserves the input array's sharding, so the cleared
        buffers keep self.recurrent_sharding / self.conv_sharding without any
        explicit device_put.

        Slot allocator state lives in HybridReqToTokenPool and is reset
        independently by the caller (e.g., HybridReqToTokenPool.clear()).
        """
        for layer in range(self.num_linear_recurrent_layers):
            self.recurrent_buffers[layer] = jnp.zeros_like(self.recurrent_buffers[layer])
            for inner in range(len(self.conv_buffers[layer])):
                self.conv_buffers[layer][inner] = jnp.zeros_like(self.conv_buffers[layer][inner])

    # --- pytree ---
    def tree_flatten(self):
        # list is a default pytree container; auto-expands to 2L leaves
        # (outer L recurrent + L inner conv lists each yielding their own leaves).
        # aux carries tuple(linear_recurrent_layer_ids) instead of num_linear_recurrent_layers, so
        # tree_unflatten can reconstruct layers_mapping (otherwise JIT donate would
        # lose the global-layer-id -> local-index mapping). list is unhashable, so
        # we wrap as tuple to satisfy aux's hashability requirement.
        # Mesh + partition axis names + sharding specs are also carried so
        # replace_buffer's device_put can use the persisted sharding after a
        # JIT donate cycle (mirrors MHATokenToKVPool aux shape).
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            tuple(self.linear_recurrent_layer_ids),
            self.max_num_reqs,
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
            self.recurrent_sharding,
            self.conv_sharding,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            linear_recurrent_layer_ids_tup,
            max_num_reqs,
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
            recurrent_sharding,
            conv_sharding,
        ) = aux_data
        obj = cls.__new__(cls)
        # Restore linear_recurrent_layer_ids + rebuild layers_mapping
        # (must rebuild here, otherwise JIT donate would lose the mapping).
        obj.linear_recurrent_layer_ids = list(linear_recurrent_layer_ids_tup)
        obj.layers_mapping = {
            layer_id: idx for idx, layer_id in enumerate(obj.linear_recurrent_layer_ids)
        }
        obj.num_linear_recurrent_layers = len(obj.linear_recurrent_layer_ids)
        obj.max_num_reqs = max_num_reqs
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
        obj.recurrent_sharding = recurrent_sharding
        obj.conv_sharding = conv_sharding
        # `children` is restored by jax; force-cast back to mutable list so that
        # subsequent list element mutation (`recurrent_buffers[l] = ...`) keeps working
        # -- jax may otherwise restore the container as a tuple.
        new_recurrent, new_conv = children
        obj.recurrent_buffers = list(new_recurrent)
        obj.conv_buffers = [list(inner) for inner in new_conv]
        return obj
