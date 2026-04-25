"""RecurrentStatePool — recurrent + conv state management for linear recurrent layers.

Design reference: RFC-0015 §RecurrentStatePool (v2, list containers; layer_id-keyed accessors).
- Dual list containers:
    recurrent_buffers: list[jax.Array] of length L, each [N+1, H, D, D] (default f32)
    conv_buffers: list[list[jax.Array]] outer L, inner currently 1, each [N+1, K-1, proj] (default bf16)
- Slot 0 is reserved as dummy; valid slots start from 1 (aligned with sglang PyTorch MambaPool).
- Does NOT inherit from KVCache (KVCache abstract methods are meaningless for recurrent state).
- Outer list is keyed by local 0..L-1 internally; the public API
  (get_/set_linear_recurrent_layer_cache) accepts the model-global layer_id and
  translates via self.layers_mapping. Mirrors sgl-jax SWAKVPool's
  swa_attention_layer_ids / layers_mapping pattern.
- Naming is generic (linear_recurrent_*) to allow Mamba / GDN reuse beyond KDA.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

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


@jax.tree_util.register_pytree_node_class
class RecurrentStatePool:
    """Recurrent + conv state pool (per-request slot indexing)."""

    def __init__(
        self,
        linear_recurrent_layer_ids: list[int],
        max_num_reqs: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int,
        temporal_dtype=None,
        conv_dtype=None,
    ):
        # dtype priority: constructor arg > env var > default
        # (use `is None` instead of `or` to avoid issues with falsy dtype objects)
        if temporal_dtype is None:
            temporal_dtype = _resolve_dtype("SGLANG_JAX_RECURRENT_STATE_DTYPE", jnp.float32)
        if conv_dtype is None:
            conv_dtype = _resolve_dtype("SGLANG_JAX_CONV_STATE_DTYPE", jnp.bfloat16)
        self.temporal_dtype = temporal_dtype
        self.conv_dtype = conv_dtype

        # linear_recurrent_layer_ids: model-global layer ids of linear recurrent layers
        # (KDA / Mamba / GDN ...); duplicates are not allowed since they would collide
        # in layers_mapping. layers_mapping: global layer_id -> local 0..L-1 index;
        # used internally so the public get_/set_linear_recurrent_layer_cache API can
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
        # Cached derived count; kept so existing alloc/clear/replace_buffer loops
        # can keep referring to self.num_layers.
        self.num_layers: int = len(self.linear_recurrent_layer_ids)

        # Dimension bookkeeping
        self.max_num_reqs = max_num_reqs
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size

        # proj_size MUST be written as proj_v + 2*proj_k.
        # Do NOT simplify to `* 3`: future GQA could diverge proj_v from proj_k,
        # and the formula would silently break.
        proj_v = num_heads * head_dim
        proj_k = num_heads * head_dim
        self.proj_size = proj_v + 2 * proj_k

        # Boundary asserts. NOTE: linear_recurrent_layer_ids may legitimately be empty
        # (degenerate pool with no recurrent layers); we do NOT assert num_layers > 0.
        assert max_num_reqs > 0, f"max_num_reqs must be > 0, got {max_num_reqs}"
        assert num_heads > 0, f"num_heads must be > 0, got {num_heads}"
        assert head_dim > 0, f"head_dim must be > 0, got {head_dim}"
        # K=1 would make conv_buffers[l][i] second dim (K-1) zero; min meaningful value is 2.
        assert conv_kernel_size >= 2, (
            f"conv_kernel_size must be >= 2 (got {conv_kernel_size}); "
            "K=1 produces empty conv buffers."
        )
        assert self.proj_size > 0, f"proj_size must be > 0, got {self.proj_size}"

        # Dual list containers; each element has +1 row reserved for dummy slot 0.
        # recurrent: list[Array] of length L, each [N+1, H, D, D].
        self.recurrent_buffers: list = [
            jnp.zeros(
                (max_num_reqs + 1, num_heads, head_dim, head_dim),
                dtype=self.temporal_dtype,
            )
            for _ in range(self.num_layers)
        ]
        # conv: list[list[Array]], outer L, inner currently 1 (reserved for future
        # multi-conv-segment expansion, mirroring PyTorch KimiLinearStateShape.conv: List[tuple]).
        self.conv_buffers: list = [
            [
                jnp.zeros(
                    (max_num_reqs + 1, conv_kernel_size - 1, self.proj_size),
                    dtype=self.conv_dtype,
                )
            ]
            for _ in range(self.num_layers)
        ]

        # Slot management: starts from 1; slot 0 is reserved as dummy.
        self.free_slots: list[int] = list(range(1, max_num_reqs + 1))

    # --- interface methods ---
    def alloc(self, need_size: int = 1):
        """Allocate need_size slots from free_slots; clear-on-alloc via list element mutation.

        - Returns None if free_slots is insufficient (state unchanged).
        - clear-on-alloc: cross-layer uses list element mutation (write back to list[l]);
          intra-layer uses vectorized scatter (clears all slots in one .at[].set(0));
          conv inner list is also iterated (does NOT assume fixed length 1).
        - **Critical pitfall**: do NOT use local-variable assignment
          (`new = self.recurrent_buffers[l].at[...].set(...)` without writing back to the list)
          — when multiple layers share the pool reference, updates from layers 0..N-1
          would all be lost.
        """
        if len(self.free_slots) < need_size:
            return None

        indices = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        # Cross-layer list element mutation + intra-layer vectorized scatter.
        idx_arr = jnp.asarray(indices, dtype=jnp.int32)
        for layer in range(self.num_layers):
            self.recurrent_buffers[layer] = self.recurrent_buffers[layer].at[idx_arr].set(0)
            for inner in range(len(self.conv_buffers[layer])):
                self.conv_buffers[layer][inner] = self.conv_buffers[layer][inner].at[idx_arr].set(0)
        return indices

    def free(self, idx):
        """Return slot(s) to free_slots. Accepts int or list[int] (mirrors ReqToTokenPool.free)."""
        if isinstance(idx, int):
            self.free_slots.append(idx)
        else:
            self.free_slots.extend(idx)

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        """Read the per-layer view, keyed by model-global layer_id.

        Mirrors sgl-jax KV pool get_fused_kv_buffer + PyTorch mamba2_layer_cache.
        Returns a 2-tuple (recurrent_per_layer, conv_per_layer); both are the
        list elements themselves (no copy; `is` relation holds with
        recurrent_buffers[idx] / conv_buffers[idx]).

        External consumers (e.g., the linear recurrent attention backend) MUST
        go through this method and the matching set_linear_recurrent_layer_cache
        instead of indexing recurrent_buffers / conv_buffers directly. This
        matches the existing KV pool convention (attention backends never
        touch kv_buffer directly).
        """
        if layer_id not in self.layers_mapping:
            raise ValueError(
                f"layer_id={layer_id} is not a registered linear recurrent layer. "
                f"Registered: {self.linear_recurrent_layer_ids}"
            )
        idx = self.layers_mapping[layer_id]
        return self.recurrent_buffers[idx], self.conv_buffers[idx]

    def set_linear_recurrent_layer_cache(
        self,
        layer_id: int,
        indices,
        new_recurrent,
        new_conv,
    ):
        """Write back the per-layer cache, keyed by model-global layer_id.

        Mirrors sgl-jax KV pool set_kv_buffer. Performs list element mutation
        internally on both recurrent_buffers[idx] and each conv_buffers[idx][i].

        **Why list element mutation matters**: list is a mutable Python container,
        and multiple layers share the same RecurrentStatePool instance. After
        layer N writes via this method, layer N+1 reads the updated value via
        get_linear_recurrent_layer_cache (because both see the same list slot).
        If the implementation accidentally assigned to a local variable instead
        of writing back into the list, layers 0..N-1 updates would silently be
        lost in a multi-layer forward.

        new_conv must be a list whose length equals the inner length of
        conv_buffers[idx] (currently always 1); the assert guards against
        future multi-conv-segment misuse.
        """
        if layer_id not in self.layers_mapping:
            raise ValueError(
                f"layer_id={layer_id} is not a registered linear recurrent layer. "
                f"Registered: {self.linear_recurrent_layer_ids}"
            )
        idx = self.layers_mapping[layer_id]
        self.recurrent_buffers[idx] = self.recurrent_buffers[idx].at[indices].set(new_recurrent)
        assert len(new_conv) == len(self.conv_buffers[idx]), (
            f"new_conv length {len(new_conv)} mismatches conv_buffers[{idx}] inner length "
            f"{len(self.conv_buffers[idx])}"
        )
        for i, new_c in enumerate(new_conv):
            self.conv_buffers[idx][i] = self.conv_buffers[idx][i].at[indices].set(new_c)

    def replace_buffer(self, buffers) -> None:
        """Update both buffer-list references after a JIT donate.

        - buffers: tuple[list[jax.Array], list[list[jax.Array]]]
            [0] = new_recurrent_buffers list (length num_layers)
            [1] = new_conv_buffers list-of-list (outer length num_layers; inner lengths must match)
        - Per-element device_put handles the tp_size==1 sharding fix
          (see sgl-project/sglang-jax#233 for the original fix).

        Sharding detection: probe each element's `.sharding` attribute for single-device.
        On the Phase 1 CPU unit-test path this triggers the single-device branch where
        device_put is effectively a no-op (no side effects).

        **Phase 1 limitation**: this fix is only exercised through the single-device no-op
        branch under CPU unit tests. Whether accessing `.sharding` on a list element after
        JIT donate is legal under real NamedSharding (mesh + tensor-axis split) depends on
        JAX implementation details and is **NOT verified in Phase 1**. Phase 2, after wiring
        up mesh, must add a real test: per-element sharding stays consistent across
        donate→replace; if the sharding metadata is unavailable or drifts, this implementation
        must switch to constructor-time injection of (mesh, axis) and explicit NamedSharding
        construction (mirroring the original fix in model_runner.py:667-681).
        """
        new_recurrent, new_conv = buffers

        # Length asserts (outer + inner).
        assert len(new_recurrent) == self.num_layers, (
            f"recurrent_buffers list length {len(new_recurrent)} "
            f"!= num_layers {self.num_layers}"
        )
        assert (
            len(new_conv) == self.num_layers
        ), f"conv_buffers outer list length {len(new_conv)} != num_layers {self.num_layers}"
        for layer in range(self.num_layers):
            assert len(new_conv[layer]) == len(self.conv_buffers[layer]), (
                f"conv_buffers[{layer}] inner length {len(new_conv[layer])} "
                f"!= existing {len(self.conv_buffers[layer])}"
            )

        # Per-element sharding fix + list element mutation write-back
        # (consistent with alloc / clear).
        for layer in range(self.num_layers):
            buf = new_recurrent[layer]
            old = self.recurrent_buffers[layer]
            if hasattr(old, "sharding") and len(old.sharding.device_set) == 1:
                buf = jax.device_put(buf, old.sharding)
            self.recurrent_buffers[layer] = buf

        for layer in range(self.num_layers):
            for i in range(len(new_conv[layer])):
                buf = new_conv[layer][i]
                old = self.conv_buffers[layer][i]
                if hasattr(old, "sharding") and len(old.sharding.device_set) == 1:
                    buf = jax.device_put(buf, old.sharding)
                self.conv_buffers[layer][i] = buf

    def clear(self) -> None:
        """Full reset: zero out every layer and reset free_slots.

        MUST use list element mutation (assigning each layer in place);
        we cannot replace the list reference wholesale because downstream
        recurrent layers may hold a captured reference to self.recurrent_buffers.
        """
        for layer in range(self.num_layers):
            self.recurrent_buffers[layer] = jnp.zeros_like(self.recurrent_buffers[layer])
            for inner in range(len(self.conv_buffers[layer])):
                self.conv_buffers[layer][inner] = jnp.zeros_like(self.conv_buffers[layer][inner])
        self.free_slots = list(range(1, self.max_num_reqs + 1))

    # --- pytree ---
    def tree_flatten(self):
        # list is a default pytree container; auto-expands to 2L leaves
        # (outer L recurrent + L inner conv lists each yielding their own leaves).
        # aux carries tuple(linear_recurrent_layer_ids) instead of num_layers, so
        # tree_unflatten can reconstruct layers_mapping (otherwise JIT donate would
        # lose the global-layer-id -> local-index mapping). list is unhashable, so
        # we wrap as tuple to satisfy aux's hashability requirement.
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            tuple(self.linear_recurrent_layer_ids),
            self.max_num_reqs,
            self.num_heads,
            self.head_dim,
            self.conv_kernel_size,
            self.temporal_dtype,
            self.conv_dtype,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            linear_recurrent_layer_ids_tup,
            max_num_reqs,
            num_heads,
            head_dim,
            conv_kernel_size,
            temporal_dtype,
            conv_dtype,
        ) = aux
        obj = cls.__new__(cls)
        # Restore linear_recurrent_layer_ids + rebuild layers_mapping
        # (must rebuild here, otherwise JIT donate would lose the mapping).
        obj.linear_recurrent_layer_ids = list(linear_recurrent_layer_ids_tup)
        obj.layers_mapping = {
            layer_id: idx for idx, layer_id in enumerate(obj.linear_recurrent_layer_ids)
        }
        obj.num_layers = len(obj.linear_recurrent_layer_ids)
        obj.max_num_reqs = max_num_reqs
        obj.num_heads = num_heads
        obj.head_dim = head_dim
        obj.conv_kernel_size = conv_kernel_size
        obj.temporal_dtype = temporal_dtype
        obj.conv_dtype = conv_dtype
        proj_v = num_heads * head_dim
        proj_k = num_heads * head_dim
        obj.proj_size = proj_v + 2 * proj_k
        # `children` is restored by jax; force-cast back to mutable list so that
        # subsequent list element mutation (`recurrent_buffers[l] = ...`) keeps working
        # — jax may otherwise restore the container as a tuple.
        new_recurrent, new_conv = children
        obj.recurrent_buffers = list(new_recurrent)
        obj.conv_buffers = [list(inner) for inner in new_conv]
        obj.free_slots = list(range(1, max_num_reqs + 1))
        return obj
