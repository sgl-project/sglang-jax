"""RecurrentStatePool — recurrent + conv state management for KDA layers.

Design reference: RFC-0015 §RecurrentStatePool object design (v2, list containers).
- Dual list containers:
    recurrent_buffers: list[jax.Array] of length L, each [N+1, H, D, D] (default f32)
    conv_buffers: list[list[jax.Array]] outer L, inner currently 1, each [N+1, K-1, proj] (default bf16)
- Slot 0 is reserved as dummy; valid slots start from 1 (aligned with sglang PyTorch MambaPool).
- Does NOT inherit from KVCache (KVCache abstract methods are meaningless for recurrent state).
- list index semantics: outer list index is the KDA-subset index (0..L-1), NOT the global model layer_id.
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
    """Recurrent + conv state pool (per-request slot indexing).

    RFC-0015 §RecurrentStatePool object design line 113-185.
    """

    def __init__(
        self,
        num_layers: int,
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

        # Dimension bookkeeping
        self.num_layers = num_layers
        self.max_num_reqs = max_num_reqs
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size

        # proj_size MUST be written as proj_v + 2*proj_k (Implementation Guide 1.1 note #4
        # mandates this; do NOT simplify to `* 3` since future GQA may diverge proj_v/proj_k).
        proj_v = num_heads * head_dim
        proj_k = num_heads * head_dim
        self.proj_size = proj_v + 2 * proj_k

        # Boundary asserts
        assert num_layers > 0, f"num_layers must be > 0, got {num_layers}"
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
            for _ in range(num_layers)
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
            for _ in range(num_layers)
        ]

        # Slot management: starts from 1; slot 0 is reserved as dummy.
        self.free_slots: list[int] = list(range(1, max_num_reqs + 1))

    # --- interface methods ---
    def alloc(self, need_size: int = 1):
        """Allocate need_size slots from free_slots; clear-on-alloc via list element mutation.

        RFC §RecurrentStatePool object design line 178 + Implementation Guide 1.1 note #6.
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

    def replace_buffer(self, buffers) -> None:
        """Update both buffer-list references after a JIT donate.

        RFC §RecurrentStatePool object design line 180-181:
        - buffers: tuple[list[jax.Array], list[list[jax.Array]]]
            [0] = new_recurrent_buffers list (length num_layers)
            [1] = new_conv_buffers list-of-list (outer length num_layers; inner lengths must match)
        - Per-element device_put handles the tp_size==1 sharding fix (issue #233).

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
        we cannot replace the list reference wholesale because downstream KDA layers
        may hold a captured reference to self.recurrent_buffers.
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
        children = (self.recurrent_buffers, self.conv_buffers)
        aux = (
            self.num_layers,
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
            num_layers,
            max_num_reqs,
            num_heads,
            head_dim,
            conv_kernel_size,
            temporal_dtype,
            conv_dtype,
        ) = aux
        obj = cls.__new__(cls)
        obj.num_layers = num_layers
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
