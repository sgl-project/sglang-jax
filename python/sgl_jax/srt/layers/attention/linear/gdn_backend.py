"""Gated-DeltaNet attention backend.

Inherits :class:`LinearRecurrentAttnBackend` for shared metadata
(``cu_q_lens`` / ``recurrent_indices`` / ``has_initial_state``) and
pytree boilerplate. Owns the (fused) conv1d weight + delta-rule params
(``A_log``, ``dt_bias``); the parent layer hands in ``mixed_qkv``
(a per-device block-concat ``[Q | K | V]`` of size ``conv_dim``
channels) plus ``b``, ``a``, and a :class:`RecurrentStatePool`. State
(conv + recurrent) is fetched from the pool internally via the base
class's :meth:`get_layer_cache` helper.

Sharding pattern: the conv + recurrence pipeline runs inside
:func:`jax.shard_map` with explicit ``in_specs`` / ``out_specs``, with
the head axis pinned to ``"tensor"`` so each device sees only its local
shard. The kernels then operate on per-shard head counts
(``n_kq // TP``, ``n_v // TP``) without relying on JAX sharding
inference. Returns ``(core_attn_out, new_conv, new_rec)`` shaped for
``RecurrentStatePool.write_layer``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.gdn import (
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


def _mesh_tp_size(mesh: jax.sharding.Mesh) -> int:
    """TP size = mesh size on the ``"tensor"`` axis (1 if absent)."""
    if mesh is None:
        return 1
    shape = getattr(mesh, "shape", None)
    if shape is None or "tensor" not in shape:
        return 1
    return int(shape["tensor"])


class GDNAttnBackend(LinearRecurrentAttnBackend):
    """Gated-DeltaNet attention backend.

    Owns the conv1d weight + delta-rule params; dispatches conv1d + ragged
    delta-rule (extend) or single-step delta-rule (decode) under
    ``jax.shard_map``. Reads ``cu_q_lens`` / ``recurrent_indices`` /
    ``has_initial_state`` from ``self.forward_metadata``, populated by
    the base class's :meth:`get_forward_metadata` before each forward.
    """

    def __init__(
        self,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__(mesh=mesh)
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size

        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        # Per-shard slicing in the kernels uses `num_*_heads // tp` (integer
        # division). `MergedColumnParallelLinear` only checks that `key_dim`
        # and `value_dim` are divisible by TP, which is not enough: e.g.
        # `num_k_heads=1, head_k_dim=128, TP=2` gives `key_dim=128` (divisible)
        # but `num_k_heads // TP = 0`, and the per-shard reshape silently
        # produces zero-head arrays. GQA also relies on `num_v_heads %
        # num_k_heads == 0` so the per-step `jnp.repeat` produces exactly
        # `num_v_heads` heads.
        tp = _mesh_tp_size(mesh)
        if num_k_heads % tp != 0:
            raise ValueError(
                f"GDNAttnBackend: num_k_heads={num_k_heads} must be divisible " f"by TP={tp}."
            )
        if num_v_heads % tp != 0:
            raise ValueError(
                f"GDNAttnBackend: num_v_heads={num_v_heads} must be divisible " f"by TP={tp}."
            )
        if self.conv_dim % tp != 0:
            raise ValueError(
                f"GDNAttnBackend: conv_dim={self.conv_dim} must be divisible "
                f"by TP={tp} for clean per-shard channel slicing."
            )
        if num_v_heads % num_k_heads != 0:
            raise ValueError(
                f"GDNAttnBackend: num_v_heads={num_v_heads} must be a multiple "
                f"of num_k_heads={num_k_heads} (GQA repeat factor)."
            )

        # Depthwise conv1d weight (HF stores [conv_dim, 1, K]; we squeeze).
        # Sharded on the conv_dim axis so each TP rank owns its own channels
        # — consistent with how `RecurrentStatePool` shards conv_state.
        #
        # IMPORTANT — per-shard channel layout is a *loader contract*, not a
        # property of this Param. The shard_map calls below use
        # ``in_specs=P("tensor", None)``, which slices axis 0 into TP
        # contiguous chunks. For the conv1d to line up with ``mixed_qkv``,
        # each rank's local rows must be the per-shard block-concat
        # ``[q_tp | k_tp | v_tp]`` (the same convention
        # :class:`MergedColumnParallelLinear` produces for ``in_proj_qkv``).
        # The HF checkpoint stores conv1d as a single
        # ``[global_q | global_k | global_v]`` block along ``conv_dim`` — a
        # naive ``device_put`` with ``P("tensor", None)`` would give rank 0
        # mostly Q channels and rank N-1 mostly V, which silently mismatches
        # the per-shard activation layout and produces wrong outputs at
        # TP > 1 with no crash.
        #
        # The model loader (CUDA sglang reference:
        # ``mamba_v2_sharded_weight_loader`` in
        # ``sglang/srt/layers/attention/mamba/mamba.py``) must therefore
        # stripe-rearrange the HF tensor so each rank's local rows are
        # ``[Q[rank * key_dim/TP : (rank+1) * key_dim/TP]
        #    | K[rank * key_dim/TP : (rank+1) * key_dim/TP]
        #    | V[rank * value_dim/TP : (rank+1) * value_dim/TP]]``
        # before placement. ``in_proj_qkv`` follows the same convention.
        #
        # A TP > 1 numerical test against an fp32 reference is the canary
        # for getting this wrong — at TP = 1 the two layouts coincide and
        # bugs hide.
        self.conv1d_weight = nnx.Param(jnp.zeros((self.conv_dim, conv_kernel_size), dtype=dtype))
        # Delta-rule params, sharded per-head. Storage dtypes follow the HF
        # Qwen3.5 checkpoint exactly:
        #   A_log:   fp32 (the recurrence's ``-exp(A_log)`` factor is
        #            numerically sensitive — checkpoint is fp32 and the
        #            gating kernel reads it as such).
        #   dt_bias: model dtype (bf16). The gating kernel upcasts to fp32
        #            internally for ``softplus(a + dt_bias)``; storing fp32
        #            here would only force a load-time cast and double the
        #            param footprint with no numerical benefit.
        self.A_log = nnx.Param(jnp.zeros((num_v_heads,), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.ones((num_v_heads,), dtype=dtype))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: jax.Array,  # [T, conv_dim]                  (None, "tensor")
        b: jax.Array,  # [T, n_v]                       (None, "tensor")
        a: jax.Array,  # [T, n_v]                       (None, "tensor")
        recurrent_state_pool,
        layer_id: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Dispatch by ``forward_batch.forward_mode``.

        Fetches per-layer ``(recurrent_state, conv_state)`` from the pool
        via the base class's :meth:`get_layer_cache`. ``conv_state`` is the
        first (only) entry of the per-layer conv-state list — GDN uses a
        single fused conv1d, so it needs exactly one conv buffer per layer
        (vs. KDA, which keeps q/k/v conv states as three list entries).

        Returns ``(core_attn_out, new_conv_state, new_rec_state)`` where
        ``new_conv_state`` and ``new_rec_state`` are the full pool tables
        with this layer's per-request slots updated (scatter happens
        inside the kernel — see :func:`ragged_gated_delta_rule_ref` /
        :func:`jax_causal_conv1d_prefill` and the decode-path equivalents).
        Caller writes these back onto the pool (e.g. via
        ``RecurrentStatePool.replace_buffer``).
        """
        recurrent_state, conv_states = self.get_layer_cache(recurrent_state_pool, layer_id)
        conv_state = conv_states[0]

        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(mixed_qkv, conv_state, recurrent_state, b, a)
        return self.forward_extend(mixed_qkv, conv_state, recurrent_state, b, a)

    # ------------------------------------------------------------------
    # Decode fast path
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        mixed_qkv: jax.Array,
        conv_state_in: jax.Array,
        recurrent_state_in: jax.Array,
        b: jax.Array,
        a: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """One token per request — single conv1d update + parallel single
        recurrence step across the batch, all inside a shard_map."""
        state_indices = self.forward_metadata.recurrent_indices
        tp = _mesh_tp_size(self.mesh)
        n_kq_tp = self.num_k_heads // tp
        n_v_tp = self.num_v_heads // tp
        d_k = self.head_k_dim
        d_v = self.head_v_dim

        def _decode_local(
            mixed_qkv_l,
            conv_state_l,
            rec_state_l,
            conv_weight_l,
            A_log_l,
            dt_bias_l,
            b_l,
            a_l,
            state_indices_l,
        ):
            conv_out, new_conv = jax_causal_conv1d_update(
                mixed_qkv_l,
                conv_state_l,
                state_indices_l,
                conv_weight_l,
                bias=None,
                activation="silu",
            )
            new_rec, out = decode_gated_delta_rule_ref(
                conv_out,
                b_l,
                a_l,
                rec_state_l,
                A_log_l,
                dt_bias_l,
                state_indices_l,
                n_kq=n_kq_tp,
                n_v=n_v_tp,
                d_k=d_k,
                d_v=d_v,
            )
            return out, new_conv, new_rec

        return jax.shard_map(
            _decode_local,
            mesh=self.mesh,
            in_specs=(
                P(None, "tensor"),  # mixed_qkv
                P(None, "tensor", None),  # conv_state
                P(None, "tensor", None, None),  # recurrent_state
                P("tensor", None),  # conv1d weight
                P("tensor"),  # A_log
                P("tensor"),  # dt_bias
                P(None, "tensor"),  # b
                P(None, "tensor"),  # a
                P(),  # state_indices (replicated)
            ),
            out_specs=(
                P(None, "tensor", None),  # out [B, n_v, d_v]
                P(None, "tensor", None),  # new_conv_state [num_blocks, conv_dim, K-1]
                P(None, "tensor", None, None),  # new_rec_state [num_blocks, n_v, d_k, d_v]
            ),
            check_vma=False,
        )(
            mixed_qkv,
            conv_state_in,
            recurrent_state_in,
            self.conv1d_weight.value,
            self.A_log.value,
            self.dt_bias.value,
            b,
            a,
            state_indices,
        )

    # ------------------------------------------------------------------
    # Extend / chunked-prefill
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        mixed_qkv: jax.Array,
        conv_state_in: jax.Array,
        recurrent_state_in: jax.Array,
        b: jax.Array,
        a: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Packed ragged batch through ``ragged_gated_delta_rule_ref``."""
        meta = self.forward_metadata
        cu_seqlens = meta.cu_q_lens
        state_indices = meta.recurrent_indices
        has_initial_state = meta.has_initial_state
        tp = _mesh_tp_size(self.mesh)
        n_kq_tp = self.num_k_heads // tp
        n_v_tp = self.num_v_heads // tp
        d_k = self.head_k_dim
        d_v = self.head_v_dim

        def _extend_local(
            mixed_qkv_l,
            conv_state_l,
            rec_state_l,
            conv_weight_l,
            A_log_l,
            dt_bias_l,
            b_l,
            a_l,
            cu_seqlens_l,
            state_indices_l,
            has_initial_state_l,
        ):
            # jax_causal_conv1d_prefill operates on [D, T] (channel-first).
            # Pass `has_initial_state` so brand-new prefills don't pick up
            # stale conv state from a freshly-allocated slot (same mask
            # contract as `ragged_gated_delta_rule_ref`).
            conv_out_dt, new_conv = jax_causal_conv1d_prefill(
                x=mixed_qkv_l.T,
                weight=conv_weight_l,
                bias=None,
                cu_seqlens=cu_seqlens_l,
                conv_state=conv_state_l,
                state_indices=state_indices_l,
                has_initial_state=has_initial_state_l,
                activation="silu",
            )
            conv_out = conv_out_dt.T  # [T, D]
            new_rec, out = ragged_gated_delta_rule_ref(
                conv_out,
                b_l,
                a_l,
                rec_state_l,
                A_log_l,
                dt_bias_l,
                cu_seqlens=cu_seqlens_l,
                state_indices=state_indices_l,
                has_initial_state=has_initial_state_l,
                n_kq=n_kq_tp,
                n_v=n_v_tp,
                d_k=d_k,
                d_v=d_v,
            )
            return out, new_conv, new_rec

        return jax.shard_map(
            _extend_local,
            mesh=self.mesh,
            in_specs=(
                P(None, "tensor"),  # mixed_qkv
                P(None, "tensor", None),  # conv_state
                P(None, "tensor", None, None),  # recurrent_state
                P("tensor", None),  # conv1d weight
                P("tensor"),  # A_log
                P("tensor"),  # dt_bias
                P(None, "tensor"),  # b
                P(None, "tensor"),  # a
                P(),  # cu_seqlens (replicated)
                P(),  # state_indices (replicated)
                P(),  # has_initial_state (replicated)
            ),
            out_specs=(
                P(None, "tensor", None),  # out [T, n_v, d_v]
                P(None, "tensor", None),  # new_conv_state [num_blocks, conv_dim, K-1]
                P(None, "tensor", None, None),  # new_rec_state [num_blocks, n_v, d_k, d_v]
            ),
            check_vma=False,
        )(
            mixed_qkv,
            conv_state_in,
            recurrent_state_in,
            self.conv1d_weight.value,
            self.A_log.value,
            self.dt_bias.value,
            b,
            a,
            cu_seqlens,
            state_indices,
            has_initial_state,
        )
