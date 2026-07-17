"""Gated-DeltaNet attention backend.

Inherits :class:`LinearRecurrentAttnBackend` for shared metadata
(``cu_q_lens`` / ``recurrent_indices`` / ``has_initial_state``) and
pytree boilerplate. **Stateless** — owns no weights. The parent
``RadixLinearAttention`` carries the fused ``conv1d`` weight container
plus the ``A_log`` / ``dt_bias`` recurrence params; this backend reads
them off ``layer.*`` at call time.

The parent hands in already-sliced ``q`` / ``k`` / ``v`` (the slice
happens in the model layer). The backend re-concatenates them along
the channel axis to feed the depthwise
``conv1d``; XLA collapses the slice→concat into a single slice of
the upstream ``in_proj_qkvz`` activation, so this rebuild adds no
HBM traffic.

Sharding pattern: the conv + recurrence pipeline runs inside
:func:`jax.shard_map` with explicit ``in_specs`` / ``out_specs``, with
the head axis pinned to ``"tensor"`` so each device sees only its local
shard. The kernels then operate on per-shard head counts
(``n_kq // TP``, ``n_v // TP``) without relying on JAX sharding
inference. Returns ``(core_attn_out, new_conv, new_rec)`` shaped for
``RecurrentStatePool.write_layer``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.gdn import (
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_chunkwise,
    ragged_gated_delta_rule_ref,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)


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

    Stateless (weights on the parent :class:`RadixLinearAttention`);
    dispatches conv1d + ragged delta-rule (extend) or single-step delta-rule
    (decode) under ``jax.shard_map``. Reads ``cu_q_lens`` /
    ``recurrent_indices`` / ``has_initial_state`` off ``self.forward_metadata``.
    """

    def __init__(
        self,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        mesh: jax.sharding.Mesh,
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

        self.requested_impl: str = os.environ.get("SGLANG_JAX_GDN_PREFILL_IMPL", "chunkwise")
        if self.requested_impl not in {"chunkwise", "reference"}:
            raise ValueError(
                "SGLANG_JAX_GDN_PREFILL_IMPL must be one of 'chunkwise' or "
                f"'reference', got {self.requested_impl!r}"
            )

        self.effective_impl: str = self.requested_impl
        self.fallback_reason: str | None = None
        self._prefill_callable: Callable = ragged_gated_delta_rule_ref
        if self.requested_impl == "chunkwise":
            if self.head_k_dim > 256:
                self.effective_impl = "reference"
                self.fallback_reason = (
                    f"head_k_dim={self.head_k_dim} exceeds chunkwise limit=256"
                )
            else:
                platforms = {device.platform.lower() for device in mesh.devices.flat}
                all_tpu = platforms == {"tpu"}
                all_non_tpu = "tpu" not in platforms
                interpret_enabled = os.environ.get("PALLAS_INTERPRET", "").lower() == "true"
                if all_tpu or (all_non_tpu and interpret_enabled):
                    self._prefill_callable = ragged_gated_delta_rule_chunkwise
                else:
                    self.effective_impl = "reference"
                    self.fallback_reason = (
                        "chunkwise prefill unsupported on platform=" + ",".join(sorted(platforms))
                    )

        logger.info(
            "GDN prefill implementation requested_impl=%s effective_impl=%s fallback_reason=%s",
            self.requested_impl,
            self.effective_impl,
            self.fallback_reason,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def __call__(
        self,
        q: jax.Array,  # [T, key_dim]   sliced from in_proj_qkvz upstream
        k: jax.Array,  # [T, key_dim]
        v: jax.Array,  # [T, value_dim]
        a: jax.Array,  # [T, num_v_heads]
        b: jax.Array,  # [T, num_v_heads]
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        recurrent_state_pool: RecurrentStatePool,
        **kwargs,  # absorb mixed_qkv=None forwarded by HybridLinearAttnBackend
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """Dispatch by ``forward_batch.forward_mode``.

        Rebuilds ``mixed_qkv`` as the per-rank ``[q_d | k_d | v_d]`` block so
        each TP shard sees its own head-striped channels (matching the rank-major
        conv1d weight stripe). A plain ``concatenate([q,k,v])`` is WRONG at TP>1:
        q/k/v have unequal per-device widths, so concatenating along the sharded
        axis gives the logical ``[Q|K|V]`` layout and scrambles the conv channels.
        Reshaping to ``[T, tp, *]`` rank-blocks first preserves the stripe (and
        collapses to plain ``[Q|K|V]`` at tp=1).

        ``conv_state`` is the single fused-conv1d buffer (GDN keeps one per
        layer, vs. KDA's three q/k/v entries). Returns ``(core_attn_out,
        (new_rec_state, [new_conv_state]))`` per the linear-backend contract.
        """
        tp = _mesh_tp_size(self.mesh)
        T = q.shape[0]
        k_tp = self.key_dim // tp
        v_tp = self.value_dim // tp
        # Per-rank head-blocks -> concat -> flatten: device d gets [q_d|k_d|v_d].
        mixed_qkv = jnp.concatenate(
            [q.reshape(T, tp, k_tp), k.reshape(T, tp, k_tp), v.reshape(T, tp, v_tp)],
            axis=-1,
        ).reshape(T, self.conv_dim)
        mixed_qkv = jax.sharding.reshard(mixed_qkv, P("data", "tensor"))

        conv1d_weight = layer.conv1d.weight.value
        A_log = layer.A_log.value
        dt_bias = layer.dt_bias.value

        recurrent_state, conv_states = self.get_layer_cache(recurrent_state_pool, layer.layer_id)
        conv_state = conv_states[0]

        if forward_batch.forward_mode.is_decode():
            out, new_conv, new_rec = self.forward_decode(
                mixed_qkv,
                conv_state,
                recurrent_state,
                b,
                a,
                conv1d_weight,
                A_log,
                dt_bias,
            )
        else:
            out, new_conv, new_rec = self.forward_extend(
                mixed_qkv,
                conv_state,
                recurrent_state,
                b,
                a,
                conv1d_weight,
                A_log,
                dt_bias,
            )
        # Flatten head dim into channel dim to match KDA's contract
        # (model layer reshapes back to [T, n_v, d_v] before output norm).
        out = out.reshape(out.shape[0], -1)
        return out, (new_rec, [new_conv])

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
        conv1d_weight: jax.Array,
        A_log: jax.Array,
        dt_bias: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """One token per request — single conv1d update + parallel single
        recurrence step across the batch, all inside a shard_map."""
        state_indices = self.forward_metadata.recurrent_indices
        has_initial_state = self.forward_metadata.has_initial_state
        track_indices = self.forward_metadata.recurrent_track_indices
        track_mask = self.forward_metadata.recurrent_track_mask
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
            has_initial_state_l,
            track_indices_l=None,
            track_mask_l=None,
        ):
            conv_out, new_conv = jax_causal_conv1d_update(
                mixed_qkv_l,
                conv_state_l,
                state_indices_l,
                conv_weight_l,
                bias=None,
                activation="silu",
                has_initial_state=has_initial_state_l,
                track_indices=track_indices_l,
                track_mask=track_mask_l,
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
                has_initial_state=has_initial_state_l,
                track_indices=track_indices_l,
                track_mask=track_mask_l,
            )
            return out, new_conv, new_rec

        in_specs = [
            P("data", "tensor"),  # mixed_qkv
            P("data", "tensor", None),  # conv_state
            P("data", "tensor", None, None),  # recurrent_state
            P("tensor", None),  # conv1d weight
            P("tensor"),  # A_log
            P("tensor"),  # dt_bias
            P("data", "tensor"),  # b
            P("data", "tensor"),  # a
            P("data"),  # state_indices (sharded by data — one slice per DP rank)
            P("data"),  # has_initial_state (sharded by data)
        ]
        args = [
            mixed_qkv,
            conv_state_in,
            recurrent_state_in,
            conv1d_weight,
            A_log,
            dt_bias,
            b,
            a,
            state_indices,
            has_initial_state,
        ]
        # OFF/no-boundary: track_indices is None -> keep the original shard_map
        # call (no track args), byte-identical to the no-track path. When a
        # boundary lands in this batch, add two P("data") track slices.
        if track_indices is not None:
            in_specs += [P("data"), P("data")]  # track_indices, track_mask
            args += [track_indices, track_mask]

        return jax.shard_map(
            _decode_local,
            mesh=self.mesh,
            in_specs=tuple(in_specs),
            out_specs=(
                P("data", "tensor", None),  # out [B, n_v, d_v]
                P("data", "tensor", None),  # new_conv_state [num_blocks, conv_dim, K-1]
                P("data", "tensor", None, None),  # new_rec_state [num_blocks, n_v, d_k, d_v]
            ),
            check_vma=False,
        )(*args)

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
        conv1d_weight: jax.Array,
        A_log: jax.Array,
        dt_bias: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Packed ragged batch through the initialization-frozen prefill callable."""
        meta = self.forward_metadata
        cu_seqlens = meta.cu_q_lens
        state_indices = meta.recurrent_indices
        has_initial_state = meta.has_initial_state
        track_indices = meta.recurrent_track_indices
        track_mask = meta.recurrent_track_mask
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
            track_indices_l=None,
            track_mask_l=None,
        ):
            if track_indices_l is not None:
                nonempty = cu_seqlens_l[1:] > cu_seqlens_l[:-1]
                track_mask_l = track_mask_l & nonempty & (state_indices_l != 0)
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
                track_indices=track_indices_l,
                track_mask=track_mask_l,
            )
            conv_out = conv_out_dt.T  # [T, D]
            new_rec, out = self._prefill_callable(
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
                track_indices=track_indices_l,
                track_mask=track_mask_l,
            )
            return out, new_conv, new_rec

        in_specs = [
            P("data", "tensor"),  # mixed_qkv
            P("data", "tensor", None),  # conv_state
            P("data", "tensor", None, None),  # recurrent_state
            P("tensor", None),  # conv1d weight
            P("tensor"),  # A_log
            P("tensor"),  # dt_bias
            P("data", "tensor"),  # b
            P("data", "tensor"),  # a
            P("data"),  # cu_seqlens (sharded by data — one slice per DP rank)
            P("data"),  # state_indices (sharded by data)
            P("data"),  # has_initial_state (sharded by data)
        ]
        args = [
            mixed_qkv,
            conv_state_in,
            recurrent_state_in,
            conv1d_weight,
            A_log,
            dt_bias,
            b,
            a,
            cu_seqlens,
            state_indices,
            has_initial_state,
        ]
        if track_indices is not None:
            in_specs += [P("data"), P("data")]  # track_indices, track_mask
            args += [track_indices, track_mask]

        return jax.shard_map(
            _extend_local,
            mesh=self.mesh,
            in_specs=tuple(in_specs),
            out_specs=(
                P("data", "tensor", None),  # out [T, n_v, d_v]
                P("data", "tensor", None),  # new_conv_state [num_blocks, conv_dim, K-1]
                P("data", "tensor", None, None),  # new_rec_state [num_blocks, n_v, d_k, d_v]
            ),
            check_vma=False,
        )(*args)
