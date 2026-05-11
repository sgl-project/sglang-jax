from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.kda import chunk_kda, naive_recurrent_kda
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.layers.attention.linear.short_convolution import short_convolution
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


def l2_normalize(x: jax.Array, epsilon: float = 1e-6) -> jax.Array:
    """L2-normalize along the last axis. Computed in fp32, cast back to input dtype."""
    norm = jnp.linalg.norm(x.astype(jnp.float32), axis=-1, keepdims=True)
    return (x.astype(jnp.float32) / jnp.maximum(norm, epsilon)).astype(x.dtype)


class KDAAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, mesh: jax.sharding.Mesh = None):
        super().__init__(
            mesh=mesh,
        )

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        a: jax.Array,
        b: jax.Array,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
        **kwargs,
    ) -> jax.Array:
        recurrent_indices = self.forward_metadata.recurrent_indices
        ssm_states, conv_states = self.get_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices
        )
        q_conv_w = layer.q_conv1d.weight.value
        k_conv_w = layer.k_conv1d.weight.value
        v_conv_w = layer.v_conv1d.weight.value
        # _unpack_conv_states splits proj_size in 3 equal pieces; only valid
        # when proj_q == proj_k == proj_v (Kimi-Linear shape).
        assert (
            q_conv_w.shape[0] == k_conv_w.shape[0] == v_conv_w.shape[0]
        ), f"unequal Q/K/V proj widths: {q_conv_w.shape[0]}/{k_conv_w.shape[0]}/{v_conv_w.shape[0]}"
        q_state, k_state, v_state = self._unpack_conv_states(conv_states)

        cu_q_lens = self.forward_metadata.cu_q_lens
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            q, q_state_new = self._short_conv_extend(
                q, q_conv_w, q_state, cu_q_lens, layer.activation
            )
            k, k_state_new = self._short_conv_extend(
                k, k_conv_w, k_state, cu_q_lens, layer.activation
            )
            v, v_state_new = self._short_conv_extend(
                v, v_conv_w, v_state, cu_q_lens, layer.activation
            )
        else:
            q, q_state_new = short_convolution(
                q,
                q_conv_w,
                q_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
            k, k_state_new = short_convolution(
                k,
                k_conv_w,
                k_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
            v, v_state_new = short_convolution(
                v,
                v_conv_w,
                v_state,
                cu_q_lens,
                forward_batch.forward_mode,
                activation=layer.activation,
            )
        new_conv_packed = self._pack_conv_states(q_state_new, k_state_new, v_state_new)

        q = q.reshape(q.shape[0], layer.num_q_heads, layer.head_q_dim)
        k = k.reshape(k.shape[0], layer.num_k_heads, layer.head_k_dim)
        v = v.reshape(v.shape[0], layer.num_v_heads, layer.head_v_dim)
        a = a.reshape(a.shape[0], layer.num_q_heads, layer.head_q_dim)
        b = b.reshape(b.shape[0], layer.num_q_heads)

        # KDA requires L2-normalized q/k for all paths; upstream fuses this
        # via use_qk_l2norm_in_kernel=True, while current kernel doesn't support.
        q = l2_normalize(q)
        k = l2_normalize(k)

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_recurrent = self._forward_extend(
                q,
                k,
                v,
                a,
                b,
                ssm_states,
                cu_q_lens,
                layer,
                scale=layer.scale,
            )
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            output, new_recurrent = self._forward_decode(
                q,
                k,
                v,
                a,
                b,
                ssm_states,
                layer,
                scale=layer.scale,
            )
        else:
            raise NotImplementedError(f"KDA does not support {forward_batch.forward_mode}")

        new_ssm_full = self.set_ssm_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, new_recurrent
        )
        new_conv_full_list = self.set_conv_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, new_conv_packed
        )
        return output.reshape(output.shape[0], -1), (new_ssm_full, new_conv_full_list)

    def get_state(self, recurrent_state_pool, layer_id, recurrent_indices):
        """Return per-request views of (ssm, conv) state for this layer."""
        recurrent_buffer, conv_buffer_list = self.get_layer_cache(
            recurrent_state_pool,
            layer_id,
        )
        assert len(conv_buffer_list) == 1, (
            f"KDA expects exactly 1 conv buffer per layer "
            f"(reserved RecurrentStatePool inner-list length); got {len(conv_buffer_list)}"
        )
        conv_buffer = conv_buffer_list[0]

        ssm = jax.shard_map(
            lambda buf, idx: buf[idx],
            mesh=self.mesh,
            in_specs=(P("data", "tensor", None, None), P("data")),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(recurrent_buffer, recurrent_indices)

        conv = jax.shard_map(
            lambda buf, idx: buf[idx],
            mesh=self.mesh,
            in_specs=(P("data", "tensor", None), P("data")),
            out_specs=P("data", "tensor", None),
            check_vma=False,
        )(conv_buffer, recurrent_indices)

        has_initial_state = self.forward_metadata.has_initial_state
        if has_initial_state is not None:
            ssm = jnp.where(has_initial_state[:, None, None, None], ssm, 0.0)
            conv = jnp.where(has_initial_state[:, None, None], conv, 0.0)

        return ssm, conv

    def set_ssm_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_recurrent):
        """Scatter per-request ``new_recurrent`` into the FULL pool buffer."""
        full_recurrent, _ = self.get_layer_cache(recurrent_state_pool, layer_id)

        return jax.shard_map(
            lambda buf, idx, val: buf.at[idx].set(val),
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None, None),
                P("data"),
                P("data", "tensor", None, None),
            ),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(full_recurrent, recurrent_indices, new_recurrent)

    def set_conv_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_conv_packed):
        """Scatter per-request packed conv state into the FULL pool buffer."""
        _, conv_buffer_list = self.get_layer_cache(recurrent_state_pool, layer_id)
        assert len(conv_buffer_list) == 1
        full_conv = conv_buffer_list[0]

        new_conv_full = jax.shard_map(
            lambda buf, idx, val: buf.at[idx].set(val),
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),
                P("data"),
                P("data", "tensor", None),
            ),
            out_specs=P("data", "tensor", None),
            check_vma=False,
        )(full_conv, recurrent_indices, new_conv_packed)
        return [new_conv_full]

    # ------------------------------------------------------------------
    # Forward mode implementations
    # ------------------------------------------------------------------

    def _short_conv_extend(
        self,
        x: jax.Array,
        weight: jax.Array,
        cache: jax.Array,
        cu_seqlens: jax.Array,
        activation,
    ) -> tuple[jax.Array, jax.Array]:
        """EXTEND-path conv wrapped in shard_map."""

        def _call(x, weight, cache, cu_seqlens):
            return short_convolution(
                x, weight, cache, cu_seqlens, ForwardMode.EXTEND, activation=activation
            )

        sharded = jax.shard_map(
            _call,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor"),  # x [T, hidden]
                P("tensor", None),  # weight [hidden, K]
                P("data", "tensor", None),  # cache [B, hidden, K-1]
                P("data"),  # cu_seqlens [B+1]
            ),
            out_specs=(
                P("data", "tensor"),  # y [T, hidden]
                P("data", "tensor", None),  # new_cache [B, hidden, K-1]
            ),
            check_vma=False,
        )
        return sharded(x, weight, cache, cu_seqlens)

    def _forward_extend(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        cu_seqlens: jax.Array,
        layer: RadixLinearAttention,
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Chunked prefill via Pallas kernel."""
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = q.shape[-2]
        # kda_gate_chunk_cumsum requires A_log shape (H,); layer stores [1,1,H,1].
        A_log = layer.A_log.value.reshape(H)
        dt_bias = layer.dt_bias.value.reshape(H, -1)
        scale = scale if scale is not None else layer.scale

        def _chunk_kda_call(q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias):
            o, final_state, *_ = chunk_kda(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
            )
            return o, final_state

        sharded = jax.shard_map(
            _chunk_kda_call,
            mesh=self.mesh,
            in_specs=(
                P(None, "data", "tensor", None),  # q [1, T, H, K]
                P(None, "data", "tensor", None),  # k [1, T, H, K]
                P(None, "data", "tensor", None),  # v [1, T, H, V]
                P(None, "data", "tensor", None),  # g [1, T, H, K]
                P(None, "data", "tensor"),  # beta [1, T, H]
                P("data", "tensor", None, None),  # initial_state [N, H, K, V]
                P("data"),  # cu_seqlens [N+1]
                P("tensor"),  # A_log [H]
                P("tensor", None),  # dt_bias [H, K]
            ),
            out_specs=(
                P(None, "data", "tensor", None),  # output [1, T, H, V]
                P("data", "tensor", None, None),  # final_state [N, H, K, V]
            ),
            check_vma=False,
        )
        # Kernel expects [1, T_packed, H, K] packed layout; strip after.
        o, final_state = sharded(
            q[None, ...],
            k[None, ...],
            v[None, ...],
            g[None, ...],
            beta[None, ...],
            initial_state,
            cu_seqlens,
            A_log,
            dt_bias,
        )
        return o[0], final_state

    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        layer: RadixLinearAttention,
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Single-step decode via naive JAX recurrence (Pallas decode TBD)."""
        g = self._fused_kda_gate(layer, g)

        def _decode_kernel(q_d, k_d, v_d, g_d, beta_d, h0):
            o, final_state = naive_recurrent_kda(
                q_d[:, None, ...],
                k_d[:, None, ...],
                v_d[:, None, ...],
                g_d[:, None, ...],
                beta_d[:, None, ...],
                scale=scale,
                initial_state=h0,
                output_final_state=True,
            )
            return o[:, 0], final_state

        sharded = jax.shard_map(
            _decode_kernel,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),  # q [B, H, K]
                P("data", "tensor", None),  # k [B, H, K]
                P("data", "tensor", None),  # v [B, H, V]
                P("data", "tensor", None),  # g [B, H, K]
                P("data", "tensor"),  # beta [B, H]
                P("data", "tensor", None, None),  # h0 [B, H, K, V]
            ),
            out_specs=(
                P("data", "tensor", None),  # o [B, H, V]
                P("data", "tensor", None, None),  # final_state [B, H, K, V]
            ),
            check_vma=False,
        )
        return sharded(q, k, v, g, beta, initial_state)

    def _fused_kda_gate(
        self,
        layer: RadixLinearAttention,
        g: jax.Array,
    ) -> jax.Array:
        """JAX-side gate activation used by the DECODE path."""
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = g.shape[-2]
        orig_dtype = g.dtype
        g32 = g.astype(jnp.float32) + layer.dt_bias.value.reshape(H, -1).astype(jnp.float32)
        out = -jnp.exp(layer.A_log.value.reshape(H, 1).astype(jnp.float32)) * jax.nn.softplus(g32)
        return out.astype(orig_dtype)

    def _unpack_conv_states(
        self,
        conv_states: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Slice ``[B, proj_size, K-1]`` into per-stream Q/K/V caches."""
        D = conv_states.shape[1]
        assert D % 3 == 0, f"conv_states channel dim {D} must be divisible by 3"
        q, k, v = jnp.split(conv_states, 3, axis=1)
        return q, k, v

    def _pack_conv_states(
        self,
        q_state: jax.Array,
        k_state: jax.Array,
        v_state: jax.Array,
    ) -> jax.Array:
        """Concat per-stream ``[B, D, K-1]`` caches → packed ``[B, proj_size, K-1]``."""
        return jnp.concatenate([q_state, k_state, v_state], axis=1)


__all__ = ["KDAAttnBackend"]
