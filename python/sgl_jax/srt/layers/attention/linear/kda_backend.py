from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.kda import chunk_kda, fused_recurrent_kda
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.layers.attention.linear.short_convolution import (
    l2_normalize,
    short_convolution,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class KDAAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, mesh: jax.sharding.Mesh = None, use_pallas_prefill: bool = False):
        super().__init__(
            mesh=mesh,
        )
        self.use_pallas_prefill = use_pallas_prefill

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
        q_state, k_state, v_state = self._unpack_conv_states(conv_states)

        q_conv_w, k_conv_w, v_conv_w = layer.conv_weights
        # LinearBase stores weights as [in_features=K, out_features=D]; the
        # short conv kernel expects depthwise layout [D, K], so transpose.
        q_conv_w = self._to_depthwise_layout(q_conv_w)
        k_conv_w = self._to_depthwise_layout(k_conv_w)
        v_conv_w = self._to_depthwise_layout(v_conv_w)
        cu_q_lens = self.forward_metadata.cu_q_lens
        q, q_state_new = short_convolution(
            q, q_conv_w, q_state, cu_q_lens, forward_batch.forward_mode
        )
        k, k_state_new = short_convolution(
            k, k_conv_w, k_state, cu_q_lens, forward_batch.forward_mode
        )
        v, v_state_new = short_convolution(
            v, v_conv_w, v_state, cu_q_lens, forward_batch.forward_mode
        )
        new_conv_states = jnp.stack([q_state_new, k_state_new, v_state_new], axis=1)

        q = q.reshape(q.shape[0], layer.num_q_heads, layer.head_q_dim)
        k = k.reshape(k.shape[0], layer.num_k_heads, layer.head_k_dim)
        v = v.reshape(v.shape[0], layer.num_v_heads, layer.head_v_dim)
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
            )
        else:
            raise NotImplementedError(f"KDA does not support {forward_batch.forward_mode}")

        new_ssm_states = self.set_ssm_state(ssm_states, recurrent_indices, new_recurrent)
        new_conv_states = self.set_conv_state(conv_states, recurrent_indices, new_conv_states)
        return output.reshape(output.shape[0], -1), (new_ssm_states, new_conv_states)

    def get_state(self, recurrent_state_pool, layer_id, recurrent_indices):
        recurrent_buffers, conv_buffers = self.get_layer_cache(
            recurrent_state_pool,
            layer_id,
        )
        return recurrent_buffers[recurrent_indices], conv_buffers[recurrent_indices]

    def set_conv_state(self, conv_buffers, recurrent_indices, new_conv_states):
        return conv_buffers.at[recurrent_indices].set(new_conv_states)

    def set_ssm_state(self, recurrent_buffers, recurrent_indices, new_ssm_states):
        return recurrent_buffers.at[recurrent_indices].set(new_ssm_states)

    # ------------------------------------------------------------------
    # Forward mode implementations
    # ------------------------------------------------------------------

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
    ) -> tuple[jax.Array, jax.Array]:
        if self.use_pallas_prefill:
            return self._forward_extend_pallas(q, k, v, g, beta, initial_state, cu_seqlens, layer)
        return self._forward_extend_naive(q, k, v, g, beta, initial_state, cu_seqlens, layer)

    def _forward_extend_pallas(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        cu_seqlens: jax.Array,
        layer: RadixLinearAttention,
    ) -> tuple[jax.Array, jax.Array]:
        """Chunked prefill via Pallas kernel.  Returns (output, new_state)."""
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        # Kernel expects [1, T_packed, H, K] packed layout.
        o, final_state, *_ = chunk_kda(
            q[None, ...],
            k[None, ...],
            v[None, ...],
            g[None, ...],
            beta[None, ...],
            scale=self._get_scale(layer),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=True,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
        )
        # Remove the B=1 packed dim: [1, T, H, V] -> [T, H, V]
        return o[0], final_state

    def _forward_extend_naive(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        cu_seqlens: jax.Array,
        layer: RadixLinearAttention,
    ) -> tuple[jax.Array, jax.Array]:
        """Prefill fallback via naive recurrent kernel."""
        g = self._activate_gate(layer, g)
        if cu_seqlens.shape[0] == 2:
            o, final_state = fused_recurrent_kda(
                q[None, ...],
                k[None, ...],
                v[None, ...],
                g[None, ...],
                beta[None, ...],
                scale=self._get_scale(layer),
                initial_state=initial_state,
                output_final_state=True,
            )
            return o[0], final_state

        q_b, k_b, v_b, g_b, beta_b = self._unpack_varlen(q, k, v, g, beta, cu_seqlens)
        o_b, final_state = fused_recurrent_kda(
            q_b,
            k_b,
            v_b,
            g_b,
            beta_b,
            scale=self._get_scale(layer),
            initial_state=initial_state,
            output_final_state=True,
        )
        return self._repack_varlen(o_b, cu_seqlens, q.shape[0]), final_state

    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        initial_state: jax.Array,
        layer: RadixLinearAttention,
    ) -> tuple[jax.Array, jax.Array]:
        """Single-step decode via naive JAX recurrence.  Returns (output, new_state)."""
        g = self._activate_gate(layer, g)
        # Kernel expects [B, T=1, H, K].
        o, final_state = fused_recurrent_kda(
            q[:, None, ...],
            k[:, None, ...],
            v[:, None, ...],
            g[:, None, ...],
            beta[:, None, ...],
            scale=self._get_scale(layer),
            initial_state=initial_state,
            output_final_state=True,
        )
        # Remove the T=1 dim: [B, 1, H, V] -> [B, H, V]
        return o[:, 0], final_state

    def _activate_gate(
        self,
        layer: RadixLinearAttention,
        g: jax.Array,
    ) -> jax.Array:
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = g.shape[-2]
        orig_dtype = g.dtype
        g32 = g.astype(jnp.float32) + layer.dt_bias.reshape(H, -1).astype(jnp.float32)
        out = -jnp.exp(layer.A_log.reshape(H, 1).astype(jnp.float32)) * jax.nn.softplus(g32)
        return out.astype(orig_dtype)

    @staticmethod
    def _unpack_varlen(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        cu_seqlens: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        T = q.shape[0]
        positions = jnp.arange(T, dtype=cu_seqlens.dtype)
        starts = cu_seqlens[:-1]
        lens = cu_seqlens[1:] - starts
        token_idx = starts[:, None] + positions[None, :]
        valid = positions[None, :] < lens[:, None]
        safe_idx = jnp.where(valid, token_idx, 0)

        def gather(x):
            padded = x[safe_idx]
            return jnp.where(valid[(...,) + (None,) * (x.ndim - 1)], padded, 0)

        beta_b = beta[safe_idx]
        beta_b = jnp.where(valid[..., None], beta_b, 0)
        return gather(q), gather(k), gather(v), gather(g), beta_b

    @staticmethod
    def _repack_varlen(
        output: jax.Array,
        cu_seqlens: jax.Array,
        total_tokens: int,
    ) -> jax.Array:
        token_idx = jnp.arange(total_tokens, dtype=cu_seqlens.dtype)
        seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
        seq_offsets = token_idx - cu_seqlens[:-1][seq_ids]
        return output[seq_ids, seq_offsets]

    @staticmethod
    def _unpack_conv_states(
        conv_states,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Pull per-stream conv caches out of the pool layout.

        Accepts either a ``(qs, ks, vs)`` tuple or a stacked ``[B, 3, C, K]``
        array (the layout written back by ``__call__``).
        """
        if isinstance(conv_states, (tuple, list)) and len(conv_states) == 3:
            return tuple(conv_states)
        # Expect [B, 3, C, K] stacked layout.
        return conv_states[:, 0], conv_states[:, 1], conv_states[:, 2]

    @staticmethod
    def _to_depthwise_layout(weight: jax.Array) -> jax.Array:
        """Coerce a conv weight to the depthwise [D, K] layout.

        ``LinearBase`` stores weights as ``[in_features, out_features]``; for
        the conv1d projections that means ``[K, D]`` (kernel size, channels).
        ``short_convolution`` expects channels-first ``[D, K]``, so we
        transpose. The ``[D, 1, K]`` PyTorch layout is passed through —
        ``short_convolution`` squeezes the singleton axis itself.
        """
        if weight.ndim == 3 and weight.shape[1] == 1:
            return weight
        if weight.ndim != 2:
            raise ValueError(
                f"conv weight must be rank 2 or [D, 1, K]; got shape {weight.shape}"
            )
        return jnp.swapaxes(weight, 0, 1)


__all__ = ["KDAAttnBackend"]
