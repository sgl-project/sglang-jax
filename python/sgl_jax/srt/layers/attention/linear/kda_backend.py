from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

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

    def __init__(self, mesh: jax.sharding.Mesh = None, use_pallas_prefill: bool = True):
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
        # TODO: add v_head_num_per_device, etc to make it more compatible
        q_state, k_state, v_state = self._unpack_conv_states(conv_states)

        # Conv weights are stored directly in depthwise ``[D, K]`` layout —
        # the format ``short_convolution`` consumes — so no reshape needed.
        # Reading ``.value`` keeps us tied to the live nnx.Param so checkpoint
        # loads done after backend construction are picked up automatically.
        q_conv_w = layer.q_conv1d.weight.value
        k_conv_w = layer.k_conv1d.weight.value
        v_conv_w = layer.v_conv1d.weight.value
        cu_q_lens = self.forward_metadata.cu_q_lens
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

        # KDA requires L2-normalized q/k for all paths (decode, naive prefill,
        # Pallas prefill). The official implementation does this inside the
        # kernel via use_qk_l2norm_in_kernel=True; our kernels don't expose
        # that flag yet, so we normalize in JAX up front.
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
        """Return per-request views of (ssm, conv) state for this layer.

        Pool stores conv as ``list[list[jax.Array]]`` (outer per-layer, inner
        currently length 1; reserved for future multi-segment expansion). We
        unwrap that length-1 inner list so the rest of the backend treats
        conv as a single ``[N+1, proj_size, K-1]`` array.
        """
        recurrent_buffer, conv_buffer_list = self.get_layer_cache(
            recurrent_state_pool,
            layer_id,
        )
        assert len(conv_buffer_list) == 1, (
            f"KDA expects exactly 1 conv buffer per layer "
            f"(reserved RecurrentStatePool inner-list length); got {len(conv_buffer_list)}"
        )
        conv_buffer = conv_buffer_list[0]
        return recurrent_buffer[recurrent_indices], conv_buffer[recurrent_indices]

    def set_ssm_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_recurrent):
        """Scatter per-request ``new_recurrent`` into the FULL pool buffer.

        Returns the full ``[N+1, H, D, D]`` recurrent buffer for this layer
        so the caller can bubble it up to ``MemoryPools.replace_all`` for
        rebinding ``RecurrentStatePool.recurrent_buffers[idx]``.
        """
        full_recurrent, _ = self.get_layer_cache(recurrent_state_pool, layer_id)
        return full_recurrent.at[recurrent_indices].set(new_recurrent)

    def set_conv_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_conv_packed):
        """Scatter per-request packed conv state into the FULL pool buffer.

        ``new_conv_packed`` is ``[B, proj_size, K-1]``; the full conv buffer
        is ``[N+1, proj_size, K-1]``. Returns a length-1 list to match the
        pool's per-layer ``list[list[jax.Array]]`` layout.
        """
        _, conv_buffer_list = self.get_layer_cache(recurrent_state_pool, layer_id)
        assert len(conv_buffer_list) == 1
        full_conv = conv_buffer_list[0]
        return [full_conv.at[recurrent_indices].set(new_conv_packed)]

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
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if self.use_pallas_prefill:
            return self._forward_extend_pallas(
                q, k, v, g, beta, initial_state, cu_seqlens, layer, scale=scale
            )
        return self._forward_extend_naive(
            q, k, v, g, beta, initial_state, cu_seqlens, layer, scale=scale
        )

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
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Chunked prefill via Pallas kernel.  Returns (output, new_state).

        chunk_kda calls pallas_call internally, so we wrap it in shard_map
        (rpav3 pattern) — H is sharded on "tensor", cu_seqlens replicated.
        scale is a static argname of chunk_kda and must be a Python float;
        binding it via closure here is fine because jax.jit caches the
        compiled product across calls.
        """
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = q.shape[-2]
        # kda_gate_chunk_cumsum requires A_log shape (H,); the layer stores
        # it as [1, 1, H, 1] (broadcast-friendly for naive paths).
        A_log = layer.A_log.value.reshape(H)
        dt_bias = layer.dt_bias.value
        scale = scale if scale is not None else layer.scale
        g = self._fused_kda_gate(layer, g)

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
                use_gate_in_kernel=False,
            )
            return o, final_state

        sharded = jax.shard_map(
            _chunk_kda_call,
            mesh=self.mesh,
            in_specs=(
                P(None, None, "tensor", None),  # q [1, T, H, K]
                P(None, None, "tensor", None),  # k [1, T, H, K]
                P(None, None, "tensor", None),  # v [1, T, H, V]
                P(None, None, "tensor", None),  # g [1, T, H, K]
                P(None, None, "tensor"),  # beta [1, T, H]
                P(None, "tensor", None, None),  # initial_state [N, H, K, V]
                P(),  # cu_seqlens [N+1]
                P("tensor"),  # A_log [H]
                P("tensor"),  # dt_bias [H*K]
            ),
            out_specs=(
                P(None, None, "tensor", None),  # output [1, T, H, V]
                P(None, "tensor", None, None),  # final_state [N, H, K, V]
            ),
            check_vma=False,
        )
        # Kernel expects [1, T_packed, H, K] packed layout.
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
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Prefill fallback via naive recurrent kernel."""
        g = self._fused_kda_gate(layer, g)
        if cu_seqlens.shape[0] == 2:
            o, final_state = fused_recurrent_kda(
                q[None, ...],
                k[None, ...],
                v[None, ...],
                g[None, ...],
                beta[None, ...],
                scale=scale,
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
            scale=scale,
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
        scale: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Single-step decode via naive JAX recurrence.  Returns (output, new_state)."""
        g = self._fused_kda_gate(layer, g)
        # Kernel expects [B, T=1, H, K].
        o, final_state = fused_recurrent_kda(
            q[:, None, ...],
            k[:, None, ...],
            v[:, None, ...],
            g[:, None, ...],
            beta[:, None, ...],
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
        )
        # Remove the T=1 dim: [B, 1, H, V] -> [B, H, V]
        return o[:, 0], final_state

    def _fused_kda_gate(
        self,
        layer: RadixLinearAttention,
        g: jax.Array,
    ) -> jax.Array:
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        H = g.shape[-2]
        orig_dtype = g.dtype
        g32 = g.astype(jnp.float32) + layer.dt_bias.value.reshape(H, -1).astype(jnp.float32)
        out = -jnp.exp(layer.A_log.value.reshape(H, 1).astype(jnp.float32)) * jax.nn.softplus(g32)
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

    def _unpack_conv_states(
        self,
        conv_states: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Slice ``[B, proj_size, K-1]`` into per-stream ``[B, D, K-1]`` caches.

        Q-K-V order along ``proj_size`` (matches ``_pack_conv_states``). For
        Kimi-Linear, ``num_k_heads == num_heads`` and ``head_k_dim == head_dim``,
        so all three sub-channels collapse to the same width.
        """
        D = conv_states.shape[1]
        assert D % 3 == 0, (
            f"conv_states channel dim {D} must be divisible by 3 (Q/K/V "
            "share head_dim assumption)."
        )
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
