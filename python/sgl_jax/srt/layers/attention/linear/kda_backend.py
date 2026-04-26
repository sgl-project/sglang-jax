from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.kda import chunk_kda, fused_recurrent_kda
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


@register_pytree_node_class
@dataclass
class KDAAttnBackendMetadata(LinearRecurrentAttnBackendMetadata):
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(cu_q_lens=children[0], recurrent_indices=children[1])


class KDAAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    metadata_cls = KDAAttnBackendMetadata

    def __call__(
        self,
        mixed_qkv: jax.Array,
        a: jax.Array,
        b: jax.Array,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
        **kwargs,
    ) -> jax.Array:
        q, k, v = self._split_qkv(mixed_qkv, layer)
        g = self._reshape_gate(a, q)
        beta = self._reshape_beta(b, q)

        if forward_batch.forward_mode in (ForwardMode.IDLE, ForwardMode.DUMMY_FIRST):
            return jnp.zeros((q.shape[0], q.shape[1] * v.shape[-1]), dtype=v.dtype)
        if forward_batch.forward_mode in (
            ForwardMode.DRAFT_EXTEND,
            ForwardMode.TARGET_VERIFY,
        ):
            raise NotImplementedError(
                f"KDA does not support {forward_batch.forward_mode}"
            )

        recurrent_indices = self.forward_metadata.recurrent_indices
        recurrent_cache, _ = self._get_layer_cache(
            recurrent_state_pool,
            layer.layer_id,
        )
        initial_state = recurrent_cache[recurrent_indices].astype(jnp.float32)

        if forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_recurrent = self._forward_extend(
                q, k, v, g, beta, initial_state,
                self.forward_metadata.cu_q_lens, layer,
            )
        elif forward_batch.forward_mode == ForwardMode.DECODE:
            output, new_recurrent = self._forward_decode(
                q, k, v, g, beta, initial_state, layer,
            )
        else:
            raise NotImplementedError(
                f"KDA does not support {forward_batch.forward_mode}"
            )

        recurrent_state_pool.set_linear_recurrent_layer_cache(
            layer.layer_id,
            recurrent_indices,
            new_recurrent,
            None,
        )
        return output.reshape(output.shape[0], -1)

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
        """Chunked prefill via Pallas kernel.  Returns (output, new_state)."""
        A_log, dt_bias = self._gate_params(layer, g)
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
            A_log=A_log,
            dt_bias=dt_bias,
        )
        # Remove the B=1 packed dim: [1, T, H, V] -> [T, H, V]
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
    ) -> tuple[jax.Array, jax.Array]:
        """Single-step decode via naive JAX recurrence.  Returns (output, new_state)."""
        A_log, dt_bias = self._gate_params(layer, g)
        g = -jnp.exp(A_log.reshape(1, g.shape[-2], 1)) * jax.nn.softplus(
            g + dt_bias.reshape(1, g.shape[-2], g.shape[-1])
        )
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

    @staticmethod
    def _as_array(value) -> jax.Array:
        return value[...] if hasattr(value, "__getitem__") else value

    @classmethod
    def _gate_params(
        cls,
        layer: RadixLinearAttention,
        g: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if layer.A_log is None or layer.dt_bias is None:
            raise ValueError("KDA gate activation requires layer.A_log and layer.dt_bias")
        A_log = jnp.asarray(cls._as_array(layer.A_log), dtype=jnp.float32).reshape(-1)
        dt_bias = jnp.asarray(cls._as_array(layer.dt_bias), dtype=jnp.float32).reshape(
            g.shape[-2],
            g.shape[-1],
        )
        return A_log, dt_bias

    # ------------------------------------------------------------------
    # QKV / gate / beta helpers (KDA-specific shapes)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_qkv(
        mixed_qkv: jax.Array | tuple[jax.Array, jax.Array, jax.Array],
        layer: RadixLinearAttention,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Split mixed QKV into separate q, k, v arrays.

        Handles two formats produced by ``KimiDeltaAttention``:
        - tuple of 3 arrays (already split)
        - flat ``[T, q_dim + k_dim + v_dim]`` concatenation
        """
        if isinstance(mixed_qkv, tuple):
            return mixed_qkv

        # Flat [T, total_dim] — split and reshape to [T, H, D].
        q_size = layer.num_q_heads * layer.head_q_dim
        k_size = layer.num_k_heads * layer.head_k_dim
        q_raw, k_raw, v_raw = jnp.split(
            mixed_qkv,
            [q_size, q_size + k_size],
            axis=-1,
        )
        return (
            q_raw.reshape(mixed_qkv.shape[0], layer.num_q_heads, layer.head_q_dim),
            k_raw.reshape(mixed_qkv.shape[0], layer.num_k_heads, layer.head_k_dim),
            v_raw.reshape(mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim),
        )

    @staticmethod
    def _reshape_gate(gate: jax.Array, q: jax.Array) -> jax.Array:
        """Ensure gate has shape ``[T, H, K]`` matching q."""
        if gate.shape == q.shape:
            return gate
        if gate.ndim == 2 and gate.shape[-1] == q.shape[1] * q.shape[2]:
            return gate.reshape(q.shape)
        raise ValueError(
            f"Gate shape {gate.shape} incompatible with q {q.shape}; "
            f"expected {q.shape} or [{q.shape[0]}, {q.shape[1] * q.shape[2]}]"
        )

    @staticmethod
    def _reshape_beta(beta: jax.Array, q: jax.Array) -> jax.Array:
        """Ensure beta has shape ``[T, H]`` matching q[:2]."""
        if beta.ndim == 3 and beta.shape[-1] == 1:
            beta = beta[..., 0]
        if beta.shape == q.shape[:2]:
            return beta
        if beta.ndim == 1 and beta.shape[0] == q.shape[0] * q.shape[1]:
            return beta.reshape(q.shape[:2])
        raise ValueError(
            f"Beta shape {beta.shape} incompatible with q {q.shape}; "
            f"expected {q.shape[:2]} or [{q.shape[0] * q.shape[1]}]"
        )


__all__ = ["KDAAttnBackend", "KDAAttnBackendMetadata"]
