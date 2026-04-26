from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.radix_linear_attention import RadixLinearAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class _LinearNoBias(nnx.Module):
    """HF-layout linear layer: weight is [out_features, in_features]."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.weight = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (out_features, in_features),
                dtype=dtype,
            )
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.matmul(x, jnp.swapaxes(self.weight[...], -1, -2))


class _DepthwiseConv1D(nnx.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.weight = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (channels, kernel_size),
                dtype=dtype,
            )
        )


class _GatedRMSNorm(nnx.Module):
    def __init__(
        self,
        head_dim: int,
        epsilon: float = 1e-6,
        param_dtype: jnp.dtype = jnp.float32,
    ):
        self.weight = nnx.Param(jnp.ones((head_dim,), dtype=param_dtype))
        self.epsilon = epsilon

    def __call__(self, x: jax.Array, gate: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + self.epsilon)
        x_norm = x_norm * self.weight[...].astype(jnp.float32)
        return (x_norm * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(orig_dtype)


class KimiDeltaAttention(nnx.Module):
    """Standalone Kimi Delta Attention layer.

    This intentionally implements only the KDA attention module. Full
    KimiDecoderLayer/KimiLinearModel assembly is handled by the model skeleton
    integration work.
    """

    def __init__(
        self,
        config,
        layer_idx: int = 0,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        del mesh
        linear_config = config.linear_attn_config
        self.config = config
        self.layer_idx = layer_idx
        self.layer_id = layer_idx
        self.hidden_size = config.hidden_size
        self.conv_size = linear_config["short_conv_kernel_size"]
        self.head_dim = linear_config["head_dim"]
        self.head_k_dim = self.head_dim
        self.num_heads = linear_config["num_heads"]
        self.num_k_heads = self.num_heads
        self.projection_k_size = self.num_k_heads * self.head_k_dim
        self.projection_size = self.num_heads * self.head_dim
        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        self.q_proj = _LinearNoBias(self.hidden_size, self.projection_k_size, dtype)
        self.k_proj = _LinearNoBias(self.hidden_size, self.projection_k_size, dtype)
        self.v_proj = _LinearNoBias(self.hidden_size, self.projection_size, dtype)

        self.q_conv1d = _DepthwiseConv1D(self.projection_k_size, self.conv_size, dtype)
        self.k_conv1d = _DepthwiseConv1D(self.projection_k_size, self.conv_size, dtype)
        self.v_conv1d = _DepthwiseConv1D(self.projection_size, self.conv_size, dtype)

        self.A_log = nnx.Param(jnp.zeros((1, 1, self.num_heads, 1), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.zeros((self.projection_size,), dtype=jnp.float32))

        self.f_a_proj = _LinearNoBias(self.hidden_size, self.head_dim, dtype)
        self.f_b_proj = _LinearNoBias(self.head_dim, self.projection_size, dtype)
        self.b_proj = _LinearNoBias(self.hidden_size, self.num_heads, dtype)
        self.g_a_proj = _LinearNoBias(self.hidden_size, self.head_dim, dtype)
        self.g_b_proj = _LinearNoBias(self.head_dim, self.projection_size, dtype)
        self.o_norm = _GatedRMSNorm(self.head_dim, epsilon=self.rms_norm_eps)
        self.o_proj = _LinearNoBias(self.projection_size, self.hidden_size, dtype)

        self.attn = RadixLinearAttention(
            layer_id=layer_idx,
            num_q_heads=self.num_heads,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_heads,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_dim,
            conv_weights=None,
            bias=None,
            activation=jax.nn.silu,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scaling=self.head_k_dim**-0.5,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: jax.Array | None,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ) -> jax.Array:
        del positions
        hidden_states, restore_batch_dim = self._flatten_hidden_states(hidden_states)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k, v = self._short_convs(q, k, v, forward_batch, recurrent_state_pool)
        q = self._split_heads(q, self.num_k_heads, self.head_k_dim)
        k = self._split_heads(k, self.num_k_heads, self.head_k_dim)
        v = self._split_heads(v, self.num_heads, self.head_dim)

        q = self._l2_normalize(q)
        k = self._l2_normalize(k)

        raw_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        raw_gate = self._split_heads(raw_gate, self.num_heads, self.head_dim)
        beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))

        o = self.attn(
            forward_batch,
            (q, k, v),
            a=raw_gate,
            b=beta,
            recurrent_state_pool=recurrent_state_pool,
        )
        o = self._split_heads(o, self.num_heads, self.head_dim)

        output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        output_gate = self._split_heads(output_gate, self.num_heads, self.head_dim)
        o = self.o_norm(o, output_gate).reshape(hidden_states.shape[0], self.projection_size)
        o = self.o_proj(o)

        if restore_batch_dim:
            return o[None, ...]
        return o

    @staticmethod
    def _flatten_hidden_states(hidden_states: jax.Array) -> tuple[jax.Array, bool]:
        if hidden_states.ndim == 2:
            return hidden_states, False
        if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
            return hidden_states[0], True
        raise ValueError(
            "KimiDeltaAttention expects hidden_states shaped [T, hidden] "
            "or [1, T, hidden]"
        )

    @staticmethod
    def _split_heads(x: jax.Array, num_heads: int, head_dim: int) -> jax.Array:
        return x.reshape(x.shape[0], num_heads, head_dim)

    @staticmethod
    def _l2_normalize(x: jax.Array, epsilon: float = 1e-6) -> jax.Array:
        norm = jnp.linalg.norm(x.astype(jnp.float32), axis=-1, keepdims=True)
        return (x.astype(jnp.float32) / jnp.maximum(norm, epsilon)).astype(x.dtype)

    def _short_convs(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        metadata = self._linear_forward_metadata(forward_batch)
        recurrent_indices = metadata.recurrent_indices
        cu_seqlens = metadata.cu_q_lens

        conv_cache = None
        recurrent_cache = None
        if recurrent_state_pool is not None:
            layer_cache = recurrent_state_pool.get_linear_recurrent_layer_cache(self.layer_id)
            recurrent_cache, conv_cache = self._split_layer_cache(layer_cache)

        q_state, k_state, v_state = self._get_conv_states(
            conv_cache,
            recurrent_indices,
            q.shape[-1],
        )

        if forward_batch.forward_mode == ForwardMode.DECODE:
            q, q_state = self._decode_conv(q, self.q_conv1d.weight[...], q_state)
            k, k_state = self._decode_conv(k, self.k_conv1d.weight[...], k_state)
            v, v_state = self._decode_conv(v, self.v_conv1d.weight[...], v_state)
        else:
            q, q_state = self._extend_conv(q, self.q_conv1d.weight[...], q_state, cu_seqlens)
            k, k_state = self._extend_conv(k, self.k_conv1d.weight[...], k_state, cu_seqlens)
            v, v_state = self._extend_conv(v, self.v_conv1d.weight[...], v_state, cu_seqlens)

        if recurrent_state_pool is not None and recurrent_cache is not None:
            conv_state = jnp.stack([q_state, k_state, v_state], axis=1)
            recurrent_state_pool.set_linear_recurrent_layer_cache(
                self.layer_id,
                recurrent_indices,
                recurrent_cache[recurrent_indices],
                conv_state,
            )

        return q, k, v

    @staticmethod
    def _linear_forward_metadata(forward_batch: ForwardBatch):
        backend = forward_batch.attn_backend
        metadata = getattr(backend, "forward_metadata", None)
        if hasattr(metadata, "recurrent_indices"):
            return metadata
        linear_backend = getattr(backend, "linear_attn_backend", None)
        metadata = getattr(linear_backend, "forward_metadata", None)
        if hasattr(metadata, "recurrent_indices"):
            return metadata
        raise ValueError("KimiDeltaAttention requires linear recurrent forward metadata")

    @staticmethod
    def _split_layer_cache(layer_cache):
        if isinstance(layer_cache, dict):
            recurrent = layer_cache.get("recurrent")
            if recurrent is None:
                recurrent = layer_cache.get("recurrent_state")
            return recurrent, layer_cache.get("conv")
        if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
            return layer_cache
        raise TypeError("linear recurrent layer cache must be a dict or a 2-tuple")

    def _get_conv_states(
        self,
        conv_cache,
        recurrent_indices: jax.Array,
        channels: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        num_sequences = recurrent_indices.shape[0]
        zeros = jnp.zeros(
            (num_sequences, channels, self.conv_size),
            dtype=self.q_conv1d.weight[...].dtype,
        )
        if conv_cache is None:
            return zeros, zeros, zeros
        if isinstance(conv_cache, (tuple, list)) and len(conv_cache) == 3:
            return tuple(cache[recurrent_indices] for cache in conv_cache)
        if conv_cache.ndim == 4 and conv_cache.shape[1] == 3:
            conv_cache = conv_cache[recurrent_indices]
            return conv_cache[:, 0], conv_cache[:, 1], conv_cache[:, 2]
        return zeros, zeros, zeros

    def _decode_conv(
        self,
        x: jax.Array,
        weight: jax.Array,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        weight = self._conv_weight(weight)
        state = jnp.concatenate([state[:, :, 1:], x[:, :, None]], axis=-1)
        y = jnp.einsum("bck,ck->bc", state.astype(jnp.float32), weight.astype(jnp.float32))
        return jax.nn.silu(y).astype(x.dtype), state

    def _extend_conv(
        self,
        x: jax.Array,
        weight: jax.Array,
        state: jax.Array,
        cu_seqlens: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        weight = self._conv_weight(weight)
        T = x.shape[0]
        K = self.conv_size
        token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
        seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
        starts = cu_seqlens[:-1][seq_ids]
        offsets = jnp.arange(K, dtype=cu_seqlens.dtype) - (K - 1)
        source_idx = token_idx[:, None] + offsets[None, :]
        from_x = source_idx >= starts[:, None]

        x_idx = jnp.clip(source_idx, 0, jnp.maximum(T - 1, 0))
        x_window = x[x_idx]

        state_pos = jnp.clip(K + source_idx - starts[:, None], 0, K - 1)
        state_window = jnp.take_along_axis(
            jnp.swapaxes(state[seq_ids], 1, 2),
            state_pos[:, :, None],
            axis=1,
        )
        window = jnp.where(from_x[:, :, None], x_window, state_window)
        y = jnp.einsum("tkc,ck->tc", window.astype(jnp.float32), weight.astype(jnp.float32))

        ends = cu_seqlens[1:]
        state_offsets = jnp.arange(K, dtype=cu_seqlens.dtype)
        final_idx = ends[:, None] - K + state_offsets[None, :]
        final_from_x = final_idx >= cu_seqlens[:-1, None]
        final_x_idx = jnp.clip(final_idx, 0, jnp.maximum(T - 1, 0))
        final_x = jnp.swapaxes(x[final_x_idx], 1, 2)
        final_state_pos = jnp.clip(K + final_idx - cu_seqlens[:-1, None], 0, K - 1)
        final_state = jnp.take_along_axis(state, final_state_pos[:, None, :], axis=2)
        new_state = jnp.where(final_from_x[:, None, :], final_x, final_state)

        return jax.nn.silu(y).astype(x.dtype), new_state

    @staticmethod
    def _conv_weight(weight: jax.Array) -> jax.Array:
        if weight.ndim == 3 and weight.shape[1] == 1:
            weight = weight[:, 0, :]
        if weight.shape[0] < weight.shape[-1]:
            weight = jnp.swapaxes(weight, 0, 1)
        return weight


__all__ = ["KimiDeltaAttention"]
