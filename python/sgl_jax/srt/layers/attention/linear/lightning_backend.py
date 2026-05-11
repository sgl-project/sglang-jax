"""LightningAttnBackend — GLA (Gated Linear Attention) backend.

Extends LinearRecurrentAttnBackend to provide:
- Chunked prefill via simple_gla_fwd (Pallas kernel, varlen — kernel pads each
  sequence internally, so cu_seqlens carries real lengths)
- Decode via fused_recurrent_simple_gla (jax.lax.scan)
- Recurrent state management through RecurrentStatePool (no conv state)

Aligns with upstream sglang's LightningAttentionBackend(MambaAttnBackendBase) pattern.
Per-layer ALiBi slope decay is owned here (indexed by layer_id), mirroring
upstream's ``self.tp_slope[layer.layer_id]``; the model layer only carries
identification + head shape via ``RadixLightningAttention``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import (
        fused_recurrent_simple_gla,
        simple_gla_fwd,
    )
except ModuleNotFoundError:
    simple_gla_fwd = None
    fused_recurrent_simple_gla = None

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_lightning_attention import RadixLightningAttention
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

_CHUNK_SIZE = 64


def _build_alibi_base_slopes(num_heads: int) -> list[float]:
    """ALiBi base slopes matching the HF BailingMoeV2.5 reference."""

    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + _build_alibi_base_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
    )


def _compute_layer_slope(layer_id: int, num_hidden_layers: int, num_heads: int) -> jnp.ndarray:
    """Per-layer slope decay used as ``g_gamma`` by the simple_gla kernels."""
    base_slopes = jnp.asarray(_build_alibi_base_slopes(num_heads), dtype=jnp.float32)
    return -base_slopes * (1 - (layer_id - 1) / (num_hidden_layers - 1) + 1e-5)


class LightningAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for GLA (Gated Linear Attention) used by BailingMoeV2.5.

    Per-layer slope (g_gamma) is pre-computed once in __init__ and indexed by
    ``layer.layer_id`` at call time, matching upstream
    ``LightningAttentionBackend.tp_slope`` ownership.
    """

    def __init__(
        self,
        mesh: jax.sharding.Mesh = None,
        chunk_size: int = _CHUNK_SIZE,
        linear_recurrent_layer_ids: list[int] | None = None,
        num_hidden_layers: int | None = None,
        num_heads: int | None = None,
    ):
        """Construct a LightningAttnBackend.

        Args:
            mesh: Required for production forward.
            chunk_size: simple_gla kernel chunk size.
            linear_recurrent_layer_ids: Global layer ids of every Lightning
                attention layer in the model.
            num_hidden_layers: Total transformer layer count, used to scale
                the per-layer slope. Required iff
                ``linear_recurrent_layer_ids`` is provided.
            num_heads: Per-layer head count, used to size the slope vector.
                Required iff ``linear_recurrent_layer_ids`` is provided.
        """
        super().__init__(mesh=mesh)
        self.chunk_size = chunk_size
        if (
            linear_recurrent_layer_ids is not None
            and num_hidden_layers is not None
            and num_heads is not None
        ):
            self.tp_slope = nnx.data(
                {
                    lid: _compute_layer_slope(lid, num_hidden_layers, num_heads)
                    for lid in linear_recurrent_layer_ids
                }
            )
        else:
            self.tp_slope = nnx.data({})

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixLightningAttention,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
        **kwargs,
    ) -> tuple[jax.Array, tuple]:
        recurrent_indices = self.forward_metadata.recurrent_indices
        ssm_states = self.get_state(recurrent_state_pool, layer.layer_id, recurrent_indices)
        try:
            slope = self.tp_slope[layer.layer_id]
        except KeyError:
            raise KeyError(
                f"LightningAttnBackend has no slope for layer_id={layer.layer_id}; "
                f"registered ids: {sorted(self.tp_slope.keys())}. "
                f"Was this backend created via attn_backend_wrapper with a "
                f"non-empty linear_recurrent_layer_ids?"
            ) from None

        if forward_batch.forward_mode == ForwardMode.DECODE:
            output, new_recurrent = self._forward_decode(q, k, v, ssm_states, slope)
        elif forward_batch.forward_mode == ForwardMode.EXTEND:
            has_init = self.forward_metadata.has_initial_state
            if has_init is not None:
                mask = has_init[:, None, None, None].astype(ssm_states.dtype)
                ssm_states = ssm_states * mask
            output, new_recurrent = self._forward_extend(q, k, v, ssm_states, slope)
        else:
            raise NotImplementedError(
                f"LightningAttnBackend does not support {forward_batch.forward_mode}"
            )

        new_ssm_full = self.set_ssm_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, new_recurrent
        )
        return output.reshape(output.shape[0], -1), (new_ssm_full, [])

    def get_state(self, recurrent_state_pool, layer_id, recurrent_indices):
        """Gather recurrent states using shard_map."""
        recurrent_buffer, _ = self.get_layer_cache(recurrent_state_pool, layer_id)

        def _gather_local(buf, indices):
            return buf[indices]

        return jax.shard_map(
            _gather_local,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None, None),
                P("data"),
            ),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(recurrent_buffer, recurrent_indices)

    def set_ssm_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_recurrent):
        """Scatter recurrent states using shard_map."""
        recurrent_buffer, _ = self.get_layer_cache(recurrent_state_pool, layer_id)

        def _scatter_local(buf, indices, state):
            return buf.at[indices].set(state)

        return jax.shard_map(
            _scatter_local,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None, None),
                P("data"),
                P("data", "tensor", None, None),
            ),
            out_specs=P("data", "tensor", None, None),
            check_vma=False,
        )(recurrent_buffer, recurrent_indices, new_recurrent)

    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        ssm_states: jax.Array,
        slope: jnp.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """Decode forward using shard_map."""
        if fused_recurrent_simple_gla is None:
            raise ImportError("simple_gla kernel is required for GLA decode")

        ssm_states = ssm_states.astype(jnp.float32)

        def _decode_fn(q_local, k_local, v_local, gamma, h0):
            q_d = q_local[:, None, :, :]
            k_d = k_local[:, None, :, :]
            v_d = v_local[:, None, :, :]
            output_d, new_state = fused_recurrent_simple_gla(
                q_d,
                k_d,
                v_d,
                g_gamma=gamma,
                initial_state=h0,
                output_final_state=True,
                scale=None,
            )
            return output_d[:, 0, :, :], new_state

        output, new_state = jax.shard_map(
            _decode_fn,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),  # q
                P("data", "tensor", None),  # k
                P("data", "tensor", None),  # v
                P("tensor"),  # slope
                P("data", "tensor", None, None),  # ssm_states
            ),
            out_specs=(
                P("data", "tensor", None),
                P("data", "tensor", None, None),
            ),
            check_vma=False,
        )(q, k, v, slope, ssm_states)

        return output, new_state

    def _forward_extend(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        ssm_states: jax.Array,
        slope: jnp.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """Extend forward using shard_map."""
        if simple_gla_fwd is None:
            raise ImportError("simple_gla kernel is required for GLA prefill")

        cu_seqlens = self.forward_metadata.cu_q_lens
        ssm_states = ssm_states.astype(jnp.float32)
        chunk_size = self.chunk_size

        def _prefill_fn(q_local, k_local, v_local, gamma, h0, cu_seqlens_p):
            output, ht = simple_gla_fwd(
                q_local[None],
                k_local[None],
                v_local[None],
                g_gamma=gamma,
                h0=h0,
                cu_seqlens_dev=cu_seqlens_p,
                scale=None,
                use_ht=True,
                chunk_size=chunk_size,
            )
            return output[0], ht

        output, new_state = jax.shard_map(
            _prefill_fn,
            mesh=self.mesh,
            in_specs=(
                P("data", "tensor", None),  # q: always has "data" axis
                P("data", "tensor", None),  # k
                P("data", "tensor", None),  # v
                P("tensor"),  # slope: replicated
                P("data", "tensor", None, None),  # ssm_states
                P("data"),  # cu_seqlens
            ),
            out_specs=(
                P("data", "tensor", None),
                P("data", "tensor", None, None),
            ),
            check_vma=False,
        )(q, k, v, slope, ssm_states, cu_seqlens)

        return output, new_state


__all__ = ["LightningAttnBackend"]
