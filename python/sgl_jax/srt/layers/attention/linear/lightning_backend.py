"""LightningAttnBackend — GLA backend.

DECODE uses ``decode_simple_gla_fused`` (Pallas, in-kernel async DMA
gather/scatter on the recurrent state buffer).

EXTEND uses the baseline ``simple_gla_fwd`` (Pallas) wrapped with JAX
gather/scatter — a fused EXTEND variant existed but was reverted as
slower than the baseline path used here.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import effective_axis
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import simple_gla_fwd
except ModuleNotFoundError:
    simple_gla_fwd = None

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla_fused import decode_simple_gla_fused
except ModuleNotFoundError:
    decode_simple_gla_fused = None

if TYPE_CHECKING:
    from sgl_jax.srt.layers.radix_lightning_attention import RadixLightningAttention
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

_CHUNK_SIZE = 64


def _head_axis(mesh: jax.sharding.Mesh | None, num_heads: int) -> str | None:
    tensor_size = mesh.shape.get("tensor", 1) if mesh else 1
    return "tensor" if num_heads % tensor_size == 0 else None


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


def _compute_layer_slope(
    layer_id: int,
    num_hidden_layers: int,
    num_heads: int,
    mesh: jax.sharding.Mesh | None = None,
) -> jax.Array:
    """Per-layer slope decay used as ``g_gamma`` by the simple_gla kernels.

    Returned array is sharded along the ``tensor`` axis when the head count is
    divisible by TP. Otherwise it stays replicated so shard_map specs can
    match the actual array sharding under JAX 0.9.2.
    """
    base = np.asarray(_build_alibi_base_slopes(num_heads), dtype=np.float32)
    slope_np = -base * (1 - (layer_id - 1) / (num_hidden_layers - 1) + 1e-5)
    if mesh is None:
        return jnp.asarray(slope_np)
    sharding = NamedSharding(mesh, P(_head_axis(mesh, num_heads)))
    return jax.make_array_from_callback(slope_np.shape, sharding, lambda idx: slope_np[idx])


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
                    lid: _compute_layer_slope(lid, num_hidden_layers, num_heads, mesh)
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
        md = self.forward_metadata
        recurrent_buffer, _ = self.get_layer_cache(recurrent_state_pool, layer.layer_id)

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
            output, new_buffer = self._forward_decode(
                q,
                k,
                v,
                recurrent_buffer,
                md.recurrent_indices,
                md.has_initial_state,
                slope,
            )
        elif forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_buffer = self._forward_extend(
                q,
                k,
                v,
                recurrent_buffer,
                md.recurrent_indices,
                md.has_initial_state,
                slope,
            )
        else:
            raise NotImplementedError(
                f"LightningAttnBackend does not support {forward_batch.forward_mode}"
            )

        return output.reshape(output.shape[0], -1), (new_buffer, [])

    @named_scope("lightning_decode")
    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        recurrent_buffer: jax.Array,
        recurrent_indices: jax.Array,
        has_initial_state: jax.Array,
        slope: jnp.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """Decode forward via fused Pallas kernel with in-kernel state DMA."""
        if decode_simple_gla_fused is None:
            raise ImportError("simple_gla_fused kernel is required for GLA decode")

        q_data = effective_axis(q, 0, "data")
        q_head = effective_axis(q, 1, "tensor")
        k_data = effective_axis(k, 0, "data")
        k_head = effective_axis(k, 1, "tensor")
        v_data = effective_axis(v, 0, "data")
        v_head = effective_axis(v, 1, "tensor")
        slope_head = effective_axis(slope, 0, "tensor")
        state_data = effective_axis(recurrent_buffer, 0, "data")
        state_head = effective_axis(recurrent_buffer, 1, "tensor")
        idx_data = effective_axis(recurrent_indices, 0, "data")
        has_data = effective_axis(has_initial_state, 0, "data")

        def _decode_fn(q_l, k_l, v_l, gamma, buf_l, idx_l, has_l):
            return decode_simple_gla_fused(
                q_l,
                k_l,
                v_l,
                recurrent_buffer=buf_l,
                recurrent_indices=idx_l,
                has_initial_state=has_l,
                g_gamma=gamma,
                scale=None,
            )

        return jax.shard_map(
            _decode_fn,
            mesh=self.mesh,
            in_specs=(
                P(q_data, q_head, None),  # q
                P(k_data, k_head, None),  # k
                P(v_data, v_head, None),  # v
                P(slope_head),  # slope
                P(state_data, state_head, None, None),  # recurrent_buffer
                P(idx_data),  # recurrent_indices
                P(has_data),  # has_initial_state
            ),
            out_specs=(
                P(q_data, q_head, None),
                P(state_data, state_head, None, None),
            ),
            check_vma=False,
        )(q, k, v, slope, recurrent_buffer, recurrent_indices, has_initial_state)

    @named_scope("lightning_extend")
    def _forward_extend(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        recurrent_buffer: jax.Array,
        recurrent_indices: jax.Array,
        has_initial_state: jax.Array,
        slope: jnp.ndarray,
    ) -> tuple[jax.Array, jax.Array]:
        """Extend forward via baseline simple_gla_fwd + JAX gather/scatter."""
        if simple_gla_fwd is None:
            raise ImportError("simple_gla kernel is required for GLA prefill")

        cu_seqlens = self.forward_metadata.cu_q_lens
        chunk_size = self.chunk_size
        q_data = effective_axis(q, 0, "data")
        q_head = effective_axis(q, 1, "tensor")
        k_data = effective_axis(k, 0, "data")
        k_head = effective_axis(k, 1, "tensor")
        v_data = effective_axis(v, 0, "data")
        v_head = effective_axis(v, 1, "tensor")
        slope_head = effective_axis(slope, 0, "tensor")
        state_data = effective_axis(recurrent_buffer, 0, "data")
        state_head = effective_axis(recurrent_buffer, 1, "tensor")
        idx_data = effective_axis(recurrent_indices, 0, "data")
        has_data = effective_axis(has_initial_state, 0, "data")
        cu_data = effective_axis(cu_seqlens, 0, "data")

        def _prefill_fn(q_l, k_l, v_l, gamma, buf_l, idx_l, has_l, cu_l):
            h0 = buf_l[idx_l]
            h0 = jnp.where(has_l[:, None, None, None], h0, 0.0)

            output, ht = simple_gla_fwd(
                q_l[None],
                k_l[None],
                v_l[None],
                g_gamma=gamma,
                h0=h0,
                cu_seqlens_dev=cu_l,
                scale=None,
                use_ht=True,
                chunk_size=chunk_size,
            )

            # Skip writing back to dummy slot 0.
            keep_mask = (idx_l == 0).reshape(-1, 1, 1, 1)
            safe_val = jnp.where(keep_mask, buf_l[idx_l], ht)
            new_buf = buf_l.at[idx_l].set(safe_val)
            return output[0], new_buf

        return jax.shard_map(
            _prefill_fn,
            mesh=self.mesh,
            in_specs=(
                P(q_data, q_head, None),  # q
                P(k_data, k_head, None),  # k
                P(v_data, v_head, None),  # v
                P(slope_head),  # slope
                P(state_data, state_head, None, None),  # recurrent_buffer
                P(idx_data),  # recurrent_indices
                P(has_data),  # has_initial_state
                P(cu_data),  # cu_seqlens
            ),
            out_specs=(
                P(q_data, q_head, None),
                P(state_data, state_head, None, None),
            ),
            check_vma=False,
        )(q, k, v, slope, recurrent_buffer, recurrent_indices, has_initial_state, cu_seqlens)


__all__ = ["LightningAttnBackend"]
