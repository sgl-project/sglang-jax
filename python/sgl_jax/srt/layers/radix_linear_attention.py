"""Radix dispatch entry for linear recurrent attention layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from flax import nnx

if TYPE_CHECKING:
    from sgl_jax.srt.layers.linear import LinearBase
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class RadixLinearAttention(nnx.Module):
    def __init__(
        self,
        layer_id: int,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        head_q_dim: int,
        head_k_dim: int,
        head_v_dim: int,
        q_conv1d: LinearBase | None = None,
        k_conv1d: LinearBase | None = None,
        v_conv1d: LinearBase | None = None,
        bias: jax.Array | None = None,
        activation=None,
        A_log: nnx.Param | None = None,
        dt_bias: nnx.Param | None = None,
        scale: float | None = None,
    ):
        self.layer_id = layer_id
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_q_dim = head_q_dim
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        # LinearBase / nnx.Param references — the live containers, not
        # ``param[...]`` snapshots, so checkpoint loads done after construction
        # propagate automatically. The conv1d LinearBases are used as parameter
        # containers only (never called); their ``weight.value`` is laid out
        # directly as ``[D, K]`` for ``short_convolution``.
        self.q_conv1d = q_conv1d
        self.k_conv1d = k_conv1d
        self.v_conv1d = v_conv1d
        self.bias = nnx.data(bias) if bias is not None else None
        self.activation = activation
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.scale = scale if scale is not None else head_v_dim**-0.5

    def __call__(
        self,
        forward_batch: ForwardBatch,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        a: jax.Array,
        b: jax.Array,
        recurrent_state_pool,
    ):
        output, recurrent_state_pool = forward_batch.attn_backend(
            q,
            k,
            v,
            a,
            b,
            self,
            forward_batch,
            recurrent_state_pool,
        )
        return output, recurrent_state_pool
