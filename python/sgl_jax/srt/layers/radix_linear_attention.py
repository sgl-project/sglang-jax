"""Radix dispatch entry for linear recurrent attention layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from flax import nnx

if TYPE_CHECKING:
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
        conv_weights: jax.Array | None = None,
        bias: jax.Array | None = None,
        activation=None,
        A_log: jax.Array | None = None,
        dt_bias: jax.Array | None = None,
        scaling: float | None = None,
    ):
        self.layer_id = layer_id
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_q_dim = head_q_dim
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_weights = conv_weights
        self.bias = bias
        self.activation = activation
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.scaling = scaling

    def __call__(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: jax.Array,
        a: jax.Array,
        b: jax.Array,
        recurrent_state_pool,
    ) -> jax.Array:
        return forward_batch.attn_backend(
            mixed_qkv,
            a,
            b,
            self,
            forward_batch,
            recurrent_state_pool=recurrent_state_pool,
        )
