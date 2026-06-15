"""Radix dispatch entry for Lightning / Gated Linear Attention layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from flax import nnx

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class RadixLightningAttention(nnx.Module):
    """Layer dispatcher for Lightning / GLA-family linear attention.

    Used by BailingMoeV2.5 (Simple GLA) and other Lightning-style attention
    where decay is a per-head static constant (ALiBi slope), not a
    data-dependent gate like KDA's A_log/dt_bias.
    """

    def __init__(
        self,
        layer_id: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(
        self,
        forward_batch: ForwardBatch,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        recurrent_state_pool,
    ):
        output, recurrent_state_pool = forward_batch.attn_backend(
            q=q,
            k=k,
            v=v,
            layer=self,
            forward_batch=forward_batch,
            pool=recurrent_state_pool,
        )
        return output, recurrent_state_pool
