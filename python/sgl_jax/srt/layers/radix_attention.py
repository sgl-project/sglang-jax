"""Radix attention."""

from enum import Enum

import jax
from flax import nnx

from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionType(Enum):
    """
    Attention type.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"


class RadixAttention(nnx.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        v_head_dim: int = -1,
        sliding_window_size: int = 0,
        logit_cap: float = 0,
        attn_type: AttentionType = AttentionType.DECODER,
    ):
        super().__init__()
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.sliding_window_size = sliding_window_size or None
        self.logit_cap = logit_cap or None
        self.attn_type = attn_type
        self.xai_temperature_len = -1

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **kwargs,
    ):
        assert k is not None
        assert v is not None

        attn_output, kv_fused = forward_batch.attn_backend(
            q,
            k,
            v,
            self,
            forward_batch,
            token_to_kv_pool,
            **kwargs,
        )
        return attn_output, kv_fused
