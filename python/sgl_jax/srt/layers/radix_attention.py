"""Radix attention."""

from enum import Enum

import jax
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

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
        self.attn_type = attn_type

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        assert k is not None
        assert v is not None

        k = k.reshape(-1, self.kv_head_num, self.qk_head_dim)
        v = v.reshape(-1, self.kv_head_num, self.v_head_dim)
        # print(f"===========print mesh: {jax.sharding.get_abstract_mesh()}")
        # print(f"===========[before with] k, shape: {k.shape}, k.sharding: {k.sharding}")
        # print(f"===========[before with] v, shape: {v.shape}, v.sharding: {v.sharding}")
        # for i,shard in enumerate(k.addressable_shards):
        #    print(f"[before k][{i}] {shard.data.shape}")
        # for i,shard in enumerate(v.addressable_shards):
        #    print(f"[before v][{i}] {shard.data.shape}")
        k = jax.lax.with_sharding_constraint(
            k, NamedSharding(jax.sharding.get_abstract_mesh(), P(None, "tensor", None))
        )
        v = jax.lax.with_sharding_constraint(
            v, NamedSharding(jax.sharding.get_abstract_mesh(), P(None, "tensor", None))
        )
        # print(f"===========[after with] k, shape: {k.shape}, k.sharding: {k.sharding}")
        # print(f"===========[after with] v, shape: {v.shape}, v.sharding: {v.sharding}")
        # for i,shard in enumerate(k.addressable_shards):
        #    print(f"[after k][{i}] {shard.data.shape}")
        # for i,shard in enumerate(v.addressable_shards):
        #    print(f"[after v][{i}] {shard.data.shape}")

        attn_output, k, v = forward_batch.attn_backend(
            q,
            k,
            v,
            self,
            forward_batch,
            **kwargs,
        )

        return attn_output, k, v
