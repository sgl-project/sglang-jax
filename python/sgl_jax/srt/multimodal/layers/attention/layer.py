import math

import jax
from flax import nnx

from sgl_jax.srt.layers.attention.utils import get_attention_impl


class USPAttention(nnx.Module):
    """
    Ulysses Sequence Parallelism with Ring Attention.

    This class implements the USP algorithm, which is a combination of
    Ulysses-style all-to-all communication for sequence-head dimension sharding
    and Ring Attention for fine-grained sequence parallelism within subgroups.

    # FIXME(pc) we will implement above features later. For now, this is a naive attentnion implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        dropout_rate: float = 0.0,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.softmax_scale = softmax_scale or 1.0 / math.sqrt(head_size)
        self.causal = causal
        self.dropout_rate = dropout_rate

        self.attention_impl = get_attention_impl()
        self.attention_backend = self.attention_impl(
            num_attn_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_size,
            page_size=1,
            kv_partition_axis="tensor",
            mesh=None,
        )

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
    ) -> jax.Array:
        """
        Forward pass for USPAttention.

            q, k, v: [B, S_local, H, D]

        Note: Replicated tensors are not supported in this implementation.
        """

        # TODO refactor flashattention backend
        return self.attention_backend(query, key, value, None, None, None, 0)
