import math

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    FlashAttentionBackend,
)


def align_to(x, a):
    return pl.cdiv(x, a) * a


def simple_attention(query, key, value, scale=None, causal=False, mask=None):
    """Simple dot-product attention for diffusion models (no KV cache).

    Args:
        query: [B, S, H, D]
        key: [B, S, H, D]
        value: [B, S, H, D]
        scale: softmax scale, default 1/sqrt(D)
        causal: whether to apply causal mask
        mask: [B, S_k] or similar broadcastable mask
    Returns:
        output: [B, S, H, D]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # [B, H, S, D]
    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    # [B, H, S, S]
    # Perform dot product in bfloat16 but immediately cast to float32 for softmax stability
    attn_weights = jnp.einsum("bhsd,bhtd->bhst", q, k) * scale
    orig_dtype = attn_weights.dtype
    attn_weights = attn_weights.astype(jnp.float32)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        attn_weights = jnp.where(~mask, float("-inf"), attn_weights)

    if causal:
        seq_len = query.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_weights = jnp.where(mask == 0, float("-inf"), attn_weights)

    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    # [B, H, S, D]
    output = jnp.einsum("bhst,bhtd->bhsd", attn_weights, v.astype(jnp.float32))

    # Downcast back
    output = output.astype(orig_dtype)

    # [B, S, H, D]
    return jnp.transpose(output, (0, 2, 1, 3))


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
        layer_id: int = 0,
        logit_cap: float | None = None,
        scaling: float | None = None,
        mesh: jax.sharding.Mesh | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.softmax_scale = softmax_scale or 1.0 / math.sqrt(head_size)
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.head_dim = head_size
        self.layer_id = layer_id
        self.logit_cap = logit_cap or None
        self.scaling = scaling
        self.mesh = mesh
        self.attention_backend = FlashAttentionBackend(
            mesh=self.mesh, sm_scale=self.softmax_scale, causal=False
        )

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        req=None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """
        Forward pass for USPAttention.

            q, k, v: [B, S_local, H, D]

        Note: Replicated tensors are not supported in this implementation.
        """
        # Use simple attention for diffusion (no KV cache needed)
        if req is None:
            return simple_attention(query, key, value, self.softmax_scale, self.causal, mask=mask)

        # TODO refactor flashattention backend
        return req.attention_backend(query, key, value, self, None, None, 0)
