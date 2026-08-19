"""K3's depth-wise residual protocol (AttnRes), factored out so it can be tested on its own.

This is the bookkeeping half of Attention Residuals. The *math* lives in
``kimi_k3_layers.AttentionResidual``; what lives here is the per-layer state machine that decides
which candidates that math sees, ported from ``KimiDecoderLayer.forward`` in
``vllm_torchtpu/models/vllm/kimi_k3/model.py``.

The protocol per layer, when ``attn_res_block_size`` is set::

    prefix_sum = hidden_states
    if block_residuals is non-empty:
        hidden_states = self_attention_res(prefix_sum, block_residuals)   # AttnRes #1
    if layer_idx % attn_res_block_size == 0:
        block_residuals = concat(block_residuals, prefix_sum)             # checkpoint this depth
        prefix_sum = None                                                 # and restart the sum
    hidden_states = self_attn(input_layernorm(hidden_states))
    prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
    hidden_states = mlp_res(prefix_sum, block_residuals)                  # AttnRes #2
    hidden_states = mlp_or_moe(post_attention_layernorm(hidden_states))
    return prefix_sum + hidden_states, block_residuals

Two things are easy to get wrong and are what the tests pin:

1. **`prefix_sum` is reset to None at a checkpoint boundary**, so the running sum restarts from the
   attention output of that layer rather than continuing across the boundary. Carrying it across
   silently changes what every downstream AttnRes sees.
2. **`block_residuals` grows by one candidate per checkpoint**, so the softmax in AttnRes is over a
   set whose size depends on depth. A layer at depth d sees ``floor(d/block_size) + 1`` candidates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def initial_block_residuals(
    batch_tokens: int, hidden_size: int, dtype: jnp.dtype = jnp.bfloat16
) -> jax.Array:
    """The empty candidate set the first layer starts from: ``[tokens, 0, hidden]``."""
    return jnp.zeros((batch_tokens, 0, hidden_size), dtype=dtype)


def residual_state_transition(
    layer_idx: int,
    attn_res_block_size: int,
    prefix_sum: jax.Array,
    block_residuals: jax.Array,
) -> tuple[jax.Array, jax.Array | None]:
    """The checkpoint half of the protocol (pre-attention).

    Returns ``(block_residuals, prefix_sum_or_None)``. ``prefix_sum`` is returned as ``None`` at a
    checkpoint boundary to signal "restart the running sum from this layer's attention output" --
    the reference uses exactly that sentinel and the distinction is load-bearing.
    """
    if layer_idx % attn_res_block_size == 0:
        block_residuals = jnp.concatenate(
            (block_residuals, jnp.expand_dims(prefix_sum, axis=-2)), axis=-2
        )
        return block_residuals, None
    return block_residuals, prefix_sum


def n_candidates_at_depth(layer_idx: int, attn_res_block_size: int) -> int:
    """How many candidate vectors AttnRes softmaxes over, entering layer ``layer_idx``.

    Useful as a shape oracle: the candidate axis is dynamic in depth, so a wrong answer here shows
    up as a silent broadcast rather than an error.
    """
    return layer_idx // attn_res_block_size + 1
