import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from functools import partial
from flax.typing import PromoteDtypeFn


@dataclasses.dataclass
class LogitsProcessorOutput:
    next_token_logits: jax.Array


class LogitsProcessor(nnx.Module):
    """Logits processor for the model."""
    _requires_weight_loading = False

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head: Embed,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            logits=_logits_processor_forward_extend(
                hidden_states,
                forward_batch.extend_start_loc,
                forward_batch.seq_lens,
                lm_head.promote_dtype,
                lm_head.embedding.value,
                lm_head.dtype,
                self.vocab_size,
            )
        else:
            logits=_logits_processor_forward_decode(
                hidden_states,
                lm_head.promote_dtype,
                lm_head.embedding.value,
                forward_batch.batch_size,
                lm_head.dtype,
                self.vocab_size,
            )
        return LogitsProcessorOutput(next_token_logits=logits)

@partial(jax.jit,static_argnums=(3,5,6))
def _logits_processor_forward_extend(
    hidden_states:jax.Array,
    extend_start_loc:jax.Array,
    seq_lens: jax.Array,
    promote_dtype:PromoteDtypeFn,
    embedding:jax.Array,
    dtype:jnp.dtype,
    vocab_size:int,
    ):
    last_token_indices = extend_start_loc + seq_lens - 1
    # Shape: [batch_size, hidden_size]
    last_hidden_states = hidden_states[last_token_indices]

    return _lm_head_forward(
        last_hidden_states,
        embedding,
        promote_dtype,
        dtype,
        vocab_size,
    )

@partial(jax.jit,static_argnums=(1,3,4,5))
def _logits_processor_forward_decode(
    hidden_states:jax.Array,
    promote_dtype:PromoteDtypeFn,
    embedding:jax.Array,
    batch_size:int,
    dtype:jnp.dtype,
    vocab_size:int,
    ):
    last_token_indices = jnp.arange(batch_size)
    # Shape: [batch_size, hidden_size]
    last_hidden_states = hidden_states[last_token_indices]
    return _lm_head_forward(
        last_hidden_states,
        embedding,
        promote_dtype,
        dtype,
        vocab_size,
    )

@partial(jax.jit,static_argnums=(2,3,4))
def _lm_head_forward(
    last_hidden_states:jax.Array,
    embedding:jax.Array,
    promote_dtype:PromoteDtypeFn, 
    dtype:jnp.dtype,
    vocab_size:int,
    ):
    last_hidden_states, embedding = promote_dtype(
            (last_hidden_states, embedding), dtype=dtype
        )
    logits=jnp.dot(last_hidden_states, embedding.T)

    logits = logits[:,
                    :vocab_size] if logits.ndim > 1 else logits[:vocab_size]
    return logits