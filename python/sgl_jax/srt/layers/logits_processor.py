import dataclasses
import logging
from functools import partial
from typing import List, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import Param
from flax.nnx.nn import dtypes
from flax.typing import PromoteDtypeFn
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@register_pytree_node_class
@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: jax.Array
    # Used by speculative decoding (EAGLE)
    # The last hidden layers
    hidden_states: Optional[jax.Array] = None

    ## Part 2: This part will be assigned in python/sglang/srt/layers/sampler.py::Sampler
    # The logprobs of the next tokens.                              shape: [#seq]
    next_token_logprobs: Optional[jax.Array] = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None
    # The logprobs and ids of the requested token ids in output positions. shape: [#seq, n] (n is the number of requested token ids)
    next_token_token_ids_logprobs_val: Optional[List] = None
    next_token_token_ids_logprobs_idx: Optional[List] = None

    ## Part 3: Prefill-only. This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logprobs of input tokens.        shape: [#token]
    input_token_logprobs: Optional[jax.Array] = None
    # The logprobs and ids of the top-k tokens in input positions.  shape: [#seq, #token, k]
    input_top_logprobs_val: List = None
    input_top_logprobs_idx: List = None
    # The logprobs and ids of the requested token ids in input positions. shape: [#seq, n] (n is the number of requested token ids)
    input_token_ids_logprobs_val: Optional[List] = None
    input_token_ids_logprobs_idx: Optional[List] = None

    def tree_flatten(self):
        children = (
            self.next_token_logits,
            self.hidden_states,
            self.next_token_logprobs,
            self.input_token_logprobs,
        )

        aux_data = {
            "next_token_top_logprobs_val": self.next_token_top_logprobs_val,
            "next_token_top_logprobs_idx": self.next_token_top_logprobs_idx,
            "next_token_token_ids_logprobs_val": self.next_token_token_ids_logprobs_val,
            "next_token_token_ids_logprobs_idx": self.next_token_token_ids_logprobs_idx,
            "input_top_logprobs_val": self.input_top_logprobs_val,
            "input_top_logprobs_idx": self.input_top_logprobs_idx,
            "input_token_ids_logprobs_val": self.input_token_ids_logprobs_val,
            "input_token_ids_logprobs_idx": self.input_token_ids_logprobs_idx,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.next_token_logits = children[0]
        obj.hidden_states = children[1]
        obj.next_token_logprobs = children[2]
        obj.input_token_logprobs = children[3]

        obj.next_token_top_logprobs_val = aux_data["next_token_top_logprobs_val"]
        obj.next_token_top_logprobs_idx = aux_data["next_token_top_logprobs_idx"]
        obj.next_token_token_ids_logprobs_val = aux_data[
            "next_token_token_ids_logprobs_val"
        ]
        obj.next_token_token_ids_logprobs_idx = aux_data[
            "next_token_token_ids_logprobs_idx"
        ]
        obj.input_top_logprobs_val = aux_data["input_top_logprobs_val"]
        obj.input_top_logprobs_idx = aux_data["input_top_logprobs_idx"]
        obj.input_token_ids_logprobs_val = aux_data["input_token_ids_logprobs_val"]
        obj.input_token_ids_logprobs_idx = aux_data["input_token_ids_logprobs_idx"]

        return obj

    def truncate_logits_processor_output(self, idx: jax.Array):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, jax.Array) or isinstance(value, list):
                # 注意：对于 jax.Array，切片操作是合法的；对于 list，也可以切片
                truncated = value[idx]
                setattr(self, field.name, truncated)


class LogitsProcessor(nnx.Module):
    """Logits processor for the model."""

    _requires_weight_loading = False

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head: Embed,
        # promote_dtype: dtypes.promote_dtype,
        # embedding: jax.Array,
        forward_batch: ForwardBatch,
        # dtype: Optional[jnp.dtype] = None,
    ) -> LogitsProcessorOutput:
        # promote_dtype = dtypes.promote_dtype
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            logits = _logits_processor_forward_extend(
                hidden_states,
                forward_batch.extend_start_loc,
                forward_batch.extend_seq_lens,
                lm_head.promote_dtype,
                lm_head.embedding,
                lm_head.dtype,
                self.vocab_size,
            )
        else:
            logits = _logits_processor_forward_decode(
                hidden_states,
                lm_head.promote_dtype,
                lm_head.embedding,
                lm_head.dtype,
                forward_batch.batch_size,
                self.vocab_size,
            )
        return LogitsProcessorOutput(next_token_logits=logits)


# @partial(jax.jit, static_argnums=(3, 5, 6))
def _logits_processor_forward_extend(
    hidden_states: jax.Array,
    extend_start_loc: jax.Array,
    extend_seq_lens: jax.Array,
    promote_dtype: PromoteDtypeFn,
    embedding: Param,
    dtype: jnp.dtype,
    vocab_size: int,
):
    last_token_indices = extend_start_loc + extend_seq_lens - 1
    # Shape: [batch_size, hidden_size]
    last_hidden_states = hidden_states[last_token_indices]

    return _lm_head_forward(
        last_hidden_states,
        embedding,
        promote_dtype,
        dtype,
        vocab_size,
    )


# @partial(jax.jit, static_argnums=(1, 3, 4, 5))
def _logits_processor_forward_decode(
    hidden_states: jax.Array,
    promote_dtype: PromoteDtypeFn,
    embedding: Param,
    dtype: jnp.dtype,
    batch_size: int,
    vocab_size: int,
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


# @partial(jax.jit, static_argnums=(2, 3, 4))
def _lm_head_forward(
    last_hidden_states: jax.Array,
    embedding: Param,
    promote_dtype: PromoteDtypeFn,
    dtype: jnp.dtype,
    vocab_size: int,
):
    last_hidden_states, embedding = promote_dtype(
        (last_hidden_states, embedding.value), dtype=dtype
    )
    logging.info("before jnp.dot(last_hidden_states, embedding.T)")
    # logits = jnp.dot(last_hidden_states, embedding.T)
    logits = last_hidden_states
    # logger.info("after jnp.dot(last_hidden_states, embedding.T)")

    # logits = logits[:, :vocab_size] if logits.ndim > 1 else logits[:vocab_size]
    return logits
