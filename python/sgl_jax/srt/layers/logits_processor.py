import dataclasses
from functools import partial
from typing import List, Optional

import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx import Param
from flax.nnx.nn import dtypes
from flax.typing import PromoteDtypeFn
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.embeddings import Embed
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.utils.jax_utils import device_array


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
        # note: here only need to truncate next_token_logits and hidden_states
        self.next_token_logits = self.next_token_logits[idx]
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states[idx]

        # for field in dataclasses.fields(self):
        #     value = getattr(self, field.name)
        #     if value is None:
        #         continue
        #     if isinstance(value, jax.Array):
        #         truncated = value[idx]
        #         setattr(self, field.name, truncated)
        #     elif isinstance(value, list):
        #         if len(value) > 0 and len(idx.shape) == 1:
        #             idx_list = idx.tolist() if hasattr(idx, 'tolist') else [int(idx)]
        #             truncated = [value[i] for i in idx_list if i < len(value)]
        #         else:
        #             truncated = value
        #         setattr(self, field.name, truncated)


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    extend_return_logprob: bool = False
    extend_return_top_logprob: bool = False
    extend_token_ids_logprob: bool = False
    extend_seq_lens: Optional[jax.Array] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None
    top_logprobs_nums: Optional[List[int]] = None
    extend_input_logprob_token_ids_device: Optional[jax.Array] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    temperature: jax.Array = None
    top_p_normalized_logprobs: bool = False
    top_p: jax.Array = None

    # for padding
    real_bs: int = -1

    @classmethod
    def from_model_worker_batch(cls, batch: ModelWorkerBatch, mesh: Mesh = None):
        if batch.forward_mode.is_extend() and batch.return_logprob:
            extend_seq_lens_cpu = jax.device_get(batch.extend_seq_lens).tolist()

            extend_return_top_logprob = any(x > 0 for x in batch.top_logprobs_nums)
            extend_token_ids_logprob = any(
                x is not None for x in batch.token_ids_logprobs
            )
            extend_return_logprob = False
            extend_logprob_pruned_lens_cpu = []
            for extend_len, start_len in zip(
                extend_seq_lens_cpu,
                batch.extend_logprob_start_lens,
            ):
                if extend_len - start_len > 0:
                    extend_return_logprob = True
                extend_logprob_pruned_lens_cpu.append(extend_len - start_len)
        else:
            extend_return_logprob = extend_return_top_logprob = (
                extend_token_ids_logprob
            ) = extend_logprob_pruned_lens_cpu = False
            extend_seq_lens_cpu = None

        return cls(
            forward_mode=batch.forward_mode,
            capture_hidden_mode=batch.capture_hidden_mode,
            extend_return_logprob=extend_return_logprob,
            extend_return_top_logprob=extend_return_top_logprob,
            extend_token_ids_logprob=extend_token_ids_logprob,
            extend_seq_lens=batch.extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=batch.extend_logprob_start_lens,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
            top_logprobs_nums=batch.top_logprobs_nums,
            token_ids_logprobs=batch.token_ids_logprobs,
            extend_input_logprob_token_ids_device=device_array(
                mesh if mesh is not None else jax.sharding.get_abstract_mesh(),
                batch.extend_input_logprob_token_ids,
            ),
            real_bs=batch.real_bs,
        )


class LogitsProcessor(nnx.Module):
    """Logits processor for the model."""

    _requires_weight_loading = False

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(
        self,
        hidden_states: jax.Array,
        lm_head: Embed,
        logits_metadata: LogitsMetadata,
        mesh: Mesh,
    ) -> LogitsProcessorOutput:
        if logits_metadata.forward_mode.is_decode_or_idle():
            pruned_states = hidden_states
            sample_indices = None
            input_logprob_indices = None
        elif (
            logits_metadata.forward_mode.is_extend()
            and not logits_metadata.extend_return_logprob
        ):
            # Prefill without input logprobs.
            # if logits_metadata.padded_static_len < 0:
            #    last_index = jnp.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            # else:
            #     # If padding_static length is 5 and extended_seq_lens is [2, 3],
            #     # then our batch looks like [t00, t01, p, p, p, t10, t11, t12, p, p]
            #     # and this retrieves t01 and t12, which are the valid last tokens
            #     idx = device_array(jax.sharding.get_abstract_mesh(),np.arange(len(logits_metadata.extend_seq_lens)))
            #     last_index = (
            #         idx * logits_metadata.padded_static_len
            #         + logits_metadata.extend_seq_lens
            #         - 1
            #     )

            last_index = jnp.cumsum(logits_metadata.extend_seq_lens, axis=0) - 1
            pruned_states = hidden_states[last_index]
            sample_indices = None
            input_logprob_indices = None
        else:
            # Input logprobs are required.
            # Find 3 different indices.
            # 1. pruned_states: hidden states that we want logprobs from.
            # 2. sample_indices: Indices that have sampled tokens.
            # 3. input_logprob_indices: Indices that have input logprob tokens.
            sample_index_pt = -1
            sample_indices = []
            input_logprob_indices_pt = 0
            input_logprob_indices = []
            pt, pruned_states = 0, []
            real_req_count = 0
            for extend_logprob_start_len, extend_len in zip(
                logits_metadata.extend_logprob_start_lens_cpu,
                logits_metadata.extend_seq_lens_cpu,
            ):
                if real_req_count >= logits_metadata.real_bs:
                    break

                # It can happen in chunked prefill. We still need to sample 1 token,
                # But we don't want to include it in input logprob.
                # if extend_len == extend_logprob_start_len:
                #     start_len = extend_logprob_start_len - 1
                # else:
                start_len = extend_logprob_start_len

                # We always need at least 1 token to sample because that's required
                # by a caller.
                assert extend_len > start_len
                pruned_states.append(hidden_states[pt + start_len : pt + extend_len])
                pt += extend_len
                sample_index_pt += extend_len - start_len
                sample_indices.append(sample_index_pt)
                input_logprob_indices.extend(
                    [
                        input_logprob_indices_pt + i
                        for i in range(extend_len - extend_logprob_start_len)
                    ]
                )
                input_logprob_indices_pt += extend_len - start_len

                real_req_count += 1

            pruned_states = jnp.concat(pruned_states)
            sample_indices = device_array(
                mesh,
                np.array(
                    sample_indices,
                    dtype=jnp.int64,
                ),
            )
            input_logprob_indices = device_array(
                mesh,
                np.array(input_logprob_indices, dtype=jnp.int64),
            )

        # Compute logits for both input and sampled tokens.
        logits = self._get_logits(pruned_states, lm_head, logits_metadata)
        sampled_logits = (
            logits[sample_indices] if sample_indices is not None else logits
        )

        hidden_states_to_store: Optional[jax.Array] = None
        if logits_metadata.capture_hidden_mode.need_capture():
            if logits_metadata.capture_hidden_mode.is_full():
                hidden_states_to_store = hidden_states
            elif logits_metadata.capture_hidden_mode.is_last():
                # Get the last token hidden states. If sample_indices is None,
                # pruned states only contain the last tokens already.
                hidden_states_to_store = (
                    pruned_states[sample_indices]
                    if sample_indices is not None
                    else pruned_states
                )
            else:
                assert False, "Should never reach"

        def get_topk(arr, topk):
            sorted_indices = jnp.argsort(arr, axis=1)[:, -topk:]
            sorted_values = jnp.take_along_axis(arr, sorted_indices, axis=1)

            topk_indices = jnp.flip(sorted_indices, axis=1)
            topk_values = jnp.flip(sorted_values, axis=1)
            return topk_values, topk_indices

        if not logits_metadata.extend_return_logprob:
            # Decode mode or extend mode without return_logprob.
            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                hidden_states=hidden_states_to_store,
            )
        else:
            input_logprobs = logits[input_logprob_indices]

            del hidden_states, logits

            # Normalize the logprob w/o temperature, top-p
            pruned_lens = device_array(
                mesh,
                np.array(
                    logits_metadata.extend_logprob_pruned_lens_cpu,
                ),
            )
            if logits_metadata.temp_scaled_logprobs:
                logits_metadata.temperature = jnp.repeat(
                    logits_metadata.temperature.reshape(-1),
                    pruned_lens,
                ).reshape(-1, 1)
            if logits_metadata.top_p_normalized_logprobs:
                logits_metadata.top_p = jnp.repeat(
                    logits_metadata.top_p,
                    pruned_lens,
                )
            input_logprobs = self.compute_temp_top_p_normalized_logprobs(
                input_logprobs, logits_metadata
            )

            # Get the logprob of top-k tokens
            if logits_metadata.extend_return_top_logprob:
                (
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                ) = self.get_top_logprobs(input_logprobs, logits_metadata)
            else:
                input_top_logprobs_val = input_top_logprobs_idx = None

            # Get the logprob of given token id
            if logits_metadata.extend_token_ids_logprob:
                (
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                ) = self.get_token_ids_logprobs(input_logprobs, logits_metadata)
            else:
                input_token_ids_logprobs_val = input_token_ids_logprobs_idx = None

            input_token_logprobs = input_logprobs[
                device_array(mesh, np.arange(input_logprobs.shape[0])),
                logits_metadata.extend_input_logprob_token_ids_device,
            ]

            return LogitsProcessorOutput(
                next_token_logits=sampled_logits,
                input_token_logprobs=input_token_logprobs,
                input_top_logprobs_val=input_top_logprobs_val,
                input_top_logprobs_idx=input_top_logprobs_idx,
                hidden_states=hidden_states_to_store,
                input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            )

    @staticmethod
    def get_token_ids_logprobs(
        all_logprobs: jax.Array, logits_metadata: LogitsMetadata
    ):
        input_token_ids_logprobs_val, input_token_ids_logprobs_idx = [], []
        pt = 0
        for token_ids, pruned_len in zip(
            logits_metadata.token_ids_logprobs,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_token_ids_logprobs_val.append([])
                input_token_ids_logprobs_idx.append([])
                continue

            input_token_ids_logprobs_val.append(
                [all_logprobs[pt + j, token_ids].tolist() for j in range(pruned_len)]
            )
            input_token_ids_logprobs_idx.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len

        return input_token_ids_logprobs_val, input_token_ids_logprobs_idx

    @staticmethod
    def get_top_logprobs(all_logprobs: jax.Array, logits_metadata: LogitsMetadata):
        max_k = max(logits_metadata.top_logprobs_nums)
        values, indices = jax.lax.top_k(all_logprobs, max_k)
        values = values.tolist()
        indices = indices.tolist()

        input_top_logprobs_val, input_top_logprobs_idx = [], []

        pt = 0
        for k, pruned_len in zip(
            logits_metadata.top_logprobs_nums,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_top_logprobs_val.append([])
                input_top_logprobs_idx.append([])
                continue

            input_top_logprobs_val.append(
                [values[pt + j][:k] for j in range(pruned_len)]
            )
            input_top_logprobs_idx.append(
                [indices[pt + j][:k] for j in range(pruned_len)]
            )
            pt += pruned_len

        return input_top_logprobs_val, input_top_logprobs_idx

    @staticmethod
    def compute_temp_top_p_normalized_logprobs(
        last_logits: jax.Array, logits_metadata: LogitsMetadata
    ) -> jax.Array:
        """
        compute logprobs for the output token from the given logits.

        Returns:
            jax.Array: logprobs from logits
        """
        # Scale logits if temperature scaling is enabled
        if logits_metadata.temp_scaled_logprobs:
            last_logits = last_logits / logits_metadata.temperature

        # Normalize logprobs if top_p normalization is enabled
        # NOTE: only normalize logprobs when top_p is set and not equal to 1.0
        if (
            logits_metadata.top_p_normalized_logprobs
            and (logits_metadata.top_p != 1.0).any()
        ):
            from sgl_jax.srt.layers.sampler import top_p_normalize_probs_jax

            probs = jnp.softmax(last_logits, axis=-1)
            del last_logits
            probs = top_p_normalize_probs_jax(probs, logits_metadata.top_p)
            return jnp.log(probs)
        else:
            # return torch.nn.functional.log_softmax(last_logits, dim=-1)
            return nn.log_softmax(last_logits, axis=-1)

    def _get_logits(
        self,
        hidden_states: jax.Array,
        lm_head: Embed,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Get logits from hidden_states.

        If sampled_logits_only is True, it means hidden_states only contain the
        last position (e.g., extend without input logprobs). The caller should
        guarantee the given hidden_states follow this constraint.
        """
        hidden_states, embedding = lm_head.promote_dtype(
            (hidden_states, lm_head.embedding.value),
            dtype=lm_head.dtype,
        )

        logits = jnp.dot(hidden_states, embedding.T)

        logits = (
            logits[:, : self.vocab_size]
            if logits.ndim > 1
            else logits[: self.vocab_size]
        )

        return logits


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
    logits = jnp.dot(last_hidden_states, embedding.T)

    logits = logits[:, :vocab_size] if logits.ndim > 1 else logits[:vocab_size]
    return logits
