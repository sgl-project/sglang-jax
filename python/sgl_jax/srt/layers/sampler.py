from functools import partial
from typing import List

import jax
import numpy as np
from flax import nnx
from jax import numpy as jnp
from jax import random
from jax.sharding import Mesh

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.utils.jax_utils import device_array


class Sampler(nnx.Module):
    def __init__(self, rngs: nnx.Rngs = None):
        self.rngs = rngs

    def __call__(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        mesh: Mesh = None,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = jnp.reshape(
            logits_output.next_token_logits,
            (-1, logits_output.next_token_logits.shape[-1]),
        )

        if sampling_info.is_all_greedy:
            batch_next_token_ids = jnp.argmax(logits, -1).flatten()
            if return_logprob:
                logprobs = jax.nn.log_softmax(logits, axis=-1)
        else:
            # Post process logits
            logits = jnp.divide(logits, sampling_info.temperatures)
            probs = jax.nn.softmax(logits, axis=-1)
            _, new_rng = jax.random.split(self.rngs.params())
            # A slower fallback implementation with torch native operations.
            batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_jax(
                probs,
                sampling_info.top_ks,
                sampling_info.top_ps,
                sampling_info.min_ps,
                sampling_info.need_min_p_sampling,
                new_rng,
            )

            if return_logprob:
                logprobs = jnp.log(probs).clip(min=jnp.finfo(probs.dtype).min)

        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                np.arange(len(batch_next_token_ids)),
                batch_next_token_ids,
            ]

        return batch_next_token_ids


def get_top_logprobs(logprobs: jax.Array, top_logprobs_nums: List[int]):
    max_k = max(top_logprobs_nums)
    values, indices = jax.lax.top_k(logprobs, max_k)
    values = values.tolist()
    indices = indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: jax.Array, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx


def top_k_top_p_min_p_sampling_from_probs_jax(
    probs: jax.Array,
    top_ks: jax.Array,
    top_ps: jax.Array,
    min_ps: jax.Array,
    need_min_p_sampling: bool,
    rng: nnx.Rngs,
):
    """A top-k, top-p and min-p sampling implementation with native jax operations."""
    probs_sort, probs_idx = _sample_part_a(
        probs, top_ks, top_ps, need_min_p_sampling, min_ps
    )

    sampled_index = random.categorical(rng, jnp.log(probs_sort)).reshape(-1, 1)

    return _sample_part_b(probs_idx, sampled_index)


def _sample_part_a(probs, top_ks, top_ps, need_min_p_sampling: bool, min_ps):
    probs_sort = jnp.sort(probs, axis=-1)[
        :, ::-1
    ]  # Sort and reverse for descending order
    probs_idx = jnp.argsort(probs, axis=-1)[:, ::-1]
    probs_sum = jnp.cumsum(probs_sort, axis=-1)

    top_k_mask = jnp.arange(0, probs.shape[-1]).reshape(1, -1) >= top_ks.reshape(-1, 1)
    probs_sort = jnp.where(top_k_mask, 0.0, probs_sort)

    top_p_mask = (probs_sum - probs_sort) > top_ps.reshape(-1, 1)
    probs_sort = jnp.where(top_p_mask, 0.0, probs_sort)

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        min_p_mask = probs_sort < min_p_thresholds.reshape(-1, 1)
        probs_sort = jnp.where(min_p_mask, 0.0, probs_sort)

    return probs_sort, probs_idx


def _sample_part_b(probs_idx, sampled_index):
    probs_idx = probs_idx.astype(jnp.int32)
    return jnp.take_along_axis(probs_idx, axis=1, indices=sampled_index).flatten()


def top_p_normalize_probs_jax(
    probs: jax.Array,
    top_ps: jax.Array,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort = jnp.sort(probs, axis=-1, descending=True)
    probs_idx = jnp.argsort(probs, axis=-1, descending=True)

    # probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    exclude_mask = (probs_sum - probs_sort) > top_ps.reshape(-1, 1)
    probs_sort = jnp.where(exclude_mask, 0.0, probs_sort)
    probs_sort = probs_sort / probs_sort.sum(axis=-1, keepdim=True)
    # return jnp.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)

    num_tokens, h = probs.shape
    row_idx = jnp.arange(num_tokens)[:, None]  # [B, 1], broadcast over H
    return jnp.zeros_like(probs).at[row_idx, probs_idx].set(probs_sort)
