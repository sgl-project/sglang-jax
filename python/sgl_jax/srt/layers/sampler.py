from functools import partial
from typing import List, Optional

import jax
import numpy as np
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax import random

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata


class Sampler(nnx.Module):
    def __init__(self, rngs: nnx.Rngs = None):
        self.rngs = rngs

    def _greedy_sampling(self, operands):
        """Greedy sampling branch"""
        logits, _, _, _ = operands
        batch_next_token_ids = jnp.argmax(logits, -1).flatten()
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        return batch_next_token_ids, logprobs

    def _regular_sampling(self, operands):
        """Regular sampling branch"""
        logits, sampling_metadata, positions, rng = operands

        # Post process logits
        processed_logits = jnp.divide(logits, sampling_metadata.temperatures).astype(
            logits.dtype
        )

        probs = jax.nn.softmax(processed_logits, axis=-1)

        batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_jax(
            probs,
            sampling_metadata.top_ks,
            sampling_metadata.top_ps,
            sampling_metadata.min_ps,
            positions,
            sampling_metadata.sampling_seeds,
            sampling_metadata.need_min_p_sampling,
            rng,
        )

        logprobs = jnp.log(probs).clip(min=jnp.finfo(probs.dtype).min)
        return batch_next_token_ids, logprobs

    def _process_logprob_results(self, operands):
        """Process logprob results when return_logprob=True"""
        logits_output, sampling_metadata, batch_next_token_ids, logprobs = operands

        # Set next_token_logprobs
        logits_output.next_token_logprobs = logprobs[
            np.arange(len(batch_next_token_ids)),
            batch_next_token_ids,
        ]

        # Set top_logprobs if needed
        if sampling_metadata.top_logprobs_nums is not None and any(
            x > 0 for x in sampling_metadata.top_logprobs_nums
        ):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, sampling_metadata.top_logprobs_nums)

        # Set token_ids_logprobs if needed
        if sampling_metadata.token_ids_logprobs is not None and any(
            x is not None for x in sampling_metadata.token_ids_logprobs
        ):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(logprobs, sampling_metadata.token_ids_logprobs)

        return None

    def __call__(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_metadata: SamplingMetadata,
        positions: jax.Array,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_metadata: Metadata for sampling
            positions: The positions of the tokens in the sequence.
        """

        logits = jnp.reshape(
            logits_output.next_token_logits,
            (-1, logits_output.next_token_logits.shape[-1]),
        )

        _, rng = jax.random.split(self.rngs.params())

        operands = (logits, sampling_metadata, positions, rng)
        batch_next_token_ids, logprobs = lax.cond(
            sampling_metadata.is_all_greedy,
            self._greedy_sampling,
            self._regular_sampling,
            operands,
        )

        logprob_operands = (
            logits_output,
            sampling_metadata,
            batch_next_token_ids,
            logprobs,
        )
        lax.cond(
            sampling_metadata.return_logprob,
            self._process_logprob_results,
            lambda operands: None,
            logprob_operands,
        )

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
    positions: jax.Array,
    sampling_seeds: jax.Array = None,
    need_min_p_sampling: bool = False,
    rng: nnx.Rngs = None,
):
    """A top-k, top-p and min-p sampling implementation with native jax operations."""
    probs_sort, probs_idx = _sample_part_a(
        probs, top_ks, top_ps, min_ps, need_min_p_sampling
    )

    multinomial_operands = (probs_sort, sampling_seeds, positions, rng)
    sampled_index = lax.cond(
        sampling_seeds is not None,
        multinomial_with_seed,
        multinomial,
        multinomial_operands,
    )

    return _sample_part_b(probs_idx, sampled_index)


def multinomial(
    operands,
) -> jax.Array:
    inputs, _, _, rng = operands
    return random.categorical(rng, jnp.log(inputs)).reshape(-1, 1)


def multinomial_with_seed(
    operands,
) -> jax.Array:
    """
    Note:
    1. This implementation is copied from https://github.com/sgl-project/sglang/blob/e2ac7888b8cb1fd6c33a7ec58d27a5f5b5b24e0c/python/sglang/srt/layers/sampler.py#L268.
    2. Based on last response in issue, the fixed four big prime numbers can be set freely. 8589934591 is out of uin32, so I replace it with 805306457.
    - issue: https://github.com/sgl-project/sglang/issues/10938

    Samples n elements from an input array `inputs` of shape (n, m) using
    a unique random seed for each row.

    Args:
        inputs: A float array of shape (n, m) representing n categorical
                distributions with m categories each. The values are treated
                as weights and do not need to sum to 1.
        seed:   An integer array of shape (n,) containing the random seed
                for each corresponding row in `inputs`.
        positions: The positions of the tokens in the sequence.

    Returns:
        A array of shape (n,) where the i-th element is an index sampled
        from the distribution in `inputs[i]` using `seed[i]`.
    """
    inputs, seed, positions, _ = operands
    if seed is None:
        # note: this codes is used to keep compatible with lax.cond
        return multinomial(operands)
    n, m = inputs.shape
    step_seed = seed * 19349663 ^ positions * 73856093
    seed_expanded = step_seed[:, None]
    col_indices = jnp.arange(m)[None, :]
    hashed = seed_expanded * 805306457 ^ col_indices * 479001599
    uniform_samples = (hashed % (2**24)).astype(jnp.float32) / (2**24)
    epsilon = 1e-9
    gumbel_noise = -jnp.log(-jnp.log(uniform_samples + epsilon) + epsilon)
    log_probs = jnp.log(inputs + epsilon)
    perturbed_log_probs = log_probs + gumbel_noise
    return jnp.argmax(perturbed_log_probs, axis=1, keepdims=True)


def _apply_min_p_filter(operands):
    """Apply min_p filtering when need_min_p_sampling=True"""
    probs_sort, min_ps = operands
    min_p_thresholds = probs_sort[:, 0] * min_ps
    min_p_mask = probs_sort < min_p_thresholds.reshape(-1, 1)
    return jnp.where(min_p_mask, 0.0, probs_sort)


def _sample_part_a(probs, top_ks, top_ps, min_ps, need_min_p_sampling: bool):
    probs_sort = jnp.sort(probs, axis=-1)[
        :, ::-1
    ]  # Sort and reverse for descending order
    probs_idx = jnp.argsort(probs, axis=-1)[:, ::-1]
    probs_sum = jnp.cumsum(probs_sort, axis=-1)

    top_k_mask = jnp.arange(0, probs.shape[-1]).reshape(1, -1) >= top_ks.reshape(-1, 1)
    probs_sort = jnp.where(top_k_mask, 0.0, probs_sort)

    top_p_mask = (probs_sum - probs_sort) > top_ps.reshape(-1, 1)
    probs_sort = jnp.where(top_p_mask, 0.0, probs_sort)

    # Use lax.cond to avoid recompilation due to need_min_p_sampling changes
    min_p_operands = (probs_sort, min_ps)
    probs_sort = lax.cond(
        need_min_p_sampling,
        _apply_min_p_filter,
        lambda operands: operands[0],  # No min_p filtering, just return probs_sort
        min_p_operands,
    )

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
