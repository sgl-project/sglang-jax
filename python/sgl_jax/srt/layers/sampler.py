from functools import partial
from typing import List

import jax
from flax import nnx
from jax import numpy as jnp
from jax import random

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo


class Sampler(nnx.Module):
    def __init__(self, rngs: nnx.Rngs = None):
        self.rngs = rngs

    def __call__(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
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
        else:
            # Post process logits
            probs = jnp.divide(logits, sampling_info.temperatures)
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
        return batch_next_token_ids


def top_k_top_p_min_p_sampling_from_probs_jax(
    probs: jax.Array,
    top_ks: jax.Array,
    top_ps: jax.Array,
    min_ps: jax.Array,
    need_min_p_sampling: bool,
    rng: nnx.Rngs,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = _sample_part_a(
        probs, top_ks, top_ps, need_min_p_sampling, min_ps
    )

    sampled_index = random.categorical(rng, probs_sort).reshape(-1, 1)

    return _sample_part_b(probs_idx, sampled_index)


@partial(jax.jit, static_argnames=("need_min_p_sampling"))
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


@partial(jax.jit)
def _sample_part_b(probs_idx, sampled_index):
    probs_idx = probs_idx.astype(jnp.int32)
    return jnp.take_along_axis(probs_idx, axis=1, indices=sampled_index).flatten()
