import jax
import numpy as np
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.sharding import Mesh

from sgl_jax.srt.layers.binary_search import topk_mask, topp_mask
from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime

_SAMPLING_EPS = 1e-5


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
        logits, sampling_metadata, positions, rng, mesh, use_sort_for_toppk_minp = operands

        # Validate broadcast compatibility for temperature division
        logits_batch_size = logits.shape[0]
        temperatures_shape = sampling_metadata.temperatures.shape

        # Temperatures should be (batch_size, 1) for proper broadcasting
        assert (
            temperatures_shape[0] == logits_batch_size
        ), f"Temperature batch size {temperatures_shape[0]} doesn't match logits batch size {logits_batch_size}"

        # Post process logits
        processed_logits = jnp.divide(logits, sampling_metadata.temperatures).astype(logits.dtype)

        probs = jax.nn.softmax(processed_logits, axis=-1)

        args = (
            logits,
            probs,
            sampling_metadata.top_ks,
            sampling_metadata.top_ps,
            sampling_metadata.min_ps,
            positions,
            sampling_metadata.temperatures,
            sampling_metadata.sampling_seeds,
            sampling_metadata.need_min_p_sampling,
            rng,
        )
        batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_jax(
            args,
            use_sort_for_toppk_minp,
        )

        log_probs = jnp.log(probs).clip(min=jnp.finfo(probs.dtype).min)
        return batch_next_token_ids, log_probs

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

    def _apply_linear_penalty(self, operands):
        """
        Args:
            logits: The input logits tensor of shape [batch_size, vocab_size]
            sampling_metadata: Metadata containing penalty information (never None)

        Returns:
            Modified logits with penalties applied
        """
        logits, sampling_metadata = operands
        penalty = sampling_metadata.linear_penalty

        # Validate penalty shape matches logits when penalty exists
        penalty = penalty.astype(logits.dtype)

        return logits + penalty

    def _apply_min_tokens_penalty(self, operands):
        """Apply min new tokens penalty to stop tokens"""
        logits, sampling_metadata = operands

        len_output = sampling_metadata.len_output_tokens
        min_new = sampling_metadata.min_new_tokens
        stop_penalties = sampling_metadata.stop_token_penalties

        # The parent lax.cond checks for None, but this branch is still traced
        # when the values are None. This guard prevents a TypeError during trace.
        if len_output is None or min_new is None or stop_penalties is None:
            return logits

        # Generate mask for sequences that haven't reached min_new_tokens
        min_new_tokens_mask = len_output < min_new

        # Apply stop token penalties only for sequences that need more tokens
        stop_penalty = jnp.where(
            min_new_tokens_mask.reshape(-1, 1),
            stop_penalties,
            jnp.array(0.0, dtype=stop_penalties.dtype),
        )

        return logits + stop_penalty.astype(logits.dtype)

    def __call__(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_metadata: SamplingMetadata,
        positions: jax.Array,
        mesh: Mesh,
        use_sort_for_toppk_minp: bool,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_metadata: Metadata for sampling
            positions: The positions of the tokens in the sequence.
            use_sort_for_toppk_minp: whether use sort when dealing with top_k, top_k and min_p.
        """
        # Apply penalties before sampling
        logits = lax.cond(
            sampling_metadata.do_penalties,
            self._apply_linear_penalty,
            lambda operands: operands[0],
            (logits_output.next_token_logits, sampling_metadata),
        )

        _, rng = jax.random.split(self.rngs.params())
        operands = (logits, sampling_metadata, positions, rng)
        regular_fn = lambda op: self._regular_sampling((*op, mesh, use_sort_for_toppk_minp))
        batch_next_token_ids, logprobs = lax.cond(
            sampling_metadata.is_all_greedy,
            self._greedy_sampling,
            regular_fn,
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


def get_top_logprobs(logprobs: jax.Array, top_logprobs_nums: list[int]):
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


def get_token_ids_logprobs(logprobs: jax.Array, token_ids_logprobs: list[list[int]]):
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


def multinomial(
    operands,
) -> jax.Array:
    inputs, _, _, rng, use_probs = operands
    if not use_probs:
        return random.categorical(rng, inputs).reshape(-1, 1)
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
    inputs, seed, positions, _, use_probs = operands
    if seed is None:
        # note: this codes is used to keep compatible with lax.cond
        return multinomial(operands)
    if not use_probs:
        inputs = jax.nn.softmax(inputs, axis=-1)
    # inputs = jax.nn.softmax(logits, axis=-1)

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


def _get_sorted_indices_np(probs_np: np.ndarray) -> np.ndarray:
    """
    CPU-side NumPy sorting index that is robust to NaNs/Infs.
    Always returns descending order indices with int32 dtype.
    """
    # 1) Map NaN -> -inf, +inf -> +inf, -inf -> -inf for stable descending order
    scores_np = np.nan_to_num(probs_np, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    # 2) argsort ascending then flip for descending
    return np.argsort(scores_np, axis=-1)[:, ::-1].astype(np.int32)


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

    num_tokens, _ = probs.shape
    row_idx = jnp.arange(num_tokens)[:, None]  # [B, 1], broadcast over H
    return jnp.zeros_like(probs).at[row_idx, probs_idx].set(probs_sort)


def _apply_min_p_filter(operands):
    """Apply min_p filtering when need_min_p_sampling=True"""
    inputs, min_ps, use_probs = operands
    if not use_probs:
        max_per_bs = jnp.max(inputs, axis=1)
        min_p_thresholds = max_per_bs * min_ps
    else:
        min_p_thresholds = inputs[:, 0] * min_ps
    min_p_mask = inputs < min_p_thresholds.reshape(-1, 1)
    return jnp.where(min_p_mask, 0.0, inputs)


def top_k_top_p_min_p_sampling_from_probs_jax(
    args,
    use_sort_for_toppk_minp,
):
    if use_sort_for_toppk_minp:
        return top_k_top_p_min_p_sampling_from_probs_jax_with_sort(args)
    return top_k_top_p_min_p_sampling_from_probs_jax_with_mask(args)


def top_k_top_p_min_p_sampling_from_probs_jax_with_sort(args):
    (
        logits,
        probs,
        top_ks,
        top_ps,
        min_ps,
        positions,
        temperatures,
        sampling_seeds,
        need_min_p_sampling,
        rng,
    ) = args

    if is_tpu_runtime():
        probs_sort = jnp.sort(probs, axis=-1)[:, ::-1]  # Sort and reverse for descending order
        probs_idx = jnp.argsort(probs, axis=-1)[:, ::-1]
    else:
        # 1) Use jax.pure_callback to compute robust descending indices on CPU
        out_spec = jnp.empty(probs.shape, dtype=jnp.int32)
        probs_idx = jax.pure_callback(
            _get_sorted_indices_np,
            out_spec,
            probs,
            vmap_method="legacy_vectorized",
        )
        # 2) Gather with sanitized probabilities (map NaNs/Infs to 0)
        sanitized_probs = jnp.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        assert probs_idx.shape == sanitized_probs.shape and probs_idx.dtype == jnp.int32

        probs_sort = jnp.take_along_axis(sanitized_probs, probs_idx, axis=-1)

    probs_sum = jnp.cumsum(probs_sort, axis=-1)

    top_k_mask = jnp.arange(0, probs.shape[-1]).reshape(1, -1) >= top_ks.reshape(-1, 1)
    probs_sort = jnp.where(top_k_mask, 0.0, probs_sort)

    top_p_mask = (probs_sum - probs_sort) > top_ps.reshape(-1, 1)
    probs_sort = jnp.where(top_p_mask, 0.0, probs_sort)

    # Use lax.cond to avoid recompilation due to need_min_p_sampling changes
    min_p_operands = (probs_sort, min_ps)
    apply_min_p_filter_fn = lambda op: _apply_min_p_filter((*op, True))
    probs_sort = lax.cond(
        need_min_p_sampling,
        apply_min_p_filter_fn,
        lambda operands: operands[0],  # No min_p filtering, just return probs_sort
        min_p_operands,
    )

    multinomial_operands = (probs_sort, sampling_seeds, positions, rng)
    multinomial_with_seed_fn = lambda op: multinomial_with_seed((*op, True))
    multinomial_fn = lambda op: multinomial((*op, True))
    sampled_index = lax.cond(
        sampling_seeds is not None,
        multinomial_with_seed_fn,
        multinomial_fn,
        multinomial_operands,
    )

    probs_idx = probs_idx.astype(jnp.int32)
    
    result = jnp.take_along_axis(probs_idx, axis=1, indices=sampled_index).flatten()
    return result


def top_k_top_p_min_p_sampling_from_probs_jax_with_mask(args):
    (
        logits,
        probs,
        top_ks,
        top_ps,
        min_ps,
        positions,
        temperatures,
        sampling_seeds,
        need_min_p_sampling,
        rng,
    ) = args
    logits = logits.astype(jnp.float32)
    logits = topk_mask(logits, top_ks, replace_val=-1e12)
    logits = topp_mask(logits, top_ps, replace_val=-1e12)

    temperatures = temperatures.astype(logits.dtype)
    logits = jnp.divide(logits, temperatures)

    min_p_operands = (logits, min_ps)
    apply_min_p_filter_fn = lambda op: _apply_min_p_filter((*op, False))
    logits = lax.cond(
        need_min_p_sampling,
        apply_min_p_filter_fn,
        lambda operands: operands[0],
        min_p_operands,
    )

    multinomial_operands = (logits, sampling_seeds, positions, rng)
    multinomial_with_seed_fn = lambda op: multinomial_with_seed((*op, False))
    multinomial_fn = lambda op: multinomial((*op, False))
    jax.debug.visualize_array_sharding(logits)
    sampled_index = lax.cond(
        sampling_seeds is not None,
        multinomial_with_seed_fn,
        multinomial_fn,
        multinomial_operands,
    )

    return sampled_index.flatten()
