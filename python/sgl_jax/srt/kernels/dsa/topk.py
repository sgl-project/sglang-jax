"""Top-k selection adapters for the DSA Indexer score matrix."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.radix_topk.tuned_configs import (
    DEFAULT_RADIX_TOPK_CONFIG,
    get_tuned_radix_topk_config,
)

_NEG_INF = float("-inf")


def _validate_scores(scores: jax.Array, k: int) -> None:
    if scores.ndim != 2:
        raise ValueError(f"DSA top-k scores must be rank 2, got shape={scores.shape}")
    if scores.dtype != jnp.float32:
        raise TypeError(f"DSA top-k scores must be float32, got dtype={scores.dtype}")
    if not 1 <= k <= scores.shape[-1]:
        raise ValueError(f"index_topk must be in [1, {scores.shape[-1]}], got {k}")


def select_indexer_radix_topk_indices(scores: jax.Array, *, k: int) -> jax.Array:
    """Return exact unordered radix top-k positions without selected scores."""

    _validate_scores(scores, k)
    from sgl_jax.srt.kernels.radix_topk import radix_topk_pallas

    config = get_tuned_radix_topk_config(scores.shape[-1], k)
    if config is None:
        config = DEFAULT_RADIX_TOPK_CONFIG
    padding = (-scores.shape[-1]) % config.input_alignment
    padded_scores = jnp.pad(scores, ((0, 0), (0, padding)), constant_values=_NEG_INF)
    return radix_topk_pallas(
        padded_scores,
        k=k,
        use_approx_top_k=False,
        num_seq_windows=config.num_seq_windows,
        digit_width=config.digit_width,
        num_digits=config.num_digits,
        use_tc_tiling_on_sc=config.use_tc_tiling_on_sc,
        indices_only=True,
    )[..., :k]


def select_indexer_topk(
    scores: jax.Array,
    *,
    k: int,
    implementation: str,
) -> tuple[jax.Array, jax.Array]:
    """Return top-k score values and sequence-local token indices.

    Score construction and masking belong to the DSA scorer. This adapter only
    selects candidates and normalizes every implementation to a
    ``(values, indices)`` ABI consumed by the cache/gather path. Candidate
    order is intentionally unspecified: DSA consumes the selected positions
    as a set.
    """

    _validate_scores(scores, k)

    if implementation == "exact_lax":
        return jax.lax.top_k(scores, k)

    if implementation == "approx":
        candidate_values, candidate_indices = jax.lax.approx_max_k(
            scores,
            k,
            recall_target=0.70,
            aggregate_to_topk=False,
        )
        return _order_candidates(candidate_values, candidate_indices, k=k)

    if implementation == "radix":
        candidate_indices = select_indexer_radix_topk_indices(scores, k=k)
        in_bounds = (candidate_indices >= 0) & (candidate_indices < scores.shape[-1])
        safe_indices = jnp.clip(candidate_indices, 0, scores.shape[-1] - 1)
        candidate_values = jnp.take_along_axis(scores, safe_indices, axis=-1)
        candidate_values = jnp.where(in_bounds, candidate_values, _NEG_INF)
        # Radix selection already returns the exact top-k set. Sorting that
        # K-sized set again adds an XLA sort but does not change sparse
        # attention, whose gather is permutation invariant.
        return candidate_values[..., :k], candidate_indices[..., :k]

    raise ValueError(f"unknown DSA top-k implementation: {implementation}")


def _order_candidates(
    candidate_values: jax.Array,
    candidate_indices: jax.Array,
    *,
    k: int,
) -> tuple[jax.Array, jax.Array]:
    """Order a K-sized candidate set and keep invalid ``-inf`` entries last."""

    values, selection = jax.lax.top_k(candidate_values, k)
    indices = jnp.take_along_axis(candidate_indices, selection, axis=-1)
    return values, indices
