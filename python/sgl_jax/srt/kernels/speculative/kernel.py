import jax
import jax.numpy as jnp

from sgl_jax.srt.layers.binary_search import float32_bsearch, topk_mask


def top_k_renorm_prob(probs, top_k_values):
    """Renormalize probabilities after top-k thresholding."""
    assert len(probs.shape) == 2, f"length of probs.shape(): {len(probs.shape)} should equal to 2"
    assert probs.shape[0] == top_k_values.shape[0], (
        f"probs.shape[0]: {probs.shape[0]} should equal to "
        f"top_k_values.shape[0]: {top_k_values.shape}"
    )

    # topk_mask keeps the largest k values per row (ties may keep more than k).
    masked_probs = topk_mask(probs, top_k_values.reshape(-1), replace_val=0.0)
    return masked_probs / jnp.sum(masked_probs, axis=-1, keepdims=True)


def top_p_renorm_prob(probs, top_p_values):
    """Renormalize probabilities after top-p thresholding."""
    assert len(probs.shape) == 2, f"length of probs.shape(): {len(probs.shape)} should equal to 2"
    assert probs.shape[0] == top_p_values.shape[0], (
        f"probs.shape[0]: {probs.shape[0]} should equal to "
        f"top_p_values.shape[0]: {top_p_values.shape}"
    )

    # Move the vocab axis to second-to-last so the per-step reductions do not
    # run across vector lanes on TPU.
    p = top_p_values.reshape(-1)
    probs_t = jnp.swapaxes(probs, -1, -2)

    def predicate(threshold):
        threshold = jax.lax.expand_dims(threshold, (0,))
        mass = jnp.sum(jnp.where(probs_t >= threshold, probs_t, 0.0), axis=0)
        return mass < p

    threshold = float32_bsearch((probs.shape[0],), predicate)
    masked_probs = jnp.where(probs >= threshold[:, None], probs, 0.0)
    return masked_probs / jnp.sum(masked_probs, axis=-1, keepdims=True)
