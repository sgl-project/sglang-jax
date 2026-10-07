"""Exact LSE merge of DCP partial attention outputs.

For rank ``r`` locally-normalized output ``o_r`` and log-sum-exp ``lse_r``
(natural log, same base the kernel used):

    lse = logsumexp_r(lse_r)
    o   = sum_r exp(lse_r - lse) * o_r

Empty shards must pass ``lse = -inf`` and ``o = 0`` so they drop out of the
sum without producing NaNs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def merge_dcp_attention(
    partial_o: npt.NDArray[np.floating],
    partial_lse: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Merge stacked DCP shards.

    Args:
        partial_o: ``[dcp_size, *batch, dim]``
        partial_lse: ``[dcp_size, *batch]`` (broadcasts over ``dim``)

    Returns:
        ``(o, lse)`` with shard axis removed. ``o`` matches ``partial_o[0]``
        shape without the leading ``dcp_size``.
    """
    if partial_o.shape[0] != partial_lse.shape[0]:
        raise ValueError(f"shard count mismatch: o {partial_o.shape} vs lse {partial_lse.shape}")
    if partial_o.ndim < 2:
        raise ValueError(f"partial_o must be [dcp, *batch, dim], got {partial_o.shape}")
    if partial_lse.shape[1:] != partial_o.shape[1:-1]:
        raise ValueError(
            f"partial_lse batch dims {partial_lse.shape[1:]} must match "
            f"partial_o {partial_o.shape[1:-1]}"
        )

    # Stabilize in float64; empty shards are -inf and must become weight 0.
    lse64 = np.asarray(partial_lse, dtype=np.float64)
    o64 = np.asarray(partial_o, dtype=np.float64)
    finite = np.isfinite(lse64)
    max_lse = np.max(np.where(finite, lse64, -np.inf), axis=0)
    # All-empty: max is -inf → treat as zero output.
    all_empty = ~np.isfinite(max_lse)
    # On an all-empty column this is (-inf) - (-inf) = nan, which the select below
    # discards -- but numpy evaluates the subtraction eagerly and warns first. The
    # result is right; silence the spurious warning on this op alone.
    with np.errstate(invalid="ignore"):
        shifted = np.where(finite, lse64 - max_lse, -np.inf)
    weights = np.exp(shifted)
    weights = np.where(finite, weights, 0.0)
    weight_sum = np.sum(weights, axis=0)
    lse = np.where(all_empty, -np.inf, max_lse + np.log(np.maximum(weight_sum, 1e-30)))
    scale = weights / np.maximum(weight_sum, 1e-30)
    scale = np.where(all_empty, 0.0, scale)
    o = np.sum(o64 * scale[..., None], axis=0)
    o = np.where(all_empty[..., None], 0.0, o)
    return o.astype(partial_o.dtype, copy=False), lse.astype(partial_lse.dtype, copy=False)


def merge_dcp_attention_jax(partial_o, partial_lse):
    """Device twin of ``merge_dcp_attention``. Same natural-log LSE merge.

    ``partial_o`` is ``[dcp_size, *batch, dim]``; ``partial_lse`` is
    ``[dcp_size, *batch]``. LSE from the kernel is ``m + log(l)`` (base-e,
    FlashAttention-2), not log2.
    """
    import jax.numpy as jnp

    if partial_o.shape[0] != partial_lse.shape[0]:
        raise ValueError(f"shard count mismatch: o {partial_o.shape} vs lse {partial_lse.shape}")
    lse64 = partial_lse.astype(jnp.float32)
    o32 = partial_o.astype(jnp.float32)
    finite = jnp.isfinite(lse64)
    max_lse = jnp.max(jnp.where(finite, lse64, jnp.float32(-jnp.inf)), axis=0)
    all_empty = ~jnp.isfinite(max_lse)
    shifted = jnp.where(finite, lse64 - max_lse, jnp.float32(-jnp.inf))
    weights = jnp.where(finite, jnp.exp(shifted), jnp.float32(0.0))
    weight_sum = jnp.sum(weights, axis=0)
    lse = jnp.where(
        all_empty,
        jnp.float32(-jnp.inf),
        max_lse + jnp.log(jnp.maximum(weight_sum, 1e-30)),
    )
    scale = weights / jnp.maximum(weight_sum, 1e-30)
    scale = jnp.where(all_empty, jnp.float32(0.0), scale)
    o = jnp.sum(o32 * scale[..., None], axis=0)
    o = jnp.where(all_empty[..., None], jnp.float32(0.0), o)
    return o.astype(partial_o.dtype), lse.astype(partial_lse.dtype)
