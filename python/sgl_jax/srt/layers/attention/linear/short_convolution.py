"""Short depthwise causal conv1d used by linear-attention backends (e.g. KDA).

The convolution is intentionally implemented as a stateless function (not an
nnx Module) so backends can freely combine it with their own weight containers
and cache layouts. Two execution paths are provided:

* ``decode`` — single-token step that rolls a per-sequence ``[B, D, K]`` cache
  left by one and writes the new token at the last slot.
* ``extend`` — variable-length packed prefill that consumes ``cu_seqlens`` to
  build a per-token sliding window mixing prior cache and in-sequence tokens.

State convention follows FLA's ``ShortConvolution``: cache has width ``K``
(not ``K-1``) and slot ``K-1`` always holds the most recent input token.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

_SUPPORTED_ACTIVATIONS = (None, "silu")


def short_convolution(
    x: jax.Array,
    weight: jax.Array,
    cache: jax.Array,
    cu_seqlens: jax.Array | None,
    forward_mode: ForwardMode,
    bias: jax.Array | None = None,
    activation: str | None = "silu",
) -> tuple[jax.Array, jax.Array]:
    """Depthwise causal conv1d with per-sequence cache.

    Args:
        x: ``[T, D]`` for ``EXTEND`` (packed varlen) or ``[B, D]`` for ``DECODE``.
        weight: depthwise kernel ``[D, K]``.
        cache: per-sequence rolling buffer ``[B, D, K]``; slot ``K-1`` is the
            most recent token of the prior context (zeros for fresh sequences).
        cu_seqlens: ``[N+1]`` cumulative sequence lengths; required for
            ``EXTEND``, ignored for ``DECODE``.
        forward_mode: ``ForwardMode.DECODE`` or ``ForwardMode.EXTEND``.
        bias: optional ``[D]`` channel bias added before the activation.
        activation: ``"silu"`` or ``None``.

    Returns:
        ``(y, new_cache)`` where ``y`` matches the leading dims of ``x`` and
        ``new_cache`` has the same shape as ``cache``.
    """
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"short_convolution activation must be one of {_SUPPORTED_ACTIVATIONS}, "
            f"got {activation!r}"
        )

    weight = _normalize_weight(weight)

    if forward_mode == ForwardMode.DECODE:
        return _decode_conv(x, weight, cache, bias, activation)
    if cu_seqlens is None:
        raise ValueError("short_convolution(EXTEND) requires cu_seqlens")
    return _extend_conv(x, weight, cache, cu_seqlens, bias, activation)


def _normalize_weight(weight: jax.Array) -> jax.Array:
    """Reduce common conv-weight layouts to ``[D, K]``."""
    # Squeeze the depthwise singleton axis if the loader handed us [D, 1, K].
    if weight.ndim == 3 and weight.shape[1] == 1:
        weight = weight[:, 0, :]
    return weight


def _apply_activation(y: jax.Array, activation: str | None) -> jax.Array:
    if activation == "silu":
        return jax.nn.silu(y)
    return y


def _decode_conv(
    x: jax.Array,
    weight: jax.Array,
    cache: jax.Array,
    bias: jax.Array | None,
    activation: str | None,
) -> tuple[jax.Array, jax.Array]:
    # Roll the cache left by one and append the new token at the last slot.
    new_cache = jnp.concatenate([cache[:, :, 1:], x[:, :, None]], axis=-1)
    y = jnp.einsum(
        "bck,ck->bc",
        new_cache.astype(jnp.float32),
        weight.astype(jnp.float32),
    )
    if bias is not None:
        y = y + bias.astype(jnp.float32)
    y = _apply_activation(y, activation)
    return y.astype(x.dtype), new_cache


def _extend_conv(
    x: jax.Array,
    weight: jax.Array,
    cache: jax.Array,
    cu_seqlens: jax.Array,
    bias: jax.Array | None,
    activation: str | None,
) -> tuple[jax.Array, jax.Array]:
    T = x.shape[0]
    K = weight.shape[-1]

    # Locate every output token within its owning sequence.
    token_idx = jnp.arange(T, dtype=cu_seqlens.dtype)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    starts = cu_seqlens[:-1][seq_ids]

    # Build the K-tap window ending at each token: source positions
    # ``[t-(K-1), ..., t]``. Positions inside the same sequence read from x;
    # positions reaching back across the sequence boundary read from cache.
    offsets = jnp.arange(K, dtype=cu_seqlens.dtype) - (K - 1)
    source_idx = token_idx[:, None] + offsets[None, :]
    from_x = source_idx >= starts[:, None]

    safe_x_idx = jnp.clip(source_idx, 0, jnp.maximum(T - 1, 0))
    x_window = x[safe_x_idx]

    cache_pos = jnp.clip(K + source_idx - starts[:, None], 0, K - 1)
    cache_window = jnp.take_along_axis(
        jnp.swapaxes(cache[seq_ids], 1, 2),
        cache_pos[:, :, None],
        axis=1,
    )
    window = jnp.where(from_x[:, :, None], x_window, cache_window)
    y = jnp.einsum(
        "tkc,ck->tc",
        window.astype(jnp.float32),
        weight.astype(jnp.float32),
    )
    if bias is not None:
        y = y + bias.astype(jnp.float32)
    y = _apply_activation(y, activation)

    # Compute the new per-sequence cache as the last K input tokens of each
    # sequence, falling back to the prior cache when the sequence is shorter
    # than K.
    ends = cu_seqlens[1:]
    state_offsets = jnp.arange(K, dtype=cu_seqlens.dtype)
    final_idx = ends[:, None] - K + state_offsets[None, :]
    final_from_x = final_idx >= cu_seqlens[:-1, None]
    safe_final_idx = jnp.clip(final_idx, 0, jnp.maximum(T - 1, 0))
    final_x = jnp.swapaxes(x[safe_final_idx], 1, 2)
    final_cache_pos = jnp.clip(K + final_idx - cu_seqlens[:-1, None], 0, K - 1)
    final_cache = jnp.take_along_axis(cache, final_cache_pos[:, None, :], axis=2)
    new_cache = jnp.where(final_from_x[:, None, :], final_x, final_cache)

    return y.astype(x.dtype), new_cache


def l2_normalize(x: jax.Array, epsilon: float = 1e-6) -> jax.Array:
    """L2-normalize ``x`` along its last axis.

    Computed in float32 for numerical stability and cast back to the input
    dtype. Used by linear-attention backends (KDA) to unit-norm the per-head
    Q/K vectors emitted by the short conv before the recurrent kernel.
    """
    norm = jnp.linalg.norm(x.astype(jnp.float32), axis=-1, keepdims=True)
    return (x.astype(jnp.float32) / jnp.maximum(norm, epsilon)).astype(x.dtype)


__all__ = ["short_convolution", "l2_normalize"]
