from collections.abc import Iterable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import dtypes
from flax.typing import Array, Axes, Dtype
from jax import lax
from jax.sharding import PartitionSpec as P


def _canonicalize_axes(rank: int, axes: Axes) -> tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple({rank + axis if axis < 0 else axis for axis in axes})


def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
        return lax.square(lax.real(x)) + lax.square(lax.imag(x))
    else:
        return lax.square(x)


class RMSNorm(nnx.Module):
    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-6,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        use_scale: bool = True,
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        axis_name: str | None = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: rnglib.Rngs,
    ):
        feature_shape = (num_features,)

        self.scale: nnx.Param[jax.Array] | None
        if use_scale:
            self.scale = nnx.Param(
                jax.random.normal(
                    jax.random.PRNGKey(0),
                    feature_shape,
                    dtype=param_dtype,
                    out_sharding=P(
                        None,
                    ),
                ),
            )
        else:
            self.scale = None

        self.num_features = num_features
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_scale = use_scale
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance

    def __call__(self, x, mask: jax.Array | None = None):
        mean, var = _compute_stats(
            x,
            self.reduction_axes,
            self.dtype,
            self.axis_name,
            self.axis_index_groups,
            use_mean=False,
            use_fast_variance=self.use_fast_variance,
            mask=mask,
        )

        return _normalize(
            x,
            mean,
            var,
            self.scale.value if self.scale else None,
            None,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.epsilon,
        )


def _compute_stats(
    x: Array,
    axes: Axes,
    dtype: Dtype | None,
    axis_name: str | None = None,
    axis_index_groups: Any = None,
    use_mean: bool = True,
    use_fast_variance: bool = True,
    mask: Array | None = None,
):
    if dtype is None:
        dtype = jnp.result_type(x)
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jnp.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)
    axes = _canonicalize_axes(x.ndim, axes)

    def maybe_distributed_mean(*xs, mask=None):
        mus = tuple(x.mean(axes, where=mask) for x in xs)
        if axis_name is None:
            return mus if len(xs) > 1 else mus[0]
        else:
            # In the distributed case we stack multiple arrays to speed comms.
            if len(xs) > 1:
                reduced_mus = lax.pmean(
                    jnp.stack(mus, axis=0),
                    axis_name,
                    axis_index_groups=axis_index_groups,
                )
                return tuple(reduced_mus[i] for i in range(len(xs)))
            else:
                return lax.pmean(mus[0], axis_name, axis_index_groups=axis_index_groups)

    if use_mean:
        if use_fast_variance:
            mu, mu2 = maybe_distributed_mean(x, _abs_sq(x), mask=mask)
            # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
            # to floating point round-off errors.
            var = jnp.maximum(0.0, mu2 - _abs_sq(mu))
        else:
            mu = maybe_distributed_mean(x, mask=mask)
            var = maybe_distributed_mean(_abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask)
    else:
        var = maybe_distributed_mean(_abs_sq(x), mask=mask)
        mu = jnp.zeros_like(var)
    return mu, var


def _normalize(
    x: Array,
    mean: Array,
    var: Array,
    scale: Array | None,
    bias: Array | None,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Dtype | None,
    epsilon: float,
):
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    args = [x]
    if scale is not None:
        scale = scale.reshape(feature_shape)
        mul *= scale
        args.append(scale)
    y *= mul
    if bias is not None:
        bias = bias.reshape(feature_shape)
        y += bias
        args.append(bias)
    dtype = dtypes.canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


class GemmaRMSNorm(nnx.Module):
    """Gemma RMS normalization."""

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        kernel_axes: Sequence[str] | None = None,
        rngs: nnx.Rngs = None,
    ):
        self.epsilon = epsilon
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.zeros, kernel_axes)(
                jax.random.PRNGKey(0), (hidden_size,)
            )
        )

    def __call__(
        self, x: jax.Array, residual: jax.Array | None = None
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + jnp.astype(residual, jnp.float16) if orig_dtype == jnp.float16 else x + residual
            residual = x

        x = x.astype(jnp.float32)
        variance = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        x = x * lax.rsqrt(variance + self.epsilon)
        x = x * (1.0 + jnp.asarray(self.weight, jnp.float32))
        x = x.astype(orig_dtype)
        return x if residual is None else (x, residual)
