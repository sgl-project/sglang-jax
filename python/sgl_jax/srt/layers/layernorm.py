from typing import Any, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import dtypes, initializers
from flax.typing import Array, Axes, Dtype, Initializer
from jax import lax


def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
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
        dtype: Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        use_scale: bool = True,
        scale_init: Initializer = initializers.ones,
        reduction_axes: Axes = -1,
        feature_axes: Axes = -1,
        axis_name: Optional[str] = None,
        axis_index_groups: Any = None,
        use_fast_variance: bool = True,
        rngs: rnglib.Rngs,
    ):
        feature_shape = (num_features,)

        self.scale: nnx.Param[jax.Array] | None
        if use_scale:
            self.scale = nnx.Param(
                scale_init(jax.random.PRNGKey(0), feature_shape, param_dtype)
            )
        else:
            self.scale = None

        self.num_features = num_features
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.use_scale = use_scale
        self.scale_init = scale_init
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.axis_name = axis_name
        self.axis_index_groups = axis_index_groups
        self.use_fast_variance = use_fast_variance

    def __call__(self, x, mask: Optional[jax.Array] = None):
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
    dtype: Optional[Dtype],
    axis_name: Optional[str] = None,
    axis_index_groups: Any = None,
    use_mean: bool = True,
    use_fast_variance: bool = True,
    mask: Optional[Array] = None,
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
            var = maybe_distributed_mean(
                _abs_sq(x - jnp.expand_dims(mu, axes)), mask=mask
            )
    else:
        var = maybe_distributed_mean(_abs_sq(x), mask=mask)
        mu = jnp.zeros_like(var)
    return mu, var


def _normalize(
    x: Array,
    mean: Array,
    var: Array,
    scale: Optional[Array],
    bias: Optional[Array],
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Optional[Dtype],
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
