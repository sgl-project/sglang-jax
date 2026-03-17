import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclasses.dataclass
class DiagonalGaussianDistribution:
    parameters: jax.Array
    deterministic: bool = False
    channel_axis: int | None = None

    def __post_init__(self):
        axis = self.channel_axis
        if axis is None:
            axis = 1 if self.parameters.ndim == 4 else self.parameters.ndim - 1
        self.mean, self.logvar = jnp.split(self.parameters, 2, axis=axis)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = jnp.zeros_like(self.mean)
            self.std = jnp.zeros_like(self.mean)

    def tree_flatten(self):
        children = (self.parameters,)
        aux_data = {
            "deterministic": self.deterministic,
            "channel_axis": self.channel_axis,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.parameters = children[0]
        obj.deterministic = aux_data["deterministic"]
        obj.channel_axis = aux_data["channel_axis"]
        obj.__post_init__()
        return obj

    def sample(self, random: int | jax.Array) -> jax.Array:
        if isinstance(random, int):
            key = jax.random.PRNGKey(random)
        else:
            random = jnp.asarray(random)
            key = jax.random.PRNGKey(int(random)) if random.ndim == 0 else random
        sample = jax.random.normal(key, self.mean.shape, dtype=self.parameters.dtype)
        return self.mean + self.std * sample

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> jax.Array:
        if self.deterministic:
            return jnp.array([0.0], dtype=self.parameters.dtype)
        reduce_axes = tuple(range(1, self.mean.ndim))
        if other is None:
            return 0.5 * jnp.sum(
                jnp.square(self.mean) + self.var - 1.0 - self.logvar,
                axis=reduce_axes,
            )
        return 0.5 * jnp.sum(
            jnp.square(self.mean - other.mean) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            axis=reduce_axes,
        )

    def nll(self, sample: jax.Array, dims: tuple[int, ...] | None = None) -> jax.Array:
        if self.deterministic:
            return jnp.array([0.0], dtype=self.parameters.dtype)
        if dims is None:
            dims = tuple(range(1, sample.ndim))
        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(
            logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var,
            axis=dims,
        )

    def mode(self) -> jax.Array:
        return self.mean
