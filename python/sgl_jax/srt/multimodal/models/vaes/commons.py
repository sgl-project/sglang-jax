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

    # todo: test this
    def __post_init__(self):
        self.mean, self.logvar = jnp.split(self.parameters, 2, axis=4)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.deterministic = self.deterministic
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def tree_flatten(self):
        children = (self.parameters,)

        aux_data = {"deterministic": self.deterministic}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.parameters = children[0]
        obj.deterministic = aux_data["deterministic"]
        obj.__post_init__()
        return obj

    def sample(self, random: int) -> jax.Array:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = jax.random.normal(
            jax.random.PRNGKey(random), self.mean.shape, dtype=self.parameters.dtype
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> jax.Array:
        if self.deterministic:
            return jax.Array([0.0])
        else:
            if other is None:
                return 0.5 * jnp.sum(
                    jnp.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    axis=[1, 2, 3],
                )
            else:
                return 0.5 * jnp.sum(
                    jnp.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    axis=[1, 2, 3],
                )

    def nll(self, sample: jax.Array, dims: tuple[int, ...] = (1, 2, 3)) -> jax.Array:
        if self.deterministic:
            return jax.Array([0.0])
        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(
            logtwopi + self.logvar + jnp.pow(sample - self.mean, 2) / self.var,
            axis=dims,
        )

    def mode(self) -> jax.Array:
        return self.mean
