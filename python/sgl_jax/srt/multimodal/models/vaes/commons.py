import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional

class DiagonalGaussianDistribution:
    # todo: test this
    def __init__(self, parameters: jax.Array, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = jnp.split(parameters, 2, axis=4)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, random:int) -> jax.Array:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = jax.random.normal(
            jax.random.PRNGKey(random),
            self.mean.shape,
            dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(
        self, other: Optional["DiagonalGaussianDistribution"] = None
    ) -> jax.Array:
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

    def nll(
        self, sample: jax.Array, dims: tuple[int, ...] = (1, 2, 3)
    ) -> jax.Array:
        if self.deterministic:
            return jax.Array([0.0])
        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(
            logtwopi + self.logvar + jnp.pow(sample - self.mean, 2) / self.var,
            axis=dims,
        )

    def mode(self) -> jax.Array:
        return self.mean
