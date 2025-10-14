import jax
import jax.numpy as jnp
from flax import nnx


class GeluAndMul(nnx.Module):
    def __init__(self, approximate: str = "tanh"):
        self.approximate = approximate

    def __call__(self, gate_up: jax.Array):
        gate, up = jnp.split(gate_up, 2, axis=-1)
        if self.approximate == "tanh":
            gelu = jax.nn.gelu(gate, approximate=True)
        else:
            gelu = jax.nn.gelu(gate, approximate=False)
        out = gelu * up
        return out, None 