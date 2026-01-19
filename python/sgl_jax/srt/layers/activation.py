import jax
from flax import nnx

from sgl_jax.srt.utils.profiling_utils import named_scope


class GeluAndMul(nnx.Module):
    def __init__(self, approximate: str = "tanh"):
        self.approximate = approximate

    @named_scope
    def __call__(self, gate: jax.Array, up: jax.Array):
        if self.approximate == "tanh":
            gelu = jax.nn.gelu(gate, approximate=True)
        else:
            gelu = jax.nn.gelu(gate, approximate=False)
        out = gelu * up
        return out, None
