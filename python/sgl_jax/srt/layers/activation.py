from functools import partial

import jax
from flax import nnx

from sgl_jax.srt.utils.profiling_utils import named_scope

# Keep the exact/approximate GELU distinction used by HF checkpoints.
ACT2FN = {
    "gelu": partial(jax.nn.gelu, approximate=False),
    "gelu_new": partial(jax.nn.gelu, approximate=True),
    "gelu_pytorch_tanh": partial(jax.nn.gelu, approximate=True),
    "relu": jax.nn.relu,
    "silu": jax.nn.silu,
    "swish": jax.nn.silu,
    "quick_gelu": lambda x: x * jax.nn.sigmoid(1.702 * x),
    "tanh": jax.nn.tanh,
}


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
