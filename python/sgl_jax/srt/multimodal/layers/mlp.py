import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.multimodal.layers.linear import ReplicatedLinear


def get_act_fn(act_type: str):
    if act_type == "gelu_pytorch_tanh":
        return lambda x: jax.nn.gelu(x, approximate=True)
    elif act_type == "silu":
        return jax.nn.silu
    elif act_type == "gelu":
        return jax.nn.gelu
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


class MLP(nnx.Module):
    """
    MLP for DiT blocks, NO gated linear units
    """

    def __init__(
        self,
        input_dim: int,
        mlp_hidden_dim: int,
        mesh: jax.sharding.Mesh,
        output_dim: int | None = None,
        bias: bool = True,
        act_type: str = "gelu_pytorch_tanh",
        dtype: jnp.dtype | None = jnp.bfloat16,
        prefix: str = "",
    ):
        self.fc_in = ReplicatedLinear(
            input_dim,
            mlp_hidden_dim,
            mesh=mesh,
            use_bias=bias,
            params_dtype=dtype,
        )

        self.act = get_act_fn(act_type)
        if output_dim is None:
            output_dim = input_dim
        self.fc_out = ReplicatedLinear(
            mlp_hidden_dim, output_dim, mesh=mesh, use_bias=bias, params_dtype=dtype
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x, _ = self.fc_in(x)
        x = self.act(x)
        x, _ = self.fc_out(x)
        return x
