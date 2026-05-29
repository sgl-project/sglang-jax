import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.linear import LinearBase


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
        output_dim: int | None = None,
        bias: bool = True,
        act_type: str = "gelu_pytorch_tanh",
        dtype: jnp.dtype | None = jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
        prefix: str = "",
    ):
        self.fc_in = LinearBase(
            input_size=input_dim,
            output_size=mlp_hidden_dim,
            use_bias=bias,
            params_dtype=dtype,
            mesh=mesh,
            kernel_axes=(None, "tensor"),
        )

        self.act = get_act_fn(act_type)
        if output_dim is None:
            output_dim = input_dim
        self.fc_out = LinearBase(
            input_size=mlp_hidden_dim,
            output_size=output_dim,
            use_bias=bias,
            params_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x, _ = self.fc_in(x)
        x = self.act(x)
        x, _ = self.fc_out(x)
        return x


def _resolve_rngs(rngs: nnx.Rngs | None) -> nnx.Rngs:
    return rngs or nnx.Rngs(0)


class FeedForward(nnx.Module):
    """
    Feed-forward block adapted for FLUX-style DiT blocks.

    This keeps the current FLUX interface, including dropout arguments, while
    remaining deterministic in the current inference/parity-test paths.
    """

    def __init__(
        self,
        dim: int,
        mesh: jax.sharding.Mesh,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        final_dropout: bool = False,
        inner_dim: int | None = None,
        bias: bool = True,
        params_dtype: jnp.dtype | None = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        _rngs = _resolve_rngs(rngs)
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out

        if activation_fn == "gelu-approximate":
            act_type = "gelu_pytorch_tanh"
        elif activation_fn == "gelu":
            act_type = "gelu"
        else:
            raise ValueError(f"Unsupported activation_fn {activation_fn!r} for FeedForward.")

        self.net = nnx.List(
            [
                LinearBase(
                    input_size=dim,
                    output_size=inner_dim,
                    use_bias=bias,
                    mesh=mesh,
                    params_dtype=params_dtype,
                    kernel_axes=(None, "tensor"),
                ),
                nnx.Dropout(dropout, rngs=_rngs),
                LinearBase(
                    input_size=inner_dim,
                    output_size=dim_out,
                    use_bias=bias,
                    mesh=mesh,
                    params_dtype=params_dtype,
                    kernel_axes=("tensor", None),
                ),
            ]
        )
        self.act = get_act_fn(act_type)
        if final_dropout:
            self.net.append(nnx.Dropout(dropout, rngs=_rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        x, _ = self.net[0](x)
        x = self.act(x)
        x = self.net[1](x, deterministic=True)
        x, _ = self.net[2](x)
        if len(self.net) == 4:
            x = self.net[3](x, deterministic=True)
        return x
