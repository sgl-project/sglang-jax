# from typing import Any

# import flax.nnx as nnx
# import jax
# from jax import numpy as jnp
# from jax.sharding import PartitionSpec as P

# from sgl_jax.srt.layers.linear import LinearBase


# class ReplicatedLinear(LinearBase):
#     """Replicated linear layer.

#     Args:
#         input_size: input dimension of the linear layer.
#         output_size: output dimension of the linear layer.
#         mesh: The mesh to use for sharding.
#         use_bias: If true, add bias.
#         skip_bias_add: If true, skip adding bias but instead return it.
#         params_dtype: Data type for the parameters.
#         quant_config: Quantization configure.
#     """

#     def __init__(
#         self,
#         input_size: int,
#         output_size: int,
#         use_bias: bool = True,
#         skip_bias_add: bool = False,
#         params_dtype: jnp.dtype | None = jnp.bfloat16,
#         quant_config: Any | None = None,
#     ):
#         super().__init__(
#             input_size=input_size,
#             output_size=output_size,
#             # FIXME(pc) use true mesh here
#             mesh=None,
#             use_bias=use_bias,
#             skip_bias_add=skip_bias_add,
#             params_dtype=params_dtype,
#             kernel_axes=(None, None),
#         )
#         self.quant_config = quant_config
#         if use_bias:
#             self.bias = nnx.Param(
#                 jax.random.normal(
#                     jax.random.PRNGKey(0),
#                     (output_size,),
#                     dtype=params_dtype,
#                     out_sharding=P(
#                         None,
#                     ),
#                 ),
#             )

#     def __call__(self, inputs: jax.Array) -> tuple[jax.Array, jax.Array | None]:
#         output = jnp.dot(inputs, self.weight[...])
#         if self.skip_bias_add:
#             return output, self.bias[...] if hasattr(self, "bias") else None
#         if hasattr(self, "bias"):
#             output = output + self.bias[...]
#         return output, self.bias[...] if hasattr(self, "bias") else None
