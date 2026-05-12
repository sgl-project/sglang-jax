from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils.profiling_utils import named_scope


class GroupRMSNorm(nnx.Module):
    """Group RMS normalization.

    Splits the last dimension into `num_groups` groups and applies
    RMSNorm independently within each group.
    """

    def __init__(
        self,
        hidden_size: int,
        num_groups: int = 8,
        epsilon: float = 1e-6,
        param_dtype: Dtype = jnp.float32,
        scope_name: str = "group_rms_norm",
        kernel_axes: Sequence[str | None] | None = None,
        mesh: Mesh | None = None,
    ):
        if hidden_size % num_groups != 0:
            raise ValueError("hidden_size must be divisible by num_groups")

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = hidden_size // num_groups
        self.epsilon = epsilon
        self.name = scope_name
        self.mesh = mesh

        tp_size = mesh.shape.get("tensor", 1) if mesh is not None else 1
        self._shard_groups = num_groups % tp_size == 0

        if not self._shard_groups:
            kernel_axes = (None,)

        self.weight = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (hidden_size,),
                dtype=param_dtype,
                out_sharding=P(*kernel_axes),
            ),
        )

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        orig_dtype = hidden_states.dtype
        orig_shape = hidden_states.shape

        if self._shard_groups:
            group_spec = P("data", "tensor", None)
            weight_spec = P("tensor", None)
        else:
            group_spec = P("data", None, None)
            weight_spec = P(None, None)

        hidden_states = hidden_states.reshape(
            orig_shape[0],
            self.num_groups,
            self.group_size,
            out_sharding=NamedSharding(self.mesh, group_spec),
        ).astype(jnp.float32)
        variance = jnp.mean(lax.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * lax.rsqrt(variance + self.epsilon)

        nw = self.weight.value.reshape(
            self.num_groups,
            self.group_size,
            out_sharding=NamedSharding(self.mesh, weight_spec),
        ).astype(jnp.float32)
        hidden_states = (nw * hidden_states).astype(orig_dtype)

        return hidden_states.reshape(
            orig_shape,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor")),
        )
