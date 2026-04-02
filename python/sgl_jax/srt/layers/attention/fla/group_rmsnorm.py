import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype
from jax import lax

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
    ):
        if hidden_size % num_groups != 0:
            raise ValueError("hidden_size must be divisible by num_groups")

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = hidden_size // num_groups
        self.epsilon = epsilon
        self.name = scope_name
        self.weight = nnx.Param(jnp.ones(hidden_size, dtype=param_dtype))

    @named_scope
    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        orig_dtype = hidden_states.dtype
        orig_shape = hidden_states.shape

        hidden_states = hidden_states.reshape(
            *orig_shape[:-1], self.num_groups, self.group_size
        ).astype(jnp.float32)
        variance = jnp.mean(lax.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * lax.rsqrt(variance + self.epsilon)

        weight = jnp.asarray(self.weight[...], jnp.float32).reshape(
            self.num_groups, self.group_size
        )
        hidden_states = (weight * hidden_states).reshape(orig_shape).astype(orig_dtype)

        return hidden_states
