import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax.sharding import NamedSharding, PartitionSpec
from jax import ShapeDtypeStruct

def materialize_abstract_state(model: nnx.Module, mesh: jax.sharding.Mesh | None = None):
    """Recursively replaces all ShapeDtypeStructs in the model's state with actual jax.Array."""
    def _replace_abstract(x):
        if isinstance(x, ShapeDtypeStruct):
            pspec = PartitionSpec()
            if hasattr(x, "sharding") and x.sharding is not None and hasattr(x.sharding, "spec"):
                pspec = x.sharding.spec
            if mesh:
                concrete_sharding = NamedSharding(mesh, pspec)
                return jax.device_put(jnp.zeros(x.shape, x.dtype), concrete_sharding)
            return jnp.zeros(x.shape, x.dtype)
        return x

    state = nnx.state(model)
    concrete_state = jax.tree_util.tree_map(_replace_abstract, state)
    nnx.update(model, concrete_state)