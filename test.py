import jax, jax.numpy as jnp
import numpy as np
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.update("jax_platform_name", "cpu")
devs = jax.devices()
mesh = Mesh(devs, axis_names=("tensor",))

def f(x, w1, w2):
    y = x @ w1
    y = y @ w2
    return jax.lax.psum(y, axis_name="tensor")

f_pjit = pjit(
    f,
    in_shardings=(
        NamedSharding(mesh, P(None, None)),
        NamedSharding(mesh, P("tensor", None)),
        NamedSharding(mesh, P("tensor", None)),
    ),
    out_shardings=NamedSharding(mesh, P("tensor", None)),
)

x  = jnp.asarray(np.random.randn(1, 12), dtype=jnp.float32)
w1 = jnp.asarray(np.random.randn(12, 4),  dtype=jnp.float32)
w2 = jnp.asarray(np.random.randn(4,  4),  dtype=jnp.float32)

with mesh:
    y = f_pjit(x, w1, w2)
    y.block_until_ready()
    print("OK:", y.shape)
