import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

devices = np.array(jax.devices())
device_mesh = devices.reshape(2, 2)

mesh = Mesh(device_mesh, axis_names=("data", "model"))

arr = jnp.arange(16).reshape(4, 4)

sharded_arr = jax.device_put(arr, NamedSharding(mesh, P("data", "model")))


@jax.jit
def func(arr: jax.Array):
    return sharded_arr.at[:1, :].set(jnp.array([0, 0, 0, 0]))


func(sharded_arr)

res = func(sharded_arr)

print(f"{sharded_arr=}")
print(f"{res=}")
