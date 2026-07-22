import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils.overlap_utils import (
    create_relay_buffers,
    gather_relay_buffers,
    update_relay_buffers,
)


def test_relay_buffer_updates_valid_request_slots():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    pool = SimpleNamespace(req_to_token=np.zeros((8, 16), dtype=np.int32))
    sharding = NamedSharding(mesh, P("data"))
    relay_sharding = NamedSharding(mesh, P("data", None))
    indices = jax.device_put(jnp.array([2, 5, 0, 0]), sharding)
    valid = jax.device_put(jnp.array([True, True, False, False]), sharding)
    tokens = jax.device_put(jnp.array([11, 22, 99, 99]), sharding)

    buffers = create_relay_buffers(mesh, pool, dp_size=1)
    buffers = jax.jit(
        lambda b, i, m, t: update_relay_buffers(
            b,
            i,
            m,
            t,
            dp_size=1,
            output_sharding=relay_sharding,
        )
    )(buffers, indices, valid, tokens)
    actual = jax.jit(
        lambda b, i: gather_relay_buffers(
            b,
            i,
            dp_size=1,
            output_sharding=relay_sharding,
        )
    )(buffers, indices)

    np.testing.assert_array_equal(np.asarray(actual), [11, 22, 0, 0])
