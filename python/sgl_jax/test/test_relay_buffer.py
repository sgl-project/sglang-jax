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
    resolve_decode_relay_inputs,
    resolve_relay_inputs,
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
    indices = jax.device_put(jnp.array([2, 5, -1, -1]), sharding)
    valid = indices >= 0
    tokens = jax.device_put(jnp.array([11, 22, 99, 99]), sharding)
    input_ids = jax.device_put(jnp.array([1, 2, 3, 4]), sharding)

    buffers = create_relay_buffers(mesh, pool, dp_size=1)
    buffers = jax.jit(
        lambda b, i, t: update_relay_buffers(
            b,
            i,
            t,
            dp_size=1,
            output_sharding=relay_sharding,
        )
    )(buffers, indices, tokens)
    np.testing.assert_array_equal(
        np.asarray(buffers.next_token_ids)[0, [0, 2, 5]],
        [0, 11, 22],
    )
    actual = jax.jit(
        lambda b, i, m, x: resolve_relay_inputs(
            b,
            i,
            m,
            x,
            dp_size=1,
            relay_sharding=relay_sharding,
            output_sharding=sharding,
        ),
        out_shardings=sharding,
    )(buffers, indices, valid, input_ids)

    np.testing.assert_array_equal(np.asarray(actual), [11, 22, 3, 4])

    decode_actual = jax.jit(
        lambda b, i, x: resolve_decode_relay_inputs(
            b,
            i,
            x,
            dp_size=1,
            relay_sharding=relay_sharding,
            output_sharding=sharding,
        ),
        out_shardings=sharding,
    )(buffers, indices, input_ids)
    np.testing.assert_array_equal(np.asarray(decode_actual), [11, 22, 3, 4])
