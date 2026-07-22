from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

RELAY_ID_SPEC = P("data", None)


class RelayBuffers(NamedTuple):
    next_token_id: jax.Array


def create_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
) -> RelayBuffers:
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    return RelayBuffers(
        next_token_id=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            sharding,
        )
    )


def update_relay_buffers(
    buffers: RelayBuffers,
    indices,
    valid_mask,
    next_token_ids,
    *,
    dp_size: int,
    output_sharding,
) -> RelayBuffers:
    per_dp_bs = indices.shape[0] // dp_size
    indices = indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    next_token_ids = next_token_ids.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.next_token_id.shape[1]),
    )
    return RelayBuffers(
        next_token_id=buffers.next_token_id.at[dp_indices, scatter_indices].set(
            next_token_ids,
            mode="drop",
            out_sharding=output_sharding,
        )
    )


def gather_relay_buffers(
    buffers: RelayBuffers,
    indices,
    *,
    dp_size: int,
    output_sharding,
) -> jax.Array:
    per_dp_bs = indices.shape[0] // dp_size
    indices_2d = indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    values = (
        buffers.next_token_id.at[dp_indices, indices_2d]
        .get(out_sharding=output_sharding)
        .reshape(indices.shape)
    )
    return values
