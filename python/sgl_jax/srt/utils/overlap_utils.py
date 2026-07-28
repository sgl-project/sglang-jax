from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

RELAY_ID_SPEC = P("data", None)


class RelayBuffers(NamedTuple):
    next_token_ids: jax.Array


def create_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
) -> RelayBuffers:
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    return RelayBuffers(
        next_token_ids=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            sharding,
        )
    )


def update_relay_buffers(
    buffers: RelayBuffers,
    req_pool_indices,
    next_token_ids,
    *,
    dp_size: int,
    output_sharding,
) -> RelayBuffers:
    per_dp_bs = req_pool_indices.shape[0] // dp_size
    req_pool_indices = req_pool_indices.reshape((dp_size, per_dp_bs))
    next_token_ids = next_token_ids.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        req_pool_indices >= 0,
        req_pool_indices,
        jnp.full_like(req_pool_indices, buffers.next_token_ids.shape[1]),
    )
    return RelayBuffers(
        next_token_ids=buffers.next_token_ids.at[dp_indices, scatter_indices].set(
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
        buffers.next_token_ids.at[dp_indices, indices_2d]
        .get(out_sharding=output_sharding)
        .reshape(indices.shape)
    )
    return values


def resolve_relay_inputs(
    buffers: RelayBuffers,
    indices,
    valid_mask,
    input_ids,
    *,
    dp_size: int,
    relay_sharding,
    output_sharding,
) -> jax.Array:
    relay_ids = gather_relay_buffers(
        buffers,
        indices,
        dp_size=dp_size,
        output_sharding=relay_sharding,
    )
    relay_ids = jax.sharding.reshard(relay_ids, output_sharding)
    return jnp.where(valid_mask, relay_ids, input_ids)


def resolve_decode_relay_inputs(
    buffers: RelayBuffers,
    req_pool_indices,
    input_ids,
    *,
    dp_size: int,
    relay_sharding,
    output_sharding,
) -> jax.Array:
    valid_mask = req_pool_indices >= 0
    safe_indices = jnp.where(valid_mask, req_pool_indices, 0)
    return resolve_relay_inputs(
        buffers,
        safe_indices,
        valid_mask,
        input_ids,
        dp_size=dp_size,
        relay_sharding=relay_sharding,
        output_sharding=output_sharding,
    )
