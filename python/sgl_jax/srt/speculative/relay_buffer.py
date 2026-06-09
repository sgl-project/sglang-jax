from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


class SpecRelayBuffers(NamedTuple):
    topk_index: jax.Array
    hidden_states: jax.Array
    verified_id: jax.Array


def create_spec_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
    num_steps: int,
    hidden_size: int,
    hidden_dtype,
) -> SpecRelayBuffers:
    """Create DP-local req-indexed buffers for cross-batch draft state relay."""
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    token_sharding = NamedSharding(mesh, P("data", None, None))
    hidden_sharding = NamedSharding(mesh, P("data", None, None))
    id_sharding = NamedSharding(mesh, P("data", None))
    return SpecRelayBuffers(
        topk_index=jax.device_put(
            jnp.zeros((dp_size, capacity, num_steps), dtype=jnp.int32),
            token_sharding,
        ),
        hidden_states=jax.device_put(
            jnp.zeros((dp_size, capacity, hidden_size), dtype=hidden_dtype),
            hidden_sharding,
        ),
        verified_id=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            id_sharding,
        ),
    )


@partial(jax.jit, static_argnames=("dp_size",))
def update_spec_relay_buffers(
    buffers: SpecRelayBuffers,
    future_indices,
    valid_mask,
    topk_index,
    hidden_states,
    verified_id,
    *,
    dp_size: int,
) -> SpecRelayBuffers:
    """Write DP-padded draft state into relay buffers without touching padded rows."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]

    topk_index = topk_index.reshape((dp_size, per_dp_bs) + topk_index.shape[1:])
    hidden_states = hidden_states.reshape((dp_size, per_dp_bs) + hidden_states.shape[1:])
    verified_id = verified_id.reshape((dp_size, per_dp_bs))

    old_topk_index = buffers.topk_index[dp_indices, indices]
    old_hidden_states = buffers.hidden_states[dp_indices, indices]
    old_verified_id = buffers.verified_id[dp_indices, indices]

    return SpecRelayBuffers(
        topk_index=buffers.topk_index.at[dp_indices, indices].set(
            jnp.where(valid[..., None], topk_index, old_topk_index)
        ),
        hidden_states=buffers.hidden_states.at[dp_indices, indices].set(
            jnp.where(valid[..., None], hidden_states, old_hidden_states)
        ),
        verified_id=buffers.verified_id.at[dp_indices, indices].set(
            jnp.where(valid, verified_id, old_verified_id)
        ),
    )


@partial(jax.jit, static_argnames=("dp_size",))
def gather_spec_relay_buffers(
    buffers: SpecRelayBuffers,
    future_indices,
    *,
    dp_size: int,
):
    """Gather DP-padded draft state for the next batch."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]

    return (
        buffers.topk_index[dp_indices, indices].reshape(
            future_indices.shape + buffers.topk_index.shape[2:]
        ),
        buffers.hidden_states[dp_indices, indices].reshape(
            future_indices.shape + buffers.hidden_states.shape[2:]
        ),
        buffers.verified_id[dp_indices, indices].reshape(future_indices.shape),
    )


def scatter_future_indices_to_dp_slots(
    per_dp_indices,
    *,
    total_bs: int,
    per_dp_bs: int,
) -> np.ndarray:
    out = np.zeros((total_bs,), dtype=np.int32)
    for dp_rank, indices in enumerate(per_dp_indices):
        start = dp_rank * per_dp_bs
        n = len(indices)
        if n:
            out[start : start + n] = np.asarray(indices, dtype=np.int32)
    return out


def make_dp_valid_mask(real_bs_per_dp, *, total_bs: int, per_dp_bs: int) -> np.ndarray:
    mask = np.zeros((total_bs,), dtype=np.bool_)
    for dp_rank, real_bs in enumerate(real_bs_per_dp):
        if real_bs:
            start = dp_rank * per_dp_bs
            mask[start : start + int(real_bs)] = True
    return mask
