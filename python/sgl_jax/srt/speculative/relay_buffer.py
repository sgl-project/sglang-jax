from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

RELAY_STATE_SPEC = P("data", None, None)
RELAY_ID_SPEC = P("data", None)


def _array_sharding(value):
    """Return sharding for either a concrete JAX array or a JIT tracer.

    Relay helpers are intentionally usable in CPU state tests *and* inside
    cached device programs.  ``Array.sharding`` is available in the former,
    whereas a tracer exposes the same information through ``jax.typeof``.
    """
    sharding = getattr(value, "sharding", None)
    return sharding if sharding is not None else jax.typeof(value).sharding


class SpecRelayBuffers(NamedTuple):
    topk_index: jax.Array
    hidden_states: jax.Array
    verified_id: jax.Array
    new_seq_lens: jax.Array


class SpecSeedRelayBuffers(NamedTuple):
    """Request-indexed single-token proposal state shared between spec rounds.

    This is intentionally smaller than :class:`SpecRelayBuffers`: top-1
    speculative algorithms need one verified token, one proposal token, and
    one hidden state per request.  A Frozen-KV row can be either a normal
    prefill-origin proposal or a target-verify-origin seed; ``is_target_seed``
    identifies the latter.  The scheduler owns the request-to-slot mapping;
    this buffer only preserves device-resident values across that
    variable-size scheduler boundary.
    """

    token_ids: jax.Array
    draft_token_ids: jax.Array
    hidden_states: jax.Array
    is_target_seed: jax.Array


class DFlashRelayBuffers(NamedTuple):
    verified_id: jax.Array
    new_seq_lens: jax.Array


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
    token_sharding = NamedSharding(mesh, RELAY_STATE_SPEC)
    hidden_sharding = NamedSharding(mesh, RELAY_STATE_SPEC)
    id_sharding = NamedSharding(mesh, RELAY_ID_SPEC)
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
        new_seq_lens=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            id_sharding,
        ),
    )


def create_spec_seed_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
    hidden_size: int,
    hidden_dtype,
) -> SpecSeedRelayBuffers:
    """Create DP-local request-indexed buffers for token/hidden seed state.

    A fixed request-pool index is stable while the scheduler merges, filters,
    and repads a live batch.  Keeping the value storage keyed by that index
    avoids a host materialization merely to reshape a variable-sized batch.
    """
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    hidden_sharding = NamedSharding(mesh, RELAY_STATE_SPEC)
    id_sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    return SpecSeedRelayBuffers(
        token_ids=jax.device_put(jnp.zeros((dp_size, capacity), dtype=jnp.int32), id_sharding),
        draft_token_ids=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32), id_sharding
        ),
        hidden_states=jax.device_put(
            jnp.zeros((dp_size, capacity, hidden_size), dtype=hidden_dtype),
            hidden_sharding,
        ),
        is_target_seed=jax.device_put(jnp.zeros((dp_size, capacity), dtype=bool), id_sharding),
    )


def create_dflash_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
) -> DFlashRelayBuffers:
    """Create the minimal req-indexed state needed by DFlash overlap."""
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    shape = (dp_size, capacity)
    return DFlashRelayBuffers(
        verified_id=jax.device_put(jnp.zeros(shape, dtype=jnp.int32), sharding),
        new_seq_lens=jax.device_put(jnp.zeros(shape, dtype=jnp.int32), sharding),
    )


def update_spec_relay_buffers(
    buffers: SpecRelayBuffers,
    future_indices,
    valid_mask,
    topk_index,
    hidden_states,
    verified_id,
    new_seq_lens,
    *,
    dp_size: int,
) -> SpecRelayBuffers:
    """Write DP-padded draft state into relay buffers without touching padded rows."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.topk_index.shape[1]),
    )

    topk_index = topk_index.reshape((dp_size, per_dp_bs) + topk_index.shape[1:])
    hidden_states = hidden_states.reshape((dp_size, per_dp_bs) + hidden_states.shape[1:])
    verified_id = verified_id.reshape((dp_size, per_dp_bs))
    new_seq_lens = new_seq_lens.reshape((dp_size, per_dp_bs))

    return SpecRelayBuffers(
        topk_index=buffers.topk_index.at[dp_indices, scatter_indices].set(
            topk_index,
            mode="drop",
            out_sharding=RELAY_STATE_SPEC,
        ),
        hidden_states=buffers.hidden_states.at[dp_indices, scatter_indices].set(
            hidden_states,
            mode="drop",
            out_sharding=RELAY_STATE_SPEC,
        ),
        verified_id=buffers.verified_id.at[dp_indices, scatter_indices].set(
            verified_id,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
        new_seq_lens=buffers.new_seq_lens.at[dp_indices, scatter_indices].set(
            new_seq_lens,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
    )


def update_spec_seed_relay_buffers(
    buffers: SpecSeedRelayBuffers,
    future_indices,
    valid_mask,
    token_ids,
    draft_token_ids,
    hidden_states,
    is_target_seed,
    *,
    dp_size: int,
) -> SpecSeedRelayBuffers:
    """Publish DP-padded target seeds without touching invalid padded rows."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.token_ids.shape[1]),
    )
    token_ids = token_ids.reshape((dp_size, per_dp_bs))
    draft_token_ids = draft_token_ids.reshape((dp_size, per_dp_bs))
    hidden_states = hidden_states.reshape((dp_size, per_dp_bs) + hidden_states.shape[1:])
    is_target_seed = is_target_seed.reshape((dp_size, per_dp_bs))
    return SpecSeedRelayBuffers(
        token_ids=buffers.token_ids.at[dp_indices, scatter_indices].set(
            token_ids,
            mode="drop",
            out_sharding=_array_sharding(buffers.token_ids),
        ),
        draft_token_ids=buffers.draft_token_ids.at[dp_indices, scatter_indices].set(
            draft_token_ids,
            mode="drop",
            out_sharding=_array_sharding(buffers.draft_token_ids),
        ),
        hidden_states=buffers.hidden_states.at[dp_indices, scatter_indices].set(
            hidden_states,
            mode="drop",
            out_sharding=_array_sharding(buffers.hidden_states),
        ),
        is_target_seed=buffers.is_target_seed.at[dp_indices, scatter_indices].set(
            is_target_seed,
            mode="drop",
            out_sharding=_array_sharding(buffers.is_target_seed),
        ),
    )


def update_dflash_relay_buffers(
    buffers: DFlashRelayBuffers,
    future_indices,
    valid_mask,
    verified_id,
    new_seq_lens,
    *,
    dp_size: int,
) -> DFlashRelayBuffers:
    """Publish one DP-padded DFlash round without writing padded slots."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.verified_id.shape[1]),
    )
    verified_id = verified_id.reshape((dp_size, per_dp_bs))
    new_seq_lens = new_seq_lens.reshape((dp_size, per_dp_bs))
    return DFlashRelayBuffers(
        verified_id=buffers.verified_id.at[dp_indices, scatter_indices].set(
            verified_id,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
        new_seq_lens=buffers.new_seq_lens.at[dp_indices, scatter_indices].set(
            new_seq_lens,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
    )


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
    topk_index = (
        buffers.topk_index.at[dp_indices, indices]
        .get(out_sharding=RELAY_STATE_SPEC)
        .reshape(future_indices.shape + buffers.topk_index.shape[2:])
    )
    hidden_states = (
        buffers.hidden_states.at[dp_indices, indices]
        .get(out_sharding=RELAY_STATE_SPEC)
        .reshape(future_indices.shape + buffers.hidden_states.shape[2:])
    )
    verified_id = (
        buffers.verified_id.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    new_seq_lens = (
        buffers.new_seq_lens.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    flat_sharding = jax.typeof(future_indices).sharding
    if isinstance(flat_sharding, NamedSharding) and not flat_sharding.mesh.empty:
        state_sharding = NamedSharding(flat_sharding.mesh, P("data", None))
        topk_index = jax.sharding.reshard(topk_index, state_sharding)
        hidden_states = jax.sharding.reshard(hidden_states, state_sharding)
        verified_id = jax.sharding.reshard(verified_id, flat_sharding)
        new_seq_lens = jax.sharding.reshard(new_seq_lens, flat_sharding)
    return topk_index, hidden_states, verified_id, new_seq_lens


def gather_spec_seed_relay_buffers(
    buffers: SpecSeedRelayBuffers,
    future_indices,
    *,
    dp_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Gather one-token proposal state into the next DP-padded draft batch."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    token_ids = (
        buffers.token_ids.at[dp_indices, indices]
        .get(out_sharding=_array_sharding(buffers.token_ids))
        .reshape(future_indices.shape)
    )
    draft_token_ids = (
        buffers.draft_token_ids.at[dp_indices, indices]
        .get(out_sharding=_array_sharding(buffers.draft_token_ids))
        .reshape(future_indices.shape)
    )
    hidden_states = (
        buffers.hidden_states.at[dp_indices, indices]
        .get(out_sharding=_array_sharding(buffers.hidden_states))
        .reshape(future_indices.shape + buffers.hidden_states.shape[2:])
    )
    is_target_seed = (
        buffers.is_target_seed.at[dp_indices, indices]
        .get(out_sharding=_array_sharding(buffers.is_target_seed))
        .reshape(future_indices.shape)
    )
    return token_ids, draft_token_ids, hidden_states, is_target_seed


def gather_dflash_relay_buffers(
    buffers: DFlashRelayBuffers,
    future_indices,
    *,
    dp_size: int,
):
    """Gather the DFlash seed token and logical length for the next round."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    verified_id = (
        buffers.verified_id.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    new_seq_lens = (
        buffers.new_seq_lens.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    flat_sharding = jax.typeof(future_indices).sharding
    if isinstance(flat_sharding, NamedSharding) and not flat_sharding.mesh.empty:
        verified_id = jax.sharding.reshard(verified_id, flat_sharding)
        new_seq_lens = jax.sharding.reshard(new_seq_lens, flat_sharding)
    return verified_id, new_seq_lens


def make_dp_valid_mask(real_bs_per_dp, *, total_bs: int, per_dp_bs: int) -> np.ndarray:
    mask = np.zeros((total_bs,), dtype=np.bool_)
    for dp_rank, real_bs in enumerate(real_bs_per_dp):
        if real_bs:
            start = dp_rank * per_dp_bs
            mask[start : start + int(real_bs)] = True
    return mask
