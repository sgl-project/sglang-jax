from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

RELAY_STATE_SPEC = P("data", None, None)
RELAY_ID_SPEC = P("data", None)


class SpecRelayBuffers(NamedTuple):
    topk_index: jax.Array
    hidden_states: jax.Array
    verified_id: jax.Array
    new_seq_lens: jax.Array


class DFlashRelayBuffers(NamedTuple):
    verified_id: jax.Array
    new_seq_lens: jax.Array


@dataclass(frozen=True)
class RelayBatchPlan:
    """Host-side request indices used to read and publish one relay round."""

    future_indices: np.ndarray
    padded_indices: np.ndarray
    valid_mask: np.ndarray


def build_relay_batch_plan(model_worker_batch) -> RelayBatchPlan:
    """Build compact state references and DP-padded safe scatter indices."""
    req_pool_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)
    selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
    valid_mask = make_dp_valid_mask(
        model_worker_batch.real_bs_per_dp,
        total_bs=req_pool_indices.shape[0],
        per_dp_bs=model_worker_batch.per_dp_bs_size,
    )
    return RelayBatchPlan(
        future_indices=req_pool_indices[selector],
        padded_indices=np.where(valid_mask, req_pool_indices, 0),
        valid_mask=valid_mask,
    )


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


def scatter_relay_buffers(
    buffers,
    future_indices,
    valid_mask,
    payload,
    *,
    dp_size: int,
):
    """Scatter a relay payload without writing DP padding rows."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    first_buffer = jax.tree_util.tree_leaves(buffers)[0]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, first_buffer.shape[1]),
    )

    def _scatter_leaf(buffer, value):
        value = value.reshape((dp_size, per_dp_bs) + value.shape[1:])
        sharding = jax.typeof(buffer).sharding
        if isinstance(sharding, NamedSharding) and not sharding.mesh.empty:
            return buffer.at[dp_indices, scatter_indices].set(
                value,
                mode="drop",
                out_sharding=sharding.spec,
            )
        return buffer.at[dp_indices, scatter_indices].set(value, mode="drop")

    return jax.tree.map(_scatter_leaf, buffers, payload)


def gather_relay_buffers(buffers, future_indices, *, dp_size: int):
    """Gather a relay payload and restore flat ``P("data", ...)`` sharding."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    flat_sharding = jax.typeof(future_indices).sharding

    def _gather_leaf(buffer):
        buffer_sharding = jax.typeof(buffer).sharding
        if isinstance(buffer_sharding, NamedSharding) and not buffer_sharding.mesh.empty:
            gathered = buffer.at[dp_indices, indices].get(out_sharding=buffer_sharding.spec)
        else:
            gathered = buffer.at[dp_indices, indices].get()
        gathered = gathered.reshape(future_indices.shape + buffer.shape[2:])
        if isinstance(flat_sharding, NamedSharding) and not flat_sharding.mesh.empty:
            flat_spec = P("data", *(None for _ in range(gathered.ndim - 1)))
            gathered = jax.sharding.reshard(
                gathered,
                NamedSharding(flat_sharding.mesh, flat_spec),
            )
        return gathered

    return jax.tree.map(_gather_leaf, buffers)


def make_dp_valid_mask(real_bs_per_dp, *, total_bs: int, per_dp_bs: int) -> np.ndarray:
    mask = np.zeros((total_bs,), dtype=np.bool_)
    for dp_rank, real_bs in enumerate(real_bs_per_dp):
        if real_bs:
            start = dp_rank * per_dp_bs
            mask[start : start + int(real_bs)] = True
    return mask
