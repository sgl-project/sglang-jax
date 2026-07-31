import logging

import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> str | None:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or the max context length. Truncated. len(origin_input_ids)=%s, max_req_input_len=%s",
                len(req.origin_input_ids),
                max_req_input_len,
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            return error_msg

    return None


@jax.jit(static_argnames=("mesh"))
def resolve_future_token_ids(input_ids, future_token_ids_map, mesh):
    input_ids_global = jax.sharding.reshard(input_ids, NamedSharding(mesh, P()))
    input_ids_global = jnp.where(
        input_ids_global < 0,
        future_token_ids_map[jnp.clip(-input_ids_global, min=0)],
        input_ids_global,
    )
    return jax.sharding.reshard(input_ids_global, NamedSharding(mesh, P("data")))


def future_slot_indices(seq_lens_np, req_pool_indices_np, map_size):
    """Per-request future-map slots: real rows -> req_pool_idx + 1 (slot 0 is
    reserved so 0 stays a valid "no placeholder" input id), padding rows ->
    map_size (out of bounds, dropped by the scatter)."""
    import numpy as np

    return np.where(
        seq_lens_np > 0,
        req_pool_indices_np.astype(np.int32) + 1,
        np.int32(map_size),
    ).astype(np.int32)


@jax.jit(static_argnames=("mesh"))
def set_future_token_ids(future_token_ids_map, slot_indices, next_token_ids, mesh):
    """Write each request's pending next token at its req-pool slot.

    Slot addressing replaces the previous ring-buffer cursor: the cursor
    advanced by the PADDED batch size (padded seq_lens), so a burst of
    prefill batches wrapped the 3x ring and overwrote outstanding
    placeholders before their first decode resolved them (silent first-token
    corruption that grows with concurrency). A per-request slot cannot be
    overwritten by another request; padding rows scatter out of bounds and
    are dropped.
    """
    next_token_ids_global = jax.sharding.reshard(next_token_ids, NamedSharding(mesh, P()))
    slot_indices_global = jax.sharding.reshard(slot_indices, NamedSharding(mesh, P()))
    return future_token_ids_map.at[slot_indices_global].set(next_token_ids_global, mode="drop")
