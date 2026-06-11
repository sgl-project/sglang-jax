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


def validate_pd_no_chunked_prefill(
    req: Req, disaggregation_mode: str, chunked_prefill_size: int | None
) -> str | None:
    """Reject PD requests whose prompt would be chunked.

    PD disaggregation does not support chunked prefill: process_prefill_chunk
    replaces process_batch_result for PD batches, so a chunked req never
    advances past its first chunk and leaks KV until OOM. Guard against it
    until chunk-prefill-transfer lands.

    Returns an error message if the request must be rejected, else None.
    """
    if disaggregation_mode == "null":
        return None
    if not chunked_prefill_size or chunked_prefill_size <= 0:
        return None
    if len(req.origin_input_ids) > chunked_prefill_size:
        return (
            f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
            f"chunked_prefill_size ({chunked_prefill_size} tokens). PD "
            f"disaggregation does not support chunked prefill; raise "
            f"--chunked-prefill-size or use a shorter input."
        )
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


@jax.jit(static_argnames=("mesh"))
def set_future_token_ids(future_token_ids_map, future_token_ids_ct, next_token_ids, mesh):
    next_token_ids_global = jax.sharding.reshard(next_token_ids, NamedSharding(mesh, P()))
    start_indices = (future_token_ids_ct + 1,)
    return jax.lax.dynamic_update_slice(future_token_ids_map, next_token_ids_global, start_indices)
