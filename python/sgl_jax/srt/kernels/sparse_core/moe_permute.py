"""SparseCore-backed MoE permute (dispatch gather) and unpermute (weighted combine).

Why: on TPU the XLA gather offload already runs MoE gathers on SparseCore, but it
cannot express the expert-parallel semantics ``EPMoE`` needs -- it gathers every
``tokens * top_k`` row (all experts) even though a device only computes its own
expert slice, and the combine gather + einsum is exposed on the TensorCore. The
tpu-inference kernels vendored next to this module gather only the local row
range (``ragged_gather_v2``) and fuse gather + top-k weighted reduce
(``ragged_gather_reduce``).

Both entry points fall back to the plain XLA formulation whenever SparseCore is
unavailable or the problem is too small for the offload to pay off, so callers
can use them unconditionally once the opt-in is on.
"""

import functools
import logging

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.utils.common_utils import get_bool_env_var

logger = logging.getLogger(__name__)

# Opt-in switch (default off) read by EPMoE when ``use_sc_permute`` is not given.
ENV_FLAG = "SGL_JAX_MOE_SC_PERMUTE"

# Below this fraction of TensorCore VMEM a plain TC gather beats the SC round-trip
# (same heuristic tpu-inference uses inside ragged_gather_reduce).
_SMALL_INPUT_VMEM_FRACTION = 0.6
_SUPPORTED_GATHER_BITS = (8, 16, 32)


def moe_sc_permute_enabled_by_env() -> bool:
    return get_bool_env_var(ENV_FLAG, "false")


@functools.lru_cache(maxsize=1)
def sparse_core_available() -> bool:
    try:
        if jax.devices()[0].platform != "tpu":
            return False
        return pltpu.get_tpu_info().sparse_core is not None
    except Exception as e:  # noqa: BLE001 - any probe failure means "not available"
        logger.debug("SparseCore probe failed: %s", e)
        return False


def _tc_vmem_bytes() -> int:
    return int(pltpu.get_tpu_info().vmem_capacity_bytes)


def _small_for_sparse_core(nbytes: int) -> bool:
    return nbytes * 2 < _tc_vmem_bytes() * _SMALL_INPUT_VMEM_FRACTION


def should_use_sparse_core(num_rows: int, hidden: int, dtype) -> bool:
    """Trace-time decision shared by EPMoE and the wrappers below.

    False means "emit exactly the XLA formulation" -- callers must not add any
    masking/range bookkeeping in that case, so decode-sized batches keep an HLO
    identical to the flag-off path (and hit the same compilation cache).
    """
    nbits = jax.dtypes.itemsize_bits(dtype)
    return (
        sparse_core_available()
        and nbits in _SUPPORTED_GATHER_BITS
        and hidden % 128 == 0
        and not _small_for_sparse_core(num_rows * hidden * nbits // 8)
    )


def reference_combine(
    intermediate: jax.Array, revert_indices: jax.Array, weights: jax.Array, top_k: int
) -> jax.Array:
    """XLA formulation of EPMoE unpermute: gather back to token order, weighted sum over top_k."""
    total_tokens = revert_indices.shape[0] // top_k
    unsorted = jnp.take(intermediate, indices=revert_indices, axis=0)
    unsorted = unsorted.reshape(total_tokens, top_k, -1).astype(jnp.float32)
    w = weights.reshape(total_tokens, top_k).astype(jnp.float32)
    return jnp.einsum("BKE,BK -> BE", unsorted, w).astype(intermediate.dtype)


def sc_dispatch_gather(
    inputs_2d: jax.Array, token_indices: jax.Array, start: jax.Array, end: jax.Array
) -> jax.Array:
    """``inputs_2d[token_indices]`` where only rows ``[start, end)`` are guaranteed valid.

    Rows outside the range are never read by the grouped matmul (they belong to
    other devices' experts), so skipping them is what makes this cheaper than XLA.
    """
    if not should_use_sparse_core(token_indices.shape[0], inputs_2d.shape[-1], inputs_2d.dtype):
        return inputs_2d[token_indices]
    from sgl_jax.srt.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2

    return ragged_gather_v2(inputs_2d, token_indices, start, end)


def sc_combine(
    intermediate: jax.Array,
    revert_indices: jax.Array,
    weights: jax.Array,
    valid_mask: jax.Array,
    top_k: int,
) -> jax.Array:
    """Weighted top-k combine of sorted expert outputs back into token order.

    ``valid_mask[i]`` marks whether routed slot ``i`` (token-major order) hit a
    local expert; masked rows contribute zero regardless of their contents.
    """
    if (
        intermediate.dtype != jnp.bfloat16
        or weights.dtype not in (jnp.float32, jnp.bfloat16)
        or not should_use_sparse_core(
            intermediate.shape[0], intermediate.shape[-1], intermediate.dtype
        )
    ):
        masked = jnp.where(valid_mask.reshape(-1, 1), weights.reshape(-1, 1), 0).reshape(-1)
        return reference_combine(intermediate, revert_indices, masked, top_k)
    from sgl_jax.srt.kernels.sparse_core.ragged_gather_reduce_v2 import (
        ragged_gather_reduce,
    )

    return ragged_gather_reduce(
        intermediate,
        revert_indices,
        weights.reshape(-1),
        valid_mask.reshape(-1).astype(jnp.bool_),
        top_k,
    )
