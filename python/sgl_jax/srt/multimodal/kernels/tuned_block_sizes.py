"""Auto-tuned block sizes for flash attention."""

import logging

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

logger = logging.getLogger(__name__)
# key
#   - device_name
#     - q dtype
#     - k dtype
#     - v dtype
#     - batch size
#     - head number
#     - q length
#     - kv length
#     - head dim
# value:
#   - (num_queries_per_block,)
TUNED_BLOCK_SIZES = {
    "TPU v6e": {
        ("float32", "float32", "bfloat16", 2, 12, 16896, 16896, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 16896, 17152, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 16896, 17408, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17152, 16896, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17152, 17152, 128): (256,),
        ("float32", "float32", "bfloat16", 2, 12, 17152, 17408, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17408, 16896, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17408, 17152, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17408, 17408, 128): (512,),
    }
}


def get_tuned_block_sizes(
    q_dtype,
    k_dtype,
    v_dtype,
    batch_size,
    head_num,
    q_len,
    kv_len,
    head_dim,
) -> tuple[int, int]:
    """Look up for the best (num_queries_per_blk,) from auto-tuned table."""

    keys = get_simplified_key(
        q_dtype,
        k_dtype,
        v_dtype,
        batch_size,
        head_num,
        q_len,
        kv_len,
        head_dim,
    )

    device_name = keys[0]

    # Default block sizes.
    bq = 256
    if device_name in TUNED_BLOCK_SIZES and keys[1:] in TUNED_BLOCK_SIZES[device_name]:
        bq = TUNED_BLOCK_SIZES[device_name][keys[1:]][0]
    else:
        logger.info("Using default block q size: bq=%s.", bq)

    return bq


def get_simplified_key(
    q_dtype,
    k_dtype,
    v_dtype,
    batch_size,
    head_num,
    q_len,
    kv_len,
    head_dim,
):
    """Get the simplified key to reduce the number of combinations."""
    device = get_device_name()
    q_dtype_name = jnp.dtype(q_dtype).name
    k_dtype_name = jnp.dtype(k_dtype).name
    v_dtype_name = jnp.dtype(v_dtype).name

    return (
        device,
        q_dtype_name,
        k_dtype_name,
        v_dtype_name,
        batch_size,
        head_num,
        q_len,
        kv_len,
        head_dim,
    )
