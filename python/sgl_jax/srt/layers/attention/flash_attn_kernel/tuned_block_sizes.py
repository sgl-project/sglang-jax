"""Auto-tuned block sizes for ragged paged attention."""

import jax.numpy as jnp

from sgl_jax.srt.layers.attention.flash_attn_kernel.util import (
    align_to,
    get_device_name,
    get_dtype_packing,
    get_tpu_version,
    next_power_of_2,
)

# key
#   - device_name
#     - q dtype
#     - kv dtype
#     - q head number
#     - kv head number
#     - head dim
#     - page_size
#     - max_num_tokens
# value:
#   - (num_kv_pages_per_block, num_queries_per_block)
TUNED_BLOCK_SIZES = {
    "TPU v6": {
        ("bfloat16", "bfloat16", 8, 2, 128, 128, 1): (32, 32),
        ("bfloat16", "bfloat16", 8, 2, 128, 128, 2): (16, 8),
        ("bfloat16", "bfloat16", 8, 2, 128, 128, 4): (16, 32),
        ("bfloat16", "bfloat16", 8, 2, 128, 128, 8): (16, 4),
        ("bfloat16", "bfloat16", 8, 2, 128, 128, 16): (16, 4),
        ("bfloat16", "bfloat16", 8, 2, 128, 128, 128): (16, 4),
        ("bfloat16", "bfloat16", 8, 2, 128, 128, 4096): (16, 128),
    },
}


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
) -> tuple[int, int]:
    """Look up for the best (num_kv_pages_per_blk, num_queries_per_blk) from auto-tuned table."""
    tpu_version = get_tpu_version()
    if tpu_version < 4:
        raise NotImplementedError("TPU version must be 4 or higher.")
    keys = get_simplified_key(
        page_size,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        max_num_tokens,
    )

    device_name = keys[0]

    # Default block sizes.
    bkv_p, bq = (2048 // page_size, 32)
    if tpu_version == 4:
        # TPUv4 has much smaller VMEM size so we pick fixed block sizes.
        bkv_p, bq = (512 // page_size, 32)
    else:
        if device_name in TUNED_BLOCK_SIZES:
            if keys in TUNED_BLOCK_SIZES[device_name]:
                bkv_p, bq = TUNED_BLOCK_SIZES[device_name][keys]

    return (min(pages_per_seq, bkv_p), min(max_num_tokens, bq))


def get_simplified_key(
    page_size,
    q_dtype,
    kv_dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_num_tokens,
):
    """Get the simplified key to reduce the number of combinations."""
    assert num_q_heads % num_kv_heads == 0
    device = get_device_name()
    q_dtype_name = jnp.dtype(q_dtype).name
    kv_dtype_name = jnp.dtype(kv_dtype).name
    num_q_heads = next_power_of_2(num_q_heads)
    num_kv_heads = next_power_of_2(num_kv_heads)

    return (
        device,
        q_dtype_name,
        kv_dtype_name,
        num_q_heads,
        num_kv_heads,
        (head_dim + 127) // 128 * 128,
        next_power_of_2(page_size),
        next_power_of_2(max_num_tokens),
    )
