"""Auto-tuned block sizes for ragged paged attention."""

import logging

import jax.numpy as jnp

from sgl_jax.srt.kernels.ragged_paged_attention.util import get_tpu_version
from sgl_jax.srt.utils.common_utils import next_power_of_2
from sgl_jax.srt.utils.jax_utils import get_device_name

logger = logging.getLogger(__name__)
# key
#   - device_name
#     - q dtype
#     - kv dtype
#     - q head number
#     - kv head number
#     - head dim
#     - page_size
#     - max_num_tokens
#     - static_q_len (Mixed: None, Prefill: Chunked_size, Decode: 1)
#     - sliding windown length
# value:
#   - (num_kv_per_block, num_queries_per_block, num_compute_kv_per_block, num_compute_queries_per_block)
TUNED_BLOCK_SIZES = {
    "TPU v7": {
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 1, 1, None): (1024, 1, 1024, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 2, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 4, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 8, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 16, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 32, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 64, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 128, 1, None): (4096, 1, 2048, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 256, 1, None): (4096, 1, 2048, 1),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 512, None, None): (512, 512, 512, 256),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 512, 8192, None): (2048, 1024, 1024, 512),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 1024, None, None): (512, 512, 512, 256),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 1024, 8192, None): (2048, 1024, 1024, 512),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 2048, None, None): (1024, 1024, 1024, 512),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 2048, 8192, None): (2048, 1024, 1024, 512),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 4096, None, None): (1024, 1024, 1024, 512),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 4096, 8192, None): (2048, 1024, 1024, 512),
        ("bfloat16", "bfloat16", 1, 1, 128, 128, 8192, None, None): (2048, 1024, 1024, 512),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 1, 1, None): (1024, 1, 1024, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 2, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 4, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 8, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 16, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 32, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 64, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 128, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 256, 1, None): (2048, 1, 2048, 1),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 8192, None, None): (4096, 512, 1024, 128),
        ("bfloat16", "bfloat16", 8, 1, 128, 128, 16384, None, None): (4096, 1024, 1024, 128),
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
    static_q_len,
    sliding_window,
) -> tuple[int, int, int, int]:
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
        static_q_len,
        sliding_window,
    )

    device_name = keys[0]

    if device_name in TUNED_BLOCK_SIZES and keys[1:] in TUNED_BLOCK_SIZES[device_name]:
        bkv_sz, bq_sz, bkv_csz, bq_csz = TUNED_BLOCK_SIZES[device_name][keys[1:]]
        return (bkv_sz, bq_sz, bkv_csz, bq_csz)
    else:
        logger.info(
            "Tuned RPA block sizes not found for %s: page_size=%s, actual_num_q_heads=%s, "
            "actual_num_kv_heads=%s, head_dim=%s, max_num_tokens=%s, pages_per_seq=%s. Using default block sizes.",
            device_name,
            page_size,
            actual_num_q_heads,
            actual_num_kv_heads,
            head_dim,
            max_num_tokens,
            pages_per_seq,
        )

        return None


def get_simplified_key(
    page_size,
    q_dtype,
    kv_dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_num_tokens,
    static_q_len,
    sliding_window,
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
        static_q_len,
        sliding_window,
    )
