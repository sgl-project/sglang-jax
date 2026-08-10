"""Auto-tuned block sizes for the BF16 MHA ``varlen_attention`` kernel.

``varlen_attention`` exposes ``num_queries_per_block`` and
``num_kv_per_block``. This table records the fastest correct ``(bq, bkv)`` per
TPU, normalized kernel shape, and maximum packed-sequence length.

Key:
  - device_name (from ``get_device_name``)
    - (heads, head_dim_aligned_to_128, max_seq_len_aligned_to_power_of_two)
Value:
  - (num_queries_per_block, num_kv_per_block)

The table only serves the BF16 MHA fast path. GQA and F32 use the packed
fallback kernel and keep its default block sizes until tuned independently.

The ``"TPU v7"`` entries were tuned on a v7x core (64 MiB VMEM) for the
Qwen2.5-VL vision shapes: BF16 MHA, 16 heads, head_dim 80 (padded to 128).
"""

import logging

import jax

from sgl_jax.srt.utils.jax_utils import get_device_name

logger = logging.getLogger(__name__)

# Fallback used off-TPU and on table misses (matches varlen_attention defaults).
DEFAULT_Q_BLOCK = 128
DEFAULT_KV_BLOCK = 256

TUNED_VARLEN_BLOCK_SIZES = {
    # BF16 MHA, 16 heads, padded head_dim 128; tuned on a v7x core.
    "TPU v7": {
        # Qwen2.5-VL local-window segments contain at most 64 tokens.
        (16, 128, 64): (128, 256),
        # Full-attention sequences.
        (16, 128, 256): (256, 256),
        (16, 128, 1024): (256, 512),
        (16, 128, 2048): (256, 1024),
        (16, 128, 4096): (256, 1024),
        (16, 128, 8192): (256, 1024),
        (16, 128, 16384): (256, 1024),
        # VMEM is bounded by the block tile (not the total sequence length), so
        # 32768 reuses the 16384 block sizes. Carried over, not re-tuned.
        (16, 128, 32768): (256, 1024),
    }
}


def _align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def get_tuned_block_sizes(
    heads: int,
    head_dim: int,
    max_seq_len: int,
) -> tuple[int, int]:
    """Look up the best BF16-MHA ``(bq, bkv)`` for a normalized shape.

    ``head_dim`` is aligned to 128 and ``max_seq_len`` to the next power of two
    before lookup. Returns the kernel defaults off TPU or on a table miss.
    """
    if heads <= 0:
        raise ValueError(f"heads must be positive, got {heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

    key = (
        heads,
        _align_to(head_dim, 128),
        _next_power_of_two(max_seq_len),
    )

    # The tuned table is TPU-only; other backends fall back without probing for
    # a TPU device name.
    if "TPU" not in jax.devices()[0].device_kind:
        return DEFAULT_Q_BLOCK, DEFAULT_KV_BLOCK

    device_name = get_device_name()
    device_table = TUNED_VARLEN_BLOCK_SIZES.get(device_name)
    if device_table is not None and key in device_table:
        return device_table[key]

    logger.info(
        "varlen: using default block sizes bq=%s bkv=%s (no tuned entry for %s %s).",
        DEFAULT_Q_BLOCK,
        DEFAULT_KV_BLOCK,
        device_name,
        key,
    )
    return DEFAULT_Q_BLOCK, DEFAULT_KV_BLOCK
