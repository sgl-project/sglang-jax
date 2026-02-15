"""Auto-tuned block configs for fused_moe.

This module mirrors the approach used by
`sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes`:

- The "tuning" is expected to be done offline via benchmarking.
- The results are stored as a maintained lookup table keyed by a simplified
  (bucketed) shape signature and device kind.
- Runtime uses the table when available, otherwise falls back to a fixed
  baseline config (and callers may apply override/validation logic).
"""

# ruff: noqa: E501

from __future__ import annotations

import logging

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

from .kernel import FusedMoEBlockConfig

logger = logging.getLogger(__name__)

# Key (without device_name):
#   - tokens dtype name
#   - weight dtype name
#   - num_tokens (padded token count used for tuning lookup / kernel shape)
#   - num_experts
#   - top_k
#   - hidden_size
#   - intermediate_size
#   - ep_size
#   - use_shared_expert
#   - use_grouped_topk
#
# Value (current):
#   - (bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse)
# fmt: off
TUNED_BLOCK_CONFIGS: dict[str, dict[tuple, tuple[int, ...]]] = {
    # Populate per-device kind, e.g. "TPU v6e", "TPU v7".
    "TPU v7": {
        ('bfloat16', 'bfloat16', 16, 128, 8, 2048, 768, 8, False, False): (2, 256, 2048, 2048, 2, 2, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 32, 128, 8, 2048, 768, 8, False, False): (4, 256, 2048, 2048, 4, 4, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 64, 128, 8, 2048, 768, 8, False, False): (8, 256, 2048, 2048, 8, 8, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 128, 128, 8, 2048, 768, 8, False, False): (16, 256, 2048, 2048, 16, 16, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 256, 128, 8, 2048, 768, 8, False, False): (32, 256, 2048, 2048, 32, 32, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 512, 128, 8, 2048, 768, 8, False, False): (64, 256, 2048, 2048, 64, 64, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 1024, 128, 8, 2048, 768, 8, False, False): (128, 256, 2048, 2048, 128, 128, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 2048, 128, 8, 2048, 768, 8, False, False): (256, 256, 2048, 2048, 256, 256, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 4096, 128, 8, 2048, 768, 8, False, False): (512, 256, 2048, 2048, 512, 512, 256, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 8192, 128, 8, 2048, 768, 8, False, False): (512, 256, 2048, 2048, 512, 512, 256, 2048, 2048, 256),

        ('bfloat16', 'float8_e4m3fn', 16, 128, 8, 2048, 768, 8, False, False): (2, 256, 2048, 2048, 2, 2, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 32, 128, 8, 2048, 768, 8, False, False): (4, 256, 2048, 2048, 4, 4, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 64, 128, 8, 2048, 768, 8, False, False): (8, 256, 2048, 2048, 8, 8, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 128, 128, 8, 2048, 768, 8, False, False): (16, 256, 2048, 2048, 16, 16, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 256, 128, 8, 2048, 768, 8, False, False): (32, 256, 2048, 2048, 32, 32, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 512, 128, 8, 2048, 768, 8, False, False): (64, 256, 2048, 2048, 64, 64, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 1024, 128, 8, 2048, 768, 8, False, False): (128, 256, 2048, 2048, 128, 128, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 2048, 128, 8, 2048, 768, 8, False, False): (256, 256, 2048, 2048, 256, 256, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 4096, 128, 8, 2048, 768, 8, False, False): (512, 256, 2048, 2048, 512, 512, 256, 2048, 2048, 256),
        ('bfloat16', 'float8_e4m3fn', 8192, 128, 8, 2048, 768, 8, False, False): (512, 256, 2048, 2048, 512, 512, 256, 2048, 2048, 256),

        ('bfloat16', 'bfloat16', 64, 256, 8, 8192, 2048, 32, False, False): (2, 2048, 2048, 2048, 2, 2, 2048, 2048, 2048, 2048),
        ('bfloat16', 'bfloat16', 128, 256, 8, 8192, 2048, 32, False, False): (4, 2048, 2048, 2048, 4, 4, 2048, 2048, 2048, 2048),
        ('bfloat16', 'bfloat16', 256, 256, 8, 8192, 2048, 32, False, False): (8, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 2048),
        ('bfloat16', 'bfloat16', 512, 256, 8, 8192, 2048, 32, False, False): (16, 2048, 2048, 2048, 16, 16, 2048, 2048, 2048, 2048),
        ('bfloat16', 'bfloat16', 1024, 256, 8, 8192, 2048, 32, False, False): (32, 512, 8192, 8192, 32, 32, 512, 8192, 8192, 512),
        ('bfloat16', 'bfloat16', 2048, 256, 8, 8192, 2048, 32, False, False): (64, 1024, 2048, 2048, 64, 64, 1024, 2048, 2048, 1024),
        ('bfloat16', 'bfloat16', 4096, 256, 8, 8192, 2048, 32, False, False): (128, 1024, 1024, 1024, 128, 128, 1024, 1024, 1024, 1024),

        ('bfloat16', 'bfloat16', 64, 256, 8, 8192, 2048, 32, True, False): (2, 2048, 2048, 2048, 2, 2, 2048, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 128, 256, 8, 8192, 2048, 32, True, False): (4, 2048, 2048, 2048, 4, 4, 2048, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 256, 256, 8, 8192, 2048, 32, True, False): (8, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 512, 256, 8, 8192, 2048, 32, True, False): (16, 2048, 1024, 1024, 16, 16, 2048, 1024, 1024, 2048),
        ('bfloat16', 'bfloat16', 1024, 256, 8, 8192, 2048, 32, True, False): (32, 2048, 1024, 1024, 32, 32, 2048, 1024, 1024, 1024),
        ('bfloat16', 'bfloat16', 2048, 256, 8, 8192, 2048, 32, True, False): (64, 1024, 2048, 2048, 64, 64, 1024, 2048, 2048, 512),
        ('bfloat16', 'bfloat16', 4096, 256, 8, 8192, 2048, 32, True, False): (128, 1024, 1024, 1024, 128, 128, 1024, 1024, 1024, 512),

        ('bfloat16', 'bfloat16', 64, 256, 8, 8192, 2048, 32, True, True): (2, 2048, 2048, 2048, 64, 64, 2048, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 128, 256, 8, 8192, 2048, 32, True, True): (4, 2048, 2048, 2048, 128, 128, 2048, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 256, 256, 8, 8192, 2048, 32, True, True): (8, 2048, 1024, 1024, 256, 256, 2048, 1024, 1024, 2048),
        ('bfloat16', 'bfloat16', 512, 256, 8, 8192, 2048, 32, True, True): (16, 2048, 1024, 1024, 512, 512, 2048, 1024, 1024, 1024),
        ('bfloat16', 'bfloat16', 1024, 256, 8, 8192, 2048, 32, True, True): (32, 2048, 1024, 1024, 32, 32, 2048, 1024, 1024, 1024),
        ('bfloat16', 'bfloat16', 2048, 256, 8, 8192, 2048, 32, True, True): (64, 1024, 2048, 2048, 64, 64, 1024, 2048, 2048, 512),
        ('bfloat16', 'bfloat16', 4096, 256, 8, 8192, 2048, 32, True, True): (128, 512, 1024, 1024, 1024, 1024, 512, 1024, 1024, 512),

        ('bfloat16', 'bfloat16', 128, 256, 8, 8192, 2048, 64, True, True): (2, 2048, 2048, 2048, 2, 2, 2048, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 256, 256, 8, 8192, 2048, 64, True, True): (4, 2048, 2048, 2048, 4, 4, 2048, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 512, 256, 8, 8192, 2048, 64, True, True): (8, 2048, 1024, 1024, 8, 8, 2048, 1024, 1024, 2048),
        ('bfloat16', 'bfloat16', 1024, 256, 8, 8192, 2048, 64, True, True): (16, 2048, 1024, 1024, 16, 16, 2048, 1024, 1024, 1024),
        ('bfloat16', 'bfloat16', 2048, 256, 8, 8192, 2048, 64, True, True): (32, 1024, 2048, 2048, 32, 32, 1024, 2048, 2048, 512),
        ('bfloat16', 'bfloat16', 4096, 256, 8, 8192, 2048, 64, True, True): (64, 512, 4096, 4096, 64, 64, 512, 4096, 4096, 128),
        ('bfloat16', 'bfloat16', 16384, 256, 8, 8192, 2048, 64, True, True): (64, 1024, 1024, 1024, 64, 64, 1024, 1024, 1024, 512),
        ('bfloat16', 'bfloat16', 32768, 256, 8, 8192, 2048, 64, True, True): (128, 512, 1024, 1024, 128, 128, 512, 1024, 1024, 128),

        ('bfloat16', 'bfloat16', 256, 256, 8, 8192, 2048, 128, True, True): (2, 2048, 2048, 2048, 2, 2, 2048, 2048, 2048, 256),
        ('bfloat16', 'bfloat16', 512, 256, 8, 8192, 2048, 128, True, True): (4, 1024, 4096, 4096, 4, 4, 1024, 4096, 4096, 128),
        ('bfloat16', 'bfloat16', 1024, 256, 8, 8192, 2048, 128, True, True): (8, 2048, 1024, 1024, 8, 8, 2048, 1024, 1024, 1024),
        ('bfloat16', 'bfloat16', 2048, 256, 8, 8192, 2048, 128, True, True): (16, 1024, 2048, 2048, 16, 16, 1024, 2048, 2048, 128),
        ('bfloat16', 'bfloat16', 4096, 256, 8, 8192, 2048, 128, True, True): (32, 512, 4096, 4096, 32, 32, 512, 4096, 4096, 128),
        ('bfloat16', 'bfloat16', 8192, 256, 8, 8192, 2048, 128, True, True): (64, 512, 2048, 2048, 64, 64, 512, 2048, 2048, 128),

        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 8192, 2048, 32, True, True): (2, 2048, 4096, 4096, 4, 4, 256, 512, 4096, 512),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 8192, 2048, 32, True, True): (4, 2048, 4096, 4096, 8, 8, 256, 512, 4096, 256),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 8192, 2048, 32, True, True): (8, 2048, 2048, 2048, 16, 16, 256, 512, 2048, 2048),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 8192, 2048, 32, True, True): (16, 2048, 2048, 2048, 32, 32, 256, 512, 2048, 1024),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 8192, 2048, 32, True, True): (32, 2048, 2048, 2048, 64, 16, 256, 512, 2048, 1024),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 8192, 2048, 32, True, True): (64, 2048, 1024, 1024, 128, 32, 256, 512, 1024, 512),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 8192, 2048, 32, True, True): (128, 1024, 1024, 1024, 256, 32, 256, 512, 1024, 512),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 8192, 2048, 32, True, True): (256, 256, 1024, 1024, 512, 128, 256, 512, 1024, 512),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 8192, 2048, 32, True, True): (128, 1024, 512, 512, 128, 128, 1024, 512, 512, 1024),
        ('bfloat16', 'float8_e4m3fn', 32768, 256, 8, 8192, 2048, 32, True, True): (128, 1024, 512, 512, 128, 128, 1024, 512, 512, 1024),

        ('bfloat16', 'float8_e4m3fn', 64, 288, 8, 8192, 2048, 32, True, True): (2, 2048, 4096, 4096, 4, 4, 256, 512, 4096, 512),
        ('bfloat16', 'float8_e4m3fn', 128, 288, 8, 8192, 2048, 32, True, True): (4, 2048, 4096, 4096, 8, 8, 256, 512, 4096, 256),
        ('bfloat16', 'float8_e4m3fn', 256, 288, 8, 8192, 2048, 32, True, True): (8, 2048, 2048, 2048, 16, 16, 256, 512, 2048, 2048),
        ('bfloat16', 'float8_e4m3fn', 512, 288, 8, 8192, 2048, 32, True, True): (16, 2048, 2048, 2048, 32, 32, 256, 512, 2048, 1024),
        ('bfloat16', 'float8_e4m3fn', 1024, 288, 8, 8192, 2048, 32, True, True): (32, 2048, 2048, 2048, 64, 16, 256, 512, 2048, 1024),
        ('bfloat16', 'float8_e4m3fn', 2048, 288, 8, 8192, 2048, 32, True, True): (64, 2048, 1024, 1024, 128, 32, 256, 512, 1024, 512),
        ('bfloat16', 'float8_e4m3fn', 4096, 288, 8, 8192, 2048, 32, True, True): (128, 1024, 1024, 1024, 256, 32, 256, 512, 1024, 512),
        ('bfloat16', 'float8_e4m3fn', 8192, 288, 8, 8192, 2048, 32, True, True): (256, 256, 1024, 1024, 512, 128, 256, 512, 1024, 512),
        ('bfloat16', 'float8_e4m3fn', 16384, 288, 8, 8192, 2048, 32, True, True): (128, 1024, 512, 512, 128, 128, 1024, 512, 512, 1024),
        ('bfloat16', 'float8_e4m3fn', 32768, 288, 8, 8192, 2048, 32, True, True): (128, 1024, 512, 512, 128, 128, 1024, 512, 512, 1024),

        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 8192, 2048, 64, True, True): (2, 2048, 4096, 4096, 2, 2, 2048, 4096, 4096, 256),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 8192, 2048, 64, True, True): (4, 1024, 8192, 8192, 4, 4, 1024, 8192, 8192, 128),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 8192, 2048, 64, True, True): (8, 2048, 2048, 2048, 8, 8, 2048, 2048, 2048, 2048),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 8192, 2048, 64, True, True): (16, 2048, 2048, 2048, 16, 16, 2048, 2048, 2048, 1024),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 8192, 2048, 64, True, True): (32, 1024, 4096, 4096, 32, 32, 1024, 4096, 4096, 512),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 8192, 2048, 64, True, True): (64, 1024, 2048, 2048, 64, 64, 1024, 2048, 2048, 512),
        ('bfloat16', 'float8_e4m3fn', 8192, 256, 8, 8192, 2048, 64, True, True): (64, 1024, 2048, 2048, 64, 64, 1024, 2048, 2048, 512),
        ('bfloat16', 'float8_e4m3fn', 16384, 256, 8, 8192, 2048, 64, True, True): (128, 512, 1024, 1024, 128, 128, 512, 1024, 1024, 1024),
        ('bfloat16', 'float8_e4m3fn', 32768, 256, 8, 8192, 2048, 64, True, True): (128, 512, 1024, 1024, 128, 128, 512, 1024, 1024, 1024),

        ('bfloat16', 'float8_e4m3fn', 64, 256, 8, 2048, 512, 32, True, True): (2, 512, 2048, 2048, 2, 2, 512, 2048, 2048, 512),
        ('bfloat16', 'float8_e4m3fn', 128, 256, 8, 2048, 512, 32, True, True): (4, 512, 2048, 2048, 4, 4, 512, 2048, 2048, 512),
        ('bfloat16', 'float8_e4m3fn', 256, 256, 8, 2048, 512, 32, True, True): (8, 512, 2048, 2048, 8, 8, 512, 2048, 2048, 512),
        ('bfloat16', 'float8_e4m3fn', 512, 256, 8, 2048, 512, 32, True, True): (16, 512, 2048, 2048, 16, 16, 512, 2048, 2048, 512),
        ('bfloat16', 'float8_e4m3fn', 1024, 256, 8, 2048, 512, 32, True, True): (32, 512, 2048, 2048, 32, 32, 512, 2048, 2048, 512),
        ('bfloat16', 'float8_e4m3fn', 2048, 256, 8, 2048, 512, 32, True, True): (64, 512, 2048, 2048, 64, 64, 512, 2048, 2048, 512),
        ('bfloat16', 'float8_e4m3fn', 4096, 256, 8, 2048, 512, 32, True, True): (128, 512, 2048, 2048, 128, 128, 512, 2048, 2048, 256),
    },
    # Fallback for any device kind.
    "*": {},
}
# fmt: on

DEFAULT_FUSED_MOE_BLOCK_CONFIG = FusedMoEBlockConfig(
    bt=32,
    bf=512,
    bd1=1024,
    bd2=1024,
    btc=32,
    bfc=512,
    bd1c=1024,
    bd2c=1024,
    bse=512,
)


def get_simplified_key(
    *,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
    use_shared_expert: bool,
    use_grouped_topk: bool,
) -> tuple:
    """Get a simplified key to reduce the number of tuned combinations."""
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")

    device = get_device_name()
    dtype_name = jnp.dtype(dtype).name
    weight_dtype_name = jnp.dtype(weight_dtype).name
    return (
        device,
        dtype_name,
        weight_dtype_name,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        ep_size,
        bool(use_shared_expert),
        bool(use_grouped_topk),
    )


def get_tuned_fused_moe_block_config(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    ep_size: int,
    use_shared_expert: bool,
    use_grouped_topk: bool,
):
    """Look up the best block config from the tuned table.

    Raises:
      KeyError: if not found and allow_fallback=False.
    """

    keys = get_simplified_key(
        dtype=dtype,
        weight_dtype=weight_dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=ep_size,
        use_shared_expert=use_shared_expert,
        use_grouped_topk=use_grouped_topk,
    )
    device_name = keys[0]
    table_key = keys[1:]

    cfg_tuple = None
    if device_name in TUNED_BLOCK_CONFIGS:
        cfg_tuple = TUNED_BLOCK_CONFIGS[device_name].get(table_key)
    if cfg_tuple is None:
        cfg_tuple = TUNED_BLOCK_CONFIGS.get("*", {}).get(table_key)

    if cfg_tuple is None:
        return DEFAULT_FUSED_MOE_BLOCK_CONFIG

    if len(cfg_tuple) != 10:
        raise ValueError(f"Unexpected tuned config tuple length: {len(cfg_tuple)}")

    logger.info("Using tuned block config: %s", cfg_tuple)

    bt, bf, bd1, bd2, bts, btc, bfc, bd1c, bd2c, bse = cfg_tuple

    cfg = FusedMoEBlockConfig(
        bt=bt,
        bf=bf,
        bd1=bd1,
        bd2=bd2,
        btc=btc,
        bfc=bfc,
        bd1c=bd1c,
        bd2c=bd2c,
        bse=bse,
        bts=bts,
    )
    return cfg
