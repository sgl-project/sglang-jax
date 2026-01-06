"""Auto-tuned block configs for fused_moe.

This module mirrors the approach used by
`sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes`:

- The "tuning" is expected to be done offline via benchmarking.
- The results are stored as a maintained lookup table keyed by a simplified
  (bucketed) shape signature and device kind.
- Runtime uses the table when available, otherwise falls back to a fixed
  baseline config (and callers may apply override/validation logic).

Note: In this fused_moe kernel, `bt` is the expert-side token tile size used for
HBM<->VMEM staging and output tiling. `FusedMoEBlockConfig.effective_for(...)`
may clamp `bt` (and `btc`) based on the runtime shape.
"""

from __future__ import annotations

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

from .kernel import FusedMoEBlockConfig

# Key (without device_name):
#   - dtype name
#   - num_tokens
#   - num_experts
#   - top_k
#   - hidden_size
#   - intermediate_size
#   - ep_size
#   - subc_quant_wsz (0 means None)
#
# Value:
#   - (bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c)
TUNED_BLOCK_CONFIGS: dict[str, dict[tuple, tuple[int, int, int, int, int, int, int, int]]] = {
    # Populate per-device kind, e.g. "TPU v6e", "TPU v7".
    "TPU v6e": {
        ("bfloat16", 128, 8, 8, 8192, 2048, 4): (32, 2048, 2048, 2048, 32, 2048, 2048, 2048),
        ("bfloat16", 4096, 8, 8, 8192, 2048, 4): (128, 1024, 2048, 2048, 128, 1024, 2048, 2048),
    },
    "TPU v7": {
        ("bfloat16", 16, 256, 8, 8192, 2048, 8): (2, 1024, 2048, 2048, 2, 1024, 2048, 2048),
        ("bfloat16", 32, 256, 8, 8192, 2048, 8): (4, 2048, 1024, 1024, 4, 2048, 1024, 1024),
        ("bfloat16", 64, 256, 8, 8192, 2048, 8): (8, 2048, 1024, 1024, 8, 2048, 1024, 1024),
        ("bfloat16", 128, 256, 8, 8192, 2048, 8): (16, 1024, 2048, 2048, 16, 1024, 2048, 2048),
        ("bfloat16", 256, 256, 8, 8192, 2048, 8): (32, 512, 4096, 4096, 32, 512, 4096, 4096),
        ("bfloat16", 512, 256, 8, 8192, 2048, 8): (64, 512, 2048, 2048, 64, 512, 2048, 2048),
        ("bfloat16", 1024, 256, 8, 8192, 2048, 8): (64, 512, 2048, 2048, 64, 512, 2048, 2048),
        ("bfloat16", 2048, 256, 8, 8192, 2048, 8): (64, 512, 2048, 2048, 64, 512, 2048, 2048),
        ("bfloat16", 4096, 256, 8, 8192, 2048, 8): (64, 1024, 1024, 1024, 64, 1024, 1024, 1024),
        ("bfloat16", 64, 256, 8, 8192, 2048, 32): (2, 2048, 2048, 2048, 2, 2048, 2048, 2048),
        ("bfloat16", 128, 256, 8, 8192, 2048, 32): (4, 2048, 2048, 2048, 4, 2048, 2048, 2048),
        ("bfloat16", 256, 256, 8, 8192, 2048, 32): (8, 2048, 2048, 2048, 8, 2048, 2048, 2048),
        ("bfloat16", 512, 256, 8, 8192, 2048, 32): (16, 2048, 2048, 2048, 16, 2048, 2048, 2048),
        ("bfloat16", 1024, 256, 8, 8192, 2048, 32): (32, 1024, 4096, 4096, 32, 1024, 4096, 4096),
        ("bfloat16", 2048, 256, 8, 8192, 2048, 32): (64, 1024, 2048, 2048, 64, 1024, 2048, 2048),
        ("bfloat16", 4096, 256, 8, 8192, 2048, 32): (128, 1024, 1024, 1024, 128, 1024, 1024, 1024),
    },
    # Fallback for any device kind.
    "*": {},
}

DEFAULT_FUSED_MOE_BLOCK_CONFIG = FusedMoEBlockConfig(
    bt=32,
    bf=1024,
    bd1=1024,
    bd2=1024,
    btc=32,
    bfc=1024,
    bd1c=1024,
    bd2c=1024,
)


def get_simplified_key(
    *,
    dtype: jnp.dtype,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
) -> tuple:
    """Get a simplified key to reduce the number of tuned combinations."""
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if num_tokens % ep_size != 0:
        raise ValueError(f"Expected {num_tokens=} to be aligned to {ep_size=}.")

    device = get_device_name()
    dtype_name = jnp.dtype(dtype).name
    return (
        device,
        dtype_name,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        ep_size,
    )


def get_tuned_fused_moe_block_config(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: jnp.dtype,
    ep_size: int,
):
    """Look up the best block config from the tuned table.

    Raises:
      KeyError: if not found and allow_fallback=False.
    """

    keys = get_simplified_key(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=ep_size,
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

    bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c = cfg_tuple
    cfg = FusedMoEBlockConfig(
        bt=bt,
        bf=bf,
        bd1=bd1,
        bd2=bd2,
        btc=btc,
        bfc=bfc,
        bd1c=bd1c,
        bd2c=bd2c,
    )
    return cfg
