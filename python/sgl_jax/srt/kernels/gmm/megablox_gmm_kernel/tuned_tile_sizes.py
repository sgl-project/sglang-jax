"""Shape-specific GMM v2 tile sizes measured on supported TPU generations."""

from __future__ import annotations

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

# Key: (lhs dtype, rhs dtype, groups, M, K, N).
# Values are (tile_m, tile_k, tile_n).
TUNED_TILE_SIZES_GMM_V2 = {
    "TPU v7": {
        # Ling-3.0-tiny replicated EPMoE, decode BS=1 hot wi shape.
        # Measured kernel latency: 0.555ms -> 0.382ms (31.1% lower).
        ("bfloat16", "bfloat16", 128, 32, 1536, 512): (32, 768, 512),
        # Ling-3.0-tiny replicated EPMoE, 2K balanced prefill hot shapes.
        ("bfloat16", "bfloat16", 128, 2048, 1536, 512): (32, 1536, 512),
        ("bfloat16", "bfloat16", 128, 2048, 512, 1536): (32, 512, 1536),
    },
}


def get_tuned_gmm_v2_tile_sizes(
    *,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    num_groups: int,
    size_m: int,
    size_k: int,
    size_n: int,
    device_name: str | None = None,
) -> tuple[int, int, int] | None:
    if device_name is None:
        device_name = get_device_name()
    table = TUNED_TILE_SIZES_GMM_V2.get(device_name)
    if table is None:
        return None
    key = (
        jnp.dtype(lhs_dtype).name,
        jnp.dtype(rhs_dtype).name,
        int(num_groups),
        int(size_m),
        int(size_k),
        int(size_n),
    )
    return table.get(key)
