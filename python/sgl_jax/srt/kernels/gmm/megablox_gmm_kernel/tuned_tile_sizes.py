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
        # GLM-5.2 ep16 batched-decode hot shapes (tp16, 16 local experts,
        # fp8 blockwise weights). lhs key is the QUANTIZED lhs dtype:
        # make_gmm_configs passes lhs_q_dtype (f8e4m3 on v7x for fp8 rhs)
        # to the tile fn, not bf16. m = concurrency x top8 rows, replicated
        # per device with group_offset. Whole-K/whole-N tiles load each
        # expert's weight exactly once; the auto-tiler's smaller tiles
        # re-read weights and cost ~1.9x. Swept on v7x (microbench sweep):
        #   wi m256 119.9->63.8us, wo m256 113.8->62.3us,
        #   wi m512 135.1->70.6us, wo m512 129.3->71.2us (all bitwise-equal).
        ("float8_e4m3fn", "float8_e4m3fn", 16, 256, 6144, 2048): (32, 6144, 2048),
        ("float8_e4m3fn", "float8_e4m3fn", 16, 256, 2048, 6144): (32, 2048, 6144),
        ("float8_e4m3fn", "float8_e4m3fn", 16, 512, 6144, 2048): (32, 6144, 2048),
        ("float8_e4m3fn", "float8_e4m3fn", 16, 512, 2048, 6144): (32, 2048, 6144),
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
