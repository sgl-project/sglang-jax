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
        # GLM-5.2 ep32 BF16 (8 local experts, wi=(6144, 2048), wo=(2048, 6144))
        ("bfloat16", "bfloat16", 8, 256, 6144, 2048): (32, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 512, 6144, 2048): (32, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 1024, 6144, 2048): (32, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 2048, 6144, 2048): (64, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 4096, 6144, 2048): (64, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 8192, 6144, 2048): (64, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 16384, 6144, 2048): (128, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 32768, 6144, 2048): (128, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 65536, 6144, 2048): (256, 1024, 2048),
        ("bfloat16", "bfloat16", 8, 131072, 6144, 2048): (512, 3072, 2048),
        ("bfloat16", "bfloat16", 8, 262144, 6144, 2048): (512, 6144, 1024),
        ("bfloat16", "bfloat16", 8, 524288, 6144, 2048): (512, 6144, 1024),
        ("bfloat16", "bfloat16", 8, 1048576, 6144, 2048): (512, 6144, 1024),
        ("bfloat16", "bfloat16", 8, 256, 2048, 6144): (32, 512, 6144),
        ("bfloat16", "bfloat16", 8, 512, 2048, 6144): (32, 512, 6144),
        ("bfloat16", "bfloat16", 8, 1024, 2048, 6144): (32, 2048, 1024),
        ("bfloat16", "bfloat16", 8, 2048, 2048, 6144): (64, 512, 6144),
        ("bfloat16", "bfloat16", 8, 4096, 2048, 6144): (64, 512, 6144),
        ("bfloat16", "bfloat16", 8, 8192, 2048, 6144): (64, 512, 6144),
        ("bfloat16", "bfloat16", 8, 16384, 2048, 6144): (128, 512, 6144),
        ("bfloat16", "bfloat16", 8, 32768, 2048, 6144): (256, 1024, 6144),
        ("bfloat16", "bfloat16", 8, 65536, 2048, 6144): (256, 512, 6144),
        ("bfloat16", "bfloat16", 8, 131072, 2048, 6144): (512, 512, 6144),
        ("bfloat16", "bfloat16", 8, 262144, 2048, 6144): (512, 2048, 3072),
        ("bfloat16", "bfloat16", 8, 524288, 2048, 6144): (512, 2048, 3072),
        ("bfloat16", "bfloat16", 8, 1048576, 2048, 6144): (512, 2048, 3072),
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
