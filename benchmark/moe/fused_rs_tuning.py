"""Pure tuning-space helpers for the GLM-5.2 fused-RS benchmark.

Keep K whole: the fused kernel accumulates K steps in FP32 scratch, so K
splitting adds accumulation work without reducing the output-side scratch that
dominates this shape.  Search N tiling and M tiling under the actual weight
pipeline modes instead:

* ``full_resident`` keeps every N step for an expert in VMEM;
* ``streaming`` uses the kernel's two-buffer rotation and reloads weights for
  each grouped-M tile, but is excluded from production tuning because the
  current path is not invariant to invalid padded routes;
* multi-step single-buffer configs are outside the pipeline contract.

This module deliberately has no JAX imports.  Candidate legality and the
declared scratch estimate can therefore be tested before consuming TPU time;
the TPU compiler remains the final VMEM authority.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

FusedRsBlockConfig = tuple[int, int, int, int, int, int, int]
ContractValue = int | bool | str

GLM52_RS_K1 = 6144
GLM52_RS_N1 = 2048
GLM52_RS_K2 = 2048
GLM52_RS_N2 = 6144

GLM52_RS_VMEM_CAPACITY_BYTES = 64 * 1024 * 1024
GLM52_RS_VMEM_LIMIT_BYTES = int(GLM52_RS_VMEM_CAPACITY_BYTES * 0.95)
GLM52_RS_COMPILER_HEADROOM_BYTES = 6 * 1024 * 1024

_BF16_BYTES = 2
_FP32_BYTES = 4
_FP8_BYTES = 1
_SUBLANE_ROWS = 8
_ROWS_PER_EXPERT = 65536 * 8 // 256
_M_ALIGNMENT = 128
_N_ALIGNMENT = 128
_MIN_N_TILE = 256
_N1_TILES = (2048, 1024, 512, 256)
_N2_TILES = (6144, 3072, 2048, 1536, 1024, 768, 512, 384, 256)

GLM52_RS_REFERENCE_CONFIG: FusedRsBlockConfig = (
    128,
    GLM52_RS_K1,
    GLM52_RS_N1,
    GLM52_RS_K2,
    GLM52_RS_N2,
    1,
    1,
)


def _buffer_mode(steps: int, buffers: int) -> str:
    if steps == 1 and buffers == 1:
        return "single_step"
    if steps > 1 and buffers == steps:
        return "full_resident"
    if steps > 2 and buffers == 2:
        return "streaming"
    return "invalid"


def _estimate_declared_vmem_bytes(config: FusedRsBlockConfig) -> int:
    """Estimate the kernel's declared VMEM scratch for the target dtype path."""
    tile_m, _, tile_n1, _, tile_n2, num_w1_bufs, num_w2_bufs = config

    # Scratch shapes mirror gmm_fused_rs_nodedup.py.  The target path uses BF16
    # activations/outputs, FP8 per-channel weights, and FP32 accumulators/scales.
    gathered_lhs = 2 * tile_m * GLM52_RS_K1 * _BF16_BYTES
    gathered_indices = 2 * tile_m * _FP32_BYTES
    gmm1_out = tile_m * GLM52_RS_N1 * _FP32_BYTES
    intermediate = tile_m * GLM52_RS_K2 * _BF16_BYTES
    tiled_out = 2 * tile_m * GLM52_RS_N2 * _BF16_BYTES
    scatter_scratch = 3 * tile_m * GLM52_RS_N2 * _BF16_BYTES
    gmm1_partial = _SUBLANE_ROWS * (2 * tile_n1) * _FP32_BYTES
    gmm2_partial = _SUBLANE_ROWS * tile_n2 * _FP32_BYTES
    shared_accumulator = tile_m * max(2 * tile_n1, tile_n2) * _FP32_BYTES

    w1_buffers = num_w1_bufs * GLM52_RS_K1 * (2 * tile_n1) * _FP8_BYTES
    w2_buffers = num_w2_bufs * GLM52_RS_K2 * tile_n2 * _FP8_BYTES
    scale_buffers = (num_w1_bufs * (2 * tile_n1) + num_w2_bufs * tile_n2) * _FP32_BYTES
    bias_scratch = (2 * tile_n1 + tile_n2) * _FP32_BYTES

    return sum(
        (
            gathered_lhs,
            gathered_indices,
            gmm1_out,
            intermediate,
            tiled_out,
            scatter_scratch,
            gmm1_partial,
            gmm2_partial,
            shared_accumulator,
            w1_buffers,
            w2_buffers,
            scale_buffers,
            bias_scratch,
        )
    )


def analyze_rs_config(config: FusedRsBlockConfig) -> dict[str, ContractValue]:
    """Return pipeline, cache, VMEM, and traffic facts for one config."""
    if len(config) != 7 or any(value <= 0 for value in config):
        raise ValueError("fused-RS configs must contain seven positive integers")

    tile_m, tile_k1, tile_n1, tile_k2, tile_n2, num_w1_bufs, num_w2_bufs = config
    divisibility = (
        GLM52_RS_K1 % tile_k1 == 0,
        GLM52_RS_N1 % tile_n1 == 0,
        GLM52_RS_K2 % tile_k2 == 0,
        GLM52_RS_N2 % tile_n2 == 0,
    )
    if not all(divisibility):
        raise ValueError(
            "fused-RS tiles must divide GLM-5.2 K1/N1/K2/N2 exactly; "
            f"got config={config}"
        )

    w1_steps = (GLM52_RS_N1 // tile_n1) * (GLM52_RS_K1 // tile_k1)
    w2_steps = (GLM52_RS_N2 // tile_n2) * (GLM52_RS_K2 // tile_k2)
    w1_buffer_mode = _buffer_mode(w1_steps, num_w1_bufs)
    w2_buffer_mode = _buffer_mode(w2_steps, num_w2_bufs)
    can_cache_w1 = w1_buffer_mode in {"single_step", "full_resident"}
    can_cache_w2 = w2_buffer_mode in {"single_step", "full_resident"}
    pipeline_contract_valid = "invalid" not in {w1_buffer_mode, w2_buffer_mode}
    # Production prefill pads every device-local token shard with topk_id=-1
    # and routing weight 0.  Target-v7x oracle tests show that the current
    # two-buffer streaming path changes valid outputs under that padding,
    # while single-step/full-resident weight paths preserve them.
    padding_contract_valid = can_cache_w1 and can_cache_w2
    full_k = tile_k1 == GLM52_RS_K1 and tile_k2 == GLM52_RS_K2
    matrix_tile_aligned = (
        tile_m % _M_ALIGNMENT == 0
        and tile_n1 >= _MIN_N_TILE
        and tile_n2 >= _MIN_N_TILE
        and tile_n1 % _N_ALIGNMENT == 0
        and tile_n2 % _N_ALIGNMENT == 0
    )

    estimated_vmem_bytes = _estimate_declared_vmem_bytes(config)
    estimated_vmem_with_headroom_bytes = (
        estimated_vmem_bytes + GLM52_RS_COMPILER_HEADROOM_BYTES
    )
    vmem_contract_valid = (
        estimated_vmem_with_headroom_bytes <= GLM52_RS_VMEM_LIMIT_BYTES
    )

    estimated_gm_tiles_per_expert = math.ceil(_ROWS_PER_EXPERT / tile_m)
    w1_loads_per_expert = 1 if can_cache_w1 else estimated_gm_tiles_per_expert
    w2_loads_per_expert = 1 if can_cache_w2 else estimated_gm_tiles_per_expert
    w1_matrix_bytes = GLM52_RS_K1 * (2 * GLM52_RS_N1) * _FP8_BYTES
    w2_matrix_bytes = GLM52_RS_K2 * GLM52_RS_N2 * _FP8_BYTES
    estimated_weight_dma_bytes_per_expert = (
        w1_loads_per_expert * w1_matrix_bytes + w2_loads_per_expert * w2_matrix_bytes
    )

    eligible_for_tuning = (
        full_k
        and matrix_tile_aligned
        and pipeline_contract_valid
        and padding_contract_valid
        and vmem_contract_valid
    )
    return {
        "tile_m": tile_m,
        "tile_k1": tile_k1,
        "tile_n1": tile_n1,
        "tile_k2": tile_k2,
        "tile_n2": tile_n2,
        "num_w1_bufs": num_w1_bufs,
        "num_w2_bufs": num_w2_bufs,
        "w1_steps": w1_steps,
        "w2_steps": w2_steps,
        "w1_buffer_mode": w1_buffer_mode,
        "w2_buffer_mode": w2_buffer_mode,
        "can_cache_w1": can_cache_w1,
        "can_cache_w2": can_cache_w2,
        "pipeline_contract_valid": pipeline_contract_valid,
        "padding_contract_valid": padding_contract_valid,
        "full_k": full_k,
        "matrix_tile_aligned": matrix_tile_aligned,
        "estimated_vmem_bytes": estimated_vmem_bytes,
        "estimated_vmem_with_headroom_bytes": estimated_vmem_with_headroom_bytes,
        "vmem_contract_valid": vmem_contract_valid,
        "estimated_gm_tiles_per_expert": estimated_gm_tiles_per_expert,
        "w1_loads_per_expert": w1_loads_per_expert,
        "w2_loads_per_expert": w2_loads_per_expert,
        "estimated_weight_dma_bytes_per_expert": estimated_weight_dma_bytes_per_expert,
        "eligible_for_tuning": eligible_for_tuning,
    }


def _best_config_for_cache_mode(
    tile_m: int,
    *,
    cache_w1: bool,
    cache_w2: bool,
) -> FusedRsBlockConfig | None:
    candidates: list[FusedRsBlockConfig] = []
    for tile_n1 in _N1_TILES:
        w1_steps = GLM52_RS_N1 // tile_n1
        num_w1_bufs = w1_steps if cache_w1 else 2
        if (w1_steps > num_w1_bufs) != (not cache_w1):
            continue
        for tile_n2 in _N2_TILES:
            w2_steps = GLM52_RS_N2 // tile_n2
            num_w2_bufs = w2_steps if cache_w2 else 2
            if (w2_steps > num_w2_bufs) != (not cache_w2):
                continue

            config = (
                tile_m,
                GLM52_RS_K1,
                tile_n1,
                GLM52_RS_K2,
                tile_n2,
                num_w1_bufs,
                num_w2_bufs,
            )
            if analyze_rs_config(config)["eligible_for_tuning"]:
                candidates.append(config)

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda config: (
            int(analyze_rs_config(config)["w1_steps"])
            + int(analyze_rs_config(config)["w2_steps"]),
            int(analyze_rs_config(config)["estimated_vmem_bytes"]),
        ),
    )


def generate_rs_tuning_configs(
    tile_ms: Iterable[int],
) -> tuple[FusedRsBlockConfig, ...]:
    """Generate a cache-aware, full-K search frontier under the VMEM budget."""
    configs: list[FusedRsBlockConfig] = []
    for tile_m in tile_ms:
        if tile_m <= 0 or tile_m % _M_ALIGNMENT != 0:
            raise ValueError(
                f"tile_m must be a positive multiple of {_M_ALIGNMENT}, got {tile_m}"
            )

        if tile_m == 128:
            # Retain the four low-M, fully resident N geometries.  They all keep
            # the complete 36 MiB of expert weights resident; the sweep measures
            # whether fewer N steps or smaller accumulator tiles win.
            configs.extend(
                (
                    (128, 6144, 2048, 2048, 6144, 1, 1),
                    (128, 6144, 1024, 2048, 6144, 2, 1),
                    (128, 6144, 2048, 2048, 3072, 1, 2),
                    (128, 6144, 1024, 2048, 3072, 2, 2),
                )
            )
            continue

        # For larger M, choose the lowest-step legal geometry in every cache
        # regime.  VMEM may remove a regime entirely (for example, both sides
        # resident at M=256/384).
        for cache_w1, cache_w2 in (
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ):
            config = _best_config_for_cache_mode(
                tile_m,
                cache_w1=cache_w1,
                cache_w2=cache_w2,
            )
            if config is not None:
                configs.append(config)

    unique_configs = tuple(dict.fromkeys(configs))
    if not unique_configs:
        raise ValueError("no full-K fused-RS candidates fit the tuning contract")
    for config in unique_configs:
        if not analyze_rs_config(config)["eligible_for_tuning"]:
            raise AssertionError(f"generated an invalid fused-RS candidate: {config}")
    return unique_configs
