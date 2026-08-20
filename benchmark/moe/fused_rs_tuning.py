"""Pure tuning-space helpers for the GLM-5.2 fused-RS benchmark.

The upstream kernel enables same-expert weight caching only when the VMEM
buffers can hold every weight step for that expert.  Keep this planner free of
JAX imports so the contract and candidate space can be checked on a CPU-only
developer machine before consuming TPU time.
"""

from __future__ import annotations

from collections.abc import Iterable

FusedRsBlockConfig = tuple[int, int, int, int, int, int, int]

GLM52_RS_K1 = 6144
GLM52_RS_N1 = 2048
GLM52_RS_K2 = 2048
GLM52_RS_N2 = 6144

GLM52_RS_REFERENCE_CONFIG: FusedRsBlockConfig = (
    128,
    GLM52_RS_K1,
    GLM52_RS_N1,
    GLM52_RS_K2,
    GLM52_RS_N2,
    1,
    1,
)


def analyze_rs_config(config: FusedRsBlockConfig) -> dict[str, int | bool]:
    """Return the weight-step/cache contract for one fixed GLM-5.2 config."""
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
    can_cache_w1 = num_w1_bufs >= w1_steps
    can_cache_w2 = num_w2_bufs >= w2_steps
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
        "can_cache_w1": can_cache_w1,
        "can_cache_w2": can_cache_w2,
        "buffer_contract_valid": can_cache_w1 and can_cache_w2,
    }


def generate_rs_tuning_configs(tile_ms: Iterable[int]) -> tuple[FusedRsBlockConfig, ...]:
    """Generate full-N, independent split-N, and combined split-N probes.

    Every split dimension gets exactly one buffer per weight step.  The old
    ``(..., 1024, ..., 3072, 1, 1)`` geometry is represented only with ``(2, 2)``
    buffers; whether that candidate compiles and fits VMEM remains a TPU result,
    not an assumption made by this planner.
    """
    configs: list[FusedRsBlockConfig] = []
    for tile_m in tile_ms:
        if tile_m <= 0:
            raise ValueError(f"tile_m must be positive, got {tile_m}")
        configs.extend(
            (
                (tile_m, 6144, 2048, 2048, 6144, 1, 1),
                (tile_m, 6144, 1024, 2048, 6144, 2, 1),
                (tile_m, 6144, 2048, 2048, 3072, 1, 2),
                (tile_m, 6144, 1024, 2048, 3072, 2, 2),
            )
        )

    unique_configs = tuple(dict.fromkeys(configs))
    for config in unique_configs:
        if not analyze_rs_config(config)["buffer_contract_valid"]:
            raise AssertionError(f"generated an invalid fused-RS candidate: {config}")
    return unique_configs
