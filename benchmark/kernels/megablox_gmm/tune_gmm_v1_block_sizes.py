"""
Tune GMM v1 block sizes for Grok-2 model configurations.

Sweeps (tm, tk, tn) tiling parameters for each (m, k, n, dtype) combination
and prints the best config in the TUNED_BLOCK_SIZES dict format used by
sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_block_sizes.

Usage:
    python -m benchmark.kernels.megablox_gmm.tune_gmm_v1_block_sizes
"""

from __future__ import annotations

import functools
from math import inf

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm

# ---------------------------------------------------------------------------
# Grok-2 MoE shape constants
#   hidden_size=8192, moe_intermediate_size=16384, num_local_experts=8,
#   num_experts_per_tok=2, TP=8, EP=1
# ---------------------------------------------------------------------------
NUM_TOTAL_GROUPS = 8  # num_local_experts
NUM_CURRENT_GROUPS = 8  # all experts on each device (ep_size=1)
TP_SIZE = 8

HIDDEN_SIZE = 8192
MOE_INTERMEDIATE_SIZE = 16384

# After TP sharding:
#   gate/up (wi_0/wi_1): [g, hidden_size, intermediate/TP] -> k=8192, n=2048
#   down    (wo):        [g, intermediate/TP, hidden_size]  -> k=2048, n=8192
KN_PAIRS: list[tuple[int, int]] = [
    (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE // TP_SIZE),  # gate/up: k=8192, n=2048
    (MOE_INTERMEDIATE_SIZE // TP_SIZE, HIDDEN_SIZE),  # down:    k=2048, n=8192
]

# batch_size * top_k (top_k=2 for Grok-2)
BATCH_SIZES: list[int] = [1, 2, 4, 8, 16, 32, 64, 8192, 16384]
TOP_K = 2
M_VALUES: list[int] = [bs * TOP_K for bs in BATCH_SIZES]

# Three dtype combos: (lhs_dtype, rhs_dtype)
DTYPE_COMBOS: list[tuple[jnp.dtype, jnp.dtype]] = [
    (jnp.float8_e4m3fn, jnp.float8_e4m3fn),
]


def _make_tiling_candidates(m: int, k: int, n: int) -> list[tuple[int, int, int]]:
    """Generate candidate (tm, tk, tn) tiling triples to sweep."""
    tm_options = set()
    for val in [m]:
        if m % val == 0:
            tm_options.add(val)
    for val in [128, 256]:
        if val <= m and m % val == 0:
            tm_options.add(val)
    tm_options = sorted(tm_options)

    tk_options = set()
    for mult in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]:
        val = 256 * mult
        if val <= k and k % val == 0:
            tk_options.add(val)
    if 128 <= k:
        tk_options.add(128)
    tk_options = sorted(tk_options)

    tn_options = set()
    for mult in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]:
        val = 256 * mult
        if val <= n and n % val == 0:
            tn_options.add(val)
    if 128 <= n:
        tn_options.add(128)
    tn_options = sorted(tn_options)

    candidates = []
    for tm in tm_options:
        for tk in tk_options:
            for tn in tn_options:
                candidates.append((tm, tk, tn))
    return candidates


def _create_inputs(
    m: int,
    k: int,
    n: int,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
):
    """Create GMM inputs for benchmarking.

    group_sizes has shape [NUM_TOTAL_GROUPS]. Tokens are spread across the
    first NUM_CURRENT_GROUPS experts; when m < NUM_CURRENT_GROUPS the extras
    get 0 tokens (mimicking sparse routing at small batch sizes).
    """
    active = np.zeros(NUM_CURRENT_GROUPS, dtype=np.int32)
    if m >= NUM_CURRENT_GROUPS:
        per_expert = m // NUM_CURRENT_GROUPS
        active[:] = per_expert
        remaining = m - per_expert * NUM_CURRENT_GROUPS
        active[:remaining] += 1
    else:
        active[:m] = 1
    inactive = np.zeros(NUM_TOTAL_GROUPS - NUM_CURRENT_GROUPS, dtype=np.int32)
    group_sizes = jnp.array(np.concatenate([active, inactive]), dtype=jnp.int32)

    lhs = jnp.zeros((m, k), dtype=lhs_dtype)
    rhs = jnp.zeros((NUM_CURRENT_GROUPS, k, n), dtype=rhs_dtype)

    rhs_scale = None
    if rhs_dtype == jnp.float8_e4m3fn:
        rhs_scale = jnp.ones((NUM_CURRENT_GROUPS, 1, 1, n), dtype=jnp.float32)

    return lhs, rhs, group_sizes, rhs_scale


def benchmark_one_config(
    m: int,
    k: int,
    n: int,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    tiling: tuple[int, int, int],
    tries: int = 3,
) -> float:
    """Benchmark a single (m, k, n, dtype, tiling) configuration.

    Returns average time in ms, or inf on failure.
    """
    lhs, rhs, group_sizes, rhs_scale = _create_inputs(m, k, n, lhs_dtype, rhs_dtype)

    if m % tiling[0] != 0:
        return inf

    @functools.partial(
        jax.jit,
        static_argnames=["preferred_element_type", "tiling"],
    )
    def jitted_gmm(lhs, rhs, group_sizes, rhs_scale, preferred_element_type, tiling):
        return gmm(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type=preferred_element_type,
            rhs_scale=rhs_scale,
            tiling=tiling,
        )

    gmm_fn = functools.partial(
        jitted_gmm,
        lhs,
        rhs,
        group_sizes,
        rhs_scale,
        jnp.float32,
        tiling,
    )

    try:
        jax.block_until_ready(gmm_fn())
    except Exception:
        return inf

    tm, tk, tn = tiling
    task = f"gmm-g_{NUM_CURRENT_GROUPS}-m_{m}-k_{k}-n_{n}-tm_{tm}-tk_{tk}-tn_{tn}"
    try:
        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: gmm_fn(),
            data_generator=lambda: (),
            task=task,
            tries=tries,
        )
        return float(np.mean(times)) if times else inf
    except Exception:
        return inf


def _format_tile_val(val: int) -> str:
    """Format a tile value as 128 or 256 * N for readability."""
    if val == 128:
        return "128"
    if val % 256 == 0:
        mult = val // 256
        if mult == 1:
            return "256 * 1"
        return f"256 * {mult}"
    return str(val)


OUTPUT_PATH = "tuned_gmm_v1_block_sizes.txt"


def main():
    print("JAX devices:", jax.devices())
    print("Device count:", jax.device_count())
    print()
    print("Tuning GMM v1 block sizes for Grok-2 configurations")
    print(f"Results will be written incrementally to {OUTPUT_PATH}")
    print("=" * 100)

    with open(OUTPUT_PATH, "a") as f:
        for lhs_dtype, rhs_dtype in DTYPE_COMBOS:
            lhs_str = jnp.dtype(lhs_dtype).name
            rhs_str = jnp.dtype(rhs_dtype).name

            print(f"\n{'='*100}")
            print(f"  dtype combo: lhs={lhs_str}, rhs={rhs_str}")
            print(f"{'='*100}")

            for k, n in KN_PAIRS:
                quant_block_size = k
                print(f"\n  (k={k}, n={n})")

                for m in M_VALUES:
                    candidates = _make_tiling_candidates(m, k, n)
                    best_time = inf
                    best_tiling: tuple[int, int, int] | None = None

                    for tiling in candidates:
                        t = benchmark_one_config(m, k, n, lhs_dtype, rhs_dtype, tiling)
                        if t < best_time:
                            best_time = t
                            best_tiling = tiling

                    if best_tiling is not None:
                        tm, tk, tn = best_tiling
                        tm_s = _format_tile_val(tm)
                        tk_s = _format_tile_val(tk)
                        tn_s = _format_tile_val(tn)
                        entry = (
                            f"    ({m}, {k}, {n}, {NUM_TOTAL_GROUPS}, {NUM_CURRENT_GROUPS}, "
                            f'"{lhs_str}", "{rhs_str}", {quant_block_size}): (\n'
                            f"        {tm_s},\n"
                            f"        {tk_s},\n"
                            f"        {tn_s},\n"
                            f"    ),"
                        )
                        f.write(entry + "\n")
                        f.flush()
                        print(
                            f"    m={m:>6}: best=({tm_s}, {tk_s}, {tn_s})  "
                            f"time={best_time:.4f} ms"
                        )
                    else:
                        print(f"    m={m:>6}: NO VALID TILING FOUND")

    print(f"\nDone. Results in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
