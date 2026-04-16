"""
Benchmark fused_moe kernel with grouped-GEMM-like MoE shapes.

Usage:
    python -m benchmark.moe.bench_fused_moe --use-shared-expert  --imbalance-mode sparse_hotspot --hotspot-ratio 1 --hotspot-count 48 --tune-block-config --num-experts 256 --topk 8 --hidden-size 8192 --intermediate-size 2048 --num-expert-group 8 --topk-group 4
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import os
import sys
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as _compilation_cache
from jax.sharding import PartitionSpec as P

from benchmark.moe.utils import (
    DEFAULT_NUM_TOKENS,
    MoEBenchmarkCase,
    MoEImbalanceSimulator,
    build_mesh,
    make_moe_cases,
    prepare_fused_moe_inputs,
    select_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    validate_fused_moe_block_config,
)
from sgl_jax.srt.layers.moe import FusedEPMoE, TopK

# Match the fused_moe kernel's current pallas VMEM limit (96 MiB).
# The estimator still applies its own MSA overhead factor, and callers can
# further tighten the search with `--tpu-vmem-headroom-ratio`.
DEFAULT_TPU_VMEM_BUDGET_MB = 96
DEFAULT_TPU_VMEM_BUDGET_BYTES = DEFAULT_TPU_VMEM_BUDGET_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# NOTE: skip some config kernel will crash when running.
# Per-case block-config exclusion list.
# Key:   (num_tokens, num_experts, top_k, hidden_size, intermediate_size, ep_size)
# Value: set of config tuples (bt, bts, bf, bd1, bd2, btc, bfc, bd1c, bd2c, bse)
#        that should be skipped during tuning for that case.
# ---------------------------------------------------------------------------
# fmt: off
EXCLUDED_BLOCK_CONFIGS: dict[tuple[int, int, int, int, int, int], set[tuple[int, ...]]] = {
    (32768, 256, 8, 8192, 2048, 64): {
        (128, 128, 256, 4096, 4096, 128, 256, 4096, 4096, 128),
        (128, 128, 256, 2048, 2048, 128, 256, 2048, 2048, 256),
    },
}
# fmt: on


def _tpu_log_recorder_compiler_options() -> dict[str, str] | None:
    enable = os.getenv("SGLANG_JAX_ENABLE_KERNEL_LOG_RECORDER", "0") in ("1", "true", "True")
    if enable and jax.default_backend() == "tpu":
        return {"xla_tpu_enable_log_recorder": "true"}
    return None


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("1", "true", "t", "yes", "y", "on"):
        return True
    if val in ("0", "false", "f", "no", "n", "off", ""):
        return False
    raise ValueError(f"Invalid boolean env var {name}={val!r} (expected 0/1/true/false).")


def _env_bool_opt(name: str) -> bool | None:
    """Return None if unset, otherwise parse as bool."""
    val = os.getenv(name)
    if val is None:
        return None
    return _env_bool(name)


def _with_all_disable(env_name: str, *, all_disable: bool) -> bool:
    """Use per-flag env override if set; otherwise fall back to all_disable."""
    specific = _env_bool_opt(env_name)
    if specific is not None:
        return specific
    return all_disable


def _dtype_packing(dtype: jnp.dtype) -> int:
    """Match get_dtype_packing() in fused_moe kernel (32-bit repack width)."""
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype packing for {dtype=} ({bits=} bits).")
    return 32 // bits


def _estimate_vmem_bytes(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    router_dtype: jnp.dtype,
    cfg: FusedMoEBlockConfig,
    intermediate_size: int,
    hidden_size: int,
    use_shared_expert: bool = False,
    quant_block_k: int | None = None,
    verbose: bool = False,
) -> int:
    """Rough VMEM estimate to avoid compile-time OOM (TPU VMEM is 64MB/core).

    Note: this intentionally overestimates a bit because the fused_moe kernel
    materializes several routing/top-k temporaries (see `get_top_k` in the
    pallas kernel body), which are not part of the explicit scratch buffers.

    Args:
        quant_block_k: Sub-channel quantization block size for weight scales.
            When set, adds VMEM for scale scratch buffers.
    """
    bt = cfg.bt
    bts = bt if cfg.bts is None else int(cfg.bts)
    bf = cfg.bf
    bd1 = cfg.bd1
    bd2 = cfg.bd2
    top_k = case.top_k
    num_devices = case.ep_size
    hidden = case.hidden_size

    token_bytes = jnp.dtype(dtype).itemsize
    weight_bytes = jnp.dtype(weight_dtype).itemsize

    t_packing = _dtype_packing(dtype)
    padded_num_experts = ((case.num_experts + 127) // 128) * 128
    padded_top_k = ((case.top_k + 127) // 128) * 128
    # Kernel scratch shapes use bt directly.
    # a2a_max_tokens = align_to(bt * num_devices, bts)
    a2a_max_tokens = ((bt * num_devices + bts - 1) // bts) * bts

    # Output-side staging (no overlap):
    # - a2a_g_acc_vmem: (2, top_k, gcd(bt, 16), t_packing, hidden_per_pack)
    # - b_output_x2_vmem: (2, bt, hidden_size)
    acc_bt = math.gcd(bt, 16)  # Must match fused_moe kernel scratch shape.
    a2a_g_acc = 2 * top_k * acc_bt * hidden * token_bytes
    # b_output_x2_vmem is double-buffered to overlap store(output_hbm) with next bt's compute.
    b_output = 2 * bt * hidden * token_bytes
    # t2e_routing_smem scratch is placed in SMEM (not VMEM).
    t2e_routing = 0

    # See kernel scratch shapes: b_w1_x2_vmem/b_w3_x2_vmem/b_w2_x2_vmem.
    w1 = 2 * bd1 * bf * weight_bytes
    w3 = 2 * bd1 * bf * weight_bytes
    w2 = 2 * bf * bd2 * weight_bytes

    # Scale scratch buffers for quantized weights (F32).
    # 1D sub-channel:
    #   b_w1/w3_scale_x2_vmem: (2, t_packing, bd1 // t_packing // quant_block_k, 1, bf)
    #   b_w2_scale_x2_vmem:    (2, t_packing, bf // quant_block_k, 1, bd2 // t_packing)
    w1_scale = 0
    w3_scale = 0
    w2_scale = 0
    if quant_block_k is not None:
        bd1_per_pack = bd1 // t_packing
        bd2_per_pack = bd2 // t_packing
        w1_scale_dim3 = bf
        w3_scale_dim3 = bf
        w2_scale_dim3 = bd2_per_pack
        w1_scale = 2 * t_packing * (bd1_per_pack // quant_block_k) * 1 * w1_scale_dim3 * 4
        w3_scale = 2 * t_packing * (bd1_per_pack // quant_block_k) * 1 * w3_scale_dim3 * 4
        w2_scale = 2 * t_packing * (bf // quant_block_k) * 1 * w2_scale_dim3 * 4

    # b_acc_vmem is F32(2, a2a_max_tokens, 1, bf)
    b_acc = 2 * a2a_max_tokens * bf * 4
    # U32 token staging for FFN1: (2, bts, bd1 // t_packing)
    # Note: Using 4 bytes for packing factor adjustment roughly
    t_stage_b32 = 2 * bts * (bd1 // 2) * 4  # Approximation
    # Kernel uses triple-buffering for a2a_s_acc staging: (3, bts, bd2 // t_packing)
    a2a_s_acc_stage_b32 = 3 * bts * (bd2 // 2) * 4  # Approximation

    # Routing / top-k temporaries in kernel (best-effort conservative estimate):
    b_topk_weights_x2_vmem = 2 * bt * padded_top_k * 4  # (2, bt, padded_top_k)
    b_topk_ids_x2_vmem = 2 * bt * padded_top_k * 4  # (2, bt, padded_top_k)
    expert_iota_vmem = 1 * 1 * padded_num_experts * 4  # (1, 1, padded_num_experts)
    routing_mask_vmem = bt * top_k * padded_num_experts * 4  # (bt, top_k, padded_num_experts)
    expert_metadata_vmem = 2 * (1 * padded_num_experts * 4)  # (2, 1, padded_num_experts)

    routing_temporaries = (
        b_topk_weights_x2_vmem
        + b_topk_ids_x2_vmem
        + expert_iota_vmem
        + routing_mask_vmem
        + expert_metadata_vmem
    )

    total_bytes = (
        a2a_g_acc
        + b_output
        + t2e_routing
        + w1
        + w3
        + w2
        + w1_scale
        + w3_scale
        + w2_scale
        + b_acc
        + t_stage_b32
        + a2a_s_acc_stage_b32
        + routing_temporaries
    )

    # Estimate compute intermediaries from scale-group fori_loops.
    # With lax.fori_loop (XLA While), only ONE iteration's intermediaries
    # are live at a time (not n_sg copies as with Python-loop unrolling).
    compute_intermediaries = 0
    if quant_block_k is not None:
        btc = cfg.btc
        bfc = cfg.bfc
        bd1c = cfg.bd1c
        bd2c = cfg.bd2c
        n_bd1c_tiles = (bd1 + bd1c - 1) // bd1c
        n_bfc_tiles = (bf + bfc - 1) // bfc
        # FFN1: each sg iteration produces 2 dot results (w1,w3) of shape (btc, bfc) f32
        ffn1_per_sg = 2 * btc * bfc * 4
        ffn1_total = n_bd1c_tiles * n_bfc_tiles * 1 * t_packing * ffn1_per_sg
        # FFN2: each sg iteration produces 1 dot result of shape (btc, bd2c_per_tp) f32
        bd2c_per_tp = bd2c // t_packing
        n_bd2c_tiles = (bd2 + bd2c - 1) // bd2c
        ffn2_per_sg = btc * bd2c_per_tp * 4
        ffn2_total = n_bd2c_tiles * 1 * t_packing * ffn2_per_sg
        compute_intermediaries = ffn1_total + ffn2_total
    total_bytes += compute_intermediaries

    # Shared expert scratch buffers.
    se_w1 = 0
    se_w3 = 0
    se_w2 = 0
    se_tokens = 0
    se_acc = 0
    se_w1_scale = 0
    se_w3_scale = 0
    se_w2_scale = 0
    if use_shared_expert:
        bse = cfg.bse
        se_w1 = 2 * bd1 * bse * weight_bytes  # (2, t_packing, bd/pack, bse)
        se_w3 = 2 * bd1 * bse * weight_bytes
        se_w2 = 2 * bse * bd2 * weight_bytes
        # Matches fused_moe kernel scratch shape (bt ping-pong x bd1-slice ping-pong):
        # (2, 2, bt, t_packing, bd1_per_pack) => 4 * bt * bd1 elements.
        se_tokens = 4 * bt * bd1 * token_bytes
        # b_se_acc_vmem: F32(2, bt, hidden_size) - accumulator for SE to avoid bf16 precision loss
        se_acc = 2 * bt * hidden * 4

        # Shared expert scale scratch buffers (F32).
        # b_se_w1_scale_x2_vmem: (2, t_packing, bd1 // t_packing // quant_block_k, 1, bse)
        # b_se_w3_scale_x2_vmem: (2, t_packing, bd1 // t_packing // quant_block_k, 1, bse)
        # b_se_w2_scale_x2_vmem: (2, t_packing, bse // quant_block_k, 1, bd2 // t_packing)
        if quant_block_k is not None:
            se_w1_scale = intermediate_size * 4
            se_w3_scale = intermediate_size * 4
            se_w2_scale = hidden_size * 4
            total_bytes += se_w1_scale + se_w3_scale + se_w2_scale

    total_bytes += se_w1 + se_w3 + se_w2 + se_tokens + se_acc
    total_bytes += se_w1_scale + se_w3_scale + se_w2_scale

    # XLA's Memory Space Assignment (MSA) allocates significantly more VMEM
    # than the sum of explicit scratch shapes due to buffer alignment, While
    # loop state copies, and packing fragmentation.  Empirically measured at
    # ~1.5x on TPU v6e with the fused MoE megakernel (67 MB scratch → 104 MB
    # XLA actual).
    total_bytes = int(total_bytes * 1.5)

    if verbose:

        def _mb(b: int) -> str:
            return f"{b / (1024 * 1024):.2f}"

        print("    VMEM Breakdown:")
        print(
            f"      b_w1_x2_vmem:           {_mb(w1)} MB  (2, {t_packing}, {bd1 // t_packing}, {bf})"
        )
        print(
            f"      b_w3_x2_vmem:           {_mb(w3)} MB  (2, {t_packing}, {bd1 // t_packing}, {bf})"
        )
        print(
            f"      b_w2_x2_vmem:           {_mb(w2)} MB  (2, {t_packing}, {bf}, {bd2 // t_packing})"
        )
        if quant_block_k is not None:
            bd1_per_pack = bd1 // t_packing
            w1_scale_dim3 = bf
            w3_scale_dim3 = bf
            w2_scale_dim3 = bd2 // t_packing
            print(
                f"      b_w1_scale_x2_vmem:     {_mb(w1_scale)} MB  "
                f"(2, {t_packing}, {bd1_per_pack // quant_block_k}, 1, {w1_scale_dim3}) f32"
            )
            print(
                f"      b_w3_scale_x2_vmem:     {_mb(w3_scale)} MB  "
                f"(2, {t_packing}, {bd1_per_pack // quant_block_k}, 1, {w3_scale_dim3}) f32"
            )
            print(
                f"      b_w2_scale_x2_vmem:     {_mb(w2_scale)} MB  "
                f"(2, {t_packing}, {bf // quant_block_k}, 1, {w2_scale_dim3}) f32"
            )
        print(f"      b_acc_vmem:             {_mb(b_acc)} MB  (2, {a2a_max_tokens}, 1, {bf}) f32")
        print(f"      b_output_x2_vmem:       {_mb(b_output)} MB  (2, {bt}, {hidden})")
        print(
            f"      a2a_g_acc_vmem:         {_mb(a2a_g_acc)} MB  (2, {top_k}, {acc_bt}, {t_packing}, {hidden // t_packing})"
        )
        print(
            f"      b_stage_x2_vmem:        {_mb(t_stage_b32)} MB  (2, {bts}, {t_packing}, {bd1 // t_packing})"
        )
        print(
            f"      a2a_s_acc_stage_x3:     {_mb(a2a_s_acc_stage_b32)} MB  (3, {bts}, {t_packing}, {bd2 // t_packing})"
        )
        print(f"      routing_temporaries:    {_mb(routing_temporaries)} MB")
        if compute_intermediaries > 0:
            print(
                f"      compute_intermediaries: {_mb(compute_intermediaries)} MB  (fori_loop single-iteration dot products)"
            )
        print("      xla_msa_overhead (1.5x): included in total")
        if use_shared_expert:
            bse = cfg.bse
            print(
                f"      b_se_w1_x2_vmem:        {_mb(se_w1)} MB  (2, {t_packing}, {bd1 // t_packing}, {bse})"
            )
            print(
                f"      b_se_w3_x2_vmem:        {_mb(se_w3)} MB  (2, {t_packing}, {bd1 // t_packing}, {bse})"
            )
            print(
                f"      b_se_w2_x2_vmem:        {_mb(se_w2)} MB  (2, {t_packing}, {bse}, {bd2 // t_packing})"
            )
            print(
                f"      b_se_tokens_vmem:       {_mb(se_tokens)} MB  (2, 2, {bt}, {t_packing}, {bd1 // t_packing})"
            )
            print(f"      b_se_acc_vmem:          {_mb(se_acc)} MB  (2, {bt}, {hidden}) f32")
            if quant_block_k is not None:
                print(
                    f"      b_se_w1_scale_x2_vmem:  {_mb(se_w1_scale)} MB  "
                    f"(1, 1, {intermediate_size}) f32"
                )
                print(
                    f"      b_se_w3_scale_x2_vmem:  {_mb(se_w3_scale)} MB  "
                    f"(1, 1, {intermediate_size}) f32"
                )
                print(
                    f"      b_se_w2_scale_x2_vmem:  {_mb(se_w2_scale)} MB  "
                    f"(1, 1, {hidden_size}) f32"
                )
        print("      ----------------------------")
        print(f"      Total:                  {_mb(total_bytes)} MB")

    return total_bytes


def select_block_configs(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    router_dtype: jnp.dtype,
    *,
    bt_candidates: list[int],
    bts_candidates: list[int] | None = None,
    bf_candidates: list[int],
    bd_candidates: list[int],
    bse_candidates: list[int] | None = None,
    tpu_vmem_budget_bytes: int,
    tpu_vmem_headroom_ratio: float,
    tpu_vmem_estimate_scale: float,
    max_configs: int,
    use_shared_expert: bool = False,
    quant_block_k: int | None = None,
    excluded_configs: set[tuple[int, ...]] | None = None,
) -> list[FusedMoEBlockConfig]:
    """Enumerate block configs from the explicit candidate lists."""
    t_packing = _dtype_packing(dtype)
    tile_align = t_packing * 128
    local_num_tokens = case.num_tokens // case.ep_size
    router_bits = jnp.dtype(router_dtype).itemsize * 8
    router_tile0 = math.gcd(256 // router_bits, local_num_tokens)

    def _pick_candidates(
        *,
        candidates: list[int],
        multiple_of: int,
    ) -> list[int]:
        out: list[int] = []
        for v in candidates:
            if v <= 0:
                continue
            if v % multiple_of != 0:
                continue
            out.append(v)
        return sorted(set(out))

    def _bt_allowed(v: int) -> bool:
        # Allow 2/4/8, otherwise require alignment to 8.
        return v in (2, 4, 8) or v % 8 == 0

    bt_candidates = [
        v
        for v in _pick_candidates(candidates=bt_candidates, multiple_of=t_packing)
        if _bt_allowed(v)
    ]
    bts_candidates_i: list[int] | None
    if bts_candidates is None:
        bts_candidates_i = None
    else:
        bts_candidates_i = _pick_candidates(candidates=list(bts_candidates), multiple_of=1)
    bf_candidates = _pick_candidates(candidates=bf_candidates, multiple_of=128)
    bd_candidates = _pick_candidates(candidates=bd_candidates, multiple_of=tile_align)

    raw_bse_candidates = bse_candidates if bse_candidates is not None else bf_candidates
    bse_candidates_i = _pick_candidates(candidates=raw_bse_candidates, multiple_of=128)

    def validate(cfg: FusedMoEBlockConfig) -> tuple[bool, str]:
        bt = cfg.bt
        bts = bt if cfg.bts is None else int(cfg.bts)
        bf = cfg.bf
        bd1 = cfg.bd1
        bd2 = cfg.bd2
        btc = cfg.btc
        bfc = cfg.bfc
        bd1c = cfg.bd1c
        bd2c = cfg.bd2c
        bse = cfg.bse

        if bt <= 0 or bf <= 0 or bd1 <= 0 or bd2 <= 0:
            return False, "non-positive tile size"
        if bt > local_num_tokens:
            return False, f"bt({bt}) > local_num_tokens({local_num_tokens})"
        if bt % t_packing != 0:
            return False, f"bt({bt}) % t_packing({t_packing}) != 0"
        if not _bt_allowed(bt):
            return False, f"bt({bt}) must be 2, 4, 8, or a multiple of 8"
        if bt % router_tile0 != 0:
            return (
                False,
                f"bt({bt}) not aligned to router_tile0({router_tile0}) for router_dtype={jnp.dtype(router_dtype).name}",
            )
        if not (0 < bts <= bt * case.ep_size):
            return False, f"bts({bts}) not in (0, bt({bt}) * ep_size({case.ep_size})]"
        if not (0 < btc <= bts):
            return False, f"btc({btc}) not in (0, bts({bts})]"
        if btc % t_packing != 0:
            return False, f"btc({btc}) % t_packing({t_packing}) != 0"
        if bts % btc != 0:
            return False, f"bts({bts}) % btc({btc}) != 0"

        if case.intermediate_size % bf != 0:
            return False, f"intermediate_size({case.intermediate_size}) % bf({bf}) != 0"
        if bf % 128 != 0:
            return False, f"bf({bf}) % 128 != 0"
        if bfc % 128 != 0:
            return False, f"bfc({bfc}) % 128 != 0"
        if bf % bfc != 0:
            return False, f"bf({bf}) % bfc({bfc}) != 0"
        if bse % 128 != 0:
            return False, f"bse({bse}) % 128 != 0"

        if case.hidden_size % bd1 != 0 or case.hidden_size % bd2 != 0:
            return (
                False,
                f"hidden_size({case.hidden_size}) not divisible by bd1({bd1})/bd2({bd2})",
            )
        if bd1 % tile_align != 0 or bd2 % tile_align != 0:
            return False, f"bd1({bd1})/bd2({bd2}) not aligned to tile_align({tile_align})"
        if bd1c % tile_align != 0 or bd2c % tile_align != 0:
            return False, f"bd1c({bd1c})/bd2c({bd2c}) not aligned to tile_align({tile_align})"
        if bd1 % bd1c != 0 or bd2 % bd2c != 0:
            return False, f"bd1({bd1}) % bd1c({bd1c}) != 0 or bd2({bd2}) % bd2c({bd2c}) != 0"

        # Pass use_shared_expert to estimate correct VMEM
        est = _estimate_vmem_bytes(
            case,
            dtype,
            weight_dtype,
            router_dtype,
            cfg,
            intermediate_size=case.intermediate_size,
            hidden_size=case.hidden_size,
            use_shared_expert=use_shared_expert,
            quant_block_k=quant_block_k,
        )
        est = int(math.ceil(est * tpu_vmem_estimate_scale))
        effective_budget = int(tpu_vmem_budget_bytes * tpu_vmem_headroom_ratio)
        if est > effective_budget:
            return (
                False,
                f"vmem_est={est / (1024 * 1024):.1f}MB > budget={effective_budget / (1024 * 1024):.1f}MB",
            )
        return True, "ok"

    configs: list[FusedMoEBlockConfig] = []
    seen: set[tuple[int, ...]] = set()

    def _ladder_div2(start: int) -> list[int]:
        out: list[int] = []
        v = int(start)
        while v > 0:
            out.append(v)
            if v == 1:
                break
            v //= 2
        return sorted(set(out), reverse=True)

    def add(*, raw: FusedMoEBlockConfig, effective: FusedMoEBlockConfig) -> None:
        ok, reason = validate(effective)
        if not ok:
            print(f"SKIP {effective}, reason: {reason}")
            return
        key = (
            effective.bt,
            effective.bf,
            effective.bd1,
            effective.bd2,
            effective.bts,
            effective.btc,
            effective.bfc,
            effective.bd1c,
            effective.bd2c,
            effective.bse,
        )
        if key in seen:
            return
        # Check against the per-case exclusion list.
        # Exclusion tuple order: (bt, bts, bf, bd1, bd2, btc, bfc, bd1c, bd2c, bse)
        if excluded_configs is not None:
            excl_key = (
                effective.bt,
                effective.bts if effective.bts is not None else effective.bt,
                effective.bf,
                effective.bd1,
                effective.bd2,
                effective.btc,
                effective.bfc,
                effective.bd1c,
                effective.bd2c,
                effective.bse,
            )
            if excl_key in excluded_configs:
                print(f"SKIP {effective.as_kwargs()}, reason: excluded by EXCLUDED_BLOCK_CONFIGS")
                return
        seen.add(key)
        configs.append(effective)

    for bt in bt_candidates:
        if bts_candidates_i is None:
            # When `bts` isn't explicitly provided, pick a small default set around the
            # *expected* per-expert token count within one `bt` tile:
            #
            #   E[dyn_sz] ~= bt * ep_size * top_k / num_experts
            #
            # This better matches the post-routing/A2A compute dimension than tying `bts`
            # directly to `bt` (which can be tiny in decode when `num_tokens/ep_size` is small).
            max_bts = bt * case.ep_size
            expected = bt * case.ep_size * case.top_k / case.num_experts

            def _pow2_floor(x: float) -> int:
                if x <= 1:
                    return 1
                return 1 << (int(math.floor(math.log2(x))))

            def _pow2_ceil(x: float) -> int:
                if x <= 1:
                    return 1
                return 1 << (int(math.ceil(math.log2(x))))

            lo = _pow2_floor(expected)
            hi = _pow2_ceil(expected)
            bts_list = [bt, lo, hi, hi * 2]
            bts_list = sorted({v for v in bts_list if 0 < v <= max_bts})
        else:
            # When explicitly provided, allow `bts` to exceed `bt` (up to `bt * ep_size`).
            bts_list = [v for v in bts_candidates_i if 0 < v <= bt * case.ep_size]
        for bts in bts_list:
            for bf in bf_candidates:
                for bd in bd_candidates:
                    current_bse_list = bse_candidates_i if use_shared_expert else [bf]

                    for bse in current_bse_list:
                        # Search a small ladder of btc values for each bts.
                        for btc in _ladder_div2(bts):
                            if btc <= 0 or btc > bts:
                                continue
                            if bts % btc != 0:
                                continue
                            raw = FusedMoEBlockConfig(
                                bt=bt,
                                bf=bf,
                                bd1=bd,
                                bd2=bd,
                                btc=btc,
                                bfc=bf,
                                bd1c=bd,
                                bd2c=bd,
                                bts=bts,
                                bse=bse,
                            )
                            effective = raw.effective_for(
                                num_tokens=case.num_tokens, ep_size=case.ep_size, dtype=dtype
                            )
                            add(raw=raw, effective=effective)

    if max_configs <= 0:
        raise ValueError(f"Expected {max_configs=} to be > 0.")

    def score(
        c: FusedMoEBlockConfig,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        return (
            c.bt,
            c.bts or c.bt,
            c.bf,
            c.bd1,
            c.bd2,
            c.btc,
            c.bfc,
            c.bd1c,
            c.bd2c,
            c.bse,
        )

    ranked = sorted(configs, key=score, reverse=True)
    if len(ranked) <= max_configs:
        return ranked

    selected: list[FusedMoEBlockConfig] = []
    selected_keys: set[tuple[int, ...]] = set()

    def _add(cfg: FusedMoEBlockConfig) -> None:
        key = score(cfg)
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(cfg)

    # Do not use a Pareto filter here. For decode-sized MoE cases, "larger in
    # every tile dimension" does not imply lower latency, and the old filter
    # could collapse hundreds of valid configs into a single candidate.
    #
    # Instead, preserve diversity across the outer token/expert tile geometry
    # first (`bt`, `bts`) and only then take larger compute tiles within each
    # bucket. This keeps tuning tractable without deleting the interesting
    # low-latency part of the search space.
    raw_buckets: dict[tuple[int, int], dict[tuple[int, ...], list[FusedMoEBlockConfig]]] = {}
    for cfg in ranked:
        bucket_key = (cfg.bt, cfg.bts or cfg.bt)
        shape_key = (cfg.bf, cfg.bd1, cfg.bd2, cfg.bfc, cfg.bd1c, cfg.bd2c, cfg.bse)
        raw_buckets.setdefault(bucket_key, {}).setdefault(shape_key, []).append(cfg)

    buckets: dict[tuple[int, int], list[FusedMoEBlockConfig]] = {}
    for bucket_key, shape_groups in raw_buckets.items():
        ordered_bucket: list[FusedMoEBlockConfig] = []
        shape_keys = sorted(shape_groups.keys(), reverse=True)
        while True:
            made_progress = False
            for shape_key in shape_keys:
                group = shape_groups[shape_key]
                if not group:
                    continue
                ordered_bucket.append(group.pop(0))
                made_progress = True
            if not made_progress:
                break
        buckets[bucket_key] = ordered_bucket

    bucket_keys = sorted(buckets.keys(), reverse=True)
    while len(selected) < max_configs:
        made_progress = False
        for bucket_key in bucket_keys:
            bucket = buckets[bucket_key]
            if not bucket:
                continue
            _add(bucket.pop(0))
            made_progress = True
            if len(selected) >= max_configs:
                break
        if not made_progress:
            break

    if len(selected) < max_configs:
        for cfg in ranked:
            _add(cfg)
            if len(selected) >= max_configs:
                break

    print(
        f"  limit: {len(configs)} valid configs -> {len(selected)} "
        f"(max={max_configs}, diversity over bt/bts buckets={len(bucket_keys)})"
    )
    return selected


def run_all(
    iters: int,
    dtype: jnp.dtype = jnp.bfloat16,
    weight_dtype: jnp.dtype = jnp.bfloat16,  # Quantize the weight dtype, the activation's dtype always is bfloat16
    *,
    warmup_iters: int = 1,
    tune_block_config: bool = False,
    bt_candidates: list[int] | None = None,
    bts_candidates: list[int] | None = None,
    bf_candidates: list[int] | None = None,
    bd_candidates: list[int] | None = None,
    bse_candidates: list[int] | None = None,
    num_tokens: list[int] | None = None,
    num_experts: int = 256,
    top_k: int = 8,
    hidden_size: int = 2048,
    intermediate_size: int = 512,
    activation: str = "silu",
    renormalize_topk_logits: bool = True,
    num_expert_group: int = 0,
    topk_group: int = 0,
    tpu_vmem_budget_bytes: int = DEFAULT_TPU_VMEM_BUDGET_BYTES,
    tpu_vmem_headroom_ratio: float = 0.90,
    tpu_vmem_estimate_scale: float = 1.0,
    max_configs: int = 9,
    use_shared_expert: bool = False,
    use_grouped_topk: bool | None = None,
    imbalance_mode: str = None,
    alpha: float = None,
    zipf_s: float = None,
    hotspot_ratio: float = None,
    hotspot_count: int = None,
    zero_expert_count: int = None,
    non_hotspot_alpha: float = None,
    token_mask_mode: str = "none",
    token_valid_ratio: float = 1.0,
    token_mask_seed: int = 0,
    quant_block_k_override: int | None = None,
    return_results: bool = False,
) -> list[dict[str, object]] | None:
    if use_grouped_topk is None:
        use_grouped_topk = bool(num_expert_group or topk_group)

    token_mask_mode = (token_mask_mode or "none").lower()
    if token_mask_mode not in ("none", "prefix", "random"):
        raise ValueError(f"Unsupported {token_mask_mode=}. Expected none|prefix|random.")
    if not (0.0 <= token_valid_ratio <= 1.0):
        raise ValueError(f"Expected {token_valid_ratio=} to be within [0.0, 1.0].")

    token_list = DEFAULT_NUM_TOKENS if num_tokens is None else num_tokens
    raw_cases = make_moe_cases(
        num_tokens=token_list,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        renormalize_topk_logits=renormalize_topk_logits,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        name_prefix="fused_moe",
    )

    if num_tokens is not None:
        requested = set(num_tokens)
        raw_cases = [case for case in raw_cases if case.num_tokens in requested]

    cases_all = list(select_cases(raw_cases))
    cases: list[MoEBenchmarkCase] = []
    for c in cases_all:
        if c.tp_size != 1:
            print(
                f"skip [case={c.name}] because tp_size={c.tp_size} (require tp_size=1 for fused_moe)"
            )
            continue
        cases.append(c)
    if not cases:
        print("No runnable fused_moe cases after filtering tp_size!=1.")
        return [] if return_results else None

    tuned_results: dict[str, dict[tuple, tuple[int, int, int, int, int, int, int, int, int]]] = {}
    if tune_block_config:
        from sgl_jax.srt.utils.jax_utils import get_device_name
    results: list[dict[str, object]] = []

    print(f"Running fused_moe benchmarks with weight_dtype={weight_dtype}")
    print(f"  features: shared_expert={use_shared_expert}, grouped_topk={use_grouped_topk}")
    if quant_block_k_override is not None:
        print(f"  quantization: 1D sub-channel (wsz={quant_block_k_override})")
    print(
        "  shape: "
        f"num_experts={num_experts}, top_k={top_k}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
        f"activation={activation}, renormalize_topk_logits={renormalize_topk_logits}, "
        f"num_expert_group={num_expert_group}, topk_group={topk_group}"
    )
    print(
        "  vmem_filter: "
        f"budget={tpu_vmem_budget_bytes / (1024 * 1024):.0f}MB, "
        f"headroom_ratio={tpu_vmem_headroom_ratio:.2f}, "
        f"estimate_scale={tpu_vmem_estimate_scale:.2f}"
    )

    for case in cases:
        t_packing = _dtype_packing(jnp.bfloat16)
        mesh = build_mesh(ep_size=case.ep_size, tp_size=case.tp_size)
        mesh_ep = mesh.shape["tensor"]
        if mesh_ep != case.ep_size:
            print(f"warning [case={case.name}] mesh_ep={mesh_ep} != case.ep_size={case.ep_size}")
        local_num_tokens = case.num_tokens // mesh_ep
        if local_num_tokens % t_packing != 0:
            print(
                f"skip [case={case.name}] because local_num_tokens={local_num_tokens} "
                f"is not aligned to t_packing={t_packing} (dtype={jnp.dtype(dtype).name}, ep_size={mesh_ep})"
            )
            continue
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}, ep_size={case.ep_size}"
        )
        print(
            f"  mesh: ep_size={case.ep_size}, tp_size={case.tp_size}, "
            f"devices_used={case.ep_size * case.tp_size}/{len(jax.devices())}"
        )
        if token_mask_mode != "none":
            print(f"  token_mask: mode={token_mask_mode}, valid_ratio={token_valid_ratio}")
        data = prepare_fused_moe_inputs(
            case,
            weight_dtype=weight_dtype,
            mesh=mesh,
            include_weights=False,
            include_shared_expert=use_shared_expert,
        )

        print(f"{imbalance_mode=}")
        if use_grouped_topk:
            custom_logits, sim_stats = MoEImbalanceSimulator.create_grouped_topk_logits(
                case.num_tokens,
                case.num_experts,
                case.top_k,
                num_groups=case.num_expert_group,
                top_k_groups=case.topk_group,
                mode=imbalance_mode,
                seed=int(case.seed) + 42,
                alpha=alpha,
                zipf_s=zipf_s,
                hotspot_ratio=hotspot_ratio,
                hotspot_count=hotspot_count,
                zero_expert_count=zero_expert_count,
                non_hotspot_alpha=non_hotspot_alpha,
            )
            print(f"  imbalance(sim): {sim_stats}")
        else:
            target_counts = MoEImbalanceSimulator.generate_counts(
                case.num_tokens,
                case.top_k,
                case.num_experts,
                mode=imbalance_mode,
                alpha=alpha,
                zipf_s=zipf_s,
                hotspot_ratio=hotspot_ratio,
                hotspot_count=hotspot_count,
                zero_expert_count=zero_expert_count,
                non_hotspot_alpha=non_hotspot_alpha,
            )
            custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
                case.num_tokens, case.num_experts, case.top_k, target_counts
            )

        data["router_logits"] = jax.device_put(
            custom_logits, jax.sharding.NamedSharding(mesh, P("tensor", None))
        )
        token_valid_mask: jax.Array | None = None
        if token_mask_mode != "none":
            num_valid = int(round(case.num_tokens * token_valid_ratio))
            num_valid = max(0, min(case.num_tokens, num_valid))
            mask_np = np.zeros((case.num_tokens,), dtype=np.int32)
            if token_mask_mode == "prefix":
                num_invalid = case.num_tokens - num_valid
                if num_valid:
                    mask_np[num_invalid:] = 1
            else:
                rng = np.random.default_rng(token_mask_seed + case.seed)
                if num_valid:
                    valid_idx = rng.choice(case.num_tokens, size=num_valid, replace=False)
                    mask_np[valid_idx] = 1
            token_valid_mask = jax.device_put(
                jnp.asarray(mask_np),
                jax.sharding.NamedSharding(
                    mesh,
                    P(
                        "tensor",
                    ),
                ),
            )
        # Determine quant_block_k for FP8 quantization
        if quant_block_k_override is not None:
            quant_block_k = quant_block_k_override
        else:
            quant_block_k = 256 if weight_dtype == jnp.float8_e4m3fn else None

        if weight_dtype == jnp.float8_e4m3fn:
            quantization_config = QuantizationConfig(
                moe_weight_dtype=weight_dtype,
                moe_activation_dtype=None,  # activation is bfloat16
            )
        else:
            quantization_config = None

        with jax.set_mesh(mesh):
            all_disable = _env_bool("FUSED_MOE_BENCHMARK_ALL_DISABLE", False)
            fused_layer = FusedEPMoE(
                hidden_size=case.hidden_size,
                num_experts=case.num_experts,
                num_experts_per_tok=case.top_k,
                ep_size=case.ep_size,
                mesh=mesh,
                intermediate_dim=case.intermediate_size,
                weight_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                activation=case.activation,
                layer_id=0,
                renormalize_topk_logits=case.renormalize_topk_logits,
                use_grouped_topk=use_grouped_topk,
                num_groups=case.num_expert_group if use_grouped_topk else 1,
                top_k_groups=case.topk_group if use_grouped_topk else 1,
                num_shared_experts=1 if use_shared_expert else 0,
                moe_shared_expert_intermediate_size=(
                    case.intermediate_size if use_shared_expert else None
                ),
                quantization_config=quantization_config,
                # Env helpers:
                # - Set `FUSED_MOE_BENCHMARK_ALL_DISABLE=1` to disable all major stages.
                # - Any specific `FUSED_MOE_BENCHMARK_DISABLE_*` overrides ALL_DISABLE.
                disable_a2a=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_A2A",
                    all_disable=all_disable,
                ),
                disable_dynamic_ffn1=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1",
                    all_disable=all_disable,
                ),
                disable_dynamic_ffn2=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2",
                    all_disable=all_disable,
                ),
                disable_weight_load=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD",
                    all_disable=all_disable,
                ),
                disable_a2a_s_tile_read=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ",
                    all_disable=all_disable,
                ),
                disable_a2a_s_acc_tile_write=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE",
                    all_disable=all_disable,
                ),
                disable_shared_expert=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT",
                    all_disable=all_disable,
                ),
                disable_all_reduce_metadata=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA",
                    all_disable=all_disable,
                ),
                disable_sync_barrier=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER",
                    all_disable=all_disable,
                ),
            )
            if quantization_config is not None:
                if quant_block_k is not None:
                    fused_layer.quant_block_k = quant_block_k
                fused_layer.quantize_weights()

            block_cfgs: list[FusedMoEBlockConfig | None]
            if tune_block_config:
                case_excl_key = (
                    case.num_tokens,
                    case.num_experts,
                    case.top_k,
                    case.hidden_size,
                    case.intermediate_size,
                    case.ep_size,
                )
                block_cfgs = select_block_configs(
                    case,
                    dtype,
                    weight_dtype=weight_dtype,
                    router_dtype=data["router_logits"].dtype,
                    bt_candidates=bt_candidates or [2, 4, 8, 16, 32, 64, 128, 256, 512],
                    bts_candidates=bts_candidates,
                    bf_candidates=bf_candidates or [128, 256, 512, 1024, 2048],
                    bd_candidates=bd_candidates or [256, 512, 1024, 2048, 4096, 8192],
                    bse_candidates=bse_candidates,
                    tpu_vmem_budget_bytes=tpu_vmem_budget_bytes,
                    tpu_vmem_headroom_ratio=tpu_vmem_headroom_ratio,
                    tpu_vmem_estimate_scale=tpu_vmem_estimate_scale,
                    max_configs=max_configs,
                    use_shared_expert=use_shared_expert,
                    quant_block_k=quant_block_k,
                    excluded_configs=EXCLUDED_BLOCK_CONFIGS.get(case_excl_key),
                )
            else:
                block_cfgs = [None]

            topk_module = TopK(
                topk=case.top_k,
                renormalize=case.renormalize_topk_logits,
                num_expert_group=case.num_expert_group if use_grouped_topk else 0,
                topk_group=case.topk_group if use_grouped_topk else 0,
                routed_scaling_factor=case.routed_scaling_factor,
                layer_id=0,
            )

            moe_def, moe_state = nnx.split(fused_layer)
            moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)

            topk_def, topk_state = nnx.split(topk_module)
            topk_state_leaves, topk_state_def = jax.tree_util.tree_flatten(topk_state)

            @partial(
                jax.jit,
                static_argnames=("moe_state_def", "topk_state_def", "block_config"),
                compiler_options=_tpu_log_recorder_compiler_options(),
            )
            def run_no_mask(
                tokens,
                router_logits,
                *,
                moe_state_def,
                moe_state_leaves,
                topk_state_def,
                topk_state_leaves,
                block_config,
            ):
                moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
                moe = nnx.merge(moe_def, moe_state)
                topk_state = jax.tree_util.tree_unflatten(topk_state_def, topk_state_leaves)
                topk = nnx.merge(topk_def, topk_state)

                topk_weights, topk_ids = topk(router_logits)
                return moe(tokens, topk_weights, topk_ids, block_config=block_config)

            @partial(
                jax.jit,
                static_argnames=("moe_state_def", "topk_state_def", "block_config"),
                compiler_options=_tpu_log_recorder_compiler_options(),
            )
            def run_with_mask(
                tokens,
                router_logits,
                token_valid_mask,
                *,
                moe_state_def,
                moe_state_leaves,
                topk_state_def,
                topk_state_leaves,
                block_config,
            ):
                moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
                moe = nnx.merge(moe_def, moe_state)
                topk_state = jax.tree_util.tree_unflatten(topk_state_def, topk_state_leaves)
                topk = nnx.merge(topk_def, topk_state)

                topk_weights, topk_ids = topk(router_logits)
                topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)
                return moe(
                    tokens,
                    topk_weights,
                    topk_ids,
                    block_config=block_config,
                )

            best: tuple[float, FusedMoEBlockConfig | None] | None = None
            default_ms: float | None = None
            for i, block_cfg in enumerate(block_cfgs):
                tag = "default" if block_cfg is None else str(i)
                if block_cfg is None:
                    print("  fused_moe blocks] -> (block_config=None)")
                else:
                    print(
                        f"  fused_moe blocks [{i+1}/{len(block_cfgs)}] -> {block_cfg.as_kwargs()}"
                    )
                    vmem_bytes = _estimate_vmem_bytes(
                        case,
                        jnp.bfloat16,
                        weight_dtype,
                        data["router_logits"].dtype,
                        block_cfg,
                        intermediate_size=case.intermediate_size,
                        hidden_size=case.hidden_size,
                        use_shared_expert=use_shared_expert,
                        quant_block_k=quant_block_k,
                        verbose=True,
                    )
                    vmem_mb = (vmem_bytes * tpu_vmem_estimate_scale) / (1024 * 1024)
                    budget_mb = tpu_vmem_budget_bytes / (1024 * 1024)
                    effective_budget_mb = budget_mb * tpu_vmem_headroom_ratio
                    vmem_remaining_mb = effective_budget_mb - vmem_mb
                    print(
                        "    => VMEM: "
                        f"{vmem_mb:.2f} MB / effective {effective_budget_mb:.2f} MB "
                        f"(raw budget: {budget_mb:.0f} MB, remaining: {vmem_remaining_mb:.2f} MB)"
                    )

                task = "fused-moe-k_.*"

                def _compute(block_cfg=block_cfg):
                    if token_valid_mask is None:
                        return run_no_mask(
                            data["tokens"],
                            data["router_logits"],
                            moe_state_def=moe_state_def,
                            moe_state_leaves=moe_state_leaves,
                            topk_state_def=topk_state_def,
                            topk_state_leaves=topk_state_leaves,
                            block_config=block_cfg,
                        )
                    return run_with_mask(
                        data["tokens"],
                        data["router_logits"],
                        token_valid_mask,
                        moe_state_def=moe_state_def,
                        moe_state_leaves=moe_state_leaves,
                        topk_state_def=topk_state_def,
                        topk_state_leaves=topk_state_leaves,
                        block_config=block_cfg,
                    )

                try:
                    if block_cfg is not None:
                        validate_fused_moe_block_config(
                            num_tokens=case.num_tokens,
                            num_experts=case.num_experts,
                            top_k=case.top_k,
                            hidden_size=case.hidden_size,
                            intermediate_size=case.intermediate_size,
                            dtype=dtype,
                            ep_size=mesh_ep,
                            quant_block_k=quant_block_k,
                            block_config=block_cfg,
                        )
                    times = multiple_iteration_timeit_from_trace(
                        compute_func=_compute,
                        data_generator=lambda: (),
                        task=task,
                        tries=iters,
                        warmup=warmup_iters,
                    )
                except ValueError as e:
                    print(f"SKIP fused_moe blocks [{i+1}/{len(block_cfgs)}], reason: {e}")
                    continue
                except SystemExit as e:
                    # In some TPU environments stderr isn't captured/aggregated, and some internal
                    # errors can surface as SystemExit(1). Print the traceback to stdout so it's
                    # visible in logs.
                    print(
                        f"ERROR fused_moe blocks [{i+1}/{len(block_cfgs)}]: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
                    raise
                except Exception as e:
                    # Some failures (e.g., TPU compilation/runtime issues or missing profiler trace
                    # output) can surface as non-ValueError exceptions. Print the full traceback to
                    # stdout so benchmark runs don't appear to exit silently.
                    print(
                        f"ERROR fused_moe blocks [{i+1}/{len(block_cfgs)}]: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
                    continue
                if len(times) > 1:
                    times = times[1:]
                mean_ms = float(np.mean(times)) if times else float("nan")
                print(f"     fused_moe[{tag}]: {mean_ms:.3f} ms (trace) | samples={times}")
                if block_cfg is None:
                    default_ms = mean_ms
                if tune_block_config and np.isfinite(mean_ms):
                    if best is None or mean_ms < best[0]:
                        best = (mean_ms, block_cfg)

            if tune_block_config and best is not None:
                best_ms, best_cfg = best
                if best_cfg is None:
                    print(f"  best: default ({best_ms:.3f} ms)")
                else:
                    device_name = get_device_name()
                    table_key = (
                        jnp.dtype(dtype).name,
                        jnp.dtype(weight_dtype).name,
                        case.num_tokens,
                        case.num_experts,
                        case.top_k,
                        case.hidden_size,
                        case.intermediate_size,
                        case.ep_size,
                        use_shared_expert,
                        use_grouped_topk,
                    )
                    cfg_tuple = (
                        best_cfg.bt,
                        best_cfg.bf,
                        best_cfg.bd1,
                        best_cfg.bd2,
                        best_cfg.bts if best_cfg.bts is not None else best_cfg.bt,
                        best_cfg.btc,
                        best_cfg.bfc,
                        best_cfg.bd1c,
                        best_cfg.bd2c,
                        best_cfg.bse,
                    )
                    print(f"  best: {best_cfg.as_kwargs()} ({best_ms:.3f} ms)")
                    print(f"  tuned_table[{device_name}][{table_key}] = {cfg_tuple}")
                    per_device = tuned_results.setdefault(device_name, {})
                    if table_key in per_device and per_device[table_key] != cfg_tuple:
                        print(
                            f"  overwrite tuned entry: {device_name}[{table_key}] "
                            f"{per_device[table_key]} -> {cfg_tuple}"
                        )
                    per_device[table_key] = cfg_tuple
            if return_results:
                best_ms: float | None
                best_cfg: FusedMoEBlockConfig | None
                if tune_block_config:
                    if best is None:
                        best_ms, best_cfg = float("nan"), None
                    else:
                        best_ms, best_cfg = best
                else:
                    best_ms, best_cfg = default_ms, None
                results.append(
                    {
                        "case": case.name,
                        "num_tokens": case.num_tokens,
                        "num_experts": case.num_experts,
                        "top_k": case.top_k,
                        "hidden_size": case.hidden_size,
                        "intermediate_size": case.intermediate_size,
                        "ep_size": case.ep_size,
                        "best_ms": best_ms,
                        "best_cfg": best_cfg.as_kwargs() if best_cfg is not None else None,
                    }
                )

    if tune_block_config and tuned_results:
        print("\n# --- Copy/paste into tuned_block_configs.py ---")
        for device_name in sorted(tuned_results.keys()):
            entries = tuned_results[device_name]
            print(f'TUNED_BLOCK_CONFIGS.setdefault("{device_name}", {{}}).update({{')
            for k in sorted(
                entries.keys(),
                key=lambda t: (t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[0]),
            ):
                print(f"    {k}: {entries[k]},")
            print("})\n")

    if return_results:
        return results
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fused_moe.")
    parser.add_argument("--iters", type=int, default=3, help="Number of benchmark iterations.")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Number of warmup iterations before profiling (per case / block_config).",
    )
    parser.add_argument(
        "--weight-dtype",
        type=str,
        default="bfloat16",
        help="Data type to benchmark.",
        choices=["bfloat16", "float8_e4m3fn"],
    )
    parser.add_argument(
        "--tune-block-config",
        action="store_true",
        help="Benchmark multiple block_config variants and print the best tuned table entry.",
    )
    parser.add_argument(
        "--bt-candidates",
        type=int,
        nargs="+",
        help=(
            "Candidate list for bt (outer token tile size). "
            "Note: bt is per-core; local_num_tokens = num_tokens / ep_size. "
            "The kernel uses output_bt=gcd(bt, local_num_tokens) for run_bt tiling."
        ),
    )
    parser.add_argument(
        "--bts-candidates",
        type=int,
        nargs="+",
        help=(
            "Candidate list for bts (token staging tile inside expert_ffn). "
            "When omitted, bts is auto-searched as a <=bt ladder (bt, bt/2, bt/4, ...) "
            "to fit within the VMEM budget (bts must be <= bt). "
            "Example: --bts-candidates 8 16 32 64"
        ),
    )
    parser.add_argument(
        "--bf-candidates",
        type=int,
        nargs="+",
        help="Candidate list for bf (intermediate tile), e.g. --bf-candidates 512 1024 2048",
    )
    parser.add_argument(
        "--bd-candidates",
        type=int,
        nargs="+",
        help="Candidate list for bd1/bd2 (hidden tile), e.g. --bd-candidates 512 1024 2048 4096",
    )
    parser.add_argument(
        "--bse-candidates",
        type=int,
        nargs="+",
        help="Candidate list for bse (shared expert intermediate tile), e.g. --bse-candidates 128 256 512",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Token counts to benchmark (e.g. --num-tokens 128 512 4096). Default: a fixed ladder.",
    )
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument(
        "--renormalize-topk-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Renormalize top-k routing weights/logits.",
    )
    parser.add_argument("--num-expert-group", type=int, default=0)
    parser.add_argument("--topk-group", type=int, default=0)
    parser.add_argument(
        "--tpu-vmem-budget-mb",
        type=int,
        default=DEFAULT_TPU_VMEM_BUDGET_MB,
        help="VMEM budget used to filter candidate block configs (MiB).",
    )
    parser.add_argument(
        "--tpu-vmem-headroom-ratio",
        type=float,
        default=0.90,
        help="Fraction of the VMEM budget exposed to the estimator after headroom reservation.",
    )
    parser.add_argument(
        "--tpu-vmem-estimate-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the VMEM estimate before candidate filtering.",
    )
    parser.add_argument(
        "--compilation-cache-dir",
        type=str,
        default=None,
        help="Optional JAX compilation cache directory to reuse compiled executables across runs.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=9,
        help="Maximum number of block configs to benchmark per case when --tune-block-config is set.",
    )
    # Feature flags
    parser.add_argument(
        "--use-shared-expert",
        action="store_true",
        help="Enable shared expert logic (allocates extra VMEM buffers).",
    )

    parser.add_argument(
        "--imbalance-mode",
        type=str,
        choices=["balanced", "dirichlet", "zipf", "hotspot", "sparse_hotspot"],
        default="balanced",
        help="All-to-all imbalance mode.",
    )
    parser.add_argument(
        "--quant-block-k",
        type=int,
        default=None,
        help="Sub-channel quantization block size (default: 256 for FP8, None for bf16).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Dirichlet concentration (smaller => more imbalanced, e.g. 0.1).",
    )
    parser.add_argument(
        "--zipf-s", type=float, default=1.1, help="Zipf exponent (larger => more imbalanced)."
    )
    parser.add_argument(
        "--hotspot-ratio",
        type=float,
        default=0.5,
        help="Fraction of tokens routed to hotspot experts (0.0-1.0).",
    )
    parser.add_argument("--hotspot-count", type=int, default=1, help="Number of hotspot experts.")
    parser.add_argument("--zero-expert-count", type=int, default=0)
    parser.add_argument("--non-hotspot-alpha", type=float, default=100.0)
    parser.add_argument(
        "--token-mask-mode",
        type=str,
        choices=["none", "prefix", "random"],
        default="none",
        help="Optional token_valid_mask pattern for exercising invalid-token logic.",
    )
    parser.add_argument(
        "--token-valid-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of tokens marked valid (0.0-1.0). "
            "If < 1.0 and --token-mask-mode is not set, defaults to --token-mask-mode=random."
        ),
    )
    parser.add_argument(
        "--token-valid-ratios",
        type=float,
        nargs="+",
        default=None,
        help=(
            "List of token_valid_ratio values to sweep in a single run. "
            "Overrides --token-valid-ratio when provided."
        ),
    )
    parser.add_argument(
        "--token-mask-seed",
        type=int,
        default=0,
        help="RNG seed for --token-mask-mode=random (combined with case.seed).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # In some TPU environments only stdout is captured/aggregated. Route fatal Python
    # traces (e.g., segfaults, timeouts) to stdout to avoid "silent" exits.
    try:
        faulthandler.enable(file=sys.stdout, all_threads=True)
    except Exception:
        pass
    args = parse_args()
    if args.token_valid_ratios is None:
        if args.token_mask_mode == "none" and args.token_valid_ratio < 1.0:
            args.token_mask_mode = "random"
    DTYPE_MAP = {
        "int8": jnp.int8,
        "float8_e4m3fn": jnp.float8_e4m3fn,
        "float8_e5m2": jnp.float8_e5m2,
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        None: None,
    }
    if args.weight_dtype not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype: {args.weight_dtype}. Supported: {list(DTYPE_MAP.keys())}"
        )
    weight_dtype = DTYPE_MAP[args.weight_dtype]
    if args.compilation_cache_dir:
        _compilation_cache.set_cache_dir(args.compilation_cache_dir)
    tpu_vmem_budget_bytes = int(args.tpu_vmem_budget_mb) * 1024 * 1024
    full_args_dict = vars(args)
    try:

        def _resolve_token_mask_mode(ratio: float, mode: str) -> str:
            if mode == "none" and ratio < 1.0:
                return "random"
            return mode

        ratios = args.token_valid_ratios or [args.token_valid_ratio]
        all_results: list[tuple[float, str, list[dict[str, object]]]] = []
        for ratio in ratios:
            token_mask_mode = _resolve_token_mask_mode(ratio, args.token_mask_mode)
            if len(ratios) > 1:
                print(f"\n# --- token_valid_ratio={ratio} (token_mask_mode={token_mask_mode}) ---")
            results = run_all(
                args.iters,
                weight_dtype=weight_dtype,
                warmup_iters=args.warmup_iters,
                tune_block_config=args.tune_block_config,
                bt_candidates=args.bt_candidates,
                bts_candidates=args.bts_candidates,
                bf_candidates=args.bf_candidates,
                bd_candidates=args.bd_candidates,
                bse_candidates=args.bse_candidates,
                num_tokens=args.num_tokens,
                num_experts=args.num_experts,
                top_k=args.top_k,
                hidden_size=args.hidden_size,
                intermediate_size=args.intermediate_size,
                activation=args.activation,
                renormalize_topk_logits=args.renormalize_topk_logits,
                num_expert_group=args.num_expert_group,
                topk_group=args.topk_group,
                tpu_vmem_budget_bytes=tpu_vmem_budget_bytes,
                tpu_vmem_headroom_ratio=args.tpu_vmem_headroom_ratio,
                tpu_vmem_estimate_scale=args.tpu_vmem_estimate_scale,
                max_configs=args.max_configs,
                use_shared_expert=args.use_shared_expert,
                use_grouped_topk=None,
                imbalance_mode=args.imbalance_mode,
                alpha=args.alpha,
                zipf_s=args.zipf_s,
                hotspot_ratio=args.hotspot_ratio,
                hotspot_count=args.hotspot_count,
                zero_expert_count=args.zero_expert_count,
                non_hotspot_alpha=args.non_hotspot_alpha,
                token_mask_mode=token_mask_mode,
                token_valid_ratio=ratio,
                token_mask_seed=args.token_mask_seed,
                quant_block_k_override=args.quant_block_k,
                return_results=True,
            )
            all_results.append((ratio, token_mask_mode, results))

        if len(all_results) > 1:
            print("\n# === token_valid_ratio summary ===")
            print("ratio | token_mask_mode | case | best_ms")
            for ratio, token_mask_mode, results in all_results:
                best_values = []
                for row in results:
                    best_ms = row.get("best_ms")
                    if isinstance(best_ms, (int, float)) and np.isfinite(best_ms):
                        best_values.append(float(best_ms))
                    best_str = (
                        f"{best_ms:.3f}"
                        if isinstance(best_ms, (int, float)) and np.isfinite(best_ms)
                        else "nan"
                    )
                    print(f"{ratio} | {token_mask_mode} | {row['case']} | {best_str}")
                if best_values:
                    avg_ms = float(np.mean(best_values))
                    print(f"{ratio} | {token_mask_mode} | __avg__ | {avg_ms:.3f}")
    except BaseException as e:
        print(f"FATAL: {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
