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
from sgl_jax.srt.kernels.fused_moe.v1.vmem_estimator import (
    DEFAULT_TPU_VMEM_BUDGET_BYTES,
    DEFAULT_TPU_VMEM_BUDGET_MB,
    dtype_packing_32bit,
    estimate_fused_moe_vmem_bytes,
    format_vmem_bytes_breakdown,
    fused_moe_vmem_breakdown_bytes,
)
from sgl_jax.srt.layers.moe import FusedEPMoE


# Env var helpers for kernel stage ablations.
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


def _dtype_packing(dtype: jnp.dtype) -> int:
    return dtype_packing_32bit(dtype)


def _estimate_vmem_bytes(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    router_dtype: jnp.dtype,
    cfg: FusedMoEBlockConfig,
    intermediate_size: int,
    hidden_size: int,
    use_shared_expert: bool = False,
    subc_quant_wsz: int | None = None,
    verbose: bool = False,
) -> int:
    total = estimate_fused_moe_vmem_bytes(
        num_tokens=case.num_tokens,
        num_experts=case.num_experts,
        top_k=case.top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=case.ep_size,
        dtype=dtype,
        weight_dtype=weight_dtype,
        router_dtype=router_dtype,
        cfg=cfg,
        use_shared_expert=use_shared_expert,
        subc_quant_wsz=subc_quant_wsz,
        include_routing_temporaries=True,
    )
    if verbose:
        total2, items = fused_moe_vmem_breakdown_bytes(
            num_tokens=case.num_tokens,
            num_experts=case.num_experts,
            top_k=case.top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            ep_size=case.ep_size,
            dtype=dtype,
            weight_dtype=weight_dtype,
            router_dtype=router_dtype,
            cfg=cfg,
            use_shared_expert=use_shared_expert,
            subc_quant_wsz=subc_quant_wsz,
            include_routing_temporaries=True,
        )
        # total2 should match `total`; keep total as the source of truth.
        print(format_vmem_bytes_breakdown(total=total2, items=items))
    return total


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
    max_configs: int,
    use_shared_expert: bool = False,
    subc_quant_wsz: int | None = None,
    excluded_configs: set[tuple[int, ...]] | None = None,
) -> list[FusedMoEBlockConfig]:
    """Enumerate block configs from the explicit candidate lists."""
    t_packing = _dtype_packing(dtype)
    tile_align = t_packing * 128
    local_num_tokens = case.num_tokens // case.ep_size

    def _ladder_div2(start: int) -> list[int]:
        out: list[int] = []
        v = int(start)
        while v > 0:
            out.append(v)
            if v == 1:
                break
            v //= 2
        return sorted(set(out), reverse=True)

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

    def _effective_bt(raw_bt: int) -> int:
        bt = min(int(raw_bt), int(local_num_tokens))
        return math.gcd(bt, int(local_num_tokens))

    # Apply the same clamp rule as the kernel (`effective_for`) to avoid
    # generating (bts, btc) pairs that become invalid after bt/bts clamping.
    bt_candidates = [
        v
        for v in (
            _effective_bt(raw) for raw in _pick_candidates(candidates=bt_candidates, multiple_of=1)
        )
        if v > 0 and _bt_allowed(v)
    ]
    bt_candidates = sorted(set(bt_candidates))
    bts_candidates_i: list[int] | None
    if bts_candidates is None:
        bts_candidates_i = None
    else:
        bts_candidates_i = _pick_candidates(candidates=list(bts_candidates), multiple_of=1)
    bf_candidates = _pick_candidates(candidates=bf_candidates, multiple_of=128)
    bd_candidates = _pick_candidates(candidates=bd_candidates, multiple_of=tile_align)

    # If sub-channel quantization is enabled, fused_moe will override `bfc/bd1c`
    # to fixed values. Pre-filter candidates so we don't enumerate configs that
    # are guaranteed to fail validation (e.g. bf < bfc, bd1 < bd1c).
    if subc_quant_wsz is not None:
        quant_bfc = int(subc_quant_wsz)
        quant_bd1c = int(subc_quant_wsz) * int(t_packing)
        bf_candidates = [bf for bf in bf_candidates if bf >= quant_bfc and bf % quant_bfc == 0]
        bd_candidates = [bd for bd in bd_candidates if bd >= quant_bd1c and bd % quant_bd1c == 0]

    raw_bse_candidates = bse_candidates if bse_candidates is not None else bf_candidates
    bse_candidates_i = _pick_candidates(candidates=raw_bse_candidates, multiple_of=128)

    def validate(cfg: FusedMoEBlockConfig) -> tuple[bool, str]:
        try:
            validate_fused_moe_block_config(
                num_tokens=case.num_tokens,
                num_experts=case.num_experts,
                top_k=case.top_k,
                hidden_size=case.hidden_size,
                intermediate_size=case.intermediate_size,
                dtype=dtype,
                ep_size=case.ep_size,
                subc_quant_wsz=subc_quant_wsz,
                block_config=cfg,
            )
        except ValueError as e:
            return False, str(e)

        est = _estimate_vmem_bytes(
            case,
            dtype,
            weight_dtype,
            router_dtype,
            cfg,
            intermediate_size=case.intermediate_size,
            hidden_size=case.hidden_size,
            use_shared_expert=use_shared_expert,
            subc_quant_wsz=subc_quant_wsz,
        )
        if est > tpu_vmem_budget_bytes:
            return (
                False,
                f"vmem_est={est / (1024 * 1024):.1f}MB > budget={tpu_vmem_budget_bytes / (1024 * 1024):.1f}MB",
            )

        # Keep bt filtering local to the benchmark to avoid compile-time errors
        # from known Mosaic tiling constraints on certain DMA slices.
        bt = cfg.bt
        if not _bt_allowed(bt):
            return False, f"bt({bt}) must be 2, 4, 8, or a multiple of 8"

        return True, "ok"

    configs: list[FusedMoEBlockConfig] = []
    seen: set[tuple[int, ...]] = set()

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
            #
            # Use a small power-of-two neighborhood to keep the search space bounded and
            # avoid odd tile sizes that can trigger Mosaic/DMA alignment pitfalls.
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
            # When explicitly provided, allow `bts` to exceed `bt` to better match
            # the post-routing per-expert token dimension (dyn_sz). The kernel
            # clamps `bts` to at most `bt * ep_size`.
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
                            # Important: include subc_quant_wsz so the effective
                            # config matches what fused_ep_moe will compile with.
                            effective = raw.effective_for(
                                num_tokens=case.num_tokens,
                                ep_size=case.ep_size,
                                dtype=dtype,
                                subc_quant_wsz=subc_quant_wsz,
                            )
                            add(raw=raw, effective=effective)

    if max_configs <= 0:
        raise ValueError(f"Expected {max_configs=} to be > 0.")

    if len(configs) <= max_configs:
        return configs

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

    # Deterministic selection: prefer larger (more aggressive) tiling configs.
    configs = sorted(configs, key=score, reverse=True)
    selected = configs[:max_configs]
    print(f"  limit: {len(configs)} valid configs -> {len(selected)} selected (max={max_configs})")
    return selected


def run_all(
    iters: int,
    dtype: jnp.dtype = jnp.bfloat16,
    weight_dtype: jnp.dtype = jnp.bfloat16,  # Quantize the weight dtype, the activation's dtype always is bfloat16
    *,
    router_dtype: jnp.dtype = jnp.float32,
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
    token_valid_ratio: list[float] | None = None,
    token_valid_mode: str = "prefix",
    token_valid_seed: int = 0,
) -> None:
    if use_grouped_topk is None:
        use_grouped_topk = bool(num_expert_group or topk_group)

    token_valid_mode = (token_valid_mode or "prefix").lower()
    if token_valid_mode not in ("prefix", "random"):
        raise ValueError(f"Unsupported {token_valid_mode=}. Expected prefix|random.")

    ratios = [1.0] if token_valid_ratio is None else [float(r) for r in token_valid_ratio]
    for r in ratios:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"Expected token_valid_ratio in [0, 1], got {r=}.")

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
        return

    tuned_results: dict[str, dict[tuple, tuple[int, int, int, int, int, int, int, int, int]]] = {}
    tuned_results_ms: dict[str, dict[tuple, float]] = {}
    if tune_block_config:
        from sgl_jax.srt.utils.jax_utils import get_device_name

    print(f"Running fused_moe benchmarks with weight_dtype={weight_dtype}")
    print(f"  features: shared_expert={use_shared_expert}, grouped_topk={use_grouped_topk}")
    print(
        "  shape: "
        f"num_experts={num_experts}, top_k={top_k}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
        f"activation={activation}, renormalize_topk_logits={renormalize_topk_logits}, "
        f"num_expert_group={num_expert_group}, topk_group={topk_group}"
    )

    for case in cases:
        mesh = build_mesh(ep_size=case.ep_size, tp_size=case.tp_size)
        mesh_ep = mesh.shape["tensor"]
        if mesh_ep != case.ep_size:
            print(f"warning [case={case.name}] mesh_ep={mesh_ep} != case.ep_size={case.ep_size}")
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}, ep_size={case.ep_size}"
        )
        print(
            f"  mesh: ep_size={case.ep_size}, tp_size={case.tp_size}, "
            f"devices_used={case.ep_size * case.tp_size}/{len(jax.devices())}"
        )
        if any(r != 1.0 for r in ratios):
            ratios_str = ", ".join(f"{r:g}" for r in ratios[:12])
            suffix = "" if len(ratios) <= 12 else f" (+{len(ratios) - 12} more)"
            print(f"  token_valid: ratios=[{ratios_str}]{suffix}, mode={token_valid_mode}")
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
        # Keep router dtype consistent with the real model path (e.g. BailingMoE often uses fp32).
        custom_logits = custom_logits.astype(router_dtype)

        data["router_logits"] = jax.device_put(
            custom_logits, jax.sharding.NamedSharding(mesh, P(("data", "tensor"), None))
        )
        # Match the fused kernel's shard_map in_specs to avoid per-call reshard/copies.
        kernel_io_sharding = jax.sharding.NamedSharding(mesh, P(("data", "tensor"), None))
        data["tokens"] = jax.sharding.reshard(data["tokens"], kernel_io_sharding)
        data["router_logits"] = jax.sharding.reshard(data["router_logits"], kernel_io_sharding)
        data["tokens"].block_until_ready()
        data["router_logits"].block_until_ready()
        # Determine subc_quant_wsz for FP8 quantization
        subc_quant_wsz = 256 if weight_dtype == jnp.float8_e4m3fn else None

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
                max_configs=max_configs,
                use_shared_expert=use_shared_expert,
                subc_quant_wsz=subc_quant_wsz,
                excluded_configs=EXCLUDED_BLOCK_CONFIGS.get(case_excl_key),
            )
        else:
            block_cfgs = [None]

        if weight_dtype == jnp.float8_e4m3fn:
            quantization_config = QuantizationConfig(
                moe_weight_dtype=weight_dtype,
                moe_activation_dtype=None,  # activation is bfloat16
            )
        else:
            quantization_config = None

        with jax.set_mesh(mesh):
            all_disable = _env_bool("FUSED_MOE_BENCHMARK_ALL_DISABLE", False)
            # For FP8 runs we still initialize weights in BF16 and then quantize them,
            # because the fused kernel expects corresponding scale tensors.
            init_weight_dtype = jnp.bfloat16 if quantization_config is not None else weight_dtype
            fused_layer = FusedEPMoE(
                hidden_size=case.hidden_size,
                num_experts=case.num_experts,
                num_experts_per_tok=case.top_k,
                ep_size=case.ep_size,
                mesh=mesh,
                intermediate_dim=case.intermediate_size,
                weight_dtype=init_weight_dtype,
                dtype=dtype,
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
                disable_topk=_with_all_disable(
                    "FUSED_MOE_BENCHMARK_DISABLE_TOPK",
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
                fused_layer.quantize_weights()
            print(
                "  dtypes: "
                f"w1={fused_layer.w1[...].dtype}, "
                f"w2={fused_layer.w2[...].dtype}, "
                f"w3={fused_layer.w3[...].dtype}, "
                f"w_scales={'yes' if fused_layer.w1_scale is not None else 'no'}"
            )

            moe_def, moe_state = nnx.split(fused_layer)
            moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)

            # Ensure all large inputs (weights/logits/tokens) are already on device before
            # starting the timed region; otherwise async device_put/reshard can leak into the
            # first few measured steps and skew tuning decisions.
            def _block_until_ready_pytree(x):
                def _block(v):
                    try:
                        return v.block_until_ready()
                    except Exception:
                        return v

                return jax.tree_util.tree_map(_block, x)

            _block_until_ready_pytree(
                (
                    data["tokens"],
                    data["router_logits"],
                    moe_state_leaves,
                )
            )

            @partial(
                jax.jit,
                static_argnames=("moe_state_def", "block_config"),
                compiler_options=_tpu_log_recorder_compiler_options(),
            )
            def run_no_mask(
                tokens,
                router_logits,
                *,
                moe_state_def,
                moe_state_leaves,
                block_config,
            ):
                moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
                moe = nnx.merge(moe_def, moe_state)
                # Call the fused kernel directly to avoid including any extra output
                # resharding/collectives that the layer wrapper might add.
                from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe

                w1_shared = moe.w1_shared[...] if moe.w1_shared is not None else None
                w2_shared = moe.w2_shared[...] if moe.w2_shared is not None else None
                w3_shared = moe.w3_shared[...] if moe.w3_shared is not None else None
                w1_scale = moe.w1_scale[...] if moe.w1_scale is not None else None
                w2_scale = moe.w2_scale[...] if moe.w2_scale is not None else None
                w3_scale = moe.w3_scale[...] if moe.w3_scale is not None else None
                w1_shared_scale = (
                    moe.w1_shared_scale[...] if moe.w1_shared_scale is not None else None
                )
                w2_shared_scale = (
                    moe.w2_shared_scale[...] if moe.w2_shared_scale is not None else None
                )
                w3_shared_scale = (
                    moe.w3_shared_scale[...] if moe.w3_shared_scale is not None else None
                )
                subc_quant_wsz = moe.subc_quant_wsz if moe.subc_quant_wsz is not None else None

                return fused_ep_moe(
                    mesh=mesh,
                    tokens=tokens,
                    w1=moe.w1[...],
                    w2=moe.w2[...],
                    w3=moe.w3[...],
                    gating_output=router_logits,
                    bias=None,
                    top_k=moe.num_experts_per_tok,
                    use_grouped_topk=moe.use_grouped_topk,
                    num_groups=moe.num_groups,
                    top_k_groups=moe.top_k_groups,
                    renormalize_topk_logits=moe.renormalize_topk_logits,
                    routed_scaling_factor=moe.routed_scaling_factor,
                    act_fn=moe.activation,
                    block_config=block_config,
                    token_valid_mask=None,
                    disable_a2a=moe.disable_a2a,
                    disable_dynamic_ffn1=moe.disable_dynamic_ffn1,
                    disable_dynamic_ffn2=moe.disable_dynamic_ffn2,
                    disable_weight_load=moe.disable_weight_load,
                    disable_a2a_s_tile_read=moe.disable_a2a_s_tile_read,
                    disable_a2a_s_acc_tile_write=moe.disable_a2a_s_acc_tile_write,
                    disable_shared_expert=moe.disable_shared_expert,
                    disable_topk=moe.disable_topk,
                    disable_all_reduce_metadata=moe.disable_all_reduce_metadata,
                    disable_sync_barrier=moe.disable_sync_barrier,
                    subc_quant_wsz=subc_quant_wsz,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    w3_scale=w3_scale,
                    w1_shared=w1_shared,
                    w2_shared=w2_shared,
                    w3_shared=w3_shared,
                    w1_shared_scale=w1_shared_scale,
                    w2_shared_scale=w2_shared_scale,
                    w3_shared_scale=w3_shared_scale,
                    b1=None,
                    b2=None,
                    b3=None,
                    dp_axis_name="data",
                    tp_axis_name="tensor",
                )

            @partial(
                jax.jit,
                static_argnames=("moe_state_def", "block_config"),
                compiler_options=_tpu_log_recorder_compiler_options(),
            )
            def run_with_mask(
                tokens,
                router_logits,
                token_valid_mask,
                *,
                moe_state_def,
                moe_state_leaves,
                block_config,
            ):
                moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
                moe = nnx.merge(moe_def, moe_state)
                from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe

                w1_shared = moe.w1_shared[...] if moe.w1_shared is not None else None
                w2_shared = moe.w2_shared[...] if moe.w2_shared is not None else None
                w3_shared = moe.w3_shared[...] if moe.w3_shared is not None else None
                w1_scale = moe.w1_scale[...] if moe.w1_scale is not None else None
                w2_scale = moe.w2_scale[...] if moe.w2_scale is not None else None
                w3_scale = moe.w3_scale[...] if moe.w3_scale is not None else None
                w1_shared_scale = (
                    moe.w1_shared_scale[...] if moe.w1_shared_scale is not None else None
                )
                w2_shared_scale = (
                    moe.w2_shared_scale[...] if moe.w2_shared_scale is not None else None
                )
                w3_shared_scale = (
                    moe.w3_shared_scale[...] if moe.w3_shared_scale is not None else None
                )
                subc_quant_wsz = moe.subc_quant_wsz if moe.subc_quant_wsz is not None else None

                return fused_ep_moe(
                    mesh=mesh,
                    tokens=tokens,
                    w1=moe.w1[...],
                    w2=moe.w2[...],
                    w3=moe.w3[...],
                    gating_output=router_logits,
                    bias=None,
                    top_k=moe.num_experts_per_tok,
                    use_grouped_topk=moe.use_grouped_topk,
                    num_groups=moe.num_groups,
                    top_k_groups=moe.top_k_groups,
                    renormalize_topk_logits=moe.renormalize_topk_logits,
                    routed_scaling_factor=moe.routed_scaling_factor,
                    act_fn=moe.activation,
                    block_config=block_config,
                    token_valid_mask=token_valid_mask,
                    disable_a2a=moe.disable_a2a,
                    disable_dynamic_ffn1=moe.disable_dynamic_ffn1,
                    disable_dynamic_ffn2=moe.disable_dynamic_ffn2,
                    disable_weight_load=moe.disable_weight_load,
                    disable_a2a_s_tile_read=moe.disable_a2a_s_tile_read,
                    disable_a2a_s_acc_tile_write=moe.disable_a2a_s_acc_tile_write,
                    disable_shared_expert=moe.disable_shared_expert,
                    disable_topk=moe.disable_topk,
                    disable_all_reduce_metadata=moe.disable_all_reduce_metadata,
                    disable_sync_barrier=moe.disable_sync_barrier,
                    subc_quant_wsz=subc_quant_wsz,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    w3_scale=w3_scale,
                    w1_shared=w1_shared,
                    w2_shared=w2_shared,
                    w3_shared=w3_shared,
                    w1_shared_scale=w1_shared_scale,
                    w2_shared_scale=w2_shared_scale,
                    w3_shared_scale=w3_shared_scale,
                    b1=None,
                    b2=None,
                    b3=None,
                    dp_axis_name="data",
                    tp_axis_name="tensor",
                )

            best: tuple[float, FusedMoEBlockConfig | None] | None = None
            for ratio in ratios:
                num_valid = int(math.floor(case.num_tokens * ratio))
                num_valid = int(max(0, min(num_valid, case.num_tokens)))

                token_valid_mask: jax.Array | None = None
                if num_valid != case.num_tokens:
                    mask_np = np.zeros((case.num_tokens,), dtype=np.int32)
                    if token_valid_mode == "prefix":
                        num_invalid = case.num_tokens - num_valid
                        if num_valid:
                            mask_np[num_invalid:] = 1
                    else:
                        rng = np.random.default_rng(int(token_valid_seed) + int(case.seed))
                        if num_valid:
                            valid_idx = rng.choice(case.num_tokens, size=num_valid, replace=False)
                            mask_np[valid_idx] = 1
                    token_valid_mask = jax.device_put(
                        jnp.asarray(mask_np),
                        jax.sharding.NamedSharding(mesh, P(("data", "tensor"))),
                    )
                    token_valid_mask.block_until_ready()

                print(f"  token_valid: ratio={ratio:g}, num_valid={num_valid}/{case.num_tokens}")

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
                            subc_quant_wsz=subc_quant_wsz,
                            verbose=True,
                        )
                        vmem_mb = vmem_bytes / (1024 * 1024)
                        vmem_remaining_mb = 64.0 - vmem_mb
                        print(
                            f"    => VMEM: {vmem_mb:.2f} MB / 64 MB (remaining: {vmem_remaining_mb:.2f} MB)"
                        )

                    task = "fused-moe-k_.*"

                    def _compute(block_cfg=block_cfg, token_valid_mask=token_valid_mask):
                        if token_valid_mask is None:
                            return run_no_mask(
                                data["tokens"],
                                data["router_logits"],
                                moe_state_def=moe_state_def,
                                moe_state_leaves=moe_state_leaves,
                                block_config=block_cfg,
                            )
                        return run_with_mask(
                            data["tokens"],
                            data["router_logits"],
                            token_valid_mask,
                            moe_state_def=moe_state_def,
                            moe_state_leaves=moe_state_leaves,
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
                                subc_quant_wsz=None,
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
                        print(
                            f"ERROR fused_moe blocks [{i+1}/{len(block_cfgs)}]: "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )
                        print(traceback.format_exc(), flush=True)
                        raise
                    except Exception as e:
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
                        int(case.num_tokens),
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
                    per_device_ms = tuned_results_ms.setdefault(device_name, {})
                    prev_ms = per_device_ms.get(table_key)
                    if prev_ms is None or best_ms < prev_ms:
                        if table_key in per_device and per_device[table_key] != cfg_tuple:
                            print(
                                f"  overwrite tuned entry: {device_name}[{table_key}] "
                                f"{per_device[table_key]} ({prev_ms:.3f} ms) -> {cfg_tuple} ({best_ms:.3f} ms)"
                            )
                        per_device[table_key] = cfg_tuple
                        per_device_ms[table_key] = float(best_ms)

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
        "--router-dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help=(
            "Dtype of `router_logits` passed into the fused MoE kernel. "
            "Set to float32 to match BailingMoE GateLogit when router_dtype is unset/fp32."
        ),
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
            "When omitted, bts uses a small heuristic set around E[dyn_sz] ~= bt*ep_size*top_k/num_experts. "
            "When explicitly provided, bts may exceed bt (up to bt*ep_size) to better match the "
            "post-routing per-expert token dimension. "
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
        "--token-valid-ratio",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional ratio(s) in [0, 1] to apply token_valid_mask (keeps padded num_tokens fixed). "
            "Example: --token-valid-ratio 1.0 0.5 0.25"
        ),
    )
    parser.add_argument(
        "--token-valid-mode",
        type=str,
        choices=["prefix", "random"],
        default="prefix",
        help="How to place valid tokens when --token-valid-ratio < 1.",
    )
    parser.add_argument(
        "--token-valid-seed",
        type=int,
        default=0,
        help="RNG seed for --token-valid-mode=random (combined with case.seed).",
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
    if args.router_dtype not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported router dtype: {args.router_dtype}. Supported: {list(DTYPE_MAP.keys())}"
        )
    router_dtype = DTYPE_MAP[args.router_dtype]
    if args.compilation_cache_dir:
        _compilation_cache.set_cache_dir(args.compilation_cache_dir)
    tpu_vmem_budget_bytes = int(args.tpu_vmem_budget_mb) * 1024 * 1024
    full_args_dict = vars(args)
    try:
        run_all(
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
            token_valid_ratio=args.token_valid_ratio,
            token_valid_mode=args.token_valid_mode,
            token_valid_seed=args.token_valid_seed,
            router_dtype=router_dtype,
        )
    except BaseException as e:
        print(f"FATAL: {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
