"""
Benchmark fused_moe kernel with grouped-GEMM-like MoE shapes.

Usage:
    python -m benchmark.moe.bench_fused_moe [--tune-block-config] --config-mode [base|mini|all] [--use-shared-expert]
"""

from __future__ import annotations

import argparse
import faulthandler
import math
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as _compilation_cache
from jax.sharding import PartitionSpec as P

from benchmark.moe.utils import (
    BAILING_BASE,
    BASE_CASES,
    GROUP_GEMM_CASES,
    MINI_CASES,
    MoEBenchmarkCase,
    MoEImbalanceSimulator,
    build_mesh,
    prepare_fused_moe_inputs,
    select_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.kernels.fused_moe.v1.kernel import (
    FusedMoEBlockConfig,
    validate_fused_moe_block_config,
)
from sgl_jax.srt.layers.moe import FusedEPMoE

# Leave headroom for compiler padding/alignment and any unmodeled VMEM usage.
DEFAULT_TPU_VMEM_BUDGET_MB = 60
DEFAULT_TPU_VMEM_BUDGET_BYTES = DEFAULT_TPU_VMEM_BUDGET_MB * 1024 * 1024


def _dtype_packing(dtype: jnp.dtype) -> int:
    """Match get_dtype_packing() in fused_moe kernel (32-bit repack width)."""
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype packing for {dtype=} ({bits=} bits).")
    return 32 // bits


def _estimate_vmem_bytes(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype,
    router_dtype: jnp.dtype,
    cfg: FusedMoEBlockConfig,
    use_shared_expert: bool = False,
    verbose: bool = False,
) -> int:
    """Rough VMEM estimate to avoid compile-time OOM (TPU VMEM is 64MB/core).

    Note: this intentionally overestimates a bit because the fused_moe kernel
    materializes several routing/top-k temporaries (see `get_top_k` in the
    pallas kernel body), which are not part of the explicit scratch buffers.
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
    weight_bytes = token_bytes
    router_bytes = jnp.dtype(router_dtype).itemsize

    t_packing = _dtype_packing(dtype)
    padded_num_experts = ((case.num_experts + 127) // 128) * 128
    padded_top_k = ((case.top_k + 127) // 128) * 128
    # Kernel scratch shapes use bt directly.
    # a2a_max_tokens = align_to(bt * num_devices, bts)
    a2a_max_tokens = ((bt * num_devices + bts - 1) // bts) * bts

    # Output-side staging (no overlap):
    # - a2a_g_acc_vmem: (1, top_k, gcd(bt, 16), t_packing, hidden_per_pack)
    # - b_output_x2_vmem: (2, bt, hidden_size)
    acc_bt = math.gcd(bt, 16)  # Must match fused_moe kernel scratch shape.
    a2a_g_acc = 1 * top_k * acc_bt * hidden * token_bytes
    # b_output_x2_vmem is double-buffered to overlap store(output_hbm) with next bt's compute.
    b_output = 2 * bt * hidden * token_bytes
    # b_gating_x2_vmem is double-buffered for run_bt overlap: (2, bt, padded_num_experts)
    b_gating = 2 * bt * padded_num_experts * router_bytes
    # t2e_routing_smem scratch is placed in SMEM (not VMEM).
    t2e_routing = 0
    # top_k_logits_vmem scratch: (bt, top_k) float32
    top_k_logits = bt * top_k * 4

    # See kernel scratch shapes: b_w1_x2_vmem/b_w3_x2_vmem/b_w2_x2_vmem.
    w1 = 2 * bd1 * bf * weight_bytes
    w3 = 2 * bd1 * bf * weight_bytes
    w2 = 2 * bf * bd2 * weight_bytes

    # b_acc_vmem is F32(2, a2a_max_tokens, 1, bf)
    b_acc = 2 * a2a_max_tokens * bf * 4
    # U32 token staging for FFN1: (2, bts, bd1 // t_packing)
    # Note: Using 4 bytes for packing factor adjustment roughly
    t_stage_b32 = 2 * bts * (bd1 // 2) * 4  # Approximation
    # Kernel uses triple-buffering for a2a_s_acc staging: (3, bts, bd2 // t_packing)
    a2a_s_acc_stage_b32 = 3 * bts * (bd2 // 2) * 4  # Approximation

    # Routing / top-k temporaries in kernel (best-effort conservative estimate):
    # - softmax + get_top_k use float32 work buffers and broadcasted iotas
    # - top_k logits are materialized as `top_k` arrays of shape (bt, padded_top_k)
    # This is separate from `t2e_routing_smem` above.
    routing_work_f32 = bt * padded_num_experts * 4  # softmax result (approx)
    get_top_k_input_f32 = bt * padded_num_experts * 4
    get_top_k_t2e = bt * padded_num_experts * 4
    get_top_k_iota = bt * padded_num_experts * 4
    get_top_k_mask = bt * padded_num_experts * 4
    get_top_k_padded_iota = bt * padded_top_k * 4
    get_top_k_t2e_routing = bt * padded_top_k * 4
    get_top_k_logits_sum = bt * padded_top_k * 4
    get_top_k_logits_lst = top_k * bt * padded_top_k * 4
    routing_temporaries = (
        routing_work_f32
        + get_top_k_input_f32
        + get_top_k_t2e
        + get_top_k_iota
        + get_top_k_mask
        + get_top_k_padded_iota
        + get_top_k_t2e_routing
        + get_top_k_logits_sum
        + get_top_k_logits_lst
    )

    total_bytes = (
        a2a_g_acc
        + b_output
        + b_gating
        + t2e_routing
        + top_k_logits
        + w1
        + w3
        + w2
        + b_acc
        + t_stage_b32
        + a2a_s_acc_stage_b32
        + routing_temporaries
    )

    if use_shared_expert:
        se_w1 = 2 * bd1 * cfg.bse * weight_bytes  # (2, t_packing, bd/pack, bse)
        se_w3 = 2 * bd1 * cfg.bse * weight_bytes
        se_w2 = 2 * cfg.bse * bd2 * weight_bytes
        # Matches fused_moe kernel scratch shape (bt ping-pong x bd1-slice ping-pong):
        # (2, 2, bt, t_packing, bd1_per_pack) => 4 * bt * bd1 elements.
        se_tokens = 4 * bt * bd1 * token_bytes
        total_bytes += se_w1 + se_w3 + se_w2 + se_tokens

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
        print(f"      b_acc_vmem:             {_mb(b_acc)} MB  (2, {a2a_max_tokens}, 1, {bf}) f32")
        print(f"      b_output_x2_vmem:       {_mb(b_output)} MB  (2, {bt}, {hidden})")
        print(
            f"      a2a_g_acc_vmem:         {_mb(a2a_g_acc)} MB  (1, {top_k}, {acc_bt}, {t_packing}, {hidden // t_packing})"
        )
        print(
            f"      b_stage_x2_vmem:        {_mb(t_stage_b32)} MB  (2, {bts}, {t_packing}, {bd1 // t_packing})"
        )
        print(
            f"      a2a_s_acc_stage_x3:     {_mb(a2a_s_acc_stage_b32)} MB  (3, {bts}, {t_packing}, {bd2 // t_packing})"
        )
        print(f"      b_gating_x2_vmem:       {_mb(b_gating)} MB  (2, {bt}, {padded_num_experts})")
        print(f"      routing_temporaries:    {_mb(routing_temporaries)} MB")
        if use_shared_expert:
            print(
                f"      b_se_w1_x2_vmem:        {_mb(se_w1)} MB  (2, {t_packing}, {bd1 // t_packing}, {bf})"
            )
            print(
                f"      b_se_w3_x2_vmem:        {_mb(se_w3)} MB  (2, {t_packing}, {bd1 // t_packing}, {bf})"
            )
            print(
                f"      b_se_w2_x2_vmem:        {_mb(se_w2)} MB  (2, {t_packing}, {bf}, {bd2 // t_packing})"
            )
            print(
                f"      b_se_tokens_vmem:       {_mb(se_tokens)} MB  (2, 2, {bt}, {t_packing}, {bd1 // t_packing})"
            )
        print("      ----------------------------")
        print(f"      Total:                  {_mb(total_bytes)} MB")

    return total_bytes


def select_block_configs(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype,
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

    bt_candidates = _pick_candidates(candidates=bt_candidates, multiple_of=t_packing)
    bts_candidates_i: list[int] | None
    if bts_candidates is None:
        bts_candidates_i = None
    else:
        bts_candidates_i = _pick_candidates(candidates=list(bts_candidates), multiple_of=t_packing)
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
        if bt % router_tile0 != 0:
            return (
                False,
                f"bt({bt}) not aligned to router_tile0({router_tile0}) for router_dtype={jnp.dtype(router_dtype).name}",
            )
        if bts % t_packing != 0:
            return False, f"bts({bts}) % t_packing({t_packing}) != 0"
        if not (0 < bts <= bt):
            return False, f"bts({bts}) not in (0, bt({bt})]"
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
            case, dtype, router_dtype, cfg, use_shared_expert=use_shared_expert
        )
        if est > tpu_vmem_budget_bytes:
            return (
                False,
                f"vmem_est={est / (1024 * 1024):.1f}MB > budget={tpu_vmem_budget_bytes / (1024 * 1024):.1f}MB",
            )
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
        seen.add(key)
        configs.append(effective)

    for bt in bt_candidates:
        if bts_candidates_i is None:
            # Match the historical behavior: when `bts` isn't explicitly tuned, it
            # defaults to `bt` (both in the kernel and in tuned table entries).
            bts_list: list[int] = [bt]
        else:
            bts_list = [v for v in bts_candidates_i if v <= bt]
        for bts in bts_list:
            for bf in bf_candidates:
                for bd in bd_candidates:
                    current_bse_list = bse_candidates_i if use_shared_expert else [bf]

                    for bse in current_bse_list:
                        raw = FusedMoEBlockConfig(
                            bt=bt,
                            bf=bf,
                            bd1=bd,
                            bd2=bd,
                            btc=bt,
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

    # Pareto-optimal filtering: remove configs that are dominated by others.
    # A config is "dominated" if another config is >= in ALL dimensions and > in at least one.
    # Since larger block sizes are generally more efficient, we keep only non-dominated configs.
    def _config_tuple(c: FusedMoEBlockConfig) -> tuple[int, ...]:
        return (c.bt, c.bts or c.bt, c.bf, c.bd1, c.bd2, c.btc, c.bfc, c.bd1c, c.bd2c)

    def _dominates(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
        """Returns True if `a` dominates `b` (a >= b in all dims, a > b in at least one)."""
        all_geq = all(x >= y for x, y in zip(a, b))
        any_gt = any(x > y for x, y in zip(a, b))
        return all_geq and any_gt

    config_tuples = [_config_tuple(c) for c in configs]
    non_dominated_indices: list[int] = []
    for i, t_i in enumerate(config_tuples):
        is_dominated = False
        for j, t_j in enumerate(config_tuples):
            if i != j and _dominates(t_j, t_i):
                is_dominated = True
                break
        if not is_dominated:
            non_dominated_indices.append(i)

    pareto_configs = [configs[i] for i in non_dominated_indices]
    print(f"  pareto filter: {len(configs)} valid configs -> {len(pareto_configs)} non-dominated")
    configs = pareto_configs

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

    selected: list[FusedMoEBlockConfig] = []
    selected_keys: set[tuple[int, ...]] = set()

    def _add(cfg: FusedMoEBlockConfig) -> None:
        key = score(cfg)
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(cfg)

    ranked = sorted(configs, key=score, reverse=True)
    for cfg in ranked:
        _add(cfg)
        if len(selected) >= max_configs:
            break

    print(f"  limit: {len(configs)} valid configs -> {len(selected)} (max={max_configs})")
    return selected


def run_all(
    iters: int,
    dtype: jnp.dtype = jnp.bfloat16,
    *,
    warmup_iters: int = 1,
    tune_block_config: bool = False,
    bt_candidates: list[int] | None = None,
    bts_candidates: list[int] | None = None,
    bf_candidates: list[int] | None = None,
    bd_candidates: list[int] | None = None,
    bse_candidates: list[int] | None = None,
    num_tokens: list[int] | None = None,
    tpu_vmem_budget_bytes: int = DEFAULT_TPU_VMEM_BUDGET_BYTES,
    max_configs: int = 9,
    config_mode: str = "base",
    use_shared_expert: bool = False,
    use_grouped_topk: bool = False,
    imbalance_mode: str = None,
    alpha: float = None,
    zipf_s: float = None,
    hotspot_ratio: float = None,
    hotspot_count: int = None,
    zero_expert_count: int = None,
    non_hotspot_alpha: float = None,
) -> None:
    raw_cases: list[MoEBenchmarkCase] | None = None
    if num_tokens is not None:
        # If num_tokens is explicit, use base config as template
        raw_cases = [
            MoEBenchmarkCase(
                name=(
                    f"custom_nt{n}_ne{BAILING_BASE['num_experts']}_tk{BAILING_BASE['top_k']}"
                    f"_h{BAILING_BASE['hidden_size']}_i{BAILING_BASE['intermediate_size']}"
                ),
                num_tokens=n,
                **BAILING_BASE,
            )
            for n in num_tokens
        ]
    else:
        if config_mode == "base":
            raw_cases = BASE_CASES
        elif config_mode == "mini":
            raw_cases = MINI_CASES
        elif config_mode == "all":
            raw_cases = GROUP_GEMM_CASES
        else:
            raise ValueError(f"Unknown config_mode: {config_mode}")

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
    if tune_block_config:
        from sgl_jax.srt.utils.jax_utils import get_device_name

    print(f"Running fused_moe benchmarks with dtype={dtype}")
    print(f"  config_mode: {config_mode}")
    print(f"  features: shared_expert={use_shared_expert}, grouped_topk={use_grouped_topk}")

    for case in cases:
        t_packing = _dtype_packing(dtype)
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
        data = prepare_fused_moe_inputs(
            case,
            dtype=dtype,
            mesh=mesh,
            include_weights=False,
            include_shared_expert=use_shared_expert,
        )

        print(f"{imbalance_mode=}")
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
        block_cfgs: list[FusedMoEBlockConfig | None]
        if tune_block_config:
            block_cfgs = select_block_configs(
                case,
                dtype,
                router_dtype=data["router_logits"].dtype,
                bt_candidates=bt_candidates or [2, 4, 8, 16, 32, 64, 128, 256, 512],
                bts_candidates=bts_candidates,
                bf_candidates=bf_candidates or [128, 256, 512, 1024, 2048],
                bd_candidates=bd_candidates or [256, 512, 1024, 2048, 4096, 8192],
                bse_candidates=bse_candidates,
                tpu_vmem_budget_bytes=tpu_vmem_budget_bytes,
                max_configs=max_configs,
                use_shared_expert=use_shared_expert,
            )
        else:
            block_cfgs = [None]

        with jax.set_mesh(mesh):
            fused_layer = FusedEPMoE(
                hidden_size=case.hidden_size,
                num_experts=case.num_experts,
                num_experts_per_tok=case.top_k,
                ep_size=case.ep_size,
                mesh=mesh,
                intermediate_dim=case.intermediate_size,
                weight_dtype=dtype,
                dtype=dtype,
                activation=case.activation,
                layer_id=0,
                renormalize_topk_logits=case.renormalize_topk_logits,
                num_groups=case.num_expert_group if use_grouped_topk else 1,
                top_k_groups=case.topk_group if use_grouped_topk else 1,
                num_shared_experts=1 if use_shared_expert else 0,
                moe_shared_expert_intermediate_size=(
                    case.intermediate_size if use_shared_expert else None
                ),
            )

            moe_def, moe_state = nnx.split(fused_layer)
            moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)

            @jax.jit(static_argnames=("moe_state_def", "block_config"))
            def run(tokens, router_logits, *, moe_state_def, moe_state_leaves, block_config):
                moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
                moe = nnx.merge(moe_def, moe_state)
                return moe(tokens, router_logits, block_config=block_config)

            best: tuple[float, FusedMoEBlockConfig | None] | None = None
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
                        dtype,
                        data["router_logits"].dtype,
                        block_cfg,
                        use_shared_expert=use_shared_expert,
                        verbose=True,
                    )
                    vmem_mb = vmem_bytes / (1024 * 1024)
                    vmem_remaining_mb = 64.0 - vmem_mb
                    print(
                        f"    => VMEM: {vmem_mb:.2f} MB / 64 MB (remaining: {vmem_remaining_mb:.2f} MB)"
                    )

                task = "fused-moe-k_.*"

                def _compute(block_cfg=block_cfg):
                    return run(
                        data["tokens"],
                        data["router_logits"],
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
        help="Override benchmark cases with custom num_tokens list (e.g. --num-tokens 8 16 256 4096).",
    )
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
    # [New] Feature flags
    parser.add_argument(
        "--config-mode",
        type=str,
        default="base",
        choices=["base", "mini", "all"],
        help="Select which set of benchmark cases to run (default: base).",
    )
    parser.add_argument(
        "--use-grouped-topk",
        action="store_true",
        help="Enable grouped top-k routing logic.",
    )
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
        help="All-to-All 不均衡负载模式",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Dirichlet 分布因子 (越小越不均衡，如 0.1)"
    )
    parser.add_argument("--zipf-s", type=float, default=1.1, help="Zipf 分布因子 (越大越不均衡)")
    parser.add_argument(
        "--hotspot-ratio", type=float, default=0.5, help="热点专家占据的 Token 总量比例 (0.0-1.0)"
    )
    parser.add_argument("--hotspot-count", type=int, default=1, help="热点专家的数量")
    parser.add_argument("--zero-expert-count", type=int, default=0)
    parser.add_argument("--non-hotspot-alpha", type=float, default=100.0)
    return parser.parse_args()


if __name__ == "__main__":
    # In some TPU environments only stdout is captured/aggregated. Route fatal Python
    # traces (e.g., segfaults, timeouts) to stdout to avoid "silent" exits.
    try:
        faulthandler.enable(file=sys.stdout, all_threads=True)
    except Exception:
        pass
    args = parse_args()
    if args.compilation_cache_dir:
        _compilation_cache.set_cache_dir(args.compilation_cache_dir)
    tpu_vmem_budget_bytes = int(args.tpu_vmem_budget_mb) * 1024 * 1024
    full_args_dict = vars(args)
    try:
        run_all(
            args.iters,
            warmup_iters=args.warmup_iters,
            tune_block_config=args.tune_block_config,
            bt_candidates=args.bt_candidates,
            bts_candidates=args.bts_candidates,
            bf_candidates=args.bf_candidates,
            bd_candidates=args.bd_candidates,
            bse_candidates=args.bse_candidates,
            num_tokens=args.num_tokens,
            tpu_vmem_budget_bytes=tpu_vmem_budget_bytes,
            max_configs=args.max_configs,
            config_mode=args.config_mode,
            use_shared_expert=args.use_shared_expert,
            use_grouped_topk=args.use_grouped_topk,
            imbalance_mode=args.imbalance_mode,
            alpha=args.alpha,
            zipf_s=args.zipf_s,
            hotspot_ratio=args.hotspot_ratio,
            hotspot_count=args.hotspot_count,
            zero_expert_count=args.zero_expert_count,
            non_hotspot_alpha=args.non_hotspot_alpha,
        )
    except BaseException as e:
        print(f"FATAL: {type(e).__name__}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise
