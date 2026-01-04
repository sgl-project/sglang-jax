"""
Benchmark fused_moe kernel with grouped-GEMM-like MoE shapes.

Usage:
    python -m benchmark.moe.bench_fused_moe [--scenario random|balanced|imbalanced] [--tune-block-config]
"""

from __future__ import annotations

import argparse
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from benchmark.moe.utils import (
    BAILING_BASE,
    MoEBenchmarkCase,
    build_mesh,
    prepare_fused_moe_inputs,
    select_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig
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


def _estimate_vmem_bytes(case: MoEBenchmarkCase, dtype: jnp.dtype, cfg: FusedMoEBlockConfig) -> int:
    """Rough VMEM estimate to avoid compile-time OOM (TPU VMEM is 64MB/core).

    Note: this intentionally overestimates a bit because the fused_moe kernel
    materializes several routing/top-k temporaries (see `get_top_k` in the
    pallas kernel body), which are not part of the explicit scratch buffers.
    """
    bt = cfg.bt
    bf = cfg.bf
    bd1 = cfg.bd1
    bd2 = cfg.bd2
    top_k = case.top_k
    num_devices = case.ep_size
    hidden = case.hidden_size

    token_bytes = jnp.dtype(dtype).itemsize
    weight_bytes = token_bytes

    t_packing = _dtype_packing(dtype)
    local_num_tokens = case.num_tokens // case.ep_size
    padded_num_experts = ((case.num_experts + 127) // 128) * 128
    padded_top_k = ((case.top_k + 127) // 128) * 128
    # Max tokens per active expert buffer is `local_num_tokens * num_devices`, padded up
    # to a multiple of `bt` for safe (bt-sized) token tiling in the kernel.
    a2a_max_tokens = ((local_num_tokens * num_devices + bt - 1) // bt) * bt

    # a2a_g_acc_vmem is token-level double-buffered: (2, top_k, 1, ..., hidden)
    a2a_g_acc = 2 * top_k * hidden * token_bytes
    # b_output is tile-level double-buffered: (2, output_bt, hidden_size)
    output_bt = math.gcd(bt, local_num_tokens)
    b_output = 2 * output_bt * hidden * token_bytes
    b_gating = local_num_tokens * padded_num_experts * token_bytes
    # t2e_routing_smem scratch: (local_num_tokens, padded_top_k) int32
    t2e_routing = local_num_tokens * padded_top_k * 4

    # See kernel scratch shapes: b_w1_x2_vmem/b_w3_x2_vmem/b_w2_x2_vmem.
    w1 = 2 * bd1 * bf * weight_bytes
    w3 = 2 * bd1 * bf * weight_bytes
    w2 = 2 * bf * bd2 * weight_bytes

    # b_acc_vmem is F32(2, a2a_max_tokens, 1, bf)
    b_acc = 2 * a2a_max_tokens * bf * 4
    # U32 token staging for FFN1: (2, bt, bd1 // t_packing)
    t_stage_b32 = 2 * bt * (bd1 // t_packing) * 4
    # U32 staging for FFN2 output slice: (2, bt, bd2 // t_packing)
    # Kernel uses triple-buffering for a2a_s_acc staging: (3, bt, bd2 // t_packing)
    a2a_s_acc_stage_b32 = 3 * bt * (bd2 // t_packing) * 4

    # Routing / top-k temporaries in kernel (best-effort conservative estimate):
    # - softmax + get_top_k use float32 work buffers and broadcasted iotas
    # - top_k_logits/indices are stored as lists of `padded_top_k` columns
    # This is separate from `t2e_routing_smem` above.
    routing_work_f32 = local_num_tokens * padded_num_experts * 4  # softmax result (approx)
    get_top_k_input_f32 = local_num_tokens * padded_num_experts * 4
    get_top_k_t2e = local_num_tokens * padded_num_experts * 4
    get_top_k_iota = local_num_tokens * padded_num_experts * 4
    get_top_k_mask = local_num_tokens * padded_num_experts * 4
    get_top_k_padded_iota = local_num_tokens * padded_top_k * 4
    get_top_k_t2e_routing = local_num_tokens * padded_top_k * 4
    get_top_k_logits_sum = local_num_tokens * padded_top_k * 4
    get_top_k_logits_lst = top_k * local_num_tokens * padded_top_k * 4
    get_top_k_indices_lst = top_k * local_num_tokens * padded_top_k * 4
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
        + get_top_k_indices_lst
    )

    # Skip optional scale/bias buffers (unused in this benchmark).
    return (
        a2a_g_acc
        + b_output
        + b_gating
        + t2e_routing
        + w1
        + w3
        + w2
        + b_acc
        + t_stage_b32
        + a2a_s_acc_stage_b32
        + routing_temporaries
    )


def select_block_configs(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype,
    *,
    bt_candidates: list[int],
    bf_candidates: list[int],
    bd_candidates: list[int],
    tpu_vmem_budget_bytes: int,
    max_configs: int,
) -> list[FusedMoEBlockConfig]:
    """Enumerate block configs from the explicit candidate lists.

    This function is intentionally "what you see is what you get": it filters
    only by basic divisibility/alignment constraints and VMEM budget, without
    applying additional heuristics (e.g. "keep only the largest two").
    """
    t_packing = _dtype_packing(dtype)
    tile_align = t_packing * 128
    local_num_tokens = case.num_tokens // case.ep_size

    def _pick_candidates(
        *,
        candidates: list[int],
        multiple_of: int,
    ) -> list[int]:
        # Keep this strictly "what you see is what you get":
        # - Do not filter by the current case shape here.
        # - Let `FusedMoEBlockConfig.effective_for(...)` apply kernel overrides
        #   (notably clamping `bt` for small local token counts), then validate.
        out: list[int] = []
        for v in candidates:
            if v <= 0:
                continue
            if v % multiple_of != 0:
                continue
            out.append(v)
        return sorted(set(out))

    bt_candidates = _pick_candidates(candidates=bt_candidates, multiple_of=t_packing)
    bf_candidates = _pick_candidates(candidates=bf_candidates, multiple_of=128)
    bd_candidates = _pick_candidates(candidates=bd_candidates, multiple_of=tile_align)

    def validate(cfg: FusedMoEBlockConfig) -> tuple[bool, str]:
        bt = cfg.bt
        bf = cfg.bf
        bd1 = cfg.bd1
        bd2 = cfg.bd2
        btc = cfg.btc
        bfc = cfg.bfc
        bd1c = cfg.bd1c
        bd2c = cfg.bd2c

        if bt <= 0 or bf <= 0 or bd1 <= 0 or bd2 <= 0:
            return False, "non-positive tile size"
        if bt > local_num_tokens * case.ep_size:
            return False, f"bt({bt}) > max_expert_tokens({local_num_tokens * case.ep_size})"
        if bt % t_packing != 0:
            return False, f"bt({bt}) % t_packing({t_packing}) != 0"
        if not (0 < btc <= bt):
            return False, f"btc({btc}) not in (0, bt({bt})]"
        if btc % t_packing != 0:
            return False, f"btc({btc}) % t_packing({t_packing}) != 0"

        if case.intermediate_size % bf != 0:
            return False, f"intermediate_size({case.intermediate_size}) % bf({bf}) != 0"
        if bf % 128 != 0:
            return False, f"bf({bf}) % 128 != 0"
        if bfc % 128 != 0:
            return False, f"bfc({bfc}) % 128 != 0"
        if bf % bfc != 0:
            return False, f"bf({bf}) % bfc({bfc}) != 0"

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

        # TPU VMEM is 64MB per core; exceeding it is a compile-time OOM.
        est = _estimate_vmem_bytes(case, dtype, cfg)
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
            effective.btc,
            effective.bfc,
            effective.bd1c,
            effective.bd2c,
        )
        if key in seen:
            return
        seen.add(key)
        configs.append(effective)

    for bt in bt_candidates:
        for bf in bf_candidates:
            for bd in bd_candidates:
                raw = FusedMoEBlockConfig(
                    bt=bt,
                    bf=bf,
                    bd1=bd,
                    bd2=bd,
                    btc=bt,
                    bfc=bf,
                    bd1c=bd,
                    bd2c=bd,
                )
                effective = raw.effective_for(
                    num_tokens=case.num_tokens, ep_size=case.ep_size, dtype=dtype
                )
                add(raw=raw, effective=effective)

    if max_configs <= 0:
        raise ValueError(f"Expected {max_configs=} to be > 0.")
    if len(configs) <= max_configs:
        return configs

    # Keep benchmark runtime bounded while retaining coverage across (btc, bf, bd).
    def score(c: FusedMoEBlockConfig) -> tuple[int, int, int, int, int, int, int, int]:
        # Lexicographic "weighted" ranking; larger tiles tend to run faster.
        return (c.bt, c.bf, c.bd1, c.bd2, c.btc, c.bfc, c.bd1c, c.bd2c)

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
    scenario: str,
    iters: int,
    dtype: jnp.dtype = jnp.bfloat16,
    *,
    warmup_iters: int = 1,
    a2a_only: bool = False,
    tune_block_config: bool = False,
    bt_candidates: list[int] | None = None,
    bf_candidates: list[int] | None = None,
    bd_candidates: list[int] | None = None,
    num_tokens: list[int] | None = None,
    tpu_vmem_budget_bytes: int = DEFAULT_TPU_VMEM_BUDGET_BYTES,
    max_configs: int = 9,
) -> None:
    raw_cases: list[MoEBenchmarkCase] | None = None
    if num_tokens is not None:
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
    cases_all = list(select_cases(raw_cases))
    cases: list[MoEBenchmarkCase] = []
    for c in cases_all:
        # Fused MoE benchmark currently assumes TP is disabled (tp_size=1).
        # For small num_tokens (e.g. decode shapes), select_cases may pick a smaller
        # ep_size so that tokens can be evenly sharded; that implies tp_size>1 when
        # device_count is larger. Skip such cases to keep results comparable.
        if c.tp_size != 1:
            print(
                f"skip [case={c.name}] because tp_size={c.tp_size} (require tp_size=1 for fused_moe)"
            )
            continue
        cases.append(c)
    if not cases:
        print("No runnable fused_moe cases after filtering tp_size!=1.")
        return

    tuned_results: dict[str, dict[tuple, tuple[int, int, int, int, int, int, int, int]]] = {}
    if tune_block_config:
        from sgl_jax.srt.utils.jax_utils import get_device_name

    print(f"Running fused_moe benchmarks with scenario='{scenario}', dtype={dtype}")
    if a2a_only:
        print("  mode: a2a_only=True")
    for case in cases:
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}, ep_size={case.ep_size}"
        )
        print(
            f"  mesh: ep_size={case.ep_size}, tp_size={case.tp_size}, "
            f"devices_used={case.ep_size * case.tp_size}/{len(jax.devices())}"
        )

        mesh = build_mesh(ep_size=case.ep_size, tp_size=case.tp_size)
        data = prepare_fused_moe_inputs(
            case, scenario, dtype=dtype, mesh=mesh, include_weights=False
        )
        block_cfgs: list[FusedMoEBlockConfig | None]
        if tune_block_config:
            block_cfgs = select_block_configs(
                case,
                dtype,
                bt_candidates=bt_candidates or [2, 4, 8, 16, 32, 64, 128, 256, 512],
                bf_candidates=bf_candidates or [128, 256, 512, 1024, 2048],
                bd_candidates=bd_candidates or [256, 512, 1024, 2048, 4096, 8192],
                tpu_vmem_budget_bytes=tpu_vmem_budget_bytes,
                max_configs=max_configs,
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
                a2a_only=a2a_only,
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

                task = "fused-moe-k_.*"

                def _compute(block_cfg=block_cfg):
                    return run(
                        data["tokens"],
                        data["router_logits"],
                        moe_state_def=moe_state_def,
                        moe_state_leaves=moe_state_leaves,
                        block_config=block_cfg,
                    )

                times = multiple_iteration_timeit_from_trace(
                    compute_func=_compute,
                    data_generator=lambda: (),
                    task=task,
                    tries=iters,
                    warmup=warmup_iters,
                )
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
                    )
                    cfg_tuple = (
                        best_cfg.bt,
                        best_cfg.bf,
                        best_cfg.bd1,
                        best_cfg.bd2,
                        best_cfg.btc,
                        best_cfg.bfc,
                        best_cfg.bd1c,
                        best_cfg.bd2c,
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
                entries.keys(), key=lambda t: (t[1], t[2], t[3], t[4], t[5], t[6], t[0])
            ):
                print(f"    {k}: {entries[k]},")
            print("})\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fused_moe.")
    parser.add_argument(
        "--scenario",
        choices=["random", "balanced", "imbalanced"],
        default="random",
        help="Router logits distribution pattern.",
    )
    parser.add_argument("--iters", type=int, default=3, help="Number of benchmark iterations.")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Number of warmup iterations before profiling (per case / block_config).",
    )
    parser.add_argument(
        "--a2a-only",
        action="store_true",
        help="Skip expert FFN compute (measure mostly routing + A2A + accumulation).",
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
            "Candidate list for btc (inner token tile). "
            "Note: outer bt is fixed to local_num_tokens in this kernel variant."
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
        "--max-configs",
        type=int,
        default=9,
        help="Maximum number of block configs to benchmark per case when --tune-block-config is set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tpu_vmem_budget_bytes = int(args.tpu_vmem_budget_mb) * 1024 * 1024
    run_all(
        args.scenario,
        args.iters,
        warmup_iters=args.warmup_iters,
        a2a_only=args.a2a_only,
        tune_block_config=args.tune_block_config,
        bt_candidates=args.bt_candidates,
        bf_candidates=args.bf_candidates,
        bd_candidates=args.bd_candidates,
        num_tokens=args.num_tokens,
        tpu_vmem_budget_bytes=tpu_vmem_budget_bytes,
        max_configs=args.max_configs,
    )
