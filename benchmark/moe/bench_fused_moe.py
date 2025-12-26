"""
Benchmark fused_moe kernel with grouped-GEMM-like MoE shapes.

Usage:
    python -m benchmark.moe.bench_fused_moe [--scenario random|balanced|imbalanced] [--tune-block-config]
"""

from __future__ import annotations

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.moe.utils import MoEBenchmarkCase, prepare_fused_moe_inputs, select_cases
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig, fused_ep_moe


def build_mesh(ep_size: int = 1):
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    devices = jax.devices()[:ep_size]
    return create_device_mesh(
        ici_parallelism=[ep_size, 1],
        dcn_parallelism=[1, 1],
        devices=devices,
        mesh_axes=("tensor", "data"),
    )


TPU_VMEM_CAP_BYTES = 64 * 1024 * 1024
# Leave headroom for compiler padding/alignment and any unmodeled VMEM usage.
TPU_VMEM_BUDGET_BYTES = 60 * 1024 * 1024


def _dtype_packing(dtype: jnp.dtype) -> int:
    """Match get_dtype_packing() in fused_moe kernel (32-bit repack width)."""
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype packing for {dtype=} ({bits=} bits).")
    return 32 // bits


def _estimate_vmem_bytes(case: MoEBenchmarkCase, dtype: jnp.dtype, cfg: FusedMoEBlockConfig) -> int:
    """Rough VMEM estimate to avoid compile-time OOM (TPU VMEM is 64MB/core)."""
    bt = cfg.bt
    bf = cfg.bf
    bd1 = cfg.bd1
    bd2 = cfg.bd2
    top_k = case.top_k
    num_devices = case.ep_size
    hidden = case.hidden_size

    token_bytes = jnp.dtype(dtype).itemsize
    weight_bytes = token_bytes

    a2a_s = 2 * bt * num_devices * hidden * token_bytes
    a2a_s_acc = 2 * bt * num_devices * hidden * token_bytes
    a2a_g_acc = top_k * bt * hidden * token_bytes
    b_output = 2 * bt * hidden * token_bytes
    b_gating = 2 * bt * ((case.num_experts + 127) // 128 * 128) * token_bytes

    # See kernel scratch shapes: b_w1_x2_vmem/b_w3_x2_vmem/b_w2_x2_vmem.
    w1 = 2 * bd1 * bf * weight_bytes
    w3 = 2 * bd1 * bf * weight_bytes
    w2 = 2 * bf * bd2 * weight_bytes

    # b_acc_vmem is F32(2, bt * num_devices, 1, bf)
    b_acc = bt * num_devices * (bf * 2) * 4

    # Skip optional scale/bias buffers (unused in this benchmark).
    return a2a_s + a2a_s_acc + a2a_g_acc + b_output + b_gating + w1 + w3 + w2 + b_acc


def select_block_configs(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype,
) -> list[FusedMoEBlockConfig]:
    """Enumerate a small set of block configs (favoring larger blocks).

    Empirically, larger block sizes tend to be faster for this kernel; to keep
    tuning time manageable, we only consider the largest 2 candidates per block
    dimension (bt/bf/bd), plus the default block size for that dimension when
    it is valid for the case (important when very large blocks would exceed
    TPU VMEM and get filtered out).
    """
    t_packing = _dtype_packing(dtype)
    tile_align = t_packing * 128
    local_num_tokens = case.num_tokens // case.ep_size

    def _pick_candidates(
        *,
        size: int,
        candidates: list[int],
        multiple_of: int,
    ) -> list[int]:
        out: list[int] = []
        for v in candidates:
            if v <= 0 or v > size:
                continue
            if size % v != 0:
                continue
            if v % multiple_of != 0:
                continue
            out.append(v)
        return sorted(set(out))

    bt_candidates = _pick_candidates(
        size=local_num_tokens,
        candidates=[t_packing * 8, 16, 32, 64, 128, 256, local_num_tokens],
        multiple_of=t_packing,
    )
    bf_candidates = _pick_candidates(
        size=case.intermediate_size,
        candidates=[128, 256, 512, 1024, 2048, 4096, case.intermediate_size],
        multiple_of=128,
    )
    bd_candidates = _pick_candidates(
        size=case.hidden_size,
        candidates=[tile_align, 512, 1024, 2048, 4096, 8192, case.hidden_size],
        multiple_of=tile_align,
    )

    def _keep_largest_two_plus_default(
        candidates: list[int],
        *,
        default_value: int,
        size: int,
        multiple_of: int,
    ) -> list[int]:
        kept = candidates if len(candidates) > 2 else list(candidates)
        if (
            default_value <= size
            and size % default_value == 0
            and default_value % multiple_of == 0
            and default_value not in kept
        ):
            kept.append(default_value)
        return sorted(set(kept))

    bt_candidates = _keep_largest_two_plus_default(
        bt_candidates,
        default_value=32,
        size=local_num_tokens,
        multiple_of=t_packing,
    )
    bf_candidates = _keep_largest_two_plus_default(
        bf_candidates,
        default_value=1024,
        size=case.intermediate_size,
        multiple_of=128,
    )
    bd_candidates = _keep_largest_two_plus_default(
        bd_candidates,
        default_value=1024,
        size=case.hidden_size,
        multiple_of=tile_align,
    )

    def is_valid(cfg: FusedMoEBlockConfig) -> bool:
        bt = cfg.bt
        bf = cfg.bf
        bd1 = cfg.bd1
        bd2 = cfg.bd2
        btc = cfg.btc
        bfc = cfg.bfc
        bd1c = cfg.bd1c
        bd2c = cfg.bd2c

        if bt <= 0 or bf <= 0 or bd1 <= 0 or bd2 <= 0:
            return False
        if local_num_tokens % bt != 0:
            return False
        if bt % t_packing != 0:
            return False
        if not (0 < btc <= bt):
            return False
        if btc % t_packing != 0:
            return False

        if case.intermediate_size % bf != 0:
            return False
        if bf % 128 != 0:
            return False
        if bfc % 128 != 0:
            return False
        if bf % bfc != 0:
            return False

        if case.hidden_size % bd1 != 0 or case.hidden_size % bd2 != 0:
            return False
        if bd1 % tile_align != 0 or bd2 % tile_align != 0:
            return False
        if bd1c % tile_align != 0 or bd2c % tile_align != 0:
            return False
        if bd1 % bd1c != 0 or bd2 % bd2c != 0:
            return False

        # TPU VMEM is 64MB per core; exceeding it is a compile-time OOM.
        return _estimate_vmem_bytes(case, dtype, cfg) <= TPU_VMEM_BUDGET_BYTES

    configs: list[FusedMoEBlockConfig] = []
    seen: set[tuple[int, ...]] = set()

    def add(cfg: FusedMoEBlockConfig) -> None:
        if not is_valid(cfg):
            return
        key = (
            cfg.bt,
            cfg.bf,
            cfg.bd1,
            cfg.bd2,
            cfg.btc,
            cfg.bfc,
            cfg.bd1c,
            cfg.bd2c,
        )
        if key in seen:
            return
        seen.add(key)
        configs.append(cfg)

    for bt in bt_candidates:
        for bf in bf_candidates:
            for bd in bd_candidates:
                cfg = FusedMoEBlockConfig(
                    bt=bt,
                    bf=bf,
                    bd1=bd,
                    bd2=bd,
                    btc=bt,
                    bfc=bf,
                    bd1c=bd,
                    bd2c=bd,
                ).effective_for(num_tokens=case.num_tokens, ep_size=case.ep_size, dtype=dtype)
                add(cfg)

    # Always include the default config as a reference point (if valid).
    add(
        FusedMoEBlockConfig(
            bt=32,
            bf=1024,
            bd1=1024,
            bd2=1024,
            btc=32,
            bfc=1024,
            bd1c=1024,
            bd2c=1024,
        ).effective_for(num_tokens=case.num_tokens, ep_size=case.ep_size, dtype=dtype)
    )

    return configs


def run_all(
    scenario: str,
    iters: int,
    dtype: jnp.dtype = jnp.bfloat16,
    *,
    tune_block_config: bool = False,
) -> None:
    cases = list(select_cases())

    print(f"Running fused_moe benchmarks with scenario='{scenario}', dtype={dtype}")
    for case in cases:
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}, ep_size={case.ep_size}"
        )

        mesh = build_mesh(ep_size=case.ep_size)
        data = prepare_fused_moe_inputs(case, scenario, dtype=dtype, mesh=mesh)
        block_cfgs: list[FusedMoEBlockConfig | None]
        if tune_block_config:
            block_cfgs = select_block_configs(case, dtype)
        else:
            block_cfgs = [None]

        best: tuple[float, FusedMoEBlockConfig | None] | None = None
        for i, block_cfg in enumerate(block_cfgs):
            tag = "default" if block_cfg is None else str(i)
            if block_cfg is None:
                print("  fused_moe blocks] -> (block_config=None)")
            else:
                print(f"  fused_moe blocks [{i+1}/{len(block_cfgs)}] -> {block_cfg.as_kwargs()}")
            fused = functools.partial(
                fused_ep_moe,
                mesh=mesh,
                top_k=case.top_k,
                renormalize_topk_logits=case.renormalize_topk_logits,
                act_fn=case.activation,
                block_config=block_cfg,
                ep_axis_name="tensor",
            )

            @jax.jit
            def run(tokens, w1, w2, w3, router_logits):
                return fused(tokens=tokens, w1=w1, w2=w2, w3=w3, gating_output=router_logits)

            jax.block_until_ready(
                run(data["tokens"], data["w1"], data["w2"], data["w3"], data["router_logits"])
            )
            times = multiple_iteration_timeit_from_trace(
                compute_func=lambda: run(
                    data["tokens"], data["w1"], data["w2"], data["w3"], data["router_logits"]
                ),
                data_generator=lambda: (),
                task=f"fused_moe_{case.name}_{tag}",
                tries=iters,
            )
            mean_ms = float(np.mean(times)) if times else float("nan")
            print(f"     fused_moe[{tag}]: {mean_ms:.3f} ms (trace) | samples={times}")
            if tune_block_config and np.isfinite(mean_ms):
                if best is None or mean_ms < best[0]:
                    best = (mean_ms, block_cfg)

        if tune_block_config and best is not None:
            from sgl_jax.srt.utils.jax_utils import get_device_name

            best_ms, best_cfg = best
            if best_cfg is None:
                print(f"  best: default ({best_ms:.3f} ms)")
            else:
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
                print(f"  tuned_table[{get_device_name()}][{table_key}] = {cfg_tuple}")


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
        "--tune-block-config",
        action="store_true",
        help="Benchmark multiple block_config variants and print the best tuned table entry.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(
        args.scenario,
        args.iters,
        tune_block_config=args.tune_block_config,
    )
