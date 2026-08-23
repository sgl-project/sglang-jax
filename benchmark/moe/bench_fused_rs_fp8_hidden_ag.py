"""Strict A/B for BF16 versus per-rank per-tensor FP8 Hidden AllGather."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax

from benchmark.moe.bench_fused_rs_moe import (
    GLM52_HIDDEN_SIZE,
    GLM52_TOP_K,
    _build_mesh,
    _comparison_metrics,
    _invalid_padding_max_abs,
    _make_inputs,
    _make_padded_inputs,
    _measure,
    _measure_rs_breakdown,
    _routing_stats,
    _rs_runner,
)
from benchmark.moe.fused_rs_fp8_hidden_ag_contract import (
    DEFAULT_REL_L2_THRESHOLD,
    evaluate_fp8_hidden_ag_contract,
)
from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    set_fused_rs_block_sizes_override,
    set_fused_rs_routing_table_impl,
)

PRODUCTION_RS_CONFIG = (256, 6144, 1024, 2048, 1024, 2, 2)
VARIANTS = (
    ("bf16-hidden-ag", False),
    ("fp8-per-tensor-hidden-ag", True),
)


def _median(samples):
    return statistics.median(samples) if samples else None


def main() -> None:
    jax.distributed.initialize()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--routing-seed", type=int, default=42)
    parser.add_argument("--padding-active-tokens-per-device", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--trace-root", default="/tmp/sglang_jax_fused_rs_fp8_hidden_ag")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument(
        "--correctness-rel-l2-threshold",
        type=float,
        default=DEFAULT_REL_L2_THRESHOLD,
    )
    args = parser.parse_args()

    visible_devices = len(jax.devices())
    supported_contracts = {
        (32, 65536, 32),
        (8, 32768, 16),
    }
    if (args.ep_size, args.tokens, visible_devices) not in supported_contracts:
        raise ValueError(
            "FP8 Hidden AG A/B requires EP32/64K on 32 devices or EP8/32K "
            f"on 16 devices; got EP{args.ep_size}/{args.tokens}/{visible_devices}"
        )
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("", encoding="utf-8")

    def emit(row: dict) -> None:
        encoded = json.dumps(row, sort_keys=True)
        print(encoded, flush=True)
        if args.jsonl is not None:
            with args.jsonl.open("a", encoding="utf-8") as output_file:
                output_file.write(encoded + "\n")

    mesh = _build_mesh(args.ep_size)
    baseline_out = None
    baseline_padded_out = None
    with jax.set_mesh(mesh):
        inputs = _make_inputs(
            mesh,
            args.tokens,
            args.ep_size,
            routing_seed=args.routing_seed,
            layer_scope=False,
            input_profile="expert_distinct",
        )
        padded_inputs, valid_mask = _make_padded_inputs(
            inputs,
            num_tokens=args.tokens,
            ep_size=args.ep_size,
            active_per_device=args.padding_active_tokens_per_device,
        )
        routing_stats = _routing_stats(inputs[8])

        for variant, fp8_hidden_all_gather in VARIANTS:
            set_fused_rs_block_sizes_override(PRODUCTION_RS_CONFIG)
            set_fused_rs_routing_table_impl("pallas")
            jax.clear_caches()
            run = _rs_runner(
                mesh,
                layer_scope=False,
                fp8_hidden_all_gather=fp8_hidden_all_gather,
            )

            compile_start = time.perf_counter()
            output = run(inputs)
            padded_output = run(padded_inputs)
            jax.block_until_ready((output, padded_output))
            compile_time_s = time.perf_counter() - compile_start

            if baseline_out is None:
                baseline_out = output
                baseline_padded_out = padded_output
            full_metrics = _comparison_metrics(baseline_out, output)
            padded_metrics = _comparison_metrics(
                baseline_padded_out,
                padded_output,
                valid_mask=valid_mask,
            )
            padding_invariance = _comparison_metrics(
                output,
                padded_output,
                valid_mask=valid_mask,
            )
            invalid_padding_max_abs = _invalid_padding_max_abs(
                padded_output,
                valid_mask,
            )
            contract = evaluate_fp8_hidden_ag_contract(
                full_all_finite=full_metrics["all_finite"],
                full_rel_l2=full_metrics["rel_l2"],
                padded_all_finite=padded_metrics["all_finite"],
                padded_rel_l2=padded_metrics["rel_l2"],
                padding_invariance_rel_l2=padding_invariance["rel_l2"],
                invalid_padding_max_abs=invalid_padding_max_abs,
                rel_l2_threshold=args.correctness_rel_l2_threshold,
            )

            standard_samples = _measure(
                run,
                inputs,
                task=r"gmm_v2_fused_rs.*",
                warmup=args.warmup,
                iters=args.iters,
                trace_root=str(Path(args.trace_root) / variant / "standard"),
            )
            breakdown = _measure_rs_breakdown(
                run,
                inputs,
                task=r"gmm_v2_fused_rs.*",
                warmup=args.warmup,
                iters=args.iters,
                trace_root=str(Path(args.trace_root) / variant / "breakdown"),
            )
            stages = breakdown["stage_samples_ms"]
            local_tokens = args.tokens // args.ep_size
            hidden_payload_item_bytes = 1 if fp8_hidden_all_gather else 2
            emit(
                {
                    "record_type": "fused_rs_fp8_hidden_ag",
                    "status": "ok" if contract["contract_ok"] else "correctness_failed",
                    "variant": variant,
                    "fp8_hidden_all_gather": fp8_hidden_all_gather,
                    "scale_semantics": (
                        "one_fp32_scale_per_ep_rank_physical_hidden_shard"
                        if fp8_hidden_all_gather
                        else None
                    ),
                    "process_count": jax.process_count(),
                    "process_index": jax.process_index(),
                    "visible_devices": visible_devices,
                    "ep_size": args.ep_size,
                    "num_tokens": args.tokens,
                    "hidden_size": GLM52_HIDDEN_SIZE,
                    "top_k": GLM52_TOP_K,
                    "rs_block_config": list(PRODUCTION_RS_CONFIG),
                    "compile_time_s": compile_time_s,
                    "correctness_contract_ok": contract["contract_ok"],
                    "correctness_rel_l2_threshold": args.correctness_rel_l2_threshold,
                    "full_vs_bf16_rel_l2": full_metrics["rel_l2"],
                    "full_vs_bf16_max_abs": full_metrics["max_abs"],
                    "full_vs_bf16_cosine": full_metrics["cosine"],
                    "padded_vs_bf16_rel_l2": padded_metrics["rel_l2"],
                    "same_config_padding_invariance_rel_l2": padding_invariance["rel_l2"],
                    "invalid_padding_max_abs": invalid_padding_max_abs,
                    "standard_samples_ms": standard_samples,
                    "standard_median_ms": _median(standard_samples),
                    "standard_timing_method": (
                        "benchmark.utils.multiple_iteration_timeit_from_trace"
                    ),
                    "call_samples_ms": breakdown["call_samples_ms"],
                    "call_median_ms": _median(breakdown["call_samples_ms"]),
                    "main_pallas_samples_ms": breakdown["task_samples_ms"],
                    "main_pallas_median_ms": _median(breakdown["task_samples_ms"]),
                    "hidden_quantize_samples_ms": stages["hidden_quantize"],
                    "hidden_quantize_median_ms": _median(stages["hidden_quantize"]),
                    "hidden_all_gather_samples_ms": stages["hidden_all_gather"],
                    "hidden_all_gather_median_ms": _median(stages["hidden_all_gather"]),
                    "hidden_scale_all_gather_samples_ms": stages[
                        "hidden_scale_all_gather"
                    ],
                    "hidden_scale_all_gather_median_ms": _median(
                        stages["hidden_scale_all_gather"]
                    ),
                    "hidden_scale_expand_samples_ms": stages[
                        "hidden_scale_expand"
                    ],
                    "hidden_scale_expand_median_ms": _median(
                        stages["hidden_scale_expand"]
                    ),
                    "hidden_dequantize_samples_ms": stages["hidden_dequantize"],
                    "hidden_dequantize_median_ms": _median(
                        stages["hidden_dequantize"]
                    ),
                    "topk_ids_all_gather_samples_ms": stages["topk_ids_all_gather"],
                    "topk_ids_all_gather_median_ms": _median(
                        stages["topk_ids_all_gather"]
                    ),
                    "routing_table_materialization_samples_ms": stages[
                        "routing_table_materialization"
                    ],
                    "routing_table_materialization_median_ms": _median(
                        stages["routing_table_materialization"]
                    ),
                    "hidden_all_gather_local_payload_bytes": (
                        local_tokens * GLM52_HIDDEN_SIZE * hidden_payload_item_bytes
                    ),
                    "hidden_all_gather_logical_output_bytes_per_device": (
                        args.tokens * GLM52_HIDDEN_SIZE * hidden_payload_item_bytes
                    ),
                    "hidden_scale_all_gather_logical_output_bytes_per_device": (
                        args.ep_size * 4 if fp8_hidden_all_gather else 0
                    ),
                    "requested_iterations": args.iters,
                    "representative_pid_semantics": True,
                    **routing_stats,
                }
            )

    set_fused_rs_block_sizes_override(None)
    set_fused_rs_routing_table_impl("jax")


if __name__ == "__main__":
    main()
