"""Strict EP32 A/B for Hidden AllGather placement and routing-table materialization."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    set_fused_rs_block_sizes_override,
    set_fused_rs_routing_table_impl,
)

from benchmark.moe.bench_fused_rs_moe import (
    GLM52_HIDDEN_SIZE,
    GLM52_TOP_K,
    _build_mesh,
    _comparison_metrics,
    _invalid_padding_max_abs,
    _make_inputs,
    _make_padded_inputs,
    _measure_rs_breakdown,
    _routing_stats,
    _rs_runner,
)

PRODUCTION_RS_CONFIG = (256, 6144, 1024, 2048, 1024, 2, 2)
PREOP_VARIANTS = (
    ("auto-jax", "auto", "jax"),
    ("tensorcore-jax", "tensorcore", "jax"),
    ("auto-pallas", "auto", "pallas"),
    ("tensorcore-pallas", "tensorcore", "pallas"),
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
    parser.add_argument("--trace-root", default="/tmp/sglang_jax_fused_rs_preop_ab")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--correctness-rel-l2-threshold", type=float, default=1e-6)
    args = parser.parse_args()

    if args.ep_size != 32 or len(jax.devices()) != 32:
        raise ValueError(
            "This A/B requires exactly EP32 on 32 visible devices; "
            f"got ep_size={args.ep_size}, devices={len(jax.devices())}"
        )
    if args.tokens != 65536:
        raise ValueError(f"This A/B is fixed to 65536 tokens; got {args.tokens}")
    if args.correctness_rel_l2_threshold <= 0:
        raise ValueError("--correctness-rel-l2-threshold must be positive")
    if args.jsonl is not None and jax.process_index() == 0:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("", encoding="utf-8")

    def emit(row: dict) -> None:
        encoded = json.dumps(row, sort_keys=True)
        if jax.process_index() == 0:
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

        for variant_name, hidden_backend, routing_impl in PREOP_VARIANTS:
            set_fused_rs_block_sizes_override(PRODUCTION_RS_CONFIG)
            set_fused_rs_routing_table_impl(routing_impl)
            jax.clear_caches()
            run = _rs_runner(
                mesh,
                layer_scope=False,
                hidden_all_gather_backend=hidden_backend,
            )

            compile_start = time.perf_counter()
            output = run(inputs)
            jax.block_until_ready(output)
            padded_output = run(padded_inputs)
            jax.block_until_ready(padded_output)
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
            correctness_ok = (
                full_metrics["all_finite"]
                and padded_metrics["all_finite"]
                and full_metrics["rel_l2"] <= args.correctness_rel_l2_threshold
                and padded_metrics["rel_l2"] <= args.correctness_rel_l2_threshold
                and invalid_padding_max_abs == 0.0
            )
            if not correctness_ok:
                raise AssertionError(
                    f"{variant_name} changed fused-RS semantics vs auto-jax: "
                    f"full_rel_l2={full_metrics['rel_l2']}, "
                    f"padded_rel_l2={padded_metrics['rel_l2']}, "
                    f"invalid_padding_max_abs={invalid_padding_max_abs}"
                )

            breakdown = _measure_rs_breakdown(
                run,
                inputs,
                task=r"gmm_v2_fused_rs.*",
                warmup=args.warmup,
                iters=args.iters,
                trace_root=str(Path(args.trace_root) / variant_name),
            )
            stage_samples = breakdown["stage_samples_ms"]
            call_samples = breakdown["call_samples_ms"]
            pallas_samples = breakdown["task_samples_ms"]
            hidden_samples = stage_samples["hidden_all_gather"]
            topk_samples = stage_samples["topk_ids_all_gather"]
            routing_table_samples = stage_samples["routing_table_materialization"]
            emit(
                {
                    "record_type": "fused_rs_preop_ab",
                    "status": "ok",
                    "variant": variant_name,
                    "hidden_all_gather_backend": hidden_backend,
                    "routing_table_impl": routing_impl,
                    "process_count": jax.process_count(),
                    "visible_devices": len(jax.devices()),
                    "ep_size": args.ep_size,
                    "num_tokens": args.tokens,
                    "hidden_size": GLM52_HIDDEN_SIZE,
                    "top_k": GLM52_TOP_K,
                    "rs_block_config": list(PRODUCTION_RS_CONFIG),
                    "compile_time_s": compile_time_s,
                    "correctness_vs_auto_jax_rel_l2": full_metrics["rel_l2"],
                    "padded_correctness_vs_auto_jax_rel_l2": padded_metrics["rel_l2"],
                    "same_config_padding_invariance_rel_l2_diagnostic": (
                        padding_invariance["rel_l2"]
                    ),
                    "invalid_padding_max_abs": invalid_padding_max_abs,
                    "call_samples_ms": call_samples,
                    "call_median_ms": _median(call_samples),
                    "main_pallas_samples_ms": pallas_samples,
                    "main_pallas_median_ms": _median(pallas_samples),
                    "hidden_all_gather_samples_ms": hidden_samples,
                    "hidden_all_gather_median_ms": _median(hidden_samples),
                    "topk_ids_all_gather_samples_ms": topk_samples,
                    "topk_ids_all_gather_median_ms": _median(topk_samples),
                    "routing_table_materialization_samples_ms": routing_table_samples,
                    "routing_table_materialization_median_ms": _median(
                        routing_table_samples
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
