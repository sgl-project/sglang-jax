"""Fixed-shape A/B for Hidden AllGather placement and routing materialization."""

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
    _hidden_all_gather_probe_runner,
    _invalid_padding_max_abs,
    _make_inputs,
    _make_padded_inputs,
    _measure_rs_breakdown,
    _routing_stats,
    _rs_runner,
)
from benchmark.moe.fused_rs_preop_contract import (
    DEFAULT_FINAL_REL_L2_THRESHOLD,
    evaluate_preop_variant_contract,
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
    parser.add_argument(
        "--correctness-rel-l2-threshold",
        type=float,
        default=DEFAULT_FINAL_REL_L2_THRESHOLD,
    )
    args = parser.parse_args()

    visible_devices = len(jax.devices())
    contract = (args.ep_size, args.tokens, visible_devices)
    supported_contracts = {
        (32, 65536, 32),  # strict production-shape measurement
        (8, 32768, 16),  # two replicated EP8 groups on an 8-chip slice
    }
    if contract not in supported_contracts:
        raise ValueError(
            "This A/B is fixed to EP32/64K on 32 devices or EP8/32K on "
            "16 devices; got "
            f"ep_size={args.ep_size}, tokens={args.tokens}, devices={visible_devices}"
        )
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
    baseline_hidden_gather = None
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
            hidden_probe = _hidden_all_gather_probe_runner(
                mesh,
                hidden_all_gather_backend=hidden_backend,
            )

            hidden_probe_compile_start = time.perf_counter()
            hidden_gather = hidden_probe(inputs[0])
            jax.block_until_ready(hidden_gather)
            hidden_probe_compile_time_s = (
                time.perf_counter() - hidden_probe_compile_start
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
                baseline_hidden_gather = hidden_gather
            hidden_gather_metrics = _comparison_metrics(
                baseline_hidden_gather,
                hidden_gather,
            )
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
            correctness_contract = evaluate_preop_variant_contract(
                hidden_gather_all_finite=hidden_gather_metrics["all_finite"],
                hidden_gather_max_abs=hidden_gather_metrics["max_abs"],
                full_all_finite=full_metrics["all_finite"],
                full_rel_l2=full_metrics["rel_l2"],
                padded_all_finite=padded_metrics["all_finite"],
                padded_rel_l2=padded_metrics["rel_l2"],
                invalid_padding_max_abs=invalid_padding_max_abs,
                final_rel_l2_threshold=args.correctness_rel_l2_threshold,
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
                    "status": (
                        "ok"
                        if correctness_contract["contract_ok"]
                        else "correctness_failed"
                    ),
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
                    "hidden_all_gather_probe_compile_time_s": (
                        hidden_probe_compile_time_s
                    ),
                    "hidden_all_gather_exact": correctness_contract[
                        "hidden_gather_exact"
                    ],
                    "hidden_all_gather_vs_auto_rel_l2": hidden_gather_metrics[
                        "rel_l2"
                    ],
                    "hidden_all_gather_vs_auto_max_abs": hidden_gather_metrics[
                        "max_abs"
                    ],
                    "final_output_ok": correctness_contract["final_output_ok"],
                    "correctness_contract_ok": correctness_contract["contract_ok"],
                    "correctness_rel_l2_threshold": (
                        args.correctness_rel_l2_threshold
                    ),
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
