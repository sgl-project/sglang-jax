"""Production-shape one-scale-per-rank FP8 calibration probe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from benchmark.moe.bench_fused_rs_moe import (
    GLM52_HIDDEN_SIZE,
    _build_mesh,
    _comparison_metrics,
    _make_inputs,
    _rs_runner,
)
from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    set_fused_rs_block_sizes_override,
    set_fused_rs_routing_table_impl,
)


PRODUCTION_RS_CONFIG = (256, 6144, 1024, 2048, 1024, 2, 2)
_TOKEN_PATTERN_PERIOD = 85
_HIDDEN_PATTERN_PERIOD = 17


def _parse_multipliers(value: str) -> tuple[float, ...]:
    multipliers = tuple(float(item) for item in value.split(","))
    if not multipliers or any(multiplier <= 0 for multiplier in multipliers):
        raise ValueError("scale multipliers must be positive")
    return multipliers


def _weighted_input_quantization_metrics(
    multiplier: float,
    *,
    num_tokens: int,
    ep_size: int,
) -> dict:
    """Evaluate the exact benchmark token pattern without materializing 32Kx6K."""
    token_residue = np.arange(_TOKEN_PATTERN_PERIOD, dtype=np.int32)[:, None]
    hidden_residue = np.arange(_HIDDEN_PATTERN_PERIOD, dtype=np.int32)[None, :]
    pattern = (
        0.015
        + ((token_residue * 7 + hidden_residue * 3) % 17).astype(np.float32)
        * 0.00025
        + (token_residue % 5).astype(np.float32) * 0.001
    )
    pattern = pattern.astype(ml_dtypes.bfloat16).astype(np.float32)
    hidden_counts = np.bincount(
        np.arange(GLM52_HIDDEN_SIZE, dtype=np.int32) % _HIDDEN_PATTERN_PERIOD,
        minlength=_HIDDEN_PATTERN_PERIOD,
    )
    local_tokens = num_tokens // ep_size
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    numerator = 0.0
    denominator = 0.0
    clipped_count = 0
    element_count = 0
    rank_scales = []
    rank_rel_l2 = []
    for rank in range(ep_size):
        token_counts = np.bincount(
            np.arange(
                rank * local_tokens,
                (rank + 1) * local_tokens,
                dtype=np.int32,
            )
            % _TOKEN_PATTERN_PERIOD,
            minlength=_TOKEN_PATTERN_PERIOD,
        )
        weights = token_counts[:, None] * hidden_counts[None, :]
        amax = float(np.max(np.abs(pattern[weights != 0])))
        scale = max(amax, 1e-12) / fp8_max * multiplier
        normalized = pattern / scale
        clipped_count += int(np.sum(weights[np.abs(normalized) > fp8_max]))
        element_count += int(np.sum(weights))
        quantized = np.clip(normalized, -fp8_max, fp8_max).astype(
            ml_dtypes.float8_e4m3fn
        )
        dequantized = quantized.astype(np.float32) * scale
        squared_delta = float(np.sum(weights * np.square(dequantized - pattern)))
        squared_reference = float(np.sum(weights * np.square(pattern)))
        numerator += squared_delta
        denominator += squared_reference
        rank_scales.append(scale)
        rank_rel_l2.append((squared_delta / squared_reference) ** 0.5)
    return {
        "rel_l2": (numerator / denominator) ** 0.5,
        "rank_rel_l2": rank_rel_l2,
        "rank_scales": rank_scales,
        "clipped_fraction": clipped_count / element_count,
    }


def main() -> None:
    jax.distributed.initialize()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--routing-seed", type=int, default=42)
    parser.add_argument(
        "--scale-multipliers",
        default="0.95,0.96,0.97,0.98,0.99,1.0",
    )
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()
    multipliers = _parse_multipliers(args.scale_multipliers)

    visible_devices = len(jax.devices())
    if (args.ep_size, args.tokens, visible_devices) != (8, 32768, 8):
        raise ValueError(
            "scale probe requires one exact EP8/32K group on 8 devices; got "
            f"EP{args.ep_size}/{args.tokens}/{visible_devices}"
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
    with jax.set_mesh(mesh):
        inputs = _make_inputs(
            mesh,
            args.tokens,
            args.ep_size,
            routing_seed=args.routing_seed,
            layer_scope=False,
            input_profile="expert_distinct",
        )
        set_fused_rs_block_sizes_override(PRODUCTION_RS_CONFIG)
        set_fused_rs_routing_table_impl("pallas")
        baseline_run = _rs_runner(mesh, layer_scope=False)
        baseline = baseline_run(inputs)
        jax.block_until_ready(baseline)

        for multiplier in multipliers:
            jax.clear_caches()
            run = _rs_runner(
                mesh,
                layer_scope=False,
                fp8_hidden_all_gather=True,
                _fp8_hidden_direct_prequantized=True,
                _fp8_hidden_scale_multiplier=multiplier,
            )
            compile_start = time.perf_counter()
            output = run(inputs)
            jax.block_until_ready(output)
            compile_time_s = time.perf_counter() - compile_start
            output_metrics = _comparison_metrics(baseline, output)
            emit(
                {
                    "record_type": "fused_rs_fp8_hidden_ag_scale_probe",
                    "status": "ok",
                    "ep_size": args.ep_size,
                    "num_tokens": args.tokens,
                    "visible_devices": visible_devices,
                    "scale_multiplier": multiplier,
                    "input_quantization": _weighted_input_quantization_metrics(
                        multiplier,
                        num_tokens=args.tokens,
                        ep_size=args.ep_size,
                    ),
                    "final_vs_bf16": output_metrics,
                    "compile_time_s": compile_time_s,
                    "rs_block_config": list(PRODUCTION_RS_CONFIG),
                }
            )

    set_fused_rs_block_sizes_override(None)
    set_fused_rs_routing_table_impl("jax")


if __name__ == "__main__":
    main()
