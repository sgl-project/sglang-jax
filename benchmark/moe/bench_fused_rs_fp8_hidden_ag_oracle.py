"""Target-TPU oracle for the prequantized FP8 Hidden AllGather path.

This diagnostic deliberately separates three error sources:

1. BF16 hidden -> one-scale-per-rank FP8 quantization;
2. the explicit FP8 MoE reference -> the per-row W8A8 reference;
3. the Pallas fused-RS result -> the explicit FP8 MoE reference.

Small full-resident M128/M256/M384 geometries keep production streaming-weight
effects out of the first diagnosis.  Production M256 remains a separate A/B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.fused_moe.fused_rs import fused_moe_func_rs
from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    set_fused_rs_block_sizes_override,
)
from sgl_jax.test.kernels.fused_moe_rs_test import (
    FP8,
    _TOP_K,
    _explicit_reference,
    _make_inputs,
    _relative_l2,
)
from sgl_jax.test.test_utils import create_device_mesh


_CASES = (
    ("m128_full_resident", (128, 512, 512, 512, 512, 1, 1), 48, 3),
    ("m256_full_resident", (256, 512, 512, 512, 512, 1, 1), 72, 5),
    ("m384_full_resident", (384, 512, 512, 512, 512, 1, 1), 136, 8),
)


def _comparison(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "rel_l2": _relative_l2(actual, expected),
        "max_abs": float(np.max(np.abs(delta))),
        "expected_l2": float(np.linalg.norm(expected.astype(np.float64))),
    }


def _padded_inputs(reference_inputs, *, ep_size: int, active_per_device: int):
    num_tokens = reference_inputs[0].shape[0]
    local_tokens = num_tokens // ep_size
    valid_mask = (
        np.arange(num_tokens, dtype=np.int32) % local_tokens
    ) < active_per_device
    valid = jnp.asarray(valid_mask)[:, None]
    topk_weights = jnp.where(valid, reference_inputs[-2], 0.0)
    topk_ids = jnp.where(valid, reference_inputs[-1], -1)
    return (*reference_inputs[:-2], topk_weights, topk_ids), valid_mask


def _run_kernel(mesh, kernel_inputs, *, topk_weights, topk_ids):
    tokens, w1, w3, w2, s1, s3, s2, _, _ = kernel_inputs
    return fused_moe_func_rs(
        hidden_states=tokens,
        w1=w1,
        w3=w3,
        w2=w2,
        w1_scale=s1,
        w3_scale=s3,
        w2_scale=s2,
        w1_bias=None,
        w2_bias=None,
        gating_output=None,
        topk=_TOP_K,
        renormalize=False,
        mesh=mesh,
        activation="silu",
        scoring_fn="softmax",
        topk_weights=topk_weights,
        topk_indices=topk_ids,
        fp8_hidden_all_gather=True,
    )


def _quantization_diagnostics(tokens: jax.Array, *, ep_size: int) -> dict:
    tokens_f32 = tokens.astype(jnp.float32)
    local_tokens = tokens.shape[0] // ep_size
    shards = tokens_f32.reshape(ep_size, local_tokens, tokens.shape[-1])
    amax = jnp.max(jnp.abs(shards), axis=(1, 2))
    fp8_max = jnp.asarray(jnp.finfo(FP8).max, dtype=jnp.float32)
    scales = jnp.maximum(amax, 1e-12) / fp8_max
    quantized = jnp.clip(
        shards / scales[:, None, None], -fp8_max, fp8_max
    ).astype(FP8)
    dequantized = quantized.astype(jnp.float32) * scales[:, None, None]
    jax.block_until_ready((amax, scales, dequantized))
    original_host = np.asarray(jax.device_get(shards), dtype=np.float32)
    dequant_host = np.asarray(jax.device_get(dequantized), dtype=np.float32)
    return {
        "rank_amax": np.asarray(jax.device_get(amax), dtype=np.float32).tolist(),
        "rank_scale": np.asarray(jax.device_get(scales), dtype=np.float32).tolist(),
        "dequant_vs_bf16": _comparison(original_host, dequant_host),
    }


def _reference(reference_inputs, token_sharding, *, fp8_hidden_all_gather: bool):
    return jax.jit(
        _explicit_reference,
        static_argnames=("quantized", "fp8_hidden_all_gather", "ep_size"),
        out_shardings=token_sharding,
    )(
        reference_inputs,
        quantized=True,
        fp8_hidden_all_gather=fp8_hidden_all_gather,
        ep_size=token_sharding.mesh.size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--kernel-rel-l2-threshold", type=float, default=0.01)
    args = parser.parse_args()
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.write_text("", encoding="utf-8")

    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    with jax.set_mesh(mesh):
        for name, config, num_tokens, active_per_device in _CASES:
            reference_inputs, kernel_inputs, token_sharding = _make_inputs(
                mesh,
                quantized=True,
                num_tokens=num_tokens,
                distinct_shard_scales=True,
            )
            padded_reference_inputs, valid_mask = _padded_inputs(
                reference_inputs,
                ep_size=mesh.size,
                active_per_device=active_per_device,
            )
            padded_weights = jax.sharding.reshard(
                padded_reference_inputs[-2], token_sharding
            )
            padded_ids = jax.sharding.reshard(
                padded_reference_inputs[-1], token_sharding
            )

            per_row_reference = _reference(
                reference_inputs, token_sharding, fp8_hidden_all_gather=False
            )
            fp8_reference = _reference(
                reference_inputs, token_sharding, fp8_hidden_all_gather=True
            )
            padded_per_row_reference = _reference(
                padded_reference_inputs,
                token_sharding,
                fp8_hidden_all_gather=False,
            )
            padded_fp8_reference = _reference(
                padded_reference_inputs,
                token_sharding,
                fp8_hidden_all_gather=True,
            )

            set_fused_rs_block_sizes_override(config)
            actual = _run_kernel(
                mesh,
                kernel_inputs,
                topk_weights=kernel_inputs[-2],
                topk_ids=kernel_inputs[-1],
            )
            padded_actual = _run_kernel(
                mesh,
                kernel_inputs,
                topk_weights=padded_weights,
                topk_ids=padded_ids,
            )
            jax.block_until_ready(
                (
                    per_row_reference,
                    fp8_reference,
                    padded_per_row_reference,
                    padded_fp8_reference,
                    actual,
                    padded_actual,
                )
            )

            host = {
                "per_row": np.asarray(
                    jax.device_get(per_row_reference), dtype=np.float32
                ),
                "fp8": np.asarray(jax.device_get(fp8_reference), dtype=np.float32),
                "padded_per_row": np.asarray(
                    jax.device_get(padded_per_row_reference), dtype=np.float32
                ),
                "padded_fp8": np.asarray(
                    jax.device_get(padded_fp8_reference), dtype=np.float32
                ),
                "actual": np.asarray(jax.device_get(actual), dtype=np.float32),
                "padded_actual": np.asarray(
                    jax.device_get(padded_actual), dtype=np.float32
                ),
            }
            kernel_full = _comparison(host["fp8"], host["actual"])
            kernel_padded = _comparison(
                host["padded_fp8"][valid_mask], host["padded_actual"][valid_mask]
            )
            padding_invariance = _comparison(
                host["actual"][valid_mask], host["padded_actual"][valid_mask]
            )
            invalid_max_abs = float(np.max(np.abs(host["padded_actual"][~valid_mask])))
            contract_ok = (
                np.isfinite(host["actual"]).all()
                and np.isfinite(host["padded_actual"]).all()
                and kernel_full["rel_l2"] <= args.kernel_rel_l2_threshold
                and kernel_padded["rel_l2"] <= args.kernel_rel_l2_threshold
                and padding_invariance["rel_l2"] <= args.kernel_rel_l2_threshold
                and invalid_max_abs == 0.0
            )
            row = {
                "record_type": "fused_rs_fp8_hidden_ag_explicit_oracle",
                "case": name,
                "status": "ok" if contract_ok else "correctness_failed",
                "contract_ok": bool(contract_ok),
                "ep_size": mesh.size,
                "num_tokens": num_tokens,
                "active_tokens_per_device": active_per_device,
                "rs_block_config": list(config),
                "kernel_rel_l2_threshold": args.kernel_rel_l2_threshold,
                "quantization": _quantization_diagnostics(
                    reference_inputs[0], ep_size=mesh.size
                ),
                "explicit_fp8_vs_per_row_full": _comparison(
                    host["per_row"], host["fp8"]
                ),
                "explicit_fp8_vs_per_row_padded": _comparison(
                    host["padded_per_row"][valid_mask],
                    host["padded_fp8"][valid_mask],
                ),
                "pallas_vs_explicit_fp8_full": kernel_full,
                "pallas_vs_explicit_fp8_padded": kernel_padded,
                "pallas_padding_invariance": padding_invariance,
                "invalid_padding_max_abs": invalid_max_abs,
            }
            encoded = json.dumps(row, sort_keys=True)
            print(encoded, flush=True)
            with args.jsonl.open("a", encoding="utf-8") as output:
                output.write(encoded + "\n")

    set_fused_rs_block_sizes_override(None)


if __name__ == "__main__":
    main()
