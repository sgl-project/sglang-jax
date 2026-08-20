"""Differential accuracy check for the GLM-5.2 fused-RS MoE backend.

This checker intentionally uses expert-, reduction-, and channel-distinct data.
It compares fused-RS against the production fused-v2 backend at the routed-only
boundary, at the routed-plus-shared layer boundary, and for the isolated shared
delta.  It is correctness-only: no timing result from this script is valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from benchmark.moe.bench_fused_rs_moe import (
    GLM52_TOP_K,
    _build_mesh,
    _make_inputs,
    _parse_csv_ints,
    _routing_stats,
    _rs_runner,
    _v2_runner,
)
from sgl_jax.srt.kernels.fused_moe.fused_rs import fused_moe_func_rs


_MICRO_TOKENS = 64
_MICRO_EXPERTS = 32
_MICRO_HIDDEN = 256
_MICRO_INTERMEDIATE = 256


def _accuracy_metrics(reference: jax.Array, candidate: jax.Array) -> dict[str, float | bool]:
    reference_f32 = reference.astype(jnp.float32)
    candidate_f32 = candidate.astype(jnp.float32)
    diff = (reference_f32 - candidate_f32).reshape(-1)
    reference_flat = reference_f32.reshape(-1)
    candidate_flat = candidate_f32.reshape(-1)
    reference_norm = jnp.linalg.norm(reference_flat)
    candidate_norm = jnp.linalg.norm(candidate_flat)
    dot_product = jnp.sum(reference_flat * candidate_flat)
    denom = jnp.maximum(reference_norm, jnp.asarray(1e-12, dtype=jnp.float32))
    cosine_denom = jnp.maximum(
        reference_norm * candidate_norm,
        jnp.asarray(1e-12, dtype=jnp.float32),
    )
    return {
        "all_finite": bool(
            jnp.all(jnp.isfinite(reference_f32))
            & jnp.all(jnp.isfinite(candidate_f32))
        ),
        "rel_l2": float(jnp.linalg.norm(diff) / denom),
        "max_abs": float(jnp.max(jnp.abs(diff))),
        "reference_max_abs": float(jnp.max(jnp.abs(reference_f32))),
        "candidate_max_abs": float(jnp.max(jnp.abs(candidate_f32))),
        # ``jnp.vdot`` leaves the output sharding of a contraction over the
        # data×tensor token axis ambiguous.  Elementwise multiply followed by
        # an explicit global reduction has the intended replicated-scalar
        # semantics on the four-process EP32 mesh.
        "cosine_similarity": float(dot_product / cosine_denom),
    }


def _write_row(path: Path | None, row: dict) -> None:
    encoded = json.dumps(row, sort_keys=True)
    if jax.process_index() == 0:
        print(encoded, flush=True)
        if path is not None:
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write(encoded + "\n")


def _with_padded_extend_routes(
    mesh,
    inputs,
    *,
    num_tokens: int,
    ep_size: int,
    active_tokens_per_shard: int,
):
    """Apply the serving layout: valid prefix inside every EP/DP token shard."""
    tokens_per_shard = num_tokens // ep_size
    if not 0 < active_tokens_per_shard < tokens_per_shard:
        raise ValueError(
            "active_tokens_per_shard must be in "
            f"[1, {tokens_per_shard - 1}], got {active_tokens_per_shard}"
        )
    token_sharding = NamedSharding(mesh, P(("data", "tensor"), None))

    def mask_routes(topk_ids):
        token = jnp.arange(num_tokens, dtype=jnp.int32)
        valid = (token % tokens_per_shard) < active_tokens_per_shard
        return jnp.where(valid[:, None], topk_ids, -1), valid

    padded_topk_ids, valid_tokens = jax.jit(
        mask_routes,
        out_shardings=(token_sharding, NamedSharding(mesh, P(("data", "tensor")))),
    )(inputs[8])
    padded_inputs = (*inputs[:8], padded_topk_ids, *inputs[9:])
    return padded_inputs, valid_tokens


def _active_token_metrics(
    reference: jax.Array,
    candidate: jax.Array,
    valid_tokens: jax.Array,
) -> dict[str, float | bool]:
    if not isinstance(reference.sharding, NamedSharding):
        raise TypeError(f"Expected NamedSharding, got {reference.sharding}")
    reference = jax.sharding.reshard(reference, reference.sharding)
    candidate = jax.sharding.reshard(candidate, reference.sharding)
    valid = jax.sharding.reshard(
        valid_tokens,
        NamedSharding(reference.sharding.mesh, P(reference.sharding.spec[0])),
    )[:, None]
    return _accuracy_metrics(
        jnp.where(valid, reference, jnp.zeros_like(reference)),
        jnp.where(valid, candidate, jnp.zeros_like(candidate)),
    )


def _padding_max_abs(output: jax.Array, valid_tokens: jax.Array) -> float:
    if not isinstance(output.sharding, NamedSharding):
        raise TypeError(f"Expected NamedSharding, got {output.sharding}")
    valid = jax.sharding.reshard(
        valid_tokens,
        NamedSharding(output.sharding.mesh, P(output.sharding.spec[0])),
    )[:, None]
    return float(
        jnp.max(
            jnp.abs(
                jnp.where(valid, jnp.zeros_like(output), output).astype(jnp.float32)
            )
        )
    )


def _make_micro_reference_inputs(mesh):
    """Build a signed, expert-distinct case small enough for an explicit oracle."""
    expert_axis = ("data", "tensor")
    token_sharding = NamedSharding(mesh, P(expert_axis, None))
    weight_sharding = NamedSharding(mesh, P(expert_axis, None, None))
    scale_sharding = NamedSharding(mesh, P(expert_axis, None, None, None))

    def _tokens():
        token = jnp.arange(_MICRO_TOKENS, dtype=jnp.int32)[:, None]
        hidden = jnp.arange(_MICRO_HIDDEN, dtype=jnp.int32)[None, :]
        code = (token * 131 + hidden * 17 + (token % 11) * (hidden % 13)) % 16
        sign = jnp.where(code < 8, -1.0, 1.0)
        magnitude = 0.25 + (code % 4).astype(jnp.float32) * 0.25
        return (sign * magnitude).astype(jnp.bfloat16)

    def _weight(offset: int, shape: tuple[int, int, int]):
        expert = jnp.arange(shape[0], dtype=jnp.int32)[:, None, None]
        reduction = jnp.arange(shape[1], dtype=jnp.int32)[None, :, None]
        output = jnp.arange(shape[2], dtype=jnp.int32)[None, None, :]
        code = (
            expert * (37 + offset)
            + reduction * (17 + offset)
            + output * (29 + offset)
            + (reduction % 13) * (output % 11)
        ) % 16
        sign = jnp.where(code < 8, -1.0, 1.0)
        magnitude = 0.25 + (code % 4).astype(jnp.float32) * 0.25
        return (sign * magnitude).astype(jnp.float8_e4m3fn)

    def _scale(offset: int, out_size: int):
        expert = jnp.arange(_MICRO_EXPERTS, dtype=jnp.int32)[:, None, None, None]
        output = jnp.arange(out_size, dtype=jnp.int32)[None, None, None, :]
        value = 0.01 + ((expert * (3 + offset) + output * (5 + offset)) % 9).astype(
            jnp.float32
        ) * 0.001
        return value

    tokens = jax.jit(_tokens, out_shardings=token_sharding)()
    w1 = jax.jit(
        lambda: _weight(1, (_MICRO_EXPERTS, _MICRO_HIDDEN, _MICRO_INTERMEDIATE)),
        out_shardings=weight_sharding,
    )()
    w3 = jax.jit(
        lambda: _weight(5, (_MICRO_EXPERTS, _MICRO_HIDDEN, _MICRO_INTERMEDIATE)),
        out_shardings=weight_sharding,
    )()
    w2 = jax.jit(
        lambda: _weight(9, (_MICRO_EXPERTS, _MICRO_INTERMEDIATE, _MICRO_HIDDEN)),
        out_shardings=weight_sharding,
    )()
    w1_scale = jax.jit(
        lambda: _scale(1, _MICRO_INTERMEDIATE), out_shardings=scale_sharding
    )()
    w3_scale = jax.jit(
        lambda: _scale(5, _MICRO_INTERMEDIATE), out_shardings=scale_sharding
    )()
    w2_scale = jax.jit(
        lambda: _scale(9, _MICRO_HIDDEN), out_shardings=scale_sharding
    )()

    def _routing():
        token = jnp.arange(_MICRO_TOKENS, dtype=jnp.int32)[:, None]
        slot = jnp.arange(GLM52_TOP_K, dtype=jnp.int32)[None, :]
        topk_ids = (token * 13 + slot * 7 + (token % 5) * (slot + 1)) % _MICRO_EXPERTS
        logits = (
            ((token * 11 + slot * 19 + 3) % 23).astype(jnp.float32) - 11.0
        ) / 7.0
        return jax.nn.softmax(logits, axis=-1), topk_ids

    topk_weights, topk_ids = jax.jit(
        _routing,
        out_shardings=(token_sharding, token_sharding),
    )()
    return tokens, w1, w3, w2, w1_scale, w3_scale, w2_scale, topk_weights, topk_ids


def _micro_explicit_reference(
    inputs,
    *,
    quantize_ffn2: bool,
    selected_weight_sharding: NamedSharding,
    selected_scale_sharding: NamedSharding,
):
    """Evaluate the per-channel W8A8 contract without either fused kernel."""
    tokens, w1, w3, w2, s1, s3, s2, topk_weights, topk_ids = inputs
    fp8_max = jnp.asarray(jnp.finfo(jnp.float8_e4m3fn).max, dtype=jnp.float32)

    tokens_f32 = tokens.astype(jnp.float32)
    token_amax = jnp.max(jnp.abs(tokens_f32), axis=-1, keepdims=True)
    token_scale = jnp.maximum(token_amax, 1e-12) / fp8_max
    tokens_q = jnp.clip(tokens_f32 / token_scale, -fp8_max, fp8_max).astype(
        jnp.float8_e4m3fn
    )

    # The expert axis is sharded while the gather indices are token-sharded.
    # State the post-gather layout explicitly so JAX inserts the required
    # collective instead of rejecting an ambiguous advanced-indexing result.
    selected_w1 = w1.at[topk_ids].get(out_sharding=selected_weight_sharding)
    selected_w3 = w3.at[topk_ids].get(out_sharding=selected_weight_sharding)
    selected_s1 = s1[:, 0, 0, :].at[topk_ids].get(
        out_sharding=selected_scale_sharding
    )
    selected_s3 = s3[:, 0, 0, :].at[topk_ids].get(
        out_sharding=selected_scale_sharding
    )
    gate = jnp.einsum(
        "th,tkhi->tki",
        tokens_q,
        selected_w1,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    up = jnp.einsum(
        "th,tkhi->tki",
        tokens_q,
        selected_w3,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    activation_scale = token_scale[:, None, :]
    gate *= activation_scale * selected_s1
    up *= activation_scale * selected_s3
    intermediate = jax.nn.silu(gate) * up

    if quantize_ffn2:
        intermediate_amax = jnp.max(jnp.abs(intermediate), axis=-1, keepdims=True)
        intermediate_scale = jnp.maximum(intermediate_amax, 1e-12) / fp8_max
        intermediate_for_dot = jnp.clip(
            intermediate / intermediate_scale, -fp8_max, fp8_max
        ).astype(jnp.float8_e4m3fn)
    else:
        intermediate_scale = jnp.ones_like(intermediate[..., :1])
        intermediate_for_dot = intermediate.astype(jnp.bfloat16)

    selected_w2 = w2.at[topk_ids].get(out_sharding=selected_weight_sharding)
    selected_s2 = s2[:, 0, 0, :].at[topk_ids].get(
        out_sharding=selected_scale_sharding
    )
    expert_output = jnp.einsum(
        "tki,tkih->tkh",
        intermediate_for_dot,
        selected_w2,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    expert_output *= intermediate_scale * selected_s2
    return jnp.sum(
        expert_output * topk_weights.astype(jnp.float32)[..., None], axis=1
    ).astype(tokens.dtype)


def _run_micro_reference_check(mesh, output_path: Path | None) -> list[str]:
    inputs = _make_micro_reference_inputs(mesh)
    tokens, w1, w3, w2, s1, s3, s2, topk_weights, topk_ids = inputs
    rs_output = fused_moe_func_rs(
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
        topk=GLM52_TOP_K,
        renormalize=False,
        mesh=mesh,
        activation="silu",
        scoring_fn="softmax",
        topk_weights=topk_weights,
        topk_indices=topk_ids,
    )
    selected_weight_sharding = NamedSharding(
        mesh, P(("data", "tensor"), None, None, None)
    )
    selected_scale_sharding = NamedSharding(
        mesh, P(("data", "tensor"), None, None)
    )
    reference_fn = jax.jit(
        _micro_explicit_reference,
        static_argnames=(
            "quantize_ffn2",
            "selected_weight_sharding",
            "selected_scale_sharding",
        ),
    )
    reference_bf16_ffn2 = reference_fn(
        inputs,
        quantize_ffn2=False,
        selected_weight_sharding=selected_weight_sharding,
        selected_scale_sharding=selected_scale_sharding,
    )
    reference_fp8_ffn2 = reference_fn(
        inputs,
        quantize_ffn2=True,
        selected_weight_sharding=selected_weight_sharding,
        selected_scale_sharding=selected_scale_sharding,
    )
    jax.block_until_ready((rs_output, reference_bf16_ffn2, reference_fp8_ffn2))

    failures: list[str] = []
    for reference_name, reference in (
        ("explicit_bf16_ffn2", reference_bf16_ffn2),
        ("explicit_fp8_ffn2", reference_fp8_ffn2),
    ):
        metrics = _accuracy_metrics(reference, rs_output)
        passed = bool(metrics["all_finite"]) and (
            reference_name != "explicit_bf16_ffn2" or metrics["rel_l2"] <= 0.05
        )
        _write_row(
            output_path,
            {
                "status": "ok" if passed else "mismatch",
                "model": "glm-5.2-moe-geometry-micro",
                "measurement_scope": "routed_backend_explicit_reference",
                "reference": reference_name,
                "correctness_only": True,
                "input_profile": "expert_distinct_signed_non_cancelling",
                "process_count": jax.process_count(),
                "visible_devices": len(jax.devices()),
                "ep_size": 32,
                "logical_mesh": "data1-tensor32",
                "num_tokens": _MICRO_TOKENS,
                "num_experts": _MICRO_EXPERTS,
                "top_k": GLM52_TOP_K,
                "hidden_size": _MICRO_HIDDEN,
                "intermediate_size": _MICRO_INTERMEDIATE,
                "max_rel_l2_threshold": 0.05 if reference_name == "explicit_bf16_ffn2" else None,
                **metrics,
            },
        )
        if not passed:
            failures.append(f"micro reference={reference_name} rel_l2={metrics['rel_l2']}")
    return failures


def _run_padded_extend_check(
    mesh,
    output_path: Path | None,
    *,
    num_tokens: int,
    ep_size: int,
    active_tokens_per_shard: int,
    routing_seed: int,
    max_rel_l2: float,
) -> list[str]:
    """Compare the production 64K physical shape with sparse valid prefixes."""
    all_valid_inputs = _make_inputs(
        mesh,
        num_tokens,
        ep_size,
        routing_mode="random",
        routing_seed=routing_seed,
        layer_scope=True,
        input_profile="expert_distinct",
    )
    padded_inputs, valid_tokens = _with_padded_extend_routes(
        mesh,
        all_valid_inputs,
        num_tokens=num_tokens,
        ep_size=ep_size,
        active_tokens_per_shard=active_tokens_per_shard,
    )
    active_tokens = active_tokens_per_shard * ep_size
    failures: list[str] = []

    for scope, layer_scope in (("routed_backend", False), ("glm52_moe_layer", True)):
        v2_run = _v2_runner(mesh, num_tokens, layer_scope=layer_scope)
        rs_run = _rs_runner(mesh, layer_scope=layer_scope)
        v2_all_valid = v2_run(all_valid_inputs)
        rs_all_valid = rs_run(all_valid_inputs)
        v2_output = v2_run(padded_inputs)
        rs_output = rs_run(padded_inputs)
        jax.block_until_ready((v2_all_valid, rs_all_valid, v2_output, rs_output))
        metrics = _active_token_metrics(v2_output, rs_output, valid_tokens)
        all_valid_active_metrics = _active_token_metrics(
            v2_all_valid, rs_all_valid, valid_tokens
        )
        v2_padding_fidelity = _active_token_metrics(
            v2_all_valid, v2_output, valid_tokens
        )
        rs_padding_fidelity = _active_token_metrics(
            rs_all_valid, rs_output, valid_tokens
        )
        passed = (
            bool(metrics["all_finite"])
            and metrics["rel_l2"] <= max_rel_l2
            and v2_padding_fidelity["max_abs"] == 0.0
            and rs_padding_fidelity["max_abs"] == 0.0
        )
        row = {
            "status": "ok" if passed else "mismatch",
            "model": "glm-5.2",
            "measurement_scope": f"padded_extend_{scope}",
            "correctness_only": True,
            "input_profile": "expert_distinct_positive_active_with_padding",
            "includes_shared_expert": layer_scope,
            "process_count": jax.process_count(),
            "visible_devices": len(jax.devices()),
            "ep_size": ep_size,
            "logical_mesh": "data1-tensor32",
            "num_tokens_physical": num_tokens,
            "active_tokens": active_tokens,
            "active_tokens_per_shard": active_tokens_per_shard,
            "padding_tokens": num_tokens - active_tokens,
            "active_routed_rows": active_tokens * GLM52_TOP_K,
            "routing_mode": "seeded_gaussian_topk8_active_padding_sentinel",
            "routing_seed": routing_seed,
            "max_rel_l2_threshold": max_rel_l2,
            "all_valid_active_rel_l2": all_valid_active_metrics["rel_l2"],
            "all_valid_active_max_abs": all_valid_active_metrics["max_abs"],
            "v2_active_padding_fidelity_rel_l2": v2_padding_fidelity["rel_l2"],
            "v2_active_padding_fidelity_max_abs": v2_padding_fidelity["max_abs"],
            "rs_active_padding_fidelity_rel_l2": rs_padding_fidelity["rel_l2"],
            "rs_active_padding_fidelity_max_abs": rs_padding_fidelity["max_abs"],
            **metrics,
        }
        if not layer_scope:
            row["v2_padding_max_abs"] = _padding_max_abs(v2_output, valid_tokens)
            row["rs_padding_max_abs"] = _padding_max_abs(rs_output, valid_tokens)
            passed = passed and row["v2_padding_max_abs"] == 0.0
            passed = passed and row["rs_padding_max_abs"] == 0.0
            row["status"] = "ok" if passed else "mismatch"
        _write_row(output_path, row)
        if not passed:
            failures.append(
                f"padded tokens={num_tokens} scope={scope} rel_l2={metrics['rel_l2']} "
                f"v2_padding={row.get('v2_padding_max_abs')} "
                f"rs_padding={row.get('rs_padding_max_abs')}"
            )

    return failures


def main() -> None:
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="64,65536")
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--routing-seed", type=int, default=42)
    parser.add_argument("--max-rel-l2", type=float, default=0.10)
    parser.add_argument("--padded-total-tokens", type=int, default=0)
    parser.add_argument("--padded-active-tokens-per-shard", type=int, default=0)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    if args.ep_size != 32:
        raise ValueError("This checker is fixed to the production EP32 contract")
    if len(jax.devices()) != 32 or jax.process_count() != 4:
        raise ValueError(
            "Expected the exact 16-chip topology: 32 devices across 4 processes; "
            f"got devices={len(jax.devices())}, processes={jax.process_count()}"
        )

    if args.jsonl is not None and jax.process_index() == 0:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("", encoding="utf-8")

    mesh = _build_mesh(args.ep_size, replicas=1)
    failures: list[str] = []
    with jax.set_mesh(mesh):
        failures.extend(_run_micro_reference_check(mesh, args.jsonl))
        for num_tokens in _parse_csv_ints(args.tokens):
            inputs = _make_inputs(
                mesh,
                num_tokens,
                args.ep_size,
                routing_mode="random",
                routing_seed=args.routing_seed,
                layer_scope=True,
                input_profile="expert_distinct",
            )
            routing = _routing_stats(inputs[8])

            outputs: dict[str, jax.Array] = {}
            for scope, layer_scope in (("routed_backend", False), ("glm52_moe_layer", True)):
                v2_output = _v2_runner(
                    mesh, num_tokens, layer_scope=layer_scope
                )(inputs)
                rs_output = _rs_runner(mesh, layer_scope=layer_scope)(inputs)
                jax.block_until_ready((v2_output, rs_output))
                outputs[f"v2_{scope}"] = v2_output
                outputs[f"rs_{scope}"] = rs_output

                metrics = _accuracy_metrics(v2_output, rs_output)
                passed = bool(metrics["all_finite"]) and metrics["rel_l2"] <= args.max_rel_l2
                row = {
                    "status": "ok" if passed else "mismatch",
                    "model": "glm-5.2",
                    "measurement_scope": scope,
                    "correctness_only": True,
                    "input_profile": "expert_distinct_positive",
                    "includes_shared_expert": layer_scope,
                    "process_count": jax.process_count(),
                    "visible_devices": len(jax.devices()),
                    "ep_size": args.ep_size,
                    "logical_mesh": "data1-tensor32",
                    "num_tokens": num_tokens,
                    "routed_rows": num_tokens * GLM52_TOP_K,
                    "routing_mode": "random",
                    "routing_seed": args.routing_seed,
                    "max_rel_l2_threshold": args.max_rel_l2,
                    **routing,
                    **metrics,
                }
                _write_row(args.jsonl, row)
                if not passed:
                    failures.append(
                        f"tokens={num_tokens} scope={scope} rel_l2={metrics['rel_l2']}"
                    )

            layer_sharding = NamedSharding(mesh, P("data", None))
            v2_routed_for_delta = jax.sharding.reshard(
                outputs["v2_routed_backend"], layer_sharding
            )
            rs_routed_for_delta = jax.sharding.reshard(
                outputs["rs_routed_backend"], layer_sharding
            )
            v2_shared_delta = (
                outputs["v2_glm52_moe_layer"].astype(jnp.float32)
                - v2_routed_for_delta.astype(jnp.float32)
            )
            rs_shared_delta = (
                outputs["rs_glm52_moe_layer"].astype(jnp.float32)
                - rs_routed_for_delta.astype(jnp.float32)
            )
            shared_metrics = _accuracy_metrics(v2_shared_delta, rs_shared_delta)
            shared_passed = (
                bool(shared_metrics["all_finite"])
                and shared_metrics["rel_l2"] <= args.max_rel_l2
            )
            shared_row = {
                "status": "ok" if shared_passed else "mismatch",
                "model": "glm-5.2",
                "measurement_scope": "shared_expert_delta",
                "correctness_only": True,
                "input_profile": "expert_distinct_positive",
                "includes_shared_expert": True,
                "process_count": jax.process_count(),
                "visible_devices": len(jax.devices()),
                "ep_size": args.ep_size,
                "logical_mesh": "data1-tensor32",
                "num_tokens": num_tokens,
                "routing_mode": "random",
                "routing_seed": args.routing_seed,
                "max_rel_l2_threshold": args.max_rel_l2,
                **shared_metrics,
            }
            _write_row(args.jsonl, shared_row)
            if not shared_passed:
                failures.append(
                    "tokens="
                    f"{num_tokens} scope=shared_expert_delta rel_l2={shared_metrics['rel_l2']}"
                )

        if args.padded_total_tokens or args.padded_active_tokens_per_shard:
            if not args.padded_total_tokens or not args.padded_active_tokens_per_shard:
                raise ValueError(
                    "--padded-total-tokens and --padded-active-tokens-per-shard "
                    "must be set together"
                )
            failures.extend(
                _run_padded_extend_check(
                    mesh,
                    args.jsonl,
                    num_tokens=args.padded_total_tokens,
                    ep_size=args.ep_size,
                    active_tokens_per_shard=args.padded_active_tokens_per_shard,
                    routing_seed=args.routing_seed,
                    max_rel_l2=args.max_rel_l2,
                )
            )

    if failures:
        raise AssertionError("fused-RS correctness mismatches: " + "; ".join(failures))


if __name__ == "__main__":
    main()
