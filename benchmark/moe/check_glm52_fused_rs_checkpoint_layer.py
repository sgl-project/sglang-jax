"""Compare fused-v2 and fused-RS on one real GLM-5.2 MoE layer.

This is a correctness-only integration check.  It loads the first MoE layer
from the channel-wise GLM-5.2 checkpoint, derives routing with that layer's
real gate, and compares the two backends with and without the shared expert.
The physical token layout matches the 64K prefill compilation: every EP32
token shard has a short valid prefix followed by padding routes marked ``-1``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from benchmark.moe.bench_fused_rs_moe import _build_mesh
from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.fused_moe import FusedEPMoERS
from sgl_jax.srt.model_loader.loader import JAXModelLoader


def _accuracy_metrics(reference: jax.Array, candidate: jax.Array) -> dict:
    reference_f32 = reference.astype(jnp.float32)
    candidate_f32 = candidate.astype(jnp.float32)
    diff = (reference_f32 - candidate_f32).reshape(-1)
    reference_flat = reference_f32.reshape(-1)
    candidate_flat = candidate_f32.reshape(-1)
    reference_norm = jnp.linalg.norm(reference_flat)
    candidate_norm = jnp.linalg.norm(candidate_flat)
    dot = jnp.sum(reference_flat * candidate_flat)
    norm_floor = jnp.asarray(1e-12, dtype=jnp.float32)
    return {
        "all_finite": bool(
            jnp.all(jnp.isfinite(reference_f32))
            & jnp.all(jnp.isfinite(candidate_f32))
        ),
        "rel_l2": float(jnp.linalg.norm(diff) / jnp.maximum(reference_norm, norm_floor)),
        "max_abs": float(jnp.max(jnp.abs(diff))),
        "reference_max_abs": float(jnp.max(jnp.abs(reference_f32))),
        "candidate_max_abs": float(jnp.max(jnp.abs(candidate_f32))),
        "cosine_similarity": float(
            dot / jnp.maximum(reference_norm * candidate_norm, norm_floor)
        ),
    }


def _active_metrics(
    reference: jax.Array,
    candidate: jax.Array,
    valid_tokens: jax.Array,
) -> dict:
    if not isinstance(reference.sharding, NamedSharding):
        raise TypeError(f"Expected NamedSharding output, got {reference.sharding}")
    valid_sharding = NamedSharding(reference.sharding.mesh, P(reference.sharding.spec[0]))
    valid = jax.sharding.reshard(valid_tokens, valid_sharding)[:, None]
    return _accuracy_metrics(
        jnp.where(valid, reference, jnp.zeros_like(reference)),
        jnp.where(valid, candidate, jnp.zeros_like(candidate)),
    )


def _padding_max_abs(output: jax.Array, valid_tokens: jax.Array) -> float:
    if not isinstance(output.sharding, NamedSharding):
        raise TypeError(f"Expected NamedSharding output, got {output.sharding}")
    valid_sharding = NamedSharding(output.sharding.mesh, P(output.sharding.spec[0]))
    valid = jax.sharding.reshard(valid_tokens, valid_sharding)[:, None]
    return float(
        jnp.max(
            jnp.abs(
                jnp.where(valid, jnp.zeros_like(output), output).astype(jnp.float32)
            )
        )
    )


def _write_row(path: Path | None, row: dict) -> None:
    encoded = json.dumps(row, sort_keys=True)
    if jax.process_index() == 0:
        print(encoded, flush=True)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write(encoded + "\n")


def _truncate_to_first_moe_layer(model_config: ModelConfig) -> int:
    """Keep checkpoint indices intact while avoiding a full-model load."""
    first_moe_layer = int(getattr(model_config.hf_config, "first_k_dense_replace", 0))
    layer_count = first_moe_layer + 1
    if layer_count > int(model_config.num_hidden_layers):
        raise ValueError(
            f"first MoE layer {first_moe_layer} exceeds num_hidden_layers="
            f"{model_config.num_hidden_layers}"
        )

    for config in (model_config.hf_config, model_config.hf_text_config):
        config.num_hidden_layers = layer_count
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is not None:
            config.indexer_types = list(indexer_types[:layer_count])
        if hasattr(config, "index_skip_topk_offset"):
            config.index_skip_topk_offset = min(
                int(config.index_skip_topk_offset), layer_count
            )
    model_config.num_hidden_layers = layer_count
    return first_moe_layer


def _array_metadata(value: jax.Array | None) -> dict | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sharding": str(value.sharding),
    }


def _routing_stats(topk_ids: jax.Array, num_experts: int) -> dict:
    local_counts = np.zeros(num_experts, dtype=np.int64)
    for shard in topk_ids.addressable_shards:
        ids = np.asarray(shard.data).reshape(-1)
        ids = ids[(ids >= 0) & (ids < num_experts)]
        local_counts += np.bincount(ids, minlength=num_experts)
    gathered = np.asarray(
        multihost_utils.process_allgather(local_counts.astype(np.int32))
    )
    counts = gathered.reshape(-1, num_experts).astype(np.int64).sum(axis=0)
    return {
        "routing_observed_rows": int(counts.sum()),
        "routing_expert_rows_min": int(counts.min()),
        "routing_expert_rows_max": int(counts.max()),
        "routing_expert_rows_mean": float(counts.mean()),
        "routing_expert_rows_std": float(counts.std()),
    }


def _make_hidden_and_mask(
    mesh,
    *,
    tokens: int,
    hidden_size: int,
    ep_size: int,
    active_tokens_per_shard: int,
    seed: int,
):
    if tokens % ep_size:
        raise ValueError(f"tokens={tokens} must be divisible by ep_size={ep_size}")
    tokens_per_shard = tokens // ep_size
    if not 0 < active_tokens_per_shard < tokens_per_shard:
        raise ValueError(
            "active_tokens_per_shard must be within one physical token shard; "
            f"got {active_tokens_per_shard} for shard size {tokens_per_shard}"
        )

    token_sharding = NamedSharding(mesh, P(("data", "tensor"), None))
    mask_sharding = NamedSharding(mesh, P(("data", "tensor")))

    def build():
        hidden = jax.random.normal(
            jax.random.key(seed),
            (tokens, hidden_size),
            dtype=jnp.bfloat16,
        )
        token = jnp.arange(tokens, dtype=jnp.int32)
        valid = (token % tokens_per_shard) < active_tokens_per_shard
        # Production ignores padding rows.  Zeroing them also makes any
        # accidental shared-expert padding output directly observable.
        hidden = jnp.where(valid[:, None], hidden, jnp.zeros_like(hidden))
        return hidden, valid

    return jax.jit(build, out_shardings=(token_sharding, mask_sharding))()


def _derive_real_routes(layer, hidden_states, valid_tokens, token_sharding):
    def route(hidden, valid):
        router_logits = layer.moe_gate(hidden)
        correction_bias = (
            layer.moe_gate.bias.value if layer.moe_gate.bias is not None else None
        )
        topk_weights, topk_ids = layer.topk(
            router_logits,
            correction_bias,
            routing_sharding=token_sharding,
        )
        topk_ids = jnp.where(valid[:, None], topk_ids, -1)
        return topk_weights, topk_ids

    return jax.jit(
        route,
        out_shardings=(token_sharding, token_sharding),
    )(hidden_states, valid_tokens)


def _run_backend(
    mlp: FusedEPMoERS,
    hidden_states: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    *,
    use_rs: bool,
    include_shared: bool,
    out_sharding: NamedSharding,
) -> jax.Array:
    original = mlp.disable_shared_expert
    mlp.disable_shared_expert = not include_shared
    try:
        output = mlp(
            hidden_states,
            topk_weights,
            topk_ids,
            use_rs=use_rs,
            out_sharding=out_sharding,
        )
        return jax.block_until_ready(output)
    finally:
        mlp.disable_shared_expert = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quant-config", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--active-tokens-per-shard", type=int, default=64)
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rel-l2", type=float, default=0.01)
    parser.add_argument("--jsonl", type=Path)
    args = parser.parse_args()

    jax.distributed.initialize()
    if args.ep_size != 32:
        raise ValueError("This checkpoint comparison is fixed to EP32")
    if len(jax.devices()) != 32 or jax.process_count() != 4:
        raise ValueError(
            "Expected exactly 16 chips = 32 devices across 4 processes; "
            f"found devices={len(jax.devices())}, processes={jax.process_count()}"
        )
    if not (args.model_path / "config.json").is_file():
        raise FileNotFoundError(args.model_path / "config.json")
    if not args.quant_config.is_file():
        raise FileNotFoundError(args.quant_config)

    mesh = _build_mesh(args.ep_size, replicas=1)
    model_config = ModelConfig(
        model_path=str(args.model_path),
        dtype="bfloat16",
        quantization_config_path=str(args.quant_config),
        moe_backend="fused_rs",
    )
    model_config.validate_tensor_parallel_config(args.ep_size)
    model_config.configure_for_tensor_parallel(args.ep_size)
    first_moe_layer = _truncate_to_first_moe_layer(model_config)
    model_config.hf_config.ep_size = args.ep_size
    model_config.hf_config.ep_num_redundant_experts = 0
    model_config.hf_config.moe_backend = "fused_rs"
    model_config.hf_config.enable_sequence_parallel = False
    model_config.hf_config.use_jax_allreduce_metadata = True

    loader = JAXModelLoader(LoadConfig(load_format="jax"), mesh)
    # The normal service loader intentionally warms every safetensors file on
    # GCSFuse.  This checker loads only the prefix ending at the first MoE
    # layer, so warming the entire 78-layer checkpoint would be wasteful.  The
    # same initialization and WeightLoader path remain in use below.
    model_class = loader._initialize_model(model_config)
    model = loader._get_model(model_class, model_config)
    layer = model.model.layers[first_moe_layer]
    if not isinstance(layer.mlp, FusedEPMoERS):
        raise TypeError(
            f"Expected FusedEPMoERS at layer {first_moe_layer}, got {type(layer.mlp)}"
        )
    mlp = layer.mlp
    if mlp.w1_shared is None:
        raise ValueError("Checkpoint layer is missing the GLM shared expert")

    hidden_states, valid_tokens = _make_hidden_and_mask(
        mesh,
        tokens=args.tokens,
        hidden_size=mlp.hidden_size,
        ep_size=args.ep_size,
        active_tokens_per_shard=args.active_tokens_per_shard,
        seed=args.seed,
    )
    token_sharding = NamedSharding(mesh, P(("data", "tensor"), None))
    output_sharding = NamedSharding(mesh, P("data", None))
    topk_weights, topk_ids = _derive_real_routes(
        layer, hidden_states, valid_tokens, token_sharding
    )
    jax.block_until_ready((topk_weights, topk_ids))

    weight_metadata = {
        name: _array_metadata(
            getattr(mlp, name).value if getattr(mlp, name, None) is not None else None
        )
        for name in (
            "w1",
            "w3",
            "w2",
            "w1_scale",
            "w3_scale",
            "w2_scale",
            "w1_shared",
            "w3_shared",
            "w2_shared",
            "w1_shared_scale",
            "w3_shared_scale",
            "w2_shared_scale",
        )
    }
    contract = {
        "kind": "checkpoint_contract",
        "model_path": str(args.model_path),
        "quant_config": args.quant_config.name,
        "first_moe_layer": first_moe_layer,
        "loaded_layer_count": first_moe_layer + 1,
        "physical_tokens": args.tokens,
        "tokens_per_shard": args.tokens // args.ep_size,
        "active_tokens_per_shard": args.active_tokens_per_shard,
        "active_tokens": args.active_tokens_per_shard * args.ep_size,
        "ep_size": args.ep_size,
        "process_count": jax.process_count(),
        "device_count": len(jax.devices()),
        "logical_mesh": "data1-tensor32",
        "seed": args.seed,
        "routing": _routing_stats(topk_ids, mlp.num_experts),
        "weights": weight_metadata,
    }
    _write_row(args.jsonl, contract)

    failures = []
    outputs = {}
    for scope, include_shared in (("routed_only", False), ("routed_plus_shared", True)):
        v2_output = _run_backend(
            mlp,
            hidden_states,
            topk_weights,
            topk_ids,
            use_rs=False,
            include_shared=include_shared,
            out_sharding=output_sharding,
        )
        rs_output = _run_backend(
            mlp,
            hidden_states,
            topk_weights,
            topk_ids,
            use_rs=True,
            include_shared=include_shared,
            out_sharding=output_sharding,
        )
        outputs[(scope, "v2")] = v2_output
        outputs[(scope, "rs")] = rs_output
        metrics = _active_metrics(v2_output, rs_output, valid_tokens)
        row = {
            "kind": "checkpoint_layer_correctness",
            "scope": scope,
            "reference": "fused_v2",
            "candidate": "fused_rs",
            "first_moe_layer": first_moe_layer,
            "physical_tokens": args.tokens,
            "active_tokens": args.active_tokens_per_shard * args.ep_size,
            "v2_padding_max_abs": _padding_max_abs(v2_output, valid_tokens),
            "rs_padding_max_abs": _padding_max_abs(rs_output, valid_tokens),
            **metrics,
        }
        _write_row(args.jsonl, row)
        if not row["all_finite"] or row["rel_l2"] > args.max_rel_l2:
            failures.append(
                f"{scope}: rel_l2={row['rel_l2']}, all_finite={row['all_finite']}"
            )
        if scope == "routed_only" and (
            row["v2_padding_max_abs"] != 0.0 or row["rs_padding_max_abs"] != 0.0
        ):
            failures.append(
                f"{scope}: padding V2={row['v2_padding_max_abs']}, "
                f"RS={row['rs_padding_max_abs']}"
            )

    # Isolate the two remaining error sources without changing the production
    # TopK8 route set.  Zeroing slots 1..7 leaves one weighted expert output per
    # token and distinguishes expert-math drift from TopK accumulation order.
    def keep_first_slot(weights):
        slot = jnp.arange(weights.shape[1], dtype=jnp.int32)
        return jnp.where(slot[None, :] == 0, weights, jnp.zeros_like(weights))

    # Run the elementwise mask under jit so Explicit mesh assignment is carried
    # by the input/output shardings rather than creating a single-device scalar
    # scatter outside a mesh context.
    top1_weights = jax.jit(
        keep_first_slot,
        out_shardings=token_sharding,
    )(topk_weights)
    top1_v2 = _run_backend(
        mlp,
        hidden_states,
        top1_weights,
        topk_ids,
        use_rs=False,
        include_shared=False,
        out_sharding=output_sharding,
    )
    top1_rs = _run_backend(
        mlp,
        hidden_states,
        top1_weights,
        topk_ids,
        use_rs=True,
        include_shared=False,
        out_sharding=output_sharding,
    )
    _write_row(
        args.jsonl,
        {
            "kind": "checkpoint_layer_diagnostic",
            "scope": "routed_first_slot_only",
            "reference": "fused_v2",
            "candidate": "fused_rs",
            **_active_metrics(top1_v2, top1_rs, valid_tokens),
        },
    )

    # Both backends return bf16 at their boundary, so the subtraction includes
    # one boundary-rounding step.  It is still a useful shared-expert diagnostic:
    # a large delta here means the routed-only discrepancy is not the whole gap.
    v2_shared_delta = (
        outputs[("routed_plus_shared", "v2")].astype(jnp.float32)
        - outputs[("routed_only", "v2")].astype(jnp.float32)
    )
    rs_shared_delta = (
        outputs[("routed_plus_shared", "rs")].astype(jnp.float32)
        - outputs[("routed_only", "rs")].astype(jnp.float32)
    )
    _write_row(
        args.jsonl,
        {
            "kind": "checkpoint_layer_diagnostic",
            "scope": "shared_delta_boundary_estimate",
            "reference": "fused_v2_full_minus_routed",
            "candidate": "fused_rs_full_minus_routed",
            **_active_metrics(v2_shared_delta, rs_shared_delta, valid_tokens),
        },
    )

    multihost_utils.sync_global_devices("checkpoint-layer-correctness-complete")
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
