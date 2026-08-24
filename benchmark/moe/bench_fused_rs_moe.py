"""Compare fused_rs with fused_v2 using the GLM-5.2 routed-MoE shape.

Timing uses the same ``multiple_iteration_timeit_from_trace`` helper as the
existing MoE benchmarks.  With ``--layer-scope``, both variants also include
GLM's shared expert and the final reshard back to the caller's
``P("data", None)`` layout. Gate/top-k remain outside because routing is held
identical across variants.

Example (32 TPU devices):
  python -m benchmark.moe.bench_fused_rs_moe \
    --tokens 65536 --ep-size 32 \
    --jsonl /tmp/glm52-fused-rs-ep32-64k.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.fused_moe.fused_rs.fused_moe_rs import (
    _fused_moe_func_rs_impl,
    get_moe_expert_axis,
)
from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    FusedRsBlockConfig,
    get_last_fused_rs_block_sizes,
    set_fused_rs_block_sizes_override,
    set_fused_rs_routing_table_impl,
)
from sgl_jax.srt.kernels.fused_moe.v2.kernel import FusedMoEBlockConfig, fused_ep_moe_v2
from sgl_jax.srt.layers.fused_moe import fused_rs_shared_expert

from benchmark.moe.fused_rs_tuning import (
    GLM52_RS_REFERENCE_CONFIG,
    analyze_rs_config,
    generate_rs_tuning_configs,
)
from benchmark.utils import (
    multiple_iteration_profile_from_trace,
    multiple_iteration_timeit_from_trace,
)

GLM52_NUM_EXPERTS = 256
GLM52_TOP_K = 8
GLM52_HIDDEN_SIZE = 6144
GLM52_INTERMEDIATE_SIZE = 2048
GLM52_QUANT_BLOCK_K = None
DEFAULT_TOKENS = (65536,)
DEFAULT_TUNE_TILE_MS = (128, 256, 384)

# Production GLM-5.2 v7x EP32 W8A8 per-channel config from
# v2/tuned_block_configs.py in the private epic/glm_5_2 baseline. Keeping
# the fixed-shape benchmark self-contained avoids importing the serving utility
# package (and its unrelated HTTP/ZMQ dependencies) in a minimal TPU image.
GLM52_V2_BLOCK_CONFIGS = {
    # Existing EP32 direct-scaled-dot config from v2/bench_compare.py.  This
    # small case is a correctness discriminator for the scalar-prefetch route;
    # it is not a new tuning result.
    64: (8, 512, 8, 256, 8),
    65536: (64, 1024, 128, 1024, 128),
}


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_rs_configs(value: str | None) -> tuple[FusedRsBlockConfig | None, ...]:
    """Parse ``default;tm,tk1,tn1,tk2,tn2,w1bufs,w2bufs``."""
    if value is None:
        return (None,)
    configs: list[FusedRsBlockConfig | None] = []
    for raw_config in value.split(";"):
        raw_config = raw_config.strip()
        if not raw_config:
            continue
        if raw_config == "default":
            configs.append(None)
            continue
        values = _parse_csv_ints(raw_config)
        if len(values) != 7:
            raise ValueError(
                "Each --rs-configs entry must be 'default' or seven integers: "
                "tile_m,tile_k1,tile_n1,tile_k2,tile_n2,num_w1_bufs,num_w2_bufs"
            )
        configs.append(values)
    if not configs:
        raise ValueError("--rs-configs did not contain any candidates")
    return tuple(configs)


def _select_rs_configs(
    *, tune_rs: bool, rs_configs: str | None, tune_tile_ms: str
) -> tuple[FusedRsBlockConfig | None, ...]:
    if not tune_rs:
        return _parse_rs_configs(rs_configs)
    if rs_configs is None:
        return generate_rs_tuning_configs(_parse_csv_ints(tune_tile_ms))
    configs = _parse_rs_configs(rs_configs)
    if any(config is None for config in configs):
        raise ValueError(
            "--tune-rs with --rs-configs requires concrete seven-integer configs"
        )
    return configs


def _config_label(config: FusedRsBlockConfig | None) -> str:
    return "default" if config is None else "-".join(map(str, config))


def _build_mesh(ep_size: int):
    visible_devices = len(jax.devices())
    if visible_devices % ep_size:
        raise ValueError(
            f"visible_devices={visible_devices} must be divisible by ep_size={ep_size}"
        )
    replica_count = visible_devices // ep_size
    # The production EP32 case keeps the model's joint data+tensor expert axis.
    # For the smaller EP8 diagnostic, name the outer axis ``replica`` so the
    # fused-RS kernel forms two independent tensor8 EP groups rather than one
    # accidental data2*tensor8 EP16 group.
    replica_axis = "data" if replica_count == 1 else "replica"
    return jax.make_mesh(
        (replica_count, ep_size),
        (replica_axis, "tensor"),
    )


def _make_array(shape, dtype, sharding, fill_value):
    return jax.jit(
        lambda: jnp.full(shape, fill_value, dtype=dtype),
        out_shardings=sharding,
    )()


def _make_patterned_array(shape, dtype, sharding, *, kind: str):
    """Create sharded, deterministic data that exposes indexing/scale mixups.

    Constant expert weights made the original V2/RS check invariant to expert
    routing and to a gate/up swap.  These patterns deliberately vary all
    semantically relevant axes while keeping FP8 payloads small and exactly
    representable.
    """

    def make_value():
        if kind == "tokens":
            token = jnp.arange(shape[0], dtype=jnp.int32)[:, None]
            hidden = jnp.arange(shape[1], dtype=jnp.int32)[None, :]
            # Keep the reference output well away from cancellation-driven
            # near-zero values, while still distinguishing token and hidden
            # coordinates.  The earlier signed pattern made ordinary FP8/BF16
            # rounding dominate rel_l2 and obscured routing errors.
            value = (
                0.015 + ((token * 7 + hidden * 3) % 17).astype(jnp.float32) * 0.00025
            )
            value += (token % 5).astype(jnp.float32) * 0.001
            return value.astype(dtype)

        if kind in ("w1", "w3", "w2"):
            base = {"w1": 0.25, "w3": 0.125, "w2": 0.375}
            expert_period = {"w1": 7, "w3": 5, "w2": 3}
            offsets = {"w1": 1, "w3": 4, "w2": 8}
            expert = jnp.arange(shape[0], dtype=jnp.int32)[:, None, None]
            reduction = jnp.arange(shape[1], dtype=jnp.int32)[None, :, None]
            output = jnp.arange(shape[2], dtype=jnp.int32)[None, None, :]
            expert_term = (expert % expert_period[kind]).astype(jnp.float32) * 0.125
            channel_term = (
                (
                    reduction * (5 + offsets[kind])
                    + output * (7 + offsets[kind])
                    + offsets[kind]
                )
                % 3
            ).astype(jnp.float32) * 0.125
            return (base[kind] + expert_term + channel_term).astype(dtype)

        if kind in ("w1_scale", "w3_scale", "w2_scale"):
            offsets = {"w1_scale": 2, "w3_scale": 5, "w2_scale": 9}
            expert = jnp.arange(shape[0], dtype=jnp.int32)[:, None, None, None]
            output = jnp.arange(shape[3], dtype=jnp.int32)[None, None, None, :]
            value = (
                0.0015
                + (expert % 5).astype(jnp.float32) * 0.0002
                + ((expert * 3 + output * 5 + offsets[kind]) % 11).astype(jnp.float32)
                * 0.00005
            )
            return jnp.broadcast_to(value, shape).astype(dtype)

        if kind in ("w1_shared", "w3_shared", "w2_shared"):
            offsets = {"w1_shared": 2, "w3_shared": 6, "w2_shared": 10}
            reduction = jnp.arange(shape[0], dtype=jnp.int32)[:, None]
            output = jnp.arange(shape[1], dtype=jnp.int32)[None, :]
            value = (
                0.25
                + (
                    reduction * (3 + offsets[kind])
                    + output * (5 + offsets[kind])
                    + offsets[kind]
                )
                % 4
                * 0.125
            )
            return value.astype(dtype)

        if kind in ("w1_shared_scale", "w3_shared_scale", "w2_shared_scale"):
            offsets = {
                "w1_shared_scale": 1,
                "w3_shared_scale": 4,
                "w2_shared_scale": 8,
            }
            output = jnp.arange(shape[-1], dtype=jnp.int32)[None, None, :]
            value = (
                0.0015
                + ((output * 7 + offsets[kind]) % 11).astype(jnp.float32) * 0.00005
            )
            return jnp.broadcast_to(value, shape).astype(dtype)

        raise ValueError(f"Unsupported patterned array kind={kind!r}")

    return jax.jit(make_value, out_shardings=sharding)()


def _make_inputs(
    mesh,
    num_tokens: int,
    ep_size: int,
    *,
    routing_seed: int,
    layer_scope: bool,
    input_profile: str = "uniform",
):
    if num_tokens % ep_size:
        raise ValueError(
            f"num_tokens={num_tokens} must be divisible by ep_size={ep_size}"
        )

    expert_axis = get_moe_expert_axis(mesh)
    token_sharding = NamedSharding(mesh, P(expert_axis, None))
    weight_sharding = NamedSharding(mesh, P(expert_axis, None, None))
    scale_sharding = NamedSharding(mesh, P(expert_axis, None, None, None))

    if input_profile == "uniform":
        tokens = _make_array(
            (num_tokens, GLM52_HIDDEN_SIZE), jnp.bfloat16, token_sharding, 0.01
        )
        w1 = _make_array(
            (GLM52_NUM_EXPERTS, GLM52_HIDDEN_SIZE, GLM52_INTERMEDIATE_SIZE),
            jnp.float8_e4m3fn,
            weight_sharding,
            1.0,
        )
        w3 = _make_array(
            (GLM52_NUM_EXPERTS, GLM52_HIDDEN_SIZE, GLM52_INTERMEDIATE_SIZE),
            jnp.float8_e4m3fn,
            weight_sharding,
            1.0,
        )
        w2 = _make_array(
            (GLM52_NUM_EXPERTS, GLM52_INTERMEDIATE_SIZE, GLM52_HIDDEN_SIZE),
            jnp.float8_e4m3fn,
            weight_sharding,
            1.0,
        )
        w1_scale = _make_array(
            (GLM52_NUM_EXPERTS, 1, 1, GLM52_INTERMEDIATE_SIZE),
            jnp.float32,
            scale_sharding,
            0.01,
        )
        w3_scale = _make_array(w1_scale.shape, jnp.float32, scale_sharding, 0.01)
        w2_scale = _make_array(
            (GLM52_NUM_EXPERTS, 1, 1, GLM52_HIDDEN_SIZE),
            jnp.float32,
            scale_sharding,
            0.01,
        )
    elif input_profile == "expert_distinct":
        tokens = _make_patterned_array(
            (num_tokens, GLM52_HIDDEN_SIZE),
            jnp.bfloat16,
            token_sharding,
            kind="tokens",
        )
        w1 = _make_patterned_array(
            (GLM52_NUM_EXPERTS, GLM52_HIDDEN_SIZE, GLM52_INTERMEDIATE_SIZE),
            jnp.float8_e4m3fn,
            weight_sharding,
            kind="w1",
        )
        w3 = _make_patterned_array(
            (GLM52_NUM_EXPERTS, GLM52_HIDDEN_SIZE, GLM52_INTERMEDIATE_SIZE),
            jnp.float8_e4m3fn,
            weight_sharding,
            kind="w3",
        )
        w2 = _make_patterned_array(
            (GLM52_NUM_EXPERTS, GLM52_INTERMEDIATE_SIZE, GLM52_HIDDEN_SIZE),
            jnp.float8_e4m3fn,
            weight_sharding,
            kind="w2",
        )
        w1_scale = _make_patterned_array(
            (GLM52_NUM_EXPERTS, 1, 1, GLM52_INTERMEDIATE_SIZE),
            jnp.float32,
            scale_sharding,
            kind="w1_scale",
        )
        w3_scale = _make_patterned_array(
            w1_scale.shape,
            jnp.float32,
            scale_sharding,
            kind="w3_scale",
        )
        w2_scale = _make_patterned_array(
            (GLM52_NUM_EXPERTS, 1, 1, GLM52_HIDDEN_SIZE),
            jnp.float32,
            scale_sharding,
            kind="w2_scale",
        )
    else:
        raise ValueError(f"Unsupported input_profile={input_profile!r}")

    if layer_scope:
        replicated = NamedSharding(mesh, P())
        if input_profile == "uniform":
            w1_shared = _make_array(
                (GLM52_HIDDEN_SIZE, GLM52_INTERMEDIATE_SIZE),
                jnp.float8_e4m3fn,
                replicated,
                1.0,
            )
            w3_shared = _make_array(
                (GLM52_HIDDEN_SIZE, GLM52_INTERMEDIATE_SIZE),
                jnp.float8_e4m3fn,
                replicated,
                1.0,
            )
            w2_shared = _make_array(
                (GLM52_INTERMEDIATE_SIZE, GLM52_HIDDEN_SIZE),
                jnp.float8_e4m3fn,
                replicated,
                1.0,
            )
            w1_shared_scale = _make_array(
                (1, 1, GLM52_INTERMEDIATE_SIZE), jnp.float32, replicated, 0.01
            )
            w3_shared_scale = _make_array(
                (1, 1, GLM52_INTERMEDIATE_SIZE), jnp.float32, replicated, 0.01
            )
            w2_shared_scale = _make_array(
                (1, 1, GLM52_HIDDEN_SIZE), jnp.float32, replicated, 0.01
            )
        else:
            w1_shared = _make_patterned_array(
                (GLM52_HIDDEN_SIZE, GLM52_INTERMEDIATE_SIZE),
                jnp.float8_e4m3fn,
                replicated,
                kind="w1_shared",
            )
            w3_shared = _make_patterned_array(
                (GLM52_HIDDEN_SIZE, GLM52_INTERMEDIATE_SIZE),
                jnp.float8_e4m3fn,
                replicated,
                kind="w3_shared",
            )
            w2_shared = _make_patterned_array(
                (GLM52_INTERMEDIATE_SIZE, GLM52_HIDDEN_SIZE),
                jnp.float8_e4m3fn,
                replicated,
                kind="w2_shared",
            )
            w1_shared_scale = _make_patterned_array(
                (1, 1, GLM52_INTERMEDIATE_SIZE),
                jnp.float32,
                replicated,
                kind="w1_shared_scale",
            )
            w3_shared_scale = _make_patterned_array(
                (1, 1, GLM52_INTERMEDIATE_SIZE),
                jnp.float32,
                replicated,
                kind="w3_shared_scale",
            )
            w2_shared_scale = _make_patterned_array(
                (1, 1, GLM52_HIDDEN_SIZE),
                jnp.float32,
                replicated,
                kind="w2_shared_scale",
            )
    else:
        w1_shared = w3_shared = w2_shared = None
        w1_shared_scale = w3_shared_scale = w2_shared_scale = None

    # Construct global arrays through a sharded computation.  A host-side
    # ``device_put`` cannot address the non-local devices in the 16-chip,
    # four-process v7x-32 run.
    def make_routing():
        # Match fused-v2's seeded synthetic route: Gaussian router logits,
        # unique TopK expert ids, then softmax over the selected logits.
        gating = jax.random.normal(
            jax.random.key(routing_seed),
            (num_tokens, GLM52_NUM_EXPERTS),
            dtype=jnp.float32,
        )
        topk_logits, topk_ids = jax.lax.top_k(gating, GLM52_TOP_K)
        topk_weights = jax.nn.softmax(topk_logits, axis=-1)
        return topk_ids.astype(jnp.int32), topk_weights.astype(jnp.float32)

    topk_ids, topk_weights = jax.jit(
        make_routing,
        out_shardings=(token_sharding, token_sharding),
    )()

    return (
        tokens,
        w1,
        w2,
        w3,
        w1_scale,
        w2_scale,
        w3_scale,
        topk_weights,
        topk_ids,
        w1_shared,
        w3_shared,
        w2_shared,
        w1_shared_scale,
        w3_shared_scale,
        w2_shared_scale,
    )


def _make_padded_inputs(
    inputs, *, num_tokens: int, ep_size: int, active_per_device: int
):
    """Mask every device-local token shard to an active prefix.

    The physical shape stays at 64K so the same compiled executable is reused.
    Invalid rows use the production ``topk_id=-1`` sentinel and zero routing
    weights.  This catches padding-sensitive routing/scatter failures without
    changing the tuning workload's all-active performance shape.
    """
    local_tokens = num_tokens // ep_size
    if not 0 < active_per_device <= local_tokens:
        raise ValueError(
            "active_per_device must be in [1, num_tokens / ep_size]; "
            f"got active_per_device={active_per_device}, local_tokens={local_tokens}"
        )

    route_sharding = inputs[8].sharding
    valid_sharding = NamedSharding(route_sharding.mesh, P(route_sharding.spec[0]))

    def mask_routes(topk_weights, topk_ids):
        valid = (
            jnp.arange(num_tokens, dtype=jnp.int32) % local_tokens
        ) < active_per_device
        return (
            jnp.where(valid[:, None], topk_weights, 0.0),
            jnp.where(valid[:, None], topk_ids, -1),
            valid,
        )

    topk_weights, topk_ids, valid_mask = jax.jit(
        mask_routes,
        out_shardings=(route_sharding, route_sharding, valid_sharding),
    )(inputs[7], inputs[8])
    padded_inputs = (*inputs[:7], topk_weights, topk_ids, *inputs[9:])
    return padded_inputs, valid_mask


def _comparison_metrics(
    reference, candidate, *, valid_mask=None
) -> dict[str, float | bool]:
    reference_f32 = reference.astype(jnp.float32)
    candidate_f32 = candidate.astype(jnp.float32)
    if valid_mask is not None:
        mask = valid_mask[:, None].astype(jnp.float32)
        reference_f32 = reference_f32 * mask
        candidate_f32 = candidate_f32 * mask

    diff = candidate_f32 - reference_f32
    scalar_sharding = NamedSharding(reference.sharding.mesh, P())
    contracting_dims = tuple(range(reference_f32.ndim))

    def global_dot(lhs, rhs):
        # Under the explicit data1 x tensor32 mesh, both array dimensions are
        # sharded.  norm/vdot leave the scalar result sharding ambiguous, so
        # state the replicated scalar contract directly for the correctness
        # reduction instead of relying on implicit propagation.
        return jax.lax.dot_general(
            lhs,
            rhs,
            ((contracting_dims, contracting_dims), ((), ())),
            preferred_element_type=jnp.float32,
            out_sharding=scalar_sharding,
        )

    reference_norm = jnp.sqrt(
        jnp.maximum(global_dot(reference_f32, reference_f32), 0.0)
    )
    candidate_norm = jnp.sqrt(
        jnp.maximum(global_dot(candidate_f32, candidate_f32), 0.0)
    )
    diff_norm = jnp.sqrt(jnp.maximum(global_dot(diff, diff), 0.0))
    rel_l2 = diff_norm / jnp.maximum(reference_norm, 1e-12)
    cosine = global_dot(reference_f32, candidate_f32) / jnp.maximum(
        reference_norm * candidate_norm, 1e-12
    )
    values = jax.device_get(
        (
            rel_l2,
            jnp.max(jnp.abs(diff)),
            cosine,
            jnp.all(jnp.isfinite(candidate_f32)),
        )
    )
    return {
        "rel_l2": float(values[0]),
        "max_abs": float(values[1]),
        "cosine": float(values[2]),
        "all_finite": bool(values[3]),
    }


def _invalid_padding_max_abs(output, valid_mask) -> float:
    invalid = (~valid_mask)[:, None].astype(jnp.float32)
    return float(jax.device_get(jnp.max(jnp.abs(output.astype(jnp.float32)) * invalid)))


def _v2_runner(mesh, num_tokens: int, *, layer_scope: bool) -> Callable:
    try:
        bt, bf, btc, bse, bts = GLM52_V2_BLOCK_CONFIGS[num_tokens]
    except KeyError as exc:
        raise ValueError(
            f"No GLM-5.2 fused_v2 block config for tokens={num_tokens}"
        ) from exc
    block_config = FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=bse, bts=bts)

    def run(inputs):
        (
            tokens,
            w1,
            w2,
            w3,
            w1_scale,
            w2_scale,
            w3_scale,
            topk_weights,
            topk_ids,
            w1_shared,
            w3_shared,
            w2_shared,
            w1_shared_scale,
            w3_shared_scale,
            w2_shared_scale,
        ) = inputs
        output = fused_ep_moe_v2(
            mesh,
            tokens,
            w1,
            w2,
            w3,
            topk_weights,
            topk_ids,
            GLM52_TOP_K,
            act_fn="silu",
            quant_block_k=None,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            block_config=block_config,
            direct_scaled_dot=True,
            enable_act_quant=True,
            w1_shared=w1_shared if layer_scope else None,
            w3_shared=w3_shared if layer_scope else None,
            w2_shared=w2_shared if layer_scope else None,
            w1_shared_scale=(
                w1_shared_scale[:, 0, :]
                if layer_scope and w1_shared_scale is not None
                else None
            ),
            w3_shared_scale=(
                w3_shared_scale[:, 0, :]
                if layer_scope and w3_shared_scale is not None
                else None
            ),
            w2_shared_scale=(
                w2_shared_scale[:, 0, :]
                if layer_scope and w2_shared_scale is not None
                else None
            ),
        )
        if layer_scope:
            output = jax.sharding.reshard(output, NamedSharding(mesh, P("data", None)))
        return output

    # Match the serving layer boundary: keep shared-expert/add/reshard in the
    # same compiled graph instead of measuring a Python sequence of dispatches.
    return jax.jit(run)


def _rs_runner(
    mesh,
    *,
    layer_scope: bool,
    hidden_all_gather_backend: str = "auto",
    fp8_hidden_all_gather: bool = False,
    _fp8_hidden_direct_prequantized: bool = False,
    _fp8_hidden_scale_multiplier: float = 1.0,
) -> Callable:
    compiler_options = (
        {
            "xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes": str(
                1 << 30
            )
        }
        if hidden_all_gather_backend == "tensorcore"
        else None
    )

    def run(inputs):
        (
            tokens,
            w1,
            w2,
            w3,
            w1_scale,
            w2_scale,
            w3_scale,
            topk_weights,
            topk_ids,
            w1_shared,
            w3_shared,
            w2_shared,
            w1_shared_scale,
            w3_shared_scale,
            w2_shared_scale,
        ) = inputs
        output = _fused_moe_func_rs_impl(
            hidden_states=tokens,
            w1=w1,
            w3=w3,
            w2=w2,
            w1_scale=w1_scale,
            w3_scale=w3_scale,
            w2_scale=w2_scale,
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
            fp8_hidden_all_gather=fp8_hidden_all_gather,
            _fp8_hidden_direct_prequantized=_fp8_hidden_direct_prequantized,
            _fp8_hidden_scale_multiplier=_fp8_hidden_scale_multiplier,
        )
        if layer_scope:
            shared_output = fused_rs_shared_expert(
                tokens,
                w1_shared,
                w3_shared,
                w2_shared,
                w1_scale=w1_shared_scale,
                w3_scale=w3_shared_scale,
                w2_scale=w2_shared_scale,
                activation_quantized_dtype=jnp.float8_e4m3fn,
                mesh=mesh,
            )
            output = (
                output.astype(jnp.float32) + shared_output.astype(jnp.float32)
            ).astype(tokens.dtype)
            output = jax.sharding.reshard(output, NamedSharding(mesh, P("data", None)))
        return output

    # ``compiler_options`` are legal only on the outermost JIT. Calling the
    # pre-jitted tensorcore variant from this jitted runner makes JAX 0.9 reject
    # the nested option before lowering. Keep one top-level compilation while
    # preserving the exact same fused-RS implementation and option value.
    if compiler_options is None:
        return jax.jit(run)
    return jax.jit(run, compiler_options=compiler_options)


def _hidden_all_gather_probe_runner(
    mesh,
    *,
    hidden_all_gather_backend: str,
) -> Callable:
    """Materialize the fused-RS input AllGather for a semantics-only check.

    This isolated executable is not a performance measurement.  It reproduces
    the shard-map collective boundary so the A/B can distinguish an incorrect
    collective payload from downstream low-precision layout/reduction drift.
    """
    expert_axis = get_moe_expert_axis(mesh)
    compiler_options = (
        {
            "xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes": str(
                1 << 30
            )
        }
        if hidden_all_gather_backend == "tensorcore"
        else None
    )

    def run(hidden_states):
        def gather_local(hidden_local):
            with jax.named_scope("fused_rs_hidden_all_gather_probe"):
                return jax.lax.all_gather(
                    hidden_local,
                    axis_name=expert_axis,
                    axis=0,
                    tiled=True,
                )

        return jax.shard_map(
            gather_local,
            mesh=mesh,
            in_specs=P(expert_axis, None),
            out_specs=P(),
            check_vma=False,
        )(hidden_states)

    if compiler_options is None:
        return jax.jit(run)
    return jax.jit(run, compiler_options=compiler_options)


def _routing_stats(topk_ids) -> dict[str, float | int]:
    """Count the routed rows exactly once across all four EP32 processes."""
    local_counts = np.zeros(GLM52_NUM_EXPERTS, dtype=np.int64)
    for shard in topk_ids.addressable_shards:
        local_counts += np.bincount(
            np.asarray(shard.data).reshape(-1),
            minlength=GLM52_NUM_EXPERTS,
        )
    gathered = np.asarray(
        multihost_utils.process_allgather(local_counts.astype(np.int32))
    )
    counts = gathered.reshape(-1, GLM52_NUM_EXPERTS).astype(np.int64).sum(axis=0)
    return {
        "routing_observed_rows": int(counts.sum()),
        "routing_expert_rows_min": int(counts.min()),
        "routing_expert_rows_max": int(counts.max()),
        "routing_expert_rows_mean": float(counts.mean()),
        "routing_expert_rows_std": float(counts.std()),
    }


def _measure(run, inputs, *, task: str, warmup: int, iters: int, trace_root: str):
    return multiple_iteration_timeit_from_trace(
        lambda current_inputs: run(current_inputs),
        lambda: (inputs,),
        task=task,
        tries=iters,
        warmup=warmup,
        trace_root=trace_root,
    )


def _measure_rs_breakdown(
    run,
    inputs,
    *,
    task: str,
    warmup: int,
    iters: int,
    trace_root: str,
):
    return multiple_iteration_profile_from_trace(
        lambda current_inputs: run(current_inputs),
        lambda: (inputs,),
        task=task,
        stage_scopes={
            "hidden_quantize": ("fused_rs_hidden_quantize", None),
            "hidden_all_gather": ("fused_rs_hidden_all_gather", "all-gather"),
            "hidden_scale_all_gather": (
                "fused_rs_hidden_scale_all_gather",
                "all-gather",
            ),
            "hidden_scale_expand": ("fused_rs_hidden_scale_expand", None),
            "hidden_dequantize": ("fused_rs_hidden_dequantize", None),
            "topk_ids_all_gather": (
                "fused_rs_topk_ids_all_gather",
                "all-gather",
            ),
            "routing_table_materialization": (
                "fused-rs-routing-table-M_|gather_offload_custom_fusion",
                None,
            ),
        },
        tries=iters,
        warmup=warmup,
        trace_root=trace_root,
    )


def main() -> None:
    # Falcon's multi-host TPU environment supplies the coordinator metadata.
    # Calling initialize before the first backend query makes all 32 devices
    # visible to all four processes; this is also the convention used by the
    # existing fused-v2 benchmark entrypoints.
    jax.distributed.initialize()

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default=",".join(map(str, DEFAULT_TOKENS)))
    parser.add_argument("--ep-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--routing-seed", type=int, default=42)
    parser.add_argument("--trace-root", default="/tmp/sglang_jax_fused_rs_trace")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument(
        "--append-jsonl",
        action="store_true",
        help="Append to --jsonl instead of truncating it before this scope.",
    )
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument(
        "--layer-scope",
        action="store_true",
        help="Include GLM shared expert and final P('data', None) output reshard.",
    )
    parser.add_argument(
        "--profile-breakdown",
        action="store_true",
        help=(
            "From the same real fused-RS trace, record the existing call span, "
            "the Pallas task event, and the separately scoped BF16 hidden/topk-id "
            "AllGather device durations. This does not alter collective semantics."
        ),
    )
    parser.add_argument(
        "--hidden-all-gather-backend",
        choices=("auto", "tensorcore"),
        default="auto",
        help=(
            "Use XLA's default Hidden AllGather placement, or keep sub-1 GiB "
            "AllGathers on TensorCore to expose SparseCore routing overlap."
        ),
    )
    parser.add_argument(
        "--routing-table-impl",
        choices=("jax", "pallas"),
        default="jax",
        help="Materialize the high-M packed routing table with JAX or Pallas.",
    )
    parser.add_argument(
        "--rs-configs",
        help=(
            "Semicolon-separated fused_rs candidates. Each candidate is "
            "tile_m,tile_k1,tile_n1,tile_k2,tile_n2,num_w1_bufs,num_w2_bufs; "
            "use 'default' for calculate_tiling. With --tune-rs, this narrows "
            "the runtime-contract-gated tuning sweep to explicit configs."
        ),
    )
    parser.add_argument(
        "--tune-rs",
        action="store_true",
        help=(
            "Generate the contract-valid GLM-5.2 full-K candidate matrix, record "
            "candidate-vs-canonical diagnostics, gate non-finite or padding-sensitive "
            "runs, then rank measurable candidates by kernel time."
        ),
    )
    parser.add_argument(
        "--tune-tile-ms",
        default=",".join(map(str, DEFAULT_TUNE_TILE_MS)),
        help="Comma-separated tile_m values used by --tune-rs.",
    )
    parser.add_argument(
        "--input-profile",
        choices=("uniform", "expert_distinct"),
        help=(
            "Synthetic tensor pattern. --tune-rs defaults to expert_distinct; "
            "regular comparisons retain the legacy uniform default."
        ),
    )
    parser.add_argument(
        "--padding-active-tokens-per-device",
        type=int,
        default=64,
        help=(
            "With --tune-rs, keep this many active tokens at the start of every "
            "device-local shard for the same-backend padding fidelity gate."
        ),
    )
    parser.add_argument(
        "--correctness-rel-l2-threshold",
        type=float,
        default=0.01,
        help=(
            "Maximum same-candidate all-active-vs-padded rel_l2. "
            "Candidate-vs-canonical-RS rel_l2 is recorded for diagnosis only; "
            "kernel correctness is covered by the independent-oracle RS test."
        ),
    )
    parser.add_argument(
        "--rs-only",
        action="store_true",
        help="Skip fused_v2 timing. Intended for multi-candidate tuning sweeps.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Emit an error row and continue when a candidate cannot compile or run.",
    )
    args = parser.parse_args()

    if args.tune_rs and args.layer_scope:
        raise ValueError(
            "Tune the routed RS kernel first; --tune-rs does not use --layer-scope"
        )
    if args.correctness_rel_l2_threshold <= 0:
        raise ValueError("--correctness-rel-l2-threshold must be positive")
    set_fused_rs_routing_table_impl(args.routing_table_impl)

    if args.ep_size != 32:
        raise ValueError("This GLM-5.2 comparison is intentionally fixed to ep_size=32")
    expected_devices = args.ep_size
    if len(jax.devices()) != expected_devices:
        raise ValueError(
            f"Expected exactly {expected_devices} devices for EP{args.ep_size}, "
            f"found {len(jax.devices())}"
        )

    mesh = _build_mesh(args.ep_size)
    rs_configs = _select_rs_configs(
        tune_rs=args.tune_rs,
        rs_configs=args.rs_configs,
        tune_tile_ms=args.tune_tile_ms,
    )
    if args.tune_rs:
        args.rs_only = True
        args.continue_on_error = True
        input_profile = args.input_profile or "expert_distinct"
    else:
        input_profile = args.input_profile or "uniform"
    if args.jsonl is not None and not args.append_jsonl and jax.process_index() == 0:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("", encoding="utf-8")

    def emit_row(row: dict) -> None:
        encoded = json.dumps(row, sort_keys=True)
        if jax.process_index() == 0:
            print(encoded, flush=True)
        if args.jsonl is not None and jax.process_index() == 0:
            with args.jsonl.open("a", encoding="utf-8") as output_file:
                output_file.write(encoded + "\n")

    with jax.set_mesh(mesh):
        for num_tokens in _parse_csv_ints(args.tokens):
            inputs = _make_inputs(
                mesh,
                num_tokens,
                args.ep_size,
                routing_seed=args.routing_seed,
                layer_scope=args.layer_scope,
                input_profile=input_profile,
            )
            routing_stats = _routing_stats(inputs[8])
            tuning_rows: list[dict] = []

            reference_run = None
            reference_out = None
            reference_compile_time_s = None
            padded_inputs = None
            valid_mask = None
            if args.tune_rs:
                padded_inputs, valid_mask = _make_padded_inputs(
                    inputs,
                    num_tokens=num_tokens,
                    ep_size=args.ep_size,
                    active_per_device=args.padding_active_tokens_per_device,
                )
                set_fused_rs_block_sizes_override(GLM52_RS_REFERENCE_CONFIG)
                jax.clear_caches()
                reference_run = _rs_runner(
                    mesh,
                    layer_scope=False,
                    hidden_all_gather_backend=args.hidden_all_gather_backend,
                )
                compile_start = time.perf_counter()
                reference_out = reference_run(inputs)
                jax.block_until_ready(reference_out)
                reference_compile_time_s = time.perf_counter() - compile_start

            v2_out = None
            v2_ms = None
            v2_kernel_samples = None
            if not args.tune_rs:
                v2_run = _v2_runner(mesh, num_tokens, layer_scope=args.layer_scope)
                if not args.no_check:
                    v2_out = v2_run(inputs)
                    jax.block_until_ready(v2_out)
                if not args.rs_only:
                    samples = _measure(
                        v2_run,
                        inputs,
                        task=r"fused-moe-v2-k_.*",
                        warmup=args.warmup,
                        iters=args.iters,
                        trace_root=str(
                            Path(args.trace_root) / str(num_tokens) / "fused_v2"
                        ),
                    )
                    v2_ms = statistics.median(samples)
                    v2_kernel_samples = samples

            for config in rs_configs:
                config_label = _config_label(config)
                contract = analyze_rs_config(config) if config is not None else {}
                row_base = {
                    "record_type": "candidate",
                    "model": "glm-5.2",
                    "measurement_scope": (
                        "glm52_moe_layer" if args.layer_scope else "routed_backend"
                    ),
                    "routing_is_precomputed": True,
                    "hidden_all_gather_backend": args.hidden_all_gather_backend,
                    "routing_table_impl": args.routing_table_impl,
                    "includes_gate_topk": False,
                    "includes_shared_expert": args.layer_scope,
                    "includes_output_reshard": args.layer_scope,
                    "process_count": jax.process_count(),
                    "ep_size": args.ep_size,
                    "num_tokens": num_tokens,
                    "routed_rows": num_tokens * GLM52_TOP_K,
                    "num_experts": GLM52_NUM_EXPERTS,
                    "top_k": GLM52_TOP_K,
                    "hidden_size": GLM52_HIDDEN_SIZE,
                    "intermediate_size": GLM52_INTERMEDIATE_SIZE,
                    "hidden_all_gather_local_payload_bytes": (
                        num_tokens
                        // args.ep_size
                        * GLM52_HIDDEN_SIZE
                        * jnp.dtype(jnp.bfloat16).itemsize
                    ),
                    "hidden_all_gather_logical_output_bytes_per_device": (
                        num_tokens
                        * GLM52_HIDDEN_SIZE
                        * jnp.dtype(jnp.bfloat16).itemsize
                    ),
                    "topk_ids_all_gather_local_payload_bytes": (
                        num_tokens
                        // args.ep_size
                        * GLM52_TOP_K
                        * jnp.dtype(jnp.int32).itemsize
                    ),
                    "quant_mode": "per_channel",
                    "quant_block_k": GLM52_QUANT_BLOCK_K,
                    "routing_mode": "seeded_random_gaussian_topk",
                    "routing_seed": args.routing_seed,
                    "input_profile": input_profile,
                    **routing_stats,
                    "rs_block_config": config_label,
                    "rs_reference_config": (
                        list(GLM52_RS_REFERENCE_CONFIG) if args.tune_rs else None
                    ),
                    "padding_fidelity_rel_l2_threshold": (
                        args.correctness_rel_l2_threshold if args.tune_rs else None
                    ),
                    "canonical_rs_comparison_is_gate": False if args.tune_rs else None,
                    "independent_oracle_test": (
                        "python/sgl_jax/test/kernels/fused_moe_rs_test.py"
                        if args.tune_rs
                        else None
                    ),
                    **contract,
                }

                if contract and not contract["eligible_for_tuning"]:
                    row = {
                        **row_base,
                        "status": "rejected",
                        "validation_stage": "tuning_contract",
                        "error_type": "InvalidTuningContract",
                        "error": (
                            "config must keep K whole, keep both weight stages "
                            "padding-safe and resident, use aligned tiles, and fit "
                            "the declared VMEM budget"
                        ),
                    }
                    emit_row(row)
                    tuning_rows.append(row)
                    continue

                set_fused_rs_block_sizes_override(config)
                if args.tune_rs and config == GLM52_RS_REFERENCE_CONFIG:
                    rs_run = reference_run
                    rs_out = reference_out
                    compile_time_s = reference_compile_time_s
                else:
                    # The override is consumed during tracing. Clear cached traces so
                    # every candidate produces an executable with its own tile shapes.
                    jax.clear_caches()
                    rs_run = _rs_runner(
                        mesh,
                        layer_scope=args.layer_scope,
                        hidden_all_gather_backend=args.hidden_all_gather_backend,
                    )
                    rs_out = None
                    compile_time_s = None

                validation_stage = "compile"
                compile_status = (
                    "ok"
                    if rs_out is not None
                    else ("pending" if args.tune_rs else None)
                )
                try:
                    rel_l2_vs_v2 = None
                    rs_reference_metrics = None
                    padding_fidelity_metrics = None
                    padding_invalid_max_abs = None
                    if args.tune_rs:
                        if rs_out is None:
                            compile_start = time.perf_counter()
                            rs_out = rs_run(inputs)
                            jax.block_until_ready(rs_out)
                            compile_time_s = time.perf_counter() - compile_start
                            compile_status = "ok"
                        validation_stage = "candidate_finiteness"
                        rs_reference_metrics = _comparison_metrics(
                            reference_out, rs_out
                        )
                        if not rs_reference_metrics["all_finite"]:
                            raise AssertionError(
                                "candidate produced non-finite all-active output"
                            )

                        validation_stage = "padding_fidelity"
                        padded_out = rs_run(padded_inputs)
                        jax.block_until_ready(padded_out)
                        padding_fidelity_metrics = _comparison_metrics(
                            rs_out,
                            padded_out,
                            valid_mask=valid_mask,
                        )
                        padding_invalid_max_abs = _invalid_padding_max_abs(
                            padded_out,
                            valid_mask,
                        )
                        if not padding_fidelity_metrics["all_finite"]:
                            raise AssertionError(
                                "candidate produced non-finite padded output"
                            )
                        if (
                            padding_fidelity_metrics["rel_l2"]
                            > args.correctness_rel_l2_threshold
                        ):
                            raise AssertionError(
                                "candidate is padding-sensitive: "
                                f"rel_l2={padding_fidelity_metrics['rel_l2']}"
                            )
                        if padding_invalid_max_abs != 0.0:
                            raise AssertionError(
                                "candidate wrote non-zero invalid padding rows: "
                                f"max_abs={padding_invalid_max_abs}"
                            )
                    elif v2_out is not None:
                        validation_stage = "informational_v2_comparison"
                        compile_start = time.perf_counter()
                        rs_out = rs_run(inputs)
                        jax.block_until_ready(rs_out)
                        compile_time_s = time.perf_counter() - compile_start
                        compile_status = "ok"
                        v2_metrics = _comparison_metrics(v2_out, rs_out)
                        rel_l2_vs_v2 = v2_metrics["rel_l2"]
                        if rel_l2_vs_v2 > 0.2:
                            raise AssertionError(
                                "fused_rs differs from fused_v2 at "
                                f"tokens={num_tokens}, config={config_label}: "
                                f"rel_l2={rel_l2_vs_v2}"
                            )

                    validation_stage = "measurement"
                    rs_trace_root = str(
                        Path(args.trace_root)
                        / str(num_tokens)
                        / f"fused_rs_{config_label}"
                    )
                    breakdown = None
                    if args.profile_breakdown:
                        breakdown = _measure_rs_breakdown(
                            rs_run,
                            inputs,
                            task=r"gmm_v2_fused_rs.*",
                            warmup=args.warmup,
                            iters=args.iters,
                            trace_root=rs_trace_root,
                        )
                        call_samples = breakdown["call_samples_ms"]
                        pallas_samples = breakdown["task_samples_ms"]
                        # Preserve the legacy output field without hiding a
                        # missing call marker.  The explicit call/Pallas fields
                        # below remain authoritative for the breakdown.
                        samples = call_samples or pallas_samples
                        legacy_sample_source = (
                            "call_marker" if call_samples else "pallas_task_fallback"
                        )
                    else:
                        samples = _measure(
                            rs_run,
                            inputs,
                            task=r"gmm_v2_fused_rs.*",
                            warmup=args.warmup,
                            iters=args.iters,
                            trace_root=rs_trace_root,
                        )
                        call_samples = None
                        pallas_samples = None
                        legacy_sample_source = "call_marker_or_task_fallback"
                    rs_ms = statistics.median(samples)
                    hidden_all_gather_samples = (
                        breakdown["stage_samples_ms"]["hidden_all_gather"]
                        if breakdown
                        else None
                    )
                    topk_ids_all_gather_samples = (
                        breakdown["stage_samples_ms"]["topk_ids_all_gather"]
                        if breakdown
                        else None
                    )
                    breakdown_complete = (
                        breakdown is not None
                        and all(
                            len(stage_samples) == args.iters
                            for stage_samples in (
                                call_samples,
                                pallas_samples,
                                hidden_all_gather_samples,
                                topk_ids_all_gather_samples,
                            )
                        )
                    )
                    effective_config = get_last_fused_rs_block_sizes()
                    row = {
                        **row_base,
                        "status": "ok",
                        "validation_stage": "measured",
                        "compile_status": compile_status,
                        "correctness_status": "passed" if args.tune_rs else None,
                        "canonical_rs_comparison_status": (
                            "diagnostic_only" if args.tune_rs else None
                        ),
                        "eligible_for_tuning": True,
                        "effective_rs_block_config": (
                            list(effective_config)
                            if effective_config is not None
                            else None
                        ),
                        "compile_time_s": compile_time_s,
                        "fused_v2_kernel_ms": v2_ms,
                        "fused_rs_kernel_ms": rs_ms,
                        "fused_v2_kernel_samples_ms": v2_kernel_samples,
                        "fused_rs_kernel_samples_ms": samples,
                        "fused_rs_legacy_kernel_field_source": legacy_sample_source,
                        "fused_rs_call_ms": (
                            statistics.median(call_samples) if call_samples else None
                        ),
                        "fused_rs_call_samples_ms": call_samples,
                        "fused_rs_pallas_ms": (
                            statistics.median(pallas_samples)
                            if pallas_samples
                            else None
                        ),
                        "fused_rs_pallas_samples_ms": pallas_samples,
                        "fused_rs_hidden_all_gather_ms": (
                            statistics.median(hidden_all_gather_samples)
                            if hidden_all_gather_samples
                            else None
                        ),
                        "fused_rs_hidden_all_gather_samples_ms": hidden_all_gather_samples,
                        "fused_rs_topk_ids_all_gather_ms": (
                            statistics.median(topk_ids_all_gather_samples)
                            if topk_ids_all_gather_samples
                            else None
                        ),
                        "fused_rs_topk_ids_all_gather_samples_ms": (
                            topk_ids_all_gather_samples
                        ),
                        "profile_breakdown_requested": args.profile_breakdown,
                        "profile_breakdown_complete": breakdown_complete,
                        "profile_breakdown_trace_dir": (
                            breakdown["trace_dir"] if breakdown else None
                        ),
                        "fused_rs_speedup_vs_v2": (
                            v2_ms / rs_ms if v2_ms is not None else None
                        ),
                        "rel_l2_vs_v2": rel_l2_vs_v2,
                        "rs_reference_rel_l2": (
                            rs_reference_metrics["rel_l2"]
                            if rs_reference_metrics
                            else None
                        ),
                        "rs_reference_max_abs": (
                            rs_reference_metrics["max_abs"]
                            if rs_reference_metrics
                            else None
                        ),
                        "rs_reference_cosine": (
                            rs_reference_metrics["cosine"]
                            if rs_reference_metrics
                            else None
                        ),
                        "padding_fidelity_rel_l2": (
                            padding_fidelity_metrics["rel_l2"]
                            if padding_fidelity_metrics
                            else None
                        ),
                        "padding_fidelity_max_abs": (
                            padding_fidelity_metrics["max_abs"]
                            if padding_fidelity_metrics
                            else None
                        ),
                        "padding_fidelity_cosine": (
                            padding_fidelity_metrics["cosine"]
                            if padding_fidelity_metrics
                            else None
                        ),
                        "padding_invalid_max_abs": padding_invalid_max_abs,
                        "padding_active_tokens_per_device": (
                            args.padding_active_tokens_per_device
                            if args.tune_rs
                            else None
                        ),
                    }
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    row = {
                        **row_base,
                        "status": "error",
                        "eligible_for_tuning": False,
                        "validation_stage": validation_stage,
                        "compile_status": (
                            "error" if validation_stage == "compile" else compile_status
                        ),
                        "correctness_status": (
                            "failed"
                            if validation_stage
                            in ("candidate_finiteness", "padding_fidelity")
                            else None
                        ),
                        "compile_time_s": compile_time_s,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                emit_row(row)
                tuning_rows.append(row)

            if args.tune_rs:
                eligible = [row for row in tuning_rows if row.get("status") == "ok"]
                eligible.sort(key=lambda row: row["fused_rs_kernel_ms"])
                summary = {
                    "record_type": "tuning_summary",
                    "status": "ok" if eligible else "error",
                    "model": "glm-5.2",
                    "measurement_scope": "routed_backend",
                    "process_count": jax.process_count(),
                    "ep_size": args.ep_size,
                    "num_tokens": num_tokens,
                    "routed_rows": num_tokens * GLM52_TOP_K,
                    "input_profile": input_profile,
                    "routing_mode": "seeded_random_gaussian_topk",
                    "routing_seed": args.routing_seed,
                    "candidate_count": len(tuning_rows),
                    "eligible_candidate_count": len(eligible),
                    "rs_reference_config": list(GLM52_RS_REFERENCE_CONFIG),
                    "canonical_rs_comparison_status": "diagnostic_only",
                    "independent_oracle_test": (
                        "python/sgl_jax/test/kernels/fused_moe_rs_test.py"
                    ),
                    "winner_rs_block_config": (
                        eligible[0]["rs_block_config"] if eligible else None
                    ),
                    "winner_fused_rs_kernel_ms": (
                        eligible[0]["fused_rs_kernel_ms"] if eligible else None
                    ),
                    "ranked_rs_block_configs": [
                        {
                            "rank": rank,
                            "rs_block_config": row["rs_block_config"],
                            "kernel_ms": row["fused_rs_kernel_ms"],
                        }
                        for rank, row in enumerate(eligible, start=1)
                    ],
                }
                emit_row(summary)
                if not eligible:
                    raise RuntimeError(
                        "No fused-RS tuning candidate passed runtime contracts and timing"
                    )

    set_fused_rs_block_sizes_override(None)


if __name__ == "__main__":
    main()
