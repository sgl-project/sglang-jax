"""Compare fused_rs with fused_v2 using the GLM-5.2 routed-MoE shape.

The benchmark reports two timing scopes. ``*_kernel_ms`` is the per-iteration
maximum Pallas-device duration across every EP device (RS therefore starts after
its JAX all-gather); ``*_backend_wall_ms`` is the per-iteration maximum host wall
time across every process while blocking on the whole routed call.  The latter
includes input communication, GMM1, SiLU, GMM2, output communication, and top-k
reduction. With ``--layer-scope``, both variants also include GLM's shared expert
and the final reshard back to the caller's ``P("data", None)`` layout. Gate/top-k
remain outside because routing is held identical across variants.

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
from sgl_jax.srt.kernels.fused_moe.fused_rs import fused_moe_func_rs
from sgl_jax.srt.kernels.fused_moe.fused_rs.gmm_fused_rs_nodedup import (
    FusedRsBlockConfig,
    get_last_fused_rs_block_sizes,
    set_fused_rs_block_sizes_override,
)
from sgl_jax.srt.kernels.fused_moe.v2.kernel import (
    FusedMoEBlockConfig,
    fused_ep_moe_v2,
)
from sgl_jax.srt.layers.fused_moe import fused_rs_shared_expert

from benchmark.utils import multiple_iteration_device_timeit_from_trace

GLM52_NUM_EXPERTS = 256
GLM52_TOP_K = 8
GLM52_HIDDEN_SIZE = 6144
GLM52_INTERMEDIATE_SIZE = 2048
GLM52_QUANT_BLOCK_K = None
DEFAULT_TOKENS = (65536,)

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


def _config_label(config: FusedRsBlockConfig | None) -> str:
    return "default" if config is None else "-".join(map(str, config))


def _build_mesh(ep_size: int):
    return jax.make_mesh((1, ep_size), ("data", "tensor"))


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
            value = 0.015 + ((token * 7 + hidden * 3) % 17).astype(
                jnp.float32
            ) * 0.00025
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
            value = 0.0015 + (expert % 5).astype(jnp.float32) * 0.0002 + (
                (expert * 3 + output * 5 + offsets[kind]) % 11
            ).astype(jnp.float32) * 0.00005
            return jnp.broadcast_to(value, shape).astype(dtype)

        if kind in ("w1_shared", "w3_shared", "w2_shared"):
            offsets = {"w1_shared": 2, "w3_shared": 6, "w2_shared": 10}
            reduction = jnp.arange(shape[0], dtype=jnp.int32)[:, None]
            output = jnp.arange(shape[1], dtype=jnp.int32)[None, :]
            value = 0.25 + (
                reduction * (3 + offsets[kind])
                + output * (5 + offsets[kind])
                + offsets[kind]
            ) % 4 * 0.125
            return value.astype(dtype)

        if kind in ("w1_shared_scale", "w3_shared_scale", "w2_shared_scale"):
            offsets = {
                "w1_shared_scale": 1,
                "w3_shared_scale": 4,
                "w2_shared_scale": 8,
            }
            output = jnp.arange(shape[-1], dtype=jnp.int32)[None, None, :]
            value = 0.0015 + ((output * 7 + offsets[kind]) % 11).astype(
                jnp.float32
            ) * 0.00005
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

    expert_axis = ("data", "tensor")
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
            output = jax.sharding.reshard(
                output, NamedSharding(mesh, P("data", None))
            )
        return output

    # Match the serving layer boundary: keep shared-expert/add/reshard in the
    # same compiled graph instead of measuring a Python sequence of dispatches.
    return jax.jit(run)


def _rs_runner(mesh, *, layer_scope: bool) -> Callable:
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
        output = fused_moe_func_rs(
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
            output = jax.sharding.reshard(
                output, NamedSharding(mesh, P("data", None))
            )
        return output

    return jax.jit(run)


def _local_critical_samples(
    samples_by_pid: dict[int, list[float]], *, iters: int
) -> list[float]:
    if not samples_by_pid:
        raise RuntimeError("No strict device_duration_ps samples found")
    invalid = {
        pid: len(samples) for pid, samples in samples_by_pid.items() if len(samples) != iters
    }
    if invalid:
        raise RuntimeError(
            f"Expected {iters} device samples per PID, got {invalid}; "
            f"all counts={ {pid: len(samples) for pid, samples in samples_by_pid.items()} }"
        )
    return [max(samples[i] for samples in samples_by_pid.values()) for i in range(iters)]


def _global_critical_samples(local_samples: list[float]) -> list[float]:
    gathered = np.asarray(
        multihost_utils.process_allgather(np.asarray(local_samples, dtype=np.float32))
    )
    if gathered.ndim != 2 or gathered.shape[0] != jax.process_count():
        raise RuntimeError(
            "Unexpected process_allgather result for critical-path timing: "
            f"shape={gathered.shape}, process_count={jax.process_count()}"
        )
    return gathered.max(axis=0).tolist()


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
    samples_by_pid = multiple_iteration_device_timeit_from_trace(
        lambda current_inputs: run(current_inputs),
        lambda: (inputs,),
        task=task,
        tries=iters,
        warmup=warmup,
        trace_root=trace_root,
    )
    local_kernel_samples = _local_critical_samples(samples_by_pid, iters=iters)
    global_kernel_samples = _global_critical_samples(local_kernel_samples)
    for _ in range(warmup):
        jax.block_until_ready(run(inputs))
    local_wall_samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(run(inputs))
        local_wall_samples.append((time.perf_counter() - start) * 1e3)
    global_wall_samples = _global_critical_samples(local_wall_samples)
    return (
        global_kernel_samples,
        global_wall_samples,
        len(samples_by_pid),
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
        "--rs-configs",
        help=(
            "Semicolon-separated fused_rs candidates. Each candidate is "
            "tile_m,tile_k1,tile_n1,tile_k2,tile_n2,num_w1_bufs,num_w2_bufs; "
            "use 'default' for calculate_tiling."
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

    if args.ep_size != 32:
        raise ValueError("This GLM-5.2 comparison is intentionally fixed to ep_size=32")
    expected_devices = args.ep_size
    if len(jax.devices()) != expected_devices:
        raise ValueError(
            f"Expected exactly {expected_devices} devices for EP{args.ep_size}, "
            f"found {len(jax.devices())}"
        )

    mesh = _build_mesh(args.ep_size)
    rs_configs = _parse_rs_configs(args.rs_configs)
    if (
        args.jsonl is not None
        and not args.append_jsonl
        and jax.process_index() == 0
    ):
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text("", encoding="utf-8")

    with jax.set_mesh(mesh):
        for num_tokens in _parse_csv_ints(args.tokens):
            inputs = _make_inputs(
                mesh,
                num_tokens,
                args.ep_size,
                routing_seed=args.routing_seed,
                layer_scope=args.layer_scope,
            )
            routing_stats = _routing_stats(inputs[8])
            v2_run = _v2_runner(mesh, num_tokens, layer_scope=args.layer_scope)
            v2_out = None
            v2_ms = None
            v2_wall_ms = None
            v2_kernel_samples = None
            v2_wall_samples = None
            v2_local_device_count = None
            if not args.no_check:
                v2_out = v2_run(inputs)
                jax.block_until_ready(v2_out)
            if not args.rs_only:
                samples, wall_samples, local_device_count = _measure(
                    v2_run,
                    inputs,
                    task=r"fused-moe-v2-k_.*",
                    warmup=args.warmup,
                    iters=args.iters,
                    trace_root=str(Path(args.trace_root) / str(num_tokens) / "fused_v2"),
                )
                v2_ms = statistics.median(samples)
                v2_wall_ms = statistics.median(wall_samples)
                v2_kernel_samples = samples
                v2_wall_samples = wall_samples
                v2_local_device_count = local_device_count

            for config in rs_configs:
                config_label = _config_label(config)
                set_fused_rs_block_sizes_override(config)
                # The override is consumed during tracing. Clear cached traces so
                # every candidate produces an executable with its own tile shapes.
                jax.clear_caches()
                rs_run = _rs_runner(mesh, layer_scope=args.layer_scope)
                try:
                    rel_l2 = None
                    if v2_out is not None:
                        rs_out = rs_run(inputs)
                        jax.block_until_ready(rs_out)
                        diff = (
                            v2_out.astype(jnp.float32) - rs_out.astype(jnp.float32)
                        ).reshape(-1)
                        rel_l2 = float(
                            jnp.linalg.norm(diff)
                            / jnp.maximum(
                                jnp.linalg.norm(v2_out.astype(jnp.float32)), 1e-6
                            )
                        )
                        if rel_l2 > 0.2:
                            raise AssertionError(
                                "fused_rs differs from fused_v2 at "
                                f"tokens={num_tokens}, config={config_label}: rel_l2={rel_l2}"
                            )

                    samples, wall_samples, local_device_count = _measure(
                        rs_run,
                        inputs,
                        task=r"gmm_v2_fused_rs.*",
                        warmup=args.warmup,
                        iters=args.iters,
                        trace_root=str(
                            Path(args.trace_root)
                            / str(num_tokens)
                            / f"fused_rs_{config_label}"
                        ),
                    )
                    rs_ms = statistics.median(samples)
                    rs_wall_ms = statistics.median(wall_samples)
                    effective_config = get_last_fused_rs_block_sizes()
                    row = {
                        "status": "ok",
                        "model": "glm-5.2",
                        "measurement_scope": (
                            "glm52_moe_layer" if args.layer_scope else "routed_backend"
                        ),
                        "includes_shared_expert": args.layer_scope,
                        "includes_output_reshard": args.layer_scope,
                        "process_count": jax.process_count(),
                        "local_device_count": local_device_count,
                        "v2_local_device_count": v2_local_device_count,
                        "ep_size": args.ep_size,
                        "num_tokens": num_tokens,
                        "routed_rows": num_tokens * GLM52_TOP_K,
                        "num_experts": GLM52_NUM_EXPERTS,
                        "top_k": GLM52_TOP_K,
                        "hidden_size": GLM52_HIDDEN_SIZE,
                        "intermediate_size": GLM52_INTERMEDIATE_SIZE,
                        "quant_mode": "per_channel",
                        "quant_block_k": GLM52_QUANT_BLOCK_K,
                        "routing_mode": "random",
                        "routing_seed": args.routing_seed,
                        **routing_stats,
                        "rs_block_config": config_label,
                        "effective_rs_block_config": (
                            list(effective_config) if effective_config is not None else None
                        ),
                        "fused_v2_ms": v2_ms,
                        "fused_rs_ms": rs_ms,
                        "fused_v2_kernel_ms": v2_ms,
                        "fused_rs_kernel_ms": rs_ms,
                        "fused_v2_backend_wall_ms": v2_wall_ms,
                        "fused_rs_backend_wall_ms": rs_wall_ms,
                        "fused_v2_kernel_global_critical_samples_ms": v2_kernel_samples,
                        "fused_rs_kernel_global_critical_samples_ms": samples,
                        "fused_v2_backend_wall_global_critical_samples_ms": v2_wall_samples,
                        "fused_rs_backend_wall_global_critical_samples_ms": wall_samples,
                        "fused_rs_speedup_vs_v2": (
                            v2_ms / rs_ms if v2_ms is not None else None
                        ),
                        "fused_rs_backend_speedup_vs_v2": (
                            v2_wall_ms / rs_wall_ms
                            if v2_wall_ms is not None
                            else None
                        ),
                        "rel_l2_vs_v2": rel_l2,
                    }
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    row = {
                        "status": "error",
                        "model": "glm-5.2",
                        "measurement_scope": (
                            "glm52_moe_layer" if args.layer_scope else "routed_backend"
                        ),
                        "process_count": jax.process_count(),
                        "ep_size": args.ep_size,
                        "num_tokens": num_tokens,
                        "routed_rows": num_tokens * GLM52_TOP_K,
                        "routing_mode": "random",
                        "routing_seed": args.routing_seed,
                        **routing_stats,
                        "rs_block_config": config_label,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                encoded = json.dumps(row, sort_keys=True)
                if jax.process_index() == 0:
                    print(encoded, flush=True)
                if args.jsonl is not None and jax.process_index() == 0:
                    with args.jsonl.open("a", encoding="utf-8") as output_file:
                        output_file.write(encoded + "\n")

    set_fused_rs_block_sizes_override(None)


if __name__ == "__main__":
    main()
