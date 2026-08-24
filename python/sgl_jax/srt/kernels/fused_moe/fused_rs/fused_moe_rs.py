# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EP MoE with ICI reduce-scatter fused into the ``gmm_fused_rs`` kernel.

A single Pallas call performs: gather -> GMM1 -> activation -> GMM2 -> ICI
reduce-scatter. Only the nodedup path is provided.
"""

import math

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .gmm_fused_rs_nodedup import _select_fused_rs_block_sizes
from .gmm_fused_rs_nodedup import gmm_v2_fused_rs as gmm_v2_fused_rs_nodedup
from .gmm_v2_gather_scatter import _recover_quant_block_size

EXPERT = ("data", "tensor")


def _flatten_partition_axes(*axis_specs):
    axes = ()
    for spec in axis_specs:
        if spec is None:
            continue
        if isinstance(spec, tuple):
            axes += _flatten_partition_axes(*spec)
        else:
            axes += (spec,)
    return axes[0] if len(axes) == 1 else axes


def combine_partition_axes(*axis_specs):
    axes = _flatten_partition_axes(*axis_specs)
    if axes is None or isinstance(axes, str):
        return axes
    combined = ()
    for a in axes:
        if a not in combined:
            combined += (a,)
    return combined[0] if len(combined) == 1 else combined


def get_moe_expert_axis(mesh: Mesh):
    """Return the sglang-jax model-mesh axes that jointly form EP."""
    axes = tuple(axis for axis in EXPERT if axis in mesh.shape)
    if not axes:
        raise ValueError(f"fused_rs requires a data/tensor mesh; got {mesh.shape}")
    return _flatten_partition_axes(*axes)


def get_mesh_shape_product(mesh: Mesh, axis_names) -> int:
    axes = (axis_names,) if isinstance(axis_names, str) else axis_names
    return math.prod(mesh.shape[axis] for axis in axes)


def apply_scoring_fn(scoring_fn: str, x: jax.Array) -> jax.Array:
    if scoring_fn == "softmax":
        return jax.nn.softmax(x, axis=-1)
    if scoring_fn == "sigmoid":
        return jax.nn.sigmoid(x)
    raise NotImplementedError(f"unsupported scoring function: {scoring_fn}")


def _routing_and_topk(
    gating_output,
    scoring_fn,
    topk,
    renormalize,
    dtype,
    mesh,
    *,
    expert_bias: jax.Array | None = None,
    route_scale: float = 1.0,
    router_output_multiplier: float | None = None,
    router_score_division_eps: float | None = None,
):
    if router_output_multiplier is not None:
        gating_output = gating_output * router_output_multiplier

    scores = apply_scoring_fn(scoring_fn, gating_output)
    expert_axis = get_moe_expert_axis(mesh)
    # Model-loaded arrays use Explicit mesh axes.  A sharding constraint may
    # only name Auto axes, whereas ``reshard`` supports both Explicit and Auto
    # meshes and states the actual data-movement contract we need here.
    scores = jax.sharding.reshard(scores, NamedSharding(mesh, P(expert_axis, None)))

    if expert_bias is not None:
        # Bias shifts selection but not the final weights.
        scores_for_topk = scores + expert_bias
        _, topk_indices = jax.lax.top_k(scores_for_topk, k=topk)
        topk_weights = jnp.take_along_axis(scores, topk_indices, axis=1)
    else:
        topk_weights, topk_indices = jax.lax.top_k(scores, k=topk)

    if renormalize:
        denom = topk_weights.sum(axis=-1, keepdims=True)
        if router_score_division_eps is not None:
            denom = denom + router_score_division_eps
        topk_weights = topk_weights / denom

    topk_weights = (topk_weights * route_scale).astype(dtype)
    return topk_weights, topk_indices


# Below this routed-row count, packed routing indices use scalar-prefetch SMEM.
# Larger prefills are staged tile-by-tile from HBM by gmm_v2_fused_rs.
_FUSED_RS_MAX_SAFE_SIZE_M = 240000


def _assert_fused_rs_smem_safe(size_m: int) -> None:
    assert size_m > 0, f"gmm_v2_fused_rs requires positive size_m, got {size_m}"


def _quantize_hidden_per_tensor(
    hidden_local: jax.Array,
    topk_indices_local: jax.Array,
    *,
    scale_multiplier: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Quantize one EP rank's physical hidden shard with one FP32 scale.

    Both the scale and payload cover the complete physical shard.  Consequently,
    changing only routing padding cannot change the communicated activation.
    Routing sentinels, rather than payload mutation, suppress invalid outputs.
    """
    del topk_indices_local
    hidden_f32 = hidden_local.astype(jnp.float32)
    fp8_max = jnp.asarray(jnp.finfo(jnp.float8_e4m3fn).max, dtype=jnp.float32)
    amax = jnp.max(jnp.abs(hidden_f32))
    scale = (
        jnp.maximum(amax, jnp.asarray(1e-12, dtype=jnp.float32))
        / fp8_max
        * scale_multiplier
    )
    quantized = jnp.clip(hidden_f32 / scale, -fp8_max, fp8_max).astype(
        jnp.float8_e4m3fn
    )
    return quantized, scale


def _dequantize_hidden_per_rank(
    payload: jax.Array,
    rank_scales: jax.Array,
    *,
    rows_per_rank: int,
    out_dtype: jnp.dtype,
) -> jax.Array:
    """Materialize an FP8 AllGather payload using its source-rank scales.

    The communication payload remains FP8.  Dequantizing after the collective
    gives the established BF16 fused-RS path a conventional row-major input,
    avoiding the target-TPU layout bug in the experimental direct-FP8 Pallas
    input while retaining the ICI bandwidth reduction.
    """
    row_scales = jnp.repeat(rank_scales.astype(jnp.float32), rows_per_rank)
    return (payload.astype(jnp.float32) * row_scales[:, None]).astype(out_dtype)


def _all_gather_token_hidden(
    token_hidden: jax.Array,
    *,
    axis_name,
    fp8_enabled: bool,
) -> jax.Array:
    """All-gather token shards, optionally quantizing the payload to FP8."""
    if not fp8_enabled:
        return jax.lax.all_gather(token_hidden, axis_name=axis_name, axis=0, tiled=True)

    with jax.named_scope("moe_fp8_post_gather"):
        out_dtype = token_hidden.dtype
        token_hidden_f32 = token_hidden.astype(jnp.float32)
        fp8_max = jnp.array(jnp.finfo(jnp.float8_e4m3fn).max, dtype=jnp.float32)
        absmax = jnp.max(jnp.abs(token_hidden_f32), axis=-1, keepdims=True)
        scale = jnp.maximum(absmax, jnp.array(1e-6, dtype=jnp.float32)) / fp8_max
        token_hidden_fp8 = jnp.clip(token_hidden_f32 / scale, -fp8_max, fp8_max).astype(
            jnp.float8_e4m3fn
        )
        gathered_fp8 = jax.lax.all_gather(token_hidden_fp8, axis_name=axis_name, axis=0, tiled=True)
        gathered_scale = jax.lax.all_gather(scale, axis_name=axis_name, axis=0, tiled=True)
        return (gathered_fp8.astype(jnp.float32) * gathered_scale).astype(out_dtype)


def _compute_rs_routing(topk_indices, *, num_experts, topk):
    """Inline routing: lhs_indices, group_sizes, output_indices, topk_slot_indices.

    Uses integer arithmetic and one-hot+sum (not gathers / bincount) to keep the
    computation cheap and fusable. dtype stays int32 for scalar-prefetch refs.

    Padded model batches mark invalid routes with ``-1``.  Sort those entries
    after every real expert so the cumulative ``group_sizes`` offsets still
    index the valid prefix of the packed routing arrays.  The kernel processes
    only that prefix; its zero-initialized output therefore leaves invalid
    token/slot contributions at zero.
    """
    topk_indices_flat = topk_indices.flatten()
    valid_routes = jnp.logical_and(
        topk_indices_flat >= 0,
        topk_indices_flat < num_experts,
    )
    sort_expert_ids = jnp.where(valid_routes, topk_indices_flat, num_experts)
    topk_argsort_indices = jnp.argsort(sort_expert_ids)
    expert_ids = jnp.arange(num_experts, dtype=jnp.int32)
    group_sizes = jnp.sum(
        (topk_indices_flat[:, None] == expert_ids[None, :]).astype(jnp.int32),
        axis=0,
    )
    lhs_indices = topk_argsort_indices // topk
    topk_slot_indices = topk_argsort_indices % topk
    output_indices = lhs_indices
    return lhs_indices, group_sizes, output_indices, topk_slot_indices


def moe_gmm_local_rs_nodedup(
    hidden_states_local: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    w1_global_scale: jax.Array | None,
    w2_global_scale: jax.Array | None,
    group_offset: jax.Array,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    post_expert_norm_weight_input: jax.Array | None = None,
    *,
    hidden_states_scale: jax.Array | None = None,
    w3: jax.Array | None = None,
    w3_scale: jax.Array | None = None,
    activation: str,
    topk: int,
    ep_size: int,
    ep_axis_name=EXPERT,
    has_post_norm: bool = False,
    sp_enabled: bool = True,
    fp8_post_gather: bool = False,
) -> jax.Array:
    """Per-chip MoE body: ICI direct-write per row, then weighted top_k reduce."""
    num_tokens = hidden_states_local.shape[0]
    hidden_size = w2.shape[-1]
    chunk_size = num_tokens // ep_size
    num_experts = w1.shape[0] * ep_size  # global num_experts
    num_local_experts = w1.shape[0]
    my_id = jax.lax.axis_index(ep_axis_name)

    # Routing inlined here so it fuses with the kernel pipeline.
    lhs_indices, group_sizes, output_indices, topk_slot_indices = _compute_rs_routing(
        topk_indices, num_experts=num_experts, topk=topk
    )
    size_m = lhs_indices.shape[0]
    _assert_fused_rs_smem_safe(size_m)

    # Local row range [local_start, local_end) for this chip's experts.
    go_val = group_offset[0]
    expert_idx = jnp.arange(num_experts, dtype=jnp.int32)
    local_start = jnp.sum(jnp.where(expert_idx < go_val, group_sizes, 0))
    local_end = local_start + jnp.sum(
        jnp.where(
            jnp.logical_and(expert_idx >= go_val, expert_idx < go_val + num_local_experts),
            group_sizes,
            0,
        )
    )

    # Rows from other chips destined for me (dest == my_id and not local).
    send_dest_chips = output_indices // chunk_size
    rows = jnp.arange(size_m, dtype=jnp.int32)
    valid_row_count = jnp.sum(group_sizes)
    row_is_valid = rows < valid_row_count
    row_is_mine = jnp.logical_and(rows >= local_start, rows < local_end)
    to_me_remote = jnp.logical_and(
        row_is_valid,
        jnp.logical_and(send_dest_chips == my_id, jnp.logical_not(row_is_mine)),
    )
    my_recv_count = jnp.sum(jnp.where(to_me_remote, 1, 0))
    total_recv_count = jnp.array([my_recv_count], dtype=jnp.int32)

    # tile_m here MUST match the value the kernel selects internally, otherwise
    # max_num_gm under-counts and the kernel's final gather DMA is left unawaited.
    block_sizes = _select_fused_rs_block_sizes(
        size_m=size_m,
        size_k1=w1.shape[1],
        size_n1=w1.shape[2] + (w3.shape[2] if w3 is not None else 0),
        size_k2=w2.shape[1],
        size_n2=w2.shape[2],
        size_group=num_local_experts,
        size_lhs_group=group_sizes.shape[0],
        ep_size=ep_size,
        out_dtype=(
            jnp.bfloat16 if hidden_states_scale is not None else hidden_states_local.dtype
        ),
        w1_dtype=w1.dtype,
        w2_dtype=w2.dtype,
        is_quantized=w1_scale is not None,
        quant_block_size=(
            _recover_quant_block_size(w1.shape[1], w1_scale.shape[1])
            if w1_scale is not None
            else None
        ),
        act_fn=activation,
        fp8_direct_write=fp8_post_gather,
    )
    tile_m = block_sizes.tile_m
    max_num_gm = jnp.array(num_experts + (size_m + tile_m - 1) // tile_m - 1, dtype=jnp.int32)

    out_buf = gmm_v2_fused_rs_nodedup(
        hidden_states_local,
        w1,
        w2,
        group_sizes,
        lhs_indices,
        output_indices,
        hidden_states_scale=hidden_states_scale,
        w3=w3,
        w1_scale=w1_scale,
        w3_scale=w3_scale,
        w2_scale=w2_scale,
        w1_global_scale=w1_global_scale,
        w2_global_scale=w2_global_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        act_fn=activation,
        output_size=num_tokens,
        group_offset=group_offset,
        topk_indices=topk_slot_indices,
        ep_size=ep_size,
        ep_axis_name=ep_axis_name,
        max_num_gm=max_num_gm,
        total_recv_count=total_recv_count,
        top_k=topk,
        fp8_direct_write=fp8_post_gather,
    )

    # SP off: topk_weights is replicated; slice this chip's shard to match out_3d.
    local_topk_weights = (
        topk_weights
        if sp_enabled
        else jax.lax.dynamic_slice_in_dim(topk_weights, my_id * chunk_size, chunk_size, axis=0)
    )
    out_3d = out_buf.reshape(chunk_size, topk, hidden_size)
    post_expert_norm_weight = post_expert_norm_weight_input if has_post_norm else None
    if post_expert_norm_weight is not None:
        norm_size = post_expert_norm_weight.shape[0]  # unpadded hidden_size
        pnw_raw = post_expert_norm_weight.astype(jnp.float32) + 1.0
        if hidden_size > norm_size:
            # Zero padded columns so they don't affect variance.
            col_idx = jnp.arange(hidden_size, dtype=jnp.int32)
            valid_mask = (col_idx < norm_size)[None, None, :]
            out_f32 = out_3d.astype(jnp.float32) * valid_mask
            pnw = jnp.concatenate([pnw_raw, jnp.zeros(hidden_size - norm_size, dtype=jnp.float32)])
        else:
            out_f32 = out_3d.astype(jnp.float32)
            pnw = pnw_raw
        var = jnp.sum(out_f32**2, axis=-1, keepdims=True) / norm_size
        out_3d = (out_f32 * jax.lax.rsqrt(var + 1e-8) * pnw[None, None, :]).astype(out_3d.dtype)

    # Match fused-v2: routing weights and top-k accumulation stay f32, while
    # the backend boundary returns the token dtype.
    token_hidden = jnp.sum(
        out_3d.astype(jnp.float32) * local_topk_weights.astype(jnp.float32)[:, :, None],
        axis=1,
    ).astype(out_3d.dtype)

    if sp_enabled:
        # Kernel reduce-scatter is the SP exit; output stays token-sharded.
        return token_hidden
    # SP off: gather the per-chip token shard back to the replicated batch.
    return _all_gather_token_hidden(
        token_hidden,
        axis_name=ep_axis_name,
        fp8_enabled=fp8_post_gather,
    )


def expert_parallel_gmm_rs(
    hidden_states: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    *,
    w3: jax.Array | None = None,
    w3_scale: jax.Array | None = None,
    activation: str,
    topk: int,
    mesh: Mesh,
    post_expert_norm_weight: jax.Array | None = None,
    w1_global_scale: jax.Array | None = None,
    w2_global_scale: jax.Array | None = None,
    fp8_post_gather: bool = False,
    fp8_hidden_all_gather: bool = False,
    _fp8_hidden_direct_prequantized: bool = False,
    _fp8_hidden_scale_multiplier: float = 1.0,
) -> jax.Array:
    """Run fused-RS with sglang-jax's token-sharded model mesh.

    The imported kernel expects every EP rank to see the full token shard before
    computing its local experts. sglang-jax instead shards tokens over the same
    ``(data, tensor)`` axes used for EP, so the shard-map body first performs the
    upstream all-gather that PR #3040 intentionally leaves outside Pallas. The
    kernel's direct writes form the matching reduce-scatter on exit.
    """
    del fp8_post_gather
    if _fp8_hidden_direct_prequantized and not fp8_hidden_all_gather:
        raise ValueError(
            "direct prequantized FP8 consumer requires FP8 Hidden AllGather"
        )
    if _fp8_hidden_scale_multiplier <= 0:
        raise ValueError("FP8 Hidden AllGather scale multiplier must be positive")
    if _fp8_hidden_scale_multiplier != 1.0 and not fp8_hidden_all_gather:
        raise ValueError(
            "FP8 Hidden AllGather scale multiplier requires FP8 Hidden AllGather"
        )
    if fp8_hidden_all_gather:
        if (
            w1_scale is None
            or w2_scale is None
            or (w3 is not None and w3_scale is None)
        ):
            raise ValueError(
                "FP8 Hidden AllGather currently requires the fused-RS W8A8 "
                "weight-scale path"
            )
        if w1.dtype != jnp.float8_e4m3fn or w2.dtype != jnp.float8_e4m3fn:
            raise ValueError(
                "FP8 Hidden AllGather currently requires float8_e4m3fn expert weights"
            )
    expert_axis = get_moe_expert_axis(mesh)
    ep_size = get_mesh_shape_product(mesh, expert_axis)
    ep_p_spec = P(expert_axis)
    token_p_spec = P(expert_axis, None)
    num_experts = w1.shape[0]
    num_experts_per_shard = num_experts // ep_size
    group_offset = jax.sharding.reshard(
        jnp.arange(0, num_experts, num_experts_per_shard),
        NamedSharding(mesh, P(expert_axis)),
    )

    w1_scale_spec = None if w1_scale is None else ep_p_spec
    w1_bias_spec = None if w1_bias is None else ep_p_spec
    w2_scale_spec = None if w2_scale is None else ep_p_spec
    w2_bias_spec = None if w2_bias is None else ep_p_spec
    w1_gs_spec = None if w1_global_scale is None else ep_p_spec
    w2_gs_spec = None if w2_global_scale is None else ep_p_spec
    w3_spec = None if w3 is None else ep_p_spec
    w3_scale_spec = None if w3_scale is None else ep_p_spec

    has_post_norm = post_expert_norm_weight is not None
    post_norm_weight = (
        post_expert_norm_weight
        if post_expert_norm_weight is not None
        else jnp.zeros((1,), jnp.bfloat16)
    )

    def _run(
        hidden_local,
        w1_local,
        w1_scale_local,
        w1_bias_local,
        w2_local,
        w2_scale_local,
        w2_bias_local,
        w1_global_scale_local,
        w2_global_scale_local,
        w3_local,
        w3_scale_local,
        group_offset_local,
        topk_weights_local,
        topk_indices_local,
        post_norm_weight_local,
    ):
        # Keep the two upstream collectives separately identifiable in a real
        # fused-RS trace.  These scopes are diagnostic metadata only: the
        # collective implementation and sharding contract remain unchanged.
        hidden_scale_global = None
        if fp8_hidden_all_gather:
            with jax.named_scope("fused_rs_hidden_quantize"):
                hidden_payload_local, hidden_scale_local = _quantize_hidden_per_tensor(
                    hidden_local,
                    topk_indices_local,
                    scale_multiplier=_fp8_hidden_scale_multiplier,
                )
            with jax.named_scope("fused_rs_hidden_all_gather"):
                hidden_global = jax.lax.all_gather(
                    hidden_payload_local,
                    axis_name=expert_axis,
                    axis=0,
                    tiled=True,
                )
            with jax.named_scope("fused_rs_hidden_scale_all_gather"):
                hidden_scale_by_rank = jax.lax.all_gather(
                    hidden_scale_local[None],
                    axis_name=expert_axis,
                    axis=0,
                    tiled=True,
                )
            if _fp8_hidden_direct_prequantized:
                with jax.named_scope("fused_rs_hidden_scale_expand"):
                    hidden_scale_global = jnp.repeat(
                        hidden_scale_by_rank,
                        hidden_global.shape[0] // ep_size,
                    )
            else:
                with jax.named_scope("fused_rs_hidden_dequantize"):
                    # Target-TPU explicit oracles prove the collective payload
                    # and scales are exact, but the direct prequantized Pallas
                    # input is not: even a uniform rank scale remains wrong.
                    # Materialize BF16 locally after the FP8 collective, then
                    # reuse the mature per-row W8A8 GMM1 path.  This preserves
                    # the 2x ICI payload reduction while keeping the broken
                    # direct-FP8 path out of the production opt-in until its
                    # VMEM/MXU layout is independently fixed.
                    hidden_global = _dequantize_hidden_per_rank(
                        hidden_global,
                        hidden_scale_by_rank,
                        rows_per_rank=hidden_global.shape[0] // ep_size,
                        out_dtype=hidden_local.dtype,
                    )
        else:
            with jax.named_scope("fused_rs_hidden_all_gather"):
                hidden_global = jax.lax.all_gather(
                    hidden_local,
                    axis_name=expert_axis,
                    axis=0,
                    tiled=True,
                )
        with jax.named_scope("fused_rs_topk_ids_all_gather"):
            topk_indices_global = jax.lax.all_gather(
                topk_indices_local,
                axis_name=expert_axis,
                axis=0,
                tiled=True,
            )
        return moe_gmm_local_rs_nodedup(
            hidden_global,
            w1_local,
            w1_scale_local,
            w1_bias_local,
            w2_local,
            w2_scale_local,
            w2_bias_local,
            w1_global_scale_local,
            w2_global_scale_local,
            group_offset_local,
            topk_weights_local,
            topk_indices_global,
            post_norm_weight_local,
            hidden_states_scale=hidden_scale_global,
            w3=w3_local,
            w3_scale=w3_scale_local,
            activation=activation,
            topk=topk,
            ep_size=ep_size,
            ep_axis_name=expert_axis,
            has_post_norm=has_post_norm,
            sp_enabled=True,
            fp8_post_gather=False,
        )

    result = jax.shard_map(
        _run,
        mesh=mesh,
        in_specs=(
            token_p_spec,
            ep_p_spec,  # w1
            w1_scale_spec,
            w1_bias_spec,
            ep_p_spec,  # w2
            w2_scale_spec,
            w2_bias_spec,
            w1_gs_spec,
            w2_gs_spec,
            w3_spec,
            w3_scale_spec,
            ep_p_spec,  # group_offset
            token_p_spec,
            token_p_spec,
            P(),  # post_expert_norm_weight
        ),
        out_specs=token_p_spec,
        check_vma=False,
    )(
        hidden_states,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        w1_global_scale,
        w2_global_scale,
        w3,
        w3_scale,
        group_offset,
        topk_weights,
        topk_indices,
        post_norm_weight,
    )

    return result


def _fused_moe_func_rs_impl(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    gating_output: jax.Array | None,
    topk: int,
    renormalize: bool,
    mesh: Mesh,
    activation: str,
    scoring_fn: str,
    post_expert_norm_weight: jax.Array | None = None,
    topk_weights: jax.Array | None = None,
    topk_indices: jax.Array | None = None,
    fp8_post_gather: bool = False,
    fp8_hidden_all_gather: bool = False,
    _fp8_hidden_direct_prequantized: bool = False,
    _fp8_hidden_scale_multiplier: float = 1.0,
    w3: jax.Array | None = None,
    w3_scale: jax.Array | None = None,
) -> jax.Array:
    """EP MoE with ICI reduce-scatter fused in kernel (gmm_fused_rs).

    Uses caller-supplied ``topk_weights``/``topk_indices`` when both are given;
    otherwise computes top-k from ``gating_output``. Then runs the fused kernel
    (gather -> GMM1 -> act -> GMM2 -> ICI reduce-scatter) and reduces over top_k.
    """
    num_tokens, hidden_size = hidden_states.shape
    global_num_experts, padded_hidden_size, _ = w1.shape
    dtype = hidden_states.dtype
    expert_axis = get_moe_expert_axis(mesh)

    assert (num_tokens * topk) % 16 == 0
    if topk_weights is not None and topk_indices is not None:
        # Honor pre-computed routing; do not recompute from gating_output.
        topk_weights = topk_weights.astype(jnp.float32)
        topk_weights = jax.sharding.reshard(topk_weights, NamedSharding(mesh, P(expert_axis, None)))
        topk_indices = jax.sharding.reshard(topk_indices, NamedSharding(mesh, P(expert_axis, None)))
    else:
        assert gating_output is not None, (
            "fused_moe_func_rs: either pre-computed topk_weights+topk_indices "
            "or gating_output must be provided."
        )
        assert gating_output.shape == (num_tokens, global_num_experts)
        topk_weights, topk_indices = _routing_and_topk(
            gating_output, scoring_fn, topk, renormalize, dtype, mesh
        )

    # Pad hidden_states to w1's K dimension if needed.
    if padded_hidden_size != hidden_size:
        hidden_states = jnp.pad(hidden_states, ((0, 0), (0, padded_hidden_size - hidden_size)))

    result = expert_parallel_gmm_rs(
        hidden_states,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        topk_weights,
        topk_indices,
        w3=w3,
        w3_scale=w3_scale,
        activation=activation,
        topk=topk,
        mesh=mesh,
        post_expert_norm_weight=post_expert_norm_weight,
        fp8_post_gather=fp8_post_gather,
        fp8_hidden_all_gather=fp8_hidden_all_gather,
        _fp8_hidden_direct_prequantized=_fp8_hidden_direct_prequantized,
        _fp8_hidden_scale_multiplier=_fp8_hidden_scale_multiplier,
    )

    return result[:num_tokens, :hidden_size]


_FUSED_MOE_RS_STATIC_ARGNAMES = (
    "topk",
    "renormalize",
    "mesh",
    "activation",
    "scoring_fn",
    "fp8_post_gather",
    "fp8_hidden_all_gather",
    "_fp8_hidden_direct_prequantized",
    "_fp8_hidden_scale_multiplier",
)

fused_moe_func_rs = jax.jit(
    _fused_moe_func_rs_impl,
    static_argnames=_FUSED_MOE_RS_STATIC_ARGNAMES,
)

# The strict EP32 trace showed that XLA placed both the 768 MiB BF16 Hidden
# AllGather and the routing gather on SparseCore.  The latter was launched
# asynchronously but could not execute until the former completed.  This
# benchmarkable variant keeps sub-1 GiB AllGathers on TensorCore so the routing
# SparseCore work can run concurrently.  It remains separate from the default
# until target-TPU A/B evidence proves the end-to-end win.
fused_moe_func_rs_tc_hidden_all_gather = jax.jit(
    _fused_moe_func_rs_impl,
    static_argnames=_FUSED_MOE_RS_STATIC_ARGNAMES,
    compiler_options={
        "xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes": str(1 << 30),
    },
)


__all__ = [
    "fused_moe_func_rs",
    "fused_moe_func_rs_tc_hidden_all_gather",
    "expert_parallel_gmm_rs",
    "moe_gmm_local_rs_nodedup",
    "_compute_rs_routing",
    "_FUSED_RS_MAX_SAFE_SIZE_M",
    "_assert_fused_rs_smem_safe",
    "_quantize_hidden_per_tensor",
]
