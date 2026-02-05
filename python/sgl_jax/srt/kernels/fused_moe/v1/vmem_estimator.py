from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import jax.numpy as jnp

if TYPE_CHECKING:
    pass


class _BlockConfigLike(Protocol):
    bt: int
    bts: int | None
    bf: int
    bd1: int
    bd2: int
    bse: int


# Leave headroom for compiler padding/alignment and any unmodeled VMEM usage.
DEFAULT_TPU_VMEM_BUDGET_MB = 60
DEFAULT_TPU_VMEM_BUDGET_BYTES = DEFAULT_TPU_VMEM_BUDGET_MB * 1024 * 1024


def dtype_packing_32bit(dtype: jnp.dtype) -> int:
    """Match get_dtype_packing() in fused_moe kernel (32-bit repack width)."""
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype packing for {dtype=} ({bits=} bits).")
    return 32 // bits


def align_to(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"{multiple=} must be positive.")
    x = int(x)
    return ((x + multiple - 1) // multiple) * multiple


def estimate_fused_moe_vmem_bytes(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    router_dtype: jnp.dtype,
    cfg: _BlockConfigLike,
    use_shared_expert: bool = False,
    subc_quant_wsz: int | None = None,
    include_routing_temporaries: bool = True,
) -> int:
    """Conservative fused_moe VMEM estimate (TPU VMEM is 64MB/core).

    This is used for benchmark candidate filtering.

    Notes:
    - Only VMEM allocations are modeled; SMEM and semaphores are excluded.
    - When `include_routing_temporaries=True`, we include a best-effort estimate
      of additional float32 temporaries materialized in the routing/top-k code.
    """
    bt = int(cfg.bt)
    bts = bt if cfg.bts is None else int(cfg.bts)
    bf = int(cfg.bf)
    bd1 = int(cfg.bd1)
    bd2 = int(cfg.bd2)

    if bt <= 0 or bts <= 0 or bf <= 0 or bd1 <= 0 or bd2 <= 0:
        raise ValueError(f"Non-positive block config: {cfg=}")
    if ep_size <= 0:
        raise ValueError(f"{ep_size=} must be positive.")

    token_bytes = jnp.dtype(dtype).itemsize
    weight_bytes = jnp.dtype(weight_dtype).itemsize
    router_bytes = jnp.dtype(router_dtype).itemsize

    t_packing = dtype_packing_32bit(dtype)
    padded_num_experts = align_to(num_experts, 128)
    padded_top_k = align_to(top_k, 128)
    a2a_max_tokens = align_to(bt * ep_size, bts)

    hidden_per_pack = hidden_size // t_packing
    bd1_per_pack = bd1 // t_packing
    bd2_per_pack = bd2 // t_packing

    # Matches fused_moe kernel scratch shapes.
    acc_bt = math.gcd(bt, 16)
    a2a_g_acc = 2 * top_k * acc_bt * t_packing * hidden_per_pack * token_bytes
    b_output = 2 * bt * hidden_size * token_bytes
    b_gating = 2 * bt * padded_num_experts * router_bytes
    top_k_logits = bt * top_k * 4  # float32

    w1 = 2 * t_packing * bd1_per_pack * bf * weight_bytes
    w3 = 2 * t_packing * bd1_per_pack * bf * weight_bytes
    w2 = 2 * t_packing * bf * bd2_per_pack * weight_bytes

    w1_scale = 0
    w3_scale = 0
    w2_scale = 0
    if subc_quant_wsz is not None:
        w1_scale = 2 * t_packing * (bd1_per_pack // subc_quant_wsz) * 1 * bf * 4
        w3_scale = 2 * t_packing * (bd1_per_pack // subc_quant_wsz) * 1 * bf * 4
        w2_scale = 2 * t_packing * (bf // subc_quant_wsz) * 1 * bd2_per_pack * 4

    b1 = 0
    b2 = 0
    b3 = 0
    # Kernel stores biases in VMEM as float32 (when present).
    # Shapes match: b_b*_x2_vmem.
    # We include these unconditionally when user passes biases (caller decides).
    # For this estimator, model presence via `subc_quant_wsz` only; bias sizes are
    # small relative to weights/scratch and not a tuning limiter.

    b_acc = 2 * a2a_max_tokens * bf * 4  # float32
    b_stage = 2 * bts * t_packing * bd1_per_pack * token_bytes
    a2a_s_acc_stage = 3 * bts * t_packing * bd2_per_pack * token_bytes
    b_bias = padded_num_experts * 4  # float32

    routing_temporaries = 0
    if include_routing_temporaries:
        # Best-effort conservative estimate of extra float32 work buffers
        # materialized in routing/top-k.
        routing_work_f32 = bt * padded_num_experts * 4
        get_top_k_input_f32 = bt * padded_num_experts * 4
        get_top_k_t2e = bt * padded_num_experts * 4
        get_top_k_iota = bt * padded_num_experts * 4
        get_top_k_mask = bt * padded_num_experts * 4
        get_top_k_padded_iota = bt * padded_top_k * 4
        get_top_k_t2e_routing = bt * padded_top_k * 4
        get_top_k_logits_sum = bt * padded_top_k * 4
        get_top_k_logits_lst = top_k * bt * padded_top_k * 4
        routing_temporaries = (
            routing_work_f32
            + get_top_k_input_f32
            + get_top_k_t2e
            + get_top_k_iota
            + get_top_k_mask
            + get_top_k_padded_iota
            + get_top_k_t2e_routing
            + get_top_k_logits_sum
            + get_top_k_logits_lst
        )

    total_bytes = (
        a2a_g_acc
        + b_output
        + b_gating
        + top_k_logits
        + w1
        + w3
        + w2
        + w1_scale
        + w3_scale
        + w2_scale
        + b_acc
        + b_stage
        + a2a_s_acc_stage
        + b_bias
        + routing_temporaries
        + b1
        + b2
        + b3
    )

    if use_shared_expert:
        bse = int(cfg.bse)
        se_w1 = 2 * t_packing * (bd1 // t_packing) * bse * weight_bytes
        se_w3 = 2 * t_packing * (bd1 // t_packing) * bse * weight_bytes
        se_w2 = 2 * t_packing * bse * (bd2 // t_packing) * weight_bytes
        se_tokens = 2 * 2 * bt * t_packing * (bd1 // t_packing) * token_bytes
        se_acc = 2 * bt * hidden_size * 4  # float32
        total_bytes += se_w1 + se_w3 + se_w2 + se_tokens + se_acc

        if subc_quant_wsz is not None:
            # These are full vectors (not per-bf tiles), matching kernel scratch.
            total_bytes += intermediate_size * 4  # w1_shared_scale_all
            total_bytes += intermediate_size * 4  # w3_shared_scale_all
            total_bytes += hidden_size * 4  # w2_shared_scale_all

    return int(total_bytes)


def fused_moe_vmem_breakdown_bytes(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    ep_size: int,
    dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    router_dtype: jnp.dtype,
    cfg: _BlockConfigLike,
    use_shared_expert: bool = False,
    subc_quant_wsz: int | None = None,
    include_routing_temporaries: bool = True,
) -> tuple[int, dict[str, int]]:
    """Return (total_bytes, per-component-bytes) for fused_moe VMEM usage."""
    bt = int(cfg.bt)
    bts = bt if cfg.bts is None else int(cfg.bts)
    bf = int(cfg.bf)
    bd1 = int(cfg.bd1)
    bd2 = int(cfg.bd2)
    bse = int(cfg.bse)

    token_bytes = jnp.dtype(dtype).itemsize
    weight_bytes = jnp.dtype(weight_dtype).itemsize
    router_bytes = jnp.dtype(router_dtype).itemsize

    t_packing = dtype_packing_32bit(dtype)
    padded_num_experts = align_to(num_experts, 128)
    padded_top_k = align_to(top_k, 128)
    a2a_max_tokens = align_to(bt * ep_size, bts)

    hidden_per_pack = hidden_size // t_packing
    bd1_per_pack = bd1 // t_packing
    bd2_per_pack = bd2 // t_packing

    out: dict[str, int] = {}

    # Routing / output-side scratch.
    acc_bt = math.gcd(bt, 16)
    out["a2a_g_acc_vmem"] = 2 * top_k * acc_bt * t_packing * hidden_per_pack * token_bytes
    out["top_k_logits_vmem"] = bt * top_k * 4  # float32
    out["b_gating_x2_vmem"] = 2 * bt * padded_num_experts * router_bytes
    out["b_output_x2_vmem"] = 2 * bt * hidden_size * token_bytes

    # Weight tiles (double-buffered).
    out["b_w1_x2_vmem"] = 2 * t_packing * bd1_per_pack * bf * weight_bytes
    out["b_w3_x2_vmem"] = 2 * t_packing * bd1_per_pack * bf * weight_bytes
    out["b_w2_x2_vmem"] = 2 * t_packing * bf * bd2_per_pack * weight_bytes

    # Quantization scale scratch buffers (F32).
    if subc_quant_wsz is not None:
        out["b_w1_scale_x2_vmem"] = 2 * t_packing * (bd1_per_pack // subc_quant_wsz) * 1 * bf * 4
        out["b_w3_scale_x2_vmem"] = 2 * t_packing * (bd1_per_pack // subc_quant_wsz) * 1 * bf * 4
        out["b_w2_scale_x2_vmem"] = 2 * t_packing * (bf // subc_quant_wsz) * 1 * bd2_per_pack * 4
    else:
        out["b_w1_scale_x2_vmem"] = 0
        out["b_w3_scale_x2_vmem"] = 0
        out["b_w2_scale_x2_vmem"] = 0

    # Accumulators / staging.
    out["b_acc_vmem"] = 2 * a2a_max_tokens * bf * 4  # float32
    out["b_stage_x2_vmem"] = 2 * bts * t_packing * bd1_per_pack * token_bytes
    out["a2a_s_acc_stage_x3_vmem"] = 3 * bts * t_packing * bd2_per_pack * token_bytes
    out["b_bias_vmem"] = padded_num_experts * 4  # float32

    if include_routing_temporaries:
        routing_work_f32 = bt * padded_num_experts * 4
        get_top_k_input_f32 = bt * padded_num_experts * 4
        get_top_k_t2e = bt * padded_num_experts * 4
        get_top_k_iota = bt * padded_num_experts * 4
        get_top_k_mask = bt * padded_num_experts * 4
        get_top_k_padded_iota = bt * padded_top_k * 4
        get_top_k_t2e_routing = bt * padded_top_k * 4
        get_top_k_logits_sum = bt * padded_top_k * 4
        get_top_k_logits_lst = top_k * bt * padded_top_k * 4
        out["routing_temporaries"] = (
            routing_work_f32
            + get_top_k_input_f32
            + get_top_k_t2e
            + get_top_k_iota
            + get_top_k_mask
            + get_top_k_padded_iota
            + get_top_k_t2e_routing
            + get_top_k_logits_sum
            + get_top_k_logits_lst
        )
    else:
        out["routing_temporaries"] = 0

    # Shared expert scratch buffers.
    if use_shared_expert:
        out["b_se_tokens_vmem"] = 2 * 2 * bt * t_packing * (bd1 // t_packing) * token_bytes
        out["b_se_w1_x2_vmem"] = 2 * t_packing * (bd1 // t_packing) * bse * weight_bytes
        out["b_se_w3_x2_vmem"] = 2 * t_packing * (bd1 // t_packing) * bse * weight_bytes
        out["b_se_w2_x2_vmem"] = 2 * t_packing * bse * (bd2 // t_packing) * weight_bytes
        out["b_se_acc_vmem"] = 2 * bt * hidden_size * 4  # float32

        if subc_quant_wsz is not None:
            out["b_se_w1_scale_all"] = intermediate_size * 4
            out["b_se_w3_scale_all"] = intermediate_size * 4
            out["b_se_w2_scale_all"] = hidden_size * 4
        else:
            out["b_se_w1_scale_all"] = 0
            out["b_se_w3_scale_all"] = 0
            out["b_se_w2_scale_all"] = 0
    else:
        out["b_se_tokens_vmem"] = 0
        out["b_se_w1_x2_vmem"] = 0
        out["b_se_w3_x2_vmem"] = 0
        out["b_se_w2_x2_vmem"] = 0
        out["b_se_acc_vmem"] = 0
        out["b_se_w1_scale_all"] = 0
        out["b_se_w3_scale_all"] = 0
        out["b_se_w2_scale_all"] = 0

    total = int(sum(out.values()))
    return total, out


def format_vmem_bytes_breakdown(*, total: int, items: dict[str, int]) -> str:
    def _mb(b: int) -> str:
        return f"{b / (1024 * 1024):.2f}"

    lines = ["    VMEM Breakdown:"]
    for k, v in sorted(items.items(), key=lambda kv: kv[1], reverse=True):
        if v == 0:
            continue
        lines.append(f"      {k:24s} {_mb(v)} MB")
    lines.append("      ----------------------------")
    lines.append(f"      Total:                  {_mb(total)} MB")
    return "\n".join(lines)
