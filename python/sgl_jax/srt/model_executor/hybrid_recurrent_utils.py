"""Hybrid recurrent state helpers used by ModelRunner.init_memory_pool.

Module-level pure functions extracted from model_runner.py to keep the runner
focused on the core forward / profile / init flow. None of these depend on
ModelRunner state — they take everything they need as arguments.

Functions cover:
- pool stack construction (KV + recurrent state + hybrid req-to-token pool)
- per-request recurrent + conv byte arithmetic
- HBM budget split between KV and recurrent state pools
- server-args constraint enforcement for hybrid recurrent state models
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool, MemoryPools
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool


def _build_hybrid_pools(
    cfg,
    max_num_reqs: int,
    max_context_len: int,
    tp_size: int,
    token_to_kv_pool,
    mesh,
):
    """Build the hybrid pool stack: RecurrentStatePool + HybridReqToTokenPool + MemoryPools wrapper.

    Caller must pass the runner's linear_recurrent_config (a config whose
    is_linear_attn is True with non-empty kda_layers); mesh is forwarded so
    RecurrentStatePool's buffers get the same TP-aware sharding pattern as
    the token_to_kv_pool.
    """
    linear_attn_config = cfg.linear_attn_config  # dict
    rsp = RecurrentStatePool(
        linear_recurrent_layer_ids=cfg.linear_layer_ids,
        max_num_reqs=max_num_reqs,
        num_heads=linear_attn_config["num_heads"],
        head_dim=linear_attn_config["head_dim"],
        conv_kernel_size=linear_attn_config["short_conv_kernel_size"],
        mesh=mesh,
    )
    hybrid_pool = HybridReqToTokenPool(
        size=max_num_reqs + 1,  # +1 for dummy slot 0
        max_context_len=max_context_len,
        dtype=np.int32,
        recurrent_state_pool=rsp,
    )
    mp = MemoryPools(
        token_to_kv_pool=token_to_kv_pool,
        recurrent_state_pool=rsp,
    )
    return rsp, hybrid_pool, mp


def _build_non_hybrid_memory_pools(token_to_kv_pool):
    """Wrap a single KV pool in MemoryPools (so _forward can uniformly call
    self.memory_pools.replace_all)."""
    return MemoryPools(token_to_kv_pool=token_to_kv_pool)


def _compute_recurrent_per_req_bytes(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    tp_size: int,
    temporal_dtype_bytes: int,
    conv_dtype_bytes: int,
    num_k_heads: int | None = None,
    head_k_dim: int | None = None,
) -> int:
    """Per-device per-request recurrent + conv buffer size in bytes.

    Mirrors RecurrentStatePool buffer shapes :
      per_req_recurrent = L * (H/tp) * D * D * temporal_dtype.itemsize
      per_req_conv      = L * (K-1) * (proj_size/tp) * conv_dtype.itemsize
                         where proj_size = num_heads*head_dim + 2*num_k_heads*head_k_dim
                         (num_k_heads / head_k_dim default to num_heads / head_dim
                          for current Kimi-Linear convention).
    """
    if num_k_heads is None:
        num_k_heads = num_heads
    if head_k_dim is None:
        head_k_dim = head_dim
    assert num_heads % tp_size == 0, f"num_heads {num_heads} must be divisible by tp_size {tp_size}"
    proj_size = num_heads * head_dim + 2 * (num_k_heads * head_k_dim)
    assert proj_size % tp_size == 0, f"proj_size {proj_size} must be divisible by tp_size {tp_size}"
    per_req_recurrent = (
        num_layers * (num_heads // tp_size) * head_dim * head_dim * temporal_dtype_bytes
    )
    per_req_conv = num_layers * (conv_kernel_size - 1) * (proj_size // tp_size) * conv_dtype_bytes
    return per_req_recurrent + per_req_conv


def _split_state_kv_budget(available_bytes: int, ratio: float) -> tuple[int, int]:
    """Split available HBM into (state_budget, kv_budget).

    state_budget = available * r/(1+r), where r = state_to_kv_ratio
    (matches sglang PyTorch mamba_full_memory_ratio formula).
    """
    assert ratio >= 0.0, f"state_to_kv_ratio must be >= 0, got {ratio}"
    state_budget = int(available_bytes * ratio / (1.0 + ratio))
    kv_budget = available_bytes - state_budget
    return state_budget, kv_budget


def _compute_max_num_reqs_from_state_budget(state_budget: int, per_req_bytes: int) -> int:
    """Floor division; returns 0 if state_budget is 0 (degenerate)."""
    if per_req_bytes <= 0:
        return 0
    return state_budget // per_req_bytes


def _per_req_state_bytes_from_config(linear_attn_config: dict, tp_size: int) -> int:
    """Per-request recurrent + conv state bytes for a hybrid recurrent model.

    Resolves temporal_dtype / conv_dtype via the same env var lookup that
    RecurrentStatePool.__init__ uses, so the budget estimate matches the
    pool's actual allocation. Wraps _compute_recurrent_per_req_bytes to keep
    the dtype resolution out of init_memory_pool.
    """
    from sgl_jax.srt.mem_cache.recurrent_state_pool import _resolve_dtype

    temporal_dtype = _resolve_dtype("SGLANG_JAX_RECURRENT_STATE_DTYPE", jnp.float32)
    conv_dtype = _resolve_dtype("SGLANG_JAX_CONV_STATE_DTYPE", jnp.bfloat16)
    return _compute_recurrent_per_req_bytes(
        num_layers=len(linear_attn_config["kda_layers"]),
        num_heads=linear_attn_config["num_heads"],
        head_dim=linear_attn_config["head_dim"],
        conv_kernel_size=linear_attn_config["short_conv_kernel_size"],
        tp_size=tp_size,
        temporal_dtype_bytes=jnp.dtype(temporal_dtype).itemsize,
        conv_dtype_bytes=jnp.dtype(conv_dtype).itemsize,
    )


def _check_state_to_kv_ratio_for_hybrid(state_to_kv_ratio: float) -> None:
    """Fail-fast if state_to_kv_ratio is non-positive when has_recurrent_state.

    Raised early with an actionable message instead of letting the error
    surface from RecurrentStatePool's `assert max_num_reqs > 0` deep inside
    the constructor — that error message would not point at the ratio config.
    """
    if state_to_kv_ratio <= 0:
        raise ValueError(
            f"state_to_kv_ratio={state_to_kv_ratio} <= 0 is invalid for "
            f"has_recurrent_state model: state budget would be 0; "
            f"set --state-to-kv-ratio > 0 (default 0.9)"
        )


def _enforce_recurrent_state_server_constraints(server_args) -> None:
    """Assert both disable_radix_cache=True and disable_overlap_schedule=True
    for hybrid recurrent state models.

    - disable_radix_cache: prefix slots being all-zero would corrupt suffix
      computation in the recurrent path. We require the user to opt in with
      --disable-radix-cache rather than silently overriding their config so
      the constraint is transparent.
    - disable_overlap_schedule: this version does not implement double-buffer
      ping-pong (mamba_ping_pong_track_buffer_size); overlap scheduler would
      race on shared recurrent state.
    """
    assert server_args.disable_radix_cache, (
        "Hybrid recurrent state models require --disable-radix-cache "
        "(prefix sharing is unsafe with recurrent state). Please pass "
        "--disable-radix-cache explicitly."
    )
    assert server_args.disable_overlap_schedule, (
        "Hybrid recurrent state models require --disable-overlap-schedule "
        "(this version does not support double-buffer ping-pong for recurrent state)."
    )
