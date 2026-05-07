"""ModelRunnerKVCacheMixin -- all pool init logic extracted from ModelRunner.

Aligns with upstream sglang model_runner_kv_cache_mixin.py:
- Mixin methods use ``self: ModelRunner`` annotations
- Pure functions at module level for independent testing
- ModelRunner inherits ModelRunnerKVCacheMixin
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool, MemoryPools
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level pure functions (from epic hybrid_recurrent_utils.py)
# ---------------------------------------------------------------------------


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
    """Per-device per-request recurrent + conv buffer size in bytes."""
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


def _split_state_kv_budget(
    available_bytes: int,
    ratio: float,
    per_req_state_bytes: int,
) -> tuple[int, int]:
    """Split available HBM into (state_max_reqs, kv_budget).

    state_budget_raw = available * r/(1+r)
    state_max_reqs   = state_budget_raw // per_req_state_bytes
    kv_budget        = available - state_max_reqs * per_req_state_bytes
    """
    assert ratio >= 0.0, f"state_to_kv_ratio must be >= 0, got {ratio}"
    assert per_req_state_bytes > 0, f"per_req_state_bytes must be > 0, got {per_req_state_bytes}"
    state_budget_raw = int(available_bytes * ratio / (1.0 + ratio))
    state_max_reqs = state_budget_raw // per_req_state_bytes
    kv_budget = available_bytes - state_max_reqs * per_req_state_bytes
    return state_max_reqs, kv_budget


def _per_req_state_bytes_from_config(linear_attn_config: dict, tp_size: int) -> int:
    """Per-request recurrent + conv state bytes for a hybrid recurrent model."""
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


def _enforce_recurrent_state_server_constraints(server_args) -> None:
    """Assert both disable_radix_cache=True and disable_overlap_schedule=True
    for hybrid recurrent state models."""
    assert server_args.disable_radix_cache, (
        "Hybrid recurrent state models require --disable-radix-cache "
        "(prefix sharing is unsafe with recurrent state). Please pass "
        "--disable-radix-cache explicitly."
    )
    assert server_args.disable_overlap_schedule, (
        "Hybrid recurrent state models require --disable-overlap-schedule "
        "(this version does not support double-buffer ping-pong for recurrent state)."
    )


def _build_hybrid_pools(
    cfg,
    max_num_reqs: int,
    max_context_len: int,
    tp_size: int,
    token_to_kv_pool,
    mesh,
    dp_size: int = 1,
) -> tuple:
    """Build RecurrentStatePool + HybridReqToTokenPool + MemoryPools."""
    linear_attn_config = cfg.linear_attn_config
    rsp = RecurrentStatePool(
        linear_recurrent_layer_ids=cfg.linear_layer_ids,
        size=max_num_reqs,
        num_heads=linear_attn_config["num_heads"],
        head_dim=linear_attn_config["head_dim"],
        conv_kernel_size=linear_attn_config["short_conv_kernel_size"],
        mesh=mesh,
        dp_size=dp_size,
    )
    hybrid_pool = HybridReqToTokenPool(
        size=max_num_reqs,
        max_context_len=max_context_len,
        dtype=np.int32,
        recurrent_state_pool=rsp,
        dp_size=dp_size,
    )
    mp = MemoryPools(
        token_to_kv_pool=token_to_kv_pool,
        recurrent_state_pool=rsp,
    )
    return rsp, hybrid_pool, mp


def _build_non_hybrid_memory_pools(token_to_kv_pool) -> MemoryPools:
    """Wrap a single KV pool in MemoryPools."""
    return MemoryPools(token_to_kv_pool=token_to_kv_pool)


# ---------------------------------------------------------------------------
# Mixin class
# ---------------------------------------------------------------------------


class ModelRunnerKVCacheMixin:

    def _compute_cell_size(self: ModelRunner) -> int:
        """Per-token KV cache cost in bytes per device, summed across layers."""

        def align128(x: int) -> int:
            return (x + 127) // 128 * 128

        dtype_size = jnp.dtype(self.kv_cache_dtype).itemsize
        num_layers = self._kv_pool_layer_count()

        if self.use_mla_backend and self.server_args.attention_backend == "fa":
            cfg = self.model_config.hf_text_config
            return (
                (align128(cfg.kv_lora_rank) + align128(cfg.qk_rope_head_dim))
                * num_layers
                * dtype_size
            )

        return (
            self.model_config.get_num_kv_heads(self.attention_tp_size)
            * align128(self.model_config.head_dim)
            * 2
            * num_layers
            * dtype_size
        )

    def _profile_available_bytes(self: ModelRunner, total_device_memory: int) -> int:
        """Profile available bytes for KV cache (+ recurrent state)."""
        available_device_memory = self.get_available_device_memory()
        rest_memory = available_device_memory - total_device_memory * (1 - self.mem_fraction_static)
        if rest_memory <= 0:
            raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")

        if self.linear_recurrent_config is not None:
            rest_memory = self.handle_recurrent_cache(int(rest_memory))

        return int(rest_memory)

    def handle_recurrent_cache(self: ModelRunner, total_rest_memory: int) -> int:
        """Split HBM between recurrent state and KV cache.

        Aligns with upstream handle_max_mamba_cache.
        """
        cfg = self.linear_recurrent_config
        ratio = self.server_args.state_to_kv_ratio
        if ratio <= 0:
            raise ValueError(
                f"state_to_kv_ratio={ratio} <= 0 is invalid for hybrid recurrent model; "
                f"set --state-to-kv-ratio > 0 (default 0.9)"
            )

        per_req_state = _per_req_state_bytes_from_config(
            cfg.linear_attn_config, self.attention_tp_size
        )
        state_max_reqs, kv_budget = _split_state_kv_budget(total_rest_memory, ratio, per_req_state)
        self._recurrent_state_max_reqs = state_max_reqs

        logger.info(
            "Hybrid recurrent budget split: per_req_state=%d bytes, "
            "state_max_reqs=%d, kv_budget=%.1f GB",
            per_req_state,
            state_max_reqs,
            kv_budget / (1024**3),
        )

        return kv_budget

    def profile_max_num_token(self: ModelRunner, total_device_memory: int) -> int:
        """Profile the maximum number of tokens that can fit in memory."""
        available_kv_cache_bytes = self._profile_available_bytes(total_device_memory)

        cell_size = self._compute_cell_size()
        max_tokens = max(1, int(available_kv_cache_bytes // cell_size))

        logger.info(
            "TPU Memory profiling: available_kv_cache=%.1fGB, max_tokens=%d, cell_size=%d bytes",
            available_kv_cache_bytes / (1024**3),
            max_tokens,
            cell_size,
        )

        return max_tokens

    def _init_kv_cache_dtype(self: ModelRunner):
        """Resolve kv_cache_dtype from server_args."""
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "bf16":
            self.kv_cache_dtype = jnp.bfloat16
        else:
            raise ValueError(f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}.")
        logger.info("ModelRunner kv_cache_dtype: %s", self.kv_cache_dtype)

    def _apply_token_constraints(
        self: ModelRunner,
        token_capacity: int,
        max_total_tokens: int | None,
        dp_size: int,
    ) -> int:
        """Apply external constraints to token capacity."""
        # CI override
        ci_size = os.environ.get("SGLANG_CI_SMALL_KV_SIZE")
        if ci_size:
            token_capacity = int(ci_size)

        # User cap
        if max_total_tokens is not None:
            if max_total_tokens > token_capacity:
                logger.warning(
                    "max_total_tokens=%s is larger than the profiled value %s. "
                    "Use the profiled value instead.",
                    max_total_tokens,
                    token_capacity,
                )
            token_capacity = min(token_capacity, max_total_tokens)

        # Page alignment
        token_capacity = token_capacity // self.server_args.page_size * self.server_args.page_size

        # DP scale
        token_capacity = token_capacity * dp_size
        logger.info(
            "ModelRunner per dp max_total_num_tokens after dp_size %s: %s",
            dp_size,
            token_capacity,
        )

        return token_capacity

    def _resolve_max_num_reqs(self: ModelRunner, max_num_reqs: int | None) -> int:
        """Compute max concurrent requests."""
        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(self.max_total_num_tokens / self.model_config.context_len * 512),
                    2048,
                ),
                4096,
            )

        if (
            self.is_draft_worker
            and self.spec_algorithm is not None
            and not self.spec_algorithm.is_none()
        ):
            max_num_reqs = self.server_args.max_num_reqs

        # Cap by recurrent state budget
        if hasattr(self, "_recurrent_state_max_reqs"):
            max_num_reqs = min(max_num_reqs, self._recurrent_state_max_reqs)

        return max_num_reqs

    def _init_pools(self: ModelRunner, max_num_reqs: int, dp_size: int):
        """Create ReqToTokenPool, KV pool, allocator, and MemoryPools."""
        from sgl_jax.srt.mem_cache.allocator import (
            PagedTokenToKVPoolAllocator,
            SWATokenToKVPoolAllocator,
            TokenToKVPoolAllocator,
        )
        from sgl_jax.srt.mem_cache.memory_pool import (
            MHATokenToKVPool,
            ReqToTokenPool,
            SWAKVPool,
        )

        has_recurrent_state = self.linear_recurrent_config is not None

        # --- ReqToTokenPool ---
        if self.req_to_token_pool is None:
            if has_recurrent_state:
                _, self.req_to_token_pool, _ = _build_hybrid_pools(
                    cfg=self.linear_recurrent_config,
                    max_num_reqs=max_num_reqs,
                    max_context_len=self.model_config.context_len + 4,
                    tp_size=self.attention_tp_size,
                    token_to_kv_pool=None,
                    mesh=self.mesh,
                    dp_size=dp_size,
                )
            else:
                self.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len + 4,
                    dtype=np.int32,
                )

        # --- KV pool ---
        if self.is_hybrid:
            swa_num_kv_heads = getattr(self.model_config.hf_config, "swa_num_key_value_heads", None)
            if swa_num_kv_heads is not None:
                swa_head_num = max(swa_num_kv_heads, self.attention_tp_size)
            else:
                swa_head_num = None
            self.token_to_kv_pool = SWAKVPool(
                size=self.full_max_total_num_tokens,
                size_swa=self.swa_max_total_num_tokens,
                page_size=self.page_size,
                swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                token_to_kv_pool_class=MHATokenToKVPool,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_total_num_kv_heads_with_replication(
                    self.attention_tp_size
                ),
                head_dim=(self.model_config.head_dim + 127) // 128 * 128,
                swa_head_num=swa_head_num,
                mesh=self.mesh,
                dp_size=dp_size,
            )
        elif self.use_mla_backend and self.server_args.attention_backend == "fa":
            from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool

            hf_text_config = self.model_config.hf_text_config
            kv_lora_rank = getattr(hf_text_config, "kv_lora_rank", None)
            qk_rope_head_dim = getattr(hf_text_config, "qk_rope_head_dim", None)
            if kv_lora_rank is None or qk_rope_head_dim is None:
                raise ValueError(
                    "MLA pool requires kv_lora_rank and qk_rope_head_dim on the "
                    "model config; got "
                    f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}."
                )
            self.token_to_kv_pool = MLATokenToKVPool(
                size=self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                layer_num=self._kv_pool_layer_count(),
                mesh=self.mesh,
                dp_size=dp_size,
            )
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                size=self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_total_num_kv_heads_with_replication(
                    self.attention_tp_size
                ),
                head_dim=(self.model_config.head_dim + 127) // 128 * 128,
                layer_num=self._kv_pool_layer_count(),
                mesh=self.mesh,
                dp_size=dp_size,
            )

        # --- Allocator ---
        if self.token_to_kv_pool_allocator is None:
            if self.is_hybrid:
                self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                    self.full_max_total_num_tokens,
                    self.swa_max_total_num_tokens,
                    kvcache=self.token_to_kv_pool,
                    page_size=self.page_size,
                    dp_size=dp_size,
                )
            elif self.page_size == 1:
                self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    kvcache=self.token_to_kv_pool,
                    dp_size=dp_size,
                )
            else:
                self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    kvcache=self.token_to_kv_pool,
                    debug_mode=False,
                    dp_size=dp_size,
                )

        # --- MemoryPools wrapper ---
        if has_recurrent_state:
            rsp = self.req_to_token_pool.recurrent_state_pool
            self.memory_pools = MemoryPools(
                token_to_kv_pool=self.token_to_kv_pool,
                recurrent_state_pool=rsp,
            )
        else:
            self.memory_pools = _build_non_hybrid_memory_pools(self.token_to_kv_pool)

    def init_memory_pool(
        self: ModelRunner,
        max_num_reqs: int | None = None,
        max_total_tokens: int | None = None,
        total_device_memory: int | None = None,
        dp_size: int = 1,
    ):
        """Initialize memory pool for KV cache (+ recurrent state if hybrid)."""
        # 1. kv_cache_dtype
        self._init_kv_cache_dtype()

        # 2. Enforce constraints for hybrid recurrent
        if self.linear_recurrent_config is not None:
            _enforce_recurrent_state_server_constraints(self.server_args)

        # 3. Profile max tokens
        self.max_total_num_tokens = self.profile_max_num_token(total_device_memory)

        # 4. Resolve max_num_reqs (needed for spec dec headroom)
        max_num_reqs = self._resolve_max_num_reqs(max_num_reqs)

        # 5. Speculative decoding headroom (needs max_num_reqs)
        if self.spec_algorithm is not None and not self.spec_algorithm.is_none():
            if self.is_draft_worker:
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                max_num_reqs = self.server_args.max_num_reqs
            else:
                self.server_args.draft_runner_cache_size = (
                    self.max_total_num_tokens
                    + max_num_reqs
                    * self.server_args.speculative_num_steps
                    * self.server_args.speculative_eagle_topk
                    + max_num_reqs * self.server_args.speculative_num_draft_tokens
                    + 100
                )
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
                self.server_args.max_num_reqs = max_num_reqs

        # 6. Apply constraints (CI, user cap, page align, dp)
        self.max_total_num_tokens = self._apply_token_constraints(
            self.max_total_num_tokens, max_total_tokens, dp_size
        )

        # 7. Hybrid SWA token split (existing logic, not moved)
        if self.is_hybrid:
            self.set_num_token_hybrid()

        if self.max_total_num_tokens <= 0:
            raise RuntimeError("Not enough memory. Please try to increase --mem-fraction-static.")

        logger.info("ModelRunner final max_total_num_tokens: %s", self.max_total_num_tokens)

        # 8. Create pools
        self._init_pools(max_num_reqs, dp_size)

        # 9. SWA index mapping on attention backend
        if self.is_hybrid and hasattr(self.token_to_kv_pool_allocator, "full_to_swa_index_mapping"):
            object.__setattr__(
                self.attn_backend,
                "swa_index_mapping",
                self.token_to_kv_pool_allocator.full_to_swa_index_mapping,
            )

    # ── Properties ──

    @property
    def linear_recurrent_config(self: ModelRunner):
        """Return linear recurrent config if the model has linear attention, else None.

        Currently returns None unconditionally — KimiLinearConfig detection
        will be wired up when the modeling layer lands.
        """
        return None

    def _kv_pool_layer_count(self: ModelRunner):
        """Layer count for KV pool sizing.

        For hybrid recurrent models, only full-attention layers need KV cache.
        """
        cfg = self.linear_recurrent_config
        if cfg is not None:
            return len(cfg.full_attention_layer_ids)
        return self.adjust_layer_num()
