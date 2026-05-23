"""Builder for the scheduler's prefix tree cache.

Extracted from ``Scheduler.init_memory_pool_and_cache`` so that future
cache variants (HiCache wrapper, PD-aware caches, etc.) can plug into a
single dispatch point instead of growing the scheduler if/elif.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.radix_cache import RadixCache
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
    from sgl_jax.srt.server_args import ServerArgs


def build_kv_cache(
    *,
    req_to_token_pool: "ReqToTokenPool",
    token_to_kv_pool_allocator: Any,
    server_args: "ServerArgs",
    model_config: "ModelConfig",
    page_size: int,
    tp_size: int,
    is_hybrid: bool,
    sliding_window_size: int | None,
    spec_algorithm: Any,
) -> BasePrefixCache:
    """Construct the scheduler's prefix-tree cache.

    Dispatch precedence (matches the pre-refactor scheduler logic):

    1. ``is_hybrid`` → :class:`SWARadixCache`
    2. ``chunked_prefill_size is not None and disable_radix_cache`` →
       :class:`ChunkCache`
    3. otherwise → :class:`RadixCache`
    """
    if is_hybrid:
        return SWARadixCache(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            sliding_window_size=sliding_window_size,
            page_size=page_size,
            disable=server_args.disable_radix_cache,
        )

    if server_args.chunked_prefill_size is not None and server_args.disable_radix_cache:
        return ChunkCache(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=page_size,
        )

    return RadixCache(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        disable=server_args.disable_radix_cache,
        kv_head_num=model_config.get_num_kv_heads(tp_size),
        head_dim=model_config.head_dim,
        layer_num=model_config.num_hidden_layers,
        max_seq_len=server_args.max_seq_len,
        is_eagle=spec_algorithm is not None and spec_algorithm.is_eagle(),
    )
