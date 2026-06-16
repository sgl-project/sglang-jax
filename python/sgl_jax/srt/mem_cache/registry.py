from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class TreeCacheBuildContext:
    server_args: ServerArgs
    params: CacheInitParams
    is_hybrid_swa: bool
    disable_radix_cache: bool
    effective_chunked_prefill_size: int | None
    model_config: ModelConfig
    tp_size: int
    is_hybrid_recurrent: bool = False


def default_radix_cache_factory(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    params = ctx.params

    if ctx.is_hybrid_swa:
        if ctx.disable_radix_cache:
            from sgl_jax.srt.mem_cache.chunk_cache import SWAChunkCache

            return SWAChunkCache(
                req_to_token_pool=params.req_to_token_pool,
                token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
                page_size=params.page_size,
                sliding_window_size=params.sliding_window_size,
            )

        from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache

        return SWARadixCache(
            req_to_token_pool=params.req_to_token_pool,
            token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
            sliding_window_size=params.sliding_window_size,
            page_size=params.page_size,
            disable=False,
        )

    if ctx.effective_chunked_prefill_size is not None and ctx.disable_radix_cache:
        from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache

        return ChunkCache(
            req_to_token_pool=params.req_to_token_pool,
            token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
            page_size=params.page_size,
        )

    if (
        ctx.is_hybrid_recurrent
        and ctx.server_args.enable_unified_radix_tree
        and not ctx.disable_radix_cache
    ):
        from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType
        from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        return UnifiedRadixCache(
            req_to_token_pool=params.req_to_token_pool,
            token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
            page_size=params.page_size,
            disable=False,
            kv_head_num=ctx.model_config.get_num_kv_heads(ctx.tp_size),
            head_dim=ctx.model_config.head_dim,
            layer_num=ctx.model_config.num_hidden_layers,
            max_seq_len=ctx.server_args.max_seq_len,
            is_eagle=params.is_eagle,
            tree_components=(ComponentType.FULL, ComponentType.RECURRENT),
            enable_mamba_extra_buffer=params.enable_mamba_extra_buffer,
        )

    if (
        ctx.server_args.enable_unified_radix_tree
        and not ctx.is_hybrid_swa
        and not ctx.is_hybrid_recurrent
        and not ctx.disable_radix_cache
    ):
        from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType
        from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        return UnifiedRadixCache(
            req_to_token_pool=params.req_to_token_pool,
            token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
            page_size=params.page_size,
            disable=False,
            kv_head_num=ctx.model_config.get_num_kv_heads(ctx.tp_size),
            head_dim=ctx.model_config.head_dim,
            layer_num=ctx.model_config.num_hidden_layers,
            max_seq_len=ctx.server_args.max_seq_len,
            is_eagle=params.is_eagle,
            tree_components=(ComponentType.FULL,),
        )

    from sgl_jax.srt.mem_cache.radix_cache import RadixCache

    return RadixCache(
        req_to_token_pool=params.req_to_token_pool,
        token_to_kv_pool_allocator=params.token_to_kv_pool_allocator,
        page_size=params.page_size,
        disable=ctx.disable_radix_cache,
        kv_head_num=ctx.model_config.get_num_kv_heads(ctx.tp_size),
        head_dim=ctx.model_config.head_dim,
        layer_num=ctx.model_config.num_hidden_layers,
        max_seq_len=ctx.server_args.max_seq_len,
        is_eagle=params.is_eagle,
    )


def create_tree_cache(ctx: TreeCacheBuildContext) -> BasePrefixCache:
    cache = default_radix_cache_factory(ctx)

    logger.info(
        "Tree cache initialized: impl=%s hybrid_swa=%s hybrid_recurrent=%s",
        type(cache).__name__,
        ctx.is_hybrid_swa,
        ctx.is_hybrid_recurrent,
    )
    return cache
