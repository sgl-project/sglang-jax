from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
    from sgl_jax.srt.server_args import ServerArgs
    from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


def build_kv_cache(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    page_size: int,
    is_hybrid: bool,
    is_hybrid_recurrent: bool = False,
    sliding_window_size: int | None,
    tp_size: int,
    spec_algorithm: SpeculativeAlgorithm | None,
) -> BasePrefixCache:
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        is_eagle=spec_algorithm is not None and spec_algorithm.is_eagle(),
        sliding_window_size=sliding_window_size,
        enable_mamba_extra_buffer=server_args.enable_mamba_extra_buffer,
    )

    return create_tree_cache(
        TreeCacheBuildContext(
            server_args=server_args,
            params=params,
            is_hybrid_swa=is_hybrid,
            is_hybrid_recurrent=is_hybrid_recurrent,
            disable_radix_cache=server_args.disable_radix_cache,
            effective_chunked_prefill_size=server_args.chunked_prefill_size,
            model_config=model_config,
            tp_size=tp_size,
        )
    )
