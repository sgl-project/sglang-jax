from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache

if TYPE_CHECKING:
    from jax.sharding import Mesh

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
    mesh: Mesh | None = None,
) -> BasePrefixCache:
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        is_eagle=spec_algorithm is not None and spec_algorithm.is_eagle(),
        sliding_window_size=sliding_window_size,
    )

    cache = create_tree_cache(
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

    if server_args.hicache_storage != "disable":
        init_hicache(cache, server_args, mesh, token_to_kv_pool_allocator)

    return cache


def init_hicache(cache, server_args, mesh, token_to_kv_pool_allocator) -> None:
    """Assemble the HiCache L2 stack and attach it to a UnifiedRadixCache.

    This is the single swap point for storage backends — the controller and
    tree cache only pass int buffer_ids across the boundary.
    """
    from sgl_jax.srt.mem_cache.hicache_controller import HiCacheController
    from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool
    from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if not isinstance(cache, UnifiedRadixCache):
        raise TypeError(f"HiCache requires UnifiedRadixCache, got {type(cache).__name__}")
    if mesh is None:
        raise ValueError("HiCache needs a mesh to build the host pool sharding")

    device_pool = token_to_kv_pool_allocator.get_kvcache()
    per_layer_shape = tuple(int(d) for d in device_pool.kv_buffer[0].shape[1:])
    page_size = device_pool.page_size
    # hicache_ratio is token-based; fold to page count for the page-addressed pool.
    host_token_budget = int(server_args.hicache_ratio * device_pool.size)
    num_pages = host_token_budget // page_size

    host_pool = LRUHostKVPool(
        device_pool=device_pool,
        pool_size=num_pages,
        page_size=page_size,
        layer_num=device_pool.layer_num,
        per_layer_shape=per_layer_shape,
        dtype=device_pool.dtype,
        mesh=mesh,
        partition_spec=device_pool.kv_sharding.spec,
    )
    controller = HiCacheController(host_pool, device_pool)

    cache.host_pool = host_pool
    cache.hicache_controller = controller
    cache.hicache_enabled = True
    cache.write_through_threshold = server_args.hicache_write_through_threshold
    cache.write_policy = server_args.hicache_write_policy
    for component in cache._components_tuple:
        component._full_kv_pool_host = host_pool

    logger.info(
        "HiCache enabled: host pool=%d pages (page_size=%d, ~%d tokens, "
        "ratio=%.1f x device size=%d), write_through_threshold=%d",
        num_pages,
        page_size,
        num_pages * page_size,
        server_args.hicache_ratio,
        device_pool.size,
        server_args.hicache_write_through_threshold,
    )
