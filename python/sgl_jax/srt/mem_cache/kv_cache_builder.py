from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sgl_jax.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.registry import (
    TreeCacheBuildContext,
    create_tree_cache,
    validate_unified_hybrid_swa_route,
)

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
        enable_recurrent_extra_buffer=server_args.enable_recurrent_extra_buffer,
        recurrent_track_interval=server_args.recurrent_track_interval,
    )

    ctx = TreeCacheBuildContext(
        server_args=server_args,
        params=params,
        is_hybrid_swa=is_hybrid,
        is_hybrid_recurrent=is_hybrid_recurrent,
        disable_radix_cache=server_args.disable_radix_cache,
        effective_chunked_prefill_size=server_args.chunked_prefill_size,
        model_config=model_config,
        tp_size=tp_size,
        has_speculative=(spec_algorithm is not None and not spec_algorithm.is_none()),
    )
    validate_unified_hybrid_swa_route(ctx)
    cache = create_tree_cache(ctx)

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

    from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, SWAKVPool
    from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType

    device_pool = token_to_kv_pool_allocator.get_kvcache()
    hybrid = cache.tree_components == (ComponentType.FULL, ComponentType.SWA)
    if hybrid:
        if not isinstance(device_pool, SWAKVPool):
            raise ValueError("FULL+SWA HiCache requires SWAKVPool")
        device_pools = {
            ComponentType.FULL: device_pool.full_kv_pool,
            ComponentType.SWA: device_pool.swa_kv_pool,
        }
    elif cache.tree_components == (ComponentType.FULL,):
        device_pools = {ComponentType.FULL: device_pool}
    else:
        raise ValueError("HiCache supports FULL or FULL+SWA components only")

    backend = getattr(server_args, "hicache_transfer_backend", "jax")
    if backend == "raiden" and any(type(p) is not MHATokenToKVPool for p in device_pools.values()):
        raise ValueError("Raiden HiCache supports MHA KV pools only (no recurrent/MLA)")
    host_pools, controllers = {}, {}
    try:
        for component, pool in device_pools.items():
            num_pages = int(server_args.hicache_ratio * pool.size) // pool.page_size
            if backend == "raiden":
                from sgl_jax.srt.mem_cache.raiden_hicache import create_raiden_hicache

                host_pool, controller = create_raiden_hicache(
                    pool, num_pages, token_to_kv_pool_allocator.dp_size
                )
            else:
                host_pool = LRUHostKVPool(
                    device_pool=pool,
                    pool_size=num_pages,
                    page_size=pool.page_size,
                    layer_num=pool.layer_num,
                    per_layer_shape=tuple(int(d) for d in pool.kv_buffer[0].shape[1:]),
                    dtype=pool.dtype,
                    mesh=mesh,
                    partition_spec=pool.kv_sharding.spec,
                    **(
                        {
                            "dp_size": token_to_kv_pool_allocator.dp_size,
                            "pool_name": f"hicache_{component}",
                        }
                        if hybrid
                        else {}
                    ),
                )
                controller = HiCacheController(host_pool, pool)
            host_pools[component] = host_pool
            controllers[component] = controller
    except Exception:
        for controller in controllers.values():
            controller.shutdown()
        raise

    cache.host_pools = host_pools
    cache.hicache_controllers = controllers
    cache.host_pool = host_pools[ComponentType.FULL]
    cache.hicache_controller = controllers[ComponentType.FULL]
    cache.hicache_enabled = True
    cache.write_through_threshold = server_args.hicache_write_through_threshold
    cache.write_policy = server_args.hicache_write_policy
    logger.info(
        "HiCache enabled: backend=%s, component host pages=%s, page_size=%d, write_policy=%s",
        backend,
        {str(ct): p.total_size() for ct, p in host_pools.items()},
        device_pool.page_size,
        cache.write_policy,
    )
