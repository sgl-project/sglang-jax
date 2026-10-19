from __future__ import annotations

import logging
import os
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


def _vfio_dma_entry_limit() -> int | None:
    """vfio_iommu_type1.dma_entry_limit (max DMA mappings per VFIO container), or None."""
    try:
        with open("/sys/module/vfio_iommu_type1/parameters/dma_entry_limit") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _host_mem_available_bytes() -> int | None:
    """MemAvailable from /proc/meminfo, bounded by the cgroup limit if any."""
    avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        return None
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
            if raw and raw != "max":
                limit = int(raw)
                avail = limit if avail is None else min(avail, limit)
            break
        except (OSError, ValueError):
            continue
    return avail


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

    device_pool = token_to_kv_pool_allocator.get_kvcache()
    per_layer_shape = tuple(int(d) for d in device_pool.kv_buffer[0].shape[1:])
    page_size = device_pool.page_size
    # hicache_ratio is token-based; fold to page count for the page-addressed pool.
    host_token_budget = int(server_args.hicache_ratio * device_pool.size)
    num_pages = host_token_budget // page_size

    def _make_pool(n_pages: int) -> LRUHostKVPool:
        return LRUHostKVPool(
            device_pool=device_pool,
            pool_size=n_pages,
            page_size=page_size,
            layer_num=device_pool.layer_num,
            per_layer_shape=per_layer_shape,
            dtype=device_pool.dtype,
            mesh=mesh,
            partition_spec=device_pool.kv_sharding.spec,
        )

    host_pool = _make_pool(num_pages)

    # Host-RAM budget: cap the pool so its pinned footprint on THIS host stays
    # under a fraction of currently-available RAM. Pinned slots are allocated
    # lazily, so without this the pool silently grows until the TPU runtime's
    # own host allocations fail (observed: posix_memalign ENOMEM at 636 GB).
    frac = float(getattr(server_args, "hicache_host_mem_fraction", 0.5))
    avail = _host_mem_available_bytes()
    per_slot = host_pool.host_bytes_per_slot_local()
    if avail is not None and per_slot > 0:
        max_pages = int(avail * frac) // per_slot
        if max_pages < 1:
            raise RuntimeError(
                f"HiCache: one host page slot needs {per_slot / 2**20:.1f} MB pinned RAM "
                f"but only {avail / 2**30:.1f} GB host RAM is available"
            )
        if max_pages < num_pages:
            logger.warning(
                "HiCache: capping host pool from %d to %d pages so pinned host RAM stays "
                "<= %.0f%% of available (%.1f GB; %.1f MB/page/host). Lower --hicache-ratio "
                "to silence this.",
                num_pages,
                max_pages,
                frac * 100,
                avail / 2**30,
                per_slot / 2**20,
            )
            num_pages = max_pages
            host_pool = _make_pool(num_pages)

    # DMA-mapping budget: every pinned-host slot shard is a separate VFIO DMA
    # mapping, and the kernel caps mappings per VFIO container at
    # vfio_iommu_type1.dma_entry_limit (default 65535). Exceeding it fails the
    # pinned allocation mid-serving ("ioctl failed ... /dev/vfio/vfio, return
    # code: 8"), after which the TPU runtime errors and the process aborts
    # (run18: ~16.6k pages with KV+indexer groups on tpu7x). Cap the pool below
    # the limit; raise the limit (write the sysfs parameter before the server
    # starts) to lift the cap.
    dma_limit = _vfio_dma_entry_limit()
    if dma_limit is not None:
        entries_per_page = int(
            os.environ.get(
                "SGL_HICACHE_DMA_ENTRIES_PER_PAGE",
                # One mapping per buffer group per device sharing a container;
                # measured 4 on tpu7x (2 groups x 2 TensorCores per chip).
                str(2 * len(getattr(host_pool, "_groups", (None,)))),
            )
        )
        dma_pages = int(dma_limit * 0.9) // max(1, entries_per_page)
        if dma_pages < num_pages:
            logger.warning(
                "HiCache: capping host pool from %d to %d pages to stay under "
                "vfio_iommu_type1.dma_entry_limit=%d (~%d DMA mappings per page). Raise it with "
                "'echo 4194304 > /sys/module/vfio_iommu_type1/parameters/dma_entry_limit' "
                "before starting the server to lift this cap.",
                num_pages,
                dma_pages,
                dma_limit,
                entries_per_page,
            )
            num_pages = dma_pages
            host_pool = _make_pool(num_pages)
    controller = HiCacheController(host_pool, device_pool)

    cache.host_pool = host_pool
    cache.hicache_controller = controller
    cache.hicache_enabled = True
    cache.write_through_threshold = server_args.hicache_write_through_threshold
    cache.write_policy = server_args.hicache_write_policy
    # Per-component host-pool hooks deferred to Stage 5 (multi-component reuse).
    # The current Stage 2/3 HiCache control plane is centralized in
    # UnifiedRadixCache, so _full_kv_pool_host has no production readers yet.

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
