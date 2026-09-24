"""Hybrid L2 control-plane tests; CPU copies do not validate TPU transport."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.kv_cache_builder import init_hicache
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType as CT
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class CopyFuture:
    def __init__(self, copy):
        self.copy = copy
        self.ready = False

    def IsReady(self):
        return self.ready

    def Await(self):
        if not self.ready:
            self.copy()
            self.ready = True


class CopyEngine:
    """Fake native completion copies CPU arrays; it is not a TPU DMA test."""

    def __init__(self, pool, rank, dp_size):
        self.pool, self.rank, self.dp_size = pool, rank, dp_size
        self.host = {}
        self.next = 0

    def d2h_auto_allocate(self, pages):
        handles = list(range(self.next, self.next + len(pages)))
        self.next += len(pages)

        def copy():
            stride = self.pool.kv_buffer[0].shape[0] // self.dp_size
            for handle, page in zip(handles, pages):
                self.host[handle] = [
                    np.asarray(b)[self.rank * stride + page].copy() for b in self.pool.kv_buffer
                ]

        return handles, CopyFuture(copy)

    def h2d(self, handles, pages):
        def copy():
            stride = self.pool.kv_buffer[0].shape[0] // self.dp_size
            for layer, buffer in enumerate(self.pool.kv_buffer):
                data = np.asarray(buffer).copy()
                for handle, page in zip(handles, pages):
                    data[self.rank * stride + page] = self.host[handle][layer]
                self.pool.kv_buffer[layer] = jax.device_put(data, buffer.sharding)

        return CopyFuture(copy)

    def unlock_blocks(self, handles):
        for handle in handles:
            del self.host[handle]


def fake_raiden(pool, num_pages, dp_size):
    from sgl_jax.srt.mem_cache.raiden_hicache import (
        RaidenHiCacheController,
        RaidenHostKVPool,
    )

    host = RaidenHostKVPool(
        {r: CopyEngine(pool, r, dp_size) for r in range(dp_size)},
        num_pages // dp_size,
        pool.kv_buffer[0].shape[0] // dp_size,
    )
    return host, RaidenHiCacheController(host, pool)


def make_cache(
    page=1,
    window=8,
    policy="write_through",
    full_pages=32,
    swa_pages=16,
    backend="jax",
    dp_size=1,
):
    mesh = Mesh(
        np.asarray(jax.devices()[:dp_size]).reshape(dp_size, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    jax.sharding.set_mesh(mesh)
    pool = SWAKVPool(
        size=full_pages * page,
        size_swa=swa_pages * page,
        page_size=page,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1, 2],
        token_to_kv_pool_class=MHATokenToKVPool,
        dtype=jnp.float32,
        head_num=2,
        head_dim=2,
        swa_head_num=1,
        mesh=mesh,
        dp_size=dp_size,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=pool.size,
        size_swa=pool.size_swa,
        kvcache=pool,
        page_size=page,
        dp_size=dp_size,
    )
    req_pool = ReqToTokenPool(size=8, max_context_len=full_pages * page, dtype=np.int32)
    params = CacheInitParams(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page,
        sliding_window_size=window,
    )
    cache = UnifiedRadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page,
        tree_components=(CT.FULL, CT.SWA),
        component_init_params=params,
    )
    args = SimpleNamespace(
        hicache_transfer_backend=backend,
        hicache_ratio=2,
        hicache_write_policy=policy,
        hicache_write_through_threshold=1,
    )
    if backend == "raiden":
        with patch("sgl_jax.srt.mem_cache.raiden_hicache.create_raiden_hicache", fake_raiden):
            init_hicache(cache, args, mesh, allocator)
    else:
        init_hicache(cache, args, mesh, allocator)
    return cache, allocator, pool


def key(tokens, rank=0):
    return RadixKey(token_ids=list(tokens), extra_key=None, dp_rank=rank)


def insert(cache, allocator, tokens, rank=0):
    indices = allocator.alloc(len(tokens), dp_rank=rank)
    assert indices is not None
    cache.insert(InsertParams(key=key(tokens, rank), value=indices))
    return (
        indices,
        cache.match_prefix(MatchPrefixParams(key=key(tokens, rank))).last_device_node,
    )


def settle(cache):
    for controller in cache.hicache_controllers.values():
        controller.drain_pending()
    cache.check_hicache_events()


def shutdown(cache):
    for controller in cache.hicache_controllers.values():
        controller.shutdown()


@pytest.mark.parametrize("backend", ["jax", "raiden"])
def test_component_pools_and_swa_only_restore_preserve_full_addresses(backend):
    cache, allocator, pool = make_cache(backend=backend)
    try:
        full, node = insert(cache, allocator, range(8))
        settle(cache)
        old_swa = node.component_data[CT.SWA].value.copy()
        assert cache.host_pools[CT.FULL] is not cache.host_pools[CT.SWA]
        assert len(node.component_data[CT.SWA].host_value) == 8
        before_full = allocator.full_available_size()
        cache.components[CT.SWA].evict_component(node)
        cache._update_aux_evictable_node_sets(node)
        result = cache.match_prefix(MatchPrefixParams(key=key(range(8))))
        assert result.host_hit_length == 8
        assert result.swa_host_hit_length == 8
        restored, last, plan = cache.init_load_back(result.last_host_node, result.host_hit_length)
        cache.finish_load_back(plan)
        np.testing.assert_array_equal(restored, full)
        np.testing.assert_array_equal(node.component_data[CT.FULL].value, full)
        assert allocator.full_available_size() == before_full
        assert last is node
        np.testing.assert_array_equal(
            allocator.translate_full_to_swa(full, dp_rank=0),
            node.component_data[CT.SWA].value,
        )
        assert len(old_swa) == len(node.component_data[CT.SWA].value)
    finally:
        shutdown(cache)


@pytest.mark.parametrize("policy", ["write_through", "write_back"])
@pytest.mark.parametrize("backend", ["jax", "raiden"])
def test_dual_restore_after_overwriting_device_slots(policy, backend):
    cache, allocator, pool = make_cache(policy=policy, backend=backend)
    try:
        for component, subpool in [
            (CT.FULL, pool.full_kv_pool),
            (CT.SWA, pool.swa_kv_pool),
        ]:
            for layer, buffer in enumerate(subpool.kv_buffer):
                subpool.kv_buffer[layer] = jnp.full_like(buffer, 20 + int(component) * 10 + layer)
        full, node = insert(cache, allocator, range(8))
        if policy == "write_through":
            settle(cache)
        cache.evict(EvictParams(num_tokens=8, dp_rank=0))
        settle(cache)
        assert node.evicted
        for subpool in [pool.full_kv_pool, pool.swa_kv_pool]:
            subpool.kv_buffer = [jnp.zeros_like(buffer) for buffer in subpool.kv_buffer]
        match = cache.match_prefix(MatchPrefixParams(key=key(range(8))))
        assert match.host_hit_length == 8
        restored, _, plan = cache.init_load_back(match.last_host_node, match.host_hit_length)
        cache.finish_load_back(plan)
        assert len(restored) == 8
        for component, subpool in [
            (CT.FULL, pool.full_kv_pool),
            (CT.SWA, pool.swa_kv_pool),
        ]:
            indices = node.component_data[component].value
            for layer, buffer in enumerate(subpool.kv_buffer):
                np.testing.assert_array_equal(
                    np.asarray(buffer)[indices], 20 + int(component) * 10 + layer
                )
    finally:
        shutdown(cache)


@pytest.mark.parametrize("backend", ["jax", "raiden"])
@pytest.mark.parametrize("policy", ["write_through", "write_back"])
@pytest.mark.parametrize("page,window", [(128, 1024), (256, 128)])
def test_paged_dp2_component_contents_and_rank_isolation(backend, policy, page, window):
    cache, allocator, pool = make_cache(
        page=page, window=window, backend=backend, policy=policy, dp_size=2
    )
    raw_swa = allocator.alloc_swa(page, dp_rank=1)
    try:
        for ct, subpool in [(CT.FULL, pool.full_kv_pool), (CT.SWA, pool.swa_kv_pool)]:
            for layer, buffer in enumerate(subpool.kv_buffer):
                stride = buffer.shape[0] // 2
                data = np.zeros(buffer.shape, dtype=np.float32)
                data[:stride] = 10 + int(ct) * 10 + layer
                data[stride:] = 50 + int(ct) * 10 + layer
                subpool.kv_buffer[layer] = jax.device_put(data, buffer.sharding)
        _, quiet = insert(cache, allocator, range(page * 2), rank=0)
        full, node = insert(cache, allocator, range(page * 2), rank=1)
        settle(cache)
        quiet_state = (
            allocator.full_available_size(0),
            allocator.swa_available_size(0),
            tuple(p.available_size(0) for p in cache.host_pools.values()),
        )
        quiet_buffers = [
            np.asarray(b)[: b.shape[0] // 2].copy()
            for p in [pool.full_kv_pool, pool.swa_kv_pool]
            for b in p.kv_buffer
        ]
        cache.evict(EvictParams(num_tokens=page * 2, dp_rank=1))
        settle(cache)
        assert not quiet.evicted
        for subpool in [pool.full_kv_pool, pool.swa_kv_pool]:
            for layer, buffer in enumerate(subpool.kv_buffer):
                data = np.asarray(buffer).copy()
                data[buffer.shape[0] // 2 :] = -99
                subpool.kv_buffer[layer] = jax.device_put(data, buffer.sharding)
        match = cache.match_prefix(MatchPrefixParams(key=key(range(page * 2), 1)))
        assert match.host_hit_length == page * 2
        restored, _, plan = cache.init_load_back(match.last_host_node, match.host_hit_length)
        cache.finish_load_back(plan)
        assert len(restored) == page * 2
        path = []
        current = node
        while current is not cache.root_node:
            path.append(current)
            current = current.parent
        for ct, subpool in [(CT.FULL, pool.full_kv_pool), (CT.SWA, pool.swa_kv_pool)]:
            for restored_node in path:
                indices = restored_node.component_data[ct].value
                if indices is None:
                    assert ct == CT.SWA
                    continue
                pages = indices[::page] // page
                for layer, buffer in enumerate(subpool.kv_buffer):
                    stride = buffer.shape[0] // 2
                    np.testing.assert_array_equal(
                        np.asarray(buffer)[stride + pages], 50 + int(ct) * 10 + layer
                    )
                if ct == CT.SWA:
                    np.testing.assert_array_equal(
                        allocator.translate_full_to_swa(
                            restored_node.component_data[CT.FULL].value, dp_rank=1
                        ),
                        indices,
                    )
        assert quiet_state == (
            allocator.full_available_size(0),
            allocator.swa_available_size(0),
            tuple(p.available_size(0) for p in cache.host_pools.values()),
        )
        for original, buffer in zip(
            quiet_buffers,
            [b for p in [pool.full_kv_pool, pool.swa_kv_pool] for b in p.kv_buffer],
        ):
            np.testing.assert_array_equal(original, np.asarray(buffer)[: buffer.shape[0] // 2])
    finally:
        allocator.free_swa_indices(raw_swa, dp_rank=1)
        shutdown(cache)


@pytest.mark.parametrize("backend", ["jax", "raiden"])
@pytest.mark.parametrize("terminal", ["finish", "retract", "abort"])
def test_restored_request_ownership_through_chunk_and_release(backend, terminal):
    from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
    from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )
    from sgl_jax.srt.mem_cache.common import release_kv_cache
    from sgl_jax.srt.sampling.sampling_params import SamplingParams

    cache, allocator, _ = make_cache(backend=backend, policy="write_back")
    try:
        _, node = insert(cache, allocator, range(8))
        cache.evict(EvictParams(num_tokens=8, dp_rank=0))
        settle(cache)
        match = cache.match_prefix(MatchPrefixParams(key=key(range(8))))
        restored, last, plan = cache.init_load_back(match.last_host_node, match.host_hit_length)
        cache.finish_load_back(plan)
        req = Req("lifecycle", "", list(range(12)), SamplingParams(max_new_tokens=1), dp_rank=0)
        req.fill_ids = list(req.origin_input_ids)
        assert cache.req_to_token_pool.alloc([req]) is not None
        tail = allocator.alloc(4)
        row = np.concatenate([restored, tail])
        cache.req_to_token_pool.write((req.req_pool_idx, slice(0, 12)), row)
        req.prefix_indices = restored
        req.last_node = last
        req.last_matched_prefix_len = req.cache_protected_len = 8
        req.kv_committed_len = req.kv_allocated_len = 12
        receipt = cache.inc_lock_ref(last)
        req.cache_lock_params = receipt.to_dec_params()
        req.swa_uuid_for_lock = receipt.swa_uuid_for_lock
        if terminal == "finish":
            cache.cache_unfinished_req(req)
            assert req.cache_protected_len == 12
            release_kv_cache(req, cache)
        elif terminal == "retract":
            batch = SimpleNamespace(
                reqs_info=[SimpleNamespace(reqs=[req])],
                tree_cache=cache,
                _evict_tree_cache_if_needed=lambda _: None,
            )
            ScheduleBatch.release_req(batch, 0, 0, 0, SimpleNamespace())
            assert req.is_retracted and req.req_pool_idx is None
            req.init_next_round_input(cache)
            assert len(req.prefix_indices) > 0
        else:
            req.finished_reason = FINISH_ABORT("test cancellation")
            scheduler = SimpleNamespace(
                chunked_reqs=[req],
                _pending_chunked_abort_reqs=[req],
                _release_prefill_host_buffer=lambda _: None,
                tree_cache=cache,
                spec_algorithm=None,
            )
            SchedulerOutputProcessorMixin._finalize_chunked_abort(scheduler, req, 0)
        settle(cache)
        assert req.req_pool_idx is None
        assert req.cache_lock_params is None
        assert cache.full_protected_size() == cache.swa_protected_size() == 0
        assert allocator.full_available_size() + cache.full_evictable_size() == allocator.size_full
        assert allocator.swa_available_size() + cache.swa_evictable_size() == allocator.size_swa
        assert np.count_nonzero(allocator.full_to_swa_index_mapping) == cache.swa_evictable_size()
        host_capacities = {ct: pool.total_size() for ct, pool in cache.host_pools.items()}
        cache.reset()
        assert all(
            pool.available_size() == host_capacities[ct] for ct, pool in cache.host_pools.items()
        )
    finally:
        shutdown(cache)
