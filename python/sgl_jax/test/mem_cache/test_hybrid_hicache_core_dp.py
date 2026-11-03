"""Multi-device CPU regression for SWA lock receipts after split (CPU4 suite)."""

from concurrent.futures import Future

import jax
import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.schedule_policy import PrefillAdder
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sgl_jax.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType as CT
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.test.mem_cache.test_hybrid_hicache import (
    insert,
    key,
    make_cache,
    settle,
    shutdown,
)
from sgl_jax.test.mem_cache.test_hybrid_hicache_core import (
    check_receipt_acquired_after_split,
)


@pytest.mark.parametrize("backend", ["jax", "raiden"])
@pytest.mark.parametrize("component", [CT.FULL, CT.SWA])
def test_host_backup_evicts_only_rank_local_shortfall(backend, component):
    cache, alloc, _ = make_cache(
        window=1024,
        policy="write_back",
        full_pages=32,
        swa_pages=32,
        backend=backend,
        dp_size=2,
    )
    try:
        _, quiet = insert(cache, alloc, range(8), rank=0)
        cache.evict(EvictParams(num_tokens=8, dp_rank=0))
        settle(cache)
        nodes = []
        for start in (100, 200, 300):
            _, node = insert(cache, alloc, range(start, start + 8), rank=1)
            cache.evict(EvictParams(num_tokens=8, dp_rank=1))
            settle(cache)
            nodes.append(node)
        pool = cache.host_pools[component]
        assert pool.available_size(1) == 8
        saved = {
            (node, ct): node.component_data[ct].host_value.copy()
            for node in [quiet, *nodes]
            for ct in (CT.FULL, CT.SWA)
        }
        quiet_before = (
            alloc.full_available_size(0),
            alloc.swa_available_size(0),
            tuple(p.available_size(0) for p in cache.host_pools.values()),
        )
        _, incoming = insert(cache, alloc, range(400, 412), rank=1)
        mapping_before = [mapping.copy() for mapping in alloc.full_to_swa_index_mapping]
        assert cache._hybrid_coordinator.backup_component(incoming, component) == 12
        settle(cache)

        # Twelve new pages need only four beyond the eight already free.
        # Evict one eight-page node; keep the other reusable host prefixes.
        assert nodes[0].component_data[component].host_value is None
        for (node, ct), handles in saved.items():
            if node is not nodes[0] or ct != component:
                np.testing.assert_array_equal(node.component_data[ct].host_value, handles)
        assert len(incoming.component_data[component].host_value) == 12
        assert pool.available_size(1) == 4
        other = CT.SWA if component == CT.FULL else CT.FULL
        assert incoming.component_data[other].host_value is None
        assert cache.host_pools[other].available_size(1) == 8
        assert quiet_before == (
            alloc.full_available_size(0),
            alloc.swa_available_size(0),
            tuple(p.available_size(0) for p in cache.host_pools.values()),
        )
        np.testing.assert_array_equal(alloc.full_to_swa_index_mapping, mapping_before)
    finally:
        shutdown(cache)


@pytest.mark.parametrize("policy", ["write_through", "write_back"])
@pytest.mark.parametrize("page", [1, 128])
@pytest.mark.parametrize("mode", ["dual", "swa_only", "mixed"])
def test_jax_restore_reclaims_other_cache_after_pinning_sources(monkeypatch, policy, page, mode):
    full_pages = 32 if mode == "dual" else 40
    swa_pages = 35 if mode == "mixed" else 32
    cache, alloc, device = make_cache(
        page=page,
        window=8 * page,
        full_pages=2 * full_pages,
        swa_pages=2 * swa_pages,
        dp_size=2,
        policy=policy,
    )
    req = None
    allocated = None
    try:
        pools = {CT.FULL: device.full_kv_pool, CT.SWA: device.swa_kv_pool}
        # Distinct coordinates ensure a stale/reused slot cannot masquerade as
        # a restored page. Keep the idle rank's actual device contents intact.
        for ct, pool in pools.items():
            for layer, buffer in enumerate(pool.kv_buffer):
                data = np.arange(buffer.size, dtype=np.float32).reshape(buffer.shape)
                data += (1 + int(ct) * 3 + layer) * 100000
                pool.kv_buffer[layer] = jax.device_put(data, buffer.sharding)
        _, quiet = insert(cache, alloc, range(4 * page), rank=0)
        full, node = insert(cache, alloc, range(8 * page), rank=1)
        if policy == "write_through":
            cache.write_backup(quiet)
            cache.write_backup(node)
        settle(cache)
        fragments = [node]
        if mode == "mixed":
            fragments.insert(0, cache._split_node(node.key, node, 4 * page))
        expected = {}
        original_pages = {}
        for ct, pool in pools.items():
            for fragment in fragments:
                ids = fragment.component_data[ct].value[::page] // page
                original_pages[ct, fragment.id] = ids.copy()
                expected[ct, fragment.id] = [
                    np.asarray(b)[b.shape[0] // 2 + ids].copy() for b in pool.kv_buffer
                ]
        if mode == "dual":
            cache.evict(EvictParams(num_tokens=8 * page, dp_rank=1))
        else:
            cache.components[CT.SWA].evict_component(fragments[0])
            cache._update_aux_evictable_node_sets(fragments[0])
        settle(cache)
        # Overwrite only released target component slots, including all missing
        # SWA pages. Resident FULL and the resident half-window must stay live.
        for ct, pool in pools.items():
            for fragment in fragments:
                if fragment.component_data[ct].value is not None:
                    continue
                for layer, buffer in enumerate(pool.kv_buffer):
                    data = np.asarray(buffer).copy()
                    ids = original_pages[ct, fragment.id]
                    data[buffer.shape[0] // 2 + ids] = -99
                    pool.kv_buffer[layer] = jax.device_put(data, buffer.sharding)
        _, victim = insert(cache, alloc, range(100 * page, 128 * page), rank=1)
        if policy == "write_through":
            cache.write_backup(victim)
        settle(cache)
        assert alloc.swa_available_size(1) < (4 if mode == "mixed" else 8) * page
        assert cache.swa_evictable_size(1) >= 28 * page
        quiet_state = (
            alloc.full_available_size(0),
            alloc.swa_available_size(0),
            cache.full_evictable_size(0),
            cache.swa_evictable_size(0),
            tuple(pool.available_size(0) for pool in cache.host_pools.values()),
            alloc.full_to_swa_index_mapping[0].copy(),
        )
        quiet_buffers = [
            np.asarray(b)[: b.shape[0] // 2].copy() for p in pools.values() for b in p.kv_buffer
        ]
        selected = cache._hybrid_coordinator.selection(node)
        sources = {
            ct: [(n, n.component_data[ct].host_value.copy()) for n in nodes]
            for ct, nodes in selected.items()
        }
        barrier = []
        evictions = []
        monkeypatch.setattr(cache, "_donation_barrier", lambda: barrier.append(True))
        evict = cache.evict

        def protected_evict(params):
            assert barrier, "donated buffers must be safe before any eviction/gather"
            assert params.dp_rank == 1
            for ct, nodes in sources.items():
                for n, handles in nodes:
                    assert n.component_data[ct].host_lock_ref > 0
                    assert all(cache.host_pools[ct]._lock_ref[int(h)] > 0 for h in handles)
            for fragment in fragments:
                for ct in (CT.FULL, CT.SWA):
                    if fragment.component_data[ct].value is not None:
                        assert fragment.component_data[ct].lock_ref > 0
            # A simultaneous host-pressure pass cannot remove the actual H2D sources.
            for ct in (CT.FULL, CT.SWA):
                cache.evict_host(1000, dp_rank=1, component_type=ct)
            result = evict(params)
            for ct, nodes in sources.items():
                for n, handles in nodes:
                    np.testing.assert_array_equal(n.component_data[ct].host_value, handles)
            evictions.append(result)
            return result

        monkeypatch.setattr(cache, "evict", protected_evict)
        req = Req(
            "jax-pressure",
            "",
            list(range(8 * page + 1)),
            SamplingParams(max_new_tokens=1),
            dp_rank=1,
        )
        req.init_next_round_input(cache)
        adder = PrefillAdder(
            page_size=page,
            tree_cache=cache,
            token_to_kv_pool_allocator=alloc,
            running_batch=None,
            new_token_ratio=1,
            rem_input_tokens=100 * page,
            rem_chunk_tokens=None,
            dp_size=2,
        )
        adder.add_one_req(req)
        assert adder.can_run_list[1] == [req]
        assert len(req.prefix_indices) == 8 * page
        assert req.extend_input_len == 1
        assert evictions and evictions[0].swa_num_tokens_evicted > 0
        if mode == "dual":
            assert victim.evicted and evictions[0].num_tokens_evicted > 0
        monkeypatch.setattr(cache, "evict", evict)
        if page == 1:
            allocated = alloc_token_slots(cache, req.extend_input_len, dp_rank=1)
        else:
            allocated = alloc_paged_token_slots_extend(
                cache,
                [len(req.prefix_indices)],
                [len(req.fill_ids)],
                [int(req.prefix_indices[-1])],
                req.extend_input_len,
                dp_rank=1,
            )
        assert len(allocated) == 1
        assert not np.intersect1d(req.prefix_indices, allocated).size
        if mode != "dual":
            np.testing.assert_array_equal(req.prefix_indices, full)
        for ct, pool in pools.items():
            for fragment in fragments:
                ids = fragment.component_data[ct].value[::page] // page
                for prior, buffer in zip(expected[ct, fragment.id], pool.kv_buffer):
                    np.testing.assert_array_equal(
                        np.asarray(buffer)[buffer.shape[0] // 2 + ids], prior
                    )
        assert not quiet.evicted
        assert quiet_state[:5] == (
            alloc.full_available_size(0),
            alloc.swa_available_size(0),
            cache.full_evictable_size(0),
            cache.swa_evictable_size(0),
            tuple(pool.available_size(0) for pool in cache.host_pools.values()),
        )
        np.testing.assert_array_equal(quiet_state[5], alloc.full_to_swa_index_mapping[0])
        for prior, buffer in zip(quiet_buffers, [b for p in pools.values() for b in p.kv_buffer]):
            np.testing.assert_array_equal(prior, np.asarray(buffer)[: buffer.shape[0] // 2])
        assert all(cd.host_lock_ref == 0 for n in fragments for cd in n.component_data)
    finally:
        if allocated is not None:
            alloc.free(allocated, dp_rank=1)
        if req is not None and req.cache_lock_params is not None:
            cache.dec_lock_ref(req.last_node, req.cache_lock_params)
        shutdown(cache)


def test_rank1_page128_receipt_acquired_after_split():
    check_receipt_acquired_after_split(page=128, dp_size=2, rank=1)


def seed_reused_prefix(cache, alloc, tokens, rank=0):
    indices, node = insert(cache, alloc, tokens, rank=rank)
    # Reinsert an already tree-owned prefix as a reuse hit, without requiring
    # a second full request allocation in these deliberately small pools.
    cache.insert(InsertParams(key=key(tokens, rank), value=indices, prev_prefix_len=len(tokens)))
    settle(cache)
    return node


@pytest.mark.parametrize("backend", ["jax", "raiden"])
@pytest.mark.parametrize("rank", [0, 1])
def test_full_host_pressure_preserves_reachable_prefix_and_quiet_rank(backend, rank):
    cache, alloc, _ = make_cache(window=4, full_pages=32, swa_pages=32, backend=backend, dp_size=2)
    try:
        quiet = seed_reused_prefix(cache, alloc, range(500, 504), rank=1 - rank)
        cache.evict(EvictParams(num_tokens=4, dp_rank=1 - rank))
        quiet_handles = quiet.component_data[CT.FULL].host_value.copy()
        quiet_free = cache.host_pools[CT.FULL].available_size(1 - rank)
        tail = seed_reused_prefix(cache, alloc, range(12), rank=rank)
        path = cache._hybrid_coordinator.path(tail)
        assert [len(n.key) for n in path] == [4, 4, 4]
        cache.evict(EvictParams(num_tokens=12, dp_rank=rank))
        capacity = cache.host_pools[CT.FULL].available_size(rank)
        for i in range(capacity // 4):
            seed_reused_prefix(cache, alloc, range(100 + 10 * i, 104 + 10 * i), rank=rank)
            cache.evict(EvictParams(num_tokens=4, dp_rank=rank))
        assert cache.host_pools[CT.FULL].available_size(rank) == 0
        seed_reused_prefix(cache, alloc, range(400, 404), rank=rank)
        assert tail.component_data[CT.FULL].host_value is None
        assert all(n.component_data[CT.FULL].host_value is not None for n in path[:-1])
        match = cache.match_prefix(MatchPrefixParams(key=key(range(12), rank)))
        assert match.host_hit_length == 8
        restored, _, _ = cache.init_load_back(match.last_host_node, match.host_hit_length)
        assert len(restored) == 8
        np.testing.assert_array_equal(quiet.component_data[CT.FULL].host_value, quiet_handles)
        assert cache.host_pools[CT.FULL].available_size(1 - rank) == quiet_free
    finally:
        shutdown(cache)


@pytest.mark.parametrize("backend", ["jax", "raiden"])
def test_full_host_eviction_retains_ancestors_of_pinned_descendant(backend):
    cache, alloc, _ = make_cache(window=4, backend=backend)
    try:
        tail = seed_reused_prefix(cache, alloc, range(12))
        path = cache._hybrid_coordinator.path(tail)
        cache.evict(EvictParams(num_tokens=12))
        handles = list(map(int, tail.component_data[CT.FULL].host_value))
        cache.host_pools[CT.FULL].pin(handles)
        tail.component_data[CT.FULL].host_lock_ref += 1
        try:
            assert cache.evict_host(12, component_type=CT.FULL) == 0
            assert all(n.component_data[CT.FULL].host_value is not None for n in path)
        finally:
            tail.component_data[CT.FULL].host_lock_ref -= 1
            cache.host_pools[CT.FULL].unpin(handles)
        assert cache.evict_host(12, component_type=CT.FULL) == 12
        assert all(n.component_data[CT.FULL].host_value is None for n in path)
    finally:
        shutdown(cache)


@pytest.mark.parametrize("length", [8, 12])
def test_full_host_eviction_protects_pending_descendants(monkeypatch, length):
    cache, alloc, _ = make_cache(window=4, policy="write_back")
    try:
        _, tail = insert(cache, alloc, range(length))
        parent = cache._hybrid_coordinator.path(tail)[0]
        cache._hybrid_coordinator.backup_component(parent, CT.FULL)
        settle(cache)
        future = Future()
        monkeypatch.setattr(cache.hicache_controllers[CT.FULL], "write", lambda *args: future)
        cache._hybrid_coordinator.backup_component(tail, CT.FULL)
        try:
            assert cache.evict_host(4, component_type=CT.FULL) == 0
            assert parent.component_data[CT.FULL].host_value is not None
        finally:
            future.set_result(None)
            cache._hybrid_coordinator.settle(wait=True)
        assert cache.evict_host(4, component_type=CT.FULL) == 4
        assert parent.component_data[CT.FULL].host_value is not None
        assert tail.component_data[CT.FULL].host_value is None
    finally:
        shutdown(cache)


def test_full_host_eviction_crosses_empty_host_nodes_and_respects_leaf_lru():
    cache, alloc, _ = make_cache(window=4, policy="write_back")
    try:
        _, tail = insert(cache, alloc, range(12))
        path = cache._hybrid_coordinator.path(tail)
        _, other = insert(cache, alloc, range(100, 104))
        for node in (path[0], tail, other):
            cache._hybrid_coordinator.backup_component(node, CT.FULL)
        settle(cache)
        assert path[1].component_data[CT.FULL].host_value is None
        path[0].last_access_time = 1
        other.last_access_time = 2
        tail.last_access_time = 3
        assert cache.evict_host(4, component_type=CT.FULL) == 4
        assert other.component_data[CT.FULL].host_value is None
        assert path[0].component_data[CT.FULL].host_value is not None
        assert cache.evict_host(8, component_type=CT.FULL) == 8
        assert path[0].component_data[CT.FULL].host_value is None
        assert tail.component_data[CT.FULL].host_value is None
    finally:
        shutdown(cache)
