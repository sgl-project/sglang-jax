"""Hybrid transaction failure, ownership and window invariants on CPU."""

from concurrent.futures import Future

import numpy as np
import pytest

from sgl_jax.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType as CT
from sgl_jax.test.mem_cache.test_hybrid_hicache import (
    insert,
    key,
    make_cache,
    settle,
    shutdown,
)


def test_pending_backup_is_not_host_hit_and_split_settles(monkeypatch):
    cache, alloc, _ = make_cache(window=8, policy="write_back")
    try:
        _, node = insert(cache, alloc, range(8))
        future = Future()
        controller = cache.hicache_controllers[CT.SWA]
        monkeypatch.setattr(controller, "write", lambda *_: future)
        cache._hybrid_coordinator.backup_component(node, CT.SWA)
        assert node.component_data[CT.SWA].host_value is None
        cache.components[CT.SWA].evict_component(node)
        assert cache.match_prefix(MatchPrefixParams(key=key(range(8)))).host_hit_length == 0
        future.set_result(None)
        # Handles become visible only after completion, then split in page units.
        cache._split_node(node.key, node, 4)
        assert len(node.component_data[CT.SWA].host_value) == 4
        assert len(node.parent.component_data[CT.SWA].host_value) == 4
    finally:
        shutdown(cache)


def test_swa_internal_writeback_and_short_window_restore():
    cache, alloc, _ = make_cache(window=4, policy="write_back")
    try:
        full, tail = insert(cache, alloc, range(12))
        middle = tail.parent
        full_available = alloc.full_available_size()
        # An internal SWA eviction does not free its FULL owner.
        cache.components[CT.SWA].evict_component(middle)
        settle(cache)
        assert middle.component_data[CT.SWA].host_value is not None
        assert alloc.full_available_size() == full_available
        assert (
            alloc.translate_full_to_swa(
                middle.component_data[CT.FULL].value, require_mapped=False, dp_rank=0
            ).sum()
            == 0
        )
        cache.evict(EvictParams(num_tokens=12))
        settle(cache)
        result = cache.match_prefix(MatchPrefixParams(key=key(range(12))))
        assert result.host_hit_length == 12
        assert result.swa_host_hit_length == 4
        assert cache.get_load_back_sizes(tail) == (12, 4)
        restored, _, _ = cache.init_load_back(tail, 12)
        assert len(restored) == 12
        assert alloc.count_swa_mapped(restored, dp_rank=0) == 4
        assert tail.parent.component_data[CT.SWA].value is None
    finally:
        shutdown(cache)


def test_second_component_failure_rolls_back_all_destinations(monkeypatch):
    cache, alloc, _ = make_cache(window=8)
    try:
        _, node = insert(cache, alloc, range(8))
        settle(cache)
        cache.evict(EvictParams(num_tokens=8))
        before = (alloc.full_available_size(), alloc.swa_available_size())

        def fail(*args):
            raise RuntimeError("injected SWA scatter failure")

        monkeypatch.setattr(cache.hicache_controllers[CT.SWA], "flush_load", fail)
        with pytest.raises(RuntimeError, match="injected SWA scatter"):
            cache.init_load_back(node, 8)
        assert node.component_data[CT.FULL].value is None
        assert node.component_data[CT.SWA].value is None
        assert (alloc.full_available_size(), alloc.swa_available_size()) == before
        assert all(cd.lock_ref == cd.host_lock_ref == 0 for cd in node.component_data)
        assert not alloc.full_to_swa_index_mapping.any()
        assert all(not pool._pending_load for pool in cache.host_pools.values())
    finally:
        shutdown(cache)


@pytest.mark.parametrize("failure", ["first_scatter", "stage_submit", "stage_worker"])
def test_failed_restore_discards_only_its_staging(monkeypatch, failure):
    cache, alloc, _ = make_cache(window=8)
    try:
        _, unrelated = insert(cache, alloc, range(20, 28))
        _, node = insert(cache, alloc, range(8))
        settle(cache)
        cache.evict(EvictParams(num_tokens=16))
        before = (alloc.full_available_size(), alloc.swa_available_size())
        retained = {}
        for ct, pool in cache.host_pools.items():
            handles = list(unrelated.component_data[ct].host_value)
            pool.stage_load(handles)
            retained[ct] = dict(pool._pending_load)

        def fail(*args):
            raise RuntimeError("injected restore failure")

        with monkeypatch.context() as patch:
            if failure == "first_scatter":
                patch.setattr(cache.hicache_controllers[CT.FULL], "flush_load", fail)
            elif failure == "stage_submit":
                patch.setattr(cache.hicache_controllers[CT.SWA], "stage_load", fail)
            else:
                pool = cache.host_pools[CT.SWA]
                stage = pool.stage_load

                def stage_then_fail(handles):
                    stage(handles)
                    fail()

                patch.setattr(pool, "stage_load", stage_then_fail)
            with pytest.raises(RuntimeError, match="injected restore failure"):
                cache.init_load_back(node, 8)

        assert (alloc.full_available_size(), alloc.swa_available_size()) == before
        assert all(cd.value is None for cd in node.component_data)
        assert all(cd.lock_ref == cd.host_lock_ref == 0 for cd in node.component_data)
        assert not alloc.full_to_swa_index_mapping.any()
        for ct, pool in cache.host_pools.items():
            assert set(pool._pending_load) == set(retained[ct])
            for handle, entry in retained[ct].items():
                assert pool._pending_load[handle] is entry
            assert not cache.hicache_controllers[ct].has_inflight(
                list(node.component_data[ct].host_value)
            )
        # Cleanup preserves the host data, so a subsequent restore can succeed.
        restored, _, _ = cache.init_load_back(node, 8)
        assert len(restored) == 8
        for ct, pool in cache.host_pools.items():
            assert set(pool._pending_load) == set(retained[ct])
    finally:
        shutdown(cache)


def test_independent_capacity_rejection_and_reset():
    cache, alloc, _ = make_cache(window=8)
    try:
        _, node = insert(cache, alloc, range(8))
        settle(cache)
        cache.evict(EvictParams(num_tokens=8))
        before = (alloc.full_available_size(), alloc.swa_available_size())
        restored, _, plan = cache.init_load_back(node, 8, swa_mem_quota=7)
        assert len(restored) == 0 and not plan
        assert (alloc.full_available_size(), alloc.swa_available_size()) == before
        cache.reset()
        for pool in cache.host_pools.values():
            assert pool.available_size() == pool.total_size()
    finally:
        shutdown(cache)


def test_swa_host_gap_rejects_but_deeper_valid_window_recovers():
    cache, alloc, _ = make_cache(window=4)
    try:
        _, tail = insert(cache, alloc, range(12))
        settle(cache)
        middle = tail.parent
        cache.components[CT.SWA].evict_component(middle)
        pool = cache.host_pools[CT.SWA]
        pool.free(list(middle.component_data[CT.SWA].host_value))
        middle.component_data[CT.SWA].host_value = None
        middle_match = cache.match_prefix(MatchPrefixParams(key=key(range(8))))
        assert len(middle_match.device_indices) == 4
        assert middle_match.host_hit_length == 0
        tail_match = cache.match_prefix(MatchPrefixParams(key=key(range(12))))
        assert len(tail_match.device_indices) == 12
    finally:
        shutdown(cache)


def test_failed_backup_releases_reservation_without_exposing_host(monkeypatch):
    cache, alloc, _ = make_cache(window=8, policy="write_back")
    try:
        _, node = insert(cache, alloc, range(8))
        future = Future()
        controller = cache.hicache_controllers[CT.SWA]
        monkeypatch.setattr(controller, "write", lambda *_: future)
        pool = cache.host_pools[CT.SWA]
        available = pool.available_size()
        cache._hybrid_coordinator.backup_component(node, CT.SWA)
        cache.components[CT.SWA].evict_component(node)
        assert node.component_data[CT.SWA].host_value is None
        future.set_exception(RuntimeError("failed host copy"))
        with pytest.raises(RuntimeError, match="failed host copy"):
            cache._split_node(node.key, node, 4)
        assert pool.available_size() == available
        assert not node.component_data[CT.SWA].metadata.get("host_pending")
        assert len(node.key) == 8
    finally:
        shutdown(cache)


def test_restore_host_sources_pinned_through_scatter(monkeypatch):
    cache, alloc, _ = make_cache(window=8)
    try:
        _, node = insert(cache, alloc, range(8))
        settle(cache)
        cache.evict(EvictParams(num_tokens=8))
        controller = cache.hicache_controllers[CT.FULL]
        flush = controller.flush_load

        def pressure(*args):
            assert cache.evict_host(100, component_type=CT.FULL) == 0
            assert cache.evict_host(100, component_type=CT.SWA) == 0
            return flush(*args)

        monkeypatch.setattr(controller, "flush_load", pressure)
        restored, _, _ = cache.init_load_back(node, 8)
        assert len(restored) == 8
        assert all(cd.host_lock_ref == 0 for cd in node.component_data)
    finally:
        shutdown(cache)


def test_locked_full_swa_only_healing_preserves_counters():
    cache, alloc, _ = make_cache(window=8)
    try:
        full, node = insert(cache, alloc, range(8))
        settle(cache)
        cache.components[CT.SWA].evict_component(node)
        lock = cache.inc_lock_ref(node)
        protected = cache.component_protected_size_[CT.FULL][0]
        restored, _, _ = cache.init_load_back(node, 8)
        np.testing.assert_array_equal(restored, full)
        assert cache.component_protected_size_[CT.FULL][0] == protected == 8
        assert cache.component_evictable_size_[CT.SWA][0] == 8
        cache.dec_lock_ref(node, lock.to_dec_params())
        assert cache.component_protected_size_[CT.FULL][0] == 0
        assert cache.component_evictable_size_[CT.FULL][0] == 8
    finally:
        shutdown(cache)


def use_fake_native(cache, pool):
    from sgl_jax.srt.mem_cache.raiden_hicache import (
        RaidenHiCacheController,
        RaidenHostKVPool,
    )
    from sgl_jax.test.mem_cache.test_raiden_hicache import Engine

    shutdown(cache)
    engines = {}
    for ct, device in [(CT.FULL, pool.full_kv_pool), (CT.SWA, pool.swa_kv_pool)]:
        engines[ct] = Engine()
        host = RaidenHostKVPool({0: engines[ct]}, 64, device.kv_buffer[0].shape[0])
        cache.host_pools[ct] = host
        cache.hicache_controllers[ct] = RaidenHiCacheController(host, device)
    cache.host_pool = cache.host_pools[CT.FULL]
    cache.hicache_controller = cache.hicache_controllers[CT.FULL]
    return engines


def test_native_partial_submit_awaits_before_healthy_rollback(monkeypatch):
    cache, alloc, pool = make_cache(window=8)
    engines = use_fake_native(cache, pool)
    try:
        _, node = insert(cache, alloc, range(8))
        cache.evict(EvictParams(num_tokens=8))
        before = (alloc.full_available_size(), alloc.swa_available_size())

        def invalid():
            raise ValueError("second component preparation rejected")

        monkeypatch.setattr(cache.hicache_controllers[CT.SWA], "prepare_transfer", invalid)
        with pytest.raises(ValueError, match="preparation rejected"):
            cache.init_load_back(node, 8)
        assert engines[CT.FULL].future.waits == 1
        assert node.evicted
        assert (alloc.full_available_size(), alloc.swa_available_size()) == before
        assert all(cd.lock_ref == cd.host_lock_ref == 0 for cd in node.component_data)
    finally:
        shutdown(cache)


def test_native_uncertain_failure_quarantines_both_component_reservations(monkeypatch):
    cache, alloc, pool = make_cache(window=8)
    engines = use_fake_native(cache, pool)
    _, node = insert(cache, alloc, range(8))
    cache.evict(EvictParams(num_tokens=8))
    before = (alloc.full_available_size(), alloc.swa_available_size())
    original = engines[CT.SWA].h2d

    def uncertain(*args):
        future = original(*args)
        future.error = RuntimeError("uncertain native DMA")
        return future

    monkeypatch.setattr(engines[CT.SWA], "h2d", uncertain)
    with pytest.raises(RuntimeError, match="uncertain native DMA"):
        cache.init_load_back(node, 8)
    assert engines[CT.FULL].future.waits == 1
    assert cache.host_pools[CT.SWA].failed
    assert node.evicted and node.component_data[CT.SWA].value is None
    assert (alloc.full_available_size(), alloc.swa_available_size()) == (
        before[0] - 8,
        before[1] - 8,
    )
    assert node.component_data[CT.FULL].lock_ref > 0
    assert all(node.component_data[ct].host_lock_ref > 0 for ct in (CT.FULL, CT.SWA))
    with pytest.raises(RuntimeError, match="quarantined"):
        cache.reset()


def test_mixed_device_and_host_window_only_allocates_missing_swa():
    cache, alloc, _ = make_cache(window=8)
    try:
        full, tail = insert(cache, alloc, range(12))
        settle(cache)
        # Split the 8-token tail into two pages-of-nodes, then keep the last
        # 4 tokens resident while restoring the preceding 4 from host.
        middle = cache._split_node(tail.key, tail, 4)
        cache.components[CT.SWA].evict_component(middle)
        before = alloc.swa_available_size()
        match = cache.match_prefix(MatchPrefixParams(key=key(range(12))))
        assert len(match.device_indices) == 4
        assert match.host_hit_length == 8
        assert match.swa_host_hit_length == 4
        restored, _, _ = cache.init_load_back(tail, 8)
        np.testing.assert_array_equal(restored, full[4:])
        assert alloc.swa_available_size() == before - 4
        np.testing.assert_array_equal(
            alloc.translate_full_to_swa(full, dp_rank=0),
            np.concatenate(
                [
                    tail.parent.parent.component_data[CT.SWA].value,
                    middle.component_data[CT.SWA].value,
                    tail.component_data[CT.SWA].value,
                ]
            ),
        )
    finally:
        shutdown(cache)


@pytest.mark.parametrize("page,window", [(128, 1024), (256, 128)])
def test_host_splits_use_page_handles_and_boundary_nodes(page, window):
    cache, alloc, _ = make_cache(page=page, window=window)
    try:
        full, tail = insert(cache, alloc, range(page * 4))
        settle(cache)
        if len(tail.key) > page:
            cache._split_node(tail.key, tail, page)
        path = cache._hybrid_coordinator.path(tail)
        for node in path:
            for ct in (CT.FULL, CT.SWA):
                assert len(node.component_data[ct].host_value) == len(node.key) // page
        cache.components[CT.SWA].evict_component(tail)
        match = cache.match_prefix(MatchPrefixParams(key=key(range(page * 4))))
        assert match.swa_host_hit_length == len(tail.key)
        assert match.swa_host_hit_length >= min(window, len(tail.key))
    finally:
        shutdown(cache)


def test_host_pressure_reclaims_both_pools_and_empty_structural_nodes():
    cache, alloc, _ = make_cache(window=4)
    try:
        _, tail = insert(cache, alloc, range(12))
        settle(cache)
        cache.evict(EvictParams(num_tokens=12))
        assert cache.evict_host(100, component_type=CT.FULL) == 12
        assert cache.evict_host(100, component_type=CT.SWA) == 12
        assert not cache.root_node.children
        for pool in cache.host_pools.values():
            assert pool.available_size() == pool.total_size()
    finally:
        shutdown(cache)


def test_pending_dual_backup_keeps_evicted_tree_node_and_hides_incomplete_hit(monkeypatch):
    cache, alloc, _ = make_cache(window=8, policy="write_back")
    try:
        _, node = insert(cache, alloc, range(8))
        futures = {ct: Future() for ct in (CT.FULL, CT.SWA)}
        for ct, future in futures.items():
            monkeypatch.setattr(cache.hicache_controllers[ct], "write", lambda *_, f=future: f)
        cache.evict(EvictParams(num_tokens=8))
        assert node.evicted and node in cache.root_node.children.values()
        assert cache.match_prefix(MatchPrefixParams(key=key(range(8)))).host_hit_length == 0
        futures[CT.FULL].set_result(None)
        assert cache.match_prefix(MatchPrefixParams(key=key(range(8)))).host_hit_length == 0
        futures[CT.SWA].set_result(None)
        assert cache.match_prefix(MatchPrefixParams(key=key(range(8)))).host_hit_length == 8
        cache.reset()
        assert all(pool.available_size() == pool.total_size() for pool in cache.host_pools.values())
    finally:
        shutdown(cache)


def test_swa_allocation_failure_rolls_back_full_reservation():
    cache, alloc, _ = make_cache(window=8)
    try:
        _, node = insert(cache, alloc, range(8))
        settle(cache)
        cache.evict(EvictParams(num_tokens=8))
        busy_swa = alloc.alloc_swa(alloc.swa_available_size())
        full_free = alloc.full_available_size()
        restored, _, _ = cache.init_load_back(node, 8)
        assert len(restored) == 0
        assert alloc.full_available_size() == full_free
        assert node.evicted
        assert all(cd.lock_ref == cd.host_lock_ref == 0 for cd in node.component_data)
        alloc.free_swa_indices(busy_swa)
    finally:
        shutdown(cache)


@pytest.mark.parametrize("fresh_healing", [False, True])
def test_skipped_tombstone_receipt_survives_split_and_healing(fresh_healing):
    cache, alloc, _ = make_cache(window=8)
    try:
        full, node = insert(cache, alloc, range(8))
        settle(cache)
        cache.components[CT.SWA].evict_component(node)
        original_receipt = cache.inc_lock_ref(node)
        saved_params = original_receipt.to_dec_params()
        assert (
            saved_params.skip_lock_node_ids[CT.SWA] is original_receipt.skip_lock_node_ids[CT.SWA]
        )
        parent = cache._split_node(node.key, node, 4)
        # Repeated splitting must preserve the original skipped fragment too.
        grandparent = cache._split_node(parent.key, parent, 2)
        if fresh_healing:
            from sgl_jax.srt.mem_cache.base_prefix_cache import InsertParams

            fresh = alloc.alloc(8)
            cache.insert(InsertParams(key=key(range(8)), value=fresh))
            # Existing FULL slots were protected, so only the new SWA mapping
            # transfers into tree ownership; dispose of unused fresh FULL slots.
            alloc.free_full(fresh, dp_rank=0)
        else:
            cache.init_load_back(node, 8)
        # A newer request genuinely owns all healed SWA fragments. Releasing
        # the old skipped receipt must neither assert nor steal these locks.
        newer_receipt = cache.inc_lock_ref(node)
        cache.dec_lock_ref(node, saved_params)
        for fragment in (grandparent, parent, node):
            assert fragment.component_data[CT.SWA].lock_ref == 1
        assert cache.component_protected_size_[CT.SWA][0] == 8
        assert cache.component_evictable_size_[CT.SWA][0] == 0
        cache.dec_lock_ref(node, newer_receipt.to_dec_params())
        assert all(
            "skip_lock_receipts" not in fragment.component_data[CT.SWA].metadata
            for fragment in (grandparent, parent, node)
        )
        assert cache.component_protected_size_[CT.SWA][0] == 0
        assert cache.component_evictable_size_[CT.SWA][0] == 8
        assert cache.component_protected_size_[CT.FULL][0] == 0
        np.testing.assert_array_equal(
            np.concatenate(
                [
                    grandparent.component_data[CT.FULL].value,
                    parent.component_data[CT.FULL].value,
                    node.component_data[CT.FULL].value,
                ]
            ),
            full,
        )
    finally:
        shutdown(cache)


@pytest.mark.parametrize("policy", ["write_through", "write_back"])
@pytest.mark.parametrize("swa_evicted", [0, 4, 8])
def test_recomputed_full_tombstone_adopts_live_swa_suffix_without_leaking(policy, swa_evicted):
    from sgl_jax.srt.mem_cache.base_prefix_cache import InsertParams

    cache, alloc, _ = make_cache(window=8, policy=policy)
    try:
        _, node = insert(cache, alloc, range(8))
        settle(cache)
        cache.evict(EvictParams(num_tokens=8))
        settle(cache)
        fresh = alloc.alloc(8)
        if swa_evicted:
            alloc.free_swa(fresh[:swa_evicted], dp_rank=0)
        cache.insert(InsertParams(key=key(range(8)), value=fresh, swa_evicted_seqlen=swa_evicted))
        settle(cache)
        assert cache.component_evictable_size_[CT.SWA][0] == 8 - swa_evicted
        path = cache._hybrid_coordinator.path(node)
        resident = [
            n.component_data[CT.SWA].value
            for n in path
            if n.component_data[CT.SWA].value is not None
        ]
        assert sum(map(len, resident)) == 8 - swa_evicted
        if resident:
            np.testing.assert_array_equal(
                np.concatenate(resident),
                alloc.translate_full_to_swa(fresh[swa_evicted:], dp_rank=0),
            )
        cache.evict(EvictParams(num_tokens=8))
        settle(cache)
        assert alloc.full_available_size() == 32
        assert alloc.swa_available_size() == 16
        assert not alloc.full_to_swa_index_mapping.any()
        assert cache.component_evictable_size_[CT.SWA][0] == 0
    finally:
        shutdown(cache)


def check_receipt_acquired_after_split(page=1, dp_size=1, rank=0):
    cache, alloc, _ = make_cache(page=page, window=page * 8, dp_size=dp_size)
    try:
        _, child = insert(cache, alloc, range(page * 8), rank=rank)
        settle(cache)
        parent = cache._split_node(child.key, child, page * 4)
        cache.components[CT.SWA].evict_component(child)
        receipt = cache.inc_lock_ref(child)
        assert parent.component_data[CT.SWA].lock_ref == 1
        assert cache.component_protected_size_[CT.SWA][rank] == page * 4
        cache.dec_lock_ref(child, receipt.to_dec_params())
        assert parent.component_data[CT.SWA].lock_ref == 0
        assert cache.component_protected_size_[CT.SWA][rank] == 0
        assert cache.component_evictable_size_[CT.SWA][rank] == page * 4
        assert "skip_lock_receipts" not in child.component_data[CT.SWA].metadata
        assert "skip_lock_receipts" not in parent.component_data[CT.SWA].metadata
    finally:
        shutdown(cache)


def test_receipt_acquired_after_split_releases_live_prefix_lock():
    check_receipt_acquired_after_split()
