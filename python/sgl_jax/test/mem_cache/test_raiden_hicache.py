"""CPU contract tests; fake native DMA does not establish TPU correctness."""

import numpy as np
import pytest

from sgl_jax.srt.mem_cache.cache_transfer import DevicePageSpan, TransferOperation
from sgl_jax.srt.mem_cache.raiden_hicache import (
    RaidenHiCacheController,
    RaidenHostKVPool,
)
from sgl_jax.test.mem_cache import test_hicache_e2e as lifecycle


class Future:
    def __init__(self, error=None):
        self.ready = False
        self.error = error
        self.waits = 0

    def IsReady(self):
        return self.ready

    def Await(self):
        self.waits += 1
        self.ready = True
        if self.error:
            raise self.error


class Engine:
    def __init__(self):
        self.next = 100
        self.unlocked = []
        self.future = None

    def d2h_auto_allocate(self, pages):
        chunks = list(range(self.next, self.next + len(pages)))
        self.next += len(pages)
        self.future = Future()
        return chunks, self.future

    def h2d(self, chunks, pages):
        self.future = Future()
        return self.future

    def unlock_blocks(self, chunks):
        self.unlocked.extend(chunks)


def test_completion_callback_once():
    future = Future()
    completed = []
    op = TransferOperation(future, lambda: completed.append(True))
    assert not op.done()
    assert future.waits == 0
    op.wait()
    op.wait()
    assert op.done() and completed == [True] and future.waits == 1


def test_rank_capacity_lifetime_and_stale_handles():
    engines = {0: Engine(), 1: Engine()}
    pool = RaidenHostKVPool(engines, 2, 16)
    handles = pool.alloc(2, dp_rank=1)
    assert pool.alloc(1, dp_rank=1) is None
    assert pool.available_size(0) == 2
    with pytest.raises(ValueError, match="ranks differ"):
        pool.submit_backup(DevicePageSpan(0, (1, 2)), handles)
    op = pool.submit_backup(DevicePageSpan(1, (1, 2)), handles)
    with pytest.raises(RuntimeError, match="in-flight"):
        pool.free(handles)
    op.wait()
    pool.pin(handles)
    first = pool.submit_restore(handles, DevicePageSpan(1, (3, 4)))
    second = pool.submit_restore(handles, DevicePageSpan(1, (5, 6)))
    first.wait()
    second.wait()
    assert pool.has_inflight(handles)
    pool.unpin(handles)
    pool.free(handles)
    assert engines[1].unlocked == [100, 101]
    assert set(pool.alloc(2, 1)).isdisjoint(handles)
    with pytest.raises(ValueError, match="stale"):
        pool.free(handles)


def test_native_error_quarantines_pages():
    engine = Engine()
    pool = RaidenHostKVPool({0: engine}, 2, 16)
    handles = pool.alloc(1)
    op = pool.submit_backup(DevicePageSpan(0, (1,)), handles)
    engine.future.error = RuntimeError("DMA failure")
    with pytest.raises(RuntimeError, match="DMA failure"):
        op.wait()
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.free(handles)
    assert engine.unlocked == []


def test_empty_and_invalid_transfers_do_not_submit():
    engine = Engine()
    pool = RaidenHostKVPool({0: engine}, 2, 16)
    pool.submit_backup(DevicePageSpan(0, ()), []).wait()
    pool.submit_restore([], DevicePageSpan(0, ())).wait()
    handles = pool.alloc(2)
    for pages in [(1,), (1, 1), (1, 16)]:
        with pytest.raises(ValueError):
            pool.submit_backup(DevicePageSpan(0, pages), handles)
    assert engine.future is None
    pool.free(handles)


class CopyEngine(Engine):
    """Copies actual CPU KV values when Await is called."""

    def __init__(self, device_pool):
        super().__init__()
        self.device_pool = device_pool
        self.host = {}

    def d2h_auto_allocate(self, pages):
        chunks, _ = super().d2h_auto_allocate(pages)
        engine = self

        class Copy(Future):
            def Await(self):
                super().Await()
                for chunk, page in zip(chunks, pages):
                    engine.host[chunk] = [
                        np.asarray(b[page]).copy() for b in engine.device_pool.kv_buffer
                    ]

        return chunks, Copy()

    def h2d(self, chunks, pages):
        engine = self

        class Copy(Future):
            def Await(self):
                super().Await()
                for layer, buffer in enumerate(engine.device_pool.kv_buffer):
                    for chunk, page in zip(chunks, pages):
                        buffer = buffer.at[page].set(
                            engine.host[chunk][layer], out_sharding=buffer.sharding
                        )
                    engine.device_pool.kv_buffer[layer] = buffer

        return Copy()

    def unlock_blocks(self, chunks):
        super().unlock_blocks(chunks)
        for chunk in chunks:
            del self.host[chunk]


# Run the existing tree lifecycle assertions against the direct adapter, with
# real CPU array contents but an intentionally deferred native future.


class DirectSetup:
    def setUp(self):
        super().setUp()
        self.controller.shutdown()
        self.host_pool = RaidenHostKVPool(
            {0: CopyEngine(self.kv_cache)}, self.HOST_PAGES, self.kv_cache.kv_buffer[0].shape[0]
        )
        self.controller = RaidenHiCacheController(self.host_pool, self.kv_cache)
        self._enable_hicache(write_through_threshold=1)


class TestDirectWriteThrough(DirectSetup, lifecycle.TestWriteThrough):
    def test_backup_triggers_on_reuse_and_releases_lock(self):
        indices, _ = self._alloc_and_fill(4, seed=1)
        params = lifecycle.InsertParams(key=lifecycle._key([10, 11, 12, 13]), value=indices)
        self.cache.insert(params)
        node = self._child_of_root()
        engine = self.host_pool.engines[0]
        original = engine.d2h_auto_allocate

        def submit(pages):
            assert not node.backuped
            assert node.component_data[0].lock_ref > 0
            chunks, future = original(pages)
            wait = future.Await

            def checked_wait():
                assert not node.backuped
                assert node.component_data[0].lock_ref > 0
                wait()

            future.Await = checked_wait
            return chunks, future

        engine.d2h_auto_allocate = submit
        params.prev_prefix_len = 4
        self.cache.insert(params)
        assert node.backuped
        assert node.component_data[0].lock_ref == 0
        assert not self.cache.ongoing_write


class TestDirectLoadBack(DirectSetup, lifecycle.TestEvictAndLoadBack):
    pass


class TestDirectWriteBack(DirectSetup, lifecycle.TestWriteBack):
    pass


class TestDirectRevival(DirectSetup, lifecycle.TestTombstoneRevival):
    pass


class TestDirectHostEviction(DirectSetup, lifecycle.TestFullMiss):
    pass


class TestDirectPage4(DirectSetup, lifecycle.TestEvictAndLoadBackPage4):
    pass


def test_failed_previous_operation_prevents_new_submission():
    engine = Engine()
    pool = RaidenHostKVPool({0: engine}, 2, 16)
    first, second = pool.alloc(2)
    pool.submit_backup(DevicePageSpan(0, (1,)), [first])
    engine.future.ready = True
    engine.future.error = RuntimeError("asynchronous failure")
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.submit_backup(DevicePageSpan(0, (2,)), [second])
    assert engine.next == 101


def test_config_and_preload_selection():
    import os
    from unittest.mock import patch

    from sgl_jax.raiden import raiden_requested
    from sgl_jax.srt.server_args import ServerArgs

    assert raiden_requested(["--hicache-transfer-backend=raiden"])
    assert raiden_requested(["--hicache-transfer-backend", "raiden"])
    assert not raiden_requested(
        ["--hicache-transfer-backend=raiden", "--hicache-transfer-backend=jax"]
    )
    assert raiden_requested(["--disaggregation-use-raiden", "--hicache-transfer-backend=jax"])
    for kwargs, message in [
        ({"hicache_storage": "disable"}, "requires --hicache-storage"),
        ({"disaggregation_use_raiden": True}, "with PD"),
        ({"pd_disaggregation": True}, "with PD"),
        ({"disaggregation_mode": "prefill"}, "with PD"),
        ({"disaggregation_mode": "decode"}, "with PD"),
        ({"device": "cpu"}, "single-host TPU"),
        ({"nnodes": 2}, "single-host TPU"),
        ({"speculative_algorithm": "EAGLE"}, "speculative"),
    ]:
        args = dict(
            model_path="dummy",
            max_seq_len=2048,
            device="tpu",
            hicache_storage="none",
            hicache_transfer_backend="raiden",
        )
        args.update(kwargs)
        with (
            patch.dict(os.environ, {"JAX_PLATFORMS": args["device"]}),
            pytest.raises(ValueError, match=message),
        ):
            ServerArgs(**args)


def test_tree_backup_failure_keeps_source_locked_without_publishing():
    case = TestDirectWriteThrough()
    case.setUp()
    indices, _ = case._alloc_and_fill(4, seed=3)
    case.cache.insert(lifecycle.InsertParams(key=lifecycle._key([1, 2, 3, 4]), value=indices))
    node = case._child_of_root()
    engine = case.host_pool.engines[0]
    engine.d2h_auto_allocate = lambda pages: (
        [100 + i for i in range(len(pages))],
        Future(RuntimeError("native failure")),
    )
    with pytest.raises(RuntimeError, match="native failure"):
        case.cache.write_backup(node)
    assert not node.backuped
    assert node.component_data[0].lock_ref > 0
    assert case.host_pool.failed and not engine.unlocked
    with pytest.raises(RuntimeError, match="quarantined"):
        case.cache.check_hicache_events()


def test_tree_restore_failure_does_not_publish_or_recycle_destination():
    case = TestDirectLoadBack()
    case.setUp()
    indices, _ = case._alloc_and_fill(4, seed=3)
    case.cache.insert(lifecycle.InsertParams(key=lifecycle._key([1, 2, 3, 4]), value=indices))
    node = case._child_of_root()
    case.cache.write_backup(node)
    case.cache.evict(lifecycle.EvictParams(num_tokens=4, dp_rank=0))
    before = case.allocator.available_size(0)
    engine = case.host_pool.engines[0]
    engine.h2d = lambda chunks, pages: Future(RuntimeError("native failure"))
    with pytest.raises(RuntimeError, match="native failure"):
        case.cache.init_load_back(node, 4)
    assert node.evicted and node.backuped
    assert case.allocator.available_size(0) == before - 4
    assert case.host_pool.failed and not engine.unlocked


def test_manager_resolution_uses_preloaded_namespace(monkeypatch):
    import sys
    from types import SimpleNamespace

    from sgl_jax import raiden

    marker = object()
    monkeypatch.setitem(sys.modules, "tpu_sync.frameworks.jax._tpu_raiden_jax", marker)
    calls = []

    def load(name):
        calls.append(name)
        return SimpleNamespace(KVCacheManager=marker)

    monkeypatch.setattr(raiden.importlib, "import_module", load)
    assert raiden.get_raiden_kv_cache_manager() is marker
    assert calls == ["tpu_sync.api.jax.kv_cache_manager"]


def test_manager_resolution_does_not_hide_missing_dependency(monkeypatch):
    import sys

    from sgl_jax import raiden

    monkeypatch.setitem(sys.modules, "tpu_sync.frameworks.jax._tpu_raiden_jax", object())

    calls = []

    def load(name):
        calls.append(name)
        raise ModuleNotFoundError("missing dependency", name="native_dependency")

    monkeypatch.setattr(raiden.importlib, "import_module", load)
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        raiden.get_raiden_kv_cache_manager()
    assert len(calls) == 1


def test_registered_buffer_replacement_fails_before_dma():
    case = TestDirectLoadBack()
    case.setUp()
    controller = RaidenHiCacheController(
        case.host_pool, case.kv_cache, check_registered_buffers=True
    )
    # Keep the original alive, so the replacement cannot reuse its allocation.
    original = list(case.kv_cache.kv_buffer)
    case.kv_cache.kv_buffer = [b + 1 for b in original]
    with pytest.raises(RuntimeError, match="allocation changed"):
        controller.prepare_transfer()
    assert case.host_pool.failed
    assert case.host_pool.engines[0].next == 100


@pytest.mark.parametrize("finished", [True, False])
@pytest.mark.parametrize("direct", [True, False])
def test_request_cleanup_keeps_recomputed_tombstone_tail(finished, direct):
    from types import SimpleNamespace

    case = TestDirectPage4() if direct else lifecycle.TestEvictAndLoadBackPage4()
    case.setUp()
    try:
        tokens = list(range(12))
        indices, _ = case._alloc_and_fill(12, seed=8)
        case.cache.insert(lifecycle.InsertParams(key=lifecycle._key(tokens), value=indices))
        node = case._child_of_root()
        case.cache.write_backup(node)
        case._settle_writes()
        case.cache.evict(lifecycle.EvictParams(num_tokens=12, dp_rank=0))
        match = case.cache.match_prefix(lifecycle.MatchPrefixParams(key=lifecycle._key(tokens[:8])))
        restored, boundary, plan = case.cache.init_load_back(match.last_host_node, 8)
        case.cache.finish_load_back(plan)
        tail = case.allocator.alloc(4, dp_rank=0)
        lock = case.cache.inc_lock_ref(boundary)
        all_indices = np.concatenate([restored, tail])
        case.req_pool.write((0, slice(0, 12)), all_indices)
        req = SimpleNamespace(
            req_pool_idx=0,
            dp_rank=0,
            radix_input_ids=tokens,
            origin_input_ids=tokens,
            output_ids=[],
            fill_ids=tokens,
            extra_key=None,
            cache_protected_len=8,
            last_matched_prefix_len=8,
            last_node=boundary,
            cache_lock_params=lock.to_dec_params(),
            prefix_indices=restored,
            pop_committed_kv_cache=lambda: 12,
        )
        if finished:
            case.cache.cache_finished_req(req)
        else:
            case.cache.cache_unfinished_req(req)
            case.cache.dec_lock_ref(req.last_node, req.cache_lock_params)
        assert case.allocator.available_size(0) + case.cache.evictable_size(0) == case.DEVICE_SIZE
        live = case.cache.match_prefix(
            lifecycle.MatchPrefixParams(key=lifecycle._key(tokens))
        ).device_indices
        np.testing.assert_array_equal(live, all_indices)
        other = case.allocator.alloc(case.allocator.available_size(0), dp_rank=0)
        assert set(other).isdisjoint(live)
    finally:
        case.controller.shutdown()


class TestDirectRestorePressure(DirectSetup, lifecycle.HiCacheE2EBase):
    DEVICE_SIZE = 12
    HOST_PAGES = 12

    def test_restore_chain_pins_sources_during_write_back_eviction(self):
        self.cache.write_policy = "write_back"
        a, expected_a = self._alloc_and_fill(4, seed=51)
        b, expected_b = self._alloc_and_fill(4, seed=52)
        key_a = lifecycle._key([1, 2, 3, 4])
        key_ab = lifecycle._key(list(range(1, 9)))
        self.cache.insert(lifecycle.InsertParams(key=key_a, value=a))
        self.cache.insert(
            lifecycle.InsertParams(key=key_ab, value=np.concatenate([a, b]), prev_prefix_len=4)
        )
        self.cache.evict(lifecycle.EvictParams(num_tokens=8, dp_rank=0))
        match = self.cache.match_prefix(lifecycle.MatchPrefixParams(key=key_ab))
        leaf = match.last_host_node
        parent = leaf.parent
        handles = [*parent.component_data[0].host_value, *leaf.component_data[0].host_value]
        assert match.host_hit_length == 8

        # A third host-only leaf fills L2; the restore must retain its entire
        # source chain while evicting this unrelated host leaf for write-back.
        spare, _ = self._alloc_and_fill(4, seed=53)
        self.cache.insert(lifecycle.InsertParams(key=lifecycle._key([30, 31, 32, 33]), value=spare))
        self.cache.evict(lifecycle.EvictParams(num_tokens=4, dp_rank=0))
        assert self.host_pool.available_size() == 0
        busy, _ = self._alloc_and_fill(12, seed=54)
        self.cache.insert(
            lifecycle.InsertParams(key=lifecycle._key(list(range(40, 52))), value=busy)
        )
        evictions = []
        original = self.cache.evict_host

        def checked_evict(count, dp_rank=None):
            assert all(self.host_pool.has_inflight([h]) for h in handles)
            freed = original(count, dp_rank)
            assert all(int(h) in self.host_pool._pages for h in handles)
            evictions.append(freed)
            return freed

        self.cache.evict_host = checked_evict
        restored, node, plan = self.cache.init_load_back(leaf, 8)
        assert evictions == [4]
        assert node is leaf and not plan and len(restored) == 8
        for i, index in enumerate(restored):
            for layer in range(self.LAYER_NUM):
                np.testing.assert_array_equal(
                    self._read_token(layer, index), (expected_a + expected_b)[i][layer]
                )
        assert not self.host_pool.has_inflight(handles)
        assert self.allocator.available_size(0) + self.cache.evictable_size(0) == self.DEVICE_SIZE

    def test_failed_device_allocation_releases_restore_pins(self):
        indices, _ = self._alloc_and_fill(4, seed=60)
        self.cache.insert(lifecycle.InsertParams(key=lifecycle._key([1, 2, 3, 4]), value=indices))
        node = self._child_of_root()
        self.cache.write_backup(node)
        self.cache.evict(lifecycle.EvictParams(num_tokens=4, dp_rank=0))
        handles = node.component_data[0].host_value.copy()
        from unittest.mock import patch

        with patch.object(self.allocator, "alloc", return_value=None):
            restored, last, plan = self.cache.init_load_back(node, 4)
        assert len(restored) == 0 and last is node and not plan
        assert node.evicted and node.backuped
        assert not self.host_pool.has_inflight(handles)
        restored, _, _ = self.cache.init_load_back(node, 4)
        assert len(restored) == 4


def test_drain_waits_other_operations_after_native_failure():
    engines = {0: Engine(), 1: Engine()}
    pool = RaidenHostKVPool(engines, 1, 4)
    pool.submit_backup(DevicePageSpan(0, (1,)), pool.alloc(1, 0))
    pool.submit_backup(DevicePageSpan(1, (1,)), pool.alloc(1, 1))
    engines[0].future.error = RuntimeError("failed shard")
    with pytest.raises(RuntimeError, match="failed shard"):
        pool.drain()
    assert [engine.future.waits for engine in engines.values()] == [1, 1]
    assert pool.failed
    assert all(not engine.unlocked for engine in engines.values())


@pytest.mark.parametrize("missing", ["native_dependency", "tpu_sync.frameworks.jax"])
def test_preload_preserves_missing_dependency_without_legacy_fallback(monkeypatch, missing):
    from types import SimpleNamespace

    from sgl_jax import raiden

    monkeypatch.setattr(raiden, "sys", SimpleNamespace(modules={}))
    calls = []

    def load(name):
        calls.append(name)
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    monkeypatch.setattr(raiden.importlib, "import_module", load)
    with pytest.raises(ModuleNotFoundError) as error:
        raiden.preload_raiden()
    assert error.value.name == missing
    assert calls == ["tpu_sync.frameworks.jax._tpu_raiden_jax"]


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("reserved_tokens", [8, 49])
def test_scheduler_restore_can_evict_without_spending_reserved_tokens(direct, reserved_tokens):
    from types import SimpleNamespace

    from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder

    case = TestDirectPage4() if direct else lifecycle.TestEvictAndLoadBackPage4()
    case.setUp()
    req = None
    try:
        target = list(range(100, 108))
        indices, original = case._alloc_and_fill(8, seed=91)
        params = lifecycle.InsertParams(key=lifecycle._key(target), value=indices)
        case.cache.insert(params)
        params.prev_prefix_len = 8
        case.cache.insert(params)
        case._settle_writes()
        case.cache.evict(lifecycle.EvictParams(num_tokens=8, dp_rank=0))
        filler, _ = case._alloc_and_fill(60, seed=92)
        case.cache.insert(
            lifecycle.InsertParams(key=lifecycle._key(list(range(200, 260))), value=filler)
        )
        assert case.allocator.available_size(0) == 4
        match = case.cache.match_prefix(lifecycle.MatchPrefixParams(key=lifecycle._key(target)))
        req = SimpleNamespace(
            dp_rank=0,
            sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=4),
            extend_input_len=12,
            host_hit_length=8,
            prefix_indices=match.device_indices,
            last_node=match.last_device_node,
            last_host_node=match.last_host_node,
            fill_ids=target + [400, 401, 402, 403],
        )
        adder = PrefillAdder(
            page_size=4,
            tree_cache=case.cache,
            token_to_kv_pool_allocator=case.allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=64,
            rem_chunk_tokens=16,
            mixed_with_decode_tokens=reserved_tokens,
        )
        # Keep chunk/input budgets independent of the reserved device capacity.
        adder.rem_input_tokens = 64
        adder.rem_chunk_tokens_list = [16]
        result = adder.add_one_req(req)
        if reserved_tokens == 49:
            assert result == AddReqResult.NO_TOKEN
            assert len(req.prefix_indices) == 0
            assert match.last_host_node.evicted
            assert case.allocator.available_size(0) == 4
            assert not adder.pending_h2d
        elif not direct:
            assert result == AddReqResult.CONTINUE
            assert len(req.prefix_indices) == 0
            assert req.extend_input_len == 12
            assert match.last_host_node.evicted
            assert case.allocator.available_size(0) == 4
            assert not adder.pending_h2d
        else:
            assert result == AddReqResult.CONTINUE
            assert len(req.prefix_indices) == 8
            assert req.extend_input_len == 4
            assert not req.last_node.evicted
            case.cache.finish_load_back(adder.pending_h2d)
            for i, index in enumerate(req.prefix_indices):
                for layer in range(case.LAYER_NUM):
                    np.testing.assert_array_equal(
                        case._read_token(layer, int(index)), original[i][layer]
                    )
    finally:
        if req is not None and hasattr(req, "cache_lock_params"):
            case.cache.dec_lock_ref(req.last_node, req.cache_lock_params)
        case.tearDown()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("page_size", [1, 4])
def test_scheduler_write_back_restore_with_both_tiers_full(direct, page_size):
    from types import SimpleNamespace

    from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder

    base = TestDirectLoadBack if direct else lifecycle.TestEvictAndLoadBack

    class Case(base):
        DEVICE_SIZE = 16
        HOST_PAGES = 8 // page_size
        PAGE_SIZE = page_size

    case = Case()
    case.setUp()
    req = None
    try:
        case.cache.write_policy = "write_back"
        target = list(range(100, 108))
        indices, expected = case._alloc_and_fill(8, seed=101)
        case.cache.insert(lifecycle.InsertParams(key=lifecycle._key(target), value=indices))
        case.cache.evict(lifecycle.EvictParams(num_tokens=8, dp_rank=0))
        case._settle_writes()
        filler, _ = case._alloc_and_fill(16, seed=102)
        case.cache.insert(
            lifecycle.InsertParams(key=lifecycle._key(list(range(200, 216))), value=filler)
        )
        assert case.allocator.available_size(0) == case.host_pool.available_size() == 0
        match = case.cache.match_prefix(lifecycle.MatchPrefixParams(key=lifecycle._key(target)))
        assert match.host_hit_length == 8
        host_handles = match.last_host_node.component_data[0].host_value.copy()
        req = SimpleNamespace(
            dp_rank=0,
            sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=1),
            extend_input_len=9,
            host_hit_length=8,
            prefix_indices=match.device_indices,
            last_node=match.last_device_node,
            last_host_node=match.last_host_node,
            fill_ids=target + [400],
        )
        adder = PrefillAdder(
            page_size=page_size,
            tree_cache=case.cache,
            token_to_kv_pool_allocator=case.allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=32,
            rem_chunk_tokens=16,
        )
        assert adder.add_one_req(req) == AddReqResult.CONTINUE
        np.testing.assert_array_equal(
            match.last_host_node.component_data[0].host_value, host_handles
        )
        assert not adder.pending_h2d
        if direct:
            assert len(req.prefix_indices) == 8
            assert req.extend_input_len == 1
            for i, index in enumerate(req.prefix_indices):
                for layer in range(case.LAYER_NUM):
                    np.testing.assert_array_equal(
                        case._read_token(layer, int(index)), expected[i][layer]
                    )
        else:
            # JAX recomputes instead of evicting its own unpinned host source.
            assert len(req.prefix_indices) == 0
            assert req.extend_input_len == 9
            assert match.last_host_node.evicted
            assert case.allocator.available_size(0) == 0
        case.cache.dec_lock_ref(req.last_node, req.cache_lock_params)
        req = None
        assert case.allocator.available_size(0) + case.cache.evictable_size(0) == 16
    finally:
        if req is not None and hasattr(req, "cache_lock_params"):
            case.cache.dec_lock_ref(req.last_node, req.cache_lock_params)
        case.tearDown()
