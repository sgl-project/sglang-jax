"""Scheduler-facing SWA cache contract tests."""

from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    IncLockRefResult,
    InsertParams,
    validate_swa_cache_ledger,
)
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache


def _empty_ledger(*, full_capacity=8, swa_capacity=8):
    return {
        "dp_rank": 0,
        "full_capacity": full_capacity,
        "full_available": full_capacity,
        "full_tree_evictable": 0,
        "full_tree_protected": 0,
        "full_request_owned": 0,
        "full_reserved_page_slack": 0,
        "swa_capacity": swa_capacity,
        "swa_available": swa_capacity,
        "swa_tree_evictable": 0,
        "swa_tree_protected": 0,
        "swa_request_owned": 0,
        "swa_reserved_page_slack": 0,
        "mapping_nonzero_count": 0,
        "mapping_invalid_count": 0,
        "mapping_duplicate_count": 0,
        "full_duplicate_request_owner_count": 0,
        "swa_duplicate_request_owner_count": 0,
        "full_request_tree_overlap_count": 0,
        "swa_request_tree_overlap_count": 0,
        "full_evicted_total": 3,
        "swa_evicted_total": 2,
        "tombstone_created_total": 1,
        "tombstone_healed_total": 1,
    }


class _FlushTree:
    page_size = 1

    def __init__(self, events, *, supports_swa=True, invalid_pre=False, invalid_field=None):
        self.events = events
        self._supports_swa = supports_swa
        self.invalid_pre = invalid_pre
        self.invalid_field = invalid_field
        self.was_reset = False

    def supports_swa(self):
        return self._supports_swa

    def cache_ledger_snapshot(self, dp_rank, live_reqs):
        assert dp_rank == 0
        assert live_reqs == []
        self.events.append("ledger:post" if self.was_reset else "ledger:pre")
        row = _empty_ledger()
        if not self.was_reset:
            row["full_available"] = 4
            row["full_tree_evictable"] = 4
            row["swa_available"] = 4
            row["swa_tree_evictable"] = 4
            row["mapping_nonzero_count"] = 4
            if self.invalid_pre:
                row["full_tree_evictable"] = 3
            if self.invalid_field is not None:
                row[self.invalid_field] = 1
        return row

    def reset(self):
        self.events.append("tree-reset")
        self.was_reset = True

    def full_evictable_size(self, dp_rank=0):
        assert dp_rank == 0
        return 3 if self.invalid_pre else 4

    def full_protected_size(self, dp_rank=0):
        assert dp_rank == 0
        return 0

    def swa_evictable_size(self, dp_rank=0):
        assert dp_rank == 0
        return 4

    def swa_protected_size(self, dp_rank=0):
        assert dp_rank == 0
        return 0


class _Clearable:
    def __init__(self, events, label):
        self.events = events
        self.label = label

    def clear(self):
        self.events.append(self.label)


class _Allocator(_Clearable):
    full_attn_allocator = SimpleNamespace(size_per_rank=8, page_size=1)
    swa_attn_allocator = SimpleNamespace(size_per_rank=8, page_size=1)

    def available_size(self, _dp_rank=0):
        return 8

    def full_available_size(self, _dp_rank=0):
        return 4

    def swa_available_size(self, _dp_rank=0):
        return 4


def _flush_scheduler(tree):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._can_flush_cache = lambda: (True, "")
    scheduler.dp_size = 1
    scheduler.is_hybrid = True
    scheduler.cur_batch = object()
    scheduler.last_batch = object()
    scheduler.running_batch = object()
    scheduler.pending_dp_reqs = []
    scheduler.chunked_reqs = [None]
    scheduler._pending_chunked_abort_reqs = [None]
    scheduler.enable_overlap = False
    scheduler.tree_cache = tree
    scheduler.req_to_token_pool = _Clearable(tree.events, "req-clear")
    scheduler.token_to_kv_pool_allocator = _Allocator(tree.events, "alloc-clear")
    scheduler.model_config = object()
    scheduler.spec_algorithm = None
    scheduler.mesh = None
    scheduler.grammar_backend = None
    scheduler.tp_worker = None
    scheduler.tp_worker_p = None
    scheduler.tp_workers_p = []
    scheduler.num_generated_tokens = 99
    scheduler.forward_ct_decode = 99
    scheduler.init_new_token_ratio = 0.75
    scheduler.new_token_ratio = 0.25
    return scheduler


def test_base_cache_does_not_claim_swa_or_a_zero_ledger():
    """Generic caches must not accidentally opt into hybrid accounting."""
    assert BasePrefixCache.supports_swa(BasePrefixCache) is False
    with pytest.raises(NotImplementedError):
        BasePrefixCache.cache_ledger_snapshot(BasePrefixCache, 0, [])


def test_request_reset_clears_full_lock_receipt_and_legacy_mirror():
    req = Req.__new__(Req)
    req.cache_lock_params = DecLockRefParams(
        swa_uuid_for_lock=17,
        skip_lock_node_ids={"swa": [3, 5]},
    )
    req.swa_uuid_for_lock = 17

    req.reset_for_retract()

    assert req.cache_lock_params is None
    assert req.swa_uuid_for_lock is None


def test_schedule_batch_dispatches_request_tail_reclaim_to_swa_cache():
    calls = []
    tree = SimpleNamespace(
        supports_swa=lambda: True,
        evict_req_swa=lambda req, pre_len, dp_rank=0: calls.append((req, pre_len, dp_rank)),
    )
    batch = SimpleNamespace(tree_cache=tree)
    req = SimpleNamespace(swa_evicted_seqlen=0)

    ScheduleBatch._evict_swa(batch, req, 32, 8, 1, dp_rank=1)

    assert calls == [(req, 32, 1)]


def test_prefill_admission_stores_the_complete_lock_receipt():
    releases = []
    lock_result = IncLockRefResult(
        swa_uuid_for_lock=17,
        skip_lock_node_ids={"swa": [3, 5]},
    )
    tree = SimpleNamespace(
        disable=False,
        hicache_enabled=False,
        inc_lock_ref=lambda node: lock_result,
        dec_lock_ref=lambda node, params: releases.append(params),
        recurrent_extra_buffer_active=lambda: False,
        evictable_size=lambda dp_rank=0: 0,
    )
    allocator = SimpleNamespace(available_size=lambda dp_rank=0: 100)
    adder = PrefillAdder(
        page_size=1,
        tree_cache=tree,
        token_to_kv_pool_allocator=allocator,
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=32,
        rem_chunk_tokens=None,
    )
    req = SimpleNamespace(
        dp_rank=0,
        extend_input_len=1,
        host_hit_length=0,
        prefix_indices=[],
        last_node=object(),
        last_host_node=None,
        fill_ids=[1],
        origin_input_ids=[1],
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=1, ignore_eos=False),
    )

    result = adder.add_one_req(req)

    assert result is AddReqResult.CONTINUE
    assert req.cache_lock_params.swa_uuid_for_lock == 17
    assert req.cache_lock_params.skip_lock_node_ids == {"swa": [3, 5]}
    assert req.swa_uuid_for_lock == 17
    assert releases[0].skip_lock_node_ids == {"swa": [3, 5]}


def test_retract_releases_complete_receipt_before_request_reset():
    receipt = DecLockRefParams(swa_uuid_for_lock=17, skip_lock_node_ids={"swa": [3]})
    events = []

    class _Req:
        cache_lock_params = receipt
        swa_uuid_for_lock = 17

        def reset_for_retract(self):
            events.append(("reset", self.cache_lock_params))
            self.cache_lock_params = None
            self.swa_uuid_for_lock = None

    req = _Req()
    batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[req])],
        tree_cache=object(),
        _evict_tree_cache_if_needed=lambda needed: events.append(("evict", needed)),
    )

    def release(observed_req, tree_cache, is_insert=True):
        events.append(("release", observed_req.cache_lock_params, is_insert))

    with patch("sgl_jax.srt.managers.schedule_batch.release_kv_cache", side_effect=release):
        ScheduleBatch.release_req(batch, 0, 0, 0, SimpleNamespace())

    assert events[0] == ("release", receipt, False)
    assert events[-1] == ("reset", receipt)
    assert req.cache_lock_params is None
    assert req.swa_uuid_for_lock is None


def test_chunked_abort_releases_complete_receipt_before_owner_cleanup():
    receipt = DecLockRefParams(swa_uuid_for_lock=17, skip_lock_node_ids={"swa": [3]})
    events = []
    req = SimpleNamespace(
        rid="abort-me",
        is_chunked=0,
        cache_lock_params=receipt,
        check_finished=lambda: None,
        finished=lambda: True,
    )
    scheduler = SimpleNamespace(
        chunked_reqs=[req],
        _pending_chunked_abort_reqs=[req],
        _release_prefill_host_buffer=lambda observed_req: events.append("host-release"),
        tree_cache=object(),
        spec_algorithm=None,
    )

    def release(observed_req, tree_cache, **kwargs):
        events.append(("cache-release", observed_req.cache_lock_params, kwargs))

    with patch(
        "sgl_jax.srt.managers.scheduler_output_processor_mixin.release_kv_cache",
        side_effect=release,
    ):
        SchedulerOutputProcessorMixin._finalize_chunked_abort(scheduler, req, 0)

    assert events[1][0:2] == ("cache-release", receipt)
    assert events[1][2]["is_insert"] is False
    assert scheduler.chunked_reqs == [None]
    assert scheduler._pending_chunked_abort_reqs == [None]


def test_idle_flush_validates_before_mutation_then_resets_and_post_validates():
    events = []
    tree = _FlushTree(events)
    scheduler = _flush_scheduler(tree)

    with patch.object(ScheduleBatch, "init_new", return_value=object()):
        result = Scheduler.flush_cache(scheduler)

    assert result == (True, "", 8)
    assert events == [
        "ledger:pre",
        "tree-reset",
        "req-clear",
        "alloc-clear",
        "ledger:post",
    ]


def test_idle_flush_rejects_preexisting_leak_without_clearing_it():
    events = []
    tree = _FlushTree(events, invalid_pre=True)
    scheduler = _flush_scheduler(tree)

    with (
        patch.object(ScheduleBatch, "init_new", return_value=object()),
        pytest.raises(ValueError, match="full.*balance"),
    ):
        Scheduler.flush_cache(scheduler)

    assert events == ["ledger:pre"]


@pytest.mark.parametrize(
    "field",
    [
        "full_duplicate_request_owner_count",
        "swa_duplicate_request_owner_count",
        "full_request_tree_overlap_count",
        "swa_request_tree_overlap_count",
        "mapping_invalid_count",
        "mapping_duplicate_count",
    ],
)
def test_idle_flush_rejects_each_ownership_diagnostic(field):
    events = []
    tree = _FlushTree(events, invalid_field=field)
    scheduler = _flush_scheduler(tree)

    with (
        patch.object(ScheduleBatch, "init_new", return_value=object()),
        pytest.raises(ValueError, match=field),
    ):
        Scheduler.flush_cache(scheduler)

    assert events == ["ledger:pre"]


def test_non_swa_idle_flush_never_calls_ledger_api():
    events = []
    tree = _FlushTree(events, supports_swa=False)

    def forbidden_ledger(*args, **kwargs):
        raise AssertionError("non-SWA flush must not call cache_ledger_snapshot")

    tree.cache_ledger_snapshot = forbidden_ledger
    scheduler = _flush_scheduler(tree)

    with patch.object(ScheduleBatch, "init_new", return_value=object()):
        result = Scheduler.flush_cache(scheduler)

    assert result[0] is True
    assert events == ["tree-reset", "req-clear", "alloc-clear"]


def test_non_idle_flush_skips_ledger_and_all_mutation():
    events = []
    tree = _FlushTree(events)
    scheduler = _flush_scheduler(tree)
    scheduler._can_flush_cache = lambda: (False, "busy")

    result = Scheduler.flush_cache(scheduler)

    assert result == (False, "busy", 0)
    assert events == []


def test_schedule_batch_accepts_unified_swa_capability():
    class _SWAAllocator:
        pass

    tree = SimpleNamespace(supports_swa=lambda: True)
    with patch(
        "sgl_jax.srt.managers.schedule_batch.SWATokenToKVPoolAllocator",
        _SWAAllocator,
    ):
        batch = ScheduleBatch.init_new(
            reqs=[[]],
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=_SWAAllocator(),
            tree_cache=tree,
            model_config=SimpleNamespace(vocab_size=32),
            enable_overlap=False,
            dp_size=1,
        )

    assert batch.is_hybrid is True
    assert batch.tree_cache is tree


@pytest.mark.parametrize("route_name", ["UnifiedRadixCache", "SWAChunkCache"])
def test_decode_oom_preserves_capacity_error_without_legacy_lru_stats(route_name):
    """Optional legacy diagnostics must not replace the allocation failure."""

    class _HybridTree:
        def full_evictable_size(self, dp_rank=0):
            assert dp_rank == 0
            return 2

        def swa_evictable_size(self, dp_rank=0):
            assert dp_rank == 0
            return 3

    tree = type(route_name, (_HybridTree,), {})()
    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.dp_size = 1
    batch.is_hybrid = True
    batch.reqs_info = [SimpleNamespace(reqs=[object()])]
    batch.tree_cache = tree
    batch.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=1,
        alloc_decode=lambda *_args, **_kwargs: None,
        full_available_size=lambda dp_rank=0: 5,
        swa_available_size=lambda dp_rank=0: 7,
    )
    batch._evict_tree_cache_if_needed = lambda _needed: None

    with pytest.raises(RuntimeError, match="Decode out of memory") as exc_info:
        batch.alloc_paged_token_slots_decode([1], [0])

    message = str(exc_info.value)
    assert "Available full tokens: 7" in message
    assert "Available swa tokens: 10" in message


def test_decode_oom_survives_failing_optional_legacy_diagnostic():
    class _HybridTree:
        def full_evictable_size(self, dp_rank=0):
            assert dp_rank == 0
            return 2

        def swa_evictable_size(self, dp_rank=0):
            assert dp_rank == 0
            return 3

        def full_lru_list_evictable_size(self):
            raise ValueError("diagnostic traversal failed")

        def swa_lru_list_evictable_size(self):
            return 4

    batch = ScheduleBatch.__new__(ScheduleBatch)
    batch.dp_size = 1
    batch.is_hybrid = True
    batch.reqs_info = [SimpleNamespace(reqs=[object()])]
    batch.tree_cache = _HybridTree()
    batch.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=1,
        alloc_decode=lambda *_args, **_kwargs: None,
        full_available_size=lambda dp_rank=0: 5,
        swa_available_size=lambda dp_rank=0: 7,
    )
    batch._evict_tree_cache_if_needed = lambda _needed: None

    with pytest.raises(RuntimeError, match="Decode out of memory") as exc_info:
        batch.alloc_paged_token_slots_decode([1], [0])

    message = str(exc_info.value)
    assert "Full LRU list evictable size: unavailable (ValueError)" in message
    assert "SWA LRU list evictable size: 4" in message


def test_unfinished_then_finished_preserves_swa_lock_uuid_symmetry():
    """An unfinished refresh receipt must be the exact finished release receipt."""
    releases = []

    class _ReceiptChunk(ChunkCache):
        def dec_lock_ref(self, _node, params=None):
            if params is not None:
                releases.append(params)
            return 0

        def inc_lock_ref(self, _node):
            return IncLockRefResult(
                swa_uuid_for_lock=23,
                skip_lock_node_ids={"swa": [9]},
            )

    allocator = SimpleNamespace(free=lambda *_args, **_kwargs: None)
    cache = _ReceiptChunk(
        SimpleNamespace(req_to_token=np.asarray([[4, 5]], dtype=np.int32)),
        allocator,
        page_size=1,
    )

    class _Req:
        rid = "r"
        req_pool_idx = 0
        dp_rank = 0
        fill_ids = [1, 2]
        last_node = object()
        cache_lock_params = DecLockRefParams(swa_uuid_for_lock=17)
        swa_uuid_for_lock = 17

        def pop_committed_kv_cache(self):
            return 2

    req = _Req()

    cache.cache_unfinished_req(req)
    refreshed = req.cache_lock_params
    cache.cache_finished_req(req)

    assert releases[0].swa_uuid_for_lock == 17
    assert releases[1] is refreshed
    assert refreshed.swa_uuid_for_lock == 23
    assert refreshed.skip_lock_node_ids == {"swa": [9]}
    assert req.cache_lock_params is None
    assert req.swa_uuid_for_lock is None


def test_retract_abort_release_request_owned_slots_once():
    """Both scheduler reclaim paths delegate a request exactly once."""
    calls = []
    tree = SimpleNamespace(
        supports_swa=lambda: True,
        evict_req_swa=lambda req, pre_len, dp_rank=0: calls.append((id(req), pre_len, dp_rank)),
    )
    batch = SimpleNamespace(tree_cache=tree)
    req = SimpleNamespace(swa_evicted_seqlen=0)

    ScheduleBatch._evict_swa(batch, req, 16, 4, 1, dp_rank=1)

    assert calls == [(id(req), 16, 1)]


def test_maybe_evict_swa_does_not_direct_free_tree_owned_prefix():
    calls = []
    tree = SimpleNamespace(
        supports_swa=lambda: True,
        evict_req_swa=lambda req, pre_len, dp_rank=0: calls.append((req.rid, pre_len, dp_rank)),
    )
    req = SimpleNamespace(rid="r", decode_batch_idx=4, seqlen=16)
    allocator = SimpleNamespace(
        page_size=1,
        free_swa=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("scheduler must not directly free SWA slots")
        ),
    )
    batch = SimpleNamespace(
        is_hybrid=True,
        model_config=SimpleNamespace(sliding_window=4),
        token_to_kv_pool_allocator=allocator,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        reqs_info=[SimpleNamespace(reqs=[req])],
        tree_cache=tree,
        enable_overlap=False,
    )
    batch._evict_swa = lambda req, pre_len, window, page, dp_rank=0: (
        ScheduleBatch._evict_swa(batch, req, pre_len, window, page, dp_rank)
    )

    ScheduleBatch.maybe_evict_swa(batch)

    assert calls == [("r", 15, 0)]


def test_swa_size_accessors_accept_dp_rank():
    calls = []
    tree = SimpleNamespace(
        full_evictable_size=lambda dp_rank=0: calls.append(("full", dp_rank)) or 2,
        swa_evictable_size=lambda dp_rank=0: calls.append(("swa", dp_rank)) or 3,
    )
    scheduler = SimpleNamespace(
        dp_size=2,
        tree_cache=tree,
        token_to_kv_pool_allocator=SimpleNamespace(
            full_available_size=lambda rank: 10,
            swa_available_size=lambda rank: 11,
        ),
        full_tokens_per_layer=24,
        swa_tokens_per_layer=28,
    )

    Scheduler._get_swa_token_info(scheduler)

    assert calls == [("full", 0), ("full", 1), ("swa", 0), ("swa", 1)]


def test_idle_hot_path_uses_counters_without_full_ledger_scan():
    calls = []

    def counter(name, value):
        return lambda dp_rank=0: calls.append((name, dp_rank)) or value

    def forbidden(*_args, **_kwargs):
        raise AssertionError("idle hot path must not scan the ownership ledger")

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._can_flush_cache = lambda: (True, "")
    scheduler.dp_size = 1
    scheduler.is_hybrid = True
    scheduler.tree_cache = SimpleNamespace(
        supports_swa=lambda: True,
        full_evictable_size=counter("full_evictable", 0),
        full_protected_size=counter("full_protected", 0),
        swa_evictable_size=counter("swa_evictable", 0),
        swa_protected_size=counter("swa_protected", 0),
        cache_ledger_snapshot=forbidden,
    )
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        full_attn_allocator=SimpleNamespace(size_per_rank=8, page_size=1),
        swa_attn_allocator=SimpleNamespace(size_per_rank=8, page_size=1),
        full_available_size=counter("full_available", 8),
        swa_available_size=counter("swa_available", 8),
    )
    scheduler.req_to_token_pool = SimpleNamespace(size=0, free_slots=[])
    scheduler._snapshot_swa_cache_ledger = forbidden
    scheduler._cache_ledger_readonly_fingerprint = forbidden
    scheduler.init_new_token_ratio = 0.75
    scheduler.new_token_ratio = 0.25

    Scheduler.on_idle(scheduler)

    assert scheduler.new_token_ratio == 0.75
    assert calls == [
        ("full_available", 0),
        ("full_evictable", 0),
        ("full_protected", 0),
        ("swa_available", 0),
        ("swa_evictable", 0),
        ("swa_protected", 0),
    ]


def test_idle_memory_check_passes_balanced_unified_swa_counters():
    scheduler = _flush_scheduler(_FlushTree([]))
    scheduler.req_to_token_pool = SimpleNamespace(size=0, free_slots=[])

    Scheduler.check_memory(scheduler)


def test_idle_memory_check_names_rank_and_pool_on_leak():
    scheduler = _flush_scheduler(_FlushTree([], invalid_pre=True))
    scheduler.req_to_token_pool = SimpleNamespace(size=0, free_slots=[])

    with pytest.raises(ValueError, match=r"dp=0.*full"):
        Scheduler.check_memory(scheduler)


def _paged_idle_scheduler(
    *,
    full_allocator,
    swa_allocator,
    full_available,
    swa_available,
    full_owned,
    swa_owned,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.dp_size = 1
    scheduler.is_hybrid = True
    scheduler.tree_cache = SimpleNamespace(
        full_evictable_size=lambda dp_rank=0: full_owned,
        full_protected_size=lambda dp_rank=0: 0,
        swa_evictable_size=lambda dp_rank=0: swa_owned,
        swa_protected_size=lambda dp_rank=0: 0,
    )
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        full_attn_allocator=full_allocator,
        swa_attn_allocator=swa_allocator,
        full_available_size=lambda dp_rank=0: full_available,
        swa_available_size=lambda dp_rank=0: swa_available,
    )
    scheduler.req_to_token_pool = SimpleNamespace(size=0, free_slots=[])
    return scheduler


def test_paged_idle_uses_each_allocator_page_size_and_usable_capacity():
    """FULL and SWA bounds come from their own child allocator, not wrapper assumptions."""
    scheduler = _paged_idle_scheduler(
        full_allocator=SimpleNamespace(size_per_rank=8, page_size=1),
        swa_allocator=SimpleNamespace(size_per_rank=17, pages_per_rank=3, page_size=4),
        full_available=4,
        swa_available=8,
        full_owned=4,
        swa_owned=1,
    )

    Scheduler.check_memory(scheduler)


@pytest.mark.parametrize(
    ("full_allocator", "full_available", "full_owned", "message"),
    [
        (
            SimpleNamespace(size_per_rank=8, pages_per_rank=2, page_size=4),
            4,
            5,
            r"owned=5.*reserved_capacity=4",
        ),
        (
            SimpleNamespace(size_per_rank=10, page_size=4),
            8,
            2,
            r"capacity=10.*page_size=4.*page-aligned",
        ),
        (
            SimpleNamespace(size_per_rank=8, pages_per_rank=2, page_size=4),
            6,
            2,
            r"available=6.*page_size=4.*page-aligned",
        ),
        (
            SimpleNamespace(size_per_rank=8, pages_per_rank=2, page_size=4),
            12,
            0,
            r"available=12.*capacity=8",
        ),
        (
            SimpleNamespace(size_per_rank=8, pages_per_rank=2, page_size=4),
            0,
            1,
            r"owned=1.*reserved_pages=2",
        ),
    ],
    ids=(
        "owned-exceeds-reserved-capacity",
        "capacity-not-page-aligned",
        "available-not-page-aligned",
        "available-out-of-range",
        "owned-below-reserved-pages",
    ),
)
def test_idle_memory_rejects_invalid_paged_pool_state(
    full_allocator, full_available, full_owned, message
):
    scheduler = _paged_idle_scheduler(
        full_allocator=full_allocator,
        swa_allocator=SimpleNamespace(size_per_rank=8, pages_per_rank=2, page_size=4),
        full_available=full_available,
        swa_available=8,
        full_owned=full_owned,
        swa_owned=0,
    )

    with pytest.raises(ValueError, match=message):
        Scheduler.check_memory(scheduler)


def test_paged_idle_accepts_real_legacy_partial_cached_page_without_ledger_scan():
    """One live owner in a reserved page leaves legitimate slack, not a leak."""
    mesh = Mesh(
        np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    kv_pool = SWAKVPool(
        size=1024,
        size_swa=1024,
        page_size=128,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        token_to_kv_pool_class=MHATokenToKVPool,
        dtype=jax.numpy.bfloat16,
        head_num=1,
        head_dim=1,
        mesh=mesh,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=1024,
        size_swa=1024,
        kvcache=kv_pool,
        page_size=128,
    )
    req_pool = ReqToTokenPool(size=1, max_context_len=128)
    cache = SWARadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        sliding_window_size=128,
        page_size=128,
    )
    full_indices = allocator.alloc_extend(
        prefix_lens=[0],
        seq_lens=[1],
        last_loc=[0],
        extend_num_tokens=1,
    )
    assert full_indices is not None
    cache.insert(InsertParams(key=RadixKey([1]), value=full_indices))

    snapshot = cache.cache_ledger_snapshot(0, [])
    assert snapshot["full_capacity"] == snapshot["swa_capacity"] == 1024
    assert snapshot["full_available"] == snapshot["swa_available"] == 896
    assert snapshot["full_tree_evictable"] == snapshot["swa_tree_evictable"] == 1
    assert snapshot["full_reserved_page_slack"] == 127
    assert snapshot["swa_reserved_page_slack"] == 127
    validate_swa_cache_ledger(snapshot, require_idle=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("idle hot path must not scan the ownership ledger")

    cache.cache_ledger_snapshot = forbidden
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._can_flush_cache = lambda: (True, "")
    scheduler.dp_size = 1
    scheduler.is_hybrid = True
    scheduler.tree_cache = cache
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.req_to_token_pool = SimpleNamespace(size=0, free_slots=[])
    scheduler._snapshot_swa_cache_ledger = forbidden
    scheduler._cache_ledger_readonly_fingerprint = forbidden
    scheduler.init_new_token_ratio = 0.75
    scheduler.new_token_ratio = 0.25

    Scheduler.on_idle(scheduler)

    assert scheduler.new_token_ratio == 0.75


def test_flush_emits_preflush_and_postflush_snapshots():
    events = []
    scheduler = _flush_scheduler(_FlushTree(events))

    with patch.object(ScheduleBatch, "init_new", return_value=object()):
        Scheduler.flush_cache(scheduler)

    assert [event for event in events if event.startswith("ledger:")] == [
        "ledger:pre",
        "ledger:post",
    ]


class _LedgerCache:
    page_size = 128
    sliding_window_size = 4096
    tree_components = ("FULL", "SWA")

    def __init__(self):
        self.allocator_available = [8, 8]

    def supports_swa(self):
        return True

    def cache_ledger_snapshot(self, dp_rank, live_reqs):
        row = _empty_ledger()
        row["dp_rank"] = dp_rank
        row["seen_rids"] = [req.rid for req in live_reqs]
        return row


def _ledger_scheduler(tree=None):
    tree = tree or _LedgerCache()
    req0 = SimpleNamespace(rid="r0", dp_rank=0, req_pool_idx=0)
    req1 = SimpleNamespace(rid="r1", dp_rank=1, req_pool_idx=1)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = tree
    scheduler.dp_size = 2
    scheduler.page_size = 128
    scheduler.sliding_window_size = 4096
    scheduler.is_hybrid = True
    scheduler.waiting_queue = [req0]
    scheduler.grammar_queue = []
    scheduler.pending_dp_reqs = []
    scheduler.running_batch = SimpleNamespace(
        reqs_info=[SimpleNamespace(reqs=[]), SimpleNamespace(reqs=[req1])],
        is_empty=lambda: False,
        batch_size=lambda: 1,
    )
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler.chunked_reqs = [None, None]
    scheduler._pending_chunked_abort_reqs = [None, None]
    scheduler.enable_overlap = False
    scheduler.disagg_prefill_queue = []
    scheduler.disagg_prealloc_queue = []
    scheduler.disagg_transfer_queue = []
    scheduler._pd_pending_bootstrap = {}
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        full_available_size=lambda rank: tree.allocator_available[rank],
        swa_available_size=lambda rank: tree.allocator_available[rank],
    )
    return scheduler


def test_internal_ledger_snapshot_returns_all_dp_ranks():
    scheduler = _ledger_scheduler()
    ranks = Scheduler._snapshot_swa_cache_ledger(scheduler, require_idle=False)

    assert [row["dp_rank"] for row in ranks] == [0, 1]
    assert ranks[0]["seen_rids"] == ["r0"]
    assert ranks[1]["seen_rids"] == ["r1"]


def test_internal_ledger_snapshot_rejects_duplicate_live_rid_objects():
    scheduler = _ledger_scheduler()
    duplicate = SimpleNamespace(rid="r0", dp_rank=0, req_pool_idx=2)
    scheduler.grammar_queue = [duplicate]

    with pytest.raises(RuntimeError, match="Multiple live request objects share rid='r0'"):
        Scheduler._snapshot_swa_cache_ledger(scheduler, require_idle=False)


def test_internal_ledger_snapshot_is_read_only():
    class MutatingCache(_LedgerCache):
        def cache_ledger_snapshot(self, dp_rank, live_reqs):
            self.allocator_available[dp_rank] -= 1
            return super().cache_ledger_snapshot(dp_rank, live_reqs)

    with pytest.raises(RuntimeError, match="mutated cache ownership state"):
        Scheduler._snapshot_swa_cache_ledger(_ledger_scheduler(MutatingCache()), require_idle=False)


def test_internal_ledger_snapshot_detects_event_container_key_mutation():
    class EventKeyMutatingCache(_LedgerCache):
        def __init__(self):
            super().__init__()
            self._ledger_event_totals = {}

        def cache_ledger_snapshot(self, dp_rank, live_reqs):
            self._ledger_event_totals.setdefault(dp_rank, {})
            return super().cache_ledger_snapshot(dp_rank, live_reqs)

    with pytest.raises(RuntimeError, match="mutated cache ownership state"):
        Scheduler._snapshot_swa_cache_ledger(
            _ledger_scheduler(EventKeyMutatingCache()), require_idle=False
        )
