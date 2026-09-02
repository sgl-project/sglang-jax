"""Scheduler-facing tests for the Unified FULL+SWA integration seam."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sgl_jax.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.mem_cache.base_prefix_cache import DecLockRefParams, IncLockRefResult
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class _FlushTree:
    page_size = 1

    def __init__(self, events):
        self.events = events

    def reset(self):
        self.events.append("tree-reset")


class _Clearable:
    def __init__(self, events, label):
        self.events = events
        self.label = label

    def clear(self):
        self.events.append(self.label)


class _Allocator(_Clearable):
    def available_size(self, _dp_rank=0):
        return 8


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


def _run_unified_prefill_admission():
    releases = []
    lock_result = IncLockRefResult(
        swa_uuid_for_lock=17,
        skip_lock_node_ids={"swa": [3, 5]},
    )
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.disable = False
    tree.hicache_enabled = False
    tree.inc_lock_ref = lambda node: lock_result
    tree.dec_lock_ref = lambda node, params: releases.append(params)
    tree.recurrent_extra_buffer_active = lambda: False
    tree.evictable_size = lambda dp_rank=0: 0
    adder = PrefillAdder(
        page_size=1,
        tree_cache=tree,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda dp_rank=0: 100),
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

    return result, req, releases


def test_unified_prefill_admission_stores_the_complete_lock_receipt():
    result, req, releases = _run_unified_prefill_admission()

    assert result is AddReqResult.CONTINUE
    assert req.cache_lock_params.swa_uuid_for_lock == 17
    assert req.cache_lock_params.skip_lock_node_ids == {"swa": [3, 5]}
    assert req.swa_uuid_for_lock == 17
    assert releases[0].skip_lock_node_ids == {"swa": [3, 5]}


def test_idle_flush_resets_cache_without_a_diagnostic_gate():
    events = []
    scheduler = _flush_scheduler(_FlushTree(events))

    with patch.object(ScheduleBatch, "init_new", return_value=object()):
        result = Scheduler.flush_cache(scheduler)

    assert result == (True, "", 8)
    assert events == ["tree-reset", "req-clear", "alloc-clear"]


def test_non_idle_flush_skips_all_mutation():
    events = []
    scheduler = _flush_scheduler(_FlushTree(events))
    scheduler._can_flush_cache = lambda: (False, "busy")

    result = Scheduler.flush_cache(scheduler)

    assert result == (False, "busy", 0)
    assert events == []


def _decode_batch(tree, req, *, enable_overlap=False):
    batch = SimpleNamespace(
        is_hybrid=True,
        model_config=SimpleNamespace(sliding_window=4),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        reqs_info=[SimpleNamespace(reqs=[req])],
        tree_cache=tree,
        enable_overlap=enable_overlap,
    )
    batch._evict_swa = lambda request, pre_len, window, page, dp_rank=0: ScheduleBatch._evict_swa(
        batch, request, pre_len, window, page, dp_rank
    )
    return batch


def test_unified_decode_delegates_request_tail_reclaim_to_cache():
    calls = []
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.evict_req_swa = lambda req, pre_len, dp_rank=0: calls.append((req.rid, pre_len, dp_rank))
    req = SimpleNamespace(rid="r", decode_batch_idx=4, seqlen=16)
    batch = _decode_batch(tree, req)

    ScheduleBatch.maybe_evict_swa(batch)

    assert calls == [("r", 15, 0)]


def test_unified_overlap_decode_waits_for_safe_offset():
    calls = []
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.evict_req_swa = lambda req, pre_len, dp_rank=0: calls.append(
        (req.decode_batch_idx, pre_len, dp_rank)
    )
    req = SimpleNamespace(decode_batch_idx=0, seqlen=8)
    batch = _decode_batch(tree, req, enable_overlap=True)
    batch.model_config.sliding_window = 1

    ScheduleBatch.maybe_evict_swa(batch)
    req.decode_batch_idx = 1
    ScheduleBatch.maybe_evict_swa(batch)

    assert calls == [(1, 7, 0)]


def test_unified_extend_delegates_request_tail_reclaim_to_cache():
    calls = []
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.evict_req_swa = lambda req, pre_len, dp_rank=0: calls.append((req.rid, pre_len, dp_rank))
    req = SimpleNamespace(rid="r", extend_batch_idx=0)
    batch = SimpleNamespace(
        is_hybrid=True,
        model_config=SimpleNamespace(sliding_window=4),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        forward_mode=SimpleNamespace(is_decode=lambda: False, is_extend=lambda: True),
        reqs_info=[SimpleNamespace(reqs=[req], prefix_lens=[12])],
        tree_cache=tree,
        enable_overlap=False,
    )
    batch._evict_swa = lambda request, pre_len, window, page, dp_rank=0: ScheduleBatch._evict_swa(
        batch, request, pre_len, window, page, dp_rank
    )

    with patch.dict(global_server_args_dict, {"chunked_prefill_size": None}):
        ScheduleBatch.maybe_evict_swa(batch)

    assert calls == [("r", 12, 0)]


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
    scheduler.tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    scheduler.tree_cache.full_evictable_size = lambda dp_rank=0: full_owned
    scheduler.tree_cache.full_protected_size = lambda dp_rank=0: 0
    scheduler.tree_cache.swa_evictable_size = lambda dp_rank=0: swa_owned
    scheduler.tree_cache.swa_protected_size = lambda dp_rank=0: 0
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        full_attn_allocator=full_allocator,
        swa_attn_allocator=swa_allocator,
        full_available_size=lambda dp_rank=0: full_available,
        swa_available_size=lambda dp_rank=0: swa_available,
    )
    scheduler.req_to_token_pool = SimpleNamespace(size=0, free_slots=[])
    return scheduler


def test_paged_idle_uses_each_allocator_page_size_and_usable_capacity():
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
            r"\[dp=0\]\[full\].*owned=5.*reserved_capacity=4",
        ),
        (
            SimpleNamespace(size_per_rank=10, page_size=4),
            8,
            2,
            r"\[dp=0\]\[full\].*capacity=10.*page_size=4.*page-aligned",
        ),
        (
            SimpleNamespace(size_per_rank=8, pages_per_rank=2, page_size=4),
            6,
            2,
            r"\[dp=0\]\[full\].*available=6.*page_size=4.*page-aligned",
        ),
        (
            SimpleNamespace(size_per_rank=8, pages_per_rank=2, page_size=4),
            12,
            0,
            r"\[dp=0\]\[full\].*available=12.*capacity=8",
        ),
        (
            SimpleNamespace(size_per_rank=8, pages_per_rank=2, page_size=4),
            0,
            1,
            r"\[dp=0\]\[full\].*owned=1.*reserved_pages=2",
        ),
    ],
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
