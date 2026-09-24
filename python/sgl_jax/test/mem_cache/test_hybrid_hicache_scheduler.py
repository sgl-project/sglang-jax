"""Admission tests for independently sized FULL and SWA L2 restoration."""

from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import IncLockRefResult
from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType as CT
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


def admission(*, swa_capacity=50, restored_swa=8, chunk=None, succeeds=True):
    allocator = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
    allocator.full_available_size = lambda dp_rank=0: 100
    allocator.swa_available_size = lambda dp_rank=0: swa_capacity
    allocator.available_size = lambda dp_rank=0: min(100, swa_capacity)
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.tree_components = (CT.FULL, CT.SWA)
    tree.components = {CT.SWA: SimpleNamespace(sliding_window_size=4)}
    tree.hicache_enabled = True
    tree.hicache_controller = SimpleNamespace(direct_transfers=False)
    tree.component_init_params = SimpleNamespace(sliding_window_size=4)
    tree.disable = False
    tree.recurrent_extra_buffer_active = lambda: False
    tree.inc_lock_ref = lambda node: IncLockRefResult()
    tree.dec_lock_ref = lambda node, receipt: None
    tree.full_evictable_size = lambda dp_rank=0: 0
    tree.swa_evictable_size = lambda dp_rank=0: 0
    tree.get_load_back_sizes = lambda node: (32, restored_swa)
    calls = []

    def restore(node, length, **quotas):
        calls.append(quotas)
        return (np.arange(32) if succeeds else np.array([], dtype=np.int32)), node, []

    tree.init_load_back = restore
    adder = PrefillAdder(
        page_size=1,
        tree_cache=tree,
        token_to_kv_pool_allocator=allocator,
        running_batch=None,
        new_token_ratio=1,
        rem_input_tokens=100,
        rem_chunk_tokens=chunk,
    )
    req = SimpleNamespace(
        dp_rank=0,
        extend_input_len=33,
        host_hit_length=32,
        swa_host_hit_length=restored_swa,
        prefix_indices=[],
        last_node=object(),
        last_host_node=object(),
        fill_ids=list(range(33)),
        origin_input_ids=list(range(33)),
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=1, ignore_eos=False),
    )
    return adder, req, calls


def test_long_full_restore_does_not_require_equal_swa_capacity():
    adder, req, calls = admission(swa_capacity=24)
    assert adder.add_one_req(req) is AddReqResult.CONTINUE
    assert calls == [{"mem_quota": 100, "swa_mem_quota": 24}]
    assert req.swa_host_hit_length == 0
    assert req.extend_input_len == 1
    assert adder.rem_swa_token_offset[0] == adder._swa_budget_for_req(1, 0)


@pytest.mark.parametrize("restored_swa", [36, 41])
def test_restore_window_overhang_falls_back_to_recompute(restored_swa):
    adder, req, calls = admission(swa_capacity=40, restored_swa=restored_swa)
    assert adder.add_one_req(req) is AddReqResult.CONTINUE
    assert not calls
    assert req.extend_input_len == 33
    assert req.host_hit_length == req.swa_host_hit_length == 0
    assert req.last_host_node is req.last_node


def test_restore_overhang_can_recompute_one_chunk():
    adder, req, calls = admission(swa_capacity=24, restored_swa=25, chunk=8)
    assert adder.add_one_req(req) is AddReqResult.CONTINUE
    assert not calls
    assert adder.can_run_list[0] == [req]
    assert req.extend_input_len == 8
    assert req.host_hit_length == req.swa_host_hit_length == 0


def test_restore_overhang_rejects_when_recompute_also_exceeds_capacity():
    adder, req, calls = admission(swa_capacity=24, restored_swa=25)
    assert adder.add_one_req(req) is AddReqResult.NO_TOKEN
    assert not calls
    assert not adder.can_run_list[0]


def test_chunk_rejection_does_not_start_restore():
    adder, req, calls = admission(chunk=0)
    assert adder.add_one_req(req) is AddReqResult.OTHER
    assert not calls


def test_failed_restore_rechecks_recompute_swa_budget():
    adder, req, calls = admission(swa_capacity=24, succeeds=False)
    assert adder.add_one_req(req) is AddReqResult.NO_TOKEN
    assert len(calls) == 1
    assert not adder.can_run_list[0]


def test_retract_discards_both_host_candidates():
    req = Req.__new__(Req)
    req.last_host_node = object()
    req.host_hit_length = 32
    req.swa_host_hit_length = 8
    req.reset_for_retract()
    assert req.last_host_node is None
    assert req.host_hit_length == req.swa_host_hit_length == 0
