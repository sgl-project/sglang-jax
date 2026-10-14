"""PrefillAdder must let HiCache reload a host-only prefix under memory pressure.

Regression for run15: ``add_one_req`` passed ``mem_quota = allocator.available_size``
to ``init_load_back``. A prefix is only host-resident *because* HBM is full, so
free HBM was ~0 whenever a restore was possible and ``init_load_back`` bailed on
its quota check before reaching its own evict-to-make-room path. Every would-be
restore became a cold prefill. The quota must be the admission budget
(free + evictable - offset), the same number the request was admitted against.
"""

import os

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.mem_cache.base_prefix_cache import DecLockRefParams, IncLockRefResult
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

FREE = 0  # HBM completely full ...
EVICTABLE = 100  # ... but plenty of cold, evictable cache entries
HOST_HIT = 8


def _tree(calls: list, restore_tokens: int):
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.disable = False
    tree.hicache_enabled = True
    tree.inc_lock_ref = lambda node: IncLockRefResult(swa_uuid_for_lock=None, skip_lock_node_ids={})
    tree.dec_lock_ref = lambda node, params: None
    tree.recurrent_extra_buffer_active = lambda: False
    tree.evictable_size = lambda dp_rank=0: EVICTABLE
    host_node = SimpleNamespace(name="host-node")

    def init_load_back(last_host_node, host_hit_length, mem_quota=None):
        calls.append(mem_quota)
        # Mirror the real all-or-nothing gate so the test fails the way prod did.
        if mem_quota is not None and restore_tokens > mem_quota:
            return np.empty((0,), dtype=np.int32), last_host_node, []
        return np.arange(restore_tokens, dtype=np.int32), host_node, [([1, 2], [3, 4])]

    tree.init_load_back = init_load_back
    return tree, host_node


def _req():
    return SimpleNamespace(
        rid="probe-r0",
        dp_rank=0,
        extend_input_len=16,
        host_hit_length=HOST_HIT,
        prefix_indices=np.empty((0,), dtype=np.int32),
        last_node=object(),
        last_host_node=SimpleNamespace(name="tombstone"),
        fill_ids=list(range(16)),
        origin_input_ids=list(range(16)),
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=1, ignore_eos=False),
    )


def _adder(tree):
    return PrefillAdder(
        page_size=1,
        tree_cache=tree,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda dp_rank=0: FREE),
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=64,
        rem_chunk_tokens=None,
    )


def test_quota_is_admission_budget_not_free_hbm():
    calls = []
    tree, host_node = _tree(calls, restore_tokens=HOST_HIT)
    adder = _adder(tree)
    req = _req()

    result = adder.add_one_req(req)

    assert result is AddReqResult.CONTINUE
    # Quota seen by init_load_back == admission budget at call time (offset was 0).
    assert calls == [FREE + EVICTABLE]  # not FREE (== 0)
    # The reload was applied: prefix grew, extend shrank, H2D plan queued.
    assert len(req.prefix_indices) == HOST_HIT
    assert req.extend_input_len == 16 - HOST_HIT
    assert req.host_hit_length == 0
    assert req.last_node is host_node and req.last_host_node is host_node
    assert adder.pending_h2d == [([1, 2], [3, 4])]


def test_free_only_quota_would_have_refused_the_reload():
    """Documents the pre-fix failure mode: with quota == free HBM (0) the stub
    returns nothing and the request is admitted as a cold prefill."""
    calls = []
    tree, _ = _tree(calls, restore_tokens=HOST_HIT)
    tree.init_load_back = lambda n, h, mem_quota=None: tree_init(n, h, FREE)  # noqa: E731
    real = _tree(calls, restore_tokens=HOST_HIT)[0].init_load_back

    def tree_init(n, h, quota):
        return real(n, h, mem_quota=quota)

    adder = _adder(tree)
    req = _req()
    assert adder.add_one_req(req) is AddReqResult.CONTINUE
    assert len(req.prefix_indices) == 0  # cold prefill
    assert req.extend_input_len == 16
    assert adder.pending_h2d == []
