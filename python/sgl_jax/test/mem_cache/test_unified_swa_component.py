"""Device-only SWA coverage for the unified radix component seam."""

# ruff: noqa: E402

from __future__ import annotations

import os
import unittest
from collections import defaultdict
from types import SimpleNamespace

# Must precede the first direct JAX import: this module owns a 2x2 CPU mesh.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

assert jax.device_count() == 4

from sgl_jax.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    global_server_args_dict,
)
from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
    build_swa_cache_ledger_snapshot,
    validate_swa_cache_ledger,
)
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.chunk_cache import SWAChunkCache
from sgl_jax.srt.mem_cache.common import release_kv_cache
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache
from sgl_jax.srt.mem_cache.unified_cache_components import (
    ComponentType,
    LRURefreshPhase,
)
from sgl_jax.srt.mem_cache.unified_radix_cache import (
    COMPONENT_REGISTRY,
    UnifiedRadixCache,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.sampling.sampling_params import SamplingParams


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices(), dtype=object).reshape(2, 2)
    mesh = Mesh(devices, axis_names=("data", "tensor"))
    jax.sharding.set_mesh(mesh)
    return mesh


def _pool_mesh() -> Mesh:
    """DP1 paged capacity has an odd physical-page dimension on data=2."""
    return Mesh(
        np.asarray(jax.devices()[:2], dtype=object).reshape(1, 2), axis_names=("data", "tensor")
    )


def _make_cache(
    *,
    page_size: int = 1,
    dp_size: int = 1,
    window: int = 4,
    kind: str = "unified",
    size: int | None = None,
    size_swa: int | None = None,
):
    _mesh()  # The module's test world is always an explicit 2x2 mesh.
    size = size or max(512, page_size * 8)
    size_swa = size if size_swa is None else size_swa
    pool = SWAKVPool(
        size=size,
        size_swa=size_swa,
        page_size=page_size,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        token_to_kv_pool_class=MHATokenToKVPool,
        dtype=jnp.bfloat16,
        head_num=2,
        head_dim=1,
        mesh=_mesh() if dp_size == 2 else _pool_mesh(),
        dp_size=dp_size,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=size, size_swa=size_swa, kvcache=pool, page_size=page_size, dp_size=dp_size
    )
    req_pool = ReqToTokenPool(size=8, max_context_len=size, dtype=np.int32)
    component_params = CacheInitParams(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        sliding_window_size=window,
    )
    if kind == "legacy":
        cache = SWARadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            sliding_window_size=window,
            page_size=page_size,
        )
    elif kind == "chunk":
        cache = SWAChunkCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            sliding_window_size=window,
        )
    else:
        cache = UnifiedRadixCache(
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            kv_head_num=2,
            head_dim=1,
            layer_num=2,
            max_seq_len=size,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
            component_init_params=component_params,
        )
    return cache, allocator


_LIFECYCLE_MODEL_CONFIG = SimpleNamespace(vocab_size=4096, sliding_window=8)


def _run_scheduler_lifecycle(
    cache,
    allocator,
    *,
    rid: str,
    input_ids: list[int],
    output_len: int = 2,
) -> None:
    """Run one request through the real scheduler/cache ownership path."""
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=SamplingParams(
            max_new_tokens=output_len,
            temperature=0,
            top_p=1.0,
            ignore_eos=True,
        ),
        dp_rank=0,
        eos_token_ids={1},
        vocab_size=_LIFECYCLE_MODEL_CONFIG.vocab_size,
    )
    req.init_next_round_input(cache)
    adder = PrefillAdder(
        page_size=cache.page_size,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=SimpleNamespace(reqs_info=[SimpleNamespace(reqs=[])]),
        new_token_ratio=0.7,
        rem_input_tokens=128,
        rem_chunk_tokens=64,
        dp_size=1,
    )
    assert adder.add_one_req(req) == AddReqResult.CONTINUE
    assert adder.can_run_list[0] == [req]

    batch = ScheduleBatch.init_new(
        reqs=[[req]],
        req_to_token_pool=cache.req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        tree_cache=cache,
        model_config=_LIFECYCLE_MODEL_CONFIG,
        enable_overlap=True,
        dp_size=1,
        spec_algorithm=None,
        mesh=None,
    )
    batch.prepare_for_extend()
    info = batch.reqs_info[0]
    info.req_pool_indices = np.asarray([req.req_pool_idx], dtype=np.int32)
    info.seq_lens = np.asarray([len(req.origin_input_ids)], dtype=np.int32)
    info.seq_lens_sum = int(info.seq_lens.sum())
    info.sampling_info = SamplingBatchInfo.from_schedule_batch(
        info,
        _LIFECYCLE_MODEL_CONFIG.vocab_size,
        batch=batch,
    )
    req.output_ids = [3000]
    for _ in range(output_len - 1):
        info.output_ids = np.asarray([3000], dtype=np.int32)
        batch.prepare_for_decode()
        req.output_ids.append(3000)
    release_kv_cache(req, cache)


_LEDGER_FIELDS = {
    "dp_rank",
    "full_capacity",
    "full_available",
    "full_tree_evictable",
    "full_tree_protected",
    "full_request_owned",
    "full_reserved_page_slack",
    "swa_capacity",
    "swa_available",
    "swa_tree_evictable",
    "swa_tree_protected",
    "swa_request_owned",
    "swa_reserved_page_slack",
    "mapping_nonzero_count",
    "mapping_invalid_count",
    "mapping_duplicate_count",
    "full_duplicate_request_owner_count",
    "swa_duplicate_request_owner_count",
    "full_request_tree_overlap_count",
    "swa_request_tree_overlap_count",
    "full_evicted_total",
    "swa_evicted_total",
    "tombstone_created_total",
    "tombstone_healed_total",
}


class _RequestOwnership:
    """Independent request-owner record used by the conservation assertions."""

    def __init__(self):
        self.full: dict[int, set[int]] = defaultdict(set)
        self.swa: dict[int, set[int]] = defaultdict(set)

    def record_alloc(self, allocator, full_indices: np.ndarray, rank: int) -> None:
        swa_indices = _swa_indices(allocator, full_indices, rank)
        assert np.all(swa_indices > 0)
        self.full[rank].update(int(index) for index in full_indices)
        self.swa[rank].update(int(index) for index in swa_indices)

    def transfer_to_tree(self, allocator, full_indices: np.ndarray, rank: int) -> None:
        self.full[rank].difference_update(int(index) for index in full_indices)
        self.swa[rank].difference_update(
            int(index) for index in _swa_indices(allocator, full_indices, rank)
        )


def _insert(
    cache,
    allocator,
    tokens: list[int],
    *,
    dp_rank: int | None = None,
    ownership: _RequestOwnership | None = None,
):
    rank = dp_rank or 0
    value = allocator.alloc(len(tokens), dp_rank=rank)
    assert value is not None
    if ownership is not None:
        ownership.record_alloc(allocator, value, rank)
    result = cache.insert(InsertParams(key=RadixKey(tokens, dp_rank=dp_rank), value=value))
    if ownership is not None:
        ownership.transfer_to_tree(allocator, value, rank)
    return result


def _swa_indices(allocator, full_indices, rank: int) -> np.ndarray:
    return allocator.translate_full_to_swa(full_indices, dp_rank=rank, require_mapped=False)


def _tree_full_indices(cache, rank: int) -> list[np.ndarray]:
    values = []
    pending = [cache.root_node]
    while pending:
        node = pending.pop()
        if node is not cache.root_node and node.key.dp_rank == rank:
            full = node.component_data[ComponentType.FULL].value
            if full is not None:
                values.append(full)
        pending.extend(node.children.values())
    return values


def _allocator_free_indices(pool, rank: int) -> set[int]:
    if hasattr(pool, "free_slots"):
        return set(int(index) for index in pool.free_slots[rank])
    pages = np.concatenate((pool.free_pages[rank], pool.release_pages[rank]))
    indices = pages[:, None] * pool.page_size + np.arange(pool.page_size)
    return set(int(index) for index in indices.reshape(-1))


def _assert_rank_ledger(
    test: unittest.TestCase,
    cache,
    allocator,
    ownership: _RequestOwnership,
    rank: int,
) -> None:
    """Assert the allocator partition from independently recorded owners."""
    full_tree: set[int] = set()
    swa_tree: set[int] = set()
    pending = [cache.root_node]
    while pending:
        node = pending.pop()
        pending.extend(node.children.values())
        if node is cache.root_node or node.key.dp_rank != rank:
            continue
        full_value = node.component_data[ComponentType.FULL].value
        if full_value is None:
            continue
        full_indices = set(int(index) for index in full_value)
        test.assertTrue(full_tree.isdisjoint(full_indices))
        full_tree.update(full_indices)
        swa_value = node.component_data[ComponentType.SWA].value
        if swa_value is None:
            continue
        expected_swa = _swa_indices(allocator, full_value, rank)
        np.testing.assert_array_equal(swa_value, expected_swa)
        test.assertTrue(np.all(expected_swa > 0))
        swa_indices = set(int(index) for index in swa_value)
        test.assertTrue(swa_tree.isdisjoint(swa_indices))
        swa_tree.update(swa_indices)

    full_free = _allocator_free_indices(allocator.full_attn_allocator, rank)
    swa_free = _allocator_free_indices(allocator.swa_attn_allocator, rank)
    full_request = ownership.full[rank]
    swa_request = ownership.swa[rank]
    for owners in ((full_free, full_tree, full_request), (swa_free, swa_tree, swa_request)):
        test.assertTrue(owners[0].isdisjoint(owners[1]))
        test.assertTrue(owners[0].isdisjoint(owners[2]))
        test.assertTrue(owners[1].isdisjoint(owners[2]))

    test.assertEqual(len(full_free), allocator.full_available_size(rank))
    test.assertEqual(len(swa_free), allocator.swa_available_size(rank))
    test.assertEqual(
        allocator.full_attn_allocator.size_per_rank,
        len(full_free | full_tree | full_request),
    )
    test.assertEqual(
        allocator.swa_attn_allocator.size_per_rank,
        len(swa_free | swa_tree | swa_request),
    )
    test.assertEqual(
        len(swa_tree),
        cache.component_evictable_size_[ComponentType.SWA][rank]
        + cache.component_protected_size_[ComponentType.SWA][rank],
    )


class TestUnifiedSWAComponentPage1(unittest.TestCase):
    def test_radix_ledgers_share_schema_units_and_real_mapping_diagnostics(self):
        """Both radix routes expose the same slot-count ledger contract."""
        for kind in ("unified", "legacy"):
            with self.subTest(kind=kind):
                cache, allocator = _make_cache(window=4, kind=kind)
                full = allocator.alloc(4, dp_rank=0)
                self.assertIsNotNone(full)
                cache.req_to_token_pool.write((0, slice(0, 4)), full)
                req = SimpleNamespace(
                    dp_rank=0,
                    req_pool_idx=0,
                    cache_protected_len=0,
                    kv_allocated_len=4,
                    swa_evicted_seqlen=0,
                )

                snapshot = cache.cache_ledger_snapshot(0, [req])

                self.assertEqual(set(snapshot), _LEDGER_FIELDS)
                self.assertEqual(
                    snapshot["full_capacity"], allocator.full_attn_allocator.size_per_rank
                )
                self.assertEqual(
                    snapshot["swa_capacity"], allocator.swa_attn_allocator.size_per_rank
                )
                self.assertEqual(snapshot["full_request_owned"], 4)
                self.assertEqual(snapshot["swa_request_owned"], 4)
                self.assertEqual(snapshot["mapping_nonzero_count"], 4)
                self.assertEqual(snapshot["mapping_invalid_count"], 0)
                self.assertEqual(snapshot["mapping_duplicate_count"], 0)
                self.assertEqual(
                    snapshot["full_available"] + snapshot["full_request_owned"],
                    snapshot["full_capacity"],
                )
                self.assertEqual(
                    snapshot["swa_available"] + snapshot["swa_request_owned"],
                    snapshot["swa_capacity"],
                )

    def test_chunk_ledger_has_zero_tree_owners_and_real_request_conservation(self):
        cache, allocator = _make_cache(window=4, kind="chunk")
        full = allocator.alloc(4, dp_rank=0)
        self.assertIsNotNone(full)
        cache.req_to_token_pool.write((0, slice(0, 4)), full)
        req = SimpleNamespace(
            dp_rank=0,
            req_pool_idx=0,
            cache_protected_len=0,
            kv_allocated_len=4,
            swa_evicted_seqlen=0,
        )

        snapshot = cache.cache_ledger_snapshot(0, [req])

        self.assertEqual(set(snapshot), _LEDGER_FIELDS)
        for field in (
            "full_tree_evictable",
            "full_tree_protected",
            "swa_tree_evictable",
            "swa_tree_protected",
        ):
            self.assertEqual(snapshot[field], 0)
        self.assertEqual(snapshot["full_request_owned"], 4)
        self.assertEqual(snapshot["swa_request_owned"], 4)
        self.assertEqual(
            snapshot["full_available"] + snapshot["full_request_owned"],
            snapshot["full_capacity"],
        )
        self.assertEqual(
            snapshot["swa_available"] + snapshot["swa_request_owned"],
            snapshot["swa_capacity"],
        )

    def test_paged_ledger_capacity_excludes_unusable_rank_remainder(self):
        paged_pool = SimpleNamespace(size_per_rank=320, pages_per_rank=2, page_size=128)
        allocator = SimpleNamespace(
            full_attn_allocator=paged_pool,
            swa_attn_allocator=paged_pool,
            page_size=128,
            dp_size=1,
            full_to_swa_index_mapping=np.zeros(448, dtype=np.int32),
            full_available_size=lambda _rank: 256,
            swa_available_size=lambda _rank: 256,
        )
        allocator.full_to_swa_index_mapping[0] = 128

        snapshot = build_swa_cache_ledger_snapshot(
            dp_rank=0,
            allocator=allocator,
            full_tree_evictable=set(),
            full_tree_protected=set(),
            swa_tree_evictable=set(),
            swa_tree_protected=set(),
            full_request_occurrences=[],
            swa_request_occurrences=[],
            event_totals={},
        )

        self.assertEqual(allocator.full_attn_allocator.size_per_rank, 320)
        self.assertEqual(snapshot["full_capacity"], 256)
        self.assertEqual(snapshot["swa_capacity"], 256)
        self.assertEqual(snapshot["full_available"], 256)
        self.assertEqual(snapshot["swa_available"], 256)
        self.assertEqual(snapshot["mapping_nonzero_count"], 1)
        self.assertEqual(snapshot["mapping_invalid_count"], 1)

    def test_chunk_ledger_exposes_duplicate_request_owners_and_mapping_duplicates(self):
        cache, allocator = _make_cache(window=4, kind="chunk")
        full = allocator.alloc(2, dp_rank=0)
        self.assertIsNotNone(full)
        mapping = allocator.full_to_swa_index_mapping
        mapping[full[1]] = mapping[full[0]]
        duplicate_row = np.repeat(full[:1], 2)
        cache.req_to_token_pool.write((0, slice(0, 2)), duplicate_row)
        cache.req_to_token_pool.write((1, slice(0, 2)), duplicate_row)
        reqs = [
            SimpleNamespace(
                dp_rank=0,
                req_pool_idx=index,
                cache_protected_len=0,
                kv_allocated_len=2,
                swa_evicted_seqlen=0,
            )
            for index in (0, 1)
        ]

        snapshot = cache.cache_ledger_snapshot(0, reqs)

        self.assertEqual(snapshot["full_duplicate_request_owner_count"], 3)
        self.assertEqual(snapshot["swa_duplicate_request_owner_count"], 3)
        self.assertEqual(snapshot["mapping_nonzero_count"], 2)
        self.assertEqual(snapshot["mapping_duplicate_count"], 1)

    def test_request_tail_reclaim_preserves_tree_mapping_for_both_radix_routes(self):
        for kind in ("unified", "legacy"):
            with self.subTest(kind=kind):
                cache, allocator = _make_cache(window=4, kind=kind)
                _insert(cache, allocator, [0, 1, 2, 3])
                match = cache.match_prefix(MatchPrefixParams(key=RadixKey([0, 1, 2, 3])))
                tree_full = np.asarray(match.device_indices).copy()
                tree_swa = _swa_indices(allocator, tree_full, 0).copy()
                tail = allocator.alloc(12, dp_rank=0)
                self.assertIsNotNone(tail)
                row = np.concatenate([tree_full, tail])
                cache.req_to_token_pool.write((0, slice(0, len(row))), row)
                req = SimpleNamespace(
                    req_pool_idx=0,
                    last_node=match.last_device_node,
                    swa_evicted_seqlen=0,
                )

                cache.evict_req_swa(req, pre_len=16, dp_rank=0)

                self.assertEqual(req.swa_evicted_seqlen, 11)
                np.testing.assert_array_equal(_swa_indices(allocator, tree_full, 0), tree_swa)
                self.assertTrue(np.all(_swa_indices(allocator, tail[:7], 0) == 0))
                self.assertTrue(np.all(_swa_indices(allocator, tail[7:], 0) > 0))

    def test_chunk_request_tail_reclaim_owns_the_whole_live_row(self):
        cache, allocator = _make_cache(window=4, kind="chunk")
        full = allocator.alloc(16, dp_rank=0)
        self.assertIsNotNone(full)
        cache.req_to_token_pool.write((0, slice(0, 16)), full)
        req = SimpleNamespace(req_pool_idx=0, swa_evicted_seqlen=0)

        cache.evict_req_swa(req, pre_len=16, dp_rank=0)

        self.assertEqual(req.swa_evicted_seqlen, 11)
        self.assertTrue(np.all(_swa_indices(allocator, full[:11], 0) == 0))
        self.assertTrue(np.all(_swa_indices(allocator, full[11:], 0) > 0))

    def test_chunk_unfinished_and_finished_refresh_then_clear_lock_receipt(self):
        cache, allocator = _make_cache(window=4, kind="chunk")
        full = allocator.alloc(4, dp_rank=0)
        self.assertIsNotNone(full)
        cache.req_to_token_pool.write((0, slice(0, 4)), full)

        class _Req:
            def pop_committed_kv_cache(self):
                return 4

        req = _Req()
        req.req_pool_idx = 0
        req.dp_rank = 0
        req.fill_ids = [0, 1, 2, 3]
        req.last_node = None
        req.cache_lock_params = DecLockRefParams(
            swa_uuid_for_lock=17, skip_lock_node_ids={"swa": [9]}
        )
        req.swa_uuid_for_lock = 17

        cache.cache_unfinished_req(req)

        self.assertIsNotNone(req.cache_lock_params)
        self.assertIsNone(req.cache_lock_params.swa_uuid_for_lock)
        self.assertEqual(req.cache_lock_params.skip_lock_node_ids, {})
        self.assertIsNone(req.swa_uuid_for_lock)

        cache.cache_finished_req(req)
        self.assertIsNone(req.cache_lock_params)
        self.assertIsNone(req.swa_uuid_for_lock)

    def test_disabled_radix_finish_still_clears_lock_receipt(self):
        for kind in ("unified", "legacy"):
            with self.subTest(kind=kind):
                cache, allocator = _make_cache(window=4, kind=kind)
                cache.disable = True
                full = allocator.alloc(4, dp_rank=0)
                self.assertIsNotNone(full)
                cache.req_to_token_pool.write((0, slice(0, 4)), full)

                class _Req:
                    def pop_committed_kv_cache(self):
                        return 4

                req = _Req()
                req.req_pool_idx = 0
                req.dp_rank = 0
                req.fill_ids = [0, 1, 2, 3]
                req.last_node = None
                req.cache_lock_params = DecLockRefParams(swa_uuid_for_lock=17)
                req.swa_uuid_for_lock = 17

                cache.cache_unfinished_req(req)
                self.assertIsNotNone(req.cache_lock_params)
                self.assertIsNone(req.cache_lock_params.swa_uuid_for_lock)
                self.assertIsNone(req.swa_uuid_for_lock)

                req.cache_lock_params = DecLockRefParams(swa_uuid_for_lock=17)
                req.swa_uuid_for_lock = 17
                cache.cache_finished_req(req)

                self.assertIsNone(req.cache_lock_params)
                self.assertIsNone(req.swa_uuid_for_lock)

    def test_swa_capability_and_per_rank_size_accessors(self):
        """Hybrid unified cache exposes the validated window and SWA ledgers."""
        cache, _ = _make_cache(window=4)
        self.assertTrue(cache.supports_swa())
        self.assertEqual(cache.sliding_window_size, 4)
        self.assertEqual(cache.full_protected_size(0), 0)
        self.assertEqual(cache.swa_protected_size(0), 0)

    def test_registry_exposes_device_swa_component(self):
        """Unified SWA must be constructible without changing factory routing."""
        self.assertEqual(jax.device_count(), 4)
        self.assertIn(ComponentType.SWA, COMPONENT_REGISTRY)

    def test_exact_shared_and_branched_matches_keep_swa_window(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(8)))
        self.assertEqual(
            len(cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(8))))).device_indices),
            8,
        )
        _insert(cache, allocator, [0, 1, 2, 3, 20, 21, 22, 23])
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey([0, 1, 2, 3])))
        self.assertEqual(len(match.device_indices), 4)
        self.assertEqual(cache.component_evictable_size_[ComponentType.SWA][0], 12)

    def test_internal_swa_eviction_keeps_full_tree_owned(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(8)))
        _insert(cache, allocator, [0, 1, 2, 3, 20, 21, 22, 23])
        parent = cache.match_prefix(MatchPrefixParams(key=RadixKey([0, 1, 2, 3]))).last_device_node
        full_before = parent.component_data[ComponentType.FULL].value.copy()
        full_free_before = allocator.full_available_size()
        swa_free_before = allocator.swa_available_size()
        freed, _ = cache.components[ComponentType.SWA].evict_component(parent)
        self.assertEqual(freed, 4)
        self.assertIsNotNone(parent.component_data[ComponentType.FULL].value)
        np.testing.assert_array_equal(parent.component_data[ComponentType.FULL].value, full_before)
        self.assertTrue(np.all(_swa_indices(allocator, full_before, 0) == 0))
        self.assertEqual(allocator.full_available_size(), full_free_before)
        self.assertEqual(allocator.swa_available_size(), swa_free_before + 4)

    def test_tombstone_gap_requires_a_new_contiguous_window(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(8)))
        _insert(cache, allocator, [0, 1, 2, 3, 20, 21, 22, 23])
        parent = cache.match_prefix(MatchPrefixParams(key=RadixKey([0, 1, 2, 3]))).last_device_node
        child = next(iter(parent.children.values()))
        component = cache.components[ComponentType.SWA]
        component.evict_component(parent)
        validate = component.create_match_validator()
        self.assertFalse(validate(parent))
        self.assertTrue(validate(child))

    def test_window_lock_protects_only_last_window_and_releases_symmetrically(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(8)))
        node = cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(8))))).last_device_node
        lock = cache.inc_lock_ref(node)
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 4)
        cache.dec_lock_ref(node, lock.to_dec_params())
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 0)

    def test_concurrent_window_locks_reuse_boundary_uuid_and_release_independently(self):
        """Two requests sharing a window must retain the same SWA boundary."""
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(8)))
        endpoint = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(8))))
        ).last_device_node
        ancestor = endpoint.parent
        initial_evictable = cache.component_evictable_size_[ComponentType.SWA][0]

        first = cache.inc_lock_ref(endpoint)
        second = cache.inc_lock_ref(endpoint)
        self.assertIsNotNone(first.swa_uuid_for_lock)
        self.assertEqual(first.swa_uuid_for_lock, second.swa_uuid_for_lock)
        self.assertEqual(
            endpoint.component_data[ComponentType.SWA].metadata["component_uuid"],
            first.swa_uuid_for_lock,
        )
        self.assertEqual(endpoint.component_data[ComponentType.SWA].lock_ref, 2)
        self.assertEqual(ancestor.component_data[ComponentType.SWA].lock_ref, 0)
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 4)
        self.assertEqual(
            cache.component_evictable_size_[ComponentType.SWA][0], initial_evictable - 4
        )
        self.assertNotIn(endpoint, cache.aux_evictable_device_nodes[ComponentType.SWA])
        self.assertIn(ancestor, cache.aux_evictable_device_nodes[ComponentType.SWA])

        cache.dec_lock_ref(endpoint, first.to_dec_params())
        self.assertEqual(endpoint.component_data[ComponentType.SWA].lock_ref, 1)
        self.assertEqual(ancestor.component_data[ComponentType.SWA].lock_ref, 0)
        self.assertEqual(
            endpoint.component_data[ComponentType.SWA].metadata["component_uuid"],
            first.swa_uuid_for_lock,
        )
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 4)
        self.assertEqual(
            cache.component_evictable_size_[ComponentType.SWA][0], initial_evictable - 4
        )
        self.assertNotIn(endpoint, cache.aux_evictable_device_nodes[ComponentType.SWA])
        self.assertIn(ancestor, cache.aux_evictable_device_nodes[ComponentType.SWA])

        cache.dec_lock_ref(endpoint, second.to_dec_params())
        self.assertEqual(endpoint.component_data[ComponentType.SWA].lock_ref, 0)
        self.assertEqual(ancestor.component_data[ComponentType.SWA].lock_ref, 0)
        self.assertEqual(
            endpoint.component_data[ComponentType.SWA].metadata["component_uuid"],
            first.swa_uuid_for_lock,
        )
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 0)
        self.assertEqual(cache.component_evictable_size_[ComponentType.SWA][0], initial_evictable)
        self.assertIn(endpoint, cache.aux_evictable_device_nodes[ComponentType.SWA])
        self.assertIn(ancestor, cache.aux_evictable_device_nodes[ComponentType.SWA])

    def test_tombstone_lock_skip_is_symmetric(self):
        cache, allocator = _make_cache(window=8)
        _insert(cache, allocator, list(range(8)))
        _insert(cache, allocator, [0, 1, 2, 3, 20, 21, 22, 23])
        parent = cache.match_prefix(MatchPrefixParams(key=RadixKey([0, 1, 2, 3]))).last_device_node
        child = next(iter(parent.children.values()))
        cache.components[ComponentType.SWA].evict_component(parent)
        lock = cache.inc_lock_ref(child)
        self.assertIn(parent.id, lock.skip_lock_node_ids[ComponentType.SWA])
        self.assertEqual(child.component_data[ComponentType.SWA].lock_ref, 1)
        cache.dec_lock_ref(child, lock.to_dec_params())
        self.assertEqual(child.component_data[ComponentType.SWA].lock_ref, 0)
        self.assertEqual(parent.component_data[ComponentType.SWA].lock_ref, 0)

    def test_tombstone_ancestor_split_unfinished_relock_and_finish_are_symmetric(self):
        cache, allocator = _make_cache(window=8)
        key = list(range(12))
        _insert(cache, allocator, key)
        _insert(cache, allocator, list(range(8)) + [20, 21, 22, 23])
        _insert(cache, allocator, list(range(4)) + [30, 31, 32, 33])
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(key), full_only=True))
        endpoint = match.last_device_node
        middle = endpoint.parent
        ancestor = middle.parent
        self.assertIsNot(ancestor, cache.root_node)
        cache.components[ComponentType.SWA].evict_component(middle)
        old_lock = cache.inc_lock_ref(endpoint)
        self.assertIn(middle.id, old_lock.skip_lock_node_ids[ComponentType.SWA])
        self.assertIsNotNone(old_lock.swa_uuid_for_lock)
        self.assertEqual(
            ancestor.component_data[ComponentType.SWA].metadata["component_uuid"],
            old_lock.swa_uuid_for_lock,
        )

        tail = allocator.alloc(4, dp_rank=0)
        self.assertIsNotNone(tail)
        row = np.concatenate([np.asarray(match.device_indices), tail])
        cache.req_to_token_pool.write((0, slice(0, 16)), row)

        class _Req:
            def pop_committed_kv_cache(self):
                self.kv_committed_freed = True
                return self.kv_committed_len

        req = _Req()
        req.req_pool_idx = 0
        req.origin_input_ids = key + [40, 41, 42, 43]
        req.radix_input_ids = list(req.origin_input_ids)
        req.output_ids = []
        req.fill_ids = list(req.origin_input_ids)
        req.prefix_indices = np.asarray(match.device_indices)
        req.last_node = endpoint
        req.extra_key = None
        req.dp_rank = None
        req.cache_protected_len = 12
        req.last_matched_prefix_len = 12
        req.swa_evicted_seqlen = 0
        old_params = old_lock.to_dec_params()
        req.cache_lock_params = old_params
        req.swa_uuid_for_lock = old_lock.swa_uuid_for_lock
        req.kv_committed_len = 16
        req.kv_allocated_len = 16
        req.kv_committed_freed = False

        cache.cache_unfinished_req(req)

        self.assertIsNot(req.cache_lock_params, old_params)
        self.assertEqual(req.cache_lock_params.skip_lock_node_ids[ComponentType.SWA], [])
        self.assertNotEqual(req.cache_lock_params.swa_uuid_for_lock, old_lock.swa_uuid_for_lock)
        self.assertEqual(middle.component_data[ComponentType.SWA].lock_ref, 0)
        self.assertEqual(ancestor.component_data[ComponentType.SWA].lock_ref, 0)
        cache.cache_finished_req(req)
        self.assertIsNone(req.cache_lock_params)
        self.assertIsNone(req.swa_uuid_for_lock)
        for node in (endpoint, middle, ancestor, req.last_node):
            self.assertEqual(node.component_data[ComponentType.FULL].lock_ref, 0)
            self.assertEqual(node.component_data[ComponentType.SWA].lock_ref, 0)

    def test_long_leaf_window_lock_splits_at_physical_tail(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(12)))
        parent = next(iter(cache.root_node.children.values()))
        child = next(iter(parent.children.values()))
        grandchild = next(iter(child.children.values()))
        self.assertEqual(len(parent.key), 4)
        self.assertEqual(len(child.key), 4)
        self.assertEqual(len(grandchild.key), 4)

    def test_long_prefix_lock_and_lru_touch_only_physical_window_suffix(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(12)))
        endpoint = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(8))))
        ).last_device_node
        component = cache.components[ComponentType.SWA]
        component.refresh_lru(endpoint, LRURefreshPhase.MATCH_END)
        self.assertLessEqual(len(endpoint.component_data[ComponentType.SWA].value), 4)
        first = next(iter(cache.root_node.children.values()))
        self.assertLess(
            first.component_data[ComponentType.SWA].metadata.get("last_access_time", 0),
            endpoint.component_data[ComponentType.SWA].metadata["last_access_time"],
        )
        lock = cache.inc_lock_ref(endpoint)
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 4)
        cache.dec_lock_ref(endpoint, lock.to_dec_params())

    def test_split_moves_window_lock_uuid_to_new_parent(self):
        cache, allocator = _make_cache(window=8)
        _insert(cache, allocator, list(range(8)))
        child = next(iter(cache.root_node.children.values()))
        lock = cache.inc_lock_ref(child)
        uuid = lock.swa_uuid_for_lock
        parent = cache._split_node(child.key, child, 4)
        self.assertEqual(parent.component_data[ComponentType.SWA].metadata["component_uuid"], uuid)
        self.assertNotIn("component_uuid", child.component_data[ComponentType.SWA].metadata)
        self.assertIsNot(
            parent.component_data[ComponentType.SWA].metadata,
            child.component_data[ComponentType.SWA].metadata,
        )
        cache.dec_lock_ref(child, lock.to_dec_params())
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 0)
        self.assertEqual(parent.component_data[ComponentType.SWA].metadata["component_uuid"], uuid)

    def test_overlap_healing_before_node_adopts_request_owned_full(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(4)))
        node = next(iter(cache.root_node.children.values()))
        old_full = node.component_data[ComponentType.FULL].value.copy()
        cache.components[ComponentType.SWA].evict_component(node)
        self.assertTrue(np.all(_swa_indices(allocator, old_full, 0) == 0))
        replacement = allocator.alloc(4)
        self.assertIsNotNone(replacement)
        full_free_before = allocator.full_available_size()
        swa_free_before = allocator.swa_available_size()
        boundary = cache.components[ComponentType.SWA].update_component_on_insert_overlap(
            node, 4, 0, replacement, InsertParams(swa_evicted_seqlen=0)
        )
        self.assertEqual(boundary, 0)
        np.testing.assert_array_equal(node.component_data[ComponentType.FULL].value, replacement)
        np.testing.assert_array_equal(
            node.component_data[ComponentType.SWA].value, _swa_indices(allocator, replacement, 0)
        )
        self.assertEqual(allocator.full_available_size(), full_free_before + 4)
        self.assertEqual(allocator.swa_available_size(), swa_free_before)

    def test_overlap_healing_records_per_rank_monotonic_event(self):
        """A revived SWA tombstone is a run-level event, not a capacity term."""
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(4)))
        node = next(iter(cache.root_node.children.values()))
        cache.components[ComponentType.SWA]._clear_swa_value(node)
        replacement = allocator.alloc(4, dp_rank=0)
        before = cache.cache_ledger_snapshot(0, [])
        cache.components[ComponentType.SWA].update_component_on_insert_overlap(
            node, 4, 0, replacement, InsertParams(swa_evicted_seqlen=0)
        )
        healed = cache.cache_ledger_snapshot(0, [])
        self.assertEqual(healed["tombstone_healed_total"], before["tombstone_healed_total"] + 1)
        cache.reset()
        self.assertEqual(
            cache.cache_ledger_snapshot(0, [])["tombstone_healed_total"],
            healed["tombstone_healed_total"],
        )

    def test_overlap_healing_inside_node_splits_at_boundary(self):
        cache, allocator = _make_cache(window=8)
        _insert(cache, allocator, list(range(8)))
        node = next(iter(cache.root_node.children.values()))
        old_full = node.component_data[ComponentType.FULL].value.copy()
        cache.components[ComponentType.SWA].evict_component(node)
        replacement = allocator.alloc(8)
        self.assertIsNotNone(replacement)
        full_free_before = allocator.full_available_size()
        swa_free_before = allocator.swa_available_size()
        boundary = cache.components[ComponentType.SWA].update_component_on_insert_overlap(
            node, 8, 0, replacement, InsertParams(swa_evicted_seqlen=4)
        )
        self.assertEqual(boundary, 4)
        parent = node.parent
        self.assertEqual(len(parent.key), 4)
        self.assertIsNone(parent.component_data[ComponentType.SWA].value)
        self.assertIsNotNone(node.component_data[ComponentType.SWA].value)
        self.assertTrue(
            np.all(_swa_indices(allocator, parent.component_data[ComponentType.FULL].value, 0) == 0)
        )
        np.testing.assert_array_equal(
            node.component_data[ComponentType.FULL].value, replacement[4:]
        )
        self.assertTrue(np.all(_swa_indices(allocator, old_full, 0) == 0))
        self.assertEqual(allocator.full_available_size(), full_free_before + 4)
        self.assertEqual(allocator.swa_available_size(), swa_free_before)

    def test_overlap_healing_after_node_keeps_tombstone(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(4)))
        node = next(iter(cache.root_node.children.values()))
        cache.components[ComponentType.SWA].evict_component(node)
        replacement = allocator.alloc(4)
        self.assertIsNotNone(replacement)
        full_free_before = allocator.full_available_size()
        swa_free_before = allocator.swa_available_size()
        boundary = cache.components[ComponentType.SWA].update_component_on_insert_overlap(
            node, 4, 0, replacement, InsertParams(swa_evicted_seqlen=4)
        )
        self.assertEqual(boundary, 4)
        self.assertIsNone(node.component_data[ComponentType.SWA].value)
        self.assertEqual(allocator.full_available_size(), full_free_before)
        self.assertEqual(allocator.swa_available_size(), swa_free_before)

    def test_fully_request_evicted_leaf_is_not_materialized(self):
        cache, allocator = _make_cache(window=4)
        request = allocator.alloc(4)
        self.assertIsNotNone(request)
        full_free_before = allocator.full_available_size()
        result = cache.insert(
            InsertParams(
                key=RadixKey(list(range(4))),
                value=request,
                swa_evicted_seqlen=4,
            )
        )
        self.assertEqual(result.prefix_len, 0)
        self.assertFalse(cache.root_node.children)
        self.assertEqual(allocator.full_available_size(), full_free_before + 4)
        self.assertEqual(allocator.swa_available_size(), allocator.swa_attn_allocator.size)

    def test_leaf_cascade_frees_full_then_swa_once(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(4)))
        node = next(iter(cache.root_node.children.values()))
        calls = []
        full_component = cache.components[ComponentType.FULL]
        original_full = full_component._free_full
        original_swa = allocator.free_swa

        def record_full(indices, *, dp_rank):
            calls.append(("full", len(indices), dp_rank))
            return original_full(indices, dp_rank=dp_rank)

        def record_swa(indices, dp_rank=0):
            calls.append(("swa", len(indices), dp_rank))
            return original_swa(indices, dp_rank=dp_rank)

        full_component._free_full = record_full
        allocator.free_swa = record_swa
        full_free_before = allocator.full_available_size()
        swa_free_before = allocator.swa_available_size()
        result = cache.evict(EvictParams(swa_num_tokens=4))
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.swa_num_tokens_evicted, 4)
        self.assertEqual(calls, [("full", 4, 0), ("swa", 4, 0)])
        self.assertEqual(allocator.full_available_size(), full_free_before + 4)
        self.assertEqual(allocator.swa_available_size(), swa_free_before + 4)
        self.assertIsNone(node.component_data[ComponentType.FULL].value)

    def test_leaf_deletion_does_not_count_as_a_retained_swa_tombstone(self):
        for kind in ("unified", "legacy"):
            with self.subTest(kind=kind):
                cache, allocator = _make_cache(window=4, kind=kind)
                _insert(cache, allocator, list(range(4)))
                before = cache.cache_ledger_snapshot(0, [])["tombstone_created_total"]

                cache.evict(EvictParams(swa_num_tokens=4))

                self.assertFalse(cache.root_node.children)
                self.assertEqual(
                    cache.cache_ledger_snapshot(0, [])["tombstone_created_total"],
                    before,
                )

    def test_write_backup_releases_the_exact_swa_window_lock_receipt(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(8)))
        node = next(iter(cache.root_node.children.values()))
        future = object()
        observed_acquire_params = []
        observed_release_params = []
        original_acquire = cache.inc_lock_ref
        original_release = cache.dec_lock_ref
        cache._reserve_host_slots = lambda count: list(range(100, 100 + count))
        cache.hicache_controller = SimpleNamespace(write=lambda device_pages, host_pages: future)

        def record_acquire(locked_node):
            result = original_acquire(locked_node)
            observed_acquire_params.append(result.to_dec_params())
            return result

        def record_release(locked_node, params=None):
            observed_release_params.append(params)
            return original_release(locked_node, params)

        cache.inc_lock_ref = record_acquire
        cache.dec_lock_ref = record_release

        self.assertEqual(cache.write_backup(node), 4)
        self.assertEqual(observed_release_params, observed_acquire_params)
        self.assertIsNotNone(observed_release_params[0].swa_uuid_for_lock)
        self.assertIn(future, cache.ongoing_write)
        for component_data in node.component_data:
            self.assertEqual(component_data.lock_ref, 0)

    def test_reset_clears_component_ledgers(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(4)))
        self.assertEqual(cache.component_evictable_size_[ComponentType.SWA][0], 4)
        cache.reset()
        self.assertEqual(cache.component_evictable_size_[ComponentType.SWA][0], 0)
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 0)
        self.assertEqual(cache.swa_evictable_size(), 0)
        self.assertEqual(cache.swa_protected_size(), 0)

    def test_internal_swa_tombstone_creation_is_counted_and_survives_reset(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, [0, 1, 2, 3, 4, 5, 6, 7])
        _insert(cache, allocator, [0, 1, 2, 3, 20, 21, 22, 23])
        parent = cache.match_prefix(MatchPrefixParams(key=RadixKey([0, 1, 2, 3]))).last_device_node
        before = cache.cache_ledger_snapshot(0, [])

        cache.components[ComponentType.SWA].evict_component(parent)

        created = cache.cache_ledger_snapshot(0, [])
        self.assertEqual(created["tombstone_created_total"], before["tombstone_created_total"] + 1)
        cache.reset()
        self.assertEqual(
            cache.cache_ledger_snapshot(0, [])["tombstone_created_total"],
            created["tombstone_created_total"],
        )

    def test_unified_ledger_does_not_invent_tree_ownership_from_mapping(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, [0, 1, 2, 3])
        node = next(iter(cache.root_node.children.values()))
        node.component_data[ComponentType.SWA].value = None

        snapshot = cache.cache_ledger_snapshot(0, [])

        self.assertEqual(snapshot["swa_tree_evictable"], 0)
        self.assertEqual(snapshot["mapping_nonzero_count"], 4)
        with self.assertRaisesRegex(ValueError, "swa.*balance|mapping ownership"):
            validate_swa_cache_ledger(snapshot, require_idle=True)

    def test_legacy_tombstone_healing_total_is_real_and_survives_reset(self):
        cache, allocator = _make_cache(window=4, kind="legacy")
        key = list(range(8))
        _insert(cache, allocator, key)
        _insert(cache, allocator, list(range(4)) + [20, 21, 22, 23])
        parent = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(4))))
        ).last_device_node
        allocator.free_swa(parent.value, dp_rank=0)
        cache.swa_lru_list.remove_node(parent)
        cache._tombstone_internal_node(parent)
        self.assertTrue(parent.swa_tombstone)
        before = cache.cache_ledger_snapshot(0, [])
        replacement = allocator.alloc(8, dp_rank=0)
        self.assertIsNotNone(replacement)

        cache.insert(
            InsertParams(
                key=RadixKey(key),
                value=replacement,
                prev_prefix_len=0,
                swa_evicted_seqlen=0,
            )
        )

        healed = cache.cache_ledger_snapshot(0, [])
        self.assertEqual(healed["tombstone_healed_total"], before["tombstone_healed_total"] + 1)
        cache.reset()
        self.assertEqual(
            cache.cache_ledger_snapshot(0, [])["tombstone_healed_total"],
            healed["tombstone_healed_total"],
        )

    def test_cold_ledger_snapshot_does_not_create_event_rank_entries(self):
        for kind in ("chunk", "legacy", "unified"):
            with self.subTest(kind=kind):
                cache, _ = _make_cache(window=4, kind=kind)
                before = dict(cache._ledger_event_totals)

                cache.cache_ledger_snapshot(0, [])

                self.assertEqual(dict(cache._ledger_event_totals), before)


class TestUnifiedSWAComponentPagedAndDP2(unittest.TestCase):
    def test_scheduler_lifecycle_heals_internal_tombstones_for_both_swa_routes(self):
        """Check healing; admission/pressure tests own capacity-triggered eviction."""
        shared = list(range(4))
        seed_inputs = [
            shared + list(range(100, 108)),
            shared + list(range(200, 208)),
        ]
        evict_params = EvictParams(swa_num_tokens=len(shared) + 8, dp_rank=0)
        previous_chunk = global_server_args_dict.get("chunked_prefill_size")
        global_server_args_dict["chunked_prefill_size"] = 64
        try:
            for kind in ("legacy", "unified"):
                with self.subTest(kind=kind):
                    cache, allocator = _make_cache(
                        page_size=4,
                        window=8,
                        kind=kind,
                        size=128,
                        size_swa=128,
                    )
                    for ordinal, input_ids in enumerate(seed_inputs):
                        _run_scheduler_lifecycle(
                            cache,
                            allocator,
                            rid=f"{kind}-seed-{ordinal}",
                            input_ids=input_ids,
                        )

                    self.assertEqual(len(cache.root_node.children), 1)
                    target = next(iter(cache.root_node.children.values()))
                    self.assertEqual(list(target.key.token_ids), shared)
                    self.assertTrue(target.children)
                    if kind == "legacy":
                        self.assertIsNotNone(target.value)
                        self.assertFalse(target.swa_tombstone)
                    else:
                        self.assertIsNotNone(target.component_data[ComponentType.FULL].value)
                        self.assertIsNotNone(target.component_data[ComponentType.SWA].value)

                    before_evict = cache.cache_ledger_snapshot(0, [])
                    evicted = cache.evict(evict_params)
                    after_evict = cache.cache_ledger_snapshot(0, [])
                    self.assertGreater(evicted.swa_num_tokens_evicted, 0)
                    self.assertTrue(target.children)
                    if kind == "legacy":
                        self.assertIsNotNone(target.value)
                        self.assertTrue(target.swa_tombstone)
                    else:
                        self.assertIsNotNone(target.component_data[ComponentType.FULL].value)
                        self.assertIsNone(target.component_data[ComponentType.SWA].value)
                    self.assertGreater(
                        after_evict["tombstone_created_total"],
                        before_evict["tombstone_created_total"],
                    )

                    _run_scheduler_lifecycle(
                        cache,
                        allocator,
                        rid=f"{kind}-healing",
                        input_ids=seed_inputs[0],
                    )

                    healed = cache.cache_ledger_snapshot(0, [])
                    self.assertGreater(
                        healed["tombstone_healed_total"],
                        after_evict["tombstone_healed_total"],
                    )
                    if kind == "legacy":
                        self.assertFalse(target.swa_tombstone)
                    else:
                        self.assertIsNotNone(target.component_data[ComponentType.SWA].value)
                    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(shared, dp_rank=0)))
                    self.assertIs(match.last_device_node, target)
                    self.assertEqual(len(match.device_indices), len(shared))
                    validate_swa_cache_ledger(healed, require_idle=True)
        finally:
            if previous_chunk is None:
                global_server_args_dict.pop("chunked_prefill_size", None)
            else:
                global_server_args_dict["chunked_prefill_size"] = previous_chunk

    def test_live_partial_page_ledger_conserves_all_three_swa_routes(self):
        """Reserved page tails remain owned without pretending to be mapped tokens."""
        for page_size in (128, 256):
            for kind in ("unified", "legacy", "chunk"):
                with self.subTest(page_size=page_size, kind=kind):
                    cache, allocator = _make_cache(
                        page_size=page_size,
                        window=page_size,
                        kind=kind,
                    )
                    full = allocator.alloc_extend(
                        prefix_lens=[0],
                        seq_lens=[1],
                        last_loc=[0],
                        extend_num_tokens=1,
                        dp_rank=0,
                    )
                    self.assertIsNotNone(full)
                    cache.req_to_token_pool.write((0, slice(0, 1)), full)
                    req = SimpleNamespace(
                        rid=f"{kind}-{page_size}",
                        dp_rank=0,
                        req_pool_idx=0,
                        cache_protected_len=0,
                        kv_allocated_len=1,
                        swa_evicted_seqlen=0,
                    )

                    snapshot = cache.cache_ledger_snapshot(0, [req])

                    self.assertEqual(snapshot["full_request_owned"], 1)
                    self.assertEqual(snapshot["swa_request_owned"], 1)
                    self.assertEqual(snapshot["full_reserved_page_slack"], page_size - 1)
                    self.assertEqual(snapshot["swa_reserved_page_slack"], page_size - 1)
                    self.assertEqual(snapshot["mapping_nonzero_count"], 1)
                    validate_swa_cache_ledger(snapshot, require_idle=False)

                    if kind == "legacy":
                        cache.insert(
                            InsertParams(
                                key=RadixKey([1]),
                                value=full,
                            )
                        )
                        idle_snapshot = cache.cache_ledger_snapshot(0, [])
                        self.assertEqual(idle_snapshot["full_tree_evictable"], 1)
                        self.assertEqual(idle_snapshot["swa_tree_evictable"], 1)
                        self.assertEqual(idle_snapshot["full_request_owned"], 0)
                        self.assertEqual(idle_snapshot["swa_request_owned"], 0)
                        self.assertEqual(idle_snapshot["full_reserved_page_slack"], page_size - 1)
                        self.assertEqual(idle_snapshot["swa_reserved_page_slack"], page_size - 1)
                        validate_swa_cache_ledger(idle_snapshot, require_idle=True)

    def test_unscoped_eviction_totals_are_attributed_to_each_rank_and_survive_reset(self):
        cache, allocator = _make_cache(page_size=128, dp_size=2, window=128)
        _insert(cache, allocator, list(range(128)), dp_rank=0)
        _insert(cache, allocator, list(range(1000, 1128)), dp_rank=1)

        result = cache.evict(EvictParams(swa_num_tokens=256, dp_rank=None))

        self.assertEqual(result.swa_num_tokens_evicted, 256)
        before_reset = [cache.cache_ledger_snapshot(rank, []) for rank in (0, 1)]
        self.assertEqual([row["full_evicted_total"] for row in before_reset], [128, 128])
        self.assertEqual([row["swa_evicted_total"] for row in before_reset], [128, 128])
        cache.reset()
        after_reset = [cache.cache_ledger_snapshot(rank, []) for rank in (0, 1)]
        self.assertEqual(
            [row["full_evicted_total"] for row in after_reset],
            [row["full_evicted_total"] for row in before_reset],
        )
        self.assertEqual(
            [row["swa_evicted_total"] for row in after_reset],
            [row["swa_evicted_total"] for row in before_reset],
        )

    def test_rank_ledger_rejects_unowned_allocator_leak(self):
        """A request allocation omitted from ownership must fail conservation."""
        cache, allocator = _make_cache(page_size=128, dp_size=2, window=128)
        ownership = _RequestOwnership()
        _insert(cache, allocator, list(range(128)), dp_rank=0, ownership=ownership)
        leaked = allocator.alloc(128, dp_rank=0)
        self.assertIsNotNone(leaked)
        with self.assertRaises(AssertionError):
            _assert_rank_ledger(self, cache, allocator, ownership, 0)

    def test_paged_insert_split_and_heal_are_page_aligned_page128(self):
        cache, allocator = _make_cache(page_size=128, window=256)
        _insert(cache, allocator, list(range(256)))
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(256))))
        ).last_device_node
        old_full = node.component_data[ComponentType.FULL].value.copy()
        cache.components[ComponentType.SWA].evict_component(node)
        replacement = allocator.alloc(256)
        self.assertIsNotNone(replacement)
        allocator.free_swa(replacement[:128])
        full_free_before = allocator.full_available_size()
        swa_free_before = allocator.swa_available_size()
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(256))),
                value=replacement,
                swa_evicted_seqlen=128,
            )
        )
        parent = node.parent
        self.assertEqual(len(parent.key), 128)
        self.assertIsNone(parent.component_data[ComponentType.SWA].value)
        self.assertEqual(len(parent.key) % cache.page_size, 0)
        np.testing.assert_array_equal(
            node.component_data[ComponentType.FULL].value, replacement[128:]
        )
        self.assertTrue(
            np.all(_swa_indices(allocator, parent.component_data[ComponentType.FULL].value, 0) == 0)
        )
        self.assertTrue(np.all(_swa_indices(allocator, old_full, 0) == 0))
        self.assertEqual(allocator.full_available_size(), full_free_before + 256)
        self.assertEqual(allocator.swa_available_size(), swa_free_before)

    def test_paged_insert_split_and_heal_are_page_aligned_page256(self):
        cache, allocator = _make_cache(page_size=256, window=256)
        _insert(cache, allocator, list(range(512)))
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(512))))
        ).last_device_node
        old_full = node.component_data[ComponentType.FULL].value.copy()
        cache.components[ComponentType.SWA].evict_component(node)
        replacement = allocator.alloc(256)
        self.assertIsNotNone(replacement)
        replacement_swa = _swa_indices(allocator, replacement, 0)
        self.assertTrue(np.all(replacement_swa > 0))
        full_free_before = _allocator_free_indices(allocator.full_attn_allocator, 0)
        swa_free_before = _allocator_free_indices(allocator.swa_attn_allocator, 0)
        boundary = cache.components[ComponentType.SWA].update_component_on_insert_overlap(
            node,
            256,
            256,
            replacement,
            InsertParams(swa_evicted_seqlen=256),
        )
        self.assertEqual(boundary, 0)
        self.assertEqual(len(node.key), 256)
        self.assertEqual(len(node.key) % cache.page_size, 0)
        np.testing.assert_array_equal(node.component_data[ComponentType.FULL].value, replacement)
        np.testing.assert_array_equal(node.component_data[ComponentType.SWA].value, replacement_swa)
        self.assertTrue(np.all(_swa_indices(allocator, old_full, 0) == 0))
        self.assertEqual(
            _allocator_free_indices(allocator.full_attn_allocator, 0),
            full_free_before | set(int(index) for index in old_full),
        )
        self.assertEqual(_allocator_free_indices(allocator.swa_attn_allocator, 0), swa_free_before)

    def test_page_larger_than_window_protects_last_physical_page(self):
        cache, allocator = _make_cache(page_size=256, window=128)
        _insert(cache, allocator, list(range(512)))
        node = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(512))))
        ).last_device_node
        lock = cache.inc_lock_ref(node)
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 256)
        cache.dec_lock_ref(node, lock.to_dec_params())
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 0)

        lock = cache.inc_lock_ref(node)
        result = cache.evict(EvictParams(swa_num_tokens=128))
        self.assertEqual(result.swa_num_tokens_evicted, 256)
        self.assertIsNotNone(node.component_data[ComponentType.SWA].value)
        cache.dec_lock_ref(node, lock.to_dec_params())

    def test_paged_evict_result_equals_allocator_page_delta(self):
        cache, allocator = _make_cache(page_size=128, window=128)
        _insert(cache, allocator, list(range(128)))
        before = allocator.swa_available_size()
        result = cache.evict(EvictParams(swa_num_tokens=128))
        self.assertEqual(result.swa_num_tokens_evicted, allocator.swa_available_size() - before)
        self.assertEqual(result.swa_num_tokens_evicted, 128)

    def test_dp2_same_local_indices_have_rank_local_swa_values(self):
        cache, allocator = _make_cache(page_size=128, dp_size=2, window=128)
        _insert(cache, allocator, list(range(128)), dp_rank=0)
        reserved = allocator.swa_attn_allocator.alloc(128, dp_rank=1)
        self.assertIsNotNone(reserved)
        _insert(cache, allocator, list(range(1000, 1128)), dp_rank=1)
        node0 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(128)), dp_rank=0))
        ).last_device_node
        node1 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(1000, 1128)), dp_rank=1))
        ).last_device_node
        np.testing.assert_array_equal(
            node0.component_data[ComponentType.FULL].value,
            node1.component_data[ComponentType.FULL].value,
        )
        self.assertFalse(
            np.array_equal(
                node0.component_data[ComponentType.SWA].value,
                node1.component_data[ComponentType.SWA].value,
            )
        )

    def test_dp2_rejects_ambiguous_missing_node_rank(self):
        cache, allocator = _make_cache(page_size=128, dp_size=2, window=128)
        with self.assertRaisesRegex(ValueError, "dp_rank"):
            _insert(cache, allocator, list(range(128)))

    def test_dp2_free_evict_heal_mutate_only_target_rank(self):
        cache, allocator = _make_cache(page_size=128, dp_size=2, window=128)
        _insert(cache, allocator, list(range(128)), dp_rank=0)
        _insert(cache, allocator, list(range(1000, 1128)), dp_rank=1)
        node0 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(128)), dp_rank=0))
        ).last_device_node
        node1 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(1000, 1128)), dp_rank=1))
        ).last_device_node
        full0 = node0.component_data[ComponentType.FULL].value.copy()
        full1 = node1.component_data[ComponentType.FULL].value.copy()
        swa0 = _swa_indices(allocator, full0, 0)
        swa1 = _swa_indices(allocator, full1, 1)
        rank0_full_free_before_evict = _allocator_free_indices(allocator.full_attn_allocator, 0)
        rank0_swa_free_before_evict = _allocator_free_indices(allocator.swa_attn_allocator, 0)
        rank1_full_free = _allocator_free_indices(allocator.full_attn_allocator, 1)
        rank1_swa_free = _allocator_free_indices(allocator.swa_attn_allocator, 1)
        cache.components[ComponentType.SWA].evict_component(node0)
        self.assertTrue(np.all(_swa_indices(allocator, full0, 0) == 0))
        self.assertTrue(np.all(_swa_indices(allocator, full1, 1) > 0))
        self.assertEqual(
            _allocator_free_indices(allocator.full_attn_allocator, 0), rank0_full_free_before_evict
        )
        self.assertEqual(
            _allocator_free_indices(allocator.swa_attn_allocator, 0),
            rank0_swa_free_before_evict | set(int(index) for index in swa0),
        )
        self.assertEqual(_allocator_free_indices(allocator.full_attn_allocator, 1), rank1_full_free)
        self.assertEqual(_allocator_free_indices(allocator.swa_attn_allocator, 1), rank1_swa_free)
        replacement = allocator.alloc(128, dp_rank=0)
        self.assertIsNotNone(replacement)
        replacement_swa = _swa_indices(allocator, replacement, 0)
        rank0_full_free_before_heal = _allocator_free_indices(allocator.full_attn_allocator, 0)
        rank0_swa_free_before_heal = _allocator_free_indices(allocator.swa_attn_allocator, 0)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(128)), dp_rank=0),
                value=replacement,
                swa_evicted_seqlen=0,
            )
        )
        np.testing.assert_array_equal(node0.component_data[ComponentType.FULL].value, replacement)
        np.testing.assert_array_equal(
            node0.component_data[ComponentType.SWA].value, replacement_swa
        )
        self.assertTrue(np.all(replacement_swa > 0))
        self.assertEqual(
            _allocator_free_indices(allocator.full_attn_allocator, 0),
            rank0_full_free_before_heal | set(int(index) for index in full0),
        )
        self.assertEqual(
            _allocator_free_indices(allocator.swa_attn_allocator, 0), rank0_swa_free_before_heal
        )
        np.testing.assert_array_equal(node1.component_data[ComponentType.FULL].value, full1)
        np.testing.assert_array_equal(node1.component_data[ComponentType.SWA].value, swa1)
        self.assertEqual(_allocator_free_indices(allocator.full_attn_allocator, 1), rank1_full_free)
        self.assertEqual(_allocator_free_indices(allocator.swa_attn_allocator, 1), rank1_swa_free)

    def test_dp2_ledger_conservation_by_rank(self):
        cache, allocator = _make_cache(page_size=128, dp_size=2, window=128)
        ownership = _RequestOwnership()
        _insert(cache, allocator, list(range(128)), dp_rank=0, ownership=ownership)
        _insert(cache, allocator, list(range(1000, 1128)), dp_rank=1, ownership=ownership)
        for rank in (0, 1):
            _assert_rank_ledger(self, cache, allocator, ownership, rank)
            snapshot = cache.cache_ledger_snapshot(rank, [])
            validate_swa_cache_ledger(snapshot, require_idle=True)
            self.assertEqual(snapshot["full_tree_evictable"], 128)
            self.assertEqual(snapshot["swa_tree_evictable"], 128)
        node0 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(128)), dp_rank=0))
        ).last_device_node
        lock = cache.inc_lock_ref(node0)
        _assert_rank_ledger(self, cache, allocator, ownership, 0)
        _assert_rank_ledger(self, cache, allocator, ownership, 1)
        cache.dec_lock_ref(node0, lock.to_dec_params())

    def test_real_retract_and_abort_release_receipts_and_window_locks(self):
        class _ReleaseReq:
            def __init__(self, cache, allocator, rid):
                tokens = list(range(4))
                _insert(cache, allocator, tokens)
                match = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
                tail = allocator.alloc(4)
                assert tail is not None
                row = np.concatenate([np.asarray(match.device_indices), tail])
                cache.req_to_token_pool.write((0, slice(0, len(row))), row)
                lock = cache.inc_lock_ref(match.last_device_node)
                self.rid = rid
                self.req_pool_idx = 0
                self.dp_rank = None
                self.origin_input_ids = tokens + list(range(10, 14))
                self.radix_input_ids = list(self.origin_input_ids)
                self.output_ids = []
                self.fill_ids = list(self.origin_input_ids)
                self.prefix_indices = np.asarray(match.device_indices)
                self.last_node = match.last_device_node
                self.extra_key = None
                self.cache_protected_len = 4
                self.last_matched_prefix_len = 4
                self.kv_committed_len = 8
                self.kv_allocated_len = 8
                self.kv_committed_freed = False
                self.kv_overallocated_freed = False
                self.cache_lock_params = lock.to_dec_params()
                self.swa_uuid_for_lock = self.cache_lock_params.swa_uuid_for_lock
                self.is_chunked = 0
                self.reset_observed_clean_release = False

            def pop_committed_kv_cache(self):
                self.kv_committed_freed = True
                return self.kv_committed_len

            def pop_overallocated_kv_cache(self):
                self.kv_overallocated_freed = True
                return self.kv_committed_len, self.kv_allocated_len

            def reset_for_retract(self):
                self.reset_observed_clean_release = self.cache_lock_params is None

            def check_finished(self):
                return None

            def finished(self):
                return True

        cache, allocator = _make_cache(window=4)
        retract_req = _ReleaseReq(cache, allocator, "retract")
        batch = SimpleNamespace(
            reqs_info=[SimpleNamespace(reqs=[retract_req])],
            tree_cache=cache,
            _evict_tree_cache_if_needed=lambda needed: None,
        )

        ScheduleBatch.release_req(batch, 0, 0, 0, SimpleNamespace())

        self.assertTrue(retract_req.reset_observed_clean_release)
        self.assertIsNone(retract_req.cache_lock_params)
        self.assertIsNone(retract_req.swa_uuid_for_lock)
        for component_data in retract_req.last_node.component_data:
            self.assertEqual(component_data.lock_ref, 0)
        validate_swa_cache_ledger(cache.cache_ledger_snapshot(0, []), require_idle=True)

        cache, allocator = _make_cache(window=4)
        abort_req = _ReleaseReq(cache, allocator, "abort")
        scheduler = SimpleNamespace(
            chunked_reqs=[abort_req],
            _pending_chunked_abort_reqs=[abort_req],
            _release_prefill_host_buffer=lambda req: None,
            tree_cache=cache,
            spec_algorithm=None,
        )

        SchedulerOutputProcessorMixin._finalize_chunked_abort(scheduler, abort_req, 0)

        self.assertIsNone(abort_req.cache_lock_params)
        self.assertIsNone(abort_req.swa_uuid_for_lock)
        self.assertIsNone(abort_req.req_pool_idx)
        for component_data in abort_req.last_node.component_data:
            self.assertEqual(component_data.lock_ref, 0)
        validate_swa_cache_ledger(cache.cache_ledger_snapshot(0, []), require_idle=True)


if __name__ == "__main__":
    unittest.main()
