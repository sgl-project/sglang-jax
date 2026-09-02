"""Device-only SWA coverage for the unified radix component seam."""

# ruff: noqa: E402

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

# Must precede the first direct JAX import: this module owns a 2x2 CPU mesh.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

assert jax.device_count() == 4

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.unified_cache_components import (
    ComponentType,
    LRURefreshPhase,
)
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sgl_jax.test.test_utils import CustomTestCase


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


def _insert(
    cache,
    allocator,
    tokens: list[int],
    *,
    dp_rank: int | None = None,
):
    rank = dp_rank or 0
    value = allocator.alloc(len(tokens), dp_rank=rank)
    assert value is not None
    result = cache.insert(InsertParams(key=RadixKey(tokens, dp_rank=dp_rank), value=value))
    return result


def _swa_indices(allocator, full_indices, rank: int) -> np.ndarray:
    return allocator.translate_full_to_swa(full_indices, dp_rank=rank, require_mapped=False)


def _allocator_free_indices(pool, rank: int) -> set[int]:
    if hasattr(pool, "free_slots"):
        return set(int(index) for index in pool.free_slots[rank])
    pages = np.concatenate((pool.free_pages[rank], pool.release_pages[rank]))
    indices = pages[:, None] * pool.page_size + np.arange(pool.page_size)
    return set(int(index) for index in indices.reshape(-1))


class TestUnifiedSWAComponentPage1(CustomTestCase):

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
        self.assertTrue(cache.supports_swa())
        self.assertEqual(cache.sliding_window_size, 4)
        self.assertEqual(cache.full_protected_size(0), 0)
        self.assertEqual(cache.swa_protected_size(0), 0)
        _insert(cache, allocator, list(range(8)))
        self.assertEqual(
            len(cache.match_prefix(MatchPrefixParams(key=RadixKey(list(range(8))))).device_indices),
            8,
        )
        _insert(cache, allocator, [0, 1, 2, 3, 20, 21, 22, 23])
        parent = cache.match_prefix(MatchPrefixParams(key=RadixKey([0, 1, 2, 3]))).last_device_node
        self.assertEqual(len(parent.key), 4)
        self.assertEqual(cache.component_evictable_size_[ComponentType.SWA][0], 12)
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

    def test_unfinished_cache_preserves_nonzero_swa_eviction_boundary(self):
        cache, allocator = _make_cache(window=4)
        tokens = [0, 1, 2, 3]
        full = allocator.alloc(len(tokens), dp_rank=0)
        self.assertIsNotNone(full)
        cache.req_to_token_pool.write((0, slice(0, len(tokens))), full)
        allocator.free_swa(full[:2], dp_rank=0)
        req = SimpleNamespace(
            req_pool_idx=0,
            origin_input_ids=tokens,
            radix_input_ids=list(tokens),
            output_ids=[],
            fill_ids=list(tokens),
            prefix_indices=np.empty((0,), dtype=np.int32),
            last_node=cache.root_node,
            extra_key=None,
            dp_rank=None,
            cache_protected_len=0,
            swa_evicted_seqlen=2,
        )

        cache.cache_unfinished_req(req)

        tombstone = next(iter(cache.root_node.children.values()))
        live_suffix = next(iter(tombstone.children.values()))
        self.assertEqual(len(tombstone.key), 2)
        self.assertIsNone(tombstone.component_data[ComponentType.SWA].value)
        np.testing.assert_array_equal(
            live_suffix.component_data[ComponentType.SWA].value,
            _swa_indices(allocator, full[2:], 0),
        )
        self.assertTrue(np.all(_swa_indices(allocator, full[:2], 0) == 0))

    def test_lru_refresh_includes_page_cushion_but_lock_stays_window_bounded(self):
        cache, allocator = _make_cache(page_size=2, window=4)
        _insert(cache, allocator, list(range(12)))
        endpoint = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(list(range(8))))
        ).last_device_node
        first = next(iter(cache.root_node.children.values()))
        descendant = next(iter(endpoint.children.values()))
        first_data = first.component_data[ComponentType.SWA]
        endpoint_data = endpoint.component_data[ComponentType.SWA]
        descendant_data = descendant.component_data[ComponentType.SWA]
        first_data.metadata["last_access_time"] = 1.0
        endpoint_data.metadata["last_access_time"] = 2.0
        descendant_data.metadata["last_access_time"] = 3.0

        component = cache.components[ComponentType.SWA]
        component.refresh_lru(LRURefreshPhase.MATCH_END, endpoint, cache.root_node)
        self.assertGreater(first_data.metadata["last_access_time"], 1.0)
        self.assertGreater(endpoint_data.metadata["last_access_time"], 2.0)
        self.assertLess(
            first_data.metadata["last_access_time"],
            endpoint_data.metadata["last_access_time"],
        )
        self.assertEqual(descendant_data.metadata["last_access_time"], 3.0)

        lock = cache.inc_lock_ref(endpoint)
        self.assertEqual(cache.component_protected_size_[ComponentType.SWA][0], 4)
        cache.dec_lock_ref(endpoint, lock.to_dec_params())

    def test_lru_refresh_counts_tombstones_in_logical_window(self):
        cache, allocator = _make_cache(page_size=2, window=4)
        matched_tokens = list(range(12))
        _insert(cache, allocator, matched_tokens)
        _insert(cache, allocator, list(range(8)) + list(range(100, 104)))
        _insert(cache, allocator, list(range(4)) + list(range(200, 204)))

        ancestor = next(iter(cache.root_node.children.values()))
        middle = next(
            child
            for child in ancestor.children.values()
            if list(child.key.token_ids) == list(range(4, 8))
        )
        endpoint = next(
            child
            for child in middle.children.values()
            if list(child.key.token_ids) == list(range(8, 12))
        )
        ancestor_data = ancestor.component_data[ComponentType.SWA]
        middle_data = middle.component_data[ComponentType.SWA]
        endpoint_data = endpoint.component_data[ComponentType.SWA]
        self.assertEqual([len(ancestor.key), len(middle.key), len(endpoint.key)], [4, 4, 4])
        cache.components[ComponentType.SWA].evict_component(middle)
        self.assertIsNone(middle_data.value)
        ancestor_data.metadata["last_access_time"] = 1.0
        endpoint_data.metadata["last_access_time"] = 2.0

        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(matched_tokens)))

        self.assertIs(match.last_device_node, endpoint)
        self.assertEqual(ancestor_data.metadata["last_access_time"], 1.0)
        self.assertGreater(endpoint_data.metadata["last_access_time"], 2.0)

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

    def test_overlap_healing_decision_table_preserves_boundary_semantics(self):
        """Before/inside/after boundaries heal only request-owned tombstone suffixes."""
        cases = (
            ("before", 0, 0, 8),
            ("inside", 4, 4, 4),
            ("after", 8, 8, 0),
        )
        for name, evicted, expected_boundary, adopted in cases:
            with self.subTest(position=name):
                cache, allocator = _make_cache(window=8)
                _insert(cache, allocator, list(range(8)))
                node = next(iter(cache.root_node.children.values()))
                old_full = node.component_data[ComponentType.FULL].value.copy()
                cache.components[ComponentType.SWA].evict_component(node)
                self.assertTrue(np.all(_swa_indices(allocator, old_full, 0) == 0))

                replacement = allocator.alloc(8, dp_rank=0)
                self.assertIsNotNone(replacement)
                full_free_before = allocator.full_available_size()
                swa_free_before = allocator.swa_available_size()

                boundary = cache.components[ComponentType.SWA].update_component_on_insert_overlap(
                    node,
                    8,
                    0,
                    replacement,
                    InsertParams(swa_evicted_seqlen=evicted),
                )

                self.assertEqual(boundary, expected_boundary)
                self.assertEqual(allocator.full_available_size(), full_free_before + adopted)
                self.assertEqual(allocator.swa_available_size(), swa_free_before)
                if name == "before":
                    np.testing.assert_array_equal(
                        node.component_data[ComponentType.FULL].value,
                        replacement,
                    )
                    np.testing.assert_array_equal(
                        node.component_data[ComponentType.SWA].value,
                        _swa_indices(allocator, replacement, 0),
                    )
                elif name == "inside":
                    parent = node.parent
                    self.assertEqual(len(parent.key), 4)
                    self.assertIsNone(parent.component_data[ComponentType.SWA].value)
                    self.assertTrue(
                        np.all(
                            _swa_indices(
                                allocator,
                                parent.component_data[ComponentType.FULL].value,
                                0,
                            )
                            == 0
                        )
                    )
                    np.testing.assert_array_equal(
                        node.component_data[ComponentType.FULL].value,
                        replacement[4:],
                    )
                    np.testing.assert_array_equal(
                        node.component_data[ComponentType.SWA].value,
                        _swa_indices(allocator, replacement[4:], 0),
                    )
                else:
                    np.testing.assert_array_equal(
                        node.component_data[ComponentType.FULL].value,
                        old_full,
                    )
                    self.assertIsNone(node.component_data[ComponentType.SWA].value)

    def test_locked_tombstone_healing_preserves_full_indices_for_existing_request(self):
        """Healing may adopt fresh SWA slots, but must not free another request's FULL KV."""
        for name, evicted in (("whole", 0), ("partial", 4)):
            with self.subTest(position=name):
                cache, allocator = _make_cache(window=8)
                tokens = list(range(8))
                _insert(cache, allocator, tokens, dp_rank=0)
                node = next(iter(cache.root_node.children.values()))
                old_full = node.component_data[ComponentType.FULL].value.copy()
                cache.components[ComponentType.SWA].evict_component(node)
                self.assertTrue(np.all(_swa_indices(allocator, old_full, 0) == 0))

                existing_request_lock = cache.inc_lock_ref(node)
                self.assertEqual(node.component_data[ComponentType.FULL].lock_ref, 1)

                replacement = allocator.alloc(8, dp_rank=0)
                self.assertIsNotNone(replacement)
                replacement_swa = _swa_indices(allocator, replacement, 0)
                result = cache.insert(
                    InsertParams(
                        key=RadixKey(tokens, dp_rank=0),
                        value=replacement,
                        swa_evicted_seqlen=evicted,
                    )
                )

                self.assertEqual(result.prefix_len, 8)
                np.testing.assert_array_equal(
                    node.component_data[ComponentType.FULL].value,
                    old_full[evicted:],
                )
                np.testing.assert_array_equal(
                    node.component_data[ComponentType.SWA].value,
                    replacement_swa[evicted:],
                )
                self.assertTrue(np.all(_swa_indices(allocator, replacement, 0) == 0))

                full_free = _allocator_free_indices(allocator.full_attn_allocator, 0)
                self.assertTrue(set(int(index) for index in old_full).isdisjoint(full_free))
                self.assertTrue(set(int(index) for index in replacement).issubset(full_free))

                cache.dec_lock_ref(node, existing_request_lock.to_dec_params())
                self.assertEqual(cache.component_protected_size_[ComponentType.FULL][0], 0)

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

    def test_leaf_eviction_reports_exact_capacity_delta(self):
        cache, allocator = _make_cache(window=4)
        _insert(cache, allocator, list(range(4)))
        node = next(iter(cache.root_node.children.values()))
        full_free_before = allocator.full_available_size()
        swa_free_before = allocator.swa_available_size()

        result = cache.evict(EvictParams(swa_num_tokens=4))

        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.swa_num_tokens_evicted, 4)
        self.assertEqual(allocator.full_available_size(), full_free_before + 4)
        self.assertEqual(allocator.swa_available_size(), swa_free_before + 4)
        self.assertIsNone(node.component_data[ComponentType.FULL].value)


class TestUnifiedSWAComponentPaged(CustomTestCase):

    def test_request_tail_reclaim_preserves_tree_prefix_and_live_window(self):
        cache, allocator = _make_cache(page_size=4, window=8)
        tokens = list(range(8))
        _insert(cache, allocator, tokens)
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        tree_full = np.asarray(match.device_indices).copy()
        tree_swa = _swa_indices(allocator, tree_full, 0).copy()
        tail = allocator.alloc(16)
        self.assertIsNotNone(tail)
        row = np.concatenate([tree_full, tail])
        cache.req_to_token_pool.write((0, slice(0, len(row))), row)
        req = SimpleNamespace(
            req_pool_idx=0,
            last_node=match.last_device_node,
            swa_evicted_seqlen=0,
        )
        full_free_before = allocator.full_available_size()
        swa_free_before = allocator.swa_available_size()

        cache.evict_req_swa(req, pre_len=len(row), dp_rank=0)

        self.assertEqual(req.swa_evicted_seqlen, 12)
        np.testing.assert_array_equal(_swa_indices(allocator, tree_full, 0), tree_swa)
        self.assertTrue(np.all(_swa_indices(allocator, row[8:12], 0) == 0))
        self.assertTrue(np.all(_swa_indices(allocator, row[12:], 0) > 0))
        self.assertEqual(allocator.full_available_size(), full_free_before)
        self.assertEqual(allocator.swa_available_size(), swa_free_before + 4)

    def test_dp2_rejects_ambiguous_missing_node_rank(self):
        cache, allocator = _make_cache(page_size=4, dp_size=2, window=4)

        with self.assertRaisesRegex(ValueError, "dp_rank"):
            _insert(cache, allocator, list(range(4)))

    def test_dp2_swa_evict_and_heal_are_rank_local(self):
        cache, allocator = _make_cache(page_size=4, dp_size=2, window=8)
        tokens0 = list(range(8))
        tokens1 = list(range(100, 108))
        _insert(cache, allocator, tokens0, dp_rank=0)
        reserved = allocator.swa_attn_allocator.alloc(4, dp_rank=1)
        self.assertIsNotNone(reserved)
        _insert(cache, allocator, tokens1, dp_rank=1)
        allocator.swa_attn_allocator.free(reserved, dp_rank=1)
        node0 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens0, dp_rank=0))
        ).last_device_node
        node1 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(tokens1, dp_rank=1))
        ).last_device_node
        full0 = node0.component_data[ComponentType.FULL].value.copy()
        full1 = node1.component_data[ComponentType.FULL].value.copy()
        swa0 = node0.component_data[ComponentType.SWA].value.copy()
        rank0_full_free = allocator.full_available_size(0)
        rank0_swa_free = allocator.swa_available_size(0)
        rank1_full_free = allocator.full_available_size(1)
        rank1_swa_free = allocator.swa_available_size(1)

        cache.components[ComponentType.SWA].evict_component(node1)

        self.assertIsNone(node1.component_data[ComponentType.SWA].value)
        np.testing.assert_array_equal(node0.component_data[ComponentType.FULL].value, full0)
        np.testing.assert_array_equal(node0.component_data[ComponentType.SWA].value, swa0)
        self.assertTrue(np.all(_swa_indices(allocator, full1, 1) == 0))
        self.assertEqual(allocator.full_available_size(0), rank0_full_free)
        self.assertEqual(allocator.swa_available_size(0), rank0_swa_free)
        self.assertEqual(allocator.full_available_size(1), rank1_full_free)
        self.assertEqual(allocator.swa_available_size(1), rank1_swa_free + 8)

        replacement = allocator.alloc(8, dp_rank=1)
        self.assertIsNotNone(replacement)
        replacement_swa = _swa_indices(allocator, replacement, 1).copy()
        cache.insert(
            InsertParams(
                key=RadixKey(tokens1, dp_rank=1),
                value=replacement,
                swa_evicted_seqlen=0,
            )
        )

        np.testing.assert_array_equal(node1.component_data[ComponentType.FULL].value, replacement)
        np.testing.assert_array_equal(
            node1.component_data[ComponentType.SWA].value,
            replacement_swa,
        )
        np.testing.assert_array_equal(node0.component_data[ComponentType.FULL].value, full0)
        np.testing.assert_array_equal(node0.component_data[ComponentType.SWA].value, swa0)
        self.assertEqual(allocator.full_available_size(0), rank0_full_free)
        self.assertEqual(allocator.swa_available_size(0), rank0_swa_free)
        self.assertEqual(allocator.full_available_size(1), rank1_full_free)
        self.assertEqual(allocator.swa_available_size(1), rank1_swa_free)

    def test_paged_mid_node_split_and_heal_are_page_aligned(self):
        """Both page sizes exercise a mid-node split; page1 covers whole-node healing."""
        for page_size in (128, 256):
            with self.subTest(page_size=page_size):
                tokens = list(range(page_size * 2))
                cache, allocator = _make_cache(page_size=page_size, window=page_size * 2)
                _insert(cache, allocator, tokens)
                node = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens))).last_device_node
                old_full = node.component_data[ComponentType.FULL].value.copy()
                cache.components[ComponentType.SWA].evict_component(node)
                replacement = allocator.alloc(page_size * 2)
                self.assertIsNotNone(replacement)
                allocator.free_swa(replacement[:page_size])
                full_free_before = allocator.full_available_size()
                swa_free_before = allocator.swa_available_size()

                cache.insert(
                    InsertParams(
                        key=RadixKey(tokens),
                        value=replacement,
                        swa_evicted_seqlen=page_size,
                    )
                )

                parent = node.parent
                self.assertEqual(len(parent.key), page_size)
                self.assertEqual(len(parent.key) % cache.page_size, 0)
                self.assertIsNone(parent.component_data[ComponentType.SWA].value)
                np.testing.assert_array_equal(
                    node.component_data[ComponentType.FULL].value,
                    replacement[page_size:],
                )
                self.assertTrue(
                    np.all(
                        _swa_indices(
                            allocator,
                            parent.component_data[ComponentType.FULL].value,
                            0,
                        )
                        == 0
                    )
                )
                self.assertTrue(np.all(_swa_indices(allocator, old_full, 0) == 0))
                self.assertEqual(
                    allocator.full_available_size(),
                    full_free_before + page_size * 2,
                )
                self.assertEqual(allocator.swa_available_size(), swa_free_before)

    def test_paged_lock_and_eviction_use_physical_page_units(self):
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

        cache, allocator = _make_cache(page_size=128, window=128)
        _insert(cache, allocator, list(range(128)))
        before = allocator.swa_available_size()
        result = cache.evict(EvictParams(swa_num_tokens=128))
        self.assertEqual(result.swa_num_tokens_evicted, allocator.swa_available_size() - before)
        self.assertEqual(result.swa_num_tokens_evicted, 128)

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


if __name__ == "__main__":
    unittest.main()
