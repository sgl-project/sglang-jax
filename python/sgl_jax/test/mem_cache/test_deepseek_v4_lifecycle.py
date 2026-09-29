"""Request-resource lifecycle contracts, without scheduler or model wiring."""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sgl_jax.srt.mem_cache.chunk_cache import DeepseekV4ChunkCache
from sgl_jax.srt.mem_cache.common import reclaim_completed_v4_swa
from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import (
    DeepseekV4CompressStatePool,
    score_slice,
)
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool


def make_cache():
    page_size = 128
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("data",))
    spec = DeepseekV4CacheSpec((0, 4, 128), head_dim=8, index_head_dim=4)
    kv = DeepseekV4TokenToKVPool(6 * page_size, 6 * page_size, page_size, spec, mesh)
    allocator = DeepseekV4TokenToKVPoolAllocator(kv)
    return DeepseekV4ChunkCache(ReqToTokenPool(1, 1024), allocator, page_size, 128)


def new_request(cache):
    req = SimpleNamespace(
        req_pool_idx=None,
        dp_rank=0,
        kv_committed_len=0,
        kv_allocated_len=0,
        swa_evicted_seqlen=0,
        prefix_indices=np.empty(0, np.int32),
        last_node=None,
        last_host_node=None,
        cache_protected_len=0,
        kv_committed_freed=False,
        kv_overallocated_freed=False,
    )
    cache.req_to_token_pool.alloc([req])
    return req


def append(cache, req, end, *, committed=None):
    start = req.kv_allocated_len
    old = cache.req_to_token_pool.read(req.req_pool_idx, start)
    loc = cache.token_to_kv_pool_allocator.alloc_extend(
        [start], [end], [int(old[-1]) if start else -1], end - start, req.dp_rank
    )
    if loc is None:
        raise AssertionError("test cache is too small")
    cache.req_to_token_pool.write((req.req_pool_idx, slice(start, end)), loc)
    req.kv_allocated_len = end
    req.kv_committed_len = end if committed is None else committed
    return cache.req_to_token_pool.read(req.req_pool_idx, end)


class TestDeepseekV4Lifecycle(unittest.TestCase):
    def test_chunk_continuation_and_request_slot_reuse(self):
        cache = make_cache()
        req = new_request(cache)
        self.assertEqual(req.req_pool_idx, 0)
        first = append(cache, req, 127)
        cache.cache_unfinished_req(req)
        np.testing.assert_array_equal(req.prefix_indices, first)
        append(cache, req, 129)
        matched = cache.match_prefix(MatchPrefixParams(key=None, req=req))
        self.assertEqual(len(matched.device_indices), 129)
        self.assertEqual(int(matched.device_indices[126]), int(first[-1]))

        cache.release_req(req)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(cache.token_to_kv_pool_allocator.full_available_size(), 6 * 128)
        self.assertEqual(cache.token_to_kv_pool_allocator.swa_available_size(), 6 * 128)
        self.assertFalse(cache.req_to_token_pool.req_to_token.any())
        cache.release_req(req)  # Repeated cleanup must not free another owner.

        replacement = new_request(cache)
        self.assertEqual(replacement.req_pool_idx, 0)
        second = append(cache, replacement, 128)
        self.assertEqual(len(second), 128)
        before_mapping = cache.token_to_kv_pool_allocator.full_to_swa_index_mapping.copy()
        before_available = cache.token_to_kv_pool_allocator.full_available_size()
        cache.release_req(req)  # An old request cannot reclaim a recycled slot.
        np.testing.assert_array_equal(cache.req_to_token_pool.read(0, 128), second)
        np.testing.assert_array_equal(
            cache.token_to_kv_pool_allocator.full_to_swa_index_mapping, before_mapping
        )
        self.assertEqual(cache.token_to_kv_pool_allocator.full_available_size(), before_available)
        cache.release_req(replacement)

    def test_host_and_numerical_state_continue_across_chunks(self):
        cache = make_cache()
        kv = cache.token_to_kv_pool_allocator.get_kvcache()
        state = DeepseekV4CompressStatePool(1, kv.spec, kv.mesh)
        req = new_request(cache)
        first = append(cache, req, 127)
        for family, layer, value in (("c4", 1, 3), ("c128", 2, 5), ("indexer", 1, 7)):
            old = state.get_buffer(family, layer)
            state.write(
                family,
                layer,
                jnp.array([req.req_pool_idx]),
                jnp.full((1, *old.shape[1:]), value, jnp.float32),
                jnp.array([True]),
            )
        cache.cache_unfinished_req(req)
        np.testing.assert_array_equal(req.prefix_indices, first)
        append(cache, req, 257)
        match = cache.match_prefix(MatchPrefixParams(key=None, req=req))
        np.testing.assert_array_equal(match.device_indices, cache.req_to_token_pool.read(0, 257))
        for family, layer, value in (("c4", 1, 3), ("c128", 2, 5), ("indexer", 1, 7)):
            self.assertTrue(np.all(np.asarray(state.get_buffer(family, layer))[0] == value))
        cache.release_req(req)

    def test_recycled_slot_uses_explicit_numerical_initialization(self):
        cache = make_cache()
        kv = cache.token_to_kv_pool_allocator.get_kvcache()
        state = DeepseekV4CompressStatePool(1, kv.spec, kv.mesh)
        req = new_request(cache)
        old = state.get_buffer("c4", 1)
        state.write("c4", 1, jnp.array([0]), jnp.ones((1, *old.shape[1:])), jnp.array([True]))
        cache.release_req(req)
        replacement = new_request(cache)
        self.assertEqual(replacement.req_pool_idx, 0)
        # R/B apply the first-forward initialization mask before this slot is read.
        state.reset(jnp.array([replacement.req_pool_idx]), jnp.array([True]))
        empty = np.asarray(state.get_buffer("c4", 1))[0]
        self.assertTrue(np.all(empty[..., : empty.shape[-1] // 2] == 0))
        self.assertTrue(np.all(np.isneginf(empty[score_slice(empty.shape)])))
        cache.release_req(replacement)

    def test_reclaim_completed_boundary_and_keep_history(self):
        cache = make_cache()
        req = new_request(cache)
        locations = append(cache, req, 400)
        allocator = cache.token_to_kv_pool_allocator
        self.assertEqual(allocator.count_swa_mapped(locations), 400)
        # R invokes this helper only when the forward producing committed_len
        # has completed. The next query still needs its containing SWA page.
        reclaim_completed_v4_swa(req, cache)
        self.assertEqual(req.swa_evicted_seqlen, 256)
        self.assertEqual(allocator.count_swa_mapped(locations[:256]), 0)
        self.assertEqual(allocator.count_swa_mapped(locations[256:]), 144)
        self.assertEqual(allocator.full_available_size(), 2 * 128)
        reclaim_completed_v4_swa(req, cache)
        self.assertEqual(allocator.swa_available_size(), 4 * 128)
        cache.release_req(req)
        self.assertEqual(allocator.full_available_size(), 6 * 128)
        self.assertEqual(allocator.swa_available_size(), 6 * 128)

    def test_reclamation_uses_committed_length_not_allocated_tail(self):
        cache = make_cache()
        req = new_request(cache)
        locations = append(cache, req, 400, committed=127)
        allocator = cache.token_to_kv_pool_allocator
        reclaim_completed_v4_swa(req, cache)
        self.assertEqual(req.swa_evicted_seqlen, 0)
        self.assertEqual(allocator.count_swa_mapped(locations), 400)
        req.kv_committed_len = 400
        reclaim_completed_v4_swa(req, cache)
        self.assertEqual(req.swa_evicted_seqlen, 256)
        self.assertEqual(allocator.count_swa_mapped(locations[256:]), 144)
        cache.release_req(req)

    def test_release_includes_allocated_tail_for_cancel_or_retract(self):
        cache = make_cache()
        req = new_request(cache)
        append(cache, req, 129, committed=127)
        cache.release_req(req)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(req.kv_allocated_len, 0)
        self.assertEqual(req.kv_committed_len, 0)
        self.assertEqual(cache.token_to_kv_pool_allocator.full_available_size(), 6 * 128)
        self.assertEqual(cache.token_to_kv_pool_allocator.swa_available_size(), 6 * 128)

    def test_retracted_request_recomputes_from_zero(self):
        cache = make_cache()
        kv = cache.token_to_kv_pool_allocator.get_kvcache()
        state = DeepseekV4CompressStatePool(1, kv.spec, kv.mesh)
        req = new_request(cache)
        old_locations = append(cache, req, 129, committed=127)
        cache.cache_unfinished_req(req)
        self.assertEqual(len(req.prefix_indices), 127)

        old_state = state.get_buffer("c4", 1)
        state.write(
            "c4",
            1,
            jnp.array([req.req_pool_idx]),
            jnp.ones((1, *old_state.shape[1:]), jnp.float32),
            jnp.array([True]),
        )
        cache.release_req(req)
        self.assertEqual(len(req.prefix_indices), 0)
        self.assertEqual(cache.token_to_kv_pool_allocator.count_swa_mapped(old_locations), 0)
        self.assertFalse(cache.req_to_token_pool.req_to_token.any())
        self.assertTrue(np.all(np.asarray(state.get_buffer("c4", 1))[0] == 1))

        self.assertEqual(cache.req_to_token_pool.alloc([req]), [0])
        self.assertEqual(
            len(cache.match_prefix(MatchPrefixParams(key=None, req=req)).device_indices), 0
        )
        # The first forward after retract resets the reused numerical state slot.
        state.reset(jnp.array([req.req_pool_idx]), jnp.array([True]))
        empty = np.asarray(state.get_buffer("c4", 1))[0]
        self.assertTrue(np.all(empty[..., : empty.shape[-1] // 2] == 0))
        self.assertTrue(np.all(np.isneginf(empty[score_slice(empty.shape)])))

        fresh_locations = append(cache, req, 257)
        cache.cache_unfinished_req(req)
        np.testing.assert_array_equal(req.prefix_indices, fresh_locations)
        self.assertEqual(cache.token_to_kv_pool_allocator.count_swa_mapped(fresh_locations), 257)
        cache.release_req(req)
        self.assertEqual(cache.token_to_kv_pool_allocator.full_available_size(), 6 * 128)
        self.assertEqual(cache.token_to_kv_pool_allocator.swa_available_size(), 6 * 128)


if __name__ == "__main__":
    unittest.main()
