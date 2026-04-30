# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_swa_allocator.py -v

import os
import unittest
from types import SimpleNamespace

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.test.test_utils import CustomTestCase


def _make_mesh():
    # SWA pools now use DP-aware sharding specs: P("data", "tensor", None).
    # Build a degenerate 1x1 mesh so unit tests can exercise that path on a
    # single local device without needing multi-host setup.
    devices = np.array(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, axis_names=("data", "tensor"))


def _make_swa_pool(size, size_swa, page_size, mesh):
    """Create a minimal SWAKVPool for testing."""
    return SWAKVPool(
        size=size,
        size_swa=size_swa,
        page_size=page_size,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        token_to_kv_pool_class=MHATokenToKVPool,
        dtype=jnp.bfloat16,
        head_num=1,
        head_dim=1,
        mesh=mesh,
    )


# ---------------------------------------------------------------------------
# Class 1: Token-level allocator (page_size=1)
# ---------------------------------------------------------------------------
class TestSWAAllocatorTokenLevel(CustomTestCase):
    def setUp(self):
        self.mesh = _make_mesh()
        self.kvcache = _make_swa_pool(size=64, size_swa=32, page_size=1, mesh=self.mesh)
        self.alloc = SWATokenToKVPoolAllocator(
            size=64, size_swa=32, kvcache=self.kvcache, page_size=1
        )

    # 1
    def test_alloc_basic_creates_mapping(self):
        """alloc(n) returns full indices and mapping points to SWA indices."""
        indices = self.alloc.alloc(4)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 4)
        # Mapping should contain non-zero SWA indices
        swa_indices = self.alloc.full_to_swa_index_mapping[indices]
        self.assertTrue(np.all(swa_indices > 0))
        # SWA indices should be unique
        self.assertEqual(len(np.unique(swa_indices)), 4)

    # 2
    def test_alloc_exceeds_full_returns_none(self):
        """Full pool exhaustion returns None."""
        # Directly exhaust the full pool via underlying allocator
        self.alloc.full_attn_allocator.alloc(64)
        # Now full pool is empty, alloc should fail even though SWA has room
        result = self.alloc.alloc(1)
        self.assertIsNone(result)

    # 3
    def test_alloc_exceeds_swa_returns_none(self):
        """SWA pool exhaustion returns None (SWA < full)."""
        # SWA is 32, full is 64 → SWA exhausts first
        result = self.alloc.alloc(32)
        self.assertIsNotNone(result)
        result = self.alloc.alloc(1)
        self.assertIsNone(result)

    # 4
    def test_available_size_returns_min(self):
        """available_size() = min(full_avail, swa_avail)."""
        self.assertEqual(self.alloc.available_size(), 32)  # min(64, 32)
        self.alloc.alloc(10)
        self.assertEqual(self.alloc.available_size(), 22)  # min(54, 22)

    # 5
    def test_free_restores_both_pools(self):
        """free() returns tokens to both full and SWA pools."""
        indices = self.alloc.alloc(10)
        self.assertEqual(self.alloc.available_size(), 22)
        self.alloc.free(indices)
        self.assertEqual(self.alloc.full_available_size(), 64)
        self.assertEqual(self.alloc.swa_available_size(), 32)

    # 6
    def test_free_swa_only_releases_swa(self):
        """free_swa() releases SWA slots but keeps full slots allocated."""
        indices = self.alloc.alloc(10)
        full_before = self.alloc.full_available_size()
        swa_before = self.alloc.swa_available_size()

        self.alloc.free_swa(indices)

        # Full pool unchanged
        self.assertEqual(self.alloc.full_available_size(), full_before)
        # SWA pool restored
        self.assertEqual(self.alloc.swa_available_size(), swa_before + 10)

    # 7
    def test_free_swa_clears_mapping(self):
        """free_swa() zeroes out the mapping for freed indices."""
        indices = self.alloc.alloc(5)
        # Mapping should be non-zero
        self.assertTrue(np.all(self.alloc.full_to_swa_index_mapping[indices] > 0))
        self.alloc.free_swa(indices)
        # Mapping should be zero
        self.assertTrue(np.all(self.alloc.full_to_swa_index_mapping[indices] == 0))

    # 8
    def test_free_swa_idempotent(self):
        """free_swa() twice on same indices does not crash or double-free."""
        indices = self.alloc.alloc(5)
        self.alloc.free_swa(indices)
        swa_after_first = self.alloc.swa_available_size()
        # Second call should be a no-op (mapping is already 0)
        self.alloc.free_swa(indices)
        self.assertEqual(self.alloc.swa_available_size(), swa_after_first)


# ---------------------------------------------------------------------------
# Class 2: Paged allocator (page_size=4)
# ---------------------------------------------------------------------------
class TestSWAAllocatorPaged(CustomTestCase):
    def setUp(self):
        self.mesh = _make_mesh()
        self.page_size = 4
        self.size = 128
        self.size_swa = 64
        self.kvcache = _make_swa_pool(
            size=self.size,
            size_swa=self.size_swa,
            page_size=self.page_size,
            mesh=self.mesh,
        )
        self.alloc = SWATokenToKVPoolAllocator(
            size=self.size,
            size_swa=self.size_swa,
            kvcache=self.kvcache,
            page_size=self.page_size,
        )

    # 9
    def test_alloc_extend_mapping_correct(self):
        """alloc_extend() produces correct full→SWA mapping."""
        # Allocate initial tokens (simulating a prefix)
        prefix = self.alloc.alloc(8)
        self.assertIsNotNone(prefix)

        # Now extend: prefix_lens=[8], seq_lens=[12], last_loc=[last of prefix]
        last_loc = [int(prefix[-1])]
        full_indices = self.alloc.alloc_extend(
            prefix_lens=[8], seq_lens=[12], last_loc=last_loc, extend_num_tokens=4
        )
        self.assertIsNotNone(full_indices)
        self.assertEqual(len(full_indices), 4)

        # Each full index should have a valid SWA mapping
        for idx in full_indices:
            swa_idx = self.alloc.full_to_swa_index_mapping[idx]
            self.assertGreater(swa_idx, 0)

    # 10
    def test_alloc_decode_mapping_correct(self):
        """alloc_decode() produces correct full→SWA mapping."""
        # Allocate initial 4 tokens (1 page)
        prefix = self.alloc.alloc(4)
        self.assertIsNotNone(prefix)

        last_loc = [int(prefix[-1])]
        full_indices = self.alloc.alloc_decode(seq_lens=[5], last_loc=last_loc)
        self.assertIsNotNone(full_indices)
        self.assertEqual(len(full_indices), 1)

        swa_idx = self.alloc.full_to_swa_index_mapping[full_indices[0]]
        self.assertGreater(swa_idx, 0)

    # 11 — Bug 1 test (should FAIL before fix)
    def test_alloc_extend_rollback_on_swa_failure(self):
        """When SWA pool is exhausted, alloc_extend must roll back full-pool pages."""
        # Exhaust SWA pool: allocate size_swa tokens
        eaten = self.alloc.alloc(self.size_swa)
        self.assertIsNotNone(eaten)
        self.assertEqual(self.alloc.swa_available_size(), 0)
        full_before = self.alloc.full_available_size()

        # Try to extend — SWA is full, should fail
        # Set up a valid extend scenario: prefix_lens=[size_swa], seq_lens=[size_swa+4]
        last_loc = [int(eaten[-1])]
        result = self.alloc.alloc_extend(
            prefix_lens=[self.size_swa],
            seq_lens=[self.size_swa + 4],
            last_loc=last_loc,
            extend_num_tokens=4,
        )
        self.assertIsNone(result)

        # Full pool must have been rolled back (no leak)
        self.assertEqual(self.alloc.full_available_size(), full_before)

    # 12 — Bug 1 test (should FAIL before fix)
    def test_alloc_decode_rollback_on_swa_failure(self):
        """When SWA pool is exhausted, alloc_decode must roll back full-pool pages."""
        # Exhaust SWA pool
        eaten = self.alloc.alloc(self.size_swa)
        self.assertIsNotNone(eaten)
        self.assertEqual(self.alloc.swa_available_size(), 0)
        full_before = self.alloc.full_available_size()

        # Try to decode — SWA is full, should fail.
        # seq_lens=[size_swa+1] means we need the next token after the prefix.
        # last_loc is the last index of the allocated prefix.
        # We need the last index to be at a page boundary so decode needs a new page.
        # With page_size=4 and size_swa=64, last token is at offset 63.
        # seq_lens = 65 → needs page 17 (index 64 is 65th position → page 16*4=64..67)
        # Actually, eaten has 64 tokens, so we want seq_lens=65.
        # last_loc should be the last slot in eaten.
        last_loc = [int(eaten[-1])]
        result = self.alloc.alloc_decode(seq_lens=[self.size_swa + 1], last_loc=last_loc)
        self.assertIsNone(result)

        # Full pool must have been rolled back (no leak)
        self.assertEqual(self.alloc.full_available_size(), full_before)

    # 13
    def test_alloc_extend_swa_last_loc_translation(self):
        """alloc_extend translates last_loc through the full→SWA mapping."""
        prefix = self.alloc.alloc(4)
        self.assertIsNotNone(prefix)

        last_full = int(prefix[-1])
        expected_swa_last = int(self.alloc.full_to_swa_index_mapping[last_full])
        self.assertGreater(expected_swa_last, 0)

        # Extend by 4 more tokens
        result = self.alloc.alloc_extend(
            prefix_lens=[4], seq_lens=[8], last_loc=[last_full], extend_num_tokens=4
        )
        self.assertIsNotNone(result)

        # Verify new mappings exist
        for idx in result:
            self.assertGreater(int(self.alloc.full_to_swa_index_mapping[idx]), 0)

    # 14
    def test_free_releases_both_paged_pools(self):
        """free() returns pages to both full and SWA paged pools."""
        indices = self.alloc.alloc(8)
        self.assertIsNotNone(indices)
        self.alloc.free(indices)
        self.assertEqual(self.alloc.full_available_size(), self.size)
        self.assertEqual(self.alloc.swa_available_size(), self.size_swa)

    # 15 — Regression test for the OOB bug (GH-231)
    def test_mapping_covers_last_page_indices(self):
        """full_to_swa_index_mapping must be large enough for max paged token index.

        Before the fix, mapping size was size_per_rank+1, but PagedTokenToKVPoolAllocator
        can produce indices up to size_per_rank + page_size - 1, causing IndexError
        when alloc_extend or alloc_decode touches the last page.
        """
        # Use a small pool so we can easily exhaust pages and hit the last page
        ps = 4
        size = 32  # 8 pages (1..8)
        size_swa = 32
        kv = _make_swa_pool(size=size, size_swa=size_swa, page_size=ps, mesh=self.mesh)
        alloc = SWATokenToKVPoolAllocator(size=size, size_swa=size_swa, kvcache=kv, page_size=ps)

        # Verify mapping array is large enough: needs size + page_size
        self.assertGreaterEqual(len(alloc.full_to_swa_index_mapping), size + ps)

        # Allocate all pages (8 pages = 32 tokens)
        all_indices = alloc.alloc(size)
        self.assertIsNotNone(all_indices)

        # Max index = pages_per_rank * ps + ps - 1 = 8*4+3 = 35
        max_idx = int(np.max(all_indices))
        self.assertEqual(max_idx, size + ps - 1)

        # This must NOT raise IndexError
        swa_val = alloc.full_to_swa_index_mapping[max_idx]
        self.assertGreater(swa_val, 0)

    # 15b — Regression: alloc_extend on last page must not OOB (GH-231)
    def test_alloc_extend_last_page_no_oob(self):
        """alloc_extend touching the last page must not raise IndexError."""
        ps = 4
        size = 32
        size_swa = 32
        kv = _make_swa_pool(size=size, size_swa=size_swa, page_size=ps, mesh=self.mesh)
        alloc = SWATokenToKVPoolAllocator(size=size, size_swa=size_swa, kvcache=kv, page_size=ps)

        # Alloc 28 tokens (7 pages), leaving 1 page (page 8) free
        prefix = alloc.alloc(28)
        self.assertIsNotNone(prefix)
        last_loc = [int(prefix[-1])]

        # Extend by 4 tokens → must allocate the 8th (last) page
        result = alloc.alloc_extend(
            prefix_lens=[28], seq_lens=[32], last_loc=last_loc, extend_num_tokens=4
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)

        # All returned indices should have valid SWA mappings
        for idx in result:
            self.assertGreater(int(alloc.full_to_swa_index_mapping[idx]), 0)

    # 15c — Regression: alloc_decode on last page must not OOB (GH-231)
    def test_alloc_decode_last_page_no_oob(self):
        """alloc_decode allocating the last page must not raise IndexError."""
        ps = 4
        size = 32
        size_swa = 32
        kv = _make_swa_pool(size=size, size_swa=size_swa, page_size=ps, mesh=self.mesh)
        alloc = SWATokenToKVPoolAllocator(size=size, size_swa=size_swa, kvcache=kv, page_size=ps)

        # Alloc 28 tokens (7 pages), then the last page boundary token
        prefix = alloc.alloc(28)
        self.assertIsNotNone(prefix)
        last_loc = [int(prefix[-1])]

        # Decode: seq_lens=[29] → needs a new page (page 8), the last one
        result = alloc.alloc_decode(seq_lens=[29], last_loc=last_loc)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

        # The new token should have a valid SWA mapping
        self.assertGreater(int(alloc.full_to_swa_index_mapping[result[0]]), 0)

    # 16
    def test_clear_resets_everything(self):
        """clear() fully resets both pools and the mapping."""
        indices = self.alloc.alloc(16)
        self.assertIsNotNone(indices)
        self.alloc.clear()
        self.assertEqual(self.alloc.full_available_size(), self.size)
        self.assertEqual(self.alloc.swa_available_size(), self.size_swa)
        # Mapping should be all zeros
        self.assertTrue(np.all(self.alloc.full_to_swa_index_mapping == 0))


# ---------------------------------------------------------------------------
# Class 3: SWA Eviction logic
# ---------------------------------------------------------------------------
class TestSWAEviction(CustomTestCase):
    """Tests for _evict_swa logic (called via maybe_evict_swa)."""

    def setUp(self):
        self.mesh = _make_mesh()
        self.page_size = 1
        self.sliding_window = 64
        self.pool_size = 256
        self.pool_size_swa = 256
        self.kvcache = _make_swa_pool(
            size=self.pool_size,
            size_swa=self.pool_size_swa,
            page_size=self.page_size,
            mesh=self.mesh,
        )
        self.alloc = SWATokenToKVPoolAllocator(
            size=self.pool_size,
            size_swa=self.pool_size_swa,
            kvcache=self.kvcache,
            page_size=self.page_size,
        )
        self.req_to_token_pool = ReqToTokenPool(size=8, max_context_len=256)

    def _make_req(self, origin_len, output_len):
        """Create a minimal Req-like object for eviction tests."""

        class FakeReq:
            def __init__(self, origin_input_ids, output_ids):
                self.origin_input_ids = origin_input_ids
                self.output_ids = output_ids
                self.swa_evicted_seqlen = 0
                self.req_pool_idx = 0
                self.decode_batch_idx = 0
                self.extend_batch_idx = 0
                self.is_chunked = 0

            @property
            def seqlen(self):
                return len(self.origin_input_ids) + len(self.output_ids)

        return FakeReq(
            origin_input_ids=list(range(origin_len)),
            output_ids=list(range(output_len)),
        )

    def _setup_req_tokens(self, req, n_tokens):
        """Allocate n_tokens and record them in req_to_token_pool."""
        indices = self.alloc.alloc(n_tokens)
        assert indices is not None, f"Failed to allocate {n_tokens} tokens"
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_tokens] = indices
        return indices

    def _evict(self, req, pre_len):
        """Wrapper around the eviction logic matching _evict_swa."""
        new_evicted = max(req.swa_evicted_seqlen, pre_len - self.sliding_window - self.page_size)
        if self.page_size > 1:
            new_evicted = (new_evicted // self.page_size) * self.page_size
        if new_evicted <= req.swa_evicted_seqlen:
            return
        free_slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.swa_evicted_seqlen : new_evicted
        ]
        self.alloc.free_swa(free_slots)
        req.swa_evicted_seqlen = new_evicted

    # 16
    def test_evict_basic(self):
        """100 tokens, window=64, page_size=1 → evicts [0, 35)."""
        req = self._make_req(origin_len=100, output_len=0)
        self._setup_req_tokens(req, 100)
        pre_len = 100  # len(origin) + len(output) - 1 = 100 + 0 - 1, but for extend pre_len=100

        swa_before = self.alloc.swa_available_size()
        self._evict(req, pre_len)
        expected_evicted = pre_len - self.sliding_window - self.page_size  # 35
        self.assertEqual(req.swa_evicted_seqlen, expected_evicted)
        self.assertEqual(self.alloc.swa_available_size(), swa_before + expected_evicted)

    # 17
    def test_evict_idempotent(self):
        """Same pre_len repeated does not double-free."""
        req = self._make_req(origin_len=100, output_len=0)
        self._setup_req_tokens(req, 100)
        self._evict(req, 100)
        swa_after_first = self.alloc.swa_available_size()

        self._evict(req, 100)
        self.assertEqual(self.alloc.swa_available_size(), swa_after_first)

    # 18
    def test_evict_incremental(self):
        """Decode step +1 → evicts exactly 1 slot."""
        req = self._make_req(origin_len=100, output_len=0)
        self._setup_req_tokens(req, 100)

        self._evict(req, 100)
        evicted_1 = req.swa_evicted_seqlen
        swa_1 = self.alloc.swa_available_size()

        # Simulate one decode step
        self._evict(req, 101)
        self.assertEqual(req.swa_evicted_seqlen, evicted_1 + 1)
        self.assertEqual(self.alloc.swa_available_size(), swa_1 + 1)

    # 19
    def test_evict_page_aligned(self):
        """page_size=4: frontier includes the extra safety page and aligns down."""
        page_size = 4
        kvcache = _make_swa_pool(size=256, size_swa=256, page_size=page_size, mesh=self.mesh)
        alloc = SWATokenToKVPoolAllocator(
            size=256, size_swa=256, kvcache=kvcache, page_size=page_size
        )
        req_pool = ReqToTokenPool(size=8, max_context_len=256)

        req = self._make_req(origin_len=100, output_len=0)
        indices = alloc.alloc(100)
        self.assertIsNotNone(indices)
        req_pool.req_to_token[req.req_pool_idx, :100] = indices

        # pre_len=100, window=64, page=4 → raw evicted=32 → page-aligned=32
        new_evicted = max(req.swa_evicted_seqlen, 100 - self.sliding_window - page_size)
        new_evicted = (new_evicted // page_size) * page_size
        self.assertEqual(new_evicted, 32)

        # pre_len=101 → raw=33 → aligned=32 (no change from 32)
        new_evicted2 = max(32, 101 - self.sliding_window - page_size)
        new_evicted2 = (new_evicted2 // page_size) * page_size
        self.assertEqual(new_evicted2, 32)

        # pre_len=104 → raw=36 → aligned=36
        new_evicted3 = max(32, 104 - self.sliding_window - page_size)
        new_evicted3 = (new_evicted3 // page_size) * page_size
        self.assertEqual(new_evicted3, 36)

    # 20
    def test_evict_within_window_noop(self):
        """pre_len <= window → nothing evicted."""
        req = self._make_req(origin_len=50, output_len=0)
        self._setup_req_tokens(req, 50)
        swa_before = self.alloc.swa_available_size()

        self._evict(req, 50)
        self.assertEqual(req.swa_evicted_seqlen, 0)
        self.assertEqual(self.alloc.swa_available_size(), swa_before)

    # 21
    def test_evict_reclaims_swa_capacity(self):
        """Eviction increases swa_available_size."""
        req = self._make_req(origin_len=128, output_len=0)
        self._setup_req_tokens(req, 128)
        swa_before = self.alloc.swa_available_size()

        self._evict(req, 128)
        expected_freed = 128 - self.sliding_window - self.page_size  # 63
        self.assertEqual(self.alloc.swa_available_size(), swa_before + expected_freed)


# ---------------------------------------------------------------------------
# Class 4: Overlap safety
# ---------------------------------------------------------------------------
class TestSWAOverlapSafety(CustomTestCase):
    """Test overlap-aware reclaim timing for decode and chunked extend."""

    def setUp(self):
        self.mesh = _make_mesh()
        self.page_size = 1
        self.sliding_window = 64
        self.chunked_prefill_size = 32
        self.pool_size = 256
        self.pool_size_swa = 256
        self.kvcache = _make_swa_pool(
            size=self.pool_size,
            size_swa=self.pool_size_swa,
            page_size=self.page_size,
            mesh=self.mesh,
        )
        self.alloc = SWATokenToKVPoolAllocator(
            size=self.pool_size,
            size_swa=self.pool_size_swa,
            kvcache=self.kvcache,
            page_size=self.page_size,
        )
        self.req_to_token_pool = ReqToTokenPool(size=8, max_context_len=256)

    def _make_req(self, origin_len, output_len):
        class FakeReq:
            def __init__(self, origin_input_ids, output_ids):
                self.origin_input_ids = origin_input_ids
                self.output_ids = output_ids
                self.swa_evicted_seqlen = 0
                self.req_pool_idx = 0
                self.decode_batch_idx = 0
                self.extend_batch_idx = 0
                self.is_chunked = 0
                # `prefix_indices` is read by maybe_evict_swa (extend path) to
                # compute pre_len. Default to empty; tests set it explicitly
                # when exercising the extend code path.
                self.prefix_indices = np.empty(0, dtype=np.int32)

            @property
            def seqlen(self):
                return len(self.origin_input_ids) + len(self.output_ids)

        return FakeReq(
            origin_input_ids=list(range(origin_len)),
            output_ids=list(range(output_len)),
        )

    def _setup_req_tokens(self, req, n_tokens):
        indices = self.alloc.alloc(n_tokens)
        assert indices is not None, f"Failed to allocate {n_tokens} tokens"
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_tokens] = indices
        # Mirror what cache_unfinished_req does in the real flow: prefix_indices
        # is the row data for the cached portion (used by maybe_evict_swa).
        req.prefix_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_tokens].copy()
        return indices

    def _make_batch(self, req, *, enable_overlap, forward_mode, prefix_lens=None, chunked_req=None):
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
        from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache

        tree_cache = (
            ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.alloc,
                page_size=self.page_size,
            )
            if forward_mode.is_extend()
            else None
        )

        reqs_info = ScheduleReqsInfo(
            reqs=[req],
            chunked_req=chunked_req,
            prefix_lens=prefix_lens,
        )
        return ScheduleBatch(
            reqs_info=[reqs_info],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.alloc,
            tree_cache=tree_cache,
            is_hybrid=True,
            model_config=SimpleNamespace(sliding_window=self.sliding_window),
            forward_mode=forward_mode,
            enable_overlap=enable_overlap,
            dp_size=1,
        )

    def test_overlap_decode_first_batch_skips_reclaim(self):
        """Overlap decode should skip reclaim while the previous batch may still read SWA pages."""
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        req = self._make_req(origin_len=100, output_len=1)
        self._setup_req_tokens(req, req.seqlen)
        req.decode_batch_idx = 0
        batch = self._make_batch(
            req,
            enable_overlap=True,
            forward_mode=ForwardMode.DECODE,
        )

        swa_before = self.alloc.swa_available_size()
        batch.maybe_evict_swa()

        self.assertEqual(req.swa_evicted_seqlen, 0)
        self.assertEqual(self.alloc.swa_available_size(), swa_before)

    def test_overlap_decode_second_batch_reclaims(self):
        """Overlap decode should reclaim once the request reaches the safe reclaim point."""
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        req = self._make_req(origin_len=100, output_len=1)
        self._setup_req_tokens(req, req.seqlen)
        req.decode_batch_idx = 1
        batch = self._make_batch(
            req,
            enable_overlap=True,
            forward_mode=ForwardMode.DECODE,
        )

        swa_before = self.alloc.swa_available_size()
        batch.maybe_evict_swa()

        expected = req.seqlen - 1 - self.sliding_window - self.page_size
        self.assertEqual(req.swa_evicted_seqlen, expected)
        self.assertEqual(self.alloc.swa_available_size(), swa_before + expected)

    def test_overlap_chunked_extend_protects_inflight_chunk(self):
        """Overlap chunked extend must never free SWA slots that the previous
        chunk's in-flight forward could still be reading.

        Real flow grows ``req.prefix_indices`` incrementally — each iter's
        ``cache_unfinished_req`` captures the full row up to the cumulative
        chunk boundary. Subtracting ``chunked_prefill_size`` from
        ``pre_len`` shifts the eviction boundary back by exactly one chunk,
        which by construction keeps ``new_evicted`` at most
        ``len_{N-2} - sliding_window`` — the lowest position the in-flight
        chunk N-1 forward could still be reading.

        We assert two things:
          1. ``swa_evicted_seqlen`` never exceeds the in-flight safe bound.
          2. Eviction does kick in once cumulative prefix grows past
             ``chunked_prefill_size + sliding_window``, so the test isn't
             vacuously satisfied by always returning 0.
        """
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

        # Pick a total length that spans enough chunks to observe both the
        # "no eviction yet" and "eviction kicks in" phases.
        total_len = 200  # > sliding_window + 2 * chunked_prefill_size
        req = self._make_req(origin_len=total_len, output_len=0)
        self._setup_req_tokens(req, total_len)
        req.is_chunked = 1
        # Snapshot the full row of slot indices so we can produce the per-iter
        # cumulative slices, mirroring what cache_unfinished_req would set.
        full_row = self.req_to_token_pool.req_to_token[req.req_pool_idx, :total_len].copy()

        batch = self._make_batch(
            req,
            enable_overlap=True,
            forward_mode=ForwardMode.EXTEND,
            chunked_req=req,
        )

        old_chunked_prefill_size = global_server_args_dict.get("chunked_prefill_size")
        global_server_args_dict["chunked_prefill_size"] = self.chunked_prefill_size
        try:
            num_chunks = total_len // self.chunked_prefill_size
            saw_eviction = False
            for k in range(num_chunks + 1):
                cumulative = k * self.chunked_prefill_size
                # Mirror cache_unfinished_req: prefix_indices is the row data
                # for the cached portion (len_{k-1} after iter k-1's process).
                req.prefix_indices = full_row[:cumulative].copy()
                req.extend_batch_idx = k

                batch.maybe_evict_swa()

                # In-flight chunk N-1 reads SWA from
                # [len_{N-2} - sliding_window, len_{N-1}). The lowest position
                # we must NOT have evicted is len_{N-2} - sliding_window.
                # len_{N-2} = max(0, cumulative - chunked_prefill_size).
                in_flight_low = max(
                    0,
                    cumulative - self.chunked_prefill_size - self.sliding_window - self.page_size,
                )
                self.assertLessEqual(
                    req.swa_evicted_seqlen,
                    in_flight_low,
                    f"iter {k}: evicted={req.swa_evicted_seqlen} would corrupt "
                    f"in-flight chunk reading from position {in_flight_low}",
                )
                if req.swa_evicted_seqlen > 0:
                    saw_eviction = True

            # Sanity: with total_len=200 the loop must reach iters where
            # eviction kicks in, otherwise the test would be vacuous.
            self.assertTrue(
                saw_eviction,
                "Test never observed any SWA eviction; widen the loop range " "or weaken the setup",
            )
        finally:
            global_server_args_dict["chunked_prefill_size"] = old_chunked_prefill_size


if __name__ == "__main__":
    unittest.main()
