# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_swa_radix_cache.py -q

import os

# Simulate multi-device on CPU to satisfy JAX Mesh creation
if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import unittest

import jax
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache


class TestSWARadixCache(unittest.TestCase):
    def setUp(self):
        # Keep KV sizes small to make tests light-weight
        self.devices = jax.devices()
        self.mesh = Mesh([self.devices[0]], axis_names=("tensor",))

        # Small buffers to avoid heavy allocations
        self.kv_head_num = 1
        self.head_dim = 1
        self.layer_num = 2
        self.dtype = jax.numpy.bfloat16

        # Token pool sizes (FULL and SWA)
        self.full_size = 4096
        self.swa_size = 4096

        # Request→token mapping pool (host-side)
        self.req_pool = ReqToTokenPool(size=64, max_context_len=512)

        # Hybrid KV pool (FULL + SWA)
        self.kv_pool = SWAKVPool(
            size=self.full_size,
            size_swa=self.swa_size,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            token_to_kv_pool_class=MHATokenToKVPool,
            page_size=1,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=self.mesh,
        )

        # Allocator over the hybrid KV pool
        self.allocator = SWATokenToKVPoolAllocator(
            size=self.full_size,
            size_swa=self.swa_size,
            kvcache=self.kv_pool,
        )

        # SWA radix tree (page_size=1, small sliding window)
        self.cache = SWARadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=64,
            page_size=1,
            disable=False,
        )

    def _alloc_indices(self, n: int) -> np.ndarray:
        idx = self.allocator.alloc(n)
        self.assertIsNotNone(idx)
        self.assertEqual(len(idx), n)
        return idx

    def test_insert_prefix_overlap_no_share(self):
        """Insert two sequences with overlapping prefix; ensure no FULL-index sharing.

        Simulates scheduler behavior where the second sequence reuses the first sequence's
        FULL indices for the matched prefix, and supplies newly-allocated indices for the suffix.
        SWARadixCache.insert should free the request-side matched segment properly so that
        no index is recorded twice in different nodes.
        """
        # First sequence A: 66 tokens
        A_len = 66
        key_A = list(range(1000, 1000 + A_len))  # token ids are arbitrary here
        val_A = self._alloc_indices(A_len)
        # Insert A as a single leaf
        self.cache.insert(key_A, value=val_A, prev_prefix_len=0)

        # Second sequence B: shares full A as prefix, then extends by 14 tokens
        B_len = 80
        key_B = list(range(1000, 1000 + B_len))  # same 66 prefix + 14 extra

        # Simulate scheduler: for the matched prefix, reuse tree indices (val_A).
        # For the suffix, allocate fresh indices from allocator.
        suffix = self._alloc_indices(B_len - A_len)
        val_B = np.concatenate([val_A, suffix])

        # Insert B with prev_prefix_len=A_len to model reuse of the matched prefix
        self.cache.insert(key_B, value=val_B, prev_prefix_len=A_len)

        # Verify: no shared FULL indices across the tree
        res = self.cache.detect_first_full_index_share()
        self.assertFalse(res.get("found", False), f"Unexpected sharing detected: {res}")

    def test_no_share_after_small_swa_evict_and_reinsert(self):
        """Evict a small amount of SWA tokens and reinsert; no FULL-index sharing should appear.

        This approximates internal-node SWA-only tombstoning followed by reinsertion using
        request-side values.
        """
        # Build two related sequences to create a small tree
        key1 = list(range(2000, 2000 + 64))
        val1 = self._alloc_indices(len(key1))
        self.cache.insert(key1, value=val1, prev_prefix_len=0)

        key2 = list(range(2000, 2000 + 96))  # extends key1 by 32
        val2 = np.concatenate([val1, self._alloc_indices(len(key2) - len(key1))])
        self.cache.insert(key2, value=val2, prev_prefix_len=len(key1))

        # Trigger a small SWA eviction (should tombstone some path segments)
        # Keep it small to avoid blowing away the entire tree; values here are heuristic.
        self.cache.evict(full_num_tokens=0, swa_num_tokens=32)

        # Reinsert the longer key with request-side values again
        val2_re = np.concatenate([val1, self._alloc_indices(len(key2) - len(key1))])
        self.cache.insert(key2, value=val2_re, prev_prefix_len=len(key1))

        # Verify no sharing exists
        res = self.cache.detect_first_full_index_share()
        self.assertFalse(res.get("found", False), f"Unexpected sharing after evict/reinsert: {res}")


if __name__ == "__main__":
    unittest.main()
