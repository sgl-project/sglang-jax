"""DP (dp_size>1) end-to-end tests for the HiCache Stage 3 control plane.

The device KV buffer is DP-sharded along its leading page axis, but the
allocator hands out per-rank LOCAL page indices ``[1, pages_per_rank]``. The
forward path indexes those locally inside a ``shard_map``; HiCache's
``stage_backup`` / ``copy_to_device`` instead gather/scatter the GLOBAL buffer,
so the tree cache globalizes the indices at the controller boundary
(``UnifiedRadixCache._to_global_device_pages``). These tests exercise that
conversion under a *real* 2-way ``data`` mesh and assert per-rank isolation: two
ranks backing up identical token ids with different KV must each load back their
own data, never the other rank's physical page. The ``Page2`` subclass repeats
the suite at ``page_size=2`` so the global-page math is covered for dp_rank>0 AND
page_size>1 simultaneously.

Multi-device CPU is forced via ``XLA_FLAGS`` at import time, so this module must
be the first to initialise the JAX backend. Run it standalone:

    XLA_FLAGS=--xla_force_host_platform_device_count=2 \\
        PYTHONPATH=python python -m pytest \\
        python/sgl_jax/test/mem_cache/test_hicache_dp.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sgl_jax.srt.mem_cache.hicache_controller import HiCacheController
from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_DP = 2


def _key(token_ids: list[int], dp_rank: int) -> RadixKey:
    return RadixKey(token_ids=token_ids, extra_key=None, dp_rank=dp_rank)


class HiCacheDPTest(unittest.TestCase):
    HEAD_NUM = 4
    HEAD_DIM = 8
    LAYER_NUM = 2
    DEVICE_SIZE = 64  # 32 tokens per rank
    HOST_PAGES = 64
    PAGE_SIZE = 1
    DTYPE = jnp.float32

    def setUp(self):
        if len(jax.devices()) < _DP:
            self.skipTest(
                f"need >= {_DP} devices for DP test; run standalone with "
                "XLA_FLAGS=--xla_force_host_platform_device_count=2"
            )
        self.mesh = create_device_mesh(ici_parallelism=[_DP, 1], dcn_parallelism=[1, 1])
        jax.sharding.set_mesh(self.mesh)

        self.kv_cache = MHATokenToKVPool(
            size=self.DEVICE_SIZE,
            page_size=self.PAGE_SIZE,
            dtype=self.DTYPE,
            head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            mesh=self.mesh,
            dp_size=_DP,
        )
        self.allocator = (
            TokenToKVPoolAllocator(size=self.DEVICE_SIZE, kvcache=self.kv_cache, dp_size=_DP)
            if self.PAGE_SIZE == 1
            else PagedTokenToKVPoolAllocator(
                size=self.DEVICE_SIZE,
                page_size=self.PAGE_SIZE,
                kvcache=self.kv_cache,
                dp_size=_DP,
            )
        )
        self.req_pool = ReqToTokenPool(size=64, max_context_len=512, dtype=np.int32)
        self.cache = UnifiedRadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            page_size=self.PAGE_SIZE,
            kv_head_num=self.HEAD_NUM,
            head_dim=self.HEAD_DIM,
            layer_num=self.LAYER_NUM,
            max_seq_len=512,
            dtype=self.DTYPE,
        )
        self.host_pool = LRUHostKVPool(
            device_pool=self.kv_cache,
            pool_size=self.HOST_PAGES,
            page_size=self.kv_cache.page_size,
            layer_num=self.kv_cache.layer_num,
            per_layer_shape=tuple(int(d) for d in self.kv_cache.kv_buffer[0].shape[1:]),
            dtype=self.kv_cache.dtype,
            mesh=self.mesh,
            partition_spec=self.kv_cache.kv_sharding.spec,
        )
        self.controller = HiCacheController(self.host_pool, self.kv_cache)
        # Mirror kv_cache_builder.init_hicache wiring.
        self.cache.host_pool = self.host_pool
        self.cache.hicache_controller = self.controller
        self.cache.hicache_enabled = True
        self.cache.write_through_threshold = 1
        for component in self.cache._components_tuple:
            component._full_kv_pool_host = self.host_pool

    def tearDown(self):
        self.controller.shutdown()

    # ---- helpers ----

    @property
    def _pages_per_shard(self) -> int:
        return self.kv_cache.kv_buffer[0].shape[0] // _DP

    def _global_page(self, local_token: int, dp_rank: int) -> int:
        """LOCAL token index -> GLOBAL physical page id (mirrors the controller's
        ``_to_global_device_pages``: local page + dp_rank * pages_per_shard)."""
        return int(local_token) // self.PAGE_SIZE + dp_rank * self._pages_per_shard

    @property
    def _gather_sharding(self):
        # A single gathered page: drop the leading (data/DP-sharded) page axis.
        from jax.sharding import NamedSharding, PartitionSpec

        spec = self.kv_cache.kv_sharding.spec
        return NamedSharding(self.mesh, PartitionSpec(*tuple(spec)[1:]))

    def _read_token(self, layer: int, local_token: int, dp_rank: int) -> np.ndarray:
        # Plain ``buf[gp]`` cannot resolve the gather out-sharding on a
        # multi-device mesh; the explicit form is required (same as the pool).
        gp = self._global_page(local_token, dp_rank)
        off = int(local_token) % self.PAGE_SIZE
        page = self.kv_cache.kv_buffer[layer].at[gp].get(out_sharding=self._gather_sharding)
        return np.asarray(page)[off]

    def _fill(self, dp_rank: int, n: int, seed: int) -> tuple[np.ndarray, list]:
        """Alloc n LOCAL slots on ``dp_rank`` and fill the corresponding GLOBAL
        physical (page, offset) with random KV (mirrors the forward shard_map
        write). Returns (local_indices, per-(slot,layer) original KV)."""
        local = self.allocator.alloc(n, dp_rank=dp_rank)
        self.assertIsNotNone(local)
        token_shape = tuple(self.kv_cache.kv_buffer[0].shape[2:])
        orig = []
        for i, lidx in enumerate(local):
            gp = self._global_page(lidx, dp_rank)
            off = int(lidx) % self.PAGE_SIZE
            per_layer = []
            for layer in range(self.LAYER_NUM):
                buf = self.kv_cache.kv_buffer[layer]
                vals = jax.random.normal(
                    jax.random.PRNGKey(seed * 10000 + dp_rank * 1000 + i * 10 + layer),
                    token_shape,
                    buf.dtype,
                )
                self.kv_cache.kv_buffer[layer] = buf.at[gp, off].set(
                    vals, out_sharding=buf.sharding
                )
                per_layer.append(self._read_token(layer, lidx, dp_rank))
            orig.append(per_layer)
        return local, orig

    def _settle(self, timeout: float = 5.0) -> None:
        self.controller.drain_pending()
        deadline = time.time() + timeout
        while self.cache.ongoing_write and time.time() < deadline:
            self.cache.check_hicache_events()
            time.sleep(0.005)
        self.assertFalse(self.cache.ongoing_write, "D2H writes did not settle")

    def _node_for(self, tokens: list[int], dp_rank: int):
        mr = self.cache.match_prefix(MatchPrefixParams(key=_key(tokens, dp_rank)))
        return mr

    # ---- tests ----

    def test_dp_roundtrip_rank_isolation(self):
        tokens = [10, 11, 12, 13]  # identical token ids on BOTH ranks
        local0, orig0 = self._fill(0, len(tokens), seed=1)
        local1, orig1 = self._fill(1, len(tokens), seed=2)  # different KV

        # Sanity: rank0/rank1 chose overlapping LOCAL indices but they live in
        # different GLOBAL shards, so the fixtures must not collide.
        self.assertEqual(list(local0), list(local1))
        self.assertNotEqual(self._global_page(local0[0], 0), self._global_page(local1[0], 1))

        for dp_rank, local in ((0, local0), (1, local1)):
            key = _key(tokens, dp_rank)
            self.cache.insert(InsertParams(key=key, value=local))
            self.cache.insert(InsertParams(key=key, value=local))  # trigger backup
        # Two distinct root subtrees, one per dp_rank.
        self.assertEqual(len(self.cache.root_node.children), 2)
        self._settle()

        # Demote both ranks' device KV to host.
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=0))
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=1))

        # Load back each rank and verify bit-exact against ITS OWN fixture.
        for dp_rank, orig in ((0, orig0), (1, orig1)):
            mr = self._node_for(tokens, dp_rank)
            self.assertEqual(len(mr.device_indices), 0)
            self.assertEqual(mr.host_hit_length, len(tokens))
            new_local, _ = self.cache.init_load_back(
                mr.last_host_node, mr.host_hit_length, mem_quota=self.DEVICE_SIZE
            )
            self.assertEqual(len(new_local), len(tokens))
            for i, lidx in enumerate(new_local):
                for layer in range(self.LAYER_NUM):
                    got = self._read_token(layer, lidx, dp_rank)
                    np.testing.assert_allclose(got, orig[i][layer])

    def test_dp_rank1_only(self):
        """A backup/evict/load cycle confined to dp_rank=1 (the non-zero rank,
        where the local->global offset is actually exercised)."""
        tokens = [20, 21, 22, 23]
        local, orig = self._fill(1, len(tokens), seed=7)
        key = _key(tokens, 1)
        self.cache.insert(InsertParams(key=key, value=local))
        self.cache.insert(InsertParams(key=key, value=local))
        self._settle()
        self.cache.evict(EvictParams(num_tokens=len(tokens), dp_rank=1))

        mr = self._node_for(tokens, 1)
        self.assertEqual(mr.host_hit_length, len(tokens))
        new_local, _ = self.cache.init_load_back(
            mr.last_host_node, mr.host_hit_length, mem_quota=self.DEVICE_SIZE
        )
        for i, lidx in enumerate(new_local):
            # Really in shard 1: global page sits past the rank boundary.
            self.assertGreaterEqual(self._global_page(lidx, 1), self._pages_per_shard)
            for layer in range(self.LAYER_NUM):
                got = self._read_token(layer, lidx, 1)
                np.testing.assert_allclose(got, orig[i][layer])


class HiCacheDPPage2Test(HiCacheDPTest):
    PAGE_SIZE = 2


if __name__ == "__main__":
    unittest.main()
