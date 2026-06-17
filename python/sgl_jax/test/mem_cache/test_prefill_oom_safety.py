# Conservative-admission + graceful-abort guards against the sustained-retraction
# "Prefill out of memory" scheduler crash.
#
# Run on CPU:
#   cd python && USE_DEVICE_TYPE=cpu python -m pytest \
#       sgl_jax/test/mem_cache/test_prefill_oom_safety.py -v

import os

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_policy import (
    CLIP_MAX_NEW_TOKENS_ESTIMATION,
    AddReqResult,
    PrefillAdder,
)
from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.chunk_cache import ChunkCache
from sgl_jax.srt.mem_cache.common import alloc_paged_token_slots_extend
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


def _mock_req(extend_input_len, max_new_tokens=8, dp_rank=0):
    """Lightweight stand-in exposing only the attributes PrefillAdder touches."""
    return SimpleNamespace(
        rid=f"r{id(object()) & 0xFFFF}",
        extend_input_len=extend_input_len,
        host_hit_length=0,
        prefix_indices=[],
        fill_ids=list(range(extend_input_len)),
        last_node=None,
        swa_uuid_for_lock=None,
        last_matched_prefix_len=0,
        dp_rank=dp_rank,
        sampling_params=SimpleNamespace(
            max_new_tokens=max_new_tokens, ignore_eos=False
        ),
    )


class TestPrefillOOMSafety(CustomTestCase):
    def setUp(self):
        self.page_size = 128
        self.pool_size = self.page_size * 8  # 1024 tokens, 8 pages
        # head_num must be divisible by the tensor-parallel device count (8 on
        # the simulated CPU mesh); the buffer contents are irrelevant here, only
        # the allocator's page bookkeeping is exercised.
        self.kv_head_num = 8
        self.head_dim = 16
        self.layer_num = 1
        self.dtype = jnp.bfloat16

    def _build(self, page_size=None, pool_size=None):
        page_size = page_size or self.page_size
        pool_size = pool_size or self.pool_size
        req_pool = ReqToTokenPool(size=16, max_context_len=pool_size, dtype=np.int32)
        kv_cache = MHATokenToKVPool(
            size=pool_size,
            page_size=page_size,
            dtype=self.dtype,
            head_num=self.kv_head_num,
            head_dim=self.head_dim,
            layer_num=self.layer_num,
            mesh=mesh,
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=pool_size, page_size=page_size, kvcache=kv_cache
        )
        tree = ChunkCache(req_pool, allocator, page_size)
        return req_pool, allocator, tree

    def _make_adder(self, allocator, tree, new_token_ratio=1.0):
        return PrefillAdder(
            page_size=self.page_size,
            tree_cache=tree,
            token_to_kv_pool_allocator=allocator,
            running_batch=None,
            new_token_ratio=new_token_ratio,
            rem_input_tokens=10**9,
            rem_chunk_tokens=None,  # non-chunked path
            mixed_with_decode_tokens=0,
            dp_size=1,
        )

    # ---- Primary fix: page-overhead reservation in the budget --------------

    def test_update_prefill_budget_reserves_page_overhead(self):
        """_update_prefill_budget must deduct one extra page_size per request,
        mirroring alloc_paged_token_slots_extend (extend + len*page_size)."""
        _, allocator, tree = self._build()
        adder = self._make_adder(allocator, tree)

        before_total = adder.rem_total_token_offset[0]
        before_cur = adder.cur_rem_token_offset[0]

        extend = self.page_size  # exactly one page of input
        max_new = 8
        adder._update_prefill_budget(0, extend, max_new, dp_rank=0)

        # extend_input_len is page-ceiled (already page-aligned here) then a full
        # page_size of overhead is added on top of extend (+ max_new for total).
        self.assertEqual(
            adder.rem_total_token_offset[0] - before_total,
            extend + max_new + self.page_size,
        )
        self.assertEqual(
            adder.cur_rem_token_offset[0] - before_cur,
            extend + self.page_size,
        )

    def test_page_size_one_has_no_overhead(self):
        """page_size==1 has no page-straddle, so no overhead is reserved."""
        _, allocator, tree = self._build(page_size=1, pool_size=1024)
        adder = PrefillAdder(
            page_size=1,
            tree_cache=tree,
            token_to_kv_pool_allocator=allocator,
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=10**9,
            rem_chunk_tokens=None,
            mixed_with_decode_tokens=0,
            dp_size=1,
        )
        before = adder.cur_rem_token_offset[0]
        adder._update_prefill_budget(0, 10, 0, dp_rank=0)
        self.assertEqual(adder.cur_rem_token_offset[0] - before, 10)

    def test_admission_rejects_when_only_page_overhead_short(self):
        """A request that fits by raw tokens but NOT once the per-request page
        overhead is reserved must be rejected (NO_TOKEN) rather than admitted
        and later OOM-ing in alloc_extend. This is the over-admission the old
        budget (no +page_size) allowed near full pool."""
        _, allocator, tree = self._build()
        adder = self._make_adder(allocator, tree)

        # Occupy all but exactly one page of the pool.
        free_tokens = allocator.available_size(dp_rank=0)
        self.assertEqual(free_tokens, self.pool_size)
        occupy = self.pool_size - self.page_size  # leave 1 page free
        allocator.alloc(occupy, dp_rank=0)
        self.assertEqual(allocator.available_size(dp_rank=0), self.page_size)

        # Request needs exactly one page of input + small decode. Raw need
        # (extend + max_new) <= one page would have been admitted by the old
        # code; with +page_size overhead total exceeds the single free page.
        req = _mock_req(extend_input_len=self.page_size - 8, max_new_tokens=4)
        res = adder.add_one_req(req)
        self.assertEqual(
            res,
            AddReqResult.NO_TOKEN,
            "over-admitted a prefill that cannot fit the actual paged allocation",
        )

    def test_admission_accepts_when_room_for_overhead(self):
        """Sanity: with enough free pages the same request is admitted."""
        _, allocator, tree = self._build()
        adder = self._make_adder(allocator, tree)
        occupy = self.pool_size - 3 * self.page_size  # leave 3 pages free
        allocator.alloc(occupy, dp_rank=0)

        req = _mock_req(extend_input_len=self.page_size - 8, max_new_tokens=4)
        res = adder.add_one_req(req)
        self.assertEqual(res, AddReqResult.CONTINUE)

    # ---- Safety net: alloc returns None instead of raising -----------------

    def test_alloc_extend_returns_none_when_raise_disabled(self):
        """alloc_paged_token_slots_extend(raise_on_oom=False) must return None
        on shortage instead of raising the RuntimeError that kills the
        scheduler. raise_on_oom defaults True to keep callers fail-loud."""
        _, allocator, tree = self._build()
        # Exhaust the pool.
        allocator.alloc(self.pool_size, dp_rank=0)
        self.assertEqual(allocator.available_size(dp_rank=0), 0)

        # Try to extend a fresh request by one full page -> cannot fit.
        prefix_lens = [0]
        seq_lens = [self.page_size]
        last_loc = [-1]  # no prior page
        extend_num_tokens = self.page_size

        out = alloc_paged_token_slots_extend(
            tree,
            prefix_lens,
            seq_lens,
            last_loc,
            extend_num_tokens,
            dp_rank=0,
            raise_on_oom=False,
        )
        self.assertIsNone(out)

    def test_alloc_extend_still_raises_by_default(self):
        """Default behavior unchanged: raises RuntimeError on OOM."""
        _, allocator, tree = self._build()
        allocator.alloc(self.pool_size, dp_rank=0)
        with self.assertRaises(RuntimeError):
            alloc_paged_token_slots_extend(
                tree,
                [0],
                [self.page_size],
                [-1],
                self.page_size,
                dp_rank=0,
            )


if __name__ == "__main__":
    unittest.main()
