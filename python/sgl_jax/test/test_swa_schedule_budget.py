import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sgl_jax.srt.managers.schedule_batch import (
    swa_eviction_interval,
    swa_eviction_peak_tokens,
)
from sgl_jax.srt.managers.schedule_policy import PrefillAdder
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator


class _DummySWAAllocator(SWATokenToKVPoolAllocator):
    def __init__(self):
        pass


class _DummySWATreeCache:
    def __init__(self, sliding_window_size: int):
        self.sliding_window_size = sliding_window_size


def _running_req(kv_allocated_len: int, swa_evicted_seqlen: int):
    return SimpleNamespace(
        kv_allocated_len=kv_allocated_len,
        swa_evicted_seqlen=swa_evicted_seqlen,
        sampling_params=SimpleNamespace(max_new_tokens=1024, ignore_eos=False),
        output_ids=[],
    )


class TestSWAScheduleBudget(unittest.TestCase):
    def test_eviction_peak_includes_interval_and_page_margins(self):
        with patch.dict(
            os.environ,
            {"SGL_JAX_SWA_EVICTION_INTERVAL_MULTIPLIER": "0.5"},
        ):
            self.assertEqual(swa_eviction_interval(100, 16), 48)
            self.assertEqual(swa_eviction_peak_tokens(100, 16), 180)

    def test_running_requests_reserve_remaining_decode_headroom_per_dp(self):
        running_batch = SimpleNamespace(
            reqs_info=[
                SimpleNamespace(
                    reqs=[
                        _running_req(kv_allocated_len=160, swa_evicted_seqlen=32),
                        _running_req(kv_allocated_len=300, swa_evicted_seqlen=32),
                    ]
                ),
                SimpleNamespace(reqs=[_running_req(kv_allocated_len=95, swa_evicted_seqlen=0)]),
            ]
        )

        with patch.dict(
            os.environ,
            {"SGL_JAX_SWA_EVICTION_INTERVAL_MULTIPLIER": "1.0"},
        ):
            adder = PrefillAdder(
                page_size=16,
                tree_cache=_DummySWATreeCache(sliding_window_size=100),
                token_to_kv_pool_allocator=_DummySWAAllocator(),
                running_batch=running_batch,
                new_token_ratio=1.0,
                rem_input_tokens=4096,
                rem_chunk_tokens=64,
                dp_size=2,
            )

            self.assertEqual(adder._swa_budget_for_req(4096, dp_rank=0), 240)

        # Peak occupancy is ceil_page(100 + 96 + 2 * 16) = 240.
        # DP0 has 128 live tokens in one request and >=240 in the other;
        # DP1 has 96 live tokens.
        self.assertEqual(adder.rem_swa_token_offset, [112, 144])


if __name__ == "__main__":
    unittest.main()
