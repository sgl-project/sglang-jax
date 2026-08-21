import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sgl_jax.srt.managers.schedule_batch import (
    swa_eviction_interval,
    swa_eviction_peak_tokens,
)
from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator


class _DummySWAAllocator(SWATokenToKVPoolAllocator):
    def __init__(self, full_available: int = 4096, swa_available: int = 4096):
        self._full_available = full_available
        self._swa_available = swa_available

    def full_available_size(self, dp_rank: int = 0):
        return self._full_available

    def swa_available_size(self, dp_rank: int = 0):
        return self._swa_available


class _DummySWATreeCache:
    def __init__(self, sliding_window_size: int):
        self.sliding_window_size = sliding_window_size

    def full_evictable_size(self, dp_rank: int = 0):
        return 0

    def swa_evictable_size(self, dp_rank: int = 0):
        return 0


def _running_req(
    kv_allocated_len: int,
    swa_evicted_seqlen: int,
    *,
    max_new_tokens: int = 1024,
    output_len: int = 0,
    ignore_eos: bool = False,
):
    return SimpleNamespace(
        kv_allocated_len=kv_allocated_len,
        swa_evicted_seqlen=swa_evicted_seqlen,
        sampling_params=SimpleNamespace(
            max_new_tokens=max_new_tokens,
            ignore_eos=ignore_eos,
        ),
        output_ids=[0] * output_len,
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

    def test_running_ignore_eos_request_reserves_full_remaining_output(self):
        running_batch = SimpleNamespace(
            reqs_info=[
                SimpleNamespace(
                    reqs=[
                        _running_req(0, 0, max_new_tokens=100, output_len=20, ignore_eos=True),
                        _running_req(0, 0, max_new_tokens=100, output_len=20),
                    ]
                )
            ]
        )

        adder = PrefillAdder(
            page_size=1,
            tree_cache=object(),
            token_to_kv_pool_allocator=object(),
            running_batch=running_batch,
            new_token_ratio=0.25,
            rem_input_tokens=4096,
            rem_chunk_tokens=None,
            dp_size=1,
        )

        # ignore_eos reserves all 80 remaining tokens; the regular request uses 25%.
        self.assertEqual(adder.rem_total_token_offset, [100])

    def test_chunk_budget_is_sampled_before_current_chunk_is_deducted(self):
        adder = PrefillAdder(
            page_size=16,
            tree_cache=_DummySWATreeCache(sliding_window_size=1),
            token_to_kv_pool_allocator=_DummySWAAllocator(),
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=4096,
            rem_chunk_tokens=128,
            dp_size=1,
        )

        adder._update_prefill_budget(0, 80, 0, dp_rank=0)

        self.assertEqual(adder.rem_swa_token_offset, [80])
        self.assertEqual(adder.rem_chunk_tokens_list, [48])

    def test_hybrid_ignore_eos_request_checks_full_pool_safety_margin(self):
        adder = PrefillAdder(
            page_size=1,
            tree_cache=_DummySWATreeCache(sliding_window_size=1),
            token_to_kv_pool_allocator=_DummySWAAllocator(full_available=64),
            running_batch=None,
            new_token_ratio=0.5,
            rem_input_tokens=4096,
            rem_chunk_tokens=None,
            dp_size=1,
        )
        req = SimpleNamespace(
            dp_rank=0,
            extend_input_len=1,
            origin_input_ids=[1],
            output_ids=[],
            sampling_params=SimpleNamespace(max_new_tokens=128, ignore_eos=True),
        )

        self.assertEqual(adder.add_one_req_ignore_eos(req), AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list[0], [])


if __name__ == "__main__":
    unittest.main()
