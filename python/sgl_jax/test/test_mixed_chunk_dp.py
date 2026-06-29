import unittest
from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sgl_jax.srt.mem_cache.base_prefix_cache import IncLockRefResult
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class _DummyAllocator:
    def available_size(self, dp_rank: int = 0):
        return 1_000_000


class _DummyTreeCache:
    def evictable_size(self, dp_rank: int = 0):
        return 0


class _DummyRadixCache:
    """Tree cache with disable=False so add_one_req takes the radix path."""

    disable = False

    def recurrent_extra_buffer_active(self) -> bool:
        return False

    def evictable_size(self, dp_rank: int = 0):
        return 0

    def inc_lock_ref(self, node):
        return IncLockRefResult(delta=0)

    def dec_lock_ref(self, node, params=None):
        pass


def _make_req(rid: str, dp_rank: int, input_len: int = 4, output_len: int = 2) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(range(input_len)),
        sampling_params=SamplingParams(max_new_tokens=8),
        dp_rank=dp_rank,
        eos_token_ids={2},
        vocab_size=32000,
    )
    req.output_ids = list(range(output_len))
    return req


def _make_extend_batch(
    reqs_per_dp: list[list[Req]],
    input_ids_per_dp: list[np.ndarray],
    out_cache_loc_per_dp: list[np.ndarray],
    req_pool_indices_per_dp: list[np.ndarray],
    seq_lens_per_dp: list[np.ndarray],
) -> ScheduleBatch:
    batch = ScheduleBatch.init_new(
        reqs=reqs_per_dp,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        model_config=SimpleNamespace(vocab_size=32000),
        enable_overlap=False,
        dp_size=len(reqs_per_dp),
        spec_algorithm=None,
        mesh=None,
    )

    for dp_rank, info in enumerate(batch.reqs_info):
        info.input_ids = input_ids_per_dp[dp_rank]
        info.out_cache_loc = out_cache_loc_per_dp[dp_rank]
        info.req_pool_indices = req_pool_indices_per_dp[dp_rank]
        info.seq_lens = seq_lens_per_dp[dp_rank]
        info.prefix_lens = [len(r.origin_input_ids) - 1 for r in info.reqs]
        info.extend_lens = [1 for _ in info.reqs]
        info.extend_num_tokens = len(info.reqs)
        info.extend_logprob_start_lens = [0 for _ in info.reqs]
        info.seq_lens_sum = int(np.sum(info.seq_lens)) if info.seq_lens is not None else 0

    return batch


class TestMixedChunkDP(unittest.TestCase):
    def test_mix_with_running_merges_per_dp_fields(self):
        # DP0: one prefill req + one running decode req
        prefill_req_dp0 = _make_req("prefill-dp0", dp_rank=0)
        running_req_dp0 = _make_req("running-dp0", dp_rank=0)
        # DP1: one prefill req + one running decode req
        prefill_req_dp1 = _make_req("prefill-dp1", dp_rank=1)
        running_req_dp1 = _make_req("running-dp1", dp_rank=1)

        new_batch = _make_extend_batch(
            reqs_per_dp=[[prefill_req_dp0], [prefill_req_dp1]],
            input_ids_per_dp=[
                np.array([10], dtype=np.int32),
                np.array([20], dtype=np.int32),
            ],
            out_cache_loc_per_dp=[
                np.array([100], dtype=np.int32),
                np.array([200], dtype=np.int32),
            ],
            req_pool_indices_per_dp=[
                np.array([1], dtype=np.int32),
                np.array([2], dtype=np.int32),
            ],
            seq_lens_per_dp=[
                np.array([len(prefill_req_dp0.origin_input_ids) + len(prefill_req_dp0.output_ids)]),
                np.array([len(prefill_req_dp1.origin_input_ids) + len(prefill_req_dp1.output_ids)]),
            ],
        )

        running_batch = _make_extend_batch(
            reqs_per_dp=[[running_req_dp0], [running_req_dp1]],
            input_ids_per_dp=[
                np.array([11], dtype=np.int32),
                np.array([21], dtype=np.int32),
            ],
            out_cache_loc_per_dp=[
                np.array([101], dtype=np.int32),
                np.array([201], dtype=np.int32),
            ],
            req_pool_indices_per_dp=[
                np.array([3], dtype=np.int32),
                np.array([4], dtype=np.int32),
            ],
            seq_lens_per_dp=[
                np.array([len(running_req_dp0.origin_input_ids) + len(running_req_dp0.output_ids)]),
                np.array([len(running_req_dp1.origin_input_ids) + len(running_req_dp1.output_ids)]),
            ],
        )

        new_batch.mix_with_running(running_batch)

        # DP0 checks
        info0 = new_batch.reqs_info[0]
        self.assertEqual([r.rid for r in info0.reqs], ["prefill-dp0", "running-dp0"])
        np.testing.assert_array_equal(info0.input_ids, np.array([10, 11], dtype=np.int32))
        np.testing.assert_array_equal(info0.out_cache_loc, np.array([100, 101], dtype=np.int32))
        self.assertEqual(info0.extend_lens, [1, 1])
        self.assertEqual(info0.extend_num_tokens, 2)
        self.assertEqual(info0.extend_logprob_start_lens, [0, 0])

        # DP1 checks
        info1 = new_batch.reqs_info[1]
        self.assertEqual([r.rid for r in info1.reqs], ["prefill-dp1", "running-dp1"])
        np.testing.assert_array_equal(info1.input_ids, np.array([20, 21], dtype=np.int32))
        np.testing.assert_array_equal(info1.out_cache_loc, np.array([200, 201], dtype=np.int32))
        self.assertEqual(info1.extend_lens, [1, 1])
        self.assertEqual(info1.extend_num_tokens, 2)
        self.assertEqual(info1.extend_logprob_start_lens, [0, 0])

        # Running requests should be converted to 1-token extend for mixed chunk.
        self.assertEqual(running_req_dp0.extend_input_len, 1)
        self.assertEqual(running_req_dp1.extend_input_len, 1)
        self.assertEqual(
            running_req_dp0.fill_ids,
            running_req_dp0.origin_input_ids + running_req_dp0.output_ids,
        )
        self.assertEqual(
            running_req_dp1.fill_ids,
            running_req_dp1.origin_input_ids + running_req_dp1.output_ids,
        )

    def test_prefill_adder_accepts_per_dp_mixed_decode_tokens(self):
        adder = PrefillAdder(
            page_size=1,
            tree_cache=_DummyTreeCache(),
            token_to_kv_pool_allocator=_DummyAllocator(),
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=100,
            rem_chunk_tokens=20,
            mixed_with_decode_tokens=[3, 1],
            dp_size=2,
        )

        self.assertEqual(adder.rem_total_token_offset, [3, 1])
        self.assertEqual(adder.cur_rem_token_offset, [3, 1])
        self.assertEqual(adder.rem_input_tokens, 96)
        self.assertEqual(adder.rem_chunk_tokens_list, [17, 19])

    def test_add_one_req_chunked_admits_all_dp_ranks_with_radix(self):
        # Regression for #1239: with radix enabled (tree_cache.disable=False) +
        # chunked prefill + dp>1 + extend_input_len >= max_prefill_tokens, the
        # untruncated rem_input_tokens gate would return OTHER on the second
        # rank, serializing prefill to one DP rank per round.
        dp_size = 4
        adder = PrefillAdder(
            page_size=256,
            tree_cache=_DummyRadixCache(),
            token_to_kv_pool_allocator=_DummyAllocator(),
            running_batch=None,
            new_token_ratio=1.0,
            rem_input_tokens=16384,
            rem_chunk_tokens=2048,
            dp_size=dp_size,
        )
        for dp in range(dp_size):
            req = _make_req(f"r{dp}", dp_rank=dp, input_len=16384)
            req.sampling_params.ignore_eos = True
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids)
            req.prefix_indices = []
            req.host_hit_length = 0
            req.last_node = object()
            res = adder.add_one_req(req)
            self.assertEqual(
                res,
                AddReqResult.CONTINUE,
                f"dp_rank={dp} got {res}; chunked admission must not serialize across DP ranks",
            )
        for dp in range(dp_size):
            self.assertEqual(len(adder.can_run_list[dp]), 1)
            self.assertIs(adder.new_chunked_reqs[dp], adder.can_run_list[dp][0])
            self.assertEqual(adder.can_run_list[dp][0].extend_input_len, 2048)


if __name__ == "__main__":
    unittest.main()
