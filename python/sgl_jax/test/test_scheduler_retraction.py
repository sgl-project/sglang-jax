import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch


class _AllocatorWithCapacity:
    def __init__(self, capacity_per_dp: dict[int, int]):
        self.capacity_per_dp = capacity_per_dp

    def available_size(self, dp_rank: int = 0):
        return self.capacity_per_dp[dp_rank]


def _request(rid: str):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1, 2, 3],
        output_ids=[4],
        sampling_params=SimpleNamespace(max_new_tokens=16),
    )


class TestSchedulerRetraction(unittest.TestCase):
    def test_retracts_only_dp_rank_with_insufficient_memory(self):
        dp0_reqs = [_request("dp0-r0"), _request("dp0-r1")]
        dp1_reqs = [_request("dp1-r0"), _request("dp1-r1")]
        batch = object.__new__(ScheduleBatch)
        batch.dp_size = 2
        batch.reqs_info = [SimpleNamespace(reqs=dp0_reqs), SimpleNamespace(reqs=dp1_reqs)]
        batch.is_hybrid = False
        batch.tree_cache = None
        batch.token_to_kv_pool_allocator = _AllocatorWithCapacity({0: 1, 1: 2})
        batch.new_tokens_required_next_decode = Mock(
            side_effect=lambda _dp_rank, indices: len(indices)
        )
        batch.release_req = Mock()
        batch.filter_batch = Mock()

        retracted, _, aborted = batch.retract_decode(SimpleNamespace())

        self.assertEqual([req.rid for req in retracted], ["dp0-r1"])
        self.assertEqual(aborted, [])
        batch.release_req.assert_called_once_with(1, 0, 1, ANY)
        batch.filter_batch.assert_called_once_with(keep_indices={0: [0], 1: [0, 1]})


if __name__ == "__main__":
    unittest.main()
