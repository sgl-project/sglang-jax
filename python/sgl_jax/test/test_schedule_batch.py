import unittest
from unittest import mock

from sgl_jax.srt.managers import schedule_batch as schedule_batch_mod
from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.sampling.sampling_params import SamplingParams


class _DummyPrefixCache:
    def __init__(
        self,
        prefix_indices,
        last_node,
        last_host_node,
        host_hit_length,
        root_node,
    ):
        self._prefix_indices = prefix_indices
        self._last_node = last_node
        self._last_host_node = last_host_node
        self._host_hit_length = host_hit_length
        self.root_node = root_node

    def match_prefix(self, key):
        del key
        return (
            list(self._prefix_indices),
            self._last_node,
            self._last_host_node,
            self._host_hit_length,
        )


def _make_req(origin_ids, output_ids):
    req = Req(
        rid="test-rid",
        origin_input_text="",
        origin_input_ids=list(origin_ids),
        sampling_params=SamplingParams(),
    )
    req.output_ids = list(output_ids)
    return req


class TestScheduleBatch(unittest.TestCase):
    def test_multinode_tpu_single_token_extend_bypasses_prefix_cache(self):
        root_node = object()
        matched_node = object()
        cache = _DummyPrefixCache(
            prefix_indices=[101, 102, 103],
            last_node=matched_node,
            last_host_node=matched_node,
            host_hit_length=3,
            root_node=root_node,
        )
        req = _make_req(origin_ids=[1, 2, 3], output_ids=[4])

        with mock.patch.object(
            schedule_batch_mod,
            "DISABLE_MULTINODE_TPU_SINGLE_TOKEN_PREFIX_CACHE",
            True,
        ), mock.patch.object(schedule_batch_mod, "is_multinode_tpu_runtime", return_value=True):
            req.init_next_round_input(cache)

        self.assertEqual(req.prefix_indices, [])
        self.assertIs(req.last_node, root_node)
        self.assertIs(req.last_host_node, root_node)
        self.assertEqual(req.host_hit_length, 0)
        self.assertEqual(req.last_matched_prefix_len, 0)
        self.assertEqual(req.extend_input_len, 4)

    def test_multinode_tpu_multi_token_extend_keeps_prefix_cache(self):
        root_node = object()
        matched_node = object()
        cache = _DummyPrefixCache(
            prefix_indices=[201, 202, 203],
            last_node=matched_node,
            last_host_node=matched_node,
            host_hit_length=3,
            root_node=root_node,
        )
        req = _make_req(origin_ids=[1, 2, 3], output_ids=[4, 5])

        with mock.patch.object(
            schedule_batch_mod,
            "DISABLE_MULTINODE_TPU_SINGLE_TOKEN_PREFIX_CACHE",
            True,
        ), mock.patch.object(schedule_batch_mod, "is_multinode_tpu_runtime", return_value=True):
            req.init_next_round_input(cache)

        self.assertEqual(req.prefix_indices, [201, 202, 203])
        self.assertIs(req.last_node, matched_node)
        self.assertIs(req.last_host_node, matched_node)
        self.assertEqual(req.host_hit_length, 3)
        self.assertEqual(req.last_matched_prefix_len, 3)
        self.assertEqual(req.extend_input_len, 2)


if __name__ == "__main__":
    unittest.main()
