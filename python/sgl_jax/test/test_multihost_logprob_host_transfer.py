import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers import tp_worker
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.managers.tp_worker import ModelWorker


class FakeArray:
    def astype(self, dtype):
        return ("astype", self, dtype)


class TestMultihostLogprobHostTransfer(unittest.TestCase):
    def test_host_logprob_array_allgathers_when_requested(self):
        fake = FakeArray()
        gathered = object()
        calls = []

        def fake_process_allgather(value, *, tiled):
            calls.append(("allgather", value, tiled))
            return gathered

        def fake_device_get(value):
            calls.append(("device_get", value))
            return np.array([1.0, 2.0], dtype=np.float32)

        with (
            mock.patch.object(tp_worker, "process_allgather", fake_process_allgather),
            mock.patch.object(tp_worker.jax, "device_get", fake_device_get),
        ):
            out = tp_worker._host_logprob_array(fake, allgather=True)

        np.testing.assert_array_equal(out, np.array([1.0, 2.0], dtype=np.float32))
        self.assertEqual(
            calls,
            [
                ("allgather", fake, True),
                ("device_get", gathered),
            ],
        )

    def test_host_logprob_array_skips_allgather_by_default(self):
        local = np.array([3.0], dtype=np.float32)
        calls = []

        def fake_process_allgather(value, *, tiled):
            calls.append(("allgather", value, tiled))
            return value

        def fake_device_get(value):
            calls.append(("device_get", value))
            return value

        with (
            mock.patch.object(tp_worker, "process_allgather", fake_process_allgather),
            mock.patch.object(tp_worker.jax, "device_get", fake_device_get),
        ):
            out = tp_worker._host_logprob_array(local)

        np.testing.assert_array_equal(out, local)
        self.assertEqual(calls, [("device_get", local)])

    def test_materialize_logprobs_to_host_preserves_selector_order(self):
        output = LogitsProcessorOutput(
            next_token_logits=np.empty((3, 0)),
            next_token_logprobs=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            next_token_top_logprobs_val=np.array(
                [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], dtype=np.float32
            ),
            next_token_top_logprobs_idx=np.array([[11, 12], [21, 22], [31, 32]], dtype=np.int32),
        )
        batch = type(
            "Batch",
            (),
            {
                "dp_size": 2,
                "top_logprobs_nums": [1, 2, 1],
                "token_ids_logprobs": None,
            },
        )()

        gathered = []

        def fake_process_allgather(value, *, tiled):
            gathered.append(value)
            return value

        with (
            mock.patch.object(tp_worker.jax, "device_get", lambda value: value),
            mock.patch.object(tp_worker, "process_allgather", fake_process_allgather),
        ):
            ModelWorker._materialize_logprobs_to_host(
                None,
                output,
                batch,
                np.array([0, 2, 1], dtype=np.int64),
            )

        np.testing.assert_array_equal(output.next_token_logprobs, np.array([10.0, 30.0, 20.0]))
        self.assertEqual(len(gathered), 2)
        self.assertEqual(len(output.next_token_top_logprobs_val), 3)
        for actual, expected in zip(
            output.next_token_top_logprobs_val,
            [[1.0], [3.0], [2.0, 2.1]],
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected)
        self.assertEqual(output.next_token_top_logprobs_idx, [[11], [31], [21, 22]])

    def test_optional_prefill_output_logprobs_can_be_absent(self):
        req = SimpleNamespace(
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
            top_logprobs_num=2,
            token_ids_logprob=[5, 6],
        )
        output = SimpleNamespace(
            next_token_logprobs=None,
            next_token_top_logprobs_val=None,
            next_token_token_ids_logprobs_val=None,
        )

        SchedulerOutputProcessorMixin.add_logprob_return_values(object(), 0, req, 0, [4], 0, output)

        self.assertEqual(req.output_token_logprobs_val, [])
        self.assertEqual(req.output_top_logprobs_val, [])
        self.assertEqual(req.output_token_ids_logprobs_val, [])


if __name__ == "__main__":
    unittest.main()
