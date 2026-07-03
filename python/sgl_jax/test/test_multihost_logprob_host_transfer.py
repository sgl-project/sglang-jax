import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import Sampler
from sgl_jax.srt.managers import tp_worker
from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm


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

    def test_materialize_input_top_logprobs_without_next_token_outputs(self):
        output = LogitsProcessorOutput(
            next_token_logits=np.empty((2, 0)),
            input_token_logprobs=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            input_top_logprobs_val=np.array(
                [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1]], dtype=np.float32
            ),
            input_top_logprobs_idx=np.array(
                [[11, 12], [21, 22], [31, 32], [41, 42]], dtype=np.int32
            ),
        )
        batch = type(
            "Batch",
            (),
            {
                "dp_size": 2,
                "per_dp_bs_size": 1,
                "real_bs_per_dp": [1, 1],
                "extend_seq_lens": np.array([2, 1], dtype=np.int32),
                "extend_logprob_start_lens": np.array([0, 0], dtype=np.int32),
                "top_logprobs_nums": [2, 1],
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
                np.array([0, 1], dtype=np.int64),
            )

        self.assertEqual(len(gathered), 2)
        np.testing.assert_allclose(output.input_token_logprobs, [0.1, 0.2, 0.3, 0.4])
        for actual, expected in zip(
            output.input_top_logprobs_val,
            [[[1.0, 1.1], [2.0, 2.1]], [[3.0]]],
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected)
        self.assertEqual(output.input_top_logprobs_idx, [[[11, 12], [21, 22]], [[31]]])


class TestSpecPrefillLogprobPaths(unittest.TestCase):
    def test_logprob_requests_do_not_skip_greedy_prefill_sample(self):
        worker = object.__new__(BaseSpecWorker)
        sampling_info = type("SamplingInfo", (), {"is_all_greedy": True})()

        batch = type(
            "Batch",
            (),
            {
                "sampling_info": sampling_info,
                "return_logprob": False,
                "return_output_logprob_only": False,
            },
        )()

        self.assertTrue(worker._can_skip_greedy_prefill_sample(batch, False))

        batch.return_logprob = True
        self.assertFalse(worker._can_skip_greedy_prefill_sample(batch, False))

        batch.return_logprob = False
        batch.return_output_logprob_only = True
        self.assertFalse(worker._can_skip_greedy_prefill_sample(batch, False))

        batch.return_output_logprob_only = False
        self.assertFalse(worker._can_skip_greedy_prefill_sample(batch, True))


class TestSpecDecodeOutputLogprobs(unittest.TestCase):
    def _run_decode(self, *, output_only: bool):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.spec_algorithm = SpeculativeAlgorithm.NEXTN
        scheduler.enable_overlap = output_only
        scheduler.draft_worker = SimpleNamespace(speculative_num_draft_tokens=4)
        scheduler.num_generated_tokens = 0
        scheduler.accept_token = 0
        scheduler.spec_num_forward_ct = 0
        scheduler.draft_token = 0
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=lambda: None,
            free_group_end=lambda: None,
        )
        scheduler.tree_cache = None
        scheduler.set_next_batch_sampling_info_done = lambda _batch: None
        scheduler.stream_output = lambda *args, **kwargs: None
        scheduler.forward_ct_decode = 0
        scheduler.server_args = SimpleNamespace(decode_log_interval=1000)
        scheduler.log_decode_stats = lambda _batch: None

        req = Req(
            rid="rid",
            origin_input_text="",
            origin_input_ids=[1, 2],
            sampling_params=SamplingParams(max_new_tokens=10),
            return_logprob=not output_only,
            top_logprobs_num=0 if output_only else 2,
            token_ids_logprob=None if output_only else [7, 8],
        )
        if output_only:
            req.return_output_logprob_only = True
            req.output_token_logprobs_val = []
            req.output_token_logprobs_idx = []

        top_vals = None
        top_idx = None
        token_vals = None
        if not output_only:
            top_vals = np.array(
                [[-1.0, -1.1], [-2.0, -2.1], [-3.0, -3.1], [-4.0, -4.1]],
                dtype=np.float32,
            )
            top_idx = np.array([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.int32)
            token_vals = np.tile(np.arange(-10.0, 0.0, dtype=np.float32), (4, 1))

        batch = SimpleNamespace(
            dp_size=2,
            per_dp_bs_size=1,
            reqs_info=[SimpleNamespace(reqs=[req]), SimpleNamespace(reqs=[])],
            return_logprob=not output_only,
            return_output_logprob_only=output_only,
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logprobs=np.array([-0.1, -0.2, -0.3, -0.4], dtype=np.float32),
                next_token_top_logprobs_val=top_vals,
                next_token_top_logprobs_idx=top_idx,
                next_token_token_ids_logprobs_val=token_vals,
                hidden_states=None,
            ),
            next_token_ids=np.array([101, 102, 103, 104], dtype=np.int32),
            accept_lens=np.array([2], dtype=np.int32),
            cache_miss_count=0,
            bid=1,
            num_accepted_tokens=None,
        )

        gathered = []

        def fake_process_allgather(value, *, tiled):
            gathered.append((value, tiled))
            return value

        with mock.patch(
            "jax.experimental.multihost_utils.process_allgather",
            fake_process_allgather,
        ):
            SchedulerOutputProcessorMixin.process_batch_result_decode(scheduler, batch, result)
        return req, gathered

    def test_spec_decode_appends_flat_output_logprobs(self):
        req, gathered = self._run_decode(output_only=False)

        self.assertEqual(req.output_ids, [101, 102])
        self.assertEqual(len(gathered), 3)
        self.assertEqual(req.output_token_logprobs_idx, [101, 102])
        np.testing.assert_allclose(req.output_token_logprobs_val, [-0.1, -0.2])
        self.assertEqual(req.output_top_logprobs_idx, [[10, 11], [20, 21]])
        self.assertEqual(req.output_token_ids_logprobs_idx, [[7, 8], [7, 8]])
        np.testing.assert_allclose(req.output_token_ids_logprobs_val, [[-3.0, -2.0]] * 2)

    def test_spec_decode_appends_output_logprob_only(self):
        req, gathered = self._run_decode(output_only=True)

        self.assertEqual(gathered, [])
        self.assertEqual(req.output_token_logprobs_idx, [101, 102])
        np.testing.assert_allclose(req.output_token_logprobs_val, [-0.1, -0.2])
        self.assertIsNone(req.output_top_logprobs_val)
        self.assertIsNone(req.output_token_ids_logprobs_val)


class TestSamplerLogprobOutput(unittest.TestCase):
    def test_process_logprob_results_preserves_hidden_states(self):
        sampler = type("SamplerLike", (), {"mesh": Mesh(np.array(jax.devices()), ("data",))})()
        hidden_states = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        logits_output = LogitsProcessorOutput(
            next_token_logits=jnp.zeros((2, 4), dtype=jnp.float32),
            hidden_states=hidden_states,
            input_token_logprobs=jnp.array([0.5, 0.25], dtype=jnp.float32),
        )
        sampling_metadata = type(
            "SamplingMetadata",
            (),
            {
                "top_logprobs_nums": None,
                "token_ids_logprobs": None,
            },
        )()

        output = Sampler._process_logprob_results(
            sampler,
            (
                logits_output,
                sampling_metadata,
                jnp.array([1, 2], dtype=jnp.int32),
                jnp.array(
                    [[0.0, -0.1, -0.2, -0.3], [-1.0, -1.1, -1.2, -1.3]],
                    dtype=jnp.float32,
                ),
            ),
        )

        np.testing.assert_array_equal(np.asarray(output.hidden_states), np.asarray(hidden_states))
        np.testing.assert_allclose(np.asarray(output.next_token_logprobs), [-0.1, -1.2])
        np.testing.assert_array_equal(
            np.asarray(output.input_token_logprobs), np.asarray(logits_output.input_token_logprobs)
        )


if __name__ == "__main__":
    unittest.main()
