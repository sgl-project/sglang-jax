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


class TestMultihostLogprobHostTransfer(unittest.TestCase):
    def test_host_logprob_array_gathers_only_when_requested(self):
        local = np.array([3.0], dtype=np.float32)
        sharded, gathered = object(), np.array([1.0, 2.0], dtype=np.float32)
        allgather = mock.Mock(return_value=gathered)
        device_get = mock.Mock(side_effect=lambda value: value)

        with (
            mock.patch.object(tp_worker, "process_allgather", allgather),
            mock.patch.object(tp_worker.jax, "device_get", device_get),
        ):
            local_out = tp_worker._host_logprob_array(local)
            gathered_out = tp_worker._host_logprob_array(sharded, allgather=True)

        np.testing.assert_array_equal(local_out, local)
        np.testing.assert_array_equal(gathered_out, gathered)
        allgather.assert_called_once_with(sharded, tiled=True)
        self.assertEqual(device_get.call_args_list, [mock.call(local), mock.call(gathered)])

    @staticmethod
    def _materialize(output, batch, selector):
        allgather = mock.Mock(side_effect=lambda value, *, tiled: value)
        with (
            mock.patch.object(tp_worker.jax, "device_get", side_effect=lambda value: value),
            mock.patch.object(tp_worker, "process_allgather", allgather),
        ):
            ModelWorker._materialize_logprobs_to_host(None, output, batch, selector)
        return allgather

    def test_materialize_logprobs_to_host_preserves_selector_order(self):
        output = LogitsProcessorOutput(
            next_token_logits=np.empty((3, 0)),
            next_token_logprobs=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            next_token_top_logprobs_val=np.array(
                [[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]], dtype=np.float32
            ),
            next_token_top_logprobs_idx=np.array([[11, 12], [21, 22], [31, 32]], dtype=np.int32),
        )
        batch = SimpleNamespace(
            dp_size=2,
            top_logprobs_nums=[1, 2, 1],
            token_ids_logprobs=None,
        )
        allgather = self._materialize(output, batch, np.array([0, 2, 1]))

        np.testing.assert_array_equal(output.next_token_logprobs, np.array([10.0, 30.0, 20.0]))
        self.assertEqual(allgather.call_count, 2)
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
        batch = SimpleNamespace(
            dp_size=2,
            per_dp_bs_size=1,
            real_bs_per_dp=[1, 1],
            extend_seq_lens=np.array([2, 1]),
            extend_logprob_start_lens=np.array([0, 0]),
            top_logprobs_nums=[2, 1],
            token_ids_logprobs=None,
        )
        allgather = self._materialize(output, batch, np.array([0, 1]))

        self.assertEqual(allgather.call_count, 2)
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
        batch = SimpleNamespace(
            sampling_info=SimpleNamespace(is_all_greedy=True),
            return_logprob=False,
            return_output_logprob_only=False,
        )
        cases = [
            (False, False, False, True),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
        ]
        for return_logprob, output_only, legacy, expected in cases:
            with self.subTest(
                return_logprob=return_logprob,
                output_only=output_only,
                legacy=legacy,
            ):
                batch.return_logprob = return_logprob
                batch.return_output_logprob_only = output_only
                self.assertEqual(worker._can_skip_greedy_prefill_sample(batch, legacy), expected)


class TestSpecDecodeOutputLogprobs(unittest.TestCase):
    def _run_decode(
        self,
        *,
        output_only: bool,
        stop_token_ids=None,
        draft_n: int = 4,
        spec_steps: int = 3,
    ):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.__dict__.update(
            spec_algorithm=SpeculativeAlgorithm.NEXTN,
            enable_overlap=output_only,
            draft_worker=SimpleNamespace(
                speculative_num_draft_tokens=draft_n,
                speculative_num_steps=spec_steps,
            ),
            num_generated_tokens=0,
            accept_token=0,
            spec_num_forward_ct=0,
            draft_token=0,
            token_to_kv_pool_allocator=SimpleNamespace(
                free_group_begin=lambda: None,
                free_group_end=lambda: None,
            ),
            tree_cache=None,
            set_next_batch_sampling_info_done=lambda _batch: None,
            stream_output=lambda *args, **kwargs: None,
            forward_ct_decode=0,
            server_args=SimpleNamespace(decode_log_interval=1000),
            log_decode_stats=lambda _batch: None,
        )

        def make_req(rid, *, stops=None):
            req = Req(
                rid=rid,
                origin_input_text="",
                origin_input_ids=[1, 2],
                sampling_params=SamplingParams(max_new_tokens=10, stop_token_ids=stops),
                return_logprob=not output_only,
                top_logprobs_num=0 if output_only else 2,
                token_ids_logprob=None if output_only else [7, 8],
            )
            if output_only:
                req.return_output_logprob_only = True
                req.output_token_logprobs_val = []
                req.output_token_logprobs_idx = []
            return req

        req0 = make_req("rank-0", stops=stop_token_ids)
        req1 = make_req("rank-1")

        rows_per_req = spec_steps + 1
        total_rows = 2 * rows_per_req
        top_vals = top_idx = token_vals = None
        if not output_only:
            shape = (total_rows, 2)
            top_vals = -np.arange(1, total_rows * 2 + 1, dtype=np.float32).reshape(shape)
            top_idx = np.arange(10, 10 + total_rows * 2, dtype=np.int32).reshape(shape)
            token_vals = np.tile(np.arange(-10.0, 0.0, dtype=np.float32), (total_rows, 1))

        batch = SimpleNamespace(
            dp_size=2,
            per_dp_bs_size=1,
            reqs_info=[SimpleNamespace(reqs=[req0]), SimpleNamespace(reqs=[req1])],
            return_logprob=not output_only,
            return_output_logprob_only=output_only,
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logprobs=-0.1 * np.arange(1, total_rows + 1, dtype=np.float32),
                next_token_top_logprobs_val=top_vals,
                next_token_top_logprobs_idx=top_idx,
                next_token_token_ids_logprobs_val=token_vals,
                hidden_states=None,
            ),
            next_token_ids=np.concatenate(
                [
                    np.arange(101, 101 + draft_n, dtype=np.int32),
                    np.arange(201, 201 + draft_n, dtype=np.int32),
                ]
            ),
            accept_lens=np.array([2, 3], dtype=np.int32),
            cache_miss_count=0,
            bid=1,
            num_accepted_tokens=None,
        )

        gathered = mock.Mock(side_effect=lambda value, *, tiled: value)
        with mock.patch("jax.experimental.multihost_utils.process_allgather", gathered):
            SchedulerOutputProcessorMixin.process_batch_result_decode(scheduler, batch, result)
        return (req0, req1), gathered

    def test_spec_decode_appends_flat_output_logprobs(self):
        (req, rank1_req), gathered = self._run_decode(output_only=False)

        self.assertEqual(req.output_ids, [101, 102])
        self.assertEqual(gathered.call_count, 3)
        self.assertEqual(req.output_token_logprobs_idx, [101, 102])
        np.testing.assert_allclose(req.output_token_logprobs_val, [-0.1, -0.2])
        self.assertEqual(req.output_top_logprobs_idx, [[10, 11], [12, 13]])
        self.assertEqual(req.output_token_ids_logprobs_idx, [[7, 8], [7, 8]])
        np.testing.assert_allclose(req.output_token_ids_logprobs_val, [[-3.0, -2.0]] * 2)
        self.assertEqual(rank1_req.output_ids, [201, 202, 203])
        self.assertEqual(rank1_req.output_token_logprobs_idx, [201, 202, 203])
        np.testing.assert_allclose(rank1_req.output_token_logprobs_val, [-0.5, -0.6, -0.7])
        self.assertEqual(rank1_req.output_top_logprobs_idx, [[18, 19], [20, 21], [22, 23]])

    def test_spec_decode_appends_output_logprob_only(self):
        (req, rank1_req), gathered = self._run_decode(output_only=True)

        gathered.assert_not_called()
        self.assertEqual(req.output_token_logprobs_idx, [101, 102])
        np.testing.assert_allclose(req.output_token_logprobs_val, [-0.1, -0.2])
        self.assertIsNone(req.output_top_logprobs_val)
        self.assertIsNone(req.output_token_ids_logprobs_val)
        self.assertEqual(rank1_req.output_token_logprobs_idx, [201, 202, 203])

    def test_non_fused_tree_uses_accept_width_for_logprob_rows(self):
        (req, rank1_req), _ = self._run_decode(
            output_only=False,
            draft_n=6,
            spec_steps=2,
        )

        self.assertEqual(req.output_token_logprobs_idx, [101, 102])
        np.testing.assert_allclose(req.output_token_logprobs_val, [-0.1, -0.2])
        self.assertEqual(rank1_req.output_token_logprobs_idx, [201, 202, 203])
        np.testing.assert_allclose(rank1_req.output_token_logprobs_val, [-0.4, -0.5, -0.6])
        self.assertEqual(rank1_req.output_top_logprobs_idx, [[16, 17], [18, 19], [20, 21]])

    def test_spec_decode_does_not_return_logprobs_after_stop(self):
        (req, _), _ = self._run_decode(output_only=False, stop_token_ids=[101])

        self.assertEqual(req.output_ids_through_stop, [101])
        self.assertEqual(req.output_token_logprobs_idx, [101])
        np.testing.assert_allclose(req.output_token_logprobs_val, [-0.1])
        self.assertEqual(req.output_top_logprobs_idx, [[10, 11]])
        self.assertEqual(req.output_token_ids_logprobs_idx, [[7, 8]])


class TestSamplerLogprobOutput(unittest.TestCase):
    def test_process_logprob_results_preserves_hidden_states(self):
        sampler = SimpleNamespace(mesh=Mesh(np.array(jax.devices()), ("data",)))
        hidden_states = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        logits_output = LogitsProcessorOutput(
            next_token_logits=jnp.zeros((2, 4), dtype=jnp.float32),
            hidden_states=hidden_states,
            input_token_logprobs=jnp.array([0.5, 0.25], dtype=jnp.float32),
        )
        sampling_metadata = SimpleNamespace(top_logprobs_nums=None, token_ids_logprobs=None)

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
