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
from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker


class TestMultihostLogprobHostTransfer(unittest.TestCase):
    @staticmethod
    def _materialize(output, batch, selector):
        materialize = mock.Mock(side_effect=np.asarray)
        with mock.patch.object(tp_worker, "materialize_to_host", materialize):
            ModelWorker._materialize_logprobs_to_host(None, output, batch, selector)
        return materialize

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
        self._materialize(output, batch, np.array([0, 2, 1]))

        np.testing.assert_array_equal(output.next_token_logprobs, np.array([10.0, 30.0, 20.0]))
        for actual, expected in zip(
            output.next_token_top_logprobs_val,
            [[1.0], [3.0], [2.0, 2.1]],
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected)
        self.assertEqual(output.next_token_top_logprobs_idx, [[11], [31], [21, 22]])

    def test_merge_logprob_metadata_preserves_dp_padding(self):
        def info(seq_lens, top_nums, token_ids):
            return SimpleNamespace(
                seq_lens=seq_lens,
                top_logprobs_nums=top_nums,
                token_ids_logprobs=token_ids,
            )

        batch = SimpleNamespace(
            return_logprob=True,
            dp_size=2,
            reqs_info=[
                info([4, 5], [1, 2], [[10], [20]]),
                info([6], [3], [[30]]),
            ],
        )

        top_nums, token_ids = ScheduleBatch._merge_logprob_metadata(batch, 3, 6)
        self.assertEqual(top_nums, [1, 2, 0, 3, 0, 0])
        self.assertEqual(token_ids, [[10], [20], None, [30], None, None])

        batch.return_logprob = False
        self.assertEqual(ScheduleBatch._merge_logprob_metadata(batch, 3, 6), (None, None))

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
        materialize = self._materialize(output, batch, np.array([0, 1]))

        self.assertEqual(materialize.call_count, 2)
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
    @staticmethod
    def _req(**kwargs):
        values = dict(
            return_logprob=True,
            finished_len=None,
            top_logprobs_num=2,
            token_ids_logprob=[1, 3],
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
        )
        values.update(kwargs)
        return SimpleNamespace(**values)

    @staticmethod
    def _output():
        return SimpleNamespace(
            next_token_logprobs=np.arange(8, dtype=np.float32),
            next_token_top_logprobs_val=np.arange(24, dtype=np.float32).reshape(8, 3),
            next_token_top_logprobs_idx=np.arange(100, 124, dtype=np.int32).reshape(8, 3),
            next_token_token_ids_logprobs_val=np.arange(40, dtype=np.float32).reshape(8, 5),
        )

    def test_uses_accepted_width_and_dp_slot(self):
        req = self._req()
        SchedulerOutputProcessorMixin.add_spec_decode_logprob_return_values(
            req,
            self._output(),
            [201, 202, 203],
            slot=1,
            rows_per_req=3,
            output_len_before=0,
            accepted_len=3,
        )

        self.assertEqual(req.output_token_logprobs_val, [3, 4, 5])
        self.assertEqual(req.output_token_logprobs_idx, [201, 202, 203])
        self.assertEqual(req.output_top_logprobs_idx, [[109, 110], [112, 113], [115, 116]])
        self.assertEqual(req.output_token_ids_logprobs_val, [[16, 18], [21, 23], [26, 28]])
        self.assertEqual(req.output_token_ids_logprobs_idx, [[1, 3]] * 3)

    def test_truncates_output_only_logprobs_at_stop(self):
        req = self._req(
            return_logprob=False,
            finished_len=3,
            top_logprobs_num=0,
            token_ids_logprob=None,
        )
        SchedulerOutputProcessorMixin.add_spec_decode_logprob_return_values(
            req,
            self._output(),
            [101, 102, 103],
            slot=0,
            rows_per_req=4,
            output_len_before=2,
            accepted_len=3,
        )

        self.assertEqual(req.output_token_logprobs_val, [0])
        self.assertEqual(req.output_token_logprobs_idx, [101])
        self.assertEqual(req.output_top_logprobs_val, [])
        self.assertEqual(req.output_token_ids_logprobs_val, [])

    def test_allows_missing_optional_logprob_outputs(self):
        req = self._req()
        output = self._output()
        output.next_token_top_logprobs_val = None
        output.next_token_token_ids_logprobs_val = None

        SchedulerOutputProcessorMixin.add_spec_decode_logprob_return_values(
            req, output, [101], 0, 4, 0, 1
        )

        self.assertEqual(req.output_token_logprobs_val, [0])
        self.assertEqual(req.output_top_logprobs_val, [])
        self.assertEqual(req.output_token_ids_logprobs_val, [])


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
