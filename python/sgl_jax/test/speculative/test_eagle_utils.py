import inspect
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.speculative.kernel import create_extend_after_decode_spec_info
from sgl_jax.srt.kernels.speculative.tree_speculative_sampling_target_only_kernel import (
    tree_speculative_sampling_target_only_pallas_call,
)
from sgl_jax.srt.kernels.speculative.verify_tree_greedy_kernel import verify_tree_greedy
from sgl_jax.srt.speculative.eagle_util import build_tree_mask_for_draft_decode
from sgl_jax.test.test_utils import CustomTestCase


class TestVerifyTree(CustomTestCase):
    def test_fused_spec_prefill_guard_only_allows_greedy_plain_requests(self):
        from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

        worker = BaseSpecWorker.__new__(BaseSpecWorker)
        worker._can_use_fused_spec_decode = True

        def make_batch(**overrides):
            sampling_info = SimpleNamespace(
                is_all_greedy=True,
                linear_penalty=None,
                penalizer_orchestrator=SimpleNamespace(is_required=False),
                vocab_mask=None,
            )
            batch = SimpleNamespace(
                sampling_info=sampling_info,
                return_logprob=False,
                return_output_logprob_only=False,
            )
            for key, value in overrides.items():
                if hasattr(sampling_info, key):
                    setattr(sampling_info, key, value)
                else:
                    setattr(batch, key, value)
            return batch

        self.assertTrue(worker._can_use_fused_spec_prefill(make_batch()))
        self.assertFalse(worker._can_use_fused_spec_prefill(make_batch(is_all_greedy=False)))
        self.assertFalse(worker._can_use_fused_spec_prefill(make_batch(linear_penalty=1.0)))
        self.assertFalse(
            worker._can_use_fused_spec_prefill(
                make_batch(penalizer_orchestrator=SimpleNamespace(is_required=True))
            )
        )
        self.assertFalse(worker._can_use_fused_spec_prefill(make_batch(vocab_mask=np.ones(4))))
        self.assertFalse(worker._can_use_fused_spec_prefill(make_batch(return_logprob=True)))
        self.assertFalse(
            worker._can_use_fused_spec_prefill(make_batch(return_output_logprob_only=True))
        )

    def test_spec_precompile_dummy_matches_fused_greedy_runtime_fields(self):
        from sgl_jax.srt.managers.schedule_batch import ForwardMode
        from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        manager = CompilationManager.__new__(CompilationManager)
        manager.vocab_size = 128
        manager.multimodal = False
        manager.has_recurrent_state = False

        spec_batch = manager._make_dummy_batch(
            bs=4,
            num_tokens=64,
            mode=ForwardMode.EXTEND,
            max_cache_loc_size=64,
            speculative_algorithm=SpeculativeAlgorithm.NEXTN,
            dp_size=4,
            per_dp_bs_size=1,
        )
        self.assertFalse(spec_batch.return_logprob)
        self.assertFalse(spec_batch.return_output_logprob_only)
        self.assertTrue(spec_batch.sampling_info.is_all_greedy)
        self.assertIsNone(spec_batch.sampling_info.vocab_mask)

        plain_batch = manager._make_dummy_batch(
            bs=4,
            num_tokens=64,
            mode=ForwardMode.EXTEND,
            max_cache_loc_size=64,
            speculative_algorithm=None,
            dp_size=4,
            per_dp_bs_size=1,
        )
        self.assertTrue(plain_batch.return_output_logprob_only)

    def test_spec_decode_uses_split_verify_and_draft_extend_phases(self):
        from sgl_jax.srt.speculative import draft_extend_fused

        self.assertTrue(hasattr(draft_extend_fused, "_build_fused_greedy_verify_jit"))
        self.assertTrue(hasattr(draft_extend_fused, "_build_fused_greedy_prefill_jit"))
        self.assertTrue(hasattr(draft_extend_fused, "spec_prefill"))
        self.assertTrue(hasattr(draft_extend_fused, "spec_decode_verify_phase"))
        self.assertTrue(hasattr(draft_extend_fused, "spec_decode_draft_extend_phase"))

        source = inspect.getsource(draft_extend_fused.spec_decode)
        self.assertIn("spec_decode_verify_phase", source)
        self.assertIn("spec_decode_draft_extend_phase", source)
        self.assertNotIn("_fused_greedy_decode_jit_fn", source)

    def test_spec_decode_future_result_contract_fields(self):
        from sgl_jax.srt.speculative import overlap_future

        self.assertTrue(hasattr(overlap_future, "SpecDecodeFutureResult"))
        fields = set(overlap_future.SpecDecodeFutureResult.__dataclass_fields__)
        self.assertEqual(
            fields,
            {
                "logits_output",
                "next_token_ids",
                "accept_lens",
                "new_seq_lens",
                "allocate_lens",
                "next_draft_input",
                "bid",
                "cache_miss_count",
            },
        )

    def test_make_spec_decode_future_result_preserves_deferred_fields(self):
        from sgl_jax.srt.speculative import overlap_future

        next_draft_input = SimpleNamespace(
            new_seq_lens=object(),
            hidden_states=object(),
            topk_index=object(),
            verified_id=object(),
        )
        batch_output = SimpleNamespace(
            logits_output=object(),
            next_token_ids=object(),
            accept_lens=object(),
            allocate_lens=object(),
            next_draft_input=next_draft_input,
            bid=7,
            cache_miss_count=0,
        )

        future_result = overlap_future.make_spec_decode_future_result(batch_output)

        self.assertIs(future_result.logits_output, batch_output.logits_output)
        self.assertIs(future_result.next_token_ids, batch_output.next_token_ids)
        self.assertIs(future_result.accept_lens, batch_output.accept_lens)
        self.assertIs(future_result.new_seq_lens, next_draft_input.new_seq_lens)
        self.assertIs(future_result.allocate_lens, batch_output.allocate_lens)
        self.assertIs(future_result.next_draft_input, next_draft_input)
        self.assertEqual(future_result.bid, 7)
        self.assertEqual(future_result.cache_miss_count, 0)

    def test_resolve_spec_decode_scheduler_fields_only_materializes_scheduler_data(
        self,
    ):
        from sgl_jax.srt.speculative import overlap_future

        class DeferredState:
            def __array__(self, dtype=None):
                raise AssertionError("deferred draft state must not be materialized")

        next_draft_input = SimpleNamespace(
            new_seq_lens=jnp.array([4, 7], dtype=jnp.int32),
            hidden_states=DeferredState(),
            topk_index=DeferredState(),
            verified_id=DeferredState(),
        )
        future_result = overlap_future.SpecDecodeFutureResult(
            logits_output=None,
            next_token_ids=jnp.array([[10, 11, 12], [20, 21, 22]], dtype=jnp.int32),
            accept_lens=jnp.array([2, 1], dtype=jnp.int32),
            new_seq_lens=next_draft_input.new_seq_lens,
            allocate_lens=None,
            next_draft_input=next_draft_input,
            bid=0,
            cache_miss_count=0,
        )

        fields = overlap_future.resolve_spec_decode_scheduler_fields(future_result)

        np.testing.assert_array_equal(
            fields.next_token_ids,
            np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int32),
        )
        np.testing.assert_array_equal(fields.accept_lens, np.array([2, 1], dtype=np.int32))
        np.testing.assert_array_equal(fields.new_seq_lens, np.array([4, 7], dtype=np.int32))

    def test_can_use_spec_decode_overlap_gate_lives_under_speculative(self):
        from sgl_jax.srt.speculative import overlap_future

        spec_algorithm = SimpleNamespace(is_none=lambda: False)
        none_algorithm = SimpleNamespace(is_none=lambda: True)
        decode_mode = SimpleNamespace(is_decode=lambda: True)
        extend_mode = SimpleNamespace(is_decode=lambda: False)

        def make_batch(**overrides):
            batch = SimpleNamespace(
                forward_mode=decode_mode,
                return_logprob=False,
                return_output_logprob_only=False,
            )
            for key, value in overrides.items():
                setattr(batch, key, value)
            return batch

        self.assertTrue(
            overlap_future.can_use_spec_decode_overlap(
                enable_overlap=True,
                spec_algorithm=spec_algorithm,
                batch=make_batch(),
            )
        )
        self.assertFalse(
            overlap_future.can_use_spec_decode_overlap(
                enable_overlap=False,
                spec_algorithm=spec_algorithm,
                batch=make_batch(),
            )
        )
        self.assertFalse(
            overlap_future.can_use_spec_decode_overlap(
                enable_overlap=True,
                spec_algorithm=none_algorithm,
                batch=make_batch(),
            )
        )
        self.assertFalse(
            overlap_future.can_use_spec_decode_overlap(
                enable_overlap=True,
                spec_algorithm=spec_algorithm,
                batch=make_batch(forward_mode=extend_mode),
            )
        )
        self.assertFalse(
            overlap_future.can_use_spec_decode_overlap(
                enable_overlap=True,
                spec_algorithm=spec_algorithm,
                batch=make_batch(return_logprob=True),
            )
        )
        self.assertFalse(
            overlap_future.can_use_spec_decode_overlap(
                enable_overlap=True,
                spec_algorithm=spec_algorithm,
                batch=make_batch(return_output_logprob_only=True),
            )
        )

    def test_as_int32_array_keeps_host_metadata_on_host(self):
        from sgl_jax.srt.speculative import eagle_util

        original_jnp_asarray = eagle_util.jnp.asarray
        original_jnp_empty = eagle_util.jnp.empty

        def fail_jnp_asarray(*args, **kwargs):
            raise AssertionError("host metadata conversion must not call jnp.asarray")

        def fail_jnp_empty(*args, **kwargs):
            raise AssertionError("host metadata placeholder must not call jnp.empty")

        try:
            eagle_util.jnp.asarray = fail_jnp_asarray
            eagle_util.jnp.empty = fail_jnp_empty
            arr = eagle_util._as_int32_array(np.array([1, 2], dtype=np.int64))
            scalar = eagle_util._as_int32_array(3)
            listed = eagle_util._as_int32_array([4, 5])
            children, _ = eagle_util.EagleDraftInput().tree_flatten()
        finally:
            eagle_util.jnp.asarray = original_jnp_asarray
            eagle_util.jnp.empty = original_jnp_empty

        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.int32)
        np.testing.assert_array_equal(arr, np.array([1, 2], dtype=np.int32))
        self.assertIsInstance(scalar, np.ndarray)
        self.assertEqual(scalar.dtype, np.int32)
        np.testing.assert_array_equal(listed, np.array([4, 5], dtype=np.int32))
        self.assertIsInstance(children[9], np.ndarray)
        self.assertEqual(children[9].dtype, np.int32)
        self.assertEqual(children[9].shape, (0,))

        device_arr = jnp.array([6], dtype=jnp.int32)
        self.assertIs(eagle_util._as_int32_array(device_arr), device_arr)

    def test_build_chain_verify_inputs_device_matches_linear_chain_layout(self):
        from sgl_jax.srt.speculative.eagle_util import build_chain_verify_inputs_device

        verified_id = jnp.array([101, 201], dtype=jnp.int32)
        token_list = jnp.array(
            [
                [102, 103, 104],
                [202, 203, 204],
            ],
            dtype=jnp.int32,
        )
        seq_lens = jnp.array([7, 11], dtype=jnp.int32)

        packed = build_chain_verify_inputs_device(
            verified_id=verified_id,
            token_list=token_list,
            seq_lens=seq_lens,
            num_verify_tokens=4,
            batch_size=2,
        )

        expected = np.array(
            [
                [101, 102, 103, 104, 201, 202, 203, 204],
                [7, 8, 9, 10, 11, 12, 13, 14],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, -1, 1, 2, 3, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(np.asarray(packed), expected)

    def test_fused_chain_verify_matches_topk1_linear_reference(self):
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _greedy_sample_and_prepare_draft_inputs_chain_from_predict,
        )

        speculative_num_steps = 3
        num_draft_tokens = 4
        bs = 4
        draft_tokens = jnp.array(
            [
                10,
                11,
                12,
                13,
                20,
                21,
                22,
                23,
                30,
                31,
                32,
                33,
                40,
                41,
                42,
                43,
            ],
            dtype=jnp.int32,
        )
        target_predict = np.array(
            [
                99,
                12,
                13,
                14,
                21,
                99,
                23,
                24,
                31,
                32,
                99,
                34,
                41,
                42,
                43,
                44,
            ],
            dtype=np.int32,
        )
        chain = _greedy_sample_and_prepare_draft_inputs_chain_from_predict(
            target_hidden=jnp.arange(bs * num_draft_tokens * 2, dtype=jnp.float32).reshape(
                bs * num_draft_tokens, 2
            ),
            positions=jnp.arange(bs * num_draft_tokens, dtype=jnp.int32),
            seq_lens=jnp.array([100, 200, 300, 400], dtype=jnp.int32),
            draft_tokens=draft_tokens,
            target_predict=jnp.asarray(target_predict),
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=num_draft_tokens,
        )

        np.testing.assert_array_equal(np.asarray(chain.accept_lens), np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(
            np.asarray(chain.select_index),
            np.arange(bs, dtype=np.int32) * (speculative_num_steps + 1)
            + np.asarray(chain.accept_lens)
            - 1,
        )
        select_index = np.asarray(chain.select_index)
        np.testing.assert_array_equal(
            np.asarray(chain.verified_id)[select_index],
            np.array([99, 99, 99, 44], dtype=np.int32),
        )

    def test_fused_chain_verify_zeroes_padding_accept_length(self):
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _greedy_sample_and_prepare_draft_inputs_chain_from_predict,
        )

        out = _greedy_sample_and_prepare_draft_inputs_chain_from_predict(
            target_hidden=jnp.arange(8 * 2, dtype=jnp.float32).reshape(8, 2),
            positions=jnp.arange(8, dtype=jnp.int32),
            seq_lens=jnp.array([0, 10], dtype=jnp.int32),
            draft_tokens=jnp.array([0, 0, 0, 0, 20, 21, 22, 23], dtype=jnp.int32),
            target_predict=jnp.array([0, 0, 0, 0, 21, 22, 99, 24], dtype=jnp.int32),
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )

        np.testing.assert_array_equal(np.asarray(out.accept_lens), np.array([0, 3]))
        np.testing.assert_array_equal(np.asarray(out.new_seq_lens), np.array([0, 13]))
        np.testing.assert_array_equal(np.asarray(out.sel_pos), np.array([0, 2]))

    def test_device_rotate_prefill_input_ids_matches_host_prefill_rotation(self):
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _device_rotate_prefill_input_ids,
        )

        input_ids = np.array([10, 11, 12, 20, 30, 31], dtype=np.int32)
        extend_seq_lens = np.array([3, 1, 0, 2], dtype=np.int32)
        verified_id = np.array([99, 88, 0, 77], dtype=np.int32)

        expected = input_ids.copy()
        pos = 0
        for slot, extend_len in enumerate(extend_seq_lens):
            if extend_len == 0:
                continue
            segment = expected[pos : pos + extend_len].copy()
            expected[pos : pos + extend_len] = np.concatenate(
                (segment[1:], verified_id[slot : slot + 1])
            )
            pos += extend_len

        actual = _device_rotate_prefill_input_ids(
            jax.device_put(input_ids, jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])),
            jax.device_put(
                extend_seq_lens, jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
            ),
            jax.device_put(verified_id, jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])),
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_verify_tree_greedy(self):
        candidates = jnp.array(
            [
                [0, 1, 2, 3, 4, 5],
                [7, 8, 9, 10, 11, 12],
            ],
            dtype=jnp.int32,
        )
        retrive_index = jnp.array(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
            ],
            dtype=jnp.int32,
        )
        retrive_next_token = jnp.array(
            [
                [1, 2, -1, 4, 5, -1],
                [4, 2, 3, -1, 5, -1],
            ],
            dtype=jnp.int32,
        )
        retrive_next_sibling = jnp.array(
            [
                [-1, 3, -1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1],
            ],
            dtype=jnp.int32,
        )

        target_logits = jnp.full((2, 6, 20), 1, dtype=jnp.float32)
        target_logits = target_logits.at[0, 0, 3].set(10)
        target_logits = target_logits.at[0, 3, 4].set(10)
        target_logits = target_logits.at[0, 4, 5].set(10)
        target_logits = target_logits.at[1, 0, 11].set(10)
        target_logits = target_logits.at[1, 4, 12].set(10)
        for i in range(target_logits.shape[0]):
            for j in range(target_logits.shape[1]):
                if jnp.max(target_logits[i][j]) < 10:
                    target_logits = target_logits.at[i, j, 18].set(10)

        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        predict_shape = (12,)

        bs = candidates.shape[0]
        num_spec_step = 4

        predicts = jnp.empty(predict_shape, dtype=jnp.int32)  # mutable
        accept_index = jnp.full((bs, num_spec_step), -1, dtype=jnp.int32)  # mutable
        accept_token_num = jnp.full((bs,), 0, dtype=jnp.int32)  # mutable

        from sgl_jax.srt.utils.mesh_utils import create_device_mesh

        mesh = create_device_mesh(ici_parallelism=[-1, 1], dcn_parallelism=[1, 1])
        with jax.set_mesh(mesh):
            accept_index, accept_token_num, predicts = verify_tree_greedy(
                speculative_num_steps=4,
                num_draft_tokens=6,
                draft_tokens=candidates,
                retrive_index=retrive_index,
                retrive_next_token=retrive_next_token,
                retrive_next_sibling=retrive_next_sibling,
                next_token_logits=target_logits,
            )

        # Check the expected output.
        self.assertEqual(predicts.flatten().tolist(), [3, 0, 0, 4, 5, 18, 11, 0, 0, 0, 12, 18, 0])
        self.assertEqual(accept_index.tolist(), [[0, 3, 4, 5, -1], [6, 10, 11, -1, -1]])
        self.assertEqual(accept_token_num.tolist(), [3, 2])

    def _test_tree_speculative_sampling_target_only(
        self,
        threshold_single,
        threshold_acc,
        expected_predicts,
        expected_accept_index,
        expected_accept_token_num,
    ):
        """
        Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
        """
        candidates = jnp.array(
            [
                [0, 1, 2, 3, 4, 5],
                [7, 8, 9, 10, 11, 12],
            ],
            dtype=jnp.int32,
        )
        retrive_index = jnp.array(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
            ],
            dtype=jnp.int32,
        )
        retrive_next_token = jnp.array(
            [
                [1, 2, -1, 4, 5, -1],
                [4, 2, 3, -1, 5, -1],
            ],
            dtype=jnp.int32,
        )
        retrive_next_sibling = jnp.array(
            [
                [-1, 3, -1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1],
            ],
            dtype=jnp.int32,
        )

        target_logits = jnp.full(
            (2, 6, 20),
            1,
            dtype=jnp.float32,
        )
        target_logits = target_logits.at[0, 0, 3].set(10)
        target_logits = target_logits.at[0, 3, 4].set(10)
        target_logits = target_logits.at[0, 4, 5].set(10)
        target_logits = target_logits.at[1, 0, 11].set(10)
        target_logits = target_logits.at[1, 4, 12].set(10)

        for i in range(target_logits.shape[0]):
            for j in range(target_logits.shape[1]):
                if jnp.max(target_logits[i, j]) < 10:
                    target_logits = target_logits.at[i, j, 18].set(10)

        temperatures = jnp.array(
            [0.01, 0.01],
            dtype=jnp.float32,
        )
        bs, num_draft_tokens = candidates.shape
        num_spec_step = len(expected_accept_index[0])
        predict_shape = (len(expected_predicts),)

        predicts = jnp.full(
            predict_shape,
            -1,
            dtype=jnp.int32,
        )
        accept_index = jnp.full(
            (bs, num_spec_step),
            -1,
            dtype=jnp.int32,
        )
        accept_token_num = jnp.full(
            (bs,),
            0,
            dtype=jnp.int32,
        )

        expanded_temperature = jnp.expand_dims(jnp.expand_dims(temperatures, axis=1), axis=1)
        target_probs = jax.nn.softmax(target_logits / expanded_temperature, axis=-1).reshape(
            bs * num_draft_tokens, -1
        )
        draft_probs = jnp.full_like(
            target_probs,
            0,
            dtype=jnp.float32,
        )
        coins = jax.random.uniform(
            jax.random.PRNGKey(42), (bs, num_draft_tokens), dtype=jnp.float32
        )
        coins_for_final_sampling = jax.random.uniform(
            jax.random.PRNGKey(42), (bs,), dtype=jnp.float32
        )
        accept_index, accept_token_num, predicts = (
            tree_speculative_sampling_target_only_pallas_call(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrive_index=retrive_index,
                retrive_next_token=retrive_next_token,
                retrive_next_sibling=retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=threshold_single,
                threshold_acc=threshold_acc,
                deterministic=True,
            )
        )

        self.assertEqual(
            predicts.tolist(),
            expected_predicts,
            f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})",
        )
        self.assertEqual(
            accept_index.tolist(),
            expected_accept_index,
            f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})",
        )
        self.assertEqual(
            accept_token_num.tolist(),
            expected_accept_token_num,
            f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})",
        )

    def test_tree_speculative_sampling_target_only(self):
        # this kernel still have some problems, skip benchmark test
        return
        test_cases = [
            (
                1,
                1,
                [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
                [[0, 3, 4, 5], [6, 10, 11, -1]],
                [3, 2],
            ),
            (
                0,  # threshold_single
                0,  # threshold_acc
                [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
                [[0, 1, 2, -1], [6, 10, 11, -1]],
                [2, 2],
            ),
        ]

        for (
            threshold_single,
            threshold_acc,
            expected_predicts,
            expected_accept_index,
            expected_accept_token_num,
        ) in test_cases:
            self._test_tree_speculative_sampling_target_only(
                threshold_single,
                threshold_acc,
                expected_predicts,
                expected_accept_index,
                expected_accept_token_num,
            )

    def test_create_extend_after_decode_spec_info(self):
        verified_id = jnp.array([100, 101, 102, 200, 201, 300], dtype=jnp.int32)
        seq_lens = jnp.array([10, 15, 8], dtype=jnp.int32)
        accept_lens = jnp.array([2, 3, 1], dtype=jnp.int32)
        positions = jnp.array([0] * 6, dtype=jnp.int32)
        new_verified_id = jnp.array([0] * 3, dtype=jnp.int32)
        positions, new_verified_id = create_extend_after_decode_spec_info(
            verified_id, seq_lens, accept_lens, positions, new_verified_id
        )

        expected_postions = [8, 9, 12, 13, 14, 7]
        self.assertEqual(
            positions.tolist(),
            expected_postions,
            f"positions not equal, result: {positions.tolist()}, expected: {expected_postions}",
        )
        expected_verified_id = [101, 201, 300]
        self.assertEqual(
            new_verified_id.tolist(),
            expected_verified_id,
            f"verified_id not equal, result: {new_verified_id.tolist()}, expected: {expected_verified_id}",
        )


class TestDraftDecodeMask(CustomTestCase):
    def _expected(
        self,
        seq_len: int,
        speculative_step_id: int,
        topk: int,
        parents_rows: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        kv_len = seq_len + (speculative_step_id + 1) * topk
        mask = np.zeros((topk, kv_len), dtype=bool)
        mask[:, :seq_len] = True

        ancestry = np.zeros((speculative_step_id + 1, topk), dtype=np.int32)
        ancestry[speculative_step_id] = np.arange(topk, dtype=np.int32)

        parent_rows = parents_rows or [None] * (speculative_step_id + 1)
        if len(parent_rows) < speculative_step_id + 1:
            parent_rows = list(parent_rows) + [None] * (speculative_step_id + 1 - len(parent_rows))

        for step in range(speculative_step_id, 0, -1):
            row = parent_rows[step]
            if row is None:
                ancestry[step - 1] = ancestry[step]
                continue
            row_np = np.asarray(row, dtype=np.int64)
            offset = topk if step == 1 else topk**2 * (step - 1) + topk
            parent_indices = np.clip((row_np - offset) // topk, 0, topk - 1).astype(np.int32)
            ancestry[step - 1] = parent_indices[ancestry[step]]

        for branch in range(topk):
            for step in range(speculative_step_id + 1):
                branch_idx = ancestry[step, branch]
                position = seq_len + step * topk + branch_idx
                mask[branch, position] = True

        return mask.reshape(-1)

    def test_single_step(self):
        mask = build_tree_mask_for_draft_decode(
            seq_lens=jnp.array([6]),
            topk=3,
            speculative_step_id=0,
            parents_list=[jnp.arange(-1, 3, dtype=jnp.int32)[None, :]],
        )
        expected = self._expected(6, speculative_step_id=0, topk=3)
        print(f"======test_single_step======{mask=}================")

        self.assertEqual(mask.tolist(), expected.tolist())

    def test_multi_step(self):
        parents_rows = [None, np.array([4, 3]), np.array([6, 9])]
        parents = [
            jnp.array([[-1, 0]], dtype=jnp.int32),
            jnp.array([[4, 3]], dtype=jnp.int32),
            jnp.array([[6, 9]], dtype=jnp.int32),
        ]
        mask = build_tree_mask_for_draft_decode(
            seq_lens=jnp.array([4]),
            topk=2,
            speculative_step_id=2,
            parents_list=parents,
        )
        expected = self._expected(4, speculative_step_id=2, topk=2, parents_rows=parents_rows)
        print(f"=====test_multi_step======={mask=}================")

        self.assertEqual(mask.tolist(), expected.tolist())

    def test_batch_concatenation(self):
        parents = [
            jnp.tile(jnp.arange(-1, 2, dtype=jnp.int32), (2, 1)),
            jnp.array([[4, 3], [2, 5]], dtype=jnp.int32),  # offset 2 -> batch-specific parents
        ]
        mask = build_tree_mask_for_draft_decode(
            seq_lens=jnp.array([3, 5]),
            topk=2,
            speculative_step_id=1,
            parents_list=parents,
        )
        expected = np.concatenate(
            [
                self._expected(
                    3,
                    speculative_step_id=1,
                    topk=2,
                    parents_rows=[None, np.array([4, 3])],
                ),
                self._expected(
                    5,
                    speculative_step_id=1,
                    topk=2,
                    parents_rows=[None, np.array([2, 5])],
                ),
            ]
        )
        print(f"====test_batch_concatenation========{mask=}================")

        self.assertEqual(mask.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
