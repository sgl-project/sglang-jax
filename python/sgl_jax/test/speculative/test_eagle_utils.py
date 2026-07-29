import unittest
from unittest.mock import Mock

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
    def test_as_int32_array_keeps_host_metadata_on_host(self):
        from sgl_jax.srt.speculative import eagle_info

        original_jnp_asarray = eagle_info.jnp.asarray
        original_jnp_empty = eagle_info.jnp.empty

        def fail_jnp_asarray(*args, **kwargs):
            raise AssertionError("host metadata conversion must not call jnp.asarray")

        def fail_jnp_empty(*args, **kwargs):
            raise AssertionError("host metadata placeholder must not call jnp.empty")

        try:
            eagle_info.jnp.asarray = fail_jnp_asarray
            eagle_info.jnp.empty = fail_jnp_empty
            arr = eagle_info._as_int32_array(np.array([1, 2], dtype=np.int64))
            scalar = eagle_info._as_int32_array(3)
            listed = eagle_info._as_int32_array([4, 5])
            children, _ = eagle_info.EagleDraftInput().tree_flatten()
        finally:
            eagle_info.jnp.asarray = original_jnp_asarray
            eagle_info.jnp.empty = original_jnp_empty

        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.int32)
        np.testing.assert_array_equal(arr, np.array([1, 2], dtype=np.int32))
        self.assertIsInstance(scalar, np.ndarray)
        self.assertEqual(scalar.dtype, np.int32)
        np.testing.assert_array_equal(listed, np.array([4, 5], dtype=np.int32))
        self.assertIsNone(children[6])
        self.assertIsInstance(children[7], np.ndarray)
        self.assertEqual(children[7].dtype, np.int32)
        self.assertEqual(children[7].shape, (0,))

        device_arr = jnp.array([6], dtype=jnp.int32)
        self.assertIs(eagle_info._as_int32_array(device_arr), device_arr)

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
        from sgl_jax.srt.speculative.draft_extend_fused import _verify_greedy

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
        chain = _verify_greedy(
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
        from sgl_jax.srt.speculative.draft_extend_fused import _verify_greedy

        out = _verify_greedy(
            target_hidden=jnp.arange(8 * 2, dtype=jnp.float32).reshape(8, 2),
            positions=jnp.arange(8, dtype=jnp.int32),
            seq_lens=jnp.array([0, 10], dtype=jnp.int32),
            draft_tokens=jnp.array([0, 0, 0, 0, 20, 21, 22, 23], dtype=jnp.int32),
            target_predict=jnp.array([0, 0, 0, 0, 21, 22, 99, 24], dtype=jnp.int32),
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )

        np.testing.assert_array_equal(np.asarray(out.accept_lens), np.array([0, 3]))
        np.testing.assert_array_equal(np.asarray(out.new_seq_lens), np.array([1, 14]))
        np.testing.assert_array_equal(np.asarray(out.sel_pos), np.array([0, 2]))

    def test_recurrent_eagle3_chain_skips_draft_model_forward(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        worker = object.__new__(EagleDraftWorker)
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE3
        worker.speculative_num_steps = 3
        worker.topk = 1
        token_chain = jnp.array([[11, 12, 13], [21, 22, 23]], dtype=jnp.int32)
        batch = SimpleNamespace(
            spec_info_padded=SimpleNamespace(
                topk_p=jnp.ones_like(token_chain, dtype=jnp.float32),
                topk_index=token_chain,
                hidden_states=jnp.zeros((2, 4), dtype=jnp.float32),
            )
        )

        scores, tokens, parents = worker.draft_forward(batch)

        self.assertIsNone(scores)
        self.assertIsNone(parents)
        np.testing.assert_array_equal(np.asarray(tokens), np.asarray(token_chain))

    def test_fused_verify_preparation_keeps_recurrent_chain_raw(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        worker = object.__new__(EagleDraftWorker)
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE3
        worker.speculative_num_steps = 3
        worker.topk = 1
        worker.hot_token_ids = object()
        worker.padding_for_decode = Mock()
        token_chain = jnp.array([[11, 12, 13], [21, 22, 23]], dtype=jnp.int32)
        worker.draft_forward = Mock(return_value=(None, token_chain, None))
        batch = SimpleNamespace(
            spec_info_padded=SimpleNamespace(topk_index=token_chain),
        )

        mapping = worker.prepare_for_fused_verify(batch)

        self.assertIs(mapping, worker.hot_token_ids)
        self.assertIs(batch.spec_info_padded.topk_index, token_chain)
        worker.padding_for_decode.assert_called_once_with(
            batch,
            map_hot_token_ids=False,
        )

    def test_fused_verify_bootstrap_chain_is_already_mapped(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        worker = object.__new__(EagleDraftWorker)
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE3
        worker.speculative_num_steps = 3
        worker.topk = 1
        worker.hot_token_ids = object()
        worker.padding_for_decode = Mock()
        seed = jnp.array([[11], [21]], dtype=jnp.int32)
        mapped_chain = jnp.array([[101, 102, 103], [201, 202, 203]], dtype=jnp.int32)
        worker.draft_forward = Mock(return_value=(None, mapped_chain, None))
        batch = SimpleNamespace(
            spec_info_padded=SimpleNamespace(topk_index=seed),
        )

        mapping = worker.prepare_for_fused_verify(batch)

        self.assertIsNone(mapping)
        self.assertIs(batch.spec_info_padded.topk_index, mapped_chain)
        worker.padding_for_decode.assert_called_once_with(
            batch,
            map_hot_token_ids=True,
        )

    def test_fused_verify_maps_and_builds_recurrent_chain_in_one_jit(self):
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _build_chain_verify_arrays,
            _map_eagle3_token_ids,
        )

        @jax.jit
        def prepare_chain(verified_id, raw_chain, seq_lens, hot_token_ids):
            mapped_chain = _map_eagle3_token_ids(raw_chain, hot_token_ids)
            return _build_chain_verify_arrays(
                verified_id=verified_id,
                token_list=mapped_chain,
                seq_lens=seq_lens,
                num_verify_tokens=4,
                batch_size=2,
            )

        packed = prepare_chain(
            jnp.array([7, 8], dtype=jnp.int32),
            jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
            jnp.array([10, 20], dtype=jnp.int32),
            jnp.arange(100, 200, dtype=jnp.int32),
        )

        np.testing.assert_array_equal(
            np.asarray(packed[0]),
            np.array([7, 101, 102, 103, 8, 104, 105, 106], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(packed[1]),
            np.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.int32),
        )

    def test_eagle3_decode_metadata_uses_one_query_per_slot(self):
        from sgl_jax.srt.layers.attention.flashattention_backend import (
            FlashAttentionMetadata,
        )
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _make_eagle3_decode_metadata,
        )

        old_metadata = FlashAttentionMetadata(
            cu_q_lens=jnp.array([0, 1, 2], dtype=jnp.int32),
            cu_kv_lens=jnp.array([0, 8, 16], dtype=jnp.int32),
            page_indices=jnp.arange(16, dtype=jnp.int32),
            swa_page_indices=None,
            seq_lens=jnp.array([4, 0], dtype=jnp.int32),
            distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
            custom_mask=None,
        )

        metadata = _make_eagle3_decode_metadata(
            old_metadata,
            seq_lens=jnp.array([5, 0], dtype=jnp.int32),
            allocated_lens=jnp.array([8, 8], dtype=jnp.int32),
            page_size=1,
            dp_size=1,
        )

        np.testing.assert_array_equal(np.asarray(metadata.cu_q_lens), np.array([0, 1, 2]))
        np.testing.assert_array_equal(np.asarray(metadata.cu_kv_lens), np.array([0, 5, 5]))
        np.testing.assert_array_equal(np.asarray(metadata.distribution), np.array([0, 0, 1]))
        np.testing.assert_array_equal(np.asarray(metadata.seq_lens), np.array([5, 0]))

    def test_eagle3_recurrent_token_keeps_raw_id_for_cross_round_state(self):
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _eagle3_raw_and_mapped_token_from_logits,
        )

        logits = jnp.array(
            [
                [0.0, 1.0, 5.0, 2.0],
                [4.0, 1.0, 0.0, 2.0],
            ],
            dtype=jnp.float32,
        )
        hot_token_ids = jnp.array([100, 101, 102, 103], dtype=jnp.int32)

        raw, mapped = _eagle3_raw_and_mapped_token_from_logits(logits, hot_token_ids)

        np.testing.assert_array_equal(np.asarray(raw), np.array([2, 0], dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(mapped), np.array([102, 100], dtype=np.int32))

    def test_greedy_prepare_uses_original_seq_lens_for_new_seq_lens(self):
        from sgl_jax.srt.speculative.draft_extend_fused import _prepare_draft_inputs

        out = _prepare_draft_inputs(
            hidden_states=jnp.arange(12 * 2, dtype=jnp.float32).reshape(12, 2),
            positions=jnp.arange(12, dtype=jnp.int32),
            seq_lens=jnp.array([103, 303], dtype=jnp.int32),
            accept_index=jnp.array([0, -1, -1, -1, 8, 9, 10, 11], dtype=jnp.int32),
            accept_length=jnp.array([1, 4], dtype=jnp.int32),
            verified_id=jnp.arange(8, dtype=jnp.int32),
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )

        np.testing.assert_array_equal(
            np.asarray(out.new_seq_lens),
            np.array([105, 308], dtype=np.int32),
        )

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
