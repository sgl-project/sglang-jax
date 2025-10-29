import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.eagle_util import (
    build_tree_mask_for_draft_decode,
    create_extend_after_decode_spec_info,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)
from sgl_jax.srt.speculative.pallas.tree_speculative_sampling_target_only_kernel import (
    tree_speculative_sampling_target_only_pallas_call,
)
from sgl_jax.test.test_utils import CustomTestCase

from sgl_jax.srt.speculative.pallas.verify_tree_greedy_kernel import verify_tree_greedy
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
class TestVerifyTree(CustomTestCase):
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

        target_predict = jnp.argmax(target_logits, axis=-1).flatten()
        predict_shape = (12,)

        bs = candidates.shape[0]
        num_spec_step = 4

        predicts = jnp.empty(predict_shape, dtype=jnp.int32)  # mutable
        accept_index = jnp.full((bs, num_spec_step), -1, dtype=jnp.int32)  # mutable
        accept_token_num = jnp.full((bs,), 0, dtype=jnp.int32)  # mutable

        # # for compatibility, 0.6.3 need to use use_mesh. set_mesh is not have __entry__ attribute.
        # # on jax >=0.7.1, we need to use set_mesh.
        try:
            ctx = jax.sharding.use_mesh(mesh)
        except AttributeError:
            try:
                ctx = jax.set_mesh(mesh)
            except AttributeError:
                ctx = mesh
        with ctx:
            accept_index, accept_token_num, predicts = verify_tree_greedy(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrive_index=retrive_index,
                retrive_next_token=retrive_next_token,
                retrive_next_sibling=retrive_next_sibling,
                target_predict=target_predict,
            )

        # Check the expected output.
        self.assertEqual(predicts.flatten().tolist(), [3, 0, 0, 4, 5, 18, 11, 0, 0, 0, 12, 18])
        self.assertEqual(accept_index.tolist(), [[0, 3, 4, 5], [6, 10, 11, -1]])
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
