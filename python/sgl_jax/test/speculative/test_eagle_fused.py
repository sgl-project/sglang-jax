import unittest
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.test.test_utils import CustomTestCase


class TestFusedEagle(CustomTestCase):
    def test_eagle_overlap_server_args_allow_linear_fa(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="target",
            speculative_algorithm="EAGLE",
            speculative_draft_model_path="draft",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
            attention_backend="fa",
            disable_overlap_schedule=False,
            grammar_backend="none",
        )

        args.check_server_args()

    def test_eagle3_overlap_server_args_allow_linear_fa(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="target",
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path="draft",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
            attention_backend="fa",
            disable_overlap_schedule=False,
            grammar_backend="none",
        )

        args.check_server_args()

    def test_nextn_overlap_server_args_allow_fused_linear_fa(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="target",
            speculative_algorithm="NEXTN",
            speculative_draft_model_path="draft",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
            attention_backend="fa",
            disable_overlap_schedule=False,
            grammar_backend="none",
        )

        args.check_server_args()

    def test_eagle3_overlap_server_args_reject_non_fa(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="target",
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path="draft",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
            attention_backend="native",
            disable_overlap_schedule=False,
            grammar_backend="none",
        )

        with self.assertRaisesRegex(ValueError, "EAGLE3\\+FA"):
            args.check_server_args()

    def test_eagle3_server_args_reject_topk_greater_than_one(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="target",
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path="draft",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=2,
            attention_backend="fa",
            grammar_backend="none",
        )

        with self.assertRaisesRegex(ValueError, "topk=1"):
            args.check_server_args()

    def test_eagle_server_args_reject_topk_greater_than_one(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="target",
            speculative_algorithm="EAGLE",
            speculative_draft_model_path="draft",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=2,
            attention_backend="fa",
            grammar_backend="none",
        )

        with self.assertRaisesRegex(ValueError, "topk=1"):
            args.check_server_args()

    def test_supported_speculative_algorithms_are_parseable(self):
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertEqual(
            SpeculativeAlgorithm.from_string("EAGLE"),
            SpeculativeAlgorithm.EAGLE,
        )
        self.assertTrue(SpeculativeAlgorithm.EAGLE.is_eagle_family())
        self.assertTrue(SpeculativeAlgorithm.EAGLE3.is_eagle_family())
        self.assertEqual(
            SpeculativeAlgorithm.from_string("NEXTN"),
            SpeculativeAlgorithm.NEXTN,
        )
        self.assertTrue(SpeculativeAlgorithm.NEXTN.is_eagle())
        self.assertTrue(SpeculativeAlgorithm.NEXTN.is_nextn())
        self.assertFalse(SpeculativeAlgorithm.NEXTN.is_eagle_family())

        with self.assertRaises(KeyError):
            SpeculativeAlgorithm.from_string("STANDALONE")

    def test_nextn_rejects_non_linear_topk(self):
        from sgl_jax.srt.server_args import ServerArgs

        args = ServerArgs(
            model_path="target",
            speculative_algorithm="NEXTN",
            speculative_draft_model_path="draft",
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=2,
            attention_backend="fa",
            grammar_backend="none",
        )

        with self.assertRaisesRegex(ValueError, "NEXTN requires.*topk=1"):
            args.check_server_args()

    def test_nextn_multi_layer_verify_requires_full_chain(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.multi_layer_draft_worker import (
            MultiLayerDraftWorker,
        )

        worker = object.__new__(MultiLayerDraftWorker)
        worker.speculative_num_steps = 3
        worker.padding_for_decode = Mock()
        token_chain = jnp.array([[11, 12, 13], [21, 22, 23]], dtype=jnp.int32)
        batch = SimpleNamespace(
            spec_info_padded=SimpleNamespace(topk_index=token_chain),
        )

        mapping = worker.prepare_for_fused_verify(batch)

        self.assertIsNone(mapping)
        worker.padding_for_decode.assert_called_once_with(batch)

        batch.spec_info_padded.topk_index = token_chain[:, :1]
        with self.assertRaisesRegex(ValueError, "precomputed token chain"):
            worker.prepare_for_fused_verify(batch)

    def test_nextn_input_rotation_builds_linear_layer_chain(self):
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _rotate_mtp_decode_input_ids,
            _rotate_mtp_prefill_input_ids,
        )

        prefill = _rotate_mtp_prefill_input_ids(
            jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32),
            jnp.array([3, 2], dtype=jnp.int32),
            jnp.array([9, 8], dtype=jnp.int32),
            dp_size=1,
            per_dp_bs=2,
        )
        np.testing.assert_array_equal(
            np.asarray(prefill),
            np.array([2, 3, 9, 5, 8], dtype=np.int32),
        )

        decode = _rotate_mtp_decode_input_ids(
            jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32),
            jnp.array([3, 2], dtype=jnp.int32),
            jnp.array([2, 1], dtype=jnp.int32),
            jnp.array([9, 8], dtype=jnp.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(decode),
            np.array([2, 3, 9, 4, 6, 8, 8, 8], dtype=np.int32),
        )

    def test_as_int32_array_keeps_host_metadata_on_host(self):
        from sgl_jax.srt.speculative import eagle_info

        arr = eagle_info._as_int32_array(np.array([1, 2], dtype=np.int64))
        scalar = eagle_info._as_int32_array(3)
        listed = eagle_info._as_int32_array([4, 5])
        children, _ = eagle_info.EagleDraftInput().tree_flatten()

        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.dtype, np.int32)
        np.testing.assert_array_equal(arr, np.array([1, 2], dtype=np.int32))
        self.assertIsInstance(scalar, np.ndarray)
        self.assertEqual(scalar.dtype, np.int32)
        np.testing.assert_array_equal(listed, np.array([4, 5], dtype=np.int32))
        self.assertIsNone(children[5])
        self.assertIsInstance(children[6], np.ndarray)
        self.assertEqual(children[6].dtype, np.int32)
        self.assertEqual(children[6].shape, (0,))

        device_arr = jnp.array([6], dtype=jnp.int32)
        self.assertIs(eagle_info._as_int32_array(device_arr), device_arr)

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
        logits = np.full((target_predict.size, 128), -1.0, dtype=np.float32)
        logits[np.arange(target_predict.size), target_predict] = 10.0
        chain = _verify_greedy(
            target_hidden=jnp.arange(bs * num_draft_tokens * 2, dtype=jnp.float32).reshape(
                bs * num_draft_tokens, 2
            ),
            positions=jnp.arange(bs * num_draft_tokens, dtype=jnp.int32),
            seq_lens=jnp.array([100, 200, 300, 400], dtype=jnp.int32),
            draft_tokens=draft_tokens,
            target_logits=jnp.asarray(logits),
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

        draft_tokens = jnp.array(
            [0, 0, 0, 0, 20, 21, 22, 23],
            dtype=jnp.int32,
        )
        target_predict = np.array([0, 0, 0, 0, 21, 22, 99, 24], dtype=np.int32)
        logits = np.full((target_predict.size, 128), -1.0, dtype=np.float32)
        logits[np.arange(target_predict.size), target_predict] = 10.0
        seq_lens = jnp.array([0, 10], dtype=jnp.int32)
        out = _verify_greedy(
            target_hidden=jnp.arange(8 * 2, dtype=jnp.float32).reshape(8, 2),
            positions=jnp.arange(8, dtype=jnp.int32),
            seq_lens=seq_lens,
            draft_tokens=draft_tokens,
            target_logits=jnp.asarray(logits),
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )

        np.testing.assert_array_equal(np.asarray(out.accept_lens), np.array([0, 3]))
        np.testing.assert_array_equal(np.asarray(out.new_seq_lens), np.array([1, 14]))
        np.testing.assert_array_equal(np.asarray(out.sel_pos), np.array([0, 2]))

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
        batch = SimpleNamespace(
            spec_info_padded=SimpleNamespace(topk_index=token_chain),
        )

        mapping = worker.prepare_for_fused_verify(batch)

        self.assertIs(mapping, worker.hot_token_ids)
        self.assertIs(batch.spec_info_padded.topk_index, token_chain)
        worker.padding_for_decode.assert_called_once_with(batch)

    def test_classic_eagle_fused_verify_uses_target_vocabulary_directly(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        worker = object.__new__(EagleDraftWorker)
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        worker.speculative_num_steps = 3
        worker.topk = 1
        worker.hot_token_ids = None
        worker.padding_for_decode = Mock()
        token_chain = jnp.array([[11, 12, 13], [21, 22, 23]], dtype=jnp.int32)
        batch = SimpleNamespace(
            spec_info_padded=SimpleNamespace(topk_index=token_chain),
        )

        mapping = worker.prepare_for_fused_verify(batch)

        self.assertIsNone(mapping)
        self.assertIs(batch.spec_info_padded.topk_index, token_chain)
        worker.padding_for_decode.assert_called_once_with(batch)

    def test_classic_eagle_shares_target_embedding_and_head(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        embed = object()
        head = object()
        target_model = Mock()
        target_model.get_embed_and_head.return_value = (embed, head)
        draft_model = Mock()
        draft_runner = SimpleNamespace(model=draft_model)
        worker = object.__new__(EagleDraftWorker)
        worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
        worker.hot_token_ids = None
        worker._worker = Mock()
        worker._worker.get_model_runner.return_value = draft_runner
        target_worker = SimpleNamespace(model_runner=SimpleNamespace(model=target_model))

        worker._share_embed_head(target_worker)

        draft_model.set_embed_and_head.assert_called_once_with(embed, head)
        draft_model.set_embed.assert_not_called()
        self.assertIsNone(worker.hot_token_ids)

    def test_classic_eagle_enables_only_the_fused_linear_worker_path(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_algorithm="EAGLE",
            page_size=64,
            attention_backend="fa",
        )
        target_worker = Mock()
        target_worker.mesh = object()
        target_worker.get_memory_pool.return_value = (object(), object())
        target_worker.get_precompile_paddings.return_value = ([128], [1], [128])

        worker = BaseSpecWorker(server_args, target_worker, Mock())

        self.assertTrue(worker._can_use_fused_eagle_verify)

    def test_nextn_enables_only_the_fused_multi_layer_path(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.base_worker import BaseSpecWorker

        server_args = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            speculative_algorithm="NEXTN",
            page_size=64,
            attention_backend="fa",
        )
        target_worker = Mock()
        target_worker.mesh = object()
        target_worker.get_memory_pool.return_value = (object(), object())
        target_worker.get_precompile_paddings.return_value = ([128], [1], [128])

        worker = BaseSpecWorker(server_args, target_worker, Mock())

        self.assertFalse(worker._can_use_fused_eagle_verify)
        self.assertTrue(worker._can_use_fused_mtp_verify)

    def test_classic_eagle_non_overlap_uses_host_round_state(self):
        from sgl_jax.srt.speculative.overlap_utils import uses_host_eagle_state
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.assertTrue(uses_host_eagle_state(False, SpeculativeAlgorithm.EAGLE))
        self.assertFalse(uses_host_eagle_state(True, SpeculativeAlgorithm.EAGLE))

    def test_fused_verify_bootstraps_raw_chain_and_returns_mapping(self):
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
        raw_chain = jnp.array([[11, 12, 13], [21, 22, 23]], dtype=jnp.int32)
        batch = SimpleNamespace(
            spec_info_padded=SimpleNamespace(topk_index=seed),
        )

        with patch(
            "sgl_jax.srt.speculative.draft_extend_fused.bootstrap_eagle_chain",
            return_value=raw_chain,
        ) as bootstrap:
            mapping = worker.prepare_for_fused_verify(batch)

        self.assertIs(mapping, worker.hot_token_ids)
        self.assertIs(batch.spec_info_padded.topk_index, raw_chain)
        worker.padding_for_decode.assert_called_once_with(batch)
        bootstrap.assert_called_once_with(worker, batch)

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

    def test_fused_chain_builder_aligns_replicated_bootstrap_to_data(self):
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.speculative.draft_extend_fused import (
            _build_chain_verify_arrays,
        )

        mesh = Mesh(
            np.asarray(jax.devices()),
            ("data",),
            axis_types=(jax.sharding.AxisType.Explicit,),
        )
        verified_id = jax.device_put(
            jnp.array([7, 8], dtype=jnp.int32),
            NamedSharding(mesh, P("data")),
        )
        token_list = jax.device_put(
            jnp.array([[11, 12, 13], [21, 22, 23]], dtype=jnp.int32),
            NamedSharding(mesh, P(None, None)),
        )
        seq_lens = jax.device_put(
            jnp.array([10, 20], dtype=jnp.int32),
            NamedSharding(mesh, P("data")),
        )

        build = jax.jit(
            lambda verified, tokens, lengths: _build_chain_verify_arrays(
                verified_id=verified,
                token_list=tokens,
                seq_lens=lengths,
                num_verify_tokens=4,
                batch_size=2,
            )
        )
        draft_tokens, *_ = build(verified_id, token_list, seq_lens)

        np.testing.assert_array_equal(
            np.asarray(draft_tokens),
            np.array([7, 11, 12, 13, 8, 21, 22, 23], dtype=np.int32),
        )

    def test_fused_verify_reuses_device_placeholders(self):
        from types import SimpleNamespace

        from jax.sharding import Mesh

        from sgl_jax.srt.speculative.draft_extend_fused import _prepare_verify
        from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

        mesh = Mesh(np.asarray(jax.devices()), ("data",))
        worker = SimpleNamespace(
            mesh=mesh,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
        )

        def make_draft_input():
            return EagleDraftInput(
                verified_id=np.array([7, 8], dtype=np.int32),
                topk_index=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
            )

        batch = SimpleNamespace(
            seq_lens=np.array([10, 20], dtype=np.int32),
            spec_info_padded=make_draft_input(),
        )
        _prepare_verify(worker, batch, draft_padding_prepared=True)
        first = batch.spec_info_padded

        batch.spec_info_padded = make_draft_input()
        _prepare_verify(worker, batch, draft_padding_prepared=True)

        self.assertIs(batch.spec_info_padded, first)

    def test_fused_verify_relay_skips_eager_hot_token_mapping(self):
        from types import SimpleNamespace

        from jax.sharding import Mesh

        from sgl_jax.srt.speculative.draft_extend_fused import _prepare_verify
        from sgl_jax.srt.speculative.eagle_info import EagleDraftInput

        calls = []
        worker = SimpleNamespace(
            mesh=Mesh(np.asarray(jax.devices()), ("data",)),
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            model_config=SimpleNamespace(hidden_size=8),
            padding_for_decode=lambda _batch: calls.append(True),
        )
        batch = SimpleNamespace(
            seq_lens=np.array([10, 20], dtype=np.int32),
            spec_info_padded=EagleDraftInput(future_indices=np.array([3, 5], dtype=np.int32)),
        )

        _prepare_verify(worker, batch)

        self.assertEqual(calls, [True])

    def test_target_verify_logits_metadata_has_no_unused_device_inputs(self):
        from types import SimpleNamespace

        from sgl_jax.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardMode,
        )
        from sgl_jax.srt.speculative.draft_extend_fused import _prepare_logits_metadata

        metadata = _prepare_logits_metadata(
            SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            ),
            mesh=None,
        )

        self.assertIsNone(metadata.extend_seq_lens)
        self.assertIsNone(metadata.logits_indices)
        self.assertIsNone(metadata.accept_lens)

    def test_paged_kv_layout_uploads_only_page_ids(self):
        from types import SimpleNamespace

        from jax.sharding import Mesh

        from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
        from sgl_jax.srt.layers.attention.flashattention_metadata import PagedKVLayout

        backend = SimpleNamespace(
            page_size=2,
            mesh=Mesh(np.asarray(jax.devices()), ("data",)),
            swa_index_mapping=None,
        )
        batch = SimpleNamespace(
            cache_loc=np.array([0, 0, 4, 0, 8, 0, 12, 0], dtype=np.int32),
            dp_size=1,
            per_dp_bs_size=2,
        )

        layout = FlashAttention.prepare_paged_kv_layout(backend, batch)

        self.assertIsInstance(layout, PagedKVLayout)
        np.testing.assert_array_equal(
            np.asarray(layout.page_indices)[:4],
            np.array([0, 2, 4, 6], dtype=np.int32),
        )
        self.assertFalse(hasattr(layout, "cu_q_lens"))

    def test_no_overlap_reuses_cache_loc_buffer_without_clearing(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker

        worker = object.__new__(EagleDraftWorker)
        worker.server_args = SimpleNamespace(disable_overlap_schedule=True)
        first = worker._get_decode_cache_loc_buffer(16)
        first[3] = 99

        second = worker._get_decode_cache_loc_buffer(16)

        self.assertIs(second, first)
        self.assertEqual(second[3], 99)

    def test_no_overlap_only_copies_padding_inputs_to_host(self):
        from types import SimpleNamespace

        from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker

        worker = object.__new__(EagleDraftWorker)
        worker.server_args = SimpleNamespace(disable_overlap_schedule=True)
        untouched = jnp.arange(4, dtype=jnp.int32)
        batch = SimpleNamespace(
            input_ids=untouched,
            seq_lens=jnp.array([8], dtype=jnp.int32),
            out_cache_loc=untouched,
            positions=untouched,
            req_pool_indices=jnp.array([3], dtype=jnp.int32),
            cache_loc=untouched,
            extend_prefix_lens=untouched,
            extend_seq_lens=untouched,
        )

        worker.copy_model_worker_batch_to_cpu(batch)

        self.assertIsInstance(batch.seq_lens, np.ndarray)
        self.assertIsInstance(batch.req_pool_indices, np.ndarray)
        self.assertIs(batch.input_ids, untouched)
        self.assertIs(batch.cache_loc, untouched)
        self.assertIs(batch.extend_seq_lens, untouched)

    def test_draft_forward_metadata_uses_one_query_per_slot(self):
        from sgl_jax.srt.layers.attention.flashattention_metadata import (
            PagedKVLayout,
            build_draft_forward_metadata,
        )

        layout = PagedKVLayout(
            page_indices=jnp.arange(16, dtype=jnp.int32),
            swa_page_indices=None,
        )

        metadata = build_draft_forward_metadata(
            layout,
            seq_lens=jnp.array([5, 0], dtype=jnp.int32),
            allocated_lens=jnp.array([8, 8], dtype=jnp.int32),
            page_size=1,
            dp_size=1,
        )

        np.testing.assert_array_equal(np.asarray(metadata.cu_q_lens), np.array([0, 1, 2]))
        np.testing.assert_array_equal(np.asarray(metadata.cu_kv_lens), np.array([0, 5, 5]))
        np.testing.assert_array_equal(np.asarray(metadata.distribution), np.array([0, 0, 1]))
        np.testing.assert_array_equal(np.asarray(metadata.seq_lens), np.array([5, 0]))

    def test_target_verify_metadata_materializes_page_layout(self):
        from sgl_jax.srt.layers.attention.flashattention_metadata import (
            PagedKVLayout,
            build_target_verify_metadata,
        )

        metadata = build_target_verify_metadata(
            PagedKVLayout(
                page_indices=jnp.arange(16, dtype=jnp.int32),
                swa_page_indices=None,
            ),
            prefix_lens=jnp.array([4, 0], dtype=jnp.int32),
            allocated_lens=jnp.array([8, 8], dtype=jnp.int32),
            draft_width=4,
            page_size=1,
            dp_size=1,
        )

        np.testing.assert_array_equal(np.asarray(metadata.cu_q_lens), np.array([0, 4, 4]))
        np.testing.assert_array_equal(np.asarray(metadata.cu_kv_lens), np.array([0, 8, 8]))
        np.testing.assert_array_equal(np.asarray(metadata.seq_lens), np.array([8, 0]))
        np.testing.assert_array_equal(np.asarray(metadata.distribution), np.array([0, 1, 1]))

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


if __name__ == "__main__":
    unittest.main()
