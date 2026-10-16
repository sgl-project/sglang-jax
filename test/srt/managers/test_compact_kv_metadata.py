"""Check compact KV metadata from scheduler assembly through device inputs."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest
from types import SimpleNamespace

import jax
import numpy as np

from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.managers.schedule_batch import ScheduleBatch, ScheduleReqsInfo
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.aot_workloads import InputContext, WorkloadInputBuilder
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

PAGE_SIZE = 4
CACHE_CAPACITY = 32


class CompactKVBatchTest(unittest.TestCase):
    def setUp(self):
        self.mesh = jax.sharding.Mesh(
            np.array(jax.devices()[:4]).reshape(2, 2),
            ("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
        )
        self.config = SimpleNamespace(
            is_in_model_multimodal=False,
            is_embedding=False,
            hf_config=SimpleNamespace(vision_encoder_parallel="tp", architectures=[]),
        )

    def _backends(self):
        return (
            FlashAttention(2, 2, 64, page_size=PAGE_SIZE, mesh=self.mesh),
            MLAAttentionBackend(2, 64, 64, 64, 64, page_size=PAGE_SIZE, mesh=self.mesh),
            NativeAttention(2, 2, self.mesh),
        )

    def _schedule_batch(self, mode, *, compact, empty_rank=False, spec=SpeculativeAlgorithm.NONE):
        pages = np.array([[8, 3], [11, 4], [5, 9]], dtype=np.int32)
        locations = (pages[:, :, None] * PAGE_SIZE + np.arange(PAGE_SIZE)).reshape(3, -1)
        infos = []
        for indices, lengths in (([0, 1], [6, 3]), ([2], [1])):
            seq = np.array(lengths, dtype=np.int32)
            queries = np.minimum(seq, 2) if mode == ForwardMode.EXTEND else np.ones_like(seq)
            prefixes = seq - queries
            output = np.concatenate(
                [locations[i, prefix:length] for i, prefix, length in zip(indices, prefixes, seq)]
            ).astype(np.int32)
            infos.append(
                ScheduleReqsInfo(
                    reqs=[SimpleNamespace(lora_id="0", mm_inputs=None) for _ in indices],
                    input_ids=np.ones(int(queries.sum()), dtype=np.int32),
                    req_pool_indices=np.array(indices, dtype=np.int32),
                    seq_lens=seq,
                    prefix_lens=prefixes.tolist(),
                    extend_lens=queries.tolist(),
                    out_cache_loc=output,
                )
            )
        if empty_rank:
            infos[1] = ScheduleReqsInfo(reqs=[])
        pool = ReqToTokenPool(size=3, max_context_len=8)
        pool.req_to_token[:] = locations
        pool.init_cache_loc_host_buffer(CACHE_CAPACITY)
        pool.supports_compact_kv_metadata = compact
        batch = ScheduleBatch.init_new(
            reqs=[[], []],
            dp_size=2,
            spec_algorithm=spec,
            model_config=self.config,
            mesh=self.mesh,
            enable_overlap=False,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            req_to_token_pool=pool,
        )
        batch.reqs_info = infos
        batch.forward_mode = mode
        return batch

    def _worker_batch(self, mode, **kwargs):
        batch = self._schedule_batch(mode, **kwargs)
        return batch.get_model_worker_batch([8], [4], [CACHE_CAPACITY], PAGE_SIZE)

    def _forward_batch(self, worker, backend):
        runner = SimpleNamespace(mesh=self.mesh, attn_backend=backend, model_config=self.config)
        return ForwardBatch.init_new(worker, runner)

    def test_compact_batches_preserve_metadata_and_token_writes(self):
        for backend in self._backends()[:2]:
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                for empty_rank in (False, True):
                    with self.subTest(backend=type(backend), mode=mode, empty_rank=empty_rank):
                        legacy = self._worker_batch(mode, compact=False, empty_rank=empty_rank)
                        compact = self._worker_batch(mode, compact=True, empty_rank=empty_rank)
                        self.assertIsNone(compact.cache_loc)
                        self.assertEqual(
                            compact.allocated_page_indices.size, CACHE_CAPACITY // PAGE_SIZE
                        )
                        np.testing.assert_array_equal(
                            compact.allocated_page_indices,
                            legacy.cache_loc[::PAGE_SIZE] // PAGE_SIZE,
                        )
                        np.testing.assert_array_equal(compact.out_cache_loc, legacy.out_cache_loc)
                        expected = backend.get_forward_metadata(legacy)
                        actual = backend.get_forward_metadata(compact)
                        for field in (
                            "page_indices",
                            "cu_q_lens",
                            "cu_kv_lens",
                            "seq_lens",
                            "distribution",
                        ):
                            np.testing.assert_array_equal(
                                getattr(actual, field), getattr(expected, field)
                            )
                        forward = self._forward_batch(compact, backend)
                        self.assertIsNone(forward.cache_loc)
                        np.testing.assert_array_equal(forward.out_cache_loc, legacy.out_cache_loc)

    def test_native_and_speculative_inputs_keep_token_locations(self):
        for backend, spec in (
            (self._backends()[2], SpeculativeAlgorithm.NONE),
            (self._backends()[0], SpeculativeAlgorithm.EAGLE3),
            (self._backends()[1], SpeculativeAlgorithm.EAGLE3),
        ):
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(backend=type(backend), mode=mode, spec=spec):
                    self.assertFalse(backend.use_compact_kv_metadata(mode, spec))
                    worker = self._worker_batch(
                        mode, compact=backend.supports_compact_kv_metadata, spec=spec
                    )
                    self.assertIsNone(worker.allocated_page_indices)
                    forward = self._forward_batch(worker, backend)
                    np.testing.assert_array_equal(forward.cache_loc, worker.cache_loc)
                    np.testing.assert_array_equal(forward.out_cache_loc, worker.out_cache_loc)

    def test_hybrid_preserves_full_and_recurrent_metadata(self):
        for full_backend in self._backends():
            hybrid = HybridLinearAttnBackend(
                full_backend, LinearRecurrentAttnBackend(self.mesh), full_attn_layers=[0]
            )
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(backend=type(full_backend), mode=mode):
                    compact = hybrid.use_compact_kv_metadata(mode, SpeculativeAlgorithm.NONE)
                    self.assertEqual(compact, not isinstance(full_backend, NativeAttention))
                    worker = self._worker_batch(mode, compact=compact)
                    legacy = self._worker_batch(mode, compact=False)
                    actual = hybrid.get_forward_metadata(worker)
                    expected = hybrid.get_forward_metadata(legacy)
                    self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
                    for got, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                        np.testing.assert_array_equal(got, ref)
                        self.assertEqual(got.sharding, ref.sharding)
                    forward = self._forward_batch(worker, hybrid)
                    self.assertEqual(forward.cache_loc is None, compact)

    def test_speculative_modes_keep_token_input_without_algorithm(self):
        for backend in self._backends()[:2]:
            for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND):
                with self.subTest(backend=type(backend), mode=mode):
                    worker = self._worker_batch(mode, compact=True, spec=None)
                    self.assertIsNone(worker.allocated_page_indices)
                    forward = self._forward_batch(worker, backend)
                    np.testing.assert_array_equal(forward.cache_loc, worker.cache_loc)

    def test_batch_rechecks_mode_and_algorithm_on_each_call(self):
        batch = self._schedule_batch(ForwardMode.EXTEND, compact=True)
        for mode, spec, compact in (
            (ForwardMode.EXTEND, SpeculativeAlgorithm.NONE, True),
            (ForwardMode.TARGET_VERIFY, SpeculativeAlgorithm.NONE, False),
            (ForwardMode.DRAFT_EXTEND, None, False),
            (ForwardMode.EXTEND, SpeculativeAlgorithm.EAGLE3, False),
            (ForwardMode.EXTEND, None, True),
        ):
            with self.subTest(mode=mode, spec=spec):
                batch.forward_mode = mode
                batch.spec_algorithm = spec
                worker = batch.get_model_worker_batch([8], [4], [CACHE_CAPACITY], PAGE_SIZE)
                self.assertEqual(worker.cache_loc is None, compact)
                self.assertEqual(worker.allocated_page_indices is not None, compact)

    def test_batch_uses_current_request_pool_capability(self):
        batch = self._schedule_batch(ForwardMode.EXTEND, compact=True)
        compact_pool = batch.req_to_token_pool
        token_pool = ReqToTokenPool(size=3, max_context_len=8)
        token_pool.req_to_token[:] = compact_pool.req_to_token
        token_pool.init_cache_loc_host_buffer(CACHE_CAPACITY)
        for pool, compact in ((compact_pool, True), (token_pool, False), (compact_pool, True)):
            with self.subTest(compact=compact):
                batch.req_to_token_pool = pool
                worker = batch.get_model_worker_batch([8], [4], [CACHE_CAPACITY], PAGE_SIZE)
                self.assertEqual(worker.cache_loc is None, compact)
                self.assertEqual(worker.allocated_page_indices is not None, compact)

    def test_pool_owner_sets_capability_and_shared_runner_preserves_it(self):
        fa, mla, native = self._backends()
        for backend in (fa, mla, native):
            with self.subTest(backend=type(backend)):
                runner = SimpleNamespace(
                    _validate_kv_pool_compatibility=lambda: None,
                    linear_recurrent_config=None,
                    req_to_token_pool=None,
                    model_config=SimpleNamespace(context_len=8),
                    attn_backend=backend,
                    _create_token_to_kv_pool=lambda dp_size, max_num_reqs: SimpleNamespace(),
                    token_to_kv_pool_allocator=object(),
                )
                ModelRunnerKVCacheMixin._init_pools(runner, max_num_reqs=4, dp_size=2)
                pool = runner.req_to_token_pool
                expected = backend.supports_compact_kv_metadata
                self.assertEqual(pool.supports_compact_kv_metadata, expected)
                # A draft runner can reuse the target's request pool with a different backend.
                runner.attn_backend = native if expected else fa
                ModelRunnerKVCacheMixin._init_pools(runner, max_num_reqs=4, dp_size=2)
                self.assertIs(runner.req_to_token_pool, pool)
                pool.clear()
                self.assertEqual(pool.supports_compact_kv_metadata, expected)

    def test_compact_batch_does_not_require_token_buffer(self):
        batch = self._schedule_batch(ForwardMode.EXTEND, compact=True)
        batch.req_to_token_pool.cache_loc_host_buf = None
        worker = batch.get_model_worker_batch([8], [4], [CACHE_CAPACITY], PAGE_SIZE)
        self.assertIsNone(worker.cache_loc)
        self.assertEqual(worker.allocated_page_indices.size, CACHE_CAPACITY // PAGE_SIZE)

    def test_warmup_and_aot_match_runtime_cache_layout(self):
        manager = SimpleNamespace(
            vocab_size=128,
            capture_hidden_states=False,
            has_recurrent_state=False,
            supports_recurrent_cow=False,
            supports_recurrent_track=False,
        )
        for backend in self._backends():
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(backend=type(backend), mode=mode):
                    compact = backend.use_compact_kv_metadata(mode, SpeculativeAlgorithm.NONE)
                    worker = self._worker_batch(mode, compact=compact)
                    runtime = self._forward_batch(worker, backend)
                    dummy = CompilationManager._make_dummy_batch(
                        manager,
                        4,
                        len(worker.input_ids),
                        mode,
                        CACHE_CAPACITY,
                        dp_size=2,
                        per_dp_bs_size=2,
                    )
                    warmup = self._forward_batch(dummy, backend)
                    builder = SimpleNamespace(
                        spec=SimpleNamespace(
                            context_length=8,
                            page_size=PAGE_SIZE,
                            request_count=4,
                            input_token_count=len(worker.input_ids),
                            cache_loc_size=CACHE_CAPACITY,
                        ),
                        forward_mode=mode,
                        spec_algorithm=worker.spec_algorithm,
                        capture_hidden_mode=worker.capture_hidden_mode,
                    )
                    context = InputContext(
                        self.config, self.mesh, backend, SimpleNamespace(recurrent_state_pool=None)
                    )
                    fields = {}
                    if mode == ForwardMode.EXTEND:
                        fields = dict(
                            extend_prefix_lens=context.vector(4), extend_seq_lens=context.vector(4)
                        )
                    aot = WorkloadInputBuilder._batch(builder, context, None, **fields)
                    self.assertEqual(jax.tree.structure(runtime), jax.tree.structure(warmup))
                    self.assertEqual(jax.tree.structure(runtime), jax.tree.structure(aot))
                    for batch in (runtime, warmup, aot):
                        self.assertEqual(batch.cache_loc is None, compact)
                        if not compact:
                            self.assertEqual(batch.cache_loc.shape, (CACHE_CAPACITY,))
                        self.assertEqual(batch.out_cache_loc.shape, worker.out_cache_loc.shape)


if __name__ == "__main__":
    unittest.main()
