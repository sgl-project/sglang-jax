"""Unit tests for AotDispatcher (CPU)."""

import os
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

os.environ["SGLANG_JAX_AOT_DISPATCH"] = "1"  # force-on regardless of arg count

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.model_executor.aot_dispatch import AotDispatcher


class TestAotDispatcher(unittest.TestCase):
    def test_batch_metadata_selects_distinct_executable(self):
        @jax.tree_util.register_pytree_node_class
        class Batch:
            def __init__(self, value, decode):
                self.value, self.decode = value, decode

            def tree_flatten(self):
                return (self.value,), self.decode

            @classmethod
            def tree_unflatten(cls, decode, children):
                return cls(children[0], decode)

        @jax.jit
        def f(batch):
            return batch.value + (1 if batch.decode else 10)

        disp = AotDispatcher(f, stable_call_args=(), stable_flat_args=(), name="metadata")
        for decode in (True, False, True, False):
            batch = Batch(jnp.ones(4), decode)
            np.testing.assert_array_equal(np.asarray(disp(batch)), np.asarray(f(batch)))

    def test_python_scalar_types_select_distinct_executable(self):
        @jax.jit
        def f(value):
            return value + 1

        disp = AotDispatcher(f, stable_call_args=(), stable_flat_args=(), name="scalar")
        for value in (2, 2.5, 3, 3.5):
            result = disp(value)
            expected = f(value)
            self.assertEqual(result.dtype, expected.dtype)
            np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def test_saved_sampler_and_logprobs_without_compiling(self):
        from flax import nnx

        from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
        from sgl_jax.srt.layers.sampler import (
            Sampler,
            jitted_compute_logprobs,
            make_jitted_sampler,
        )
        from sgl_jax.srt.model_executor.aot_executable import ExecutableStore
        from sgl_jax.srt.model_executor.aot_inputs import AbstractSampler
        from sgl_jax.srt.model_executor.aot_server import _export_sampling
        from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
        from sgl_jax.srt.utils.mesh_utils import create_device_mesh

        mesh = create_device_mesh([1, 1], [1, 1], devices=jax.devices()[:1])
        manager = object.__new__(CompilationManager)
        manager.vocab_size = 128
        manager.enable_static_lora = False
        manager.capture_hidden_states = False
        manager.has_recurrent_state = False
        manager.supports_recurrent_cow = False
        manager.supports_recurrent_track = False
        batch = manager._make_dummy_batch(4, 4, ForwardMode.DECODE, 4)
        logits = LogitsProcessorOutput(
            jax.device_put(
                np.random.default_rng(7).normal(size=(4, 128)).astype(np.float32),
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", "tensor")),
            )
        )
        graph, state = nnx.split(Sampler(nnx.Rngs(42), mesh=mesh))
        leaves, state_def = jax.tree_util.tree_flatten(state)
        fn = make_jitted_sampler(jax.random.PRNGKey(42))
        stable = (graph, state_def, leaves)
        step = jax.device_put(
            np.int32(0), jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"sampling": []}
            _export_sampling(
                AbstractSampler(mesh, 42),
                jax.tree.map(
                    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), logits
                ),
                manager,
                mesh,
                None,
                Path(directory),
                manifest,
            )
            self.assertEqual(len(manifest["sampling"]), 3)
            store = ExecutableStore(directory, mesh)
            cases = []
            for seeded in (False, True):
                batch.sampling_info.sampling_seeds = (
                    np.arange(4, dtype=np.int32) if seeded else None
                )
                for greedy in (False, True):
                    batch.sampling_info.is_all_greedy = greedy
                    batch.sampling_info.linear_penalty = np.full((4, 128), 0.25, dtype=np.float32)
                    metadata = SamplingMetadata.from_model_worker_batch(batch, 0, mesh, 128)
                    metadata.update_vocab_mask(np.full((4, 4), -1, dtype=np.int32), mesh, 128)
                    result = fn(*stable, step, logits, metadata)
                    (tokens, logprobs, _), next_step = result
                    selected = jitted_compute_logprobs(mesh, logprobs, tokens)
                    cases.append((metadata, result, selected))
                    self.assertEqual(int(next_step), 1)
            jax.clear_caches()
            for fast in (False, True):
                sample = AotDispatcher(
                    fn,
                    stable,
                    (graph, leaves),
                    "sampler",
                    executable_store=store,
                    allow_fast_dispatch=fast,
                )
                logprob = AotDispatcher(
                    jitted_compute_logprobs,
                    (mesh,),
                    (),
                    "logprobs",
                    executable_store=store,
                    allow_fast_dispatch=fast,
                )
                with patch(
                    "jax._src.compiler.backend_compile_and_load",
                    side_effect=AssertionError("Unexpected JIT"),
                ):
                    for metadata, expected, selected in cases:
                        actual = sample(step, logits, metadata)
                        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
                            np.testing.assert_array_equal(a, b)
                        (tokens, logprobs, _), _ = actual
                        np.testing.assert_array_equal(logprob(logprobs, tokens), selected)

    def test_saved_executable_shares_dispatch_cache_without_compiling(self):
        from sgl_jax.srt.model_executor.aot_executable import ExecutableStore
        from sgl_jax.srt.model_executor.compilation_manager import CompilationManager

        @partial(jax.jit, static_argnums=(1,), donate_argnums=(3,))
        def f(weights, scale, batch, pool, unused):
            return {"output": weights["w"] * scale + batch, "pool": pool + 1}

        mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("x",))
        weights = {"unused": jnp.zeros(3), "w": jnp.arange(4.0)}
        batch = jnp.ones(4)
        unused = jnp.zeros(7)
        pools = [jnp.zeros(4) for _ in range(3)]
        lowered = f.lower(weights, 2, batch, pools[0], unused)
        with tempfile.TemporaryDirectory() as directory:
            CompilationManager.get_executable(lowered, mesh, output=Path(directory))
            store = ExecutableStore(directory, mesh)
            dispatcher = AotDispatcher(
                f,
                stable_call_args=(weights, 2),
                stable_flat_args=(weights,),
                name="offline",
                executable_store=store,
            )
            # Explicit loading works with optional fast dispatch disabled too.
            with (
                patch("sgl_jax.srt.model_executor.aot_dispatch._ENV", "0"),
                patch(
                    "jax._src.compiler.backend_compile_and_load",
                    side_effect=AssertionError,
                ),
                patch.object(store, "load", wraps=store.load) as load,
            ):
                for pool in pools:
                    result = dispatcher(batch, pool, unused)
                    np.testing.assert_array_equal(result["output"], np.arange(4.0) * 2 + 1)
                    np.testing.assert_array_equal(result["pool"], np.ones(4))
                self.assertEqual(load.call_count, 1)
                self.assertEqual(len(dispatcher._cache), 1)
                with patch("sgl_jax.srt.model_executor.aot_dispatch._ENV", "1"):
                    dispatcher.invalidate()
                    for _ in range(2):
                        result = dispatcher(batch, result["pool"], unused)
                        np.testing.assert_array_equal(result["output"], np.arange(4.0) * 2 + 1)
                self.assertEqual(load.call_count, 2)
                # A distinct model/static value must never fall back to compilation.
                incompatible = AotDispatcher(
                    f, (weights, 3), (weights,), "mismatch", executable_store=store
                )
                with self.assertRaisesRegex(ValueError, "No matching AOT"):
                    incompatible(batch, result["pool"], unused)

    def _make(self):
        @partial(jax.jit, static_argnames=["state_def", "flag"], donate_argnames=["pool"])
        def f(weights_def, state_def, leaves, flag, batch, pool, meta):
            scale = 2.0 if flag else 1.0
            w = weights_def["w"] + leaves[0]
            return batch * scale + w + meta, pool + 1.0

        weights_def = {"w": jnp.arange(4.0)}
        leaves = [jnp.ones(4) * 3]
        disp = AotDispatcher(
            f,
            stable_call_args=(weights_def, "STATE", leaves, True),
            stable_flat_args=(weights_def, leaves),
            name="test",
        )

        def ref(batch, pool, meta):
            return f(weights_def, "STATE", leaves, True, batch, pool, meta)

        return disp, ref

    def test_matches_checked_path_across_calls(self):
        disp, ref = self._make()
        batch = jnp.ones(4)
        meta = jnp.float32(0.5)
        # first call (checked path) and steady-state calls agree with pjit
        for step in range(3):
            out, new_pool = disp(batch + step, jnp.zeros(4), meta)
            eout, _ = ref(batch + step, jnp.zeros(4), meta)
            np.testing.assert_allclose(np.asarray(out), np.asarray(eout))

    def test_multiple_shape_keys(self):
        disp, ref = self._make()
        for n in (4, 4, 4):
            out, _ = disp(jnp.ones(n), jnp.zeros(n), jnp.float32(1.0))
        # a second shape gets its own entry and still matches
        out8, _ = disp(jnp.ones(4) * 8, jnp.zeros(4), jnp.float32(2.0))
        eout8, _ = ref(jnp.ones(4) * 8, jnp.zeros(4), jnp.float32(2.0))
        np.testing.assert_allclose(np.asarray(out8), np.asarray(eout8))
        self.assertEqual(len(disp._cache), 1)  # same shapes -> one entry

    def test_stable_replacement_invalidates(self):
        disp, ref = self._make()
        disp(jnp.ones(4), jnp.zeros(4), jnp.float32(0.0))
        n_before = len(disp._cache)
        self.assertGreaterEqual(n_before, 1)
        disp.invalidate()
        self.assertEqual(len(disp._cache), 0)
        out, _ = disp(jnp.ones(4), jnp.zeros(4), jnp.float32(0.0))
        eout, _ = ref(jnp.ones(4), jnp.zeros(4), jnp.float32(0.0))
        np.testing.assert_allclose(np.asarray(out), np.asarray(eout))

    def test_rebound_stable_list_via_ensure_stable_args(self):
        """LoRA-style reload: caller rebinds the leaves list to a new object;
        ensure_stable_args must drop the cache so new weights take effect."""

        @partial(jax.jit, static_argnames=["state_def"])
        def f(weights_def, state_def, leaves, batch):
            return batch + weights_def["w"] + leaves[0]

        weights_def = {"w": jnp.arange(4.0)}
        leaves = [jnp.ones(4) * 3]
        disp = AotDispatcher(
            f,
            stable_call_args=(weights_def, "STATE", leaves),
            stable_flat_args=(weights_def, leaves),
            name="test-rebind",
        )
        disp.ensure_stable_args((weights_def, "STATE", leaves), (weights_def, leaves))
        out1 = disp(jnp.zeros(4))
        np.testing.assert_allclose(np.asarray(out1), np.arange(4.0) + 3)

        # steady-state call (cached executable), then rebind to new leaves
        out1b = disp(jnp.zeros(4))
        np.testing.assert_allclose(np.asarray(out1b), np.arange(4.0) + 3)
        new_leaves = [jnp.ones(4) * 10]  # new list object, new weights
        disp.ensure_stable_args((weights_def, "STATE", new_leaves), (weights_def, new_leaves))
        self.assertEqual(len(disp._cache), 0)
        out2 = disp(jnp.zeros(4))
        np.testing.assert_allclose(np.asarray(out2), np.arange(4.0) + 10)
        # unchanged containers -> no-op, cache preserved
        n = len(disp._cache)
        disp.ensure_stable_args((weights_def, "STATE", new_leaves), (weights_def, new_leaves))
        self.assertEqual(len(disp._cache), n)


if __name__ == "__main__":
    unittest.main()
