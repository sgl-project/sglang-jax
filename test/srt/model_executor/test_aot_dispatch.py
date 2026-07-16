"""Unit tests for AotDispatcher (CPU)."""

import os
import unittest
from functools import partial

os.environ["SGLANG_JAX_AOT_DISPATCH"] = "1"  # force-on regardless of arg count

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.model_executor.aot_dispatch import AotDispatcher


class TestAotDispatcher(unittest.TestCase):
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
        ref = lambda batch, pool, meta: f(weights_def, "STATE", leaves, True, batch, pool, meta)
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
