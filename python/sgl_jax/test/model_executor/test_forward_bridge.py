"""_forward dict-or-list bridge for Phase 3 incremental model migration.

During Phase 3, models migrate from returning bare list[jax.Array]
(Phase 2 behavior) to returning dict[pool_name, list[jax.Array]] (Phase 3
target). _forward must handle both shapes so each per-model migration is
an independent, non-breaking commit.

Once all 14 models return dict, Phase 3 Task 5 deletes the legacy branch.
"""

import unittest

import jax.numpy as jnp


class TestForwardBridgeAcceptsBothShapes(unittest.TestCase):
    """The bridge in _forward maps either:
      - layers_kv_fused (list) -> {"token_to_kv_pool": list}  (legacy)
      - layers_kv_fused (dict) -> dict directly                (new)

    We test the helper extraction without spinning up a full ModelRunner.
    """

    def test_dict_passes_through(self):
        from sgl_jax.srt.model_executor.model_runner import _wrap_pool_updates

        d = {"token_to_kv_pool": [jnp.zeros((2,))]}
        out = _wrap_pool_updates(d)
        # Same dict instance returned; no copying.
        self.assertIs(out, d)

    def test_list_gets_wrapped(self):
        from sgl_jax.srt.model_executor.model_runner import _wrap_pool_updates

        lst = [jnp.zeros((2,)), jnp.zeros((2,))]
        out = _wrap_pool_updates(lst)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out.keys()), {"token_to_kv_pool"})
        self.assertIs(out["token_to_kv_pool"], lst)

    def test_tuple_also_wraps_as_token_to_kv_pool(self):
        """Some JIT outputs come back as tuple instead of list; treat same as list."""
        from sgl_jax.srt.model_executor.model_runner import _wrap_pool_updates

        tup = (jnp.zeros((2,)), jnp.zeros((2,)))
        out = _wrap_pool_updates(tup)
        self.assertIsInstance(out, dict)
        self.assertIs(out["token_to_kv_pool"], tup)


class TestMockModelRunnerHasMemoryPools(unittest.TestCase):
    """D5: MockModelRunner.__init__ must set self.memory_pools so its inherited
    _forward can call self.memory_pools.replace_all(...)."""

    def test_mock_model_runner_class_init_sets_memory_pools(self):
        """Class source must construct memory_pools at __init__ time."""
        import inspect

        from sgl_jax.srt.model_executor import model_runner

        src = inspect.getsource(model_runner.MockModelRunner.__init__)
        self.assertIn("self.memory_pools", src)
        self.assertIn("_build_non_hybrid_memory_pools", src)


class TestForwardCallsBridgeHelper(unittest.TestCase):
    """C4 fix: bridge test must verify _forward actually invokes
    _wrap_pool_updates, not just that the helper exists. Otherwise a future
    refactor that drops the call would silently bypass the bridge during
    Tasks 2-4 (and we'd only catch it via runtime crash)."""

    def test_forward_invokes_wrap_pool_updates(self):
        import inspect

        from sgl_jax.srt.model_executor import model_runner

        src = inspect.getsource(model_runner.ModelRunner._forward)
        self.assertIn(
            "_wrap_pool_updates(",
            src,
            "_forward must call _wrap_pool_updates(...) for the Phase 3 transition; "
            "without this call, half-migrated models will crash mid-Phase-3.",
        )


if __name__ == "__main__":
    unittest.main()
