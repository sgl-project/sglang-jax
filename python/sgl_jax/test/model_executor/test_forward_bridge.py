"""Phase 3 Task 5 lock-in: _wrap_pool_updates bridge removed.

Once all 13 edited models return dict (and llama_eagle3 inherits the
migrated parent __call__), the legacy bridge helper is deleted.
TestForwardBridgeAcceptsBothShapes (transient, asserted bridge behavior
during Tasks 2-4) is replaced by TestForwardBridgeRemoved (asserts the
helper no longer exists).
"""

import unittest


class TestForwardBridgeRemoved(unittest.TestCase):
    """Phase 3 Task 5: once all 13 edited models return dict (and llama_eagle3
    inherits the migrated parent __call__), the legacy bridge helper must be
    deleted (no half-finished implementations)."""

    def test_wrap_pool_updates_helper_removed(self):
        from sgl_jax.srt.model_executor import model_runner

        self.assertFalse(
            hasattr(model_runner, "_wrap_pool_updates"),
            "_wrap_pool_updates should be removed; "
            "_forward now expects dict return from models directly.",
        )


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


if __name__ == "__main__":
    unittest.main()
