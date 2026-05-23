"""Surface-level test for PR-1-3 TreeComponent ABC + enums.

Validates:
- Enum members exist with stable string values (so logs / metrics
  built on them are not silently re-numbered).
- :class:`TreeComponent` is genuinely abstract (cannot be instantiated
  without all three hooks + ``component_type``).
- A minimal compliant subclass can be instantiated and its
  ``component_type`` dispatches correctly.
"""

from __future__ import annotations

import unittest

from sgl_jax.srt.mem_cache.tree_component import (
    CacheTransferPhase,
    ComponentType,
    TreeComponent,
)


class TestTreeComponentSurface(unittest.TestCase):
    def test_component_type_values_are_stable(self):
        self.assertEqual(ComponentType.FULL.value, "full")
        self.assertEqual(ComponentType.SWA.value, "swa")
        self.assertEqual(ComponentType.RECURRENT_STATE.value, "recurrent_state")
        self.assertEqual(len(ComponentType), 3)

    def test_cache_transfer_phase_values_are_stable(self):
        expected = {
            "IDLE": "idle",
            "SCHEDULED": "scheduled",
            "IN_FLIGHT": "in_flight",
            "DONE": "done",
            "FAILED": "failed",
        }
        for name, value in expected.items():
            self.assertEqual(CacheTransferPhase[name].value, value)
        self.assertEqual(len(CacheTransferPhase), len(expected))

    def test_tree_component_is_abstract(self):
        with self.assertRaises(TypeError):
            TreeComponent()  # type: ignore[abstract]

    def test_subclass_missing_hook_still_abstract(self):
        class HalfBaked(TreeComponent):
            @property
            def component_type(self):
                return ComponentType.FULL

            # write_through / write_back / prefetch intentionally missing

        with self.assertRaises(TypeError):
            HalfBaked()  # type: ignore[abstract]

    def test_minimal_compliant_subclass_can_be_instantiated(self):
        class NoOpComponent(TreeComponent):
            @property
            def component_type(self):
                return ComponentType.FULL

            def write_through(self, *, node, device_indices, host_indices):
                return CacheTransferPhase.SCHEDULED

            def write_back(self, *, node, host_indices):
                return CacheTransferPhase.DONE

            def prefetch(self, *, node, host_indices, device_indices):
                return CacheTransferPhase.IN_FLIGHT

        c = NoOpComponent()
        self.assertIs(c.component_type, ComponentType.FULL)
        self.assertIs(
            c.write_through(node=None, device_indices=None, host_indices=None),
            CacheTransferPhase.SCHEDULED,
        )
        self.assertIs(c.write_back(node=None, host_indices=None), CacheTransferPhase.DONE)
        self.assertIs(
            c.prefetch(node=None, host_indices=None, device_indices=None),
            CacheTransferPhase.IN_FLIGHT,
        )


if __name__ == "__main__":
    unittest.main()
