import asyncio
import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from sgl_jax.srt.lora.constants import BASE_LORA_ID, BASE_LORA_SLOT
from sgl_jax.srt.lora.lora_manager import LoRAManager
from sgl_jax.srt.lora.lora_memory_pool import EMPTY_SLOT, LoRAMemoryPool
from sgl_jax.srt.lora.lora_registry import LoRARef, LoRARegistry
from sgl_jax.srt.lora.utils import LoRABatchPlan
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch


class TestLoRAManagerOptimization(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_mesh = MagicMock()
        self.mock_mesh.shape = {"tensor": 1}

        # Mock LoRAMemoryPool
        self.pool = MagicMock(spec=LoRAMemoryPool)
        self.pool.uid_to_buffer_id = {}
        self.pool.buffer_id_to_uid = [EMPTY_SLOT] * 5
        self.pool.max_loras_per_batch = 4
        self.pool.num_lora_slots = 5
        self.pool.target_modules = {"q_proj"}
        self.pool.get_buffer_id.side_effect = lambda uid: self.pool.uid_to_buffer_id.get(uid, 0)

        # We need to use the REAL prepare_lora_batch method from LoRAMemoryPool
        # to test its return value logic, but mock the heavy lifting parts.
        # So we will partial-mock LoRAMemoryPool.

    def test_memory_pool_prepare_batch_return_value(self):
        """Test that LoRAMemoryPool.prepare_lora_batch returns correct boolean."""
        pool = LoRAMemoryPool(
            max_loras_per_batch=4,
            max_lora_rank=8,
            num_layers=1,
            target_modules={"q_proj"},
            mesh=self.mock_mesh,
            dtype=jnp.float32,
        )

        # Mock the heavy methods
        pool.load_lora_weight_to_buffer = MagicMock()
        pool._get_lora_a_shape = MagicMock(return_value=(1, 1, 1))
        pool._get_lora_b_shape = MagicMock(return_value=(1, 1, 1))
        pool._get_lora_a_sharding = MagicMock()
        pool._get_lora_b_sharding = MagicMock()
        # Mock init_buffers to avoid actual JAX allocation if called (it is called in init)
        # But we instantiated it, so init_buffers was called.
        # We should have mocked it or we assume JAX can handle small allocs on CPU.
        # To be safe, let's just test the logic by mocking the method on the instance
        # BEFORE calling prepare_lora_batch.

        # Scenario 1: Load new LoRA
        cur_uids = {"lora1"}
        lora_adapters = {"lora1": MagicMock()}

        # Reset state
        pool.uid_to_buffer_id = {BASE_LORA_ID: BASE_LORA_SLOT, None: BASE_LORA_SLOT}
        pool.buffer_id_to_uid = [BASE_LORA_ID] + [EMPTY_SLOT] * pool.max_loras_per_batch

        has_new = pool.prepare_lora_batch(cur_uids, lora_adapters)
        self.assertTrue(has_new, "Should return True when loading new LoRA")
        self.assertIn("lora1", pool.uid_to_buffer_id)

        # Scenario 2: Reuse loaded LoRA
        # "lora1" is already in pool from previous step
        cur_uids = {"lora1"}
        has_new = pool.prepare_lora_batch(cur_uids, lora_adapters)
        self.assertFalse(has_new, "Should return False when reusing loaded LoRA")

        # Scenario 3: Load another new LoRA
        cur_uids = {"lora1", "lora2"}
        lora_adapters["lora2"] = MagicMock()
        has_new = pool.prepare_lora_batch(cur_uids, lora_adapters)
        self.assertTrue(has_new, "Should return True when loading at least one new LoRA")
        self.assertIn("lora2", pool.uid_to_buffer_id)

    def test_memory_pool_capacity_excludes_base_slot(self):
        """max_loras_per_batch counts real adapters; slot 0 is reserved for base."""
        pool = LoRAMemoryPool(
            max_loras_per_batch=2,
            max_lora_rank=8,
            num_layers=1,
            target_modules={"q_proj"},
            mesh=self.mock_mesh,
            dtype=jnp.float32,
        )
        pool.load_lora_weight_to_buffer = MagicMock()

        self.assertEqual(pool.num_lora_slots, 3)
        has_new = pool.prepare_lora_batch(
            [None, "lora1", "lora2"],
            {"lora1": MagicMock(), "lora2": MagicMock()},
        )

        self.assertTrue(has_new)
        self.assertEqual(pool.get_buffer_id(None), BASE_LORA_SLOT)
        self.assertNotEqual(pool.get_buffer_id("lora1"), BASE_LORA_SLOT)
        self.assertNotEqual(pool.get_buffer_id("lora2"), BASE_LORA_SLOT)

        with self.assertRaises(ValueError):
            pool.prepare_lora_batch(
                [None, "lora1", "lora2", "lora3"],
                {"lora1": MagicMock(), "lora2": MagicMock(), "lora3": MagicMock()},
            )

    def test_base_lora_id_uses_reserved_pool_slot(self):
        """Base-model requests use reserved slot 0 without loading adapter weights."""
        pool = LoRAMemoryPool(
            max_loras_per_batch=4,
            max_lora_rank=8,
            num_layers=1,
            target_modules={"q_proj"},
            mesh=self.mock_mesh,
            dtype=jnp.float32,
        )
        pool.load_lora_weight_to_buffer = MagicMock()

        has_new = pool.prepare_lora_batch([None, BASE_LORA_ID], {})

        self.assertFalse(has_new)
        self.assertEqual(pool.get_buffer_id(None), BASE_LORA_SLOT)
        self.assertEqual(pool.get_buffer_id(BASE_LORA_ID), BASE_LORA_SLOT)
        pool.load_lora_weight_to_buffer.assert_not_called()

    def test_registry_treats_base_lora_id_as_sentinel(self):
        """The registry should not create counters for base-model requests."""
        lora_ref = LoRARef(lora_name="adapter", lora_path="/tmp/adapter")
        registry = LoRARegistry([lora_ref])

        acquired_ids = asyncio.run(registry.acquire([None, "adapter"]))

        self.assertEqual(acquired_ids, [BASE_LORA_ID, lora_ref.lora_id])
        asyncio.run(registry.release(acquired_ids))
        asyncio.run(registry.release(BASE_LORA_ID))

        with self.assertRaises(ValueError):
            LoRARef(lora_id=BASE_LORA_ID, lora_name="reserved")

    def test_memory_pool_zero_buffer_slot_clears_selected_module_layer(self):
        pool = LoRAMemoryPool(
            max_loras_per_batch=1,
            max_lora_rank=2,
            num_layers=2,
            target_modules={"q_proj"},
            mesh=self.mock_mesh,
            dtype=jnp.float32,
            hidden_size=4,
            num_attention_heads=1,
        )
        pool.A_buffer = {"q_proj": [jnp.ones((2, 2, 4), dtype=jnp.float32) for _ in range(2)]}
        pool.B_buffer = {"q_proj": [jnp.ones((2, 4, 2), dtype=jnp.float32) for _ in range(2)]}

        pool._zero_buffer_slot(buffer_id=1, module_name="q_proj", layer_id=0)

        self.assertTrue(jnp.all(pool.A_buffer["q_proj"][0][1] == 0))
        self.assertTrue(jnp.all(pool.B_buffer["q_proj"][0][1] == 0))
        self.assertTrue(jnp.all(pool.A_buffer["q_proj"][1][1] == 1))
        self.assertTrue(jnp.all(pool.B_buffer["q_proj"][1][1] == 1))

    def test_lora_batch_plan_validates_slot_metadata(self):
        plan = LoRABatchPlan(
            weight_indices=[0, 2, 1],
            ranks_by_slot=[0, 8, 16],
            scalings_by_slot=[0.0, 0.5, 1.0],
        )

        self.assertEqual(plan.ranks_for_requests(), (0, 16, 8))
        self.assertEqual(plan.scalings_for_requests(), (0.0, 1.0, 0.5))

        with self.assertRaises(ValueError):
            LoRABatchPlan(weight_indices=[3], ranks_by_slot=[0, 8], scalings_by_slot=[0.0, 1.0])
        with self.assertRaises(ValueError):
            LoRABatchPlan(weight_indices=[-1], ranks_by_slot=[0, 8], scalings_by_slot=[0.0, 1.0])
        with self.assertRaises(ValueError):
            LoRABatchPlan(weight_indices=[1], ranks_by_slot=[0, -8], scalings_by_slot=[0.0, 1.0])
        with self.assertRaises(ValueError):
            LoRABatchPlan(weight_indices=[1], ranks_by_slot=[0, 8], scalings_by_slot=[0.0])
        with self.assertRaises(ValueError):
            LoRABatchPlan(weight_indices=[1], ranks_by_slot=[0, 8], scalings_by_slot=[0.0, -1.0])
        with self.assertRaises(ValueError):
            LoRABatchPlan(
                weight_indices=[1],
                ranks_by_slot=[0, 8],
                scalings_by_slot=[0.0, float("inf")],
            )
        with self.assertRaises(ValueError):
            LoRABatchPlan(
                weight_indices=[1],
                ranks_by_slot=[0, 8],
                scalings_by_slot=[0.0, float("nan")],
            )

        empty_plan = LoRABatchPlan(weight_indices=[], ranks_by_slot=[], scalings_by_slot=[])
        self.assertEqual(empty_plan.ranks_for_requests(), ())

        static_plan = LoRABatchPlan.for_static_lora(
            batch_size=3,
            num_lora_slots=4,
            rank=8,
            scaling=0.25,
        )
        self.assertEqual(static_plan.weight_indices, (0, 0, 0))
        self.assertEqual(static_plan.ranks_by_slot, (8, 8, 8, 8))
        self.assertEqual(static_plan.scalings_by_slot, (0.25, 0.25, 0.25, 0.25))

    @patch("sgl_jax.srt.lora.lora_manager.LoRAMemoryPool")
    def test_manager_conditional_update(self, MockPoolClass):
        """Test that LoRAManager conditionally calls update_lora_info."""
        # Setup Mock Pool instance
        mock_pool_instance = MockPoolClass.return_value
        mock_pool_instance.target_modules = {"q_proj"}

        # Setup Manager
        # We mock init_state to avoid complex initialization
        with patch.object(LoRAManager, "init_state"):
            manager = LoRAManager(
                base_model=None,
                base_hf_config=MagicMock(
                    num_hidden_layers=1, hidden_size=128, num_attention_heads=4
                ),
                max_loras_per_batch=4,
                dtype=jnp.float32,
                mesh=self.mock_mesh,
            )

        # Manually set attributes needed for prepare_lora_batch
        manager.memory_pool = mock_pool_instance
        mock_lora = MagicMock()
        mock_lora.config.r = 8
        mock_lora.scaling = 0.75
        manager.loras = {"lora1": mock_lora}
        manager.max_loras_per_batch = 4
        manager.num_lora_slots = 5
        manager.lora_backend = MagicMock()
        manager.update_lora_info = MagicMock()

        # Test case 1: memory_pool returns True (new weights)
        mock_pool_instance.prepare_lora_batch.return_value = True
        mock_pool_instance.get_buffer_id.return_value = 1

        batch = MagicMock(spec=ModelWorkerBatch)
        batch.lora_ids = ["lora1"]

        manager.prepare_lora_batch(batch)

        manager.lora_backend.prepare_lora_batch.assert_called_once()
        call_kwargs = manager.lora_backend.prepare_lora_batch.call_args.kwargs
        self.assertIs(call_kwargs["model_worker_batch"], batch)
        self.assertEqual(
            call_kwargs["batch_plan"],
            LoRABatchPlan(
                weight_indices=[1],
                ranks_by_slot=[0, 8, 0, 0, 0],
                scalings_by_slot=[0.0, 0.75, 0.0, 0.0, 0.0],
            ),
        )
        manager.update_lora_info.assert_called_once()
        manager.update_lora_info.reset_mock()
        manager.lora_backend.prepare_lora_batch.reset_mock()

        # Test case 2: memory_pool returns False (no new weights)
        mock_pool_instance.prepare_lora_batch.return_value = False

        manager.prepare_lora_batch(batch)

        manager.lora_backend.prepare_lora_batch.assert_called_once()
        manager.update_lora_info.assert_not_called()

    @patch("sgl_jax.srt.lora.lora_manager.LoRAMemoryPool")
    def test_manager_normalizes_lora_ids_without_mutating_batch(self, MockPoolClass):
        mock_pool_instance = MockPoolClass.return_value
        mock_pool_instance.target_modules = {"q_proj"}
        mock_pool_instance.prepare_lora_batch.return_value = False
        mock_pool_instance.get_buffer_id.side_effect = lambda uid: 0 if uid == BASE_LORA_ID else 1

        with patch.object(LoRAManager, "init_state"):
            manager = LoRAManager(
                base_model=None,
                base_hf_config=MagicMock(
                    num_hidden_layers=1, hidden_size=128, num_attention_heads=4
                ),
                max_loras_per_batch=4,
                dtype=jnp.float32,
                mesh=self.mock_mesh,
            )

        mock_lora = MagicMock()
        mock_lora.config.r = 8
        mock_lora.scaling = 0.75
        manager.memory_pool = mock_pool_instance
        manager.loras = {"lora1": mock_lora}
        manager.max_loras_per_batch = 4
        manager.num_lora_slots = 5
        manager.lora_backend = MagicMock()

        batch = MagicMock(spec=ModelWorkerBatch)
        batch.lora_ids = [None, "lora1"]

        manager.prepare_lora_batch(batch)

        self.assertEqual(batch.lora_ids, [None, "lora1"])
        mock_pool_instance.prepare_lora_batch.assert_called_once_with(
            cur_uids=[BASE_LORA_ID, "lora1"],
            lora_adapters=manager.loras,
        )
        self.assertEqual(
            manager.lora_backend.prepare_lora_batch.call_args.kwargs["batch_plan"],
            LoRABatchPlan(
                weight_indices=[0, 1],
                ranks_by_slot=[0, 8, 0, 0, 0],
                scalings_by_slot=[0.0, 0.75, 0.0, 0.0, 0.0],
            ),
        )

    @patch("sgl_jax.srt.lora.lora_manager.LoRAMemoryPool")
    def test_manager_rejects_unknown_dynamic_lora(self, MockPoolClass):
        mock_pool_instance = MockPoolClass.return_value
        mock_pool_instance.target_modules = {"q_proj"}

        with patch.object(LoRAManager, "init_state"):
            manager = LoRAManager(
                base_model=None,
                base_hf_config=MagicMock(
                    num_hidden_layers=1, hidden_size=128, num_attention_heads=4
                ),
                max_loras_per_batch=4,
                dtype=jnp.float32,
                mesh=self.mock_mesh,
            )

        manager.memory_pool = mock_pool_instance
        manager.loras = {}
        manager.max_lora_rank = 8
        manager.lora_backend = MagicMock()

        batch = MagicMock(spec=ModelWorkerBatch)
        batch.lora_ids = ["missing-lora"]

        with self.assertRaisesRegex(ValueError, "not loaded"):
            manager.prepare_lora_batch(batch)

        mock_pool_instance.prepare_lora_batch.assert_not_called()
        manager.lora_backend.prepare_lora_batch.assert_not_called()

    @patch("sgl_jax.srt.lora.lora_manager.LoRAMemoryPool")
    def test_manager_static_lora_builds_static_plan(self, MockPoolClass):
        mock_pool_instance = MockPoolClass.return_value
        mock_pool_instance.target_modules = {"q_proj"}
        server_args = MagicMock(enable_static_lora=True, lora_scaling=0.25)

        with patch.object(LoRAManager, "init_state"):
            manager = LoRAManager(
                base_model=None,
                base_hf_config=MagicMock(
                    num_hidden_layers=1, hidden_size=128, num_attention_heads=4
                ),
                max_loras_per_batch=4,
                max_lora_rank=8,
                dtype=jnp.float32,
                mesh=self.mock_mesh,
                server_args=server_args,
            )

        manager.memory_pool = mock_pool_instance
        manager.loras = {}
        manager.max_lora_rank = 8
        manager.lora_backend = MagicMock()

        batch = MagicMock(spec=ModelWorkerBatch)
        batch.lora_ids = [None, None]

        manager.prepare_lora_batch(batch)

        manager.lora_backend.prepare_lora_batch.assert_called_once()
        self.assertEqual(
            manager.lora_backend.prepare_lora_batch.call_args.kwargs["batch_plan"],
            LoRABatchPlan(
                weight_indices=[0, 0],
                ranks_by_slot=[8, 8, 8, 8],
                scalings_by_slot=[0.25, 0.25, 0.25, 0.25],
            ),
        )
        mock_pool_instance.prepare_lora_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
