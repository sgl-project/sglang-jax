import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp

from sgl_jax.srt.lora.lora_manager import LoRAManager
from sgl_jax.srt.lora.lora_memory_pool import EMPTY_SLOT, LoRAMemoryPool
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch


class TestLoRAManagerOptimization(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_mesh = MagicMock()
        self.mock_mesh.shape = {"tensor": 1}

        # Mock LoRAMemoryPool
        self.pool = MagicMock(spec=LoRAMemoryPool)
        self.pool.uid_to_buffer_id = {}
        self.pool.buffer_id_to_uid = [EMPTY_SLOT] * 4
        self.pool.max_loras_per_batch = 4
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
        pool.uid_to_buffer_id = {}
        pool.buffer_id_to_uid = [EMPTY_SLOT] * 4

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
        manager.loras = {}
        manager.max_loras_per_batch = 4
        manager.lora_backend = MagicMock()
        manager.update_lora_info = MagicMock()

        # Test case 1: memory_pool returns True (new weights)
        mock_pool_instance.prepare_lora_batch.return_value = True

        batch = MagicMock(spec=ModelWorkerBatch)
        batch.lora_ids = ["lora1"]

        manager.prepare_lora_batch(batch)

        manager.update_lora_info.assert_called_once()
        manager.update_lora_info.reset_mock()

        # Test case 2: memory_pool returns False (no new weights)
        mock_pool_instance.prepare_lora_batch.return_value = False

        manager.prepare_lora_batch(batch)

        manager.update_lora_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
