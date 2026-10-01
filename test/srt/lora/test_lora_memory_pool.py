import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.lora.lora_memory_pool import LoRAMemoryPool


class TestLoRAMemoryPool(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh(np.array(jax.devices()[:1]), ("tensor",))
        self.pool = self.make_pool()
        self.adapters = {uid: self.make_adapter() for uid in ("a", "b", "c", "d")}

    def make_pool(self, capacity=3):
        pool = LoRAMemoryPool(
            max_loras_per_batch=capacity,
            max_lora_rank=4,
            num_layers=2,
            target_modules={"q_proj", "v_proj"},
            mesh=self.mesh,
            dtype=jnp.float32,
            hidden_size=4,
            num_attention_heads=1,
            num_kv_heads=1,
        )
        pool.init_buffers()
        return pool

    def make_adapter(self, rank=4, modules=("q_proj", "v_proj"), value=1):
        layers = []
        for layer_id in range(2):
            weights = {}
            for module in modules:
                prefix = f"base_model.model.layers.{layer_id}.self_attn.{module}"
                weights[f"{prefix}.lora_A.weight"] = jnp.full((rank, 4), value, jnp.float32)
                weights[f"{prefix}.lora_B.weight"] = jnp.full((4, rank), value, jnp.float32)
            layers.append(SimpleNamespace(weights=weights))
        return SimpleNamespace(layers=layers)

    def test_empty_slots_and_hits(self):
        self.assertTrue(self.pool.prepare_lora_batch({"a"}, self.adapters))
        a_slot = self.pool.get_buffer_id("a")
        self.assertTrue(self.pool.prepare_lora_batch({"b"}, self.adapters))
        self.assertEqual(self.pool.get_buffer_id("a"), a_slot)
        self.assertEqual(set(self.pool.uid_to_buffer_id.values()), {1, 2})
        with patch.object(self.pool, "load_lora_weight_to_buffer") as load:
            self.assertFalse(self.pool.prepare_lora_batch({"a", "b"}, self.adapters))
            load.assert_not_called()

    def test_lru_hit_refresh_and_reload(self):
        self.pool.prepare_lora_batch({"a", "b"}, self.adapters)
        b_slot = self.pool.get_buffer_id("b")
        self.pool.prepare_lora_batch({"a"}, self.adapters)
        self.assertTrue(self.pool.prepare_lora_batch({"c"}, self.adapters))
        self.assertNotIn("b", self.pool.uid_to_buffer_id)
        self.assertEqual(self.pool.get_buffer_id("c"), b_slot)
        self.pool.prepare_lora_batch({"b"}, self.adapters)
        self.assertNotIn("a", self.pool.uid_to_buffer_id)
        for uid, slot in self.pool.uid_to_buffer_id.items():
            self.assertEqual(self.pool.buffer_id_to_uid[slot], uid)

    def test_entire_current_batch_is_protected(self):
        self.pool.prepare_lora_batch({"c"}, self.adapters)
        c_slot = self.pool.get_buffer_id("c")
        self.pool.prepare_lora_batch({"b"}, self.adapters)
        # "a" is visited before the oldest resident "c", which must still be protected.
        self.pool.prepare_lora_batch({"a", "c"}, self.adapters)
        self.assertEqual(self.pool.get_buffer_id("c"), c_slot)
        self.assertNotIn("b", self.pool.uid_to_buffer_id)

    def test_multiple_replacements(self):
        self.pool.prepare_lora_batch({"a", "b"}, self.adapters)
        self.pool.prepare_lora_batch({"c", "d"}, self.adapters)
        self.assertEqual(set(self.pool.uid_to_buffer_id), {"c", "d"})
        self.assertEqual(set(self.pool.uid_to_buffer_id.values()), {1, 2})

    def test_pinned_adapter_is_protected(self):
        self.pool.prepare_lora_batch({"a", "b"}, self.adapters)
        a_slot = self.pool.get_buffer_id("a")
        self.pool.prepare_lora_batch({"c"}, self.adapters, pinned_uids={"a"})
        self.assertEqual(self.pool.get_buffer_id("a"), a_slot)
        self.assertNotIn("b", self.pool.uid_to_buffer_id)

    def test_capacity_errors_do_not_mutate_pool(self):
        self.pool.prepare_lora_batch({"a", "b"}, self.adapters)
        before = list(self.pool.uid_to_buffer_id.items())
        for required, pinned in [
            ({"c"}, {"a", "b"}),
            ({"b", "c"}, {"a"}),
            ({"a", "b", "c"}, set()),
        ]:
            with self.subTest(required=required, pinned=pinned):
                with self.assertRaisesRegex(ValueError, "No available buffer slots"):
                    self.pool.prepare_lora_batch(required, self.adapters, pinned)
                self.assertEqual(list(self.pool.uid_to_buffer_id.items()), before)

    def test_unknown_adapter_is_rejected_before_loading(self):
        with patch.object(self.pool, "load_lora_weight_to_buffer") as load:
            with self.assertRaisesRegex(ValueError, "not loaded"):
                self.pool.prepare_lora_batch({"a", "unknown"}, self.adapters)
            load.assert_not_called()
        self.assertEqual(self.pool.uid_to_buffer_id, {})

    def test_base_and_padding_share_reserved_zero_slot(self):
        self.pool.prepare_lora_batch({None, "0", "a", "b"}, self.adapters)
        self.pool.prepare_lora_batch({None, "0", "c"}, self.adapters)
        self.assertEqual(self.pool.get_buffer_id(None), 0)
        self.assertEqual(self.pool.get_buffer_id("0"), 0)
        self.assertIsNone(self.pool.buffer_id_to_uid[0])
        self.assertFalse(self.pool.prepare_lora_batch({None, "0"}, self.adapters))
        for buffers in (self.pool.A_buffer, self.pool.B_buffer):
            for layers in buffers.values():
                for array in layers:
                    np.testing.assert_array_equal(array[0], 0)

    def test_base_only_pool(self):
        pool = self.make_pool(capacity=1)
        self.assertFalse(pool.prepare_lora_batch({None, "0"}, self.adapters))
        with self.assertRaisesRegex(ValueError, "No available buffer slots"):
            pool.prepare_lora_batch({"a"}, self.adapters)

    def test_reused_slot_matches_fresh_slot(self):
        self.pool.prepare_lora_batch({"a", "b"}, self.adapters)
        old_slot = self.pool.get_buffer_id("a")
        b_slot = self.pool.get_buffer_id("b")
        self.adapters["c"] = self.make_adapter(rank=1, modules=("q_proj",), value=2)
        self.pool.prepare_lora_batch({"c"}, self.adapters)
        self.assertEqual(self.pool.get_buffer_id("c"), old_slot)

        fresh = self.make_pool()
        fresh.prepare_lora_batch({"c"}, self.adapters)
        fresh_slot = fresh.get_buffer_id("c")
        for module in self.pool.target_modules:
            for layer_id in range(self.pool.num_layers):
                for actual_buffers, fresh_buffers in [
                    (self.pool.A_buffer, fresh.A_buffer),
                    (self.pool.B_buffer, fresh.B_buffer),
                ]:
                    actual = actual_buffers[module][layer_id]
                    np.testing.assert_array_equal(
                        actual[old_slot], fresh_buffers[module][layer_id][fresh_slot]
                    )
                    np.testing.assert_array_equal(actual[b_slot], 1)
                    self.assertEqual(actual.dtype, jnp.float32)
                    self.assertEqual(actual.sharding, fresh_buffers[module][layer_id].sharding)
                a = self.pool.A_buffer[module][layer_id][old_slot]
                b = self.pool.B_buffer[module][layer_id][old_slot]
                expected_a = fresh.A_buffer[module][layer_id][fresh_slot]
                expected_b = fresh.B_buffer[module][layer_id][fresh_slot]
                x = jnp.arange(4, dtype=jnp.float32)
                np.testing.assert_allclose(b @ (a @ x), expected_b @ (expected_a @ x))
                if module == "v_proj":
                    np.testing.assert_array_equal(a, 0)
                    np.testing.assert_array_equal(b, 0)
                else:
                    np.testing.assert_array_equal(a[1:], 0)
                    np.testing.assert_array_equal(b[:, 1:], 0)

    def test_failed_replacement_preserves_resident(self):
        self.pool.prepare_lora_batch({"a", "b"}, self.adapters)
        mappings = list(self.pool.uid_to_buffer_id.items())
        a_buffers, b_buffers = self.pool.A_buffer, self.pool.B_buffer
        with (
            patch.object(
                self.pool,
                "_extract_module_weights",
                side_effect=[
                    (jnp.full((4, 4), 2.0), jnp.full((4, 4), 2.0)),
                    ValueError("invalid weights"),
                ],
            ),
            self.assertRaisesRegex(ValueError, "invalid weights"),
        ):
            self.pool.prepare_lora_batch({"c"}, self.adapters)
        self.assertEqual(list(self.pool.uid_to_buffer_id.items()), mappings)
        self.assertIs(self.pool.A_buffer, a_buffers)
        self.assertIs(self.pool.B_buffer, b_buffers)
        self.pool.prepare_lora_batch({"c"}, self.adapters)
        self.assertNotIn("a", self.pool.uid_to_buffer_id)

    def test_pytree_round_trip_preserves_lru_order(self):
        self.pool.prepare_lora_batch({"a", "b"}, self.adapters)
        self.pool.prepare_lora_batch({"a"}, self.adapters)
        leaves, tree = jax.tree_util.tree_flatten(self.pool)
        restored = jax.tree_util.tree_unflatten(tree, leaves)
        restored.prepare_lora_batch({"c"}, self.adapters)
        self.assertNotIn("b", restored.uid_to_buffer_id)
        self.assertIn("a", restored.uid_to_buffer_id)


if __name__ == "__main__":
    unittest.main()
