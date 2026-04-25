import os
import unittest

import jax
import jax.numpy as jnp


class _Base(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make(self, **overrides):
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        kwargs = dict(
            linear_recurrent_layer_ids=[0, 1],
            max_num_reqs=4,
            num_heads=2,
            head_dim=4,
            conv_kernel_size=4,
        )
        kwargs.update(overrides)
        return RecurrentStatePool(**kwargs)


class TestInit(_Base):
    """Initialization with default params.

    Default params linear_recurrent_layer_ids=[0, 1], max_num_reqs=4, num_heads=2, head_dim=4, conv_kernel_size=4
    -> recurrent_buffers: list of length 2, each shape (5, 2, 4, 4) f32
    -> conv_buffers: list-of-list outer length 2 inner length 1, each shape (5, 3, 24) bf16
       (proj_v + 2*proj_k = 2*4 + 2*(2*4) = 24)
    """

    def test_recurrent_buffers_is_list_with_correct_length(self):
        pool = self._make()
        self.assertIsInstance(pool.recurrent_buffers, list)
        self.assertEqual(len(pool.recurrent_buffers), 2)

    def test_recurrent_buffers_element_shape_and_dtype(self):
        pool = self._make()
        for layer in range(2):
            self.assertEqual(pool.recurrent_buffers[layer].shape, (5, 2, 4, 4))
            self.assertEqual(pool.recurrent_buffers[layer].dtype, jnp.float32)
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer] == 0)))

    def test_conv_buffers_is_list_of_list(self):
        pool = self._make()
        self.assertIsInstance(pool.conv_buffers, list)
        self.assertEqual(len(pool.conv_buffers), 2)
        for layer in range(2):
            self.assertIsInstance(pool.conv_buffers[layer], list)
            self.assertEqual(len(pool.conv_buffers[layer]), 1)

    def test_conv_buffers_element_shape_and_dtype(self):
        pool = self._make()
        for layer in range(2):
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertEqual(pool.conv_buffers[layer][inner].shape, (5, 3, 24))
                self.assertEqual(pool.conv_buffers[layer][inner].dtype, jnp.bfloat16)
                self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner] == 0)))

    def test_free_slots_starts_from_one(self):
        pool = self._make()
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])

    def test_proj_size_attribute(self):
        pool = self._make()
        self.assertEqual(pool.proj_size, 24)


class TestPytree(_Base):
    """children = (recurrent_buffers, conv_buffers); list auto-expands to 2L leaves."""

    def test_leaves_count_equals_2L(self):
        pool = self._make()  # L=2, conv inner length 1 -> leaves = L + L*1 = 4
        leaves, _ = jax.tree_util.tree_flatten(pool)
        self.assertEqual(len(leaves), 4)

    def test_roundtrip_preserves_buffers_and_metadata(self):
        pool = self._make()
        # Use list element mutation to write non-zero values
        # (roundtrip must verify more than the all-zero case).
        pool.recurrent_buffers[0] = pool.recurrent_buffers[0].at[1].set(1.5)
        pool.conv_buffers[0][0] = pool.conv_buffers[0][0].at[1].set(jnp.bfloat16(2.5))

        leaves, treedef = jax.tree_util.tree_flatten(pool)
        pool2 = jax.tree_util.tree_unflatten(treedef, leaves)

        # Buffers are equal layer-by-layer + inner-by-inner.
        self.assertEqual(len(pool2.recurrent_buffers), len(pool.recurrent_buffers))
        for layer in range(pool.num_layers):
            self.assertTrue(
                bool(jnp.all(pool.recurrent_buffers[layer] == pool2.recurrent_buffers[layer]))
            )
            self.assertEqual(len(pool2.conv_buffers[layer]), len(pool.conv_buffers[layer]))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertTrue(
                    bool(
                        jnp.all(pool.conv_buffers[layer][inner] == pool2.conv_buffers[layer][inner])
                    )
                )

        # Metadata
        self.assertEqual(pool2.num_layers, pool.num_layers)
        self.assertEqual(pool2.max_num_reqs, pool.max_num_reqs)
        self.assertEqual(pool2.num_heads, pool.num_heads)
        self.assertEqual(pool2.head_dim, pool.head_dim)
        self.assertEqual(pool2.conv_kernel_size, pool.conv_kernel_size)
        self.assertEqual(pool2.proj_size, pool.proj_size)
        self.assertEqual(pool2.free_slots, pool.free_slots)
        self.assertEqual(pool2.recurrent_buffers[0].dtype, jnp.float32)
        self.assertEqual(pool2.conv_buffers[0][0].dtype, jnp.bfloat16)

    def test_roundtrip_recovers_list_containers(self):
        """JAX pytree may restore lists as tuples; unflatten MUST coerce back to mutable list."""
        pool = self._make()
        leaves, treedef = jax.tree_util.tree_flatten(pool)
        pool2 = jax.tree_util.tree_unflatten(treedef, leaves)

        # Must be list (mutable); otherwise subsequent list element mutation would fail.
        self.assertIsInstance(pool2.recurrent_buffers, list)
        self.assertIsInstance(pool2.conv_buffers, list)
        self.assertIsInstance(pool2.conv_buffers[0], list)
        # Verify list element mutation works on the restored container.
        pool2.recurrent_buffers[0] = pool2.recurrent_buffers[0].at[1].set(7.0)


class TestDtype(_Base):
    """dtype priority: constructor arg > env var > default."""

    def test_constructor_dtype_overrides_default(self):
        pool = self._make(temporal_dtype=jnp.bfloat16, conv_dtype=jnp.float16)
        self.assertEqual(pool.recurrent_buffers[0].dtype, jnp.bfloat16)
        self.assertEqual(pool.conv_buffers[0][0].dtype, jnp.float16)

    def test_env_var_overrides_default(self):
        os.environ["SGLANG_JAX_RECURRENT_STATE_DTYPE"] = "bfloat16"
        os.environ["SGLANG_JAX_CONV_STATE_DTYPE"] = "float32"
        try:
            pool = self._make()
            self.assertEqual(pool.recurrent_buffers[0].dtype, jnp.bfloat16)
            self.assertEqual(pool.conv_buffers[0][0].dtype, jnp.float32)
        finally:
            del os.environ["SGLANG_JAX_RECURRENT_STATE_DTYPE"]
            del os.environ["SGLANG_JAX_CONV_STATE_DTYPE"]

    def test_constructor_arg_overrides_env_var(self):
        os.environ["SGLANG_JAX_RECURRENT_STATE_DTYPE"] = "bfloat16"
        try:
            pool = self._make(temporal_dtype=jnp.float32)
            self.assertEqual(pool.recurrent_buffers[0].dtype, jnp.float32)
        finally:
            del os.environ["SGLANG_JAX_RECURRENT_STATE_DTYPE"]


class TestEdgeCases(_Base):
    """Constructor boundaries (minimal dimensions, odd num_heads, proj_size formula, conv_kernel_size)."""

    def test_minimal_dimensions(self):
        pool = self._make(
            linear_recurrent_layer_ids=[0],
            max_num_reqs=1,
            num_heads=1,
            head_dim=1,
            conv_kernel_size=2,
        )
        self.assertEqual(len(pool.recurrent_buffers), 1)
        self.assertEqual(pool.recurrent_buffers[0].shape, (2, 1, 1, 1))
        # proj_size = 1 + 2*1 = 3
        self.assertEqual(len(pool.conv_buffers), 1)
        self.assertEqual(len(pool.conv_buffers[0]), 1)
        self.assertEqual(pool.conv_buffers[0][0].shape, (2, 1, 3))
        self.assertEqual(pool.free_slots, [1])

    def test_odd_num_heads_supported(self):
        pool = self._make(num_heads=3)
        self.assertEqual(pool.recurrent_buffers[0].shape[1], 3)
        # proj_size = 3*4 + 2*(3*4) = 36
        self.assertEqual(pool.proj_size, 36)

    def test_proj_size_formula_not_simplified(self):
        pool = self._make(num_heads=5, head_dim=7)
        self.assertEqual(pool.proj_size, 5 * 7 + 2 * (5 * 7))
        self.assertEqual(pool.conv_buffers[0][0].shape[-1], 105)

    def test_conv_kernel_size_one_raises(self):
        # K=1 makes conv_buffers[l][i] second dim (K-1) zero;
        # constructor must assert and reject.
        with self.assertRaises(AssertionError):
            self._make(conv_kernel_size=1)

    def test_zero_or_negative_dimensions_raise(self):
        for kwargs in [
            {"max_num_reqs": 0},
            {"num_heads": 0},
            {"head_dim": 0},
        ]:
            with self.subTest(**kwargs), self.assertRaises(AssertionError):
                self._make(**kwargs)


class TestAllocFree(_Base):
    """alloc / free / clear-on-alloc behavior."""

    def test_alloc_single_returns_first_free_slot(self):
        pool = self._make()
        slots = pool.alloc(1)
        self.assertEqual(slots, [1])
        self.assertEqual(pool.free_slots, [2, 3, 4])

    def test_alloc_batch_returns_consecutive_slots(self):
        pool = self._make()
        slots = pool.alloc(3)
        self.assertEqual(slots, [1, 2, 3])
        self.assertEqual(pool.free_slots, [4])

    def test_alloc_exceeding_returns_none_without_state_change(self):
        pool = self._make()
        pool.alloc(3)
        before = list(pool.free_slots)
        # Compare list refs with `is` + per-element `is` to assert nothing was replaced.
        before_recurrent_refs = list(pool.recurrent_buffers)
        before_conv_refs = [list(inner) for inner in pool.conv_buffers]
        self.assertIsNone(pool.alloc(2))
        self.assertEqual(pool.free_slots, before)
        for layer in range(pool.num_layers):
            self.assertIs(pool.recurrent_buffers[layer], before_recurrent_refs[layer])
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertIs(pool.conv_buffers[layer][inner], before_conv_refs[layer][inner])

    def test_alloc_clears_both_buffers_at_allocated_slots(self):
        pool = self._make()
        # Write non-zero via list element mutation (matches the implementation style).
        for layer in range(pool.num_layers):
            pool.recurrent_buffers[layer] = jnp.ones_like(pool.recurrent_buffers[layer])
            for inner in range(len(pool.conv_buffers[layer])):
                pool.conv_buffers[layer][inner] = jnp.ones_like(pool.conv_buffers[layer][inner])
        slots = pool.alloc(2)  # [1, 2]
        for layer in range(pool.num_layers):
            for idx in slots:
                self.assertTrue(
                    bool(jnp.all(pool.recurrent_buffers[layer][idx] == 0)),
                    f"recurrent_buffers[{layer}][{idx}] not cleared",
                )
            for inner in range(len(pool.conv_buffers[layer])):
                for idx in slots:
                    self.assertTrue(
                        bool(jnp.all(pool.conv_buffers[layer][inner][idx] == 0)),
                        f"conv_buffers[{layer}][{inner}][{idx}] not cleared",
                    )
        # Unallocated slot 3 keeps its previous value (per layer + per inner).
        for layer in range(pool.num_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer][3] == 1)))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner][3] == 1)))

    def test_alloc_does_not_touch_dummy_slot_zero(self):
        pool = self._make()
        # Stamp slot 0; alloc must not touch it.
        for layer in range(pool.num_layers):
            pool.recurrent_buffers[layer] = pool.recurrent_buffers[layer].at[0].set(7.0)
        pool.alloc(4)
        for layer in range(pool.num_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer][0] == 7)))

    def test_free_returns_slot_to_pool(self):
        pool = self._make()
        slots = pool.alloc(2)
        pool.free(slots[0])
        self.assertIn(slots[0], pool.free_slots)

    def test_free_then_realloc_clears_again(self):
        pool = self._make()
        slots = pool.alloc(1)  # [1]
        # Write non-zero into slot 1 (per-layer list element mutation).
        for layer in range(pool.num_layers):
            pool.recurrent_buffers[layer] = pool.recurrent_buffers[layer].at[slots[0]].set(99.0)
        pool.free(slots[0])
        new_slots = pool.alloc(1)
        idx = new_slots[0]
        # After re-alloc the slot is cleared in every layer.
        for layer in range(pool.num_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer][idx] == 0)))

    def test_free_accepts_int_or_list(self):
        pool = self._make()
        pool.alloc(3)
        pool.free(1)
        pool.free([2, 3])
        self.assertEqual(sorted(pool.free_slots), [1, 2, 3, 4])


class TestReplaceBufferAndClear(_Base):
    def test_replace_buffer_swaps_both_buffers(self):
        pool = self._make()
        new_recurrent = [jnp.ones_like(b) * 3.0 for b in pool.recurrent_buffers]
        new_conv = [
            [jnp.ones_like(c) * jnp.bfloat16(4.0) for c in inner] for inner in pool.conv_buffers
        ]
        pool.replace_buffer((new_recurrent, new_conv))
        for layer in range(pool.num_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer] == 3)))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner] == jnp.bfloat16(4))))

    def test_replace_buffer_preserves_dtypes(self):
        pool = self._make()
        new_recurrent = [jnp.zeros_like(b) for b in pool.recurrent_buffers]
        new_conv = [[jnp.zeros_like(c) for c in inner] for inner in pool.conv_buffers]
        pool.replace_buffer((new_recurrent, new_conv))
        for layer in range(pool.num_layers):
            self.assertEqual(pool.recurrent_buffers[layer].dtype, jnp.float32)
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertEqual(pool.conv_buffers[layer][inner].dtype, jnp.bfloat16)

    def test_replace_buffer_does_not_touch_free_slots(self):
        pool = self._make()
        pool.alloc(2)
        new_recurrent = [jnp.zeros_like(b) for b in pool.recurrent_buffers]
        new_conv = [[jnp.zeros_like(c) for c in inner] for inner in pool.conv_buffers]
        pool.replace_buffer((new_recurrent, new_conv))
        self.assertEqual(pool.free_slots, [3, 4])

    def test_alloc_after_replace_uses_correct_slot_order(self):
        pool = self._make()
        pool.alloc(2)
        new_recurrent = [jnp.ones_like(b) for b in pool.recurrent_buffers]
        new_conv = [[jnp.ones_like(c) for c in inner] for inner in pool.conv_buffers]
        pool.replace_buffer((new_recurrent, new_conv))
        slots = pool.alloc(1)
        self.assertEqual(slots, [3])

    def test_replace_buffer_assert_recurrent_length_mismatch(self):
        """recurrent list length must equal num_layers."""
        pool = self._make()  # num_layers = 2
        short_recurrent = [pool.recurrent_buffers[0]]  # length 1
        new_conv = [[jnp.zeros_like(c) for c in inner] for inner in pool.conv_buffers]
        with self.assertRaises(AssertionError):
            pool.replace_buffer((short_recurrent, new_conv))

    def test_replace_buffer_assert_conv_outer_length_mismatch(self):
        pool = self._make()
        new_recurrent = [jnp.zeros_like(b) for b in pool.recurrent_buffers]
        short_conv = [pool.conv_buffers[0]]  # length 1
        with self.assertRaises(AssertionError):
            pool.replace_buffer((new_recurrent, short_conv))

    def test_replace_buffer_assert_conv_inner_length_mismatch(self):
        """conv inner length per layer must match existing
        (guards against future multi-conv-segment misuse)."""
        pool = self._make()  # inner length 1
        new_recurrent = [jnp.zeros_like(b) for b in pool.recurrent_buffers]
        # Stuff one extra conv segment into each inner list.
        new_conv = [
            [jnp.zeros_like(inner[0]), jnp.zeros_like(inner[0])] for inner in pool.conv_buffers
        ]
        with self.assertRaises(AssertionError):
            pool.replace_buffer((new_recurrent, new_conv))

    def test_clear_zeros_both_buffers_and_resets_free_slots(self):
        pool = self._make()
        # Write non-zero via list element mutation (matches implementation style).
        for layer in range(pool.num_layers):
            pool.recurrent_buffers[layer] = jnp.ones_like(pool.recurrent_buffers[layer])
            for inner in range(len(pool.conv_buffers[layer])):
                pool.conv_buffers[layer][inner] = jnp.ones_like(pool.conv_buffers[layer][inner])
        pool.alloc(3)
        pool.clear()
        for layer in range(pool.num_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer] == 0)))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner] == 0)))
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])


class TestLayersMapping(_Base):
    """linear_recurrent_layer_ids -> layers_mapping translation."""

    def test_mapping_for_default_consecutive_ids(self):
        pool = self._make()  # [0, 1]
        self.assertEqual(pool.linear_recurrent_layer_ids, [0, 1])
        self.assertEqual(pool.layers_mapping, {0: 0, 1: 1})
        self.assertEqual(pool.num_layers, 2)

    def test_mapping_for_non_consecutive_ids(self):
        pool = self._make(linear_recurrent_layer_ids=[3, 7])
        self.assertEqual(pool.linear_recurrent_layer_ids, [3, 7])
        self.assertEqual(pool.layers_mapping, {3: 0, 7: 1})
        self.assertEqual(pool.num_layers, 2)
        # buffers length tracks num_layers, not the max id
        self.assertEqual(len(pool.recurrent_buffers), 2)
        self.assertEqual(len(pool.conv_buffers), 2)

    def test_mapping_for_kimi_linear_like_layout(self):
        # Kimi-Linear-48B layout: 27 layers total. Full attention layers
        # (0-indexed: {3, 7, 11, 15, 19, 23, 26}) are excluded; the remaining
        # 20 layers are KDA (recurrent).
        full_attn = {3, 7, 11, 15, 19, 23, 26}
        kda_layers = [i for i in range(27) if i not in full_attn]
        pool = self._make(linear_recurrent_layer_ids=kda_layers, max_num_reqs=2)
        self.assertEqual(pool.num_layers, 20)
        self.assertEqual(len(pool.layers_mapping), 20)
        # Spot-check: first KDA layer (id 0) maps to local idx 0;
        # last KDA layer (id 25) maps to local idx 19.
        self.assertEqual(pool.layers_mapping[0], 0)
        self.assertEqual(pool.layers_mapping[25], 19)
        # Full attention layer 3 must NOT be in mapping.
        self.assertNotIn(3, pool.layers_mapping)

    def test_duplicate_layer_ids_raises(self):
        with self.assertRaises(AssertionError):
            self._make(linear_recurrent_layer_ids=[0, 1, 0])

    def test_empty_layer_ids_is_legal(self):
        # Degenerate but valid: zero recurrent layers.
        pool = self._make(linear_recurrent_layer_ids=[])
        self.assertEqual(pool.num_layers, 0)
        self.assertEqual(pool.linear_recurrent_layer_ids, [])
        self.assertEqual(pool.layers_mapping, {})
        self.assertEqual(pool.recurrent_buffers, [])
        self.assertEqual(pool.conv_buffers, [])
        # alloc / clear must still work on the empty pool.
        slots = pool.alloc(2)
        self.assertEqual(slots, [1, 2])
        pool.clear()
        self.assertEqual(pool.free_slots, [1, 2, 3, 4])


class TestPytreeRoundtripPreservesMapping(_Base):
    """JIT donate must not lose linear_recurrent_layer_ids / layers_mapping."""

    def test_aux_carries_layer_ids_tuple(self):
        pool = self._make(linear_recurrent_layer_ids=[3, 7])
        _, aux = pool.tree_flatten()
        # aux[0] must be tuple (hashable), not list.
        self.assertEqual(aux[0], (3, 7))
        self.assertIsInstance(aux[0], tuple)

    def test_unflatten_rebuilds_mapping(self):
        pool = self._make(linear_recurrent_layer_ids=[3, 7])
        leaves, treedef = jax.tree_util.tree_flatten(pool)
        pool2 = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual(pool2.linear_recurrent_layer_ids, [3, 7])
        self.assertEqual(pool2.layers_mapping, {3: 0, 7: 1})
        self.assertEqual(pool2.num_layers, 2)
        # External API still works after JIT donate roundtrip.
        rec, conv = pool2.get_linear_recurrent_layer_cache(7)
        self.assertEqual(rec.shape, pool.recurrent_buffers[1].shape)
        self.assertEqual(conv[0].shape, pool.conv_buffers[1][0].shape)


class TestGetLinearRecurrentLayerCache(_Base):
    """get_linear_recurrent_layer_cache external API."""

    def test_get_returns_two_tuple_of_list_elements(self):
        pool = self._make()  # [0, 1]
        rec0, conv0 = pool.get_linear_recurrent_layer_cache(0)
        rec1, conv1 = pool.get_linear_recurrent_layer_cache(1)
        # `is` relation: get returns the list element itself, no copy.
        self.assertIs(rec0, pool.recurrent_buffers[0])
        self.assertIs(rec1, pool.recurrent_buffers[1])
        self.assertIs(conv0, pool.conv_buffers[0])
        self.assertIs(conv1, pool.conv_buffers[1])

    def test_get_with_global_layer_id_translates(self):
        pool = self._make(linear_recurrent_layer_ids=[3, 7])
        # global layer_id 7 -> local idx 1
        rec, conv = pool.get_linear_recurrent_layer_cache(7)
        self.assertIs(rec, pool.recurrent_buffers[1])
        self.assertIs(conv, pool.conv_buffers[1])

    def test_get_unregistered_layer_id_raises_value_error(self):
        pool = self._make()  # registered: [0, 1]
        with self.assertRaises(ValueError) as ctx:
            pool.get_linear_recurrent_layer_cache(99)
        self.assertIn("99", str(ctx.exception))
        self.assertIn("Registered", str(ctx.exception))

    def test_no_setter_method_exposed(self):
        """RFC-0015: backend uses functional .at[].set() return; pool exposes no setter."""
        pool = self._make()
        self.assertFalse(hasattr(pool, "set_linear_recurrent_layer_cache"))


class TestBackendFunctionalReturnNoPoolSideEffect(_Base):
    """RFC-0015 functional pattern: backend reads via get, applies
    .at[indices].set(...) to build a new layer buffer, and returns it. The
    pool's internal recurrent_buffers / conv_buffers list elements stay
    referentially unchanged across the backend call. Persistence happens only
    when replace_buffer is invoked with the collected per-layer outputs.
    """

    def test_backend_call_does_not_mutate_pool_buffers(self):
        pool = self._make(linear_recurrent_layer_ids=[3, 7])
        slots = pool.alloc(2)  # [1, 2]

        # Capture per-layer references before the simulated backend call.
        recurrent_refs_before = [pool.recurrent_buffers[i] for i in range(pool.num_layers)]
        conv_refs_before = [
            [pool.conv_buffers[i][j] for j in range(len(pool.conv_buffers[i]))]
            for i in range(pool.num_layers)
        ]

        idx_arr = jnp.asarray(slots, dtype=jnp.int32)
        new_state_l3 = jnp.full((2, 2, 4, 4), 3.0, dtype=jnp.float32)
        new_state_l7 = jnp.full((2, 2, 4, 4), 7.0, dtype=jnp.float32)
        new_conv_l3 = jnp.full((2, 3, 24), jnp.bfloat16(3.5), dtype=jnp.bfloat16)
        new_conv_l7 = jnp.full((2, 3, 24), jnp.bfloat16(7.5), dtype=jnp.bfloat16)

        layers_recurrent = []
        layers_conv = []
        # Simulate backend per-layer: read -> functional update -> append return.
        for layer_id, new_state, new_conv in [
            (3, new_state_l3, new_conv_l3),
            (7, new_state_l7, new_conv_l7),
        ]:
            cur_recurrent_layer, cur_conv_layer = pool.get_linear_recurrent_layer_cache(layer_id)
            new_recurrent_layer = cur_recurrent_layer.at[idx_arr].set(new_state)
            new_conv_layer = [cur_conv_layer[0].at[idx_arr].set(new_conv)]
            layers_recurrent.append(new_recurrent_layer)
            layers_conv.append(new_conv_layer)

        # `is` relation must hold: backend never wrote into pool's internal lists.
        for i in range(pool.num_layers):
            self.assertIs(pool.recurrent_buffers[i], recurrent_refs_before[i])
            for j in range(len(pool.conv_buffers[i])):
                self.assertIs(pool.conv_buffers[i][j], conv_refs_before[i][j])

        # The returned per-layer buffers must be NEW objects (functional update).
        for i in range(pool.num_layers):
            self.assertIsNot(layers_recurrent[i], recurrent_refs_before[i])
            self.assertIsNot(layers_conv[i][0], conv_refs_before[i][0])

        # Now persist via replace_buffer; only at this point do the pool's
        # list element references swap away from the pre-call captures.
        # Note: replace_buffer goes through jax.device_put on the single-device
        # CPU path, which may re-wrap the array — so we check value equality
        # against the returned per-layer buffers rather than `is` identity.
        pool.replace_buffer((layers_recurrent, layers_conv))
        for i in range(pool.num_layers):
            self.assertIsNot(pool.recurrent_buffers[i], recurrent_refs_before[i])
            self.assertTrue(bool(jnp.array_equal(pool.recurrent_buffers[i], layers_recurrent[i])))
            for j in range(len(pool.conv_buffers[i])):
                self.assertIsNot(pool.conv_buffers[i][j], conv_refs_before[i][j])
                self.assertTrue(bool(jnp.array_equal(pool.conv_buffers[i][j], layers_conv[i][j])))

        # Value check: layer 3 (idx 0) carries 3.0 / 3.5 at the written slots.
        for slot in slots:
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[0][slot] == 3.0)))
            self.assertTrue(bool(jnp.all(pool.conv_buffers[0][0][slot] == jnp.bfloat16(3.5))))
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[1][slot] == 7.0)))
            self.assertTrue(bool(jnp.all(pool.conv_buffers[1][0][slot] == jnp.bfloat16(7.5))))


class TestJitMultiLayerSharingContract(_Base):
    """RFC-0015 functional contract under JIT: the model layer collects
    per-layer backend outputs into ``(layers_recurrent, layers_conv)`` and
    returns them. Each layer's contribution must be preserved (no layer drops
    out) when the pool is later updated via replace_buffer. This replaces the
    earlier "layer N+1 reads layer N's setter write" contract, which no
    longer holds since the setter is gone.
    """

    def test_per_layer_updates_all_persist_via_replace_buffer(self):
        import functools

        pool = self._make(linear_recurrent_layer_ids=[3, 7])

        @functools.partial(jax.jit, donate_argnames=["sp"])
        def two_layer_forward(sp):
            # Mirror the model-level loop: per layer, read -> functional update
            # -> append the new buffer to the per-layer list. The pool itself
            # is never written to inside the JIT trace.
            layer_ids = [3, 7]
            layers_recurrent = []
            layers_conv = []
            for layer_id in layer_ids:
                cur_recurrent, cur_conv = sp.get_linear_recurrent_layer_cache(layer_id)
                indices = jnp.asarray([1], dtype=jnp.int32)
                new_state = jnp.full((1, 2, 4, 4), float(layer_id), dtype=jnp.float32)
                new_conv = jnp.full(
                    (1, 3, 24), jnp.bfloat16(float(layer_id) + 0.5), dtype=jnp.bfloat16
                )
                layers_recurrent.append(cur_recurrent.at[indices].set(new_state))
                layers_conv.append([cur_conv[0].at[indices].set(new_conv)])
            return layers_recurrent, layers_conv

        layers_recurrent, layers_conv = two_layer_forward(pool)
        pool.replace_buffer((layers_recurrent, layers_conv))

        # Both layers' updates must be visible after replace_buffer
        # (proves the model-layer collection did not drop earlier layers).
        self.assertEqual(float(pool.recurrent_buffers[0][1, 0, 0, 0]), 3.0)
        self.assertEqual(float(pool.recurrent_buffers[1][1, 0, 0, 0]), 7.0)
        self.assertEqual(float(pool.conv_buffers[0][0][1, 0, 0]), 3.5)
        self.assertEqual(float(pool.conv_buffers[1][0][1, 0, 0]), 7.5)


if __name__ == "__main__":
    unittest.main()
