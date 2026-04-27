import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh


def _mesh():
    """Single-device mesh with the canonical "tensor" axis name; matches the
    sharding axis RecurrentStatePool partitions H / proj_size on."""
    return Mesh(np.array(jax.devices()), ("tensor",))


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
            mesh=_mesh(),
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

    def test_free_slots_no_longer_attribute(self):
        """Lock-in: slot allocator state was moved to HybridReqToTokenPool;
        RecurrentStatePool must not regrow a free_slots attribute."""
        pool = self._make()
        self.assertFalse(hasattr(pool, "free_slots"))

    def test_proj_size_attribute(self):
        pool = self._make()
        self.assertEqual(pool.proj_size, 24)


class TestKHeadsAndKDimGqaApi(_Base):
    """num_k_heads / head_k_dim default to num_heads / head_dim (current
    Kimi-Linear convention) but accept explicit overrides for GQA-style
    linear-recurrent attention models. Mirrors sglang upstream
    KimiLinearStateShape.create() defaults."""

    def test_default_k_dims_collapse_to_num_heads_head_dim(self):
        """Backward-compat: omitting num_k_heads / head_k_dim falls back to V
        dims so existing Kimi-Linear callers see proj_size = 3*num_heads*head_dim."""
        pool = self._make(num_heads=4, head_dim=8)
        self.assertEqual(pool.num_k_heads, 4)
        self.assertEqual(pool.head_k_dim, 8)
        # proj_size = 4*8 + 2*(4*8) = 96
        self.assertEqual(pool.proj_size, 96)
        # Conv buffer last dim equals proj_size.
        self.assertEqual(pool.conv_buffers[0][0].shape[-1], 96)

    def test_explicit_gqa_k_dims_change_proj_size(self):
        """num_k_heads / head_k_dim explicitly set: proj_size reflects the
        override, decoupled from V dims."""
        pool = self._make(
            num_heads=4,
            head_dim=8,
            num_k_heads=2,
            head_k_dim=4,
        )
        self.assertEqual(pool.num_k_heads, 2)
        self.assertEqual(pool.head_k_dim, 4)
        # proj_size = 4*8 + 2*(2*4) = 48
        self.assertEqual(pool.proj_size, 48)
        # Conv buffer last dim matches the GQA proj_size.
        self.assertEqual(pool.conv_buffers[0][0].shape[-1], 48)


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
        for layer in range(pool.num_linear_recurrent_layers):
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
        self.assertEqual(pool2.num_linear_recurrent_layers, pool.num_linear_recurrent_layers)
        self.assertEqual(pool2.max_num_reqs, pool.max_num_reqs)
        self.assertEqual(pool2.num_heads, pool.num_heads)
        self.assertEqual(pool2.head_dim, pool.head_dim)
        self.assertEqual(pool2.conv_kernel_size, pool.conv_kernel_size)
        self.assertEqual(pool2.proj_size, pool.proj_size)
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


class TestClearSlot(_Base):
    """clear_slot helper used by HybridReqToTokenPool for clear-on-alloc."""

    def test_clear_slot_int_zeros_recurrent_and_conv(self):
        pool = self._make()
        # Stamp non-zero into slot 1 across every layer + every conv inner.
        for layer in range(pool.num_linear_recurrent_layers):
            pool.recurrent_buffers[layer] = pool.recurrent_buffers[layer].at[1].set(9.0)
            for inner in range(len(pool.conv_buffers[layer])):
                pool.conv_buffers[layer][inner] = (
                    pool.conv_buffers[layer][inner].at[1].set(jnp.bfloat16(2.5))
                )
        pool.clear_slot(1)
        for layer in range(pool.num_linear_recurrent_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer][1] == 0)))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner][1] == 0)))

    def test_clear_slot_list_zeros_multiple(self):
        pool = self._make()
        for layer in range(pool.num_linear_recurrent_layers):
            pool.recurrent_buffers[layer] = jnp.ones_like(pool.recurrent_buffers[layer])
            for inner in range(len(pool.conv_buffers[layer])):
                pool.conv_buffers[layer][inner] = jnp.ones_like(pool.conv_buffers[layer][inner])
        pool.clear_slot([1, 2])
        for layer in range(pool.num_linear_recurrent_layers):
            for slot in (1, 2):
                self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer][slot] == 0)))
                for inner in range(len(pool.conv_buffers[layer])):
                    self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner][slot] == 0)))
            # Untouched slots keep their previous content.
            for slot in (0, 3, 4):
                self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer][slot] == 1)))

    def test_clear_slot_does_not_touch_other_slots(self):
        pool = self._make()
        for layer in range(pool.num_linear_recurrent_layers):
            pool.recurrent_buffers[layer] = pool.recurrent_buffers[layer].at[3].set(7.0)
        pool.clear_slot(1)
        for layer in range(pool.num_linear_recurrent_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer][3] == 7)))

    def test_clear_slot_empty_iterable_is_noop(self):
        pool = self._make()
        # Refs captured before should equal refs captured after (no scatter issued).
        before_recurrent_refs = list(pool.recurrent_buffers)
        before_conv_refs = [list(inner) for inner in pool.conv_buffers]
        pool.clear_slot([])
        for layer in range(pool.num_linear_recurrent_layers):
            self.assertIs(pool.recurrent_buffers[layer], before_recurrent_refs[layer])
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertIs(pool.conv_buffers[layer][inner], before_conv_refs[layer][inner])


class TestNoSlotAllocatorState(_Base):
    """Lock-in: alloc / free / free_slots were moved to HybridReqToTokenPool.
    The buffer pool must NOT regrow them, even via a pytree roundtrip
    (a regression here would re-introduce the silent free_slots reset that
    motivated the refactor)."""

    def test_attributes_absent_on_fresh_pool(self):
        pool = self._make()
        self.assertFalse(hasattr(pool, "free_slots"))
        self.assertFalse(hasattr(pool, "alloc"))
        self.assertFalse(hasattr(pool, "free"))

    def test_attributes_absent_after_pytree_roundtrip(self):
        pool = self._make()
        leaves, treedef = jax.tree_util.tree_flatten(pool)
        pool2 = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertFalse(hasattr(pool2, "free_slots"))
        self.assertFalse(hasattr(pool2, "alloc"))
        self.assertFalse(hasattr(pool2, "free"))


class TestReplaceBufferAndClear(_Base):
    def test_replace_buffer_swaps_both_buffers(self):
        pool = self._make()
        new_recurrent = [jnp.ones_like(b) * 3.0 for b in pool.recurrent_buffers]
        new_conv = [
            [jnp.ones_like(c) * jnp.bfloat16(4.0) for c in inner] for inner in pool.conv_buffers
        ]
        pool.replace_buffer((new_recurrent, new_conv))
        for layer in range(pool.num_linear_recurrent_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer] == 3)))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner] == jnp.bfloat16(4))))

    def test_replace_buffer_preserves_dtypes(self):
        pool = self._make()
        new_recurrent = [jnp.zeros_like(b) for b in pool.recurrent_buffers]
        new_conv = [[jnp.zeros_like(c) for c in inner] for inner in pool.conv_buffers]
        pool.replace_buffer((new_recurrent, new_conv))
        for layer in range(pool.num_linear_recurrent_layers):
            self.assertEqual(pool.recurrent_buffers[layer].dtype, jnp.float32)
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertEqual(pool.conv_buffers[layer][inner].dtype, jnp.bfloat16)

    def test_replace_buffer_does_not_touch_buffer_shapes(self):
        pool = self._make()
        # Sanity: replace_buffer of all-zero buffers leaves shapes intact and
        # does not depend on any allocator state living on the pool.
        new_recurrent = [jnp.zeros_like(b) for b in pool.recurrent_buffers]
        new_conv = [[jnp.zeros_like(c) for c in inner] for inner in pool.conv_buffers]
        pool.replace_buffer((new_recurrent, new_conv))
        for layer in range(pool.num_linear_recurrent_layers):
            self.assertEqual(pool.recurrent_buffers[layer].shape, (5, 2, 4, 4))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertEqual(pool.conv_buffers[layer][inner].shape, (5, 3, 24))

    def test_replace_buffer_assert_recurrent_length_mismatch(self):
        """recurrent list length must equal num_linear_recurrent_layers."""
        pool = self._make()  # num_linear_recurrent_layers = 2
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

    def test_clear_zeros_both_buffers(self):
        pool = self._make()
        # Write non-zero via list element mutation (matches implementation style).
        for layer in range(pool.num_linear_recurrent_layers):
            pool.recurrent_buffers[layer] = jnp.ones_like(pool.recurrent_buffers[layer])
            for inner in range(len(pool.conv_buffers[layer])):
                pool.conv_buffers[layer][inner] = jnp.ones_like(pool.conv_buffers[layer][inner])
        pool.clear()
        for layer in range(pool.num_linear_recurrent_layers):
            self.assertTrue(bool(jnp.all(pool.recurrent_buffers[layer] == 0)))
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertTrue(bool(jnp.all(pool.conv_buffers[layer][inner] == 0)))


class TestLayersMapping(_Base):
    """linear_recurrent_layer_ids -> layers_mapping translation."""

    def test_mapping_for_default_consecutive_ids(self):
        pool = self._make()  # [0, 1]
        self.assertEqual(pool.linear_recurrent_layer_ids, [0, 1])
        self.assertEqual(pool.layers_mapping, {0: 0, 1: 1})
        self.assertEqual(pool.num_linear_recurrent_layers, 2)

    def test_mapping_for_non_consecutive_ids(self):
        pool = self._make(linear_recurrent_layer_ids=[3, 7])
        self.assertEqual(pool.linear_recurrent_layer_ids, [3, 7])
        self.assertEqual(pool.layers_mapping, {3: 0, 7: 1})
        self.assertEqual(pool.num_linear_recurrent_layers, 2)
        # buffers length tracks num_linear_recurrent_layers, not the max id
        self.assertEqual(len(pool.recurrent_buffers), 2)
        self.assertEqual(len(pool.conv_buffers), 2)

    def test_mapping_for_kimi_linear_like_layout(self):
        # Kimi-Linear-48B layout: 27 layers total. Full attention layers
        # (0-indexed: {3, 7, 11, 15, 19, 23, 26}) are excluded; the remaining
        # 20 layers are KDA (recurrent).
        full_attn = {3, 7, 11, 15, 19, 23, 26}
        kda_layers = [i for i in range(27) if i not in full_attn]
        pool = self._make(linear_recurrent_layer_ids=kda_layers, max_num_reqs=2)
        self.assertEqual(pool.num_linear_recurrent_layers, 20)
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
        self.assertEqual(pool.num_linear_recurrent_layers, 0)
        self.assertEqual(pool.linear_recurrent_layer_ids, [])
        self.assertEqual(pool.layers_mapping, {})
        self.assertEqual(pool.recurrent_buffers, [])
        self.assertEqual(pool.conv_buffers, [])
        # clear / clear_slot must remain no-ops on the empty pool.
        pool.clear()
        pool.clear_slot([1, 2])
        self.assertEqual(pool.recurrent_buffers, [])


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
        self.assertEqual(pool2.num_linear_recurrent_layers, 2)
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
        """Backend uses functional .at[].set() return; pool exposes no setter."""
        pool = self._make()
        self.assertFalse(hasattr(pool, "set_linear_recurrent_layer_cache"))


class TestBackendFunctionalReturnNoPoolSideEffect(_Base):
    """Functional update pattern: backend reads via get, applies
    .at[indices].set(...) to build a new layer buffer, and returns it. The
    pool's internal recurrent_buffers / conv_buffers list elements stay
    referentially unchanged across the backend call. Persistence happens only
    when replace_buffer is invoked with the collected per-layer outputs.
    """

    def test_backend_call_does_not_mutate_pool_buffers(self):
        pool = self._make(linear_recurrent_layer_ids=[3, 7])
        # Slots are picked by HybridReqToTokenPool in production; for this
        # buffer-pool-only test we just pick the first two valid slots directly.
        slots = [1, 2]

        # Capture per-layer references before the simulated backend call.
        recurrent_refs_before = [
            pool.recurrent_buffers[i] for i in range(pool.num_linear_recurrent_layers)
        ]
        conv_refs_before = [
            [pool.conv_buffers[i][j] for j in range(len(pool.conv_buffers[i]))]
            for i in range(pool.num_linear_recurrent_layers)
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
        for i in range(pool.num_linear_recurrent_layers):
            self.assertIs(pool.recurrent_buffers[i], recurrent_refs_before[i])
            for j in range(len(pool.conv_buffers[i])):
                self.assertIs(pool.conv_buffers[i][j], conv_refs_before[i][j])

        # The returned per-layer buffers must be NEW objects (functional update).
        for i in range(pool.num_linear_recurrent_layers):
            self.assertIsNot(layers_recurrent[i], recurrent_refs_before[i])
            self.assertIsNot(layers_conv[i][0], conv_refs_before[i][0])

        # Now persist via replace_buffer; only at this point do the pool's
        # list element references swap away from the pre-call captures.
        # Note: replace_buffer goes through jax.device_put on the single-device
        # CPU path, which may re-wrap the array — so we check value equality
        # against the returned per-layer buffers rather than `is` identity.
        pool.replace_buffer((layers_recurrent, layers_conv))
        for i in range(pool.num_linear_recurrent_layers):
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
    """Functional contract under JIT: the model layer collects
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


class TestShardingPersistence(_Base):
    """Sharding lock-in: buffers ship with the recurrent_sharding /
    conv_sharding NamedSharding from _create_buffers, and clear() preserves
    that sharding via jnp.zeros_like (decision 3 sanity check; if a future
    JAX release stops propagating sharding through zeros_like, this test
    will fail and force the implementation to switch to _create_buffers
    rebuild)."""

    def test_initial_buffers_carry_persisted_sharding(self):
        pool = self._make()
        for layer in range(pool.num_linear_recurrent_layers):
            self.assertEqual(
                pool.recurrent_buffers[layer].sharding,
                pool.recurrent_sharding,
                f"layer {layer} recurrent buffer must ship with recurrent_sharding",
            )
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertEqual(
                    pool.conv_buffers[layer][inner].sharding,
                    pool.conv_sharding,
                    f"layer {layer} conv[{inner}] must ship with conv_sharding",
                )

    def test_clear_preserves_sharding(self):
        """Decision 3: clear() uses jnp.zeros_like which must preserve the
        sharded array's sharding metadata. If JAX behavior changes, the
        cleared buffer would lose its NamedSharding -- caught here."""
        pool = self._make()
        # Mutate buffers first so clear() actually has work to do.
        pool.recurrent_buffers[0] = pool.recurrent_buffers[0].at[1].set(1.5)
        pool.conv_buffers[0][0] = pool.conv_buffers[0][0].at[1].set(jnp.bfloat16(2.5))

        pool.clear()

        for layer in range(pool.num_linear_recurrent_layers):
            self.assertEqual(
                pool.recurrent_buffers[layer].sharding,
                pool.recurrent_sharding,
                f"clear() must preserve recurrent_sharding on layer {layer}",
            )
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertEqual(
                    pool.conv_buffers[layer][inner].sharding,
                    pool.conv_sharding,
                    f"clear() must preserve conv_sharding on layer {layer}[{inner}]",
                )


class TestReplaceBufferTpDegenerateGuard(unittest.TestCase):
    """Lock-in for the issue #233 fix trigger condition: replace_buffer
    applies the device_put sharding restore ONLY when the tensor axis is
    degenerate (mesh.shape["tensor"] == 1), matching ServerArgs tp_size==1.

    aolemila's original fix in _set_kv_cache_after_forward (commit 30f405ec)
    used `if self.tp_size == 1`. When the fix moved into the pool, the
    trigger MUST stay equivalent: we read mesh.shape["tensor"] (== tp_size)
    and NOT len(kv_sharding.device_set) (== mesh total devices), because
    the two diverge in multi-device tp_size=1 deployments.
    """

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _src(self, module_path):
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(module_path))

    def test_mha_pool_uses_tensor_axis_size_not_device_set(self):
        src = self._src("sgl_jax.srt.mem_cache.memory_pool")
        # Positive: the new tensor-axis check exists.
        self.assertIn('self.mesh.shape.get("tensor", 1) == 1', src)
        # Negative: the old device_set check is gone (would silently misfire
        # on multi-device tp_size=1 deployments).
        self.assertNotIn("len(self.kv_sharding.device_set) == 1", src)

    def test_recurrent_pool_uses_tensor_axis_size_not_device_set(self):
        src = self._src("sgl_jax.srt.mem_cache.recurrent_state_pool")
        self.assertIn('self.mesh.shape.get("tensor", 1) == 1', src)
        self.assertNotIn("len(self.recurrent_sharding.device_set) == 1", src)
        self.assertNotIn("len(self.conv_sharding.device_set) == 1", src)


class TestPytreeRoundtripPreservesMeshAndSharding(_Base):
    """Lock-in: tree_flatten aux must carry mesh + partition axis names +
    NamedSharding objects so JIT donate cycles (which round-trip through
    flatten / unflatten) don't drop sharding metadata. This is the same class
    of bug  hit when layers_mapping was missing from aux."""

    def test_aux_is_hashable(self):
        """JAX requires aux_data to be hashable (it lives in the treedef)."""
        pool = self._make()
        _, aux = pool.tree_flatten()
        # Will raise TypeError if any aux element is unhashable.
        hash(aux)

    def test_roundtrip_preserves_mesh_and_partition_axes(self):
        pool = self._make()
        leaves, treedef = jax.tree_util.tree_flatten(pool)
        pool2 = jax.tree_util.tree_unflatten(treedef, leaves)

        self.assertIs(pool2.mesh, pool.mesh)
        self.assertEqual(pool2.recurrent_partition_axis, pool.recurrent_partition_axis)
        self.assertEqual(pool2.conv_partition_axis, pool.conv_partition_axis)

    def test_roundtrip_preserves_sharding_specs(self):
        pool = self._make()
        leaves, treedef = jax.tree_util.tree_flatten(pool)
        pool2 = jax.tree_util.tree_unflatten(treedef, leaves)

        self.assertEqual(pool2.recurrent_sharding, pool.recurrent_sharding)
        self.assertEqual(pool2.conv_sharding, pool.conv_sharding)

    def test_roundtrip_preserves_per_layer_buffer_sharding(self):
        """After unflatten, the restored per-layer buffers must still report
        the original sharding -- otherwise replace_buffer's persisted-sharding
        device_put fix would drift from what JIT sees on the next trace."""
        pool = self._make()
        leaves, treedef = jax.tree_util.tree_flatten(pool)
        pool2 = jax.tree_util.tree_unflatten(treedef, leaves)

        for layer in range(pool.num_linear_recurrent_layers):
            self.assertEqual(
                pool2.recurrent_buffers[layer].sharding,
                pool.recurrent_sharding,
            )
            for inner in range(len(pool.conv_buffers[layer])):
                self.assertEqual(
                    pool2.conv_buffers[layer][inner].sharding,
                    pool.conv_sharding,
                )


if __name__ == "__main__":
    unittest.main()
