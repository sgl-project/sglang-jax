"""Qwen4Exp's N-gram short conv as a peer conv state in RecurrentStatePool.

Layer 1 is both a GDN layer and the N-gram layer, so it carries two conv
states of different shapes. They are described the same way and allocated by
the same loop; order carries no meaning -- consumers ask by name.
"""

import unittest

import jax
import numpy as np
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.mem_cache.recurrent_state_pool import (
    LINEAR_CONV,
    SHORT_CONV,
    ConvStateSpec,
    RecurrentStatePool,
)
from sgl_jax.test.test_utils import CustomTestCase

LAYERS = [0, 1, 2, 4]  # 3 is full attention
PLE_LAYER = 1
NUM_HEADS = 4
HEAD_DIM = 8
CONV_KERNEL = 4
PROJ = NUM_HEADS * HEAD_DIM + 2 * (NUM_HEADS * HEAD_DIM)  # 96, GDN's channels
CHANNELS = 32  # the N-gram conv's, unrelated to PROJ
STATE_LEN = 9  # (4-1)*3, dilated
SIZE = 8

LINEAR = ConvStateSpec(LINEAR_CONV, tuple(LAYERS), PROJ, CONV_KERNEL - 1)
SHORT = ConvStateSpec(SHORT_CONV, (PLE_LAYER,), CHANNELS, STATE_LEN)


def _make_mesh():
    devices = np.array(jax.devices())
    return Mesh(
        devices[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_pool(conv_states=(LINEAR, SHORT)):
    return RecurrentStatePool(
        linear_recurrent_layer_ids=LAYERS,
        size=SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=CONV_KERNEL,
        mesh=_make_mesh(),
        conv_states=conv_states,
    )


class TestConvStateSpecs(CustomTestCase):
    def test_specs_are_peers(self):
        pool = _make_pool()
        self.assertEqual([s.name for s in pool.conv_specs], [LINEAR_CONV, SHORT_CONV])

    def test_derives_gdn_only_when_conv_states_is_none(self):
        """Every family but Qwen4Exp passes nothing and gets the old layout."""
        pool = _make_pool(conv_states=None)
        self.assertEqual([s.name for s in pool.conv_specs], [LINEAR_CONV])
        self.assertEqual(pool.conv_specs[0].channels, pool.proj_size)
        self.assertEqual(pool.conv_specs[0].state_len, CONV_KERNEL - 1)
        for layer in LAYERS:
            self.assertEqual(len(pool.get_linear_recurrent_layer_cache(layer)[1]), 1)

    def test_spec_order_does_not_matter(self):
        """Consumers ask by name, so the tuple order carries no meaning."""
        pool = _make_pool(conv_states=(SHORT, LINEAR))
        self.assertEqual(
            pool.get_linear_conv_state(PLE_LAYER).shape,
            (pool.total_slots, PROJ, CONV_KERNEL - 1),
        )
        self.assertEqual(
            pool.get_conv_state(PLE_LAYER, SHORT_CONV).shape,
            (pool.total_slots, CHANNELS, STATE_LEN),
        )

    def test_only_the_ple_layer_gets_the_second_buffer(self):
        pool = _make_pool()
        for layer in LAYERS:
            _, conv = pool.get_linear_recurrent_layer_cache(layer)
            self.assertEqual(len(conv), 2 if layer == PLE_LAYER else 1)
        self.assertEqual(
            pool.get_short_conv_state(PLE_LAYER).shape,
            (pool.total_slots, CHANNELS, STATE_LEN),
        )

    def test_the_two_conv_states_have_different_shapes(self):
        """Same layer, both conv state; confusing them is silent."""
        pool = _make_pool()
        gdn = pool.get_linear_conv_state(PLE_LAYER)
        short = pool.get_conv_state(PLE_LAYER, SHORT_CONV)
        self.assertEqual(gdn.shape, (pool.total_slots, PROJ, CONV_KERNEL - 1))
        self.assertEqual(short.shape, (pool.total_slots, CHANNELS, STATE_LEN))

    def test_unknown_name_or_layer_raises(self):
        pool = _make_pool()
        with self.assertRaises(ValueError):
            pool.get_conv_state(0, SHORT_CONV)  # layer has no such state
        with self.assertRaises(ValueError):
            pool.get_conv_state(PLE_LAYER, "nope")
        with self.assertRaises(ValueError):
            pool.get_conv_state(3, LINEAR_CONV)  # not a recurrent layer at all

    def test_rejects_a_layer_the_pool_does_not_own(self):
        """A PLE layer on full attention has no slot to hang the state on."""
        with self.assertRaises(AssertionError):
            _make_pool(conv_states=(LINEAR, ConvStateSpec(SHORT_CONV, (3,), CHANNELS, STATE_LEN)))

    def _overwrite(self, pool, name, value_fn):
        buf = np.asarray(pool.get_conv_state(PLE_LAYER, name)).copy()
        value_fn(buf)
        idx = pool.layers_mapping[PLE_LAYER]
        pool.conv_buffers[idx][pool.conv_buffer_index(PLE_LAYER, name)] = jax.device_put(
            buf, pool.conv_sharding
        )

    def test_copy_slots_clones_both_conv_states(self):
        """Fork must carry the N-gram history, not just GDN's."""
        pool = _make_pool()
        self._overwrite(pool, LINEAR_CONV, lambda b: b.__setitem__(2, 5.0))
        self._overwrite(pool, SHORT_CONV, lambda b: b.__setitem__(2, 7.0))

        src = np.zeros(pool.total_slots, np.int32)
        dst = np.zeros(pool.total_slots, np.int32)
        src[0], dst[0] = 2, 5
        sharding = jax.sharding.NamedSharding(
            pool.mesh, jax.sharding.PartitionSpec(pool.data_partition_axis)
        )
        _, new_conv = pool.copy_slots(jax.device_put(src, sharding), jax.device_put(dst, sharding))

        idx = pool.layers_mapping[PLE_LAYER]
        self.assertTrue(np.all(np.asarray(new_conv[idx][0])[5] == 5.0))
        self.assertTrue(np.all(np.asarray(new_conv[idx][1])[5] == 7.0))

    def test_clear_zeros_both(self):
        pool = _make_pool()
        self._overwrite(pool, SHORT_CONV, lambda b: b.__setitem__(slice(None), 3.0))
        pool.clear()
        self.assertTrue(np.all(np.asarray(pool.get_short_conv_state(PLE_LAYER)) == 0))

    def test_pytree_round_trip_keeps_the_specs(self):
        """The pool crosses a jit boundary; dropped aux loses the second state."""
        pool = _make_pool()
        leaves, treedef = jax.tree_util.tree_flatten(pool)
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertEqual([s.name for s in back.conv_specs], [LINEAR_CONV, SHORT_CONV])
        self.assertEqual(
            back.get_short_conv_state(PLE_LAYER).shape,
            (pool.total_slots, CHANNELS, STATE_LEN),
        )

    def test_specs_are_hashable(self):
        """aux_data lands in the jit cache key, so a list field would break it."""
        self.assertEqual(hash(SHORT), hash(SHORT))
        _, aux = _make_pool().tree_flatten()
        hash(aux)


class TestMemoryBudget(CustomTestCase):
    def _bytes(self, conv_states):
        from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
            _compute_recurrent_per_req_bytes,
        )

        return _compute_recurrent_per_req_bytes(
            num_layers=len(LAYERS),
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            conv_kernel_size=CONV_KERNEL,
            tp_size=1,
            temporal_dtype_bytes=4,
            conv_dtype_bytes=2,
            conv_states=conv_states,
        )

    def test_the_second_conv_state_is_charged(self):
        delta = self._bytes((LINEAR, SHORT)) - self._bytes(None)
        self.assertEqual(delta, CHANNELS * STATE_LEN * 2)

    def test_none_matches_an_explicit_gdn_only_list(self):
        self.assertEqual(self._bytes(None), self._bytes((LINEAR,)))

    def test_released_checkpoint_per_request_bytes(self):
        """Real numbers: N-gram 180 KiB against GDN's ~110 MiB."""
        from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
            _compute_recurrent_per_req_bytes,
        )

        layers = tuple(range(36))
        proj = 48 * 128 + 2 * (16 * 128)  # 10240
        kw = dict(
            num_layers=36,
            num_heads=48,
            head_dim=128,
            conv_kernel_size=4,
            tp_size=1,
            temporal_dtype_bytes=4,
            conv_dtype_bytes=2,
            num_k_heads=16,
            head_k_dim=128,
        )
        base = _compute_recurrent_per_req_bytes(**kw)
        total = _compute_recurrent_per_req_bytes(
            **kw,
            conv_states=(
                ConvStateSpec(LINEAR_CONV, layers, proj, 3),
                ConvStateSpec(SHORT_CONV, (1,), 4 * 2560, 9),
            ),
        )
        self.assertEqual(total - base, 10240 * 9 * 2)  # 180 KiB
        self.assertEqual(total - base, 184320)
        self.assertLess((total - base) / base, 0.002)  # 0.16% of the per-req state


class TestForwardBatchPleField(CustomTestCase):
    def _batch(self, **kw):
        import jax.numpy as jnp

        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

        return ForwardBatch(
            bid=0,
            forward_mode=ForwardMode.DECODE,
            batch_size=2,
            input_ids=jnp.zeros((2,), jnp.int32),
            req_pool_indices=jnp.zeros((2,), jnp.int32),
            seq_lens=jnp.ones((2,), jnp.int32),
            out_cache_loc=jnp.zeros((2,), jnp.int32),
            **kw,
        )

    def test_ple_embeddings_survives_a_pytree_round_trip(self):
        """A field left out of tree_flatten silently becomes None inside jit."""
        import jax.numpy as jnp

        fb = self._batch(ple_embeddings=jnp.arange(8, dtype=jnp.float32).reshape(2, 4))
        leaves, treedef = jax.tree_util.tree_flatten(fb)
        back = jax.tree_util.tree_unflatten(treedef, leaves)
        np.testing.assert_array_equal(
            np.asarray(back.ple_embeddings), np.asarray(fb.ple_embeddings)
        )

    def test_absent_ple_embeddings_stays_none(self):
        fb = self._batch()
        leaves, treedef = jax.tree_util.tree_flatten(fb)
        self.assertIsNone(jax.tree_util.tree_unflatten(treedef, leaves).ple_embeddings)


class TestConfigSuppliesBothSpecs(CustomTestCase):
    """conv_state_specs is a sibling of linear_state_params, not nested in it."""

    def test_the_two_are_siblings(self):
        from sgl_jax.srt.configs.qwen4_exp import _Qwen4ExpTextConfig

        cfg = _Qwen4ExpTextConfig(ple_layer_ids=[2], ple_embed_dim=2560)
        self.assertFalse(hasattr(cfg.linear_state_params, "conv_states"))
        self.assertFalse(hasattr(cfg.linear_state_params, "short_conv"))
        self.assertEqual(len(cfg.conv_state_specs), 2)

    def test_qwen4_exp_conv_state_specs(self):
        from sgl_jax.srt.configs.qwen4_exp import _Qwen4ExpTextConfig

        cfg = _Qwen4ExpTextConfig(ple_layer_ids=[2], ple_embed_dim=2560)
        specs = cfg.conv_state_specs
        self.assertEqual([s.name for s in specs], [LINEAR_CONV, SHORT_CONV])

        gdn, short = specs
        self.assertEqual(gdn.channels, 10240)  # 48*128 + 2*16*128
        self.assertEqual(gdn.state_len, cfg.linear_conv_kernel_dim - 1)  # 3
        self.assertEqual(len(gdn.layers), 36)

        # ple_layer_ids is 1-based; the pool keys on 0-based decoder indices.
        self.assertEqual(short.layers, (1,))
        self.assertIn(1, gdn.layers)  # and layers.1 is also a GDN layer
        self.assertEqual(short.channels, cfg.hidden_size * cfg.hc_count)  # 10240
        self.assertEqual(short.state_len, (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size)  # 9
        # Equal channel counts are a coincidence; the lengths are what differ.
        self.assertEqual(gdn.channels, short.channels)
        self.assertNotEqual(gdn.state_len, short.state_len)

    def test_no_ple_layer_means_gdn_only(self):
        from sgl_jax.srt.configs.qwen4_exp import _Qwen4ExpTextConfig

        specs = _Qwen4ExpTextConfig().conv_state_specs
        self.assertEqual([s.name for s in specs], [LINEAR_CONV])


if __name__ == "__main__":
    unittest.main()
