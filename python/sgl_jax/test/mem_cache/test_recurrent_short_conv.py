"""Qwen4Exp's N-gram short conv as a peer conv state in RecurrentStatePool.

Layer 1 is both a GDN layer and the N-gram layer, so it carries two conv
states of different shapes. They are described the same way and allocated by
the same loop; order carries no meaning -- consumers ask by name.
"""

import unittest

import jax
import jax.numpy as jnp
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
        conv_dtype=jnp.float32,
        conv_states=conv_states,
    )


class TestConvStateSpecs(CustomTestCase):
    def test_named_ownership_and_lifecycle(self):
        for specs in (None, (LINEAR,), (LINEAR, SHORT), (SHORT, LINEAR)):
            with self.subTest(specs=specs):
                pool = _make_pool(specs)
                has_ple = specs is not None and SHORT in specs
                for layer in LAYERS:
                    self.assertEqual(
                        pool.get_linear_conv_state(layer).shape,
                        (pool.total_slots, PROJ, CONV_KERNEL - 1),
                    )
                    if layer != PLE_LAYER or not has_ple:
                        with self.assertRaises(ValueError):
                            pool.get_short_conv_state(layer)
                names = (LINEAR_CONV, SHORT_CONV) if has_ple else (LINEAR_CONV,)
                layer_index = pool.layers_mapping[PLE_LAYER]
                for value, name in enumerate(names, 5):
                    state = pool.get_conv_state(PLE_LAYER, name)
                    state = jax.device_put(
                        np.full(state.shape, value, np.float32), pool.conv_sharding
                    )
                    pool.conv_buffers[layer_index] = pool.with_conv_state(PLE_LAYER, name, state)
                if has_ple:
                    self.assertEqual(
                        pool.get_short_conv_state(PLE_LAYER).shape,
                        (pool.total_slots, CHANNELS, STATE_LEN),
                    )
                leaves, treedef = jax.tree_util.tree_flatten(pool)
                pool = jax.tree_util.tree_unflatten(treedef, leaves)
                src, dst = np.zeros(pool.total_slots, np.int32), np.zeros(
                    pool.total_slots, np.int32
                )
                src[0], dst[0] = 2, 5
                shard = jax.sharding.NamedSharding(pool.mesh, jax.sharding.PartitionSpec("data"))
                # Change just the source slot so copy_slots must really copy.
                for name in names:
                    state = np.asarray(pool.get_conv_state(PLE_LAYER, name)).copy()
                    state[2] += 10
                    state = jax.device_put(state, pool.conv_sharding)
                    pool.conv_buffers[layer_index] = pool.with_conv_state(PLE_LAYER, name, state)
                buffers = pool.copy_slots(jax.device_put(src, shard), jax.device_put(dst, shard))
                pool.replace_buffer(buffers)
                for value, name in enumerate(names, 5):
                    state = np.asarray(pool.get_conv_state(PLE_LAYER, name))
                    self.assertTrue(np.all(state[5] == value + 10))
                    self.assertTrue(np.all(state[[0, 1, 3, 4, 6, 7, 8]] == value))
                pool.clear()
                for name in names:
                    self.assertFalse(np.any(np.asarray(pool.get_conv_state(PLE_LAYER, name))))

    def test_invalid_ownership(self):
        pool = _make_pool()
        for layer, name in ((0, SHORT_CONV), (PLE_LAYER, "nope"), (3, LINEAR_CONV)):
            with self.subTest(layer=layer, name=name), self.assertRaises(ValueError):
                pool.get_conv_state(layer, name)
        with self.assertRaises(AssertionError):
            _make_pool((LINEAR, ConvStateSpec(SHORT_CONV, (3,), CHANNELS, STATE_LEN)))


class TestProductionStateConsumers(CustomTestCase):
    def test_gdn_and_kda_preserve_peer_state_in_either_order(self):
        # Reuse the existing attention fixtures, but call the real layer/backend
        # and runner writeback with both named state layouts, not a mock accessor.
        from sgl_jax.test.test_gdn_attention import create_test_data as gdn_data
        from sgl_jax.test.test_kda_attention import create_test_data as kda_data

        mesh = _make_mesh()
        for family, fixture, dims in (
            (
                "gdn",
                gdn_data,
                dict(
                    num_k_heads=NUM_HEADS,
                    num_v_heads=NUM_HEADS,
                    head_k_dim=HEAD_DIM,
                    head_v_dim=HEAD_DIM,
                ),
            ),
            ("kda", kda_data, dict(num_heads=NUM_HEADS, head_dim=HEAD_DIM)),
        ):
            with jax.set_mesh(mesh):
                fb, _, layer, q, k, v, a, b, *_ = fixture(
                    mode="decode",
                    seq_lens=[1, 1],
                    conv_kernel_size=CONV_KERNEL,
                    dtype=jnp.float32,
                    rng=np.random.default_rng(42),
                    test_mesh=mesh,
                    layer_id=PLE_LAYER,
                    all_have_initial_state=[False, True],
                    **dims,
                )
                shard = jax.sharding.NamedSharding(
                    mesh, jax.sharding.PartitionSpec("data", "tensor")
                )
                inputs = [jax.device_put(x, shard) for x in (q, k, v, a, b)]
                meta = fb.attn_backend.forward_metadata
                index_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
                meta.recurrent_track_indices = jax.device_put(
                    np.array([4, 5], np.int32), index_shard
                )
                meta.recurrent_track_mask = jax.device_put(np.array([True, False]), index_shard)
                run = jax.jit(
                    lambda pool, layer=layer, fb=fb, inputs=inputs: layer(fb, *inputs, pool)
                )
                reference = None
                for specs in (None, (LINEAR, SHORT), (SHORT, LINEAR)):
                    with self.subTest(family=family, specs=specs):
                        pool = _make_pool(specs)
                        index = pool.layers_mapping[PLE_LAYER]
                        pool.recurrent_buffers[index] = jnp.full_like(
                            pool.recurrent_buffers[index], 0.2
                        )
                        linear = jnp.full_like(pool.get_linear_conv_state(PLE_LAYER), 0.3)
                        pool.conv_buffers[index] = pool.with_conv_state(
                            PLE_LAYER, LINEAR_CONV, linear
                        )
                        if specs:
                            peer = jnp.full_like(pool.get_short_conv_state(PLE_LAYER), 7)
                            pool.conv_buffers[index] = pool.with_conv_state(
                                PLE_LAYER, SHORT_CONV, peer
                            )
                        results = []
                        for _ in range(2):  # feed written state back into the next forward
                            output, (rec, conv) = run(pool)
                            recurrent, convs = list(pool.recurrent_buffers), list(pool.conv_buffers)
                            recurrent[index], convs[index] = rec, conv
                            pool.replace_buffer((recurrent, convs))
                            linear = np.asarray(pool.get_linear_conv_state(PLE_LAYER))
                            results.append((np.asarray(output), np.asarray(rec), linear.copy()))
                            np.testing.assert_array_equal(linear[4], linear[1])  # track snapshot
                            self.assertTrue(np.all(linear[0] == 0.3))  # dummy untouched
                            self.assertTrue(np.all(linear[5] == 0.3))  # masked track untouched
                            if specs:
                                np.testing.assert_array_equal(
                                    np.asarray(pool.get_short_conv_state(PLE_LAYER)), 7
                                )
                        if reference is None:
                            reference = results
                        else:
                            for got_step, want_step in zip(results, reference, strict=True):
                                for got, want in zip(got_step, want_step, strict=True):
                                    np.testing.assert_array_equal(got, want)


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
        for specs in (None, (LINEAR,), (LINEAR, SHORT), (SHORT, LINEAR)):
            with self.subTest(specs=specs):
                extra = CHANNELS * STATE_LEN * 2 if specs and SHORT in specs else 0
                self.assertEqual(self._bytes(specs) - self._bytes(None), extra)

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


class TestConfigSuppliesBothSpecs(CustomTestCase):
    def test_config_specs_and_budget(self):
        from sgl_jax.srt.configs.qwen4_exp import _Qwen4ExpTextConfig

        for enabled in (False, True):
            with self.subTest(ple=enabled):
                cfg = _Qwen4ExpTextConfig(ple_layer_ids=[2] if enabled else [], ple_embed_dim=2560)
                specs = {spec.name: spec for spec in cfg.conv_state_specs}
                linear = specs[LINEAR_CONV]
                self.assertEqual(
                    (linear.channels, linear.state_len, len(linear.layers)), (10240, 3, 36)
                )
                self.assertEqual(
                    set(specs), {LINEAR_CONV, SHORT_CONV} if enabled else {LINEAR_CONV}
                )
                if enabled:
                    short = specs[SHORT_CONV]
                    self.assertEqual(short.layers, (1,))  # config is 1-based; pool is 0-based
                    self.assertIn(1, linear.layers)
                    self.assertEqual(
                        (short.channels, short.state_len), (cfg.hidden_size * cfg.hc_count, 9)
                    )


if __name__ == "__main__":
    unittest.main()
