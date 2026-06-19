"""Backend track-slot writeback for the recurrent extra-buffer (S5a PR#2).

When a track boundary lands in a forward, the KDA / GDN backends scatter the
SAME per-request FINAL state they already write to the running slot ALSO into
each request's track slot, gated on ``track_mask`` AND the ``idx == 0`` guard.
These pure-JAX scatters are exercised directly on a forced multi-device CPU
mesh (mirrors ``test_kda_attention`` / ``test_gdn_attention``):

* KDA ``set_ssm_track_state`` / ``set_conv_track_state``.
* GDN ``_scatter_track`` plus the 4 kernel sites that thread track args.
* OFF / None: with no track metadata the kernel scatter output is byte-identical
  to the no-track path (no extra shard_map, same numerics + graph).
* Lightning / GLA: track metadata present -> fail fast.

Track boundaries are forward ends (Task 5), so the running-scatter value IS the
boundary snapshot — there is no mid-forward extraction here.
"""

from __future__ import annotations

import os
import unittest

# Force a multi-device CPU mesh before JAX initializes. The CI cpu-test job sets
# USE_DEVICE_TYPE=cpu; standalone runs (suite's unittest runner) need the same
# pins so >=2 devices exist for the sharded track scatter.
if os.environ.get("USE_DEVICE_TYPE") == "cpu" or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.gdn.gated_delta import (
    _scatter_idx0_safe,
    _scatter_track,
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sgl_jax.srt.layers.attention.linear.lightning_backend import LightningAttnBackend
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Forced multi-device CPU mesh (dp=1, tp=2) over a 2-device subset. The CPU
# suite pins JAX_PLATFORMS=cpu + xla_force_host_platform_device_count=8. We put
# both devices on the "tensor" axis so the FULL pool buffer is visible per DP
# rank (absolute slot indices stay valid) while the head dim is sharded -- the
# same shard_map the production scatter runs through. The track scatter is pure
# JAX, so a 2-way tensor split exercises it.
_DEVICES = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
mesh = create_device_mesh(
    ici_parallelism=[1, len(_DEVICES)], dcn_parallelism=[1, 1], devices=_DEVICES
)
jax.sharding.set_mesh(mesh)


def _data(arr) -> jax.Array:
    return jax.device_put(np.asarray(arr), NamedSharding(mesh, P("data")))


class TestKDATrackScatter(unittest.TestCase):
    """KDA parallel track scatter onto the running-scatter result."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.backend = KDAAttnBackend(mesh=mesh)
        self.rng = np.random.default_rng(0)
        # 4 slots: idx 0 = dummy. recurrent [N, H, K, V]; conv [N, D, K-1].
        self.N, self.H, self.K, self.V = 4, 2, 8, 8
        self.D, self.Km1 = 6, 3
        self.ssm_buf = jnp.asarray(
            self.rng.standard_normal((self.N, self.H, self.K, self.V)), jnp.float32
        )
        self.conv_buf = jnp.asarray(
            self.rng.standard_normal((self.N, self.D, self.Km1)), jnp.float32
        )

    def test_ssm_track_mask_on_writes_slot(self):
        # Two requests routed to running slots 1,2; track slots 3, 0 (dummy).
        track_idx = _data(np.array([3, 1], dtype=np.int32))
        track_mask = _data(np.array([1, 1], dtype=np.int32))
        new_rec = jnp.asarray(self.rng.standard_normal((2, self.H, self.K, self.V)), jnp.float32)
        new_rec = jax.device_put(new_rec, NamedSharding(mesh, P("data", "tensor", None, None)))
        buf = jax.device_put(self.ssm_buf, NamedSharding(mesh, P("data", "tensor", None, None)))

        out = self.backend.set_ssm_track_state(buf, track_idx, track_mask, new_rec)
        out = np.asarray(out)
        # track slot 3 (mask=1, idx!=0) takes new_rec[0]; slot 1 takes new_rec[1].
        np.testing.assert_array_equal(out[3], np.asarray(new_rec)[0])
        np.testing.assert_array_equal(out[1], np.asarray(new_rec)[1])
        # untouched slot 2 unchanged.
        np.testing.assert_array_equal(out[2], np.asarray(self.ssm_buf)[2])

    def test_ssm_track_mask_off_preserves_slot(self):
        track_idx = _data(np.array([3, 2], dtype=np.int32))
        track_mask = _data(np.array([0, 0], dtype=np.int32))  # no boundary
        new_rec = jnp.asarray(self.rng.standard_normal((2, self.H, self.K, self.V)), jnp.float32)
        new_rec = jax.device_put(new_rec, NamedSharding(mesh, P("data", "tensor", None, None)))
        buf = jax.device_put(self.ssm_buf, NamedSharding(mesh, P("data", "tensor", None, None)))

        out = np.asarray(self.backend.set_ssm_track_state(buf, track_idx, track_mask, new_rec))
        np.testing.assert_array_equal(out[3], np.asarray(self.ssm_buf)[3])
        np.testing.assert_array_equal(out[2], np.asarray(self.ssm_buf)[2])

    def test_ssm_track_idx0_is_noop(self):
        track_idx = _data(np.array([0, 0], dtype=np.int32))  # dummy slot
        track_mask = _data(np.array([1, 1], dtype=np.int32))  # mask on but idx==0
        new_rec = jnp.asarray(self.rng.standard_normal((2, self.H, self.K, self.V)), jnp.float32)
        new_rec = jax.device_put(new_rec, NamedSharding(mesh, P("data", "tensor", None, None)))
        buf = jax.device_put(self.ssm_buf, NamedSharding(mesh, P("data", "tensor", None, None)))

        out = np.asarray(self.backend.set_ssm_track_state(buf, track_idx, track_mask, new_rec))
        np.testing.assert_array_equal(out[0], np.asarray(self.ssm_buf)[0])

    def test_conv_track_mask_on_writes_slot(self):
        track_idx = _data(np.array([3, 1], dtype=np.int32))
        track_mask = _data(np.array([1, 0], dtype=np.int32))
        new_conv = jnp.asarray(self.rng.standard_normal((2, self.D, self.Km1)), jnp.float32)
        new_conv = jax.device_put(new_conv, NamedSharding(mesh, P("data", "tensor", None)))
        buf = jax.device_put(self.conv_buf, NamedSharding(mesh, P("data", "tensor", None)))

        out = np.asarray(
            self.backend.set_conv_track_state([buf], track_idx, track_mask, new_conv)[0]
        )
        np.testing.assert_array_equal(out[3], np.asarray(new_conv)[0])  # mask on
        np.testing.assert_array_equal(out[1], np.asarray(self.conv_buf)[1])  # mask off


class TestGDNScatterTrack(unittest.TestCase):
    """GDN ``_scatter_track`` helper + the 4 kernel-site track scatters."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.rng = np.random.default_rng(1)

    def test_scatter_track_mask_and_idx_guard(self):
        buf = jnp.asarray(self.rng.standard_normal((4, 3, 5)), jnp.float32)
        val = jnp.asarray(self.rng.standard_normal((3, 3, 5)), jnp.float32)
        track_idx = jnp.array([2, 0, 3], dtype=np.int32)  # idx 0 = dummy
        track_mask = jnp.array([1, 1, 0], dtype=np.int32)  # row2 mask off
        out = np.asarray(_scatter_track(buf, track_idx, track_mask, val))
        np.testing.assert_array_equal(out[2], np.asarray(val)[0])  # mask on, idx!=0
        np.testing.assert_array_equal(out[0], np.asarray(buf)[0])  # idx==0 guard
        np.testing.assert_array_equal(out[3], np.asarray(buf)[3])  # mask off

    def test_conv_prefill_track_scatter(self):
        D, K, B = 4, 3, 2
        T = 6
        x = jnp.asarray(self.rng.standard_normal((D, T)), jnp.float32)
        weight = jnp.asarray(self.rng.standard_normal((D, K)), jnp.float32)
        conv_state = jnp.asarray(self.rng.standard_normal((4, D, K - 1)), jnp.float32)
        cu = jnp.array([0, 3, 6], dtype=np.int32)
        state_idx = jnp.array([1, 2], dtype=np.int32)
        has_init = jnp.array([True, True])
        track_idx = jnp.array([3, 0], dtype=np.int32)
        track_mask = jnp.array([1, 1], dtype=np.int32)

        _, base = jax_causal_conv1d_prefill(
            x, weight, None, cu, conv_state, state_idx, has_init, "silu"
        )
        _, tracked = jax_causal_conv1d_prefill(
            x,
            weight,
            None,
            cu,
            conv_state,
            state_idx,
            has_init,
            "silu",
            track_indices=track_idx,
            track_mask=track_mask,
        )
        base, tracked = np.asarray(base), np.asarray(tracked)
        # track slot 3 (req 0, mask on) takes req 0's running-slot final state.
        np.testing.assert_array_equal(tracked[3], base[1])
        np.testing.assert_array_equal(tracked[0], base[0])  # idx==0 untouched
        # running slots still match the no-track scatter.
        np.testing.assert_array_equal(tracked[1], base[1])
        np.testing.assert_array_equal(tracked[2], base[2])

    def test_conv_update_track_scatter(self):
        D, K, B = 4, 3, 2
        x = jnp.asarray(self.rng.standard_normal((B, D)), jnp.float32)
        weight = jnp.asarray(self.rng.standard_normal((D, K)), jnp.float32)
        conv_state = jnp.asarray(self.rng.standard_normal((4, D, K - 1)), jnp.float32)
        state_idx = jnp.array([1, 2], dtype=np.int32)
        track_idx = jnp.array([3, 0], dtype=np.int32)
        track_mask = jnp.array([1, 0], dtype=np.int32)

        _, base = jax_causal_conv1d_update(
            x, conv_state, state_idx, weight, None, "silu", jnp.array([True, True])
        )
        _, tracked = jax_causal_conv1d_update(
            x,
            conv_state,
            state_idx,
            weight,
            None,
            "silu",
            jnp.array([True, True]),
            track_indices=track_idx,
            track_mask=track_mask,
        )
        base, tracked = np.asarray(base), np.asarray(tracked)
        np.testing.assert_array_equal(tracked[3], base[1])  # req 0 -> track slot 3
        np.testing.assert_array_equal(tracked[1], base[1])  # running preserved

    def test_ragged_recurrent_track_scatter(self):
        n_kq, n_v, d_k, d_v = 2, 2, 4, 4
        conv_dim = 2 * n_kq * d_k + n_v * d_v
        T = 6
        mixed = jnp.asarray(self.rng.standard_normal((T, conv_dim)), jnp.float32)
        b = jnp.asarray(self.rng.standard_normal((T, n_v)), jnp.float32)
        a = jnp.asarray(self.rng.standard_normal((T, n_v)), jnp.float32)
        rec = jnp.asarray(self.rng.standard_normal((4, n_v, d_k, d_v)), jnp.float32)
        A_log = jnp.asarray(self.rng.standard_normal((n_v,)), jnp.float32)
        dt_bias = jnp.asarray(self.rng.standard_normal((n_v,)), jnp.float32)
        cu = jnp.array([0, 3, 6], dtype=np.int32)
        state_idx = jnp.array([1, 2], dtype=np.int32)
        has_init = jnp.array([True, True])
        track_idx = jnp.array([3, 0], dtype=np.int32)
        track_mask = jnp.array([1, 1], dtype=np.int32)

        base, _ = ragged_gated_delta_rule_ref(
            mixed,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu,
            state_idx,
            has_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        tracked, _ = ragged_gated_delta_rule_ref(
            mixed,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            cu,
            state_idx,
            has_init,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            track_indices=track_idx,
            track_mask=track_mask,
        )
        base, tracked = np.asarray(base), np.asarray(tracked)
        np.testing.assert_array_equal(tracked[3], base[1])  # req 0 -> track slot 3
        np.testing.assert_array_equal(tracked[0], base[0])  # idx==0 dummy untouched
        np.testing.assert_array_equal(tracked[1], base[1])  # running preserved

    def test_decode_recurrent_track_scatter(self):
        n_kq, n_v, d_k, d_v = 2, 2, 4, 4
        conv_dim = 2 * n_kq * d_k + n_v * d_v
        B = 2
        mixed = jnp.asarray(self.rng.standard_normal((B, conv_dim)), jnp.float32)
        b = jnp.asarray(self.rng.standard_normal((B, n_v)), jnp.float32)
        a = jnp.asarray(self.rng.standard_normal((B, n_v)), jnp.float32)
        rec = jnp.asarray(self.rng.standard_normal((4, n_v, d_k, d_v)), jnp.float32)
        A_log = jnp.asarray(self.rng.standard_normal((n_v,)), jnp.float32)
        dt_bias = jnp.asarray(self.rng.standard_normal((n_v,)), jnp.float32)
        state_idx = jnp.array([1, 2], dtype=np.int32)
        track_idx = jnp.array([3, 0], dtype=np.int32)
        track_mask = jnp.array([1, 1], dtype=np.int32)

        base, _ = decode_gated_delta_rule_ref(
            mixed,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            state_idx,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        tracked, _ = decode_gated_delta_rule_ref(
            mixed,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            state_idx,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            track_indices=track_idx,
            track_mask=track_mask,
        )
        base, tracked = np.asarray(base), np.asarray(tracked)
        np.testing.assert_array_equal(tracked[3], base[1])  # req 0 -> track slot 3
        np.testing.assert_array_equal(tracked[1], base[1])  # running preserved


class TestTrackScatterOffByteIdentical(unittest.TestCase):
    """track_indices=None -> identical scatter output to the no-track path."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.rng = np.random.default_rng(2)

    def test_decode_recurrent_none_equals_no_track(self):
        n_kq, n_v, d_k, d_v = 2, 2, 4, 4
        conv_dim = 2 * n_kq * d_k + n_v * d_v
        B = 2
        mixed = jnp.asarray(self.rng.standard_normal((B, conv_dim)), jnp.float32)
        b = jnp.asarray(self.rng.standard_normal((B, n_v)), jnp.float32)
        a = jnp.asarray(self.rng.standard_normal((B, n_v)), jnp.float32)
        rec = jnp.asarray(self.rng.standard_normal((4, n_v, d_k, d_v)), jnp.float32)
        A_log = jnp.asarray(self.rng.standard_normal((n_v,)), jnp.float32)
        dt_bias = jnp.asarray(self.rng.standard_normal((n_v,)), jnp.float32)
        state_idx = jnp.array([1, 2], dtype=np.int32)

        no_arg, no_out = decode_gated_delta_rule_ref(
            mixed,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            state_idx,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        none_arg, none_out = decode_gated_delta_rule_ref(
            mixed,
            b,
            a,
            rec,
            A_log,
            dt_bias,
            state_idx,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            track_indices=None,
            track_mask=None,
        )
        np.testing.assert_array_equal(np.asarray(no_arg), np.asarray(none_arg))
        np.testing.assert_array_equal(np.asarray(no_out), np.asarray(none_out))

    def test_conv_prefill_none_equals_no_track(self):
        D, K = 4, 3
        T = 6
        x = jnp.asarray(self.rng.standard_normal((D, T)), jnp.float32)
        weight = jnp.asarray(self.rng.standard_normal((D, K)), jnp.float32)
        conv_state = jnp.asarray(self.rng.standard_normal((4, D, K - 1)), jnp.float32)
        cu = jnp.array([0, 3, 6], dtype=np.int32)
        state_idx = jnp.array([1, 2], dtype=np.int32)
        has_init = jnp.array([True, True])

        _, no_arg = jax_causal_conv1d_prefill(
            x, weight, None, cu, conv_state, state_idx, has_init, "silu"
        )
        _, none_arg = jax_causal_conv1d_prefill(
            x,
            weight,
            None,
            cu,
            conv_state,
            state_idx,
            has_init,
            "silu",
            track_indices=None,
            track_mask=None,
        )
        np.testing.assert_array_equal(np.asarray(no_arg), np.asarray(none_arg))


class TestLightningTrackFailFast(unittest.TestCase):
    """GLA/Lightning fails fast when track metadata reaches the backend."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_track_metadata_raises(self):
        backend = LightningAttnBackend(mesh=mesh)
        backend.forward_metadata = LinearRecurrentAttnBackendMetadata(
            recurrent_track_indices=_data(np.array([1, 2], dtype=np.int32)),
            recurrent_track_mask=_data(np.array([1, 1], dtype=np.int32)),
        )
        layer = type("L", (), {"layer_id": 0})()
        fb = type("FB", (), {"forward_mode": None})()
        with self.assertRaises(NotImplementedError):
            backend(None, None, None, layer=layer, forward_batch=fb, recurrent_state_pool=None)


if __name__ == "__main__":
    unittest.main()
