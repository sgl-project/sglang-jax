"""CPU tests for the single-layer MTP chain hidden-state relay (SGLANG_JAX_MTP_HIDDEN_RELAY).

GLM-5.2 ships one MTP block that the fused draft loops apply ``num_steps`` times.
The block consumes ``(h_i, tok_{i+1})`` pairs; after ``_rotate_input_ids`` shifts the
token window left by one, the hidden window must shift with it and the last slot must
take the previous step's output hidden (the draft's own ``h_{t+j}``), exactly like
sglang's EAGLE draft loop (``hidden_states = logits_output.hidden_states``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.speculative import draft_extend_fused as def_mod
from sgl_jax.srt.speculative.draft_extend_fused import (
    _rotate_hidden,
    _rotate_input_ids,
    _rotate_prefill_hidden,
    _rotate_prefill_input_ids,
)
from sgl_jax.srt.utils.jax_utils import device_array

H = 8


def _mesh(dp):
    devices = np.array(jax.devices()[:4]).reshape(dp, 4 // dp)
    return jax.sharding.Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _rows(bs, tpr, tag):
    # row value = tag*1000 + req*10 + slot, broadcast over H -> any mix-up is visible
    v = (tag * 1000 + np.arange(bs)[:, None] * 10 + np.arange(tpr)[None, :]).astype(np.float32)
    return np.repeat(v.reshape(bs * tpr, 1), H, axis=1)


class RotateHiddenTest(unittest.TestCase):
    def _case(self, dp):
        mesh = _mesh(dp)
        bs, tpr = 2 * dp, 4
        ext_np = np.full((bs,), tpr, np.int32)
        ext_np[-1] = 2  # short window
        if bs > 2:
            ext_np[1] = 0  # padding request keeps its rows
        sel_np = np.clip(ext_np - 1, 0, None).astype(np.int32)
        hid_np, prev_np = _rows(bs, tpr, 1), _rows(bs, tpr, 2)
        sh2 = NamedSharding(mesh, P("data", None))
        sh1 = NamedSharding(mesh, P("data"))
        with jax.set_mesh(mesh):
            hid = device_array(hid_np, sharding=sh2)
            prev = device_array(prev_np, sharding=sh2)
            ext = device_array(ext_np, sharding=sh1)
            sel = device_array(sel_np, sharding=sh1)
            out = jax.jit(_rotate_hidden)(hid, ext, sel, prev)
        self.assertEqual(out.sharding.spec, sh2.spec)
        got = np.asarray(out).reshape(bs, tpr, H)[:, :, 0]
        h2, p2 = hid_np.reshape(bs, tpr, H)[:, :, 0], prev_np.reshape(bs, tpr, H)[:, :, 0]
        for b in range(bs):
            if ext_np[b] == 0:
                np.testing.assert_array_equal(got[b], h2[b])
                continue
            for k in range(tpr):
                if k == sel_np[b]:
                    self.assertEqual(got[b, k], p2[b, k], f"req{b} last slot takes prev output")
                elif k < tpr - 1:
                    self.assertEqual(got[b, k], h2[b, k + 1], f"req{b} slot {k} shifts left")

    def test_dp1(self):
        self._case(1)

    def test_dp2(self):
        self._case(2)

    def test_three_step_chain_pairs(self):
        # Simulate the fused loop with a fake MTP block: output hidden = 100*h + tok.
        bs, tpr, steps = 1, 4, 3
        ext = jnp.array([tpr], jnp.int32)
        sel = ext - 1
        target_hidden = jnp.repeat(jnp.arange(tpr, dtype=jnp.float32)[:, None] + 10.0, H, axis=1)
        ids = jnp.arange(tpr, dtype=jnp.int32) + 100  # tokens t-2..t+1 (last = verified)
        hidden, drafts, inputs = target_hidden, [], []
        for j in range(steps):
            inputs.append((np.asarray(hidden)[:, 0].copy(), np.asarray(ids).copy()))
            out_hidden = 100.0 * hidden + ids[:, None].astype(jnp.float32)
            d = jnp.array([1000 + j], jnp.int32)  # draft token d_{j+1}
            drafts.append(int(d[0]))
            if j < steps - 1:
                ids = _rotate_input_ids(ids, ext, sel, d)
                hidden = _rotate_hidden(hidden, ext, sel, out_hidden)
        # step 0: target hidden, verified id in the last slot
        np.testing.assert_array_equal(inputs[0][0], np.arange(tpr) + 10.0)
        for j in range(1, steps):
            prev_h, prev_ids = inputs[j - 1]
            cur_h, cur_ids = inputs[j]
            # last slot: hidden = previous step's OUTPUT at the last slot, token = d_j
            self.assertEqual(cur_h[-1], 100.0 * prev_h[-1] + prev_ids[-1])
            self.assertEqual(int(cur_ids[-1]), drafts[j - 1])
            # other slots: both windows shift left together, so the pairs stay aligned
            np.testing.assert_array_equal(cur_h[:-1], prev_h[1:])
            np.testing.assert_array_equal(cur_ids[:-1], prev_ids[1:])


class RotatePrefillHiddenTest(unittest.TestCase):
    def _case(self, dp):
        mesh = _mesh(dp)
        per_dp_bs, per_dp_tokens = 2, 8
        bs = dp * per_dp_bs
        ext_np = np.tile(np.array([3, 4], np.int32), dp)  # ragged: 3 + 4 tokens, 1 pad row
        n = dp * per_dp_tokens
        hid_np = np.repeat((np.arange(n) + 10.0).astype(np.float32)[:, None], H, axis=1)
        prev_np = np.repeat((np.arange(n) + 500.0).astype(np.float32)[:, None], H, axis=1)
        sh2 = NamedSharding(mesh, P("data", None))
        sh1 = NamedSharding(mesh, P("data"))
        with jax.set_mesh(mesh):
            hid = device_array(hid_np, sharding=sh2)
            prev = device_array(prev_np, sharding=sh2)
            ext = device_array(ext_np, sharding=sh1)
            out = jax.jit(lambda h, e, p: _rotate_prefill_hidden(h, e, p, dp, per_dp_bs))(
                hid, ext, prev
            )
            # same layout contract as the id rotation
            ids = device_array(np.arange(n, dtype=np.int32), sharding=sh1)
            ver = device_array(np.full((bs,), 999, np.int32), sharding=sh1)
            rot_ids = jax.jit(lambda i, e, v: _rotate_prefill_input_ids(i, e, v, dp, per_dp_bs))(
                ids, ext, ver
            )
        self.assertEqual(out.sharding.spec, sh2.spec)
        got = np.asarray(out)[:, 0]
        rid = np.asarray(rot_ids)
        for r in range(dp):
            base = r * per_dp_tokens
            segs = [(base, base + 3), (base + 3, base + 7)]
            for a, b in segs:
                for t in range(a, b - 1):
                    self.assertEqual(got[t], hid_np[t + 1, 0])
                    self.assertEqual(rid[t], t + 1)
                self.assertEqual(got[b - 1], prev_np[b - 1, 0])  # last row: previous output
                self.assertEqual(rid[b - 1], 999)
            self.assertEqual(got[base + 7], hid_np[base + 7, 0])  # pad row untouched

    def test_dp1(self):
        self._case(1)

    def test_dp2(self):
        self._case(2)


class RelayGateTest(unittest.TestCase):
    def test_auto_by_chaining(self):
        # single block chained 3 times (GLM-5.2) -> relay; one block per step -> no relay
        with mock.patch.object(def_mod, "_HIDDEN_RELAY_ENV", None):
            self.assertIsNone(
                def_mod.mtp_hidden_relay_enabled(SimpleNamespace(num_nextn_predict_layers=1))
            )
            self.assertTrue(def_mod._chained_relay(None, num_steps=3, num_blocks=1))
            self.assertFalse(def_mod._chained_relay(None, num_steps=3, num_blocks=3))
            self.assertFalse(def_mod._chained_relay(None, num_steps=1, num_blocks=1))

    def test_env_override(self):
        with mock.patch.object(def_mod, "_HIDDEN_RELAY_ENV", "0"):
            self.assertFalse(def_mod._chained_relay(def_mod.mtp_hidden_relay_enabled(None), 3, 1))
        with mock.patch.object(def_mod, "_HIDDEN_RELAY_ENV", "1"):
            self.assertTrue(def_mod._chained_relay(def_mod.mtp_hidden_relay_enabled(None), 3, 3))


if __name__ == "__main__":
    unittest.main()
