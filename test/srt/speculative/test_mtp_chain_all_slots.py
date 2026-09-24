"""Slot contract of the per-step metadata shift (SGLANG_JAX_MTP_CHAIN_POOL=all).

For chained draft step i the rotated window row k must land in the slot step 0
used for row k + i (same token, same position), and the rows that fall off the
end must land in the i slots right after step 0's window. Checked through the
real metadata builders and the backend's slot mapping on an explicit CPU mesh,
including a window that straddles a page boundary.
"""

import os

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    )
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.dsa_sparse_backend import _spec_token_slots
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.speculative.draft_extend_fused import (
    _make_draft_extend_metadata,
    _shift_draft_extend_metadata,
)

PS = 128
N = 4  # draft window rows per request
STEPS = 3


def _mesh():
    devices = np.array(jax.devices()[:4]).reshape(1, 4)
    return jax.sharding.Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _window_metadata(mesh, alloc_np):
    """Host-built extend metadata of a draft-extend batch: N query rows per
    request, pages for ``alloc_np`` tokens (what the scheduler allocated)."""
    bs = len(alloc_np)
    pages = int(np.sum((alloc_np + PS - 1) // PS))
    backend = SimpleNamespace(mesh=mesh, page_size=PS, attention_data_partition_axis="data")
    batch = SimpleNamespace(
        dp_size=1,
        per_dp_bs_size=bs,
        seq_lens=alloc_np,
        extend_seq_lens=np.full((bs,), N, np.int32),
        cache_loc=np.arange(pages * PS, dtype=np.int32),
        forward_mode=ForwardMode.EXTEND,
    )
    return MLAAttentionBackend.get_forward_metadata(backend, batch)


class ChainAllSlotContractTest(unittest.TestCase):
    def test_shifted_windows(self):
        mesh = _mesh()
        # verified lengths: one mid-page, one whose step-0 window straddles a page end
        verified = np.array([300, 2 * PS - 2], np.int32)  # window rows at L-1 .. L+2
        base_np = verified + STEPS  # metadata seq_len of step 0 (verify_seq_lens + num_steps)
        alloc_np = verified + 8  # overlap scheduling keeps >= 2 * ALLOC_LEN_PER_DECODE ahead
        with jax.set_mesh(mesh):
            data = NamedSharding(mesh, P("data"))
            rep = NamedSharding(mesh, P())
            md_orig = _window_metadata(mesh, alloc_np)
            base = jax.device_put(jnp.asarray(base_np), data)
            alloc = jax.device_put(jnp.asarray(alloc_np), data)
            bs = len(verified)

            def slots(md):
                def _f(sl, cq, ck, pi):
                    return _spec_token_slots(sl, cq, ck, pi, bs * N, PS)

                return np.asarray(
                    jax.sharding.auto_axes(_f, out_sharding=rep)(
                        md.seq_lens, md.cu_q_lens, md.cu_kv_lens, md.page_indices
                    )
                ).reshape(bs, N)

            md0 = _make_draft_extend_metadata(md_orig, base, alloc, page_size=PS, dp_size=1)
            loc0 = slots(md0)
            self.assertTrue((loc0 >= 0).all(), loc0)
            for i in range(1, STEPS):
                seq_i, md_i = _shift_draft_extend_metadata(
                    md_orig, base, alloc, i, page_size=PS, dp_size=1
                )
                np.testing.assert_array_equal(np.asarray(seq_i), base_np + i)
                loc_i = slots(md_i)
                self.assertTrue((loc_i >= 0).all(), (i, loc_i))
                for r in range(bs):
                    # rows that keep a token: same slot as step 0 gave that token
                    np.testing.assert_array_equal(loc_i[r, : N - i], loc0[r, i:], (i, r))
                    # rows that fell off: the i slots right after step 0's window
                    np.testing.assert_array_equal(
                        loc_i[r, N - i :], loc0[r, -1] + 1 + np.arange(i), (i, r)
                    )
                    # never past what the scheduler allocated
                    page_of = loc_i[r] // PS
                    self.assertTrue((page_of < np.sum((alloc_np + PS - 1) // PS)).all())
            # padding request (seq_len 0) stays 0 / dropped
            base_pad = jax.device_put(jnp.asarray(np.array([base_np[0], 0], np.int32)), data)
            seq_p, md_p = _shift_draft_extend_metadata(
                md_orig, base_pad, alloc, 2, page_size=PS, dp_size=1
            )
            self.assertEqual(int(np.asarray(seq_p)[1]), 0)
            self.assertTrue((slots(md_p)[1] == -1).all())


class ChainAllHeadroomGuardTest(unittest.TestCase):
    def test_guard(self):
        from sgl_jax.srt.speculative.draft_extend_fused import chain_all_headroom_ok

        vsl = np.array([641, 300, 0], np.int32)  # third slot is padding
        # overlap scheduling: allocated >= committed + 8 -> plenty for 3 steps
        self.assertTrue(chain_all_headroom_ok(vsl + 8, vsl, 3))
        # exactly enough: window end (vsl + 3) + 2 more slots
        self.assertTrue(chain_all_headroom_ok(vsl + 5, vsl, 3))
        # one short on a live request -> fall back
        self.assertFalse(chain_all_headroom_ok(np.array([641 + 4, 300 + 8, 0], np.int32), vsl, 3))
        # padding request's allocation is ignored
        self.assertTrue(chain_all_headroom_ok(np.array([649, 308, 0], np.int32), vsl, 3))
        self.assertTrue(chain_all_headroom_ok(np.zeros(3, np.int32), np.zeros(3, np.int32), 3))


if __name__ == "__main__":
    unittest.main()
