"""CPU tests for GLM-5.2 MTP IndexShare in the fused draft loop.

Covers the step-0 seed broadcast used by ``_build_draft_extend`` (single step,
multi step, cross-request isolation, padding requests) and the enable gate.
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

from sgl_jax.srt.speculative.draft_extend_fused import (
    _seed_topk_pages_from_step0,
    mtp_index_share_enabled,
)
from sgl_jax.srt.utils.jax_utils import device_array

# Production shape: explicit-sharding mesh with a size-1 "data" axis (dp=1). The
# helper runs inside the fused draft-extend JIT with data-sharded inputs; the
# size-1 axis is what turns a [T, k] -> [bs, tpr, k] reshape into P(None, data, ..)
# and broke the first TPU verify+draft step, so every case here goes through it.
_MESH = jax.sharding.Mesh(
    np.array(jax.devices()[:4]).reshape(1, 4),
    axis_names=("data", "tensor"),
    axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
)


def _pages(bs, tokens_per_req, k):
    # Distinct values per (request, row, slot) so any mix-up is visible.
    return jnp.arange(bs * tokens_per_req * k, dtype=jnp.int32).reshape(bs * tokens_per_req, k)


def _run_seed(pages, ext, sel):
    """Call the helper the way fused_draft_extend does: data-sharded, under JIT."""
    pages_sh = NamedSharding(_MESH, P("data", None))
    vec_sh = NamedSharding(_MESH, P("data"))
    with jax.set_mesh(_MESH):
        pages_d = device_array(np.asarray(pages, dtype=np.int32), sharding=pages_sh)
        ext_d = device_array(np.asarray(ext, dtype=np.int32), sharding=vec_sh)
        sel_d = device_array(np.asarray(sel, dtype=np.int32), sharding=vec_sh)
        out = jax.jit(_seed_topk_pages_from_step0)(pages_d, ext_d, sel_d)
    # The backend shard_maps the seed with P("data", None): placement must round-trip.
    assert out.sharding.spec == pages_sh.spec, out.sharding
    return np.asarray(out)


class TestSeedTopkPages(unittest.TestCase):
    def test_multi_step_rows_all_take_last_verified_row(self):
        bs, tpr, k = 3, 4, 5
        pages = _pages(bs, tpr, k)
        ext = jnp.array([4, 2, 3], dtype=jnp.int32)
        sel = jnp.clip(ext - 1, 0)
        out = _run_seed(pages, ext, sel).reshape(bs, tpr, k)
        src = np.asarray(pages).reshape(bs, tpr, k)
        for b in range(bs):
            for t in range(tpr):
                np.testing.assert_array_equal(out[b, t], src[b, int(sel[b])])

    def test_cross_request_isolation(self):
        bs, tpr, k = 4, 3, 2
        pages = _pages(bs, tpr, k)
        ext = jnp.array([3, 3, 3, 3], dtype=jnp.int32)
        sel = ext - 1
        out = _run_seed(pages, ext, sel).reshape(bs, tpr, k)
        # Every request's rows come from its own block, never a neighbour's.
        for b in range(bs):
            lo, hi = b * tpr * k, (b + 1) * tpr * k
            self.assertTrue(((out[b] >= lo) & (out[b] < hi)).all())

    def test_padding_requests_keep_step0_rows(self):
        bs, tpr, k = 2, 3, 2
        pages = _pages(bs, tpr, k)
        ext = jnp.array([3, 0], dtype=jnp.int32)
        sel = jnp.clip(ext - 1, 0)
        out = _run_seed(pages, ext, sel).reshape(bs, tpr, k)
        src = np.asarray(pages).reshape(bs, tpr, k)
        np.testing.assert_array_equal(out[1], src[1])
        np.testing.assert_array_equal(out[0], np.broadcast_to(src[0, 2], (tpr, k)))

    def test_single_row_window_is_identity(self):
        bs, tpr, k = 2, 1, 3
        pages = _pages(bs, tpr, k)
        ext = jnp.array([1, 1], dtype=jnp.int32)
        out = _run_seed(pages, ext, ext - 1)
        np.testing.assert_array_equal(out, np.asarray(pages))

    def test_minus_one_padding_preserved(self):
        pages = jnp.array([[7, -1], [3, -1], [9, 9], [-1, -1]], dtype=jnp.int32)
        ext = jnp.array([2, 2], dtype=jnp.int32)
        out = _run_seed(pages, ext, ext - 1)
        np.testing.assert_array_equal(out, [[3, -1], [3, -1], [-1, -1], [-1, -1]])


class TestEnableGate(unittest.TestCase):
    def test_follows_config_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_JAX_MTP_INDEX_SHARE", None)
            self.assertTrue(
                mtp_index_share_enabled(SimpleNamespace(index_share_for_mtp_iteration=True), 1)
            )
            self.assertFalse(
                mtp_index_share_enabled(SimpleNamespace(index_share_for_mtp_iteration=False), 1)
            )
            self.assertFalse(mtp_index_share_enabled(SimpleNamespace(), 1))
            self.assertFalse(mtp_index_share_enabled(None, 1))

    def test_topk_gt_one_disables(self):
        with mock.patch.dict(os.environ, {"SGLANG_JAX_MTP_INDEX_SHARE": "1"}):
            self.assertFalse(mtp_index_share_enabled(SimpleNamespace(), 2))

    def test_env_override(self):
        cfg = SimpleNamespace(index_share_for_mtp_iteration=True)
        with mock.patch.dict(os.environ, {"SGLANG_JAX_MTP_INDEX_SHARE": "0"}):
            self.assertFalse(mtp_index_share_enabled(cfg, 1))
        with mock.patch.dict(os.environ, {"SGLANG_JAX_MTP_INDEX_SHARE": "1"}):
            self.assertTrue(mtp_index_share_enabled(SimpleNamespace(), 1))


if __name__ == "__main__":
    unittest.main()
