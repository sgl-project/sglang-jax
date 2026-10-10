"""``a2a_merge_dcp_attention`` == gather-merge-then-slice, on CPU.

The a2a merge is on by default (``DSA_DCP_MERGE_A2A=1``) and is the only part of
the DCP decode rework that alters numerics, so this pins it against both
``merge_scatter_dcp_attention`` and a dense NumPy merge.

The interesting part is the packing, not the merge: the LSE is bitcast into the
*output* dtype and concatenated onto the feature axis, so with a bf16 ``o`` one
f32 LSE becomes **two** bf16 lanes and the trailing axis is ``dv + 2``. Serving
runs bf16, so the bf16 case is the one that matters and the f32 case (``dv + 1``)
is the one that accidentally works if the lane arithmetic is wrong. Both are
covered here.

Needs its own process for the device-count flag::

    JAX_PLATFORMS=cpu PYTHONPATH=python \\
      python -m sgl_jax.test.test_dcp_merge_a2a
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.dcp.comm import (
    a2a_merge_dcp_attention,
    merge_scatter_dcp_attention,
)
from sgl_jax.srt.layers.dcp.merge import merge_dcp_attention

DCP = 8
T, H_ALL, DV = 5, 16, 8
H_LOCAL = H_ALL // DCP


def _partials(dtype=np.float32):
    """Per-rank ``(o, lse)`` with a spread wide enough that weights differ."""
    rng = np.random.default_rng(7)
    o = rng.standard_normal((DCP, T, H_ALL, DV)).astype(np.float32)
    lse = (rng.standard_normal((DCP, T, H_ALL)) * 3.0).astype(np.float32)
    # rank 3 owns nothing anywhere; query-head (0, 0) is empty on every rank.
    lse[3, :, :] = -np.inf
    lse[:, 0, 0] = -np.inf
    o = np.where(np.isfinite(lse)[..., None], o, 0.0)
    return o.astype(dtype), lse


def _run(kernel, o, lse, out_dtype=None):
    mesh = Mesh(np.array(jax.devices()[:DCP]), ("tensor",))
    fn = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=(P("tensor", None, None, None), P("tensor", None, None)),
            out_specs=P(None, "tensor", None),
            check_vma=False,
        )
    )
    out = fn(jnp.asarray(o, dtype=out_dtype) if out_dtype else jnp.asarray(o), jnp.asarray(lse))
    return np.asarray(out.astype(jnp.float32))


def _a2a(po, pl):
    return a2a_merge_dcp_attention(po.squeeze(0), pl.squeeze(0), H_LOCAL)


def _scatter(po, pl):
    return merge_scatter_dcp_attention(po.squeeze(0), pl.squeeze(0), H_LOCAL)


@unittest.skipIf(jax.device_count() < DCP, f"needs {DCP} devices")
class TestDcpMergeA2A(unittest.TestCase):
    def test_f32_matches_dense_and_scatter(self):
        """dv + 1 lane case: one f32 LSE bitcasts to one f32 lane."""
        o, lse = _partials()
        ref_o, _ = merge_dcp_attention(o, lse)

        got = _run(_a2a, o, lse)
        self.assertEqual(got.shape, (T, H_ALL, DV))
        np.testing.assert_allclose(got, ref_o, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(got, _run(_scatter, o, lse), rtol=1e-5, atol=1e-5)

    def test_bf16_matches_dense(self):
        """dv + 2 lane case — the configuration serving actually runs.

        A wrong trailing-lane count here does not crash: the reshape still
        succeeds and the LSE is silently read out of the feature lanes, which
        would show up as subtly wrong attention output.
        """
        o, lse = _partials()
        ref_o, _ = merge_dcp_attention(o, lse)

        got = _run(_a2a, o, lse, out_dtype=jnp.bfloat16)
        self.assertEqual(got.shape, (T, H_ALL, DV))
        # bf16 inputs, f32 accumulation: ~3 decimal digits is all that is owed.
        np.testing.assert_allclose(got, ref_o, rtol=3e-2, atol=3e-2)

    def test_lse_survives_the_bitcast_round_trip(self):
        """Pin the packing itself, independent of the merge.

        If ``bitcast_convert_type`` and the ``dv:`` slice disagree, the recovered
        LSE is garbage; the merge would still run and produce plausible numbers.
        """
        for dt in (jnp.float32, jnp.bfloat16):
            with self.subTest(dtype=dt.__name__):
                rng = np.random.default_rng(3)
                lse = (rng.standard_normal((T, H_ALL)) * 3.0).astype(np.float32)
                bits = jax.lax.bitcast_convert_type(jnp.asarray(lse), dt)
                if bits.ndim == 2:
                    bits = bits[..., None]
                back = jax.lax.bitcast_convert_type(
                    bits if bits.shape[-1] > 1 else bits[..., 0], jnp.float32
                )
                np.testing.assert_array_equal(np.asarray(back).reshape(T, H_ALL), lse)

    def test_all_empty_query_head_is_zero(self):
        o, lse = _partials()
        got = _run(_a2a, o, lse)
        np.testing.assert_array_equal(got[0, 0], np.zeros(DV, np.float32))
        self.assertTrue(np.isfinite(got).all())

    def test_head_ownership_is_not_permuted(self):
        """Rank r must receive exactly the heads it owns.

        ``all_to_all(split_axis=1, concat_axis=0)`` then ``reshape(n, t, h_local, …)``
        is the one step that silently reorders heads if the axes are swapped, and a
        head permutation is wrong output rather than an error. Pinned by giving each
        (rank, head) a unique value and making a single rank the sole owner.
        """
        o = np.zeros((DCP, T, H_ALL, DV), np.float32)
        lse = np.full((DCP, T, H_ALL), -np.inf, np.float32)
        for h in range(H_ALL):
            owner = h // H_LOCAL  # rank that must end up holding head h
            lse[owner, :, h] = 0.0
            o[owner, :, h, :] = float(h + 1)

        got = _run(_a2a, o, lse)
        for h in range(H_ALL):
            np.testing.assert_allclose(
                got[:, h, :], np.full((T, DV), float(h + 1)), rtol=1e-6, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
