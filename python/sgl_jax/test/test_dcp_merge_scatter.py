"""``merge_scatter_dcp_attention`` == all-gather-merge-then-slice, on CPU.

The reduce-scatter merge replaces the all-gather that measured 71.9% of device
self time at 8k/DCP=16. It must be bit-comparable to the path it replaces, so
this pins it against both ``gather_merge_dcp_attention`` and a dense NumPy
merge, including empty shards (``lse = -inf``).

Needs its own process for the device-count flag::

    JAX_PLATFORMS=cpu PYTHONPATH=python \\
      python -m sgl_jax.test.test_dcp_merge_scatter
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
    allgather_heads,
    gather_merge_dcp_attention,
    merge_scatter_dcp_attention,
    slice_local_heads,
)
from sgl_jax.srt.layers.dcp.merge import merge_dcp_attention

DCP = 8
T, H_ALL, DV = 5, 16, 7
H_LOCAL = H_ALL // DCP


def _partials():
    """Per-rank ``(o, lse)`` with a spread wide enough that weights differ."""
    rng = np.random.default_rng(7)
    o = rng.standard_normal((DCP, T, H_ALL, DV)).astype(np.float32)
    lse = (rng.standard_normal((DCP, T, H_ALL)) * 3.0).astype(np.float32)
    # rank 3 owns nothing anywhere; query-head (0, 0) is empty on every rank.
    lse[3, :, :] = -np.inf
    lse[:, 0, 0] = -np.inf
    o = np.where(np.isfinite(lse)[..., None], o, 0.0).astype(np.float32)
    return o, lse


def _run(kernel, o, lse):
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
    return np.asarray(fn(jnp.asarray(o), jnp.asarray(lse)))


@unittest.skipIf(jax.device_count() < DCP, f"needs {DCP} devices")
class TestDcpMergeScatter(unittest.TestCase):
    def test_scatter_merge_matches_gather_merge_and_dense(self):
        o, lse = _partials()
        ref_o, _ = merge_dcp_attention(o, lse)

        got = _run(
            lambda po, pl: merge_scatter_dcp_attention(po.squeeze(0), pl.squeeze(0), H_LOCAL),
            o,
            lse,
        )
        gathered = _run(
            lambda po, pl: slice_local_heads(
                gather_merge_dcp_attention(po.squeeze(0), pl.squeeze(0)), H_LOCAL
            ),
            o,
            lse,
        )

        self.assertEqual(got.shape, (T, H_ALL, DV))
        # Same order of operations as the gather path, so this is tight.
        np.testing.assert_allclose(got, gathered, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(got, ref_o, rtol=1e-5, atol=1e-5)

    def test_all_empty_query_head_is_zero(self):
        o, lse = _partials()
        got = _run(
            lambda po, pl: merge_scatter_dcp_attention(po.squeeze(0), pl.squeeze(0), H_LOCAL),
            o,
            lse,
        )
        np.testing.assert_array_equal(got[0, 0], np.zeros(DV, np.float32))
        self.assertTrue(np.isfinite(got).all())

    def test_allgather_heads_order_matches_gather_then_transpose(self):
        """A silent head reorder here would be wrong output, not a crash."""
        rng = np.random.default_rng(11)
        d, dpe = 6, 3
        q = rng.standard_normal((DCP, T, H_LOCAL, d)).astype(np.float32)
        qpe = rng.standard_normal((DCP, T, H_LOCAL, dpe)).astype(np.float32)

        mesh = Mesh(np.array(jax.devices()[:DCP]), ("tensor",))

        def run(kernel):
            fn = jax.jit(
                jax.shard_map(
                    kernel,
                    mesh=mesh,
                    in_specs=(P("tensor", None, None, None),) * 2,
                    out_specs=P("tensor", None, None),
                    check_vma=False,
                )
            )
            # Every rank returns the same replicated array; keep rank 0's copy.
            return np.asarray(fn(jnp.asarray(q), jnp.asarray(qpe)))[:T]

        def ref(a, b):
            g = jax.lax.all_gather(a.squeeze(0), "tensor", axis=0)
            r, t, h, dd = g.shape
            return g.transpose(1, 0, 2, 3).reshape(t, r * h, dd)

        got = run(lambda a, b: allgather_heads(a.squeeze(0), b.squeeze(0))[0])
        np.testing.assert_array_equal(got, run(ref))
        # Rank r's heads must land at [r*H_LOCAL, (r+1)*H_LOCAL).
        for r in range(DCP):
            np.testing.assert_array_equal(got[:, r * H_LOCAL : (r + 1) * H_LOCAL], q[r])

        got_pe = run(lambda a, b: allgather_heads(a.squeeze(0), b.squeeze(0))[1])
        self.assertEqual(got_pe.shape, (T, H_LOCAL * DCP, dpe))
        for r in range(DCP):
            np.testing.assert_array_equal(got_pe[:, r * H_LOCAL : (r + 1) * H_LOCAL], qpe[r])

    def test_allgather_then_slice_is_the_identity(self):
        """slice_local_heads must undo allgather_heads on every rank."""
        rng = np.random.default_rng(12)
        d = 6
        q = rng.standard_normal((DCP, T, H_LOCAL, d)).astype(np.float32)
        mesh = Mesh(np.array(jax.devices()[:DCP]), ("tensor",))

        def kernel(a):
            full, _ = allgather_heads(a.squeeze(0), a.squeeze(0))
            return slice_local_heads(full, H_LOCAL)[None]

        fn = jax.jit(
            jax.shard_map(
                kernel,
                mesh=mesh,
                in_specs=P("tensor", None, None, None),
                out_specs=P("tensor", None, None, None),
                check_vma=False,
            )
        )
        np.testing.assert_array_equal(np.asarray(fn(jnp.asarray(q))), q)

    def test_bf16_scatter_is_close_but_provably_in_effect(self):
        """bf16 on the wire is the default: bounded drift, and demonstrably applied."""
        o, lse = _partials()
        f32 = _run(
            lambda po, pl: merge_scatter_dcp_attention(po.squeeze(0), pl.squeeze(0), H_LOCAL),
            o,
            lse,
        )
        bf16 = _run(
            lambda po, pl: merge_scatter_dcp_attention(
                po.squeeze(0), pl.squeeze(0), H_LOCAL, scatter_dtype=jnp.bfloat16
            ),
            o,
            lse,
        )
        scale = float(np.abs(f32).max())
        err = float(np.abs(bf16 - f32).max())
        # bf16 keeps 8 mantissa bits, so a DCP-way weighted sum should stay ~1%.
        self.assertLess(err, 0.02 * scale, f"bf16 merge drifted {err:.4f} of {scale:.4f}")
        # It must NOT be bit-identical, or scatter_dtype was silently ignored and
        # a "bf16 is free" measurement would really be measuring the f32 path.
        self.assertGreater(err, 0.0, "scatter_dtype=bfloat16 had no effect")
        # An all-empty query-head must still be exactly zero, not a rounded zero.
        np.testing.assert_array_equal(bf16[0, 0], np.zeros(DV, np.float32))

    def test_a_flat_merge_would_not_pass(self):
        """Guard the guard: the weights must actually differ across ranks."""
        o, lse = _partials()
        ref_o, _ = merge_dcp_attention(o, lse)
        finite = np.isfinite(lse)
        uniform = np.where(finite, 1.0 / np.maximum(finite.sum(0), 1), 0.0)
        flat = np.sum(o * uniform[..., None], axis=0)
        self.assertGreater(np.abs(flat - ref_o).max(), 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
