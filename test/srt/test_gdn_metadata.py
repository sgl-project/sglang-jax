"""Unit tests for :class:`GDNAttnMetadata` and :class:`GDNMetadataBuilder`.

The metadata carries per-forward packed-batch boundaries (``cu_seqlens``)
through JIT. Tests check the host-side cumsum construction, decode-vs-
extend dispatch, and pytree round-trip semantics that let it ride through
``ForwardBatch`` under ``jax.jit``.

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest test/srt/test_gdn_metadata.py -v
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.linear.gdn_metadata import (
    GDNAttnMetadata,
    GDNMetadataBuilder,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


# Minimal duck-typed batch — the builder only reads ``forward_mode`` and
# ``extend_seq_lens``.
class _FakeBatch:
    def __init__(self, forward_mode, extend_seq_lens=None):
        self.forward_mode = forward_mode
        self.extend_seq_lens = extend_seq_lens


class GDNAttnMetadataPytreeTest(unittest.TestCase):
    def test_round_trip_with_cu_seqlens(self):
        """tree_flatten / tree_unflatten preserve cu_seqlens contents."""
        cu = jnp.array([0, 3, 7], dtype=jnp.int32)
        md = GDNAttnMetadata(cu_seqlens=cu)
        children, aux = md.tree_flatten()
        rebuilt = GDNAttnMetadata.tree_unflatten(aux, children)
        np.testing.assert_array_equal(rebuilt.cu_seqlens, cu)

    def test_round_trip_with_none(self):
        """Decode batches carry cu_seqlens=None; pytree round-trip is identity."""
        md = GDNAttnMetadata()
        children, aux = md.tree_flatten()
        rebuilt = GDNAttnMetadata.tree_unflatten(aux, children)
        self.assertIsNone(rebuilt.cu_seqlens)

    def test_jit_through_pytree(self):
        """Passing GDNAttnMetadata through a jit'd function preserves the
        array — exercises the pytree registration on a real jit boundary."""

        @jax.jit
        def f(md):
            return md.cu_seqlens + 1

        cu = jnp.array([0, 2, 5], dtype=jnp.int32)
        out = f(GDNAttnMetadata(cu_seqlens=cu))
        np.testing.assert_array_equal(out, cu + 1)


class GDNMetadataBuilderTest(unittest.TestCase):
    def test_decode_returns_none_cu_seqlens(self):
        """DECODE batches carry no boundary metadata."""
        builder = GDNMetadataBuilder(mesh=None)
        batch = _FakeBatch(forward_mode=ForwardMode.DECODE)
        md = builder.get_forward_metadata(batch)
        self.assertIsInstance(md, GDNAttnMetadata)
        self.assertIsNone(md.cu_seqlens)

    def test_extend_builds_cumsum_from_extend_seq_lens(self):
        """EXTEND batches: cu_seqlens = [0, cumsum(extend_seq_lens)...]."""
        builder = GDNMetadataBuilder(mesh=None)
        batch = _FakeBatch(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=np.array([3, 5, 2], dtype=np.int32),
        )
        md = builder.get_forward_metadata(batch)
        np.testing.assert_array_equal(md.cu_seqlens, np.array([0, 3, 8, 10], dtype=np.int32))
        self.assertEqual(md.cu_seqlens.dtype, jnp.int32)

    def test_extend_single_request(self):
        """Smoke check: a single-request extend → cu_seqlens=[0, L]."""
        builder = GDNMetadataBuilder(mesh=None)
        batch = _FakeBatch(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=np.array([7], dtype=np.int32),
        )
        md = builder.get_forward_metadata(batch)
        np.testing.assert_array_equal(md.cu_seqlens, np.array([0, 7], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
