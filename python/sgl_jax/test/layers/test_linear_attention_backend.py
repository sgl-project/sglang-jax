"""Tests for LinearAttentionBackend and scatter/gather helpers.

Run with: pytest python/sgl_jax/test/layers/test_linear_attention_backend.py -v
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.fla.linear_attention_backend import (
    LinearAttentionBackend,
    LinearAttentionMetadata,
    gather_from_packed,
    scatter_to_packed,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

# ---------------------------------------------------------------------------
# Mock batch helpers
# ---------------------------------------------------------------------------


def _make_batch(forward_mode, extend_seq_lens, seq_lens, T_outer=None):
    esl = np.array(extend_seq_lens, dtype=np.int32) if extend_seq_lens is not None else None
    if T_outer is None and esl is not None:
        T_outer = int(np.sum(esl))
    return SimpleNamespace(
        forward_mode=forward_mode,
        extend_seq_lens=esl,
        seq_lens=np.array(seq_lens, dtype=np.int32),
        input_ids=np.zeros(T_outer or 0, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# cu_seqlens tests
# ---------------------------------------------------------------------------


class TestCuSeqlens:
    def test_single_aligned_request(self):
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.EXTEND, [64], [64])
        metadata = backend.get_forward_metadata(batch)
        expected = np.array([0, 64], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens_dev), expected)

    def test_single_unaligned_request(self):
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.EXTEND, [100], [100])
        metadata = backend.get_forward_metadata(batch)
        # ceil(100/64)*64 = 128
        expected = np.array([0, 128], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens_dev), expected)

    def test_multiple_requests(self):
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.EXTEND, [30, 64, 50], [30, 64, 50])
        metadata = backend.get_forward_metadata(batch)
        # 30->64, 64->64, 50->64; cumsum: [0,64,128,192]
        expected = np.array([0, 64, 128, 192], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens_dev), expected)

    def test_padded_batch_zero_length(self):
        backend = LinearAttentionBackend()
        # Third request has length 0 (padding slot)
        T_outer = 64 + 128  # tight sum of real requests
        batch = _make_batch(ForwardMode.EXTEND, [64, 128, 0], [64, 128, 0], T_outer=T_outer)
        metadata = backend.get_forward_metadata(batch)
        # 64->64, 128->128, 0->0; cumsum: [0,64,192,192]
        expected = np.array([0, 64, 192, 192], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens_dev), expected)


# ---------------------------------------------------------------------------
# T_packed_bucket tests
# ---------------------------------------------------------------------------


class TestTPackedBucket:
    def test_two_unaligned_requests(self):
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.EXTEND, [30, 50], [30, 50])
        backend.get_forward_metadata(batch)
        # 30->64, 50->64; total=128
        assert backend.T_packed_bucket == 128


# ---------------------------------------------------------------------------
# scatter_idx tests
# ---------------------------------------------------------------------------


class TestScatterIdx:
    def test_shape(self):
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.EXTEND, [30, 50], [30, 50], T_outer=128)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        assert idx.shape == (128,)

    def test_single_aligned_request(self):
        backend = LinearAttentionBackend()
        # T_outer == T_tight == 64, all tokens map to themselves
        batch = _make_batch(ForwardMode.EXTEND, [64], [64], T_outer=64)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        np.testing.assert_array_equal(idx[:64], np.arange(64, dtype=np.int32))

    def test_two_requests_mapping(self):
        backend = LinearAttentionBackend()
        # extend=[30,50], T_outer=128
        # request 0: seq_len=30 -> chunk slot [0..63], real tokens [0..29]
        # request 1: seq_len=50 -> chunk slot [64..127], real tokens [0..49]
        batch = _make_batch(ForwardMode.EXTEND, [30, 50], [30, 50], T_outer=128)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        T_pb = backend.T_packed_bucket  # 128

        # First 30 tokens map to packed positions 0..29
        np.testing.assert_array_equal(idx[:30], np.arange(0, 30, dtype=np.int32))
        # Next 50 tokens map to packed positions 64..113
        np.testing.assert_array_equal(idx[30:80], np.arange(64, 114, dtype=np.int32))
        # Remaining 48 outer positions map to dummy slot T_pb
        np.testing.assert_array_equal(idx[80:], np.full(48, T_pb, dtype=np.int32))

    def test_padding_slot_maps_to_dummy(self):
        backend = LinearAttentionBackend()
        # extend=[30, 0], T_outer=64 (outer bucket pads to 64)
        batch = _make_batch(ForwardMode.EXTEND, [30, 0], [30, 0], T_outer=64)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        T_pb = backend.T_packed_bucket  # 64 (only one real chunk of 64)

        # First 30 tokens map to 0..29
        np.testing.assert_array_equal(idx[:30], np.arange(30, dtype=np.int32))
        # Remaining 34 outer positions map to dummy slot
        np.testing.assert_array_equal(idx[30:], np.full(34, T_pb, dtype=np.int32))


# ---------------------------------------------------------------------------
# Decode no-op test
# ---------------------------------------------------------------------------


class TestDecodeNoOp:
    def test_decode_does_not_crash(self):
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.DECODE, None, [10, 20], T_outer=2)
        # Should return immediately without raising
        metadata = backend.get_forward_metadata(batch)
        # State unchanged from init
        assert backend.T_packed_bucket == 0
        # Decode returns an empty LinearAttentionMetadata with None fields.
        assert isinstance(metadata, LinearAttentionMetadata)
        assert metadata.cu_seqlens_dev is None
        assert metadata.scatter_idx is None


# ---------------------------------------------------------------------------
# scatter_to_packed / gather_from_packed tests
# ---------------------------------------------------------------------------


class TestScatterGather:
    def _make_backend_and_batch(self, extend_seq_lens, T_outer=None):
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.EXTEND, extend_seq_lens, extend_seq_lens, T_outer=T_outer)
        metadata = backend.get_forward_metadata(batch)
        return backend, metadata

    def test_scatter_output_shape(self):
        backend, metadata = self._make_backend_and_batch([30, 50], T_outer=128)
        H, K = 4, 8
        x = jnp.ones((128, H, K), dtype=jnp.float32)
        scatter_idx = metadata.scatter_idx
        T_pb = backend.T_packed_bucket
        out = scatter_to_packed(x, scatter_idx, T_pb)
        assert out.shape == (1, T_pb, H, K)

    def test_gather_roundtrip_aligned(self):
        """When T_outer == T_tight and no padding, scatter then gather recovers original."""
        backend, metadata = self._make_backend_and_batch([64], T_outer=64)
        H, K = 2, 4
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((64, H, K)).astype(np.float32)
        x = jnp.array(x_np)
        scatter_idx = metadata.scatter_idx
        T_pb = backend.T_packed_bucket  # 64

        packed = scatter_to_packed(x, scatter_idx, T_pb)
        recovered = gather_from_packed(packed, scatter_idx)

        np.testing.assert_allclose(np.asarray(recovered[:64]), x_np, rtol=1e-6, atol=1e-6)

    def test_gather_roundtrip_multi_request(self):
        """Real tokens (first T_tight) roundtrip exactly; outer padding gathers zeros."""
        extend_seq_lens = [30, 50]
        T_tight = 80
        T_outer = 128
        backend, metadata = self._make_backend_and_batch(extend_seq_lens, T_outer=T_outer)
        H, K = 3, 6
        rng = np.random.default_rng(1)
        # Only set real token values; outer padding is zeros
        x_np = np.zeros((T_outer, H, K), dtype=np.float32)
        x_np[:T_tight] = rng.standard_normal((T_tight, H, K)).astype(np.float32)
        x = jnp.array(x_np)

        scatter_idx = metadata.scatter_idx
        T_pb = backend.T_packed_bucket

        packed = scatter_to_packed(x, scatter_idx, T_pb)
        recovered = gather_from_packed(packed, scatter_idx)

        # Real tokens roundtrip exactly
        np.testing.assert_allclose(
            np.asarray(recovered[:T_tight]), x_np[:T_tight], rtol=1e-6, atol=1e-6
        )
        # Outer padding positions gather zeros (dummy slot)
        np.testing.assert_allclose(
            np.asarray(recovered[T_tight:]), np.zeros((T_outer - T_tight, H, K)), atol=1e-7
        )

    def test_gather_roundtrip_outer_padding(self):
        """T_outer > sum(extend_seq_lens): tail padding tokens map to dummy slot and gather zeros."""
        extend_seq_lens = [50]
        T_tight = 50
        T_outer = 128  # scheduler bucket pads beyond real tokens
        backend, metadata = self._make_backend_and_batch(extend_seq_lens, T_outer=T_outer)
        H, K = 2, 4
        rng = np.random.default_rng(2)
        x_np = np.zeros((T_outer, H, K), dtype=np.float32)
        x_np[:T_tight] = rng.standard_normal((T_tight, H, K)).astype(np.float32)
        x = jnp.array(x_np)

        scatter_idx = metadata.scatter_idx
        T_pb = backend.T_packed_bucket  # ceil(50/64)*64 = 64

        # Verify tail positions in scatter_idx map to dummy slot
        idx_np = np.asarray(scatter_idx)
        np.testing.assert_array_equal(
            idx_np[T_tight:], np.full(T_outer - T_tight, T_pb, dtype=np.int32)
        )

        packed = scatter_to_packed(x, scatter_idx, T_pb)
        recovered = gather_from_packed(packed, scatter_idx)

        # Real tokens roundtrip exactly
        np.testing.assert_allclose(
            np.asarray(recovered[:T_tight]), x_np[:T_tight], rtol=1e-6, atol=1e-6
        )
        # Tail padding positions gather zeros from dummy slot
        np.testing.assert_allclose(
            np.asarray(recovered[T_tight:]), np.zeros((T_outer - T_tight, H, K)), atol=1e-7
        )


# ---------------------------------------------------------------------------
# JIT safety tests — verify LinearAttentionMetadata survives jax.jit
# ---------------------------------------------------------------------------


class TestJitSafety:
    """Verify that LinearAttentionMetadata flows correctly through jax.jit.

    These tests catch the original bug where nnx.data fields accessed via
    `.value` would crash inside JIT with 'DynamicJaxprTracer has no attribute
    value'. The pytree-based metadata must survive flatten/unflatten and be
    accessible as normal traced arrays inside JIT.
    """

    def test_metadata_pytree_survives_jit(self):
        """LinearAttentionMetadata arrays are accessible inside jax.jit."""
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.EXTEND, [30, 50], [30, 50], T_outer=128)
        metadata = backend.get_forward_metadata(batch)

        @jax.jit
        def use_metadata(meta):
            # Access fields — this would crash with the old .value interface
            return meta.cu_seqlens_dev[0] + meta.scatter_idx[0]

        result = use_metadata(metadata)
        expected = np.asarray(metadata.cu_seqlens_dev)[0] + np.asarray(metadata.scatter_idx)[0]
        assert int(result) == int(expected)

    def test_metadata_in_container_survives_jit(self):
        """Metadata nested in a container (simulating ForwardBatch) works in JIT."""
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.EXTEND, [64], [64], T_outer=64)
        metadata = backend.get_forward_metadata(batch)

        # Simulate ForwardBatch by nesting metadata in a tuple (pytree container)
        @jax.jit
        def scatter_inside_jit(x, meta):
            packed = scatter_to_packed(x, meta.scatter_idx, backend.T_packed_bucket)
            recovered = gather_from_packed(packed, meta.scatter_idx)
            return recovered

        H, K = 2, 4
        x = jax.random.normal(jax.random.PRNGKey(0), (64, H, K))
        recovered = scatter_inside_jit(x, metadata)

        np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), rtol=1e-6, atol=1e-6)

    def test_decode_metadata_none_fields_in_jit(self):
        """DECODE metadata (None fields) can be passed through JIT without crash."""
        backend = LinearAttentionBackend()
        batch = _make_batch(ForwardMode.DECODE, None, [10, 20], T_outer=2)
        metadata = backend.get_forward_metadata(batch)

        @jax.jit
        def check_decode_meta(meta):
            # In decode, both fields are None — verify JIT doesn't crash
            # Return a constant to prove the function executed
            return jnp.array(42)

        result = check_decode_meta(metadata)
        assert int(result) == 42
