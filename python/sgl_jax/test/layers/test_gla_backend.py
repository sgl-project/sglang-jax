"""Tests for GLA attention backends (GLAMetadataBackend metadata + LightningAttnBackend).

Run with: pytest python/sgl_jax/test/layers/test_gla_backend.py -v
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.linear.gla_metadata import (
    GLAMetadataBackend,
    GLAMetadata,
    gather_from_packed,
    scatter_to_packed,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    HybridLinearAttnBackendMetadata,
    MockRecurrentStatePool,
)
from sgl_jax.srt.layers.attention.linear.lightning_backend import (
    LightningAttnBackend,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import (
        fused_recurrent_simple_gla,
        simple_gla_fwd,
    )

    HAS_SIMPLE_GLA = True
except ImportError:
    HAS_SIMPLE_GLA = False

requires_simple_gla = pytest.mark.skipif(
    not HAS_SIMPLE_GLA, reason="simple_gla kernel not available"
)

_HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="chunk kernel requires TPU")

_H = 4
_K = 128

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata_batch(forward_mode, extend_seq_lens, seq_lens, T_outer=None):
    esl = np.array(extend_seq_lens, dtype=np.int32) if extend_seq_lens is not None else None
    if T_outer is None and esl is not None:
        T_outer = int(np.sum(esl))
    return SimpleNamespace(
        forward_mode=forward_mode,
        extend_seq_lens=esl,
        seq_lens=np.array(seq_lens, dtype=np.int32),
        input_ids=np.zeros(T_outer or 0, dtype=np.int32),
    )


def _make_batch(forward_mode, extend_seq_lens=None, input_ids=None, recurrent_indices=None):
    batch = SimpleNamespace(forward_mode=forward_mode)
    if extend_seq_lens is not None:
        batch.extend_seq_lens = np.asarray(extend_seq_lens, dtype=np.int32)
    if input_ids is not None:
        batch.input_ids = np.asarray(input_ids, dtype=np.int32)
    if recurrent_indices is not None:
        batch.recurrent_indices = np.asarray(recurrent_indices, dtype=np.int32)
    if forward_mode == ForwardMode.DECODE:
        n_seqs = len(recurrent_indices) if recurrent_indices is not None else 1
        batch.seq_lens = np.ones(n_seqs, dtype=np.int32)
    return batch


def _make_mock_pool(layer_id, recurrent_state, recurrent_indices=None):
    B = recurrent_state.shape[0]
    if recurrent_indices is None:
        recurrent_indices = np.arange(1, B + 1, dtype=np.int32)
    N_plus_1 = int(max(recurrent_indices)) + 1
    buf = jnp.zeros((N_plus_1,) + recurrent_state.shape[1:], dtype=recurrent_state.dtype)
    buf = buf.at[jnp.array(recurrent_indices)].set(recurrent_state)
    return MockRecurrentStatePool(layer_caches={layer_id: (buf, [])}), recurrent_indices


def _make_fake_layer(layer_id=5, slope=None):
    if slope is None:
        slope = jnp.array([-0.1, -0.2, -0.3, -0.4], dtype=jnp.float32)
    return SimpleNamespace(
        layer_id=layer_id,
        slope=slope,
        mesh=mesh,
        num_heads=_H,
        head_dim=_K,
    )


def _extract_state(pool_updates, recurrent_indices):
    new_ssm_full, conv_list = pool_updates
    assert conv_list == [] or conv_list is None
    return new_ssm_full[jnp.array(recurrent_indices)]


# ---------------------------------------------------------------------------
# scatter_idx tests
# ---------------------------------------------------------------------------


class TestScatterIdx:
    def test_two_requests_mapping(self):
        backend = GLAMetadataBackend()
        # extend=[30,50], T_outer=128
        # request 0: seq_len=30 -> chunk slot [0..63], real tokens [0..29]
        # request 1: seq_len=50 -> chunk slot [64..127], real tokens [0..49]
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 50], [30, 50], T_outer=128)
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
        backend = GLAMetadataBackend()
        # extend=[30, 0], T_outer=64 (outer bucket pads to 64)
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 0], [30, 0], T_outer=64)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        T_pb = backend.T_packed_bucket  # 64 (only one real chunk of 64)

        # First 30 tokens map to 0..29
        np.testing.assert_array_equal(idx[:30], np.arange(30, dtype=np.int32))
        # Remaining 34 outer positions map to dummy slot
        np.testing.assert_array_equal(idx[30:], np.full(34, T_pb, dtype=np.int32))


# ---------------------------------------------------------------------------
# scatter_to_packed / gather_from_packed tests
# ---------------------------------------------------------------------------


class TestScatterGather:
    def _make_backend_and_batch(self, extend_seq_lens, T_outer=None):
        backend = GLAMetadataBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, extend_seq_lens, extend_seq_lens, T_outer=T_outer)
        metadata = backend.get_forward_metadata(batch)
        return backend, metadata

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


# ---------------------------------------------------------------------------
# JIT safety tests --- verify GLAMetadata survives jax.jit
# ---------------------------------------------------------------------------


class TestJitSafety:
    """Verify that GLAMetadata flows correctly through jax.jit.

    These tests catch the original bug where nnx.data fields accessed via
    `.value` would crash inside JIT with 'DynamicJaxprTracer has no attribute
    value'. The pytree-based metadata must survive flatten/unflatten and be
    accessible as normal traced arrays inside JIT.
    """

    def test_metadata_pytree_survives_jit(self):
        """GLAMetadata arrays are accessible inside jax.jit."""
        backend = GLAMetadataBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 50], [30, 50], T_outer=128)
        metadata = backend.get_forward_metadata(batch)

        @jax.jit
        def use_metadata(meta):
            # Access fields --- this would crash with the old .value interface
            return meta.cu_seqlens_dev[0] + meta.scatter_idx[0]

        result = use_metadata(metadata)
        expected = np.asarray(metadata.cu_seqlens_dev)[0] + np.asarray(metadata.scatter_idx)[0]
        assert int(result) == int(expected)


# ---------------------------------------------------------------------------
# LightningAttnBackend tests


class TestGetForwardMetadata:
    def test_scatter_metadata_matches_old_backend(self):
        """Scatter/gather metadata should match GLAMetadataBackend output."""
        old_backend = GLAMetadataBackend(mesh=mesh)
        new_backend = LightningAttnBackend(mesh=mesh)

        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=np.array([100, 60, 128], dtype=np.int32),
            seq_lens=np.array([100, 60, 128], dtype=np.int32),
            input_ids=np.zeros(288, dtype=np.int32),
            recurrent_indices=np.array([1, 2, 3], dtype=np.int32),
        )
        old_meta = old_backend.get_forward_metadata(batch)
        new_backend.get_forward_metadata(batch)

        np.testing.assert_array_equal(
            np.array(old_meta.cu_seqlens_dev),
            np.array(new_backend.cu_seqlens_aligned),
        )
        np.testing.assert_array_equal(
            np.array(old_meta.scatter_idx),
            np.array(new_backend.scatter_idx),
        )
        assert old_backend.T_packed_bucket == new_backend.T_packed_bucket


class TestForwardDecode:
    @requires_simple_gla
    def test_decode_matches_direct_kernel(self):
        """Backend decode should match direct fused_recurrent_simple_gla call."""
        backend = LightningAttnBackend(mesh=mesh)
        layer_id = 5

        with jax.set_mesh(mesh):
            state_init = jnp.zeros((1, _H, _K, _K), dtype=jnp.float32)
            pool, rec_indices = _make_mock_pool(layer_id, state_init)

            batch = _make_batch(
                ForwardMode.DECODE,
                recurrent_indices=rec_indices,
            )
            metadata = backend.get_forward_metadata(batch)
            backend.forward_metadata = metadata

            q = jax.random.normal(jax.random.PRNGKey(0), (1, _H, _K), dtype=jnp.bfloat16)
            k = jax.random.normal(jax.random.PRNGKey(1), (1, _H, _K), dtype=jnp.bfloat16)
            v = jax.random.normal(jax.random.PRNGKey(2), (1, _H, _K), dtype=jnp.bfloat16)
            slope = jnp.array([-0.1, -0.2, -0.3, -0.4], dtype=jnp.float32)
            layer = _make_fake_layer(layer_id=layer_id, slope=slope)
            fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

            output, pool_updates = backend(q, k, v, layer=layer, forward_batch=fb, recurrent_state_pool=pool)
            backend_state = _extract_state(pool_updates, rec_indices)

            direct_out, direct_state = fused_recurrent_simple_gla(
                q[:, None, :, :],
                k[:, None, :, :],
                v[:, None, :, :],
                g_gamma=slope,
                initial_state=state_init.astype(jnp.float32),
                output_final_state=True,
                scale=None,
            )
            direct_out = direct_out[:, 0, :, :].reshape(1, -1)

        np.testing.assert_allclose(
            np.array(output), np.array(direct_out), atol=1e-5,
            err_msg="Backend decode output != direct kernel",
        )
        np.testing.assert_allclose(
            np.array(backend_state), np.array(direct_state), atol=1e-5,
            err_msg="Backend decode state != direct kernel state",
        )


class TestHybridIntegration:
    def test_dispatch_routes_to_lightning(self):
        from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

        with jax.set_mesh(mesh):
            full_backend = FlashAttention(
                num_attn_heads=_H, num_kv_heads=_H, head_dim=_K,
                page_size=1, mesh=mesh,
            )
            lightning = LightningAttnBackend(mesh=mesh)
            hybrid = HybridLinearAttnBackend(full_backend, lightning, full_attn_layers=[0, 1])

        assert 5 not in hybrid.full_attn_layers
        assert isinstance(hybrid.linear_attn_backend, LightningAttnBackend)

