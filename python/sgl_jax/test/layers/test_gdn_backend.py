"""Unit tests for :class:`GDNAttnBackend`.

The kernels themselves are covered in ``test_gated_delta.py`` and
``test_ragged_gated_delta_rule_ref.py``; this file exercises the backend
glue: ``__init__`` parameter ownership, decode/extend dispatch, and the
``shard_map``-wrapped conv + recurrence pipeline.

The backend inherits ``LinearRecurrentAttnBackend`` and reads
``cu_q_lens`` / ``recurrent_indices`` / ``has_initial_state`` from
``self.forward_metadata`` (normally populated by
``get_forward_metadata(batch)`` before the forward; we set it directly
here for unit testing). State is fetched from a
``recurrent_state_pool``-shaped object via ``get_layer_cache``.

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest test/srt/test_gdn_backend.py -v
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mesh():
    """Single-device 1×1 mesh with both 'data' and 'tensor' axes."""
    devices = mesh_utils.create_device_mesh((8,))[:1].reshape((1, 1))
    return Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_backend(mesh, n_kq=1, n_v=2, d_k=4, d_v=8, K=3):
    backend = GDNAttnBackend(
        num_k_heads=n_kq,
        num_v_heads=n_v,
        head_k_dim=d_k,
        head_v_dim=d_v,
        conv_kernel_size=K,
        mesh=mesh,
        dtype=jnp.bfloat16,
    )
    rng = jax.random.split(jax.random.key(0), 3)
    backend.conv1d_weight = nnx.Param(
        jax.random.normal(rng[0], (backend.conv_dim, K), dtype=jnp.bfloat16) * 0.1
    )
    backend.A_log = nnx.Param(jax.random.normal(rng[1], (n_v,)) * 0.3)
    backend.dt_bias = nnx.Param(jax.random.normal(rng[2], (n_v,)) * 0.3)
    return backend


class _FakeForwardMode:
    def __init__(self, decode: bool):
        self._decode = decode

    def is_decode(self):
        return self._decode


class _FakeForwardBatch:
    """Minimal forward batch — the new backend only reads forward_mode here.

    All ragged-batch info (cu_seqlens, recurrent_indices, has_initial_state)
    is read from ``backend.forward_metadata`` instead, which the tests set
    directly via ``_set_metadata``.
    """

    def __init__(self, is_decode: bool):
        self.forward_mode = _FakeForwardMode(is_decode)


class _FakePool:
    """Stand-in for :class:`RecurrentStatePool` that exposes the single
    method ``get_linear_recurrent_layer_cache`` the backend uses.

    Holds one ``recurrent_buffer`` and a one-element conv-buffer list per
    layer (GDN has a single fused conv per layer; KDA would have three).
    """

    def __init__(self, recurrent_buffer, conv_buffer):
        self._rec = recurrent_buffer
        self._conv_list = [conv_buffer]

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        return self._rec, self._conv_list


def _set_metadata(backend, cu_q_lens=None, recurrent_indices=None, has_initial_state=None):
    backend.forward_metadata = LinearRecurrentAttnBackendMetadata(
        cu_q_lens=cu_q_lens,
        recurrent_indices=recurrent_indices,
        has_initial_state=has_initial_state,
    )


def _sharded_state(mesh, shape, spec, dtype, rng=None):
    """Allocate an array with explicit sharding (matches what
    ``RecurrentStatePool`` produces in production)."""
    out_sharding = NamedSharding(mesh, spec)
    if rng is None:
        return jnp.zeros(shape, dtype=dtype, out_sharding=out_sharding)
    return jax.random.normal(rng, shape, dtype=dtype, out_sharding=out_sharding)


def _sharded_proj(mesh, shape, spec, rng, scale=0.3):
    """Mimic a `LinearBase` output: sharded on the last (head/channel) axis."""
    return (
        jax.random.normal(
            rng,
            shape,
            dtype=jnp.bfloat16,
            out_sharding=NamedSharding(mesh, spec),
        )
        * scale
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class GDNAttnBackendInitTest(unittest.TestCase):
    def test_param_shapes_match_config(self):
        """``conv_dim`` is derived from head counts/dims; param shapes track it."""
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            n_kq, n_v, d_k, d_v, K = 2, 4, 8, 16, 4
            backend = GDNAttnBackend(
                num_k_heads=n_kq,
                num_v_heads=n_v,
                head_k_dim=d_k,
                head_v_dim=d_v,
                conv_kernel_size=K,
                mesh=mesh,
                dtype=jnp.bfloat16,
            )
            self.assertEqual(backend.key_dim, n_kq * d_k)
            self.assertEqual(backend.value_dim, n_v * d_v)
            self.assertEqual(backend.conv_dim, 2 * backend.key_dim + backend.value_dim)
            self.assertEqual(backend.conv1d_weight.value.shape, (backend.conv_dim, K))
            self.assertEqual(backend.A_log.value.shape, (n_v,))
            self.assertEqual(backend.dt_bias.value.shape, (n_v,))


class GDNAttnBackendDispatchTest(unittest.TestCase):
    """``__call__`` routes to forward_decode/forward_extend on forward_mode."""

    def _make_inputs(self, mesh, T, backend, rng_root):
        rng = jax.random.split(rng_root, 3)
        mixed_qkv = _sharded_proj(mesh, (T, backend.conv_dim), P(None, "tensor"), rng[0])
        b = _sharded_proj(mesh, (T, backend.num_v_heads), P(None, "tensor"), rng[1])
        a = _sharded_proj(mesh, (T, backend.num_v_heads), P(None, "tensor"), rng[2])
        return mixed_qkv, b, a

    def _make_pool(self, mesh, backend, B):
        cs = _sharded_state(
            mesh,
            (B + 1, backend.conv_dim, backend.conv_kernel_size - 1),
            P(None, "tensor", None),
            jnp.bfloat16,
        )
        rs = _sharded_state(
            mesh,
            (B + 1, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim),
            P(None, "tensor", None, None),
            jnp.float32,
        )
        return _FakePool(recurrent_buffer=rs, conv_buffer=cs)

    def test_decode_dispatch(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            backend = _make_backend(mesh)
            B = 2
            mixed_qkv, b, a = self._make_inputs(mesh, B, backend, jax.random.key(1))
            pool = self._make_pool(mesh, backend, B)
            _set_metadata(
                backend,
                recurrent_indices=jnp.array([1, 2], dtype=jnp.int32),
            )
            fb = _FakeForwardBatch(is_decode=True)
            out, new_conv, new_rec = backend(fb, mixed_qkv, b, a, pool, layer_id=0)
            self.assertEqual(out.shape, (B, backend.num_v_heads, backend.head_v_dim))
            # `new_conv` / `new_rec` are the full pool tables (kernel scatters
            # per-request slots in place). Pool was sized `B + 1`.
            self.assertEqual(
                new_conv.shape, (B + 1, backend.conv_dim, backend.conv_kernel_size - 1)
            )
            self.assertEqual(
                new_rec.shape, (B + 1, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim)
            )

    def test_extend_dispatch(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            backend = _make_backend(mesh)
            T = 5  # 2 reqs of lengths [3, 2]
            B = 2
            mixed_qkv, b, a = self._make_inputs(mesh, T, backend, jax.random.key(2))
            pool = self._make_pool(mesh, backend, B)
            _set_metadata(
                backend,
                cu_q_lens=jnp.array([0, 3, 5], dtype=jnp.int32),
                recurrent_indices=jnp.array([1, 2], dtype=jnp.int32),
                has_initial_state=jnp.array([False, False], dtype=jnp.bool_),
            )
            fb = _FakeForwardBatch(is_decode=False)
            out, new_conv, new_rec = backend(fb, mixed_qkv, b, a, pool, layer_id=0)
            self.assertEqual(out.shape, (T, backend.num_v_heads, backend.head_v_dim))
            self.assertEqual(
                new_conv.shape, (B + 1, backend.conv_dim, backend.conv_kernel_size - 1)
            )
            self.assertEqual(
                new_rec.shape, (B + 1, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim)
            )


class GDNAttnBackendExtendStateTest(unittest.TestCase):
    """forward_extend should return per-request new states with finite values."""

    def test_extend_returns_per_request_state_shape(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            backend = _make_backend(mesh)
            lens = [4, 2, 1]
            T = sum(lens)
            B = len(lens)
            rng = jax.random.split(jax.random.key(50), 3)
            mixed_qkv = _sharded_proj(mesh, (T, backend.conv_dim), P(None, "tensor"), rng[0])
            b = _sharded_proj(mesh, (T, backend.num_v_heads), P(None, "tensor"), rng[1])
            a = _sharded_proj(mesh, (T, backend.num_v_heads), P(None, "tensor"), rng[2])
            cs = _sharded_state(
                mesh,
                (B + 1, backend.conv_dim, backend.conv_kernel_size - 1),
                P(None, "tensor", None),
                jnp.bfloat16,
                rng=jax.random.key(60),
            )
            rs = _sharded_state(
                mesh,
                (B + 1, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim),
                P(None, "tensor", None, None),
                jnp.float32,
                rng=jax.random.key(61),
            )
            pool = _FakePool(recurrent_buffer=rs, conv_buffer=cs)
            _set_metadata(
                backend,
                cu_q_lens=jnp.array([0, 4, 6, 7], dtype=jnp.int32),
                recurrent_indices=jnp.array([1, 2, 3], dtype=jnp.int32),
                has_initial_state=jnp.array([False, False, False], dtype=jnp.bool_),
            )
            fb = _FakeForwardBatch(is_decode=False)
            out, new_conv, new_rec = backend(fb, mixed_qkv, b, a, pool, layer_id=0)
            self.assertEqual(out.shape, (T, backend.num_v_heads, backend.head_v_dim))
            self.assertEqual(
                new_conv.shape, (B + 1, backend.conv_dim, backend.conv_kernel_size - 1)
            )
            self.assertEqual(
                new_rec.shape, (B + 1, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim)
            )
            self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
            self.assertTrue(bool(jnp.all(jnp.isfinite(new_rec))))


if __name__ == "__main__":
    unittest.main()
