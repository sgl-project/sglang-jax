"""Tests for per-layer KV scatter (decode-side write-back OOM fix).

Symmetric to ``test_kv_gather_per_layer.py``. Validates that the per-layer jit
scatter writes the same values as direct numpy assignment, across all page
bucket sizes, reuses compiled kernels across layers, and donates its input
buffer (the property that lets XLA update in place and avoid the ~884 MB
double-allocation that OOM'd the decode event loop).
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.disaggregation.decode import _jit_scatter_one_layer
from sgl_jax.srt.disaggregation.prefill import _KV_GATHER_PAGE_BUCKETS

# Realistic model params (Qwen3-8B style, scaled down for CPU test)
_NUM_LAYERS = 4
_NUM_PAGES_POOL = 32
_PAGE_SIZE = 8
_HEAD_NUM_KV = 4  # num_kv_heads
_PACKING = 2
_HEAD_DIM = 16

_LAYER_SHAPE = (
    _NUM_PAGES_POOL,
    _PAGE_SIZE,
    _HEAD_NUM_KV * 2 // _PACKING,
    _PACKING,
    _HEAD_DIM,
)


def _make_mesh():
    """Single-device mesh for CPU testing (1x1 for data x tensor)."""
    devices = jax.devices("cpu")[:1]
    return Mesh(
        np.array(devices).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _pool_sharding(mesh):
    return NamedSharding(mesh, P("data", None, "tensor", None, None))


def _out_sharding(mesh, pool_sharding):
    """Page axis unsharded, mirroring decode._write_kv_to_pool."""
    return NamedSharding(mesh, P(None, *pool_sharding.spec[1:]))


def _make_layer_buffer(mesh, pool_sharding, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    data = rng.standard_normal(_LAYER_SHAPE).astype(np.float32)
    return jax.device_put(jnp.array(data), pool_sharding), data


def _make_values(mesh, out_sharding, num_pages, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    shape = (num_pages,) + _LAYER_SHAPE[1:]
    data = rng.standard_normal(shape).astype(np.float32)
    return jax.device_put(jnp.array(data), out_sharding), data


class TestPerLayerScatter:
    """Verify per-layer scatter correctness + donation."""

    @pytest.fixture
    def env(self):
        mesh = _make_mesh()
        pool_sharding = _pool_sharding(mesh)
        out_sharding = _out_sharding(mesh, pool_sharding)
        idx_sharding = NamedSharding(mesh, P(None))
        return mesh, pool_sharding, out_sharding, idx_sharding

    @pytest.mark.parametrize("num_pages", _KV_GATHER_PAGE_BUCKETS)
    def test_scatter_matches_naive(self, env, num_pages):
        """Per-layer scatter output matches numpy fancy-index assignment."""
        mesh, pool_sharding, out_sharding, idx_sharding = env
        # page ids wrap into the pool; distinct so order is unambiguous.
        page_ids_np = (np.arange(num_pages, dtype=np.int32)) % _NUM_PAGES_POOL
        buf, buf_np = _make_layer_buffer(mesh, pool_sharding)
        vals, vals_np = _make_values(mesh, out_sharding, num_pages)
        page_ids = jax.device_put(jnp.asarray(page_ids_np), idx_sharding)

        result = _jit_scatter_one_layer(buf, page_ids, vals, out_sharding)

        expected = buf_np.copy()
        expected[page_ids_np] = vals_np  # last-write-wins, same as JAX .set
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_scatter_output_shape(self, env):
        """Scatter returns the full per-layer buffer, shape unchanged."""
        mesh, pool_sharding, out_sharding, idx_sharding = env
        buf, _ = _make_layer_buffer(mesh, pool_sharding)
        vals, _ = _make_values(mesh, out_sharding, 4)
        page_ids = jax.device_put(jnp.array([0, 1, 2, 3], dtype=jnp.int32), idx_sharding)
        result = _jit_scatter_one_layer(buf, page_ids, vals, out_sharding)
        assert result.shape == _LAYER_SHAPE

    def test_scatter_preserves_untouched_pages(self, env):
        """Pages not in page_indices keep their original contents."""
        mesh, pool_sharding, out_sharding, idx_sharding = env
        buf, buf_np = _make_layer_buffer(mesh, pool_sharding)
        page_ids_np = np.array([2, 5, 9], dtype=np.int32)
        vals, vals_np = _make_values(mesh, out_sharding, len(page_ids_np))
        page_ids = jax.device_put(jnp.asarray(page_ids_np), idx_sharding)

        result = np.asarray(_jit_scatter_one_layer(buf, page_ids, vals, out_sharding))

        expected = buf_np.copy()
        expected[page_ids_np] = vals_np
        np.testing.assert_array_equal(result, expected)
        # spot-check an untouched page is byte-for-byte original
        np.testing.assert_array_equal(result[0], buf_np[0])

    def test_scatter_duplicate_tail_indices(self, env):
        """Repeated trailing page ids (the bucket-pad pattern) take the last
        written value, matching _write_kv_to_pool's tail-repeat padding."""
        mesh, pool_sharding, out_sharding, idx_sharding = env
        buf, buf_np = _make_layer_buffer(mesh, pool_sharding)
        # 2 real pages (0, 3) then padded tail repeats id 3.
        page_ids_np = np.array([0, 3, 3, 3], dtype=np.int32)
        vals, vals_np = _make_values(mesh, out_sharding, 4)
        page_ids = jax.device_put(jnp.asarray(page_ids_np), idx_sharding)

        result = np.asarray(_jit_scatter_one_layer(buf, page_ids, vals, out_sharding))

        expected = buf_np.copy()
        expected[page_ids_np] = vals_np  # last write to page 3 wins
        np.testing.assert_array_equal(result, expected)
        # page 3 ends up with the LAST value among the duplicates.
        np.testing.assert_array_equal(result[3], vals_np[3])

    def test_scatter_donates_input(self, env):
        """donate_argnums=(0,) consumes the input buffer (enables in-place)."""
        mesh, pool_sharding, out_sharding, idx_sharding = env
        buf, _ = _make_layer_buffer(mesh, pool_sharding)
        vals, _ = _make_values(mesh, out_sharding, 2)
        page_ids = jax.device_put(jnp.array([0, 1], dtype=jnp.int32), idx_sharding)

        _ = _jit_scatter_one_layer(buf, page_ids, vals, out_sharding)

        assert buf.is_deleted(), "input buffer should be donated/deleted"

    def test_multi_layer_independent(self, env):
        """Scattering each layer independently writes correct per-layer values."""
        mesh, pool_sharding, out_sharding, idx_sharding = env
        page_ids_np = np.array([1, 4, 7], dtype=np.int32)
        page_ids = jax.device_put(jnp.asarray(page_ids_np), idx_sharding)
        for layer in range(_NUM_LAYERS):
            buf, buf_np = _make_layer_buffer(mesh, pool_sharding, rng_seed=100 + layer)
            vals, vals_np = _make_values(mesh, out_sharding, 3, rng_seed=200 + layer)
            result = np.asarray(
                _jit_scatter_one_layer(buf, page_ids, vals, out_sharding)
            )
            expected = buf_np.copy()
            expected[page_ids_np] = vals_np
            np.testing.assert_array_equal(
                result, expected, err_msg=f"Layer {layer}"
            )


class TestScatterCompileCaching:
    """Verify per-layer jit reuses compiled kernels across same-shape layers."""

    def test_same_shape_reuses_cache(self):
        mesh = _make_mesh()
        pool_sharding = _pool_sharding(mesh)
        out_sharding = _out_sharding(mesh, pool_sharding)
        idx_sharding = NamedSharding(mesh, P(None))
        page_ids = jax.device_put(jnp.array([0, 1], dtype=jnp.int32), idx_sharding)

        buf0, _ = _make_layer_buffer(mesh, pool_sharding, rng_seed=1)
        buf1, _ = _make_layer_buffer(mesh, pool_sharding, rng_seed=2)
        vals, _ = _make_values(mesh, out_sharding, 2)

        # lower() does not execute, so it does not consume the donated buffer.
        lowered_0 = _jit_scatter_one_layer.lower(buf0, page_ids, vals, out_sharding)
        lowered_1 = _jit_scatter_one_layer.lower(buf1, page_ids, vals, out_sharding)
        # Same HLO text => same compiled kernel reused across layers.
        assert lowered_0.as_text() == lowered_1.as_text()
