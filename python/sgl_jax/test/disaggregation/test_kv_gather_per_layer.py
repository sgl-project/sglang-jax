"""Tests for per-layer KV gather (OOM fix).

Validates that the per-layer jit gather produces identical results to the
original all-layers-in-one-jit approach, across all page bucket sizes.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.disaggregation.prefill import (
    _KV_GATHER_PAGE_BUCKETS,
    _jit_gather_all_layers,
    _jit_gather_one_layer,
    _pad_to_page_bucket,
)

# Realistic model params (Qwen3-8B style, scaled down for CPU test)
_NUM_LAYERS = 4
_NUM_PAGES_POOL = 32
_PAGE_SIZE = 8
_HEAD_NUM_KV = 4  # num_kv_heads
_PACKING = 2
_HEAD_DIM = 16


def _make_mesh():
    """Single-device mesh for CPU testing (1x1 for data x tensor)."""
    devices = jax.devices("cpu")[:1]
    return Mesh(
        np.array(devices).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_kv_buffers(mesh, num_layers=_NUM_LAYERS, rng_seed=42):
    """Create per-layer KV buffers mimicking memory_pool.py layout."""
    rng = np.random.default_rng(rng_seed)
    shape = (_NUM_PAGES_POOL, _PAGE_SIZE, _HEAD_NUM_KV * 2 // _PACKING, _PACKING, _HEAD_DIM)
    sharding = NamedSharding(mesh, P("data", None, "tensor", None, None))
    buffers = []
    for _ in range(num_layers):
        data = rng.standard_normal(shape).astype(np.float32)
        buf = jax.device_put(jnp.array(data), sharding)
        buffers.append(buf)
    return buffers, sharding


class TestPerLayerGather:
    """Verify per-layer gather correctness."""

    @pytest.fixture
    def setup(self):
        mesh = _make_mesh()
        buffers, pool_sharding = _make_kv_buffers(mesh)
        gather_pspec = P(None, *pool_sharding.spec[1:])
        gather_sharding = NamedSharding(mesh, gather_pspec)
        idx_sharding = NamedSharding(mesh, P(None))
        return buffers, gather_sharding, idx_sharding, mesh

    @pytest.mark.parametrize("num_pages", _KV_GATHER_PAGE_BUCKETS)
    def test_gather_matches_naive(self, setup, num_pages):
        """Per-layer gather output matches direct numpy indexing."""
        buffers, gather_sharding, idx_sharding, mesh = setup
        page_indices = jax.device_put(
            jnp.arange(num_pages, dtype=jnp.int32) % _NUM_PAGES_POOL,
            idx_sharding,
        )
        results = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        assert len(results) == len(buffers)
        for i, (result, buf) in enumerate(zip(results, buffers)):
            expected_np = np.asarray(buf)[np.asarray(page_indices)]
            np.testing.assert_array_equal(
                np.asarray(result),
                expected_np,
                err_msg=f"Layer {i}, num_pages={num_pages}",
            )

    @pytest.mark.parametrize("num_pages", [1, 4, 16, 64])
    def test_gather_output_shape(self, setup, num_pages):
        """Output shape is (num_pages, page_size, heads, packing, head_dim)."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(
            jnp.arange(num_pages, dtype=jnp.int32) % _NUM_PAGES_POOL,
            idx_sharding,
        )
        results = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        expected_shape = (num_pages, _PAGE_SIZE, _HEAD_NUM_KV * 2 // _PACKING, _PACKING, _HEAD_DIM)
        for result in results:
            assert result.shape == expected_shape

    def test_gather_single_layer(self, setup):
        """_jit_gather_one_layer works correctly standalone."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(jnp.array([0, 3, 7], dtype=jnp.int32), idx_sharding)
        result = _jit_gather_one_layer(buffers[0], page_indices, gather_sharding)
        expected = np.asarray(buffers[0])[[0, 3, 7]]
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_gather_stack_matches_monolithic(self, setup):
        """jnp.stack(per_layer_results) matches what a monolithic gather would produce."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(
            jnp.array([1, 5, 10, 20], dtype=jnp.int32), idx_sharding
        )
        per_layer = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        stacked = jnp.stack(per_layer, axis=0)
        assert stacked.shape[0] == _NUM_LAYERS
        assert stacked.shape[1] == 4  # num_pages queried
        for i, buf in enumerate(buffers):
            expected = np.asarray(buf)[[1, 5, 10, 20]]
            np.testing.assert_array_equal(
                np.asarray(stacked[i]),
                expected,
                err_msg=f"Layer {i}",
            )

    def test_gather_duplicate_indices(self, setup):
        """Gathering same page multiple times works correctly."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(
            jnp.array([0, 0, 0, 0], dtype=jnp.int32), idx_sharding
        )
        results = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        for i, result in enumerate(results):
            page0 = np.asarray(buffers[i])[0]
            for j in range(4):
                np.testing.assert_array_equal(np.asarray(result)[j], page0)

    def test_gather_last_page(self, setup):
        """Can gather the last page in the pool without error."""
        buffers, gather_sharding, idx_sharding, _ = setup
        page_indices = jax.device_put(
            jnp.array([_NUM_PAGES_POOL - 1], dtype=jnp.int32), idx_sharding
        )
        results = _jit_gather_all_layers(buffers, page_indices, gather_sharding)
        for i, result in enumerate(results):
            expected = np.asarray(buffers[i])[_NUM_PAGES_POOL - 1: _NUM_PAGES_POOL]
            np.testing.assert_array_equal(np.asarray(result), expected)


class TestPadToPageBucket:
    """Test the page bucketing utility."""

    @pytest.mark.parametrize(
        "input_pages,expected_bucket",
        [
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 8),
            (8, 8),
            (9, 16),
            (16, 16),
            (17, 32),
            (32, 32),
            (33, 64),
            (64, 64),
            (65, 64),  # clamped to max bucket
        ],
    )
    def test_bucket_selection(self, input_pages, expected_bucket):
        assert _pad_to_page_bucket(input_pages) == expected_bucket


class TestGatherCompileCaching:
    """Verify that per-layer jit reuses compiled kernels across layers."""

    def test_same_shape_reuses_cache(self):
        """All layers with same buffer shape should hit the same compiled kernel."""
        mesh = _make_mesh()
        buffers, pool_sharding = _make_kv_buffers(mesh, num_layers=4)
        gather_sharding = NamedSharding(mesh, P(None, *pool_sharding.spec[1:]))
        idx_sharding = NamedSharding(mesh, P(None))
        page_indices = jax.device_put(jnp.array([0, 1], dtype=jnp.int32), idx_sharding)

        # Warm up: first call compiles
        _ = _jit_gather_one_layer(buffers[0], page_indices, gather_sharding)

        # Subsequent calls should use cached compilation
        lowered_0 = _jit_gather_one_layer.lower(buffers[0], page_indices, gather_sharding)
        lowered_1 = _jit_gather_one_layer.lower(buffers[1], page_indices, gather_sharding)
        # Same HLO text means same compiled kernel will be reused
        assert lowered_0.as_text() == lowered_1.as_text()
