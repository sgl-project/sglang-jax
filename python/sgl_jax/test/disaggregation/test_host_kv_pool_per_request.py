"""Per-request QueueHostKVPool layout: reserve/release, per-layer staging."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, PartitionSpec

from sgl_jax.srt.mem_cache.host_kv_pool import QueueHostKVPool, StagedData

_LAYER_NUM = 4
_MAX_PAGES = 8
_PER_LAYER_SHAPE = (2, 3, 5)  # (page_size, kv_head, head_dim)-ish tail
_DTYPE = jnp.float32


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices("cpu")[:1]).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_pool(pool_size=2):
    return QueueHostKVPool(
        pool_size=pool_size,
        max_padded_pages=_MAX_PAGES,
        layer_num=_LAYER_NUM,
        per_layer_shape=_PER_LAYER_SHAPE,
        dtype=_DTYPE,
        mesh=_mesh(),
        partition_spec=PartitionSpec(None, "tensor", None),
    )


def _layers(padded_pages, seed=0):
    rng = np.random.default_rng(seed)
    return [
        jnp.asarray(rng.standard_normal((padded_pages, *_PER_LAYER_SHAPE)).astype(np.float32))
        for _ in range(_LAYER_NUM)
    ]


def test_reserve_release_accounting():
    pool = _make_pool(pool_size=2)
    assert pool.total_size() == 2
    assert pool.available_size() == 2
    a = pool.reserve()
    b = pool.reserve()
    assert {a, b} == {0, 1}
    assert pool.available_size() == 0
    assert pool.reserve() is None  # exhausted
    pool.release(a)
    assert pool.available_size() == 1


def test_copy_from_device_values_match_numpy():
    pool = _make_pool(pool_size=2)
    padded_pages = 4
    layers = _layers(padded_pages, seed=7)
    bid = pool.reserve()
    staged = pool.copy_from_device(layers, bid)
    assert isinstance(staged, StagedData)
    assert staged.buffer_id == bid
    assert len(staged.array_pytree) == _LAYER_NUM
    for i, layer in enumerate(layers):
        got = np.asarray(jax.device_get(staged.array_pytree[i]))
        assert got.shape == (padded_pages, *_PER_LAYER_SHAPE)
        np.testing.assert_allclose(got, np.asarray(layer), rtol=0, atol=0)


def test_double_free_guard():
    pool = _make_pool(pool_size=1)
    bid = pool.reserve()
    pool.release(bid)
    with pytest.raises(RuntimeError):
        pool.release(bid)
