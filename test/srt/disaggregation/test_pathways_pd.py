"""Pathways single-controller cross-slice PD unit tests.

Run: XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
     pytest test/srt/disaggregation/test_pathways_pd.py -v
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.pathways_pd import (
    PathwaysPDKVTransfer,
    group_by_slice,
    make_slice_meshes,
    slots_to_ordered_pages,
)

NUM_PAGES = 16
PAGE_SIZE = 4
HEAD_DIM = 8
N_LAYERS = 3


@pytest.fixture(scope="module")
def meshes():
    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip(f"need >=8 devices, got {len(devs)}")
    p_meshes, d_mesh = make_slice_meshes(dp_size=1, tp_per_side=len(devs) // 2)
    assert isinstance(p_meshes, list) and len(p_meshes) == 1
    return p_meshes[0], d_mesh


def _make_pool(mesh, fill: str):
    sh = NamedSharding(mesh, P("data", None, None, None))
    bufs = []
    for layer in range(N_LAYERS):
        if fill == "seq":
            host = (
                np.arange(NUM_PAGES * PAGE_SIZE * HEAD_DIM, dtype=np.float32).reshape(
                    NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM
                )
                + layer * 1000
            ).astype(jnp.bfloat16)
        else:
            host = np.zeros((NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM), jnp.bfloat16)
        bufs.append(jax.device_put(host, sh))
    [b.block_until_ready() for b in bufs]
    return SimpleNamespace(
        kv_buffer=bufs, layer_num=N_LAYERS, kv_sharding=sh, _donate_lock=threading.Lock()
    )


@pytest.mark.unit
def test_group_by_slice_fallback():
    devs = jax.devices()
    groups = group_by_slice(devs)
    assert len(groups) == 2
    assert sum(len(g) for g in groups.values()) == len(devs)
    all_ids = {d.id for g in groups.values() for d in g}
    assert len(all_ids) == len(devs)


@pytest.mark.unit
def test_make_slice_meshes(meshes):
    p_mesh, d_mesh = meshes
    assert p_mesh.devices.size == d_mesh.devices.size
    assert p_mesh.axis_names == ("data", "tensor")
    p_ids = {d.id for d in p_mesh.devices.flatten()}
    d_ids = {d.id for d in d_mesh.devices.flatten()}
    assert p_ids.isdisjoint(d_ids)


@pytest.mark.unit
def test_transfer_byte_equal(meshes):
    p_mesh, d_mesh = meshes
    p_pool = _make_pool(p_mesh, fill="seq")
    d_pool = _make_pool(d_mesh, fill="zero")
    xfer = PathwaysPDKVTransfer(p_mesh, d_mesh, p_pool, d_pool)

    p_pages = np.array([1, 3, 5, 7], np.int32)
    d_pages = np.array([0, 2, 4, 6], np.int32)
    xfer.transfer(p_pages, d_pages)

    for layer in range(N_LAYERS):
        d_host = np.asarray(d_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])
        untouched = np.array([1, 3, 5, 7, 8, 9], np.int32)
        np.testing.assert_array_equal(d_host[untouched], np.zeros_like(d_host[untouched]))


@pytest.mark.unit
def test_transfer_bucket_padding(meshes):
    p_mesh, d_mesh = meshes
    p_pool = _make_pool(p_mesh, fill="seq")
    d_pool = _make_pool(d_mesh, fill="zero")
    xfer = PathwaysPDKVTransfer(p_mesh, d_mesh, p_pool, d_pool)

    p_pages = np.array([2, 4, 6], np.int32)
    d_pages = np.array([1, 3, 5], np.int32)
    xfer.transfer(p_pages, d_pages)

    for layer in range(N_LAYERS):
        d_host = np.asarray(d_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])


@pytest.mark.unit
def test_transfer_on_d_mesh(meshes):
    p_mesh, d_mesh = meshes
    p_pool = _make_pool(p_mesh, fill="seq")
    d_pool = _make_pool(d_mesh, fill="zero")
    xfer = PathwaysPDKVTransfer(p_mesh, d_mesh, p_pool, d_pool)
    xfer.transfer(np.array([0], np.int32), np.array([0], np.int32))

    d_devs = set(d_mesh.devices.flatten().tolist())
    for buf in d_pool.kv_buffer:
        assert set(buf.sharding.device_set) == d_devs


@pytest.mark.unit
def test_slots_to_ordered_pages():
    slots = np.array([8, 9, 10, 11, 4, 5, 6, 7, 20, 21], np.int32)
    np.testing.assert_array_equal(slots_to_ordered_pages(slots, 4), [2, 1, 5])
    # mid-page resume (chunked req): [6,7] then [8,9,10] -> pages [1,2]
    np.testing.assert_array_equal(
        slots_to_ordered_pages(np.array([6, 7, 8, 9, 10], np.int32), 4), [1, 2]
    )


def _make_swa_pool(mesh, fill: str, n_swa_layers: int = 2):
    full = _make_pool(mesh, fill)
    swa = SimpleNamespace(
        kv_buffer=[_make_pool(mesh, fill).kv_buffer[i] for i in range(n_swa_layers)],
        layer_num=n_swa_layers,
        kv_sharding=full.kv_sharding,
    )
    return SimpleNamespace(full_kv_pool=full, swa_kv_pool=swa, _donate_lock=threading.Lock())


def _identity_swa_alloc():
    # full slot i <-> swa slot i (page-aligned 1:1, the PD invariant)
    return SimpleNamespace(
        full_to_swa_index_mapping=np.arange(NUM_PAGES * PAGE_SIZE, dtype=np.int64)
    )


@pytest.mark.unit
def test_swa_dual_pool_transfer(meshes):
    """SWAKVPool path: both full and swa sub-pools are gathered/scattered,
    and swa pages are derived via full_to_swa_index_mapping on each side."""
    p_mesh, d_mesh = meshes
    p_pool = _make_swa_pool(p_mesh, "seq")
    d_pool = _make_swa_pool(d_mesh, "zero")
    xfer = PathwaysPDKVTransfer(
        p_mesh,
        d_mesh,
        p_pool,
        d_pool,
        page_size=PAGE_SIZE,
        p_alloc=_identity_swa_alloc(),
        d_alloc=_identity_swa_alloc(),
    )
    assert xfer.is_swa and len(xfer._jits) == 2

    p_pages = np.array([1, 3, 5], np.int32)
    d_pages = np.array([0, 2, 4], np.int32)
    xfer.transfer(p_pages, d_pages)

    for layer in range(N_LAYERS):
        d_host = np.asarray(d_pool.full_kv_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.full_kv_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])
    for layer in range(2):
        d_host = np.asarray(d_pool.swa_kv_pool.kv_buffer[layer])
        p_host = np.asarray(p_pool.swa_kv_pool.kv_buffer[layer])
        np.testing.assert_array_equal(d_host[d_pages], p_host[p_pages])
        # swa pages outside d_pages stay zero (no over-write from bucket pad)
        np.testing.assert_array_equal(d_host[[1, 3, 6, 7]], 0)


@pytest.mark.unit
def test_swa_page_mapping(meshes):
    """_swa_pages must remap full-space pages through each side's mapping."""
    p_mesh, d_mesh = meshes
    p_pool = _make_swa_pool(p_mesh, "seq")
    d_pool = _make_swa_pool(d_mesh, "zero")
    # P side: full page k -> swa page (k+2) mod NUM_PAGES (page-aligned shift)
    p_map = np.arange(NUM_PAGES * PAGE_SIZE, dtype=np.int64)
    p_map = ((p_map // PAGE_SIZE + 2) % NUM_PAGES) * PAGE_SIZE + p_map % PAGE_SIZE
    p_alloc = SimpleNamespace(full_to_swa_index_mapping=p_map)
    xfer = PathwaysPDKVTransfer(
        p_mesh,
        d_mesh,
        p_pool,
        d_pool,
        page_size=PAGE_SIZE,
        p_alloc=p_alloc,
        d_alloc=_identity_swa_alloc(),
    )
    full_pages = np.array([0, 1, 5], np.int32)
    np.testing.assert_array_equal(xfer._swa_pages(full_pages, xfer.p_mapping), [2, 3, 7])
    np.testing.assert_array_equal(xfer._swa_pages(full_pages, xfer.d_mapping), full_pages)
