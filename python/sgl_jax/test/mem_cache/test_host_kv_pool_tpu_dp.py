"""TPU round-trip tests for :class:`LRUHostKVPool` (HiCache Stage 0), tp=2/dp=2.

Companion to ``test_host_kv_pool_tpu.py`` -- see that module's docstring for why
the dp layouts live in separate files (a single process cannot switch device
mesh topology). This file exercises the dp>1 page-axis sharding: the KV buffer's
page axis is split across the data axis, so the H2D scatter must route each
GLOBAL device page to its owning shard as a shard-local offset (``flush_load``).
Registered in ``unit-test-tpu-v6e-4``; skipped off TPU / with <4 devices.
"""

from __future__ import annotations

import unittest

import numpy as np

from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.mem_cache.test_host_kv_pool_tpu import (
    _fill,
    _is_tpu_4chip,
    _make_device_pool,
    _make_pool,
    _read_page,
)


@unittest.skipUnless(_is_tpu_4chip(), "requires a TPU with >=4 chips")
class TestLRUHostKVPoolTPURoundTripDP2(unittest.TestCase):
    """Bit-exact round-trip + padding skip on the dp>1 page-axis sharding."""

    def _roundtrip(self, page_size=1):
        mesh = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        device_pool = _make_device_pool(mesh, dp_size=2, page_size=page_size)
        pool = _make_pool(device_pool, mesh, pool_size=8, page_size=page_size)

        # Distinct src/dst device pages so a no-op (forgot to write) can't pass.
        pairs = [(2, 5), (3, 6), (4, 7)]  # 3 pages -> exercises bucket padding too
        origs = {s: _fill(device_pool, s, seed=s + 1) for s, _ in pairs}
        host_pages = [int(p) for p in pool.alloc(len(pairs))]
        srcs = [s for s, _ in pairs]
        dsts = [d for _, d in pairs]

        pool.copy_into(srcs, host_pages)
        pool.copy_to_device(host_pages, dsts)

        for s, d in pairs:
            for layer in range(device_pool.layer_num):
                got = _read_page(device_pool, layer, d)
                np.testing.assert_array_equal(got, origs[s][layer])

    def test_roundtrip_tp2_dp2(self):
        self._roundtrip()

    def test_roundtrip_tp2_dp2_page_size4(self):
        self._roundtrip(page_size=4)

    def test_roundtrip_crosses_dp_shard_boundary(self):
        # Targets span both data shards (pages_per_shard=17 for size=32/page_size=1):
        # 3 in shard 0 (<17) and 2 in shard 1 (>=17). Catches the global->shard
        # routing regression directly -- a global-id scatter writes the shard-1
        # pages to the wrong place.
        mesh = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        device_pool = _make_device_pool(mesh, dp_size=2)
        pool = _make_pool(device_pool, mesh, pool_size=8)
        pairs = [(2, 5), (3, 6), (4, 7), (8, 20), (9, 25)]
        origs = {s: _fill(device_pool, s, seed=s + 1) for s, _ in pairs}
        host_pages = [int(p) for p in pool.alloc(len(pairs))]
        pool.copy_into([s for s, _ in pairs], host_pages)
        pool.copy_to_device(host_pages, [d for _, d in pairs])
        for s, d in pairs:
            for layer in range(device_pool.layer_num):
                got = _read_page(device_pool, layer, d)
                np.testing.assert_array_equal(got, origs[s][layer])

    def test_padding_does_not_clobber_other_pages(self):
        # Device page 0 is the gather/scatter padding target; a 3-page transfer
        # excluding it must leave it untouched (loc=-1 skip works on the real
        # kernel, not just the CPU shim).
        mesh = create_device_mesh(ici_parallelism=[2, 2], dcn_parallelism=[1, 1])
        device_pool = _make_device_pool(mesh, dp_size=2)
        pool = _make_pool(device_pool, mesh, pool_size=8)
        guard = _fill(device_pool, 0, seed=42)
        srcs, dsts = [2, 3, 4], [5, 6, 7]
        for s in srcs:
            _fill(device_pool, s, seed=s + 1)
        host_pages = [int(p) for p in pool.alloc(3)]
        pool.copy_into(srcs, host_pages)
        pool.copy_to_device(host_pages, dsts)
        for layer in range(device_pool.layer_num):
            got = _read_page(device_pool, layer, 0)
            np.testing.assert_array_equal(got, guard[layer])


if __name__ == "__main__":
    unittest.main()
