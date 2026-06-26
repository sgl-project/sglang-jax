"""Route D: Pathways single-controller cross-slice P/D split.

Architecture: one Python process (Pathways head) sees all devices across
N slices via IFRT proxy. Build prefill_mesh on slice 0, decode_mesh on
slice 1, run two ModelRunners. KV transfer = jax.device_put across meshes
(IFRT runtime handles per-host DCN fan-out, ~14 GB/s/host on v7x vs ~4.5
for jax.experimental.transfer on the same hardware).

Reuses ici_pd's gather/scatter jit kernels; replaces the
embed->ppermute->extract path with a single device_put.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.disaggregation.ici_pd import (
    _NPG_BATCH_MAX,
    _bucket_npg,
    migrate_reqs_p_to_d,
    slots_to_ordered_pages,
)

__all__ = [
    "group_by_slice",
    "make_slice_meshes",
    "PathwaysPDKVTransfer",
    "migrate_reqs_p_to_d",
    "slots_to_ordered_pages",
]

logger = logging.getLogger(__name__)


def group_by_slice(devs: list) -> dict[int, list]:
    """Group devices by slice_index. Fallback: split in half by id order."""
    groups: dict[int, list] = defaultdict(list)
    for d in devs:
        sid = getattr(d, "slice_index", None)
        if sid is not None:
            groups[sid].append(d)
    if len(groups) >= 2:
        return dict(groups)
    ordered = sorted(devs, key=lambda d: d.id)
    half = len(ordered) // 2
    return {0: ordered[:half], 1: ordered[half:]}


def make_slice_meshes(
    dp_size: int, tp_per_side: int, n_prefill: int = 1
) -> tuple[list[Mesh], Mesh]:
    """Build ([prefill_mesh_0..n_prefill-1], decode_mesh) from the first
    n_prefill+1 slices. Stage 6 multi-P: n_prefill>1 fans prefill across
    slices to lift R_prefill (the bottleneck on device-bound models)."""
    devs = jax.devices()
    groups = group_by_slice(devs)
    sids = sorted(groups)
    need = n_prefill + 1
    if len(sids) < need:
        raise RuntimeError(
            f"pathways_pd n_prefill={n_prefill} requires >={need} slices, got {len(sids)} "
            f"(devices={len(devs)}, backend={jax.default_backend()})"
        )
    tp_axis = tp_per_side // dp_size
    et = (jax.sharding.AxisType.Explicit,) * 2

    def _mk(ds: list) -> Mesh:
        if len(ds) != tp_per_side:
            raise RuntimeError(f"slice size {len(ds)} != tp_per_side {tp_per_side}")
        return Mesh(
            np.asarray(ds).reshape(dp_size, tp_axis),
            axis_names=("data", "tensor"),
            axis_types=et,
        )

    p_meshes = [_mk(groups[sids[i]]) for i in range(n_prefill)]
    d_mesh = _mk(groups[sids[n_prefill]])
    logger.info(
        "[pathways_pd] slices=%d n_prefill=%d p_slices=%s d_slice=%d shape=%s",
        len(sids),
        n_prefill,
        sids[:n_prefill],
        sids[n_prefill],
        d_mesh.shape,
    )
    return p_meshes, d_mesh


class PathwaysPDKVTransfer:
    """Paged KV pool P-mesh -> D-mesh via cross-slice device_put.

    gather(P pages) -> stack L layers -> device_put to d_mesh sharding
    -> scatter into D pool (donate). Single transfer per batch maximizes
    payload (P0 measured 28.9 GB/s aggregate at 2048 MiB on 2x v7x-8).
    """

    def __init__(self, p_mesh: Mesh, d_mesh: Mesh, p_pool, d_pool) -> None:
        self.p_mesh = p_mesh
        self.d_mesh = d_mesh
        self.p_pool = p_pool
        self.d_pool = d_pool
        self.L = p_pool.layer_num
        self.page_shape = p_pool.kv_buffer[0].shape[1:]
        # kv_buffer real spec is P("data", None, "tensor", None, None) (5-dim fused
        # KV); the previous hardcoded 4-dim P("data",None,None,None) made XLA
        # all-gather dim2 (tensor=16) across 36 layers -> ~170ms/scatter on D
        # device (xprof confirmed: scatter_custom_fusion 0.005ms vs module 170ms).
        kv_spec = d_pool.kv_sharding.spec
        page_spec = P(None, *kv_spec[1:])  # gathered pages: dim0=npg unsharded
        stack_spec = P(None, *kv_spec)  # stacked: [L, num_pages_dim, ...]
        self._d_stack_shard = NamedSharding(d_mesh, stack_spec)

        self._gather_stack = jax.jit(
            lambda bufs, idx: jnp.stack([b.at[idx].get(out_sharding=page_spec) for b in bufs])
        )
        self._scatter_jit = jax.jit(
            lambda bufs, idx, stacked: tuple(
                b.at[idx].set(stacked[i], out_sharding=kv_spec) for i, b in enumerate(bufs)
            ),
            donate_argnums=(0,),
        )
        self._p_idx_shard = NamedSharding(p_mesh, P(None))
        self._d_idx_shard = NamedSharding(d_mesh, P(None))

    def gather_to_dmesh(self, p_pages: np.ndarray) -> tuple[jax.Array, int]:
        """gather P pool pages -> stack -> device_put to D mesh -> block.

        Runs in the prefill thread; blocks here so the main (decode) thread
        never waits on cross-slice transfer. Returns (d_stacked, bucket_len).
        """
        p_pages = np.asarray(p_pages, np.int32)
        npg = len(p_pages)
        bucket = _bucket_npg(npg)
        if bucket > npg:
            p_pages = np.concatenate([p_pages, np.full(bucket - npg, p_pages[-1], np.int32)])
        p_idx = jax.device_put(p_pages, self._p_idx_shard)
        with jax.set_mesh(self.p_mesh):
            p_stacked = self._gather_stack(tuple(self.p_pool.kv_buffer), p_idx)
        # Do NOT block on d_stacked: kRecvRefs on the dst slice is hard-coded
        # kSerializedOnDeviceThread (worker_op_dispatcher.cc GetOpSyncMode) and
        # queues behind concurrent decode kExecute (~15*17ms backpressure).
        # Return the future; main-thread scatter consumes it via data-dep so
        # the RecvRefs lands in the D Processor queue early and interleaves
        # with decode instead of stalling this thread for ~250ms.
        d_stacked = jax.device_put(p_stacked, self._d_stack_shard)
        return d_stacked, bucket

    def scatter_from_dmesh(self, d_pages: np.ndarray, d_stacked: jax.Array, bucket: int) -> None:
        """scatter d_stacked into D pool. Called on the main thread so the
        donated d_pool.kv_buffer reassignment is ordered before the next decode
        forward dispatch (data-dependency guarantees device-side ordering)."""
        d_pages = np.asarray(d_pages, np.int32)
        npg = len(d_pages)
        if bucket > npg:
            d_pages = np.concatenate([d_pages, np.full(bucket - npg, d_pages[-1], np.int32)])
        d_idx = jax.device_put(d_pages, self._d_idx_shard)
        with self.d_pool._donate_lock, jax.set_mesh(self.d_mesh):
            new_bufs = self._scatter_jit(tuple(self.d_pool.kv_buffer), d_idx, d_stacked)
            self.d_pool.kv_buffer = list(new_bufs)

    def transfer(self, p_page_indices: np.ndarray, d_page_indices: np.ndarray) -> None:
        p_page_indices = np.asarray(p_page_indices, np.int32)
        d_page_indices = np.asarray(d_page_indices, np.int32)
        n = len(p_page_indices)
        assert n == len(d_page_indices)
        for i in range(0, n, _NPG_BATCH_MAX):
            p = p_page_indices[i : i + _NPG_BATCH_MAX]
            d = d_page_indices[i : i + _NPG_BATCH_MAX]
            if len(p) == 0:
                continue
            d_stacked, bucket = self.gather_to_dmesh(p)
            self.scatter_from_dmesh(d, d_stacked, bucket)
