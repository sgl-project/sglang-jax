"""Slice 4 I0/I1 on a 16-device TPU slice (jax.distributed, 2 hosts).

I0: compile ``owned_topk_jax`` inside shard_map over ``tensor``.
I1: planted needle at virtual 5 (owner rank 5, physical 0) is in that rank's
owned top-k and dropped on every other rank.

Run via the 2-host launcher (single-host jax init hangs on this slice)::

    JAX_DIST_ADDR=host0:63100 JAX_NPROC=2 JAX_PID=0|1 \\
      python -m sgl_jax.test.test_dcp_indexer_tpu
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def _init_distributed() -> int:
    addr = os.environ.get("JAX_DIST_ADDR")
    if not addr:
        raise SystemExit("JAX_DIST_ADDR is required (2-host TPU slice)")
    nproc = int(os.environ.get("JAX_NPROC", "2"))
    pid = int(os.environ.get("JAX_PID", "0"))
    jax.distributed.initialize(coordinator_address=addr, num_processes=nproc, process_id=pid)
    return pid


def main() -> None:
    pid = _init_distributed()
    devices = jax.devices()
    local = jax.local_device_count()
    if pid == 0:
        print(f"I0/I1 mesh: global={len(devices)} local={local}")
    if len(devices) != 16 or local != 8:
        raise SystemExit(f"expected 16 global / 8 local, got {len(devices)} / {local}")

    mesh = Mesh(np.array(devices).reshape(1, 16), ("data", "tensor"))
    dcp_size = 16
    k = 3
    needle_rank = 5
    needle_virt = 5  # physical 0 * 16 + rank 5

    def kernel(_):
        from sgl_jax.srt.layers.dcp.indexer import owned_topk_jax

        rank = jax.lax.axis_index("tensor")
        scores = jnp.full((1, 4), -jnp.inf, dtype=jnp.float32)
        scores = scores.at[0, 0].set(
            jnp.where(
                rank == needle_rank,
                jnp.float32(100.0),
                jnp.where(rank == 0, jnp.float32(1.0), -jnp.inf),
            )
        )
        return owned_topk_jax(scores, dcp_size, rank, k)

    dummy = jnp.zeros((1,), dtype=jnp.float32)
    fn = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=P("data"),
            out_specs=P("tensor", None),
            check_vma=False,
        )
    )
    t0 = time.perf_counter()
    local = jax.block_until_ready(fn(dummy))
    from jax.experimental.multihost_utils import process_allgather

    out = np.asarray(process_allgather(local, tiled=True))
    compile_s = time.perf_counter() - t0
    if pid != 0:
        return
    print(f"I0 compile+run_s={compile_s:.3f} owned_shape={out.shape}")
    print("owned_per_rank", out.tolist())
    if out.shape != (dcp_size, k):
        raise SystemExit(f"bad owned shape {out.shape}")
    if needle_virt not in out[needle_rank]:
        raise SystemExit(f"I1 FAIL: needle {needle_virt} missing on owner rank {needle_rank}")
    for r in range(dcp_size):
        if r == needle_rank:
            continue
        if needle_virt in out[r]:
            raise SystemExit(f"I1 FAIL: needle {needle_virt} leaked onto rank {r}")
    if 0 not in out[0]:
        raise SystemExit("I1 FAIL: rank 0 dummy virtual 0 missing")
    print("I1 PASS")


if __name__ == "__main__":
    main()
