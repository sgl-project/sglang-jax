"""Slice 5 D0-on-device: gather + LSE merge on a 16-device TPU slice.

Each rank attends one striped key; merge must match full softmax.
Run via the 2-host launcher (single-host jax init hangs)::

    JAX_DIST_ADDR=host0:63102 JAX_NPROC=2 JAX_PID=0|1 \\
      python -m sgl_jax.test.test_dcp_merge_tpu
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
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


def _softmax_attn(q, k, v):
    scores = q @ k.T
    m = np.max(scores, axis=-1, keepdims=True)
    ex = np.exp(scores - m)
    z = np.sum(ex, axis=-1, keepdims=True)
    out = (ex / z) @ v
    lse = (m.squeeze(-1) + np.log(z.squeeze(-1))).astype(np.float32)
    return out.astype(np.float32), lse


def main() -> None:
    pid = _init_distributed()
    devices = jax.devices()
    local = jax.local_device_count()
    if pid == 0:
        print(f"D0-tpu mesh: global={len(devices)} local={local}")
    if len(devices) != 16 or local != 8:
        raise SystemExit(f"expected 16 global / 8 local, got {len(devices)} / {local}")

    rng = np.random.default_rng(0)
    d, s, dcp_size = 8, 16, 16
    q = rng.normal(size=(1, d)).astype(np.float32)
    k = rng.normal(size=(s, d)).astype(np.float32)
    v = rng.normal(size=(s, d)).astype(np.float32)
    full_o, full_lse = _softmax_attn(q, k, v)

    mesh = Mesh(np.array(devices).reshape(1, 16), ("data", "tensor"))
    q_j, k_j, v_j = jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)

    def kernel(_):
        from sgl_jax.srt.layers.dcp.comm import gather_merge_dcp_attention

        rank = jax.lax.axis_index("tensor")
        virt = jnp.arange(s // dcp_size, dtype=jnp.int32) * dcp_size + rank
        k_local = jnp.take(k_j, virt, axis=0)
        v_local = jnp.take(v_j, virt, axis=0)
        scores = q_j @ k_local.T
        m = jnp.max(scores, axis=-1, keepdims=True)
        ex = jnp.exp(scores - m)
        z = jnp.sum(ex, axis=-1, keepdims=True)
        o_r = (ex / z) @ v_local
        lse_r = (m.squeeze(-1) + jnp.log(z.squeeze(-1))).astype(jnp.float32)
        return gather_merge_dcp_attention(o_r, lse_r)

    dummy = jnp.zeros((1,), dtype=jnp.float32)
    fn = jax.jit(
        jax.shard_map(
            kernel,
            mesh=mesh,
            in_specs=P("data"),
            out_specs=P("data", None),
            check_vma=False,
        )
    )
    t0 = time.perf_counter()
    merged = np.asarray(process_allgather(jax.block_until_ready(fn(dummy)), tiled=True))
    compile_s = time.perf_counter() - t0
    if pid != 0:
        return
    got = merged.reshape(-1, d)[0]
    print(f"D0-tpu compile+run {compile_s:.2f}s")
    np.testing.assert_allclose(got, full_o[0], rtol=1e-4, atol=1e-4)
    print(f"D0-tpu PASS (ref lse={full_lse[0]:.4f})")


if __name__ == "__main__":
    main()
