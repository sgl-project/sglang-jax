"""Can XLA:TPU gather rows from a host-resident table? No -- this proves it.

The whole PLE design turns on this: the table is 95.4 GiB and v6e HBM is
31.24 GiB, so the table lives in host RAM, so the gather is numpy, so the hash
that feeds it is numpy too. If this ever starts working, the host round trip
can go away and the hash should move to the device with it.

WARNING: the third case aborts the process. A gather whose operand is in host
memory fails an XLA CHECK, not a Python exception:

    F lowering_util.cc:2168] Check failed:
        source->memory_space() != MemorySpace::kHost (host vs. host)

Run it on its own, not inside a test suite.

    python benchmark/kernels/ngram/probe_host_gather.py
"""

import jax
import jax.numpy as jnp
import numpy as np

ROWS, DIM = 1 << 20, 160  # small stand-in for the 320M-row table
dev = jax.devices()[0]
print("device:", dev.device_kind)
print("memory kinds:", [m.kind for m in dev.addressable_memories()])

tbl = jnp.asarray(np.zeros((ROWS, DIM), np.float32))
idx = jnp.asarray(np.random.default_rng(0).integers(0, ROWS, 4096, np.int32))


def take(t, i):
    return jnp.take(t, i, axis=0)


for kind in ("device", "pinned_host", "unpinned_host"):
    try:
        s = jax.sharding.SingleDeviceSharding(dev, memory_kind=kind)
        t = jax.device_put(tbl, s)
    except Exception as e:
        print(f"  {kind:<14} place FAILED: {type(e).__name__}: {str(e)[:110]}")
        continue
    for out_kind in (kind, "device"):
        try:
            out_s = jax.sharding.SingleDeviceSharding(dev, memory_kind=out_kind)
            f = jax.jit(take, out_shardings=out_s)
            r = jax.block_until_ready(f(t, idx))
            print(f"  {kind:<14} -> out {out_kind:<14} OK   {r.shape} {r.sharding.memory_kind}")
        except Exception as e:
            print(f"  {kind:<14} -> out {out_kind:<14} FAIL {type(e).__name__}: {str(e)[:150]}")
