"""Isolate conv-state write-back costs in `jax_causal_conv1d_update`.

The variants separately remove:

  barrier   `jax.lax.optimization_barrier(conv_state)` (guards a donated-pool
            aliasing race under multi-host SPMD -- measured, not removed)
  keep      the `buf[state_indices]` gather inside `_scatter_idx0_safe`
  scatter   `buf.at[state_indices].set(...)`

    python benchmark/kernels/ngram/bench_conv_state_writeback.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

CHANNELS = 10240  # HC * HS
STATE_LEN = 9  # (kernel - 1) * dilation
KERNEL = 4
DILATION = 3
DTYPE = jnp.bfloat16


def variants(mesh):
    """Same shard_map wrapper the layer uses: the pool is slot-sharded on
    `data`, channels on `tensor`, and the slot gather is device-local."""

    def wrap(local):
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                P("data", "tensor", None),  # conv_state
                P("data"),  # state_indices
                P("data", "tensor"),  # x
                P("tensor", None),  # weight
            ),
            out_specs=(P("data", "tensor"), P("data", "tensor", None)),
            check_vma=False,
        )

    def body(
        conv_state, state_indices, x, weight, *, barrier: bool, keep: bool, writeback: bool = True
    ):
        if barrier:
            conv_state = jax.lax.optimization_barrier(conv_state)
        state = conv_state[state_indices]  # [B, C, S]
        window = jnp.concatenate([state, x[..., None]], axis=-1)
        y = jax.nn.silu(jnp.einsum("bdk,dk->bd", window[..., ::DILATION], weight.astype(x.dtype)))
        if not writeback:
            return y, conv_state
        new_state = window[..., 1:].astype(conv_state.dtype)
        if keep:
            hold = (state_indices == 0).reshape((-1,) + (1,) * (conv_state.ndim - 1))
            new_state = jnp.where(hold, conv_state[state_indices], new_state)
        return y, conv_state.at[state_indices].set(new_state)

    import functools

    return {
        "as shipped": wrap(functools.partial(body, barrier=True, keep=True)),
        "no barrier": wrap(functools.partial(body, barrier=False, keep=True)),
        "no barrier/keep": wrap(functools.partial(body, barrier=False, keep=False)),
        "no write-back": wrap(functools.partial(body, barrier=False, keep=False, writeback=False)),
    }


def timeit(fn, argv, donated=0, reps=10, warm=3):
    sharding, shape = argv[donated].sharding, argv[donated].shape
    ts = []
    for i in range(warm + reps):
        call = list(argv)
        call[donated] = jax.block_until_ready(jax.device_put(jnp.zeros(shape, DTYPE), sharding))
        t = time.perf_counter()
        jax.block_until_ready(fn(*call))
        dt = (time.perf_counter() - t) * 1e3
        if i >= warm:
            ts.append(dt)
    return statistics.median(ts)


def main():
    mesh = Mesh(
        np.array(jax.devices())[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    rng = np.random.default_rng(0)
    print(f"{jax.devices()[0].device_kind}  conv_state=[slots, {CHANNELS}, {STATE_LEN}] bf16")

    jitted = {k: jax.jit(v, donate_argnums=(0,)) for k, v in variants(mesh).items()}
    widths = list(jitted)
    print(
        f"\n{'B':>5}{'slots':>7}{'pool MiB':>10}" + "".join(f"{k:>18}" for k in widths), flush=True
    )

    def put(x, spec):
        return jax.device_put(jnp.asarray(x), NamedSharding(mesh, spec))

    for batch in (256, 512):
        for num_slots in (256, 1024, 2048):
            if num_slots < batch:
                continue
            argv = [
                put(jnp.zeros((num_slots, CHANNELS, STATE_LEN), DTYPE), P("data", "tensor", None)),
                put(np.arange(batch, dtype=np.int32), P("data")),
                put(
                    jnp.asarray(rng.standard_normal((batch, CHANNELS)), DTYPE), P("data", "tensor")
                ),
                put(jnp.asarray(rng.standard_normal((CHANNELS, KERNEL)), DTYPE), P("tensor", None)),
            ]
            mib = num_slots * CHANNELS * STATE_LEN * 2 / 2**20
            row = "".join(f"{timeit(fn, argv):>18.3f}" for fn in jitted.values())
            print(f"{batch:>5}{num_slots:>7}{mib:>10.0f}{row}", flush=True)


if __name__ == "__main__":
    main()
