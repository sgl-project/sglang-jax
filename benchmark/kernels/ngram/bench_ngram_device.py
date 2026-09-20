"""Measure NGramEmbedding gate and post-gate time on TPU.

The sweep uses the runner's bf16 donated conv state and varies `num_slots` to
expose state-pool costs. `bench_conv_state_writeback.py` isolates them further.

    python benchmark/kernels/ngram/bench_ngram_device.py [--hlo]
"""

from __future__ import annotations

import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.ngram_embedding import NGramEmbedding

# Released Qwen4Exp checkpoint.
HIDDEN_SIZE = 2560
HC_COUNT = 4
HYPER_SIZE = HIDDEN_SIZE * HC_COUNT  # 10240
PLE_EMBED_DIM = 2560
NGRAM_SIZE = 3
CONV_KERNEL = 4
CONV_STATE_LEN = (CONV_KERNEL - 1) * NGRAM_SIZE  # 9
ACT_DTYPE = jnp.bfloat16
CONV_DTYPE = jnp.bfloat16  # recurrent_state_dtype().conv

DECODE_ARGS = ("hyper_input", "ple_embeddings", "conv_state", "state_indices", "has_initial_state")
EXTEND_ARGS = (
    "hyper_input",
    "ple_embeddings",
    "conv_state",
    "state_indices",
    "cu_seqlens",
    "has_initial_state",
)


class _Config:
    hidden_size = HIDDEN_SIZE
    hc_count = HC_COUNT
    ple_embed_dim = PLE_EMBED_DIM
    ple_conv_kernel_size = CONV_KERNEL
    ngram_size = NGRAM_SIZE
    rms_norm_eps = 1e-6


def build(mesh, rng):
    with jax.set_mesh(mesh):
        layer = NGramEmbedding(_Config(), mesh, params_dtype=ACT_DTYPE)

    def put(param, shape, spec):
        param[...] = jax.device_put(
            jnp.asarray(rng.standard_normal(shape) * 0.02, ACT_DTYPE),
            NamedSharding(mesh, spec),
        )

    put(layer.key_proj.weight, (PLE_EMBED_DIM, HYPER_SIZE), P(None, None))
    put(layer.value_proj.weight, (PLE_EMBED_DIM, HIDDEN_SIZE), P(None, None))
    put(layer.norm_key.weight, (HYPER_SIZE,), P(None))
    put(layer.norm_query.weight, (HYPER_SIZE,), P(None))
    put(layer.norm_conv.weight, (HYPER_SIZE,), P(None))
    put(layer.conv1d_weight, (HYPER_SIZE, CONV_KERNEL), P(None, None))
    return layer


def inputs(mesh, rng, num_tokens, num_reqs, num_slots):
    def put(x, spec):
        return jax.device_put(jnp.asarray(x), NamedSharding(mesh, spec))

    lens = np.full(num_reqs, num_tokens // num_reqs, np.int32)
    lens[: num_tokens % num_reqs] += 1
    cu = np.zeros(num_reqs + 1, np.int32)
    np.cumsum(lens, out=cu[1:])
    act = rng.standard_normal((num_tokens, HYPER_SIZE)).astype(np.float32)
    emb = rng.standard_normal((num_tokens, PLE_EMBED_DIM)).astype(np.float32)
    return {
        "hyper_input": put(jnp.asarray(act, ACT_DTYPE), P("data", "tensor")),
        "ple_embeddings": put(jnp.asarray(emb, ACT_DTYPE), P("data", None)),
        "conv_state": put(
            jnp.zeros((num_slots, HYPER_SIZE, CONV_STATE_LEN), CONV_DTYPE),
            P("data", "tensor", None),
        ),
        "state_indices": put(np.arange(num_reqs, dtype=np.int32), P("data")),
        "has_initial_state": put(np.ones(num_reqs, bool), P("data")),
        "cu_seqlens": put(cu, P("data")),
    }


def timeit(fn, argv, donated=None, reps=30, warm=5):
    """`donated` is the index of the donated positional arg; donation deletes
    the buffer, so it is re-placed before each call and only the call is timed."""
    ts = []
    sharding = argv[donated].sharding if donated is not None else None
    shape = argv[donated].shape if donated is not None else None
    for i in range(warm + reps):
        call = list(argv)
        if donated is not None:
            call[donated] = jax.block_until_ready(
                jax.device_put(jnp.zeros(shape, CONV_DTYPE), sharding)
            )
        t = time.perf_counter()
        jax.block_until_ready(fn(*call))
        dt = (time.perf_counter() - t) * 1e3
        if i >= warm:
            ts.append(dt)
    return statistics.median(ts)


def hlo_shape(compiled, label):
    lines = compiled.as_text().splitlines()
    print(
        f"    [{label}] fusions={sum('fusion(' in x for x in lines)} "
        f"custom-calls={sum('custom-call' in x for x in lines)} "
        f"dots={sum('dot(' in x for x in lines)} lines={len(lines)}"
    )
    if "--hlo" in sys.argv:
        for line in lines:
            if any(k in line for k in ("fusion(", "custom-call", "dot(", "ROOT")):
                print("       ", line.strip()[:180])


def main():
    mesh = Mesh(
        np.array(jax.devices())[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    rng = np.random.default_rng(0)
    layer = build(mesh, rng)
    print(
        f"{jax.devices()[0].device_kind} x{jax.device_count()}  C={HYPER_SIZE}  "
        f"ple_dim={PLE_EMBED_DIM}  conv_state=[slots, {HYPER_SIZE}, {CONV_STATE_LEN}] bf16"
    )

    gate_fn = jax.jit(layer.gate)
    decode_fn = jax.jit(layer.forward_decode, donate_argnums=(2,))
    extend_fn = jax.jit(layer.forward_extend, donate_argnums=(2,))

    print(
        f"\n{'case':<15}{'slots':>7}{'pool MiB':>10}{'full ms':>9}"
        f"{'gate ms':>9}{'post ms':>9}{'post %':>8}"
    )
    for name, num_tokens, num_reqs, mode in [
        ("decode B=256", 256, 256, "decode"),
        ("decode B=512", 512, 512, "decode"),
        ("extend T=2048", 2048, 4, "extend"),
        ("extend T=8192", 8192, 4, "extend"),
    ]:
        for num_slots in (256, 1024, 4096):
            if num_slots < num_reqs:
                continue
            args = inputs(mesh, rng, num_tokens, num_reqs, num_slots)
            fn, names = (decode_fn, DECODE_ARGS) if mode == "decode" else (extend_fn, EXTEND_ARGS)
            argv = [args[k] for k in names]
            full = timeit(fn, argv, donated=2)
            only_gate = timeit(gate_fn, [args["hyper_input"], args["ple_embeddings"]])
            post_gate = full - only_gate
            mib = num_slots * HYPER_SIZE * CONV_STATE_LEN * 2 / 2**20
            print(
                f"{name:<15}{num_slots:>7}{mib:>10.0f}{full:>9.3f}"
                f"{only_gate:>9.3f}{post_gate:>9.3f}{100 * post_gate / full:>7.0f}%"
            )
            if num_slots == 1024:
                hlo_shape(fn.lower(*argv).compile(), name)


if __name__ == "__main__":
    main()
