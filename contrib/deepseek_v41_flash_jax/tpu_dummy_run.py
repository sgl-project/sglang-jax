"""Real-dimension V4.1 with RANDOM weights on a TPU slice (compile time, memory, step time). Run the SAME command on every host:
    python tpu_dummy_run.py <hf config.json> [dp tp] [layers]
Multi-host SPMD: jax.distributed.initialize with host 0 as coordinator (env: HEAD_IP, default 127.0.0.1).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_real as C
import engine as EN
import jax
import jax.numpy as jnp
import model_ref as J
import sharding as SH
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

cfg = sys.argv[1]
dp, tp = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (4, 8)
n_layers = int(sys.argv[4]) if len(sys.argv) > 4 else None
rank = int(os.environ.get("RANK", "0"))
nproc = int(os.environ.get("NPROC", "8"))
head = os.environ.get("HEAD_IP", "127.0.0.1")
if nproc > 1:
    jax.distributed.initialize(f"{head}:{os.environ.get('PORT', '5000')}", nproc, rank)
log = lambda *m: print(f"[{rank}] {time.strftime('%H:%M:%S')}", *m, flush=True)
log(
    "devices:", jax.device_count(), "local:", jax.local_device_count(), jax.devices()[0].device_kind
)

a = C.args_from_hf_config(cfg, max_seq_len=int(os.environ.get("MAX_SEQ", "16384")))
if n_layers:  # cut-down model for quick tests (keeps the first n layers' types)
    import dataclasses

    a = dataclasses.replace(
        a,
        n_layers=n_layers,
        kv_source_layers=tuple(x for x in a.kv_source_layers if x < n_layers),
        index_source_layers=tuple(x for x in a.index_source_layers if x < n_layers),
        candidate_source_layer=(
            a.candidate_source_layer if a.candidate_source_layer < n_layers else -1
        ),
        engram_layer_ids=(),
    )
mesh = Mesh(mesh_utils.create_device_mesh((dp, tp)), ("dp", "tp"))
B = dp  # one batch row per data-parallel group
table = C.param_table(a)


def make_param(name, shape, dtype, key):
    """Cheap deterministic fill (iota patterns) -- values don't matter for timing, random generation is slow at this size."""
    sh = NamedSharding(mesh, SH.param_spec(name))
    n = int(np.prod(shape))

    def f(_):
        i = jnp.arange(n, dtype=jnp.int32).reshape(shape)
        if dtype == jnp.float8_e4m3fn:
            return (((i * 7) % 25 - 12).astype(jnp.float32) * 0.25).astype(jnp.float8_e4m3fn)
        if dtype == jnp.int8:
            return ((i * 7) % 25 - 12).astype(jnp.int8)
        if dtype == jnp.uint8:
            return (120 + i % 6).astype(jnp.uint8)
        if (
            name.endswith("norm.weight")
            or name.endswith("_scale")
            or name.endswith("_rowscale")
            or name.endswith("q_weight")
            or name.endswith("k_weight")
        ):
            return jnp.ones(shape, dtype)
        if name.endswith("attn_sink") or name.endswith("gate.bias") or name.endswith("_base"):
            return jnp.zeros(shape, dtype)
        return (jnp.sin(i.astype(jnp.float32) * 0.00137) * 0.02).astype(dtype)

    return jax.jit(f, out_shardings=sh)(0)


t0 = time.time()
params = {}
key = jax.random.PRNGKey(0)
for i, (name, (shape, dtype)) in enumerate(table.items()):
    key, sub = jax.random.split(key)
    params[name] = make_param(name, shape, dtype, sub)
jax.block_until_ready(params)
log(f"params built: {len(params)} tensors in {time.time()-t0:.0f}s")
ms = jax.local_devices()[0].memory_stats() or {}
log(
    "HBM in use after params (device 0): %.2f GB of %.2f GB"
    % (ms.get("bytes_in_use", 0) / 1e9, ms.get("bytes_limit", 0) / 1e9)
)

rep = NamedSharding(mesh, P())
ropes = [
    tuple(jax.device_put(np.asarray(x), rep) for x in J.make_rope(a, l)) for l in range(a.n_layers)
]
states = SH.shard_states(EN.init_states(a, B, jnp.bfloat16), mesh)
step = jax.jit(
    EN.chunk_forward, static_argnames=("a", "want_logits", "engram_fn"), donate_argnames=("states",)
)


def toks(S, seed):
    r = np.random.default_rng(seed)
    return jax.device_put(
        r.integers(0, a.vocab_size, (B, S), dtype=np.int32), NamedSharding(mesh, P("dp", None))
    )


def run(S, pos, seed):
    global states
    t = time.time()
    lg, states = step(params, a, toks(S, seed), jax.device_put(np.int32(pos), rep), states, ropes)
    jax.block_until_ready(lg)
    return time.time() - t, lg


with mesh:
    for S in (int(x) for x in os.environ.get("CHUNKS", "128").split(",")):
        dt, lg = run(S, 0, 1)
        log(f"prefill chunk S={S}: first call (compile+run) {dt:.1f}s")
        dt, lg = run(S, S, 2)
        log(
            f"prefill chunk S={S}: second call {dt*1000:.0f} ms  logits finite: {bool(np.isfinite(np.asarray(lg.addressable_shards[0].data)).all())}"
        )
    pos = int(os.environ.get("CHUNKS", "128").split(",")[-1]) * 2
    dt, lg = run(1, pos, 3)
    log(f"decode S=1: first call (compile+run) {dt:.1f}s")
    ts = []
    for i in range(20):
        dt, lg = run(1, pos + 1 + i, 4 + i)
        ts.append(dt)
    log(
        f"decode S=1: median {np.median(ts)*1000:.1f} ms/step  min {np.min(ts)*1000:.1f}  max {np.max(ts)*1000:.1f}  (batch rows={B})"
    )
    if os.environ.get("PROFILE_DIR"):
        with jax.profiler.trace(os.environ["PROFILE_DIR"]):
            for i in range(6):
                run(1, pos + 30 + i, 50 + i)
        log("profile written to", os.environ["PROFILE_DIR"])
ms = jax.local_devices()[0].memory_stats() or {}
log(
    "HBM peak (device 0): %.2f GB of %.2f GB"
    % (ms.get("peak_bytes_in_use", 0) / 1e9, ms.get("bytes_limit", 0) / 1e9)
)
log("DONE")
