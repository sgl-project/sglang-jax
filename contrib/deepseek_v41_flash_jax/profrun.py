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
EN.MESH = mesh
B = int(os.environ.get("BATCH", dp))  # batch rows (default one per data-parallel group)
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
_rc = {}
for l in range(a.n_layers):
    k = bool(a.compress_ratios[l])
    if k not in _rc:
        _rc[k] = tuple(jax.device_put(np.asarray(x), rep) for x in J.make_rope(a, l))
ropes = {"c": _rc.get(True, _rc.get(False)), "b": _rc.get(False, _rc.get(True))}
states = SH.init_states_sharded(lambda: EN.init_states(a, B, jnp.bfloat16), mesh)
step = jax.jit(
    EN.chunk_forward,
    static_argnames=("a", "want_logits", "engram_fn", "cap"),
    donate_argnames=("states",),
)


def toks(S, seed):
    r = np.random.default_rng(seed)
    full = r.integers(
        0, a.vocab_size, (B, S), dtype=np.int32
    )  # same on every host (same seed); local-shard assembly, no cross-host check
    return jax.make_array_from_callback(
        (B, S), NamedSharding(mesh, P("dp", None)), lambda idx: full[idx]
    )


cap = int(os.environ.get("CAP", "1048576")) or None
modes = os.environ.get("MODES", "dense,grouped").split(",")
shapes = [int(x) for x in os.environ.get("SHAPES", "1,512").split(",")]
for B in [int(x) for x in os.environ.get("BATCHES", str(B)).split(",")]:
    states = SH.init_states_sharded(lambda: EN.init_states(a, B, jnp.bfloat16), mesh)
    act = jax.device_put(np.ones(B, bool), rep)
    log(f"=== batch rows B={B}")

    erows = lambda S: (
        {
            l: jax.device_put(
                np.zeros(
                    (B, S, a.engram_n_heads * (a.engram_max_ngram_size - 1), a.engram_head_dim),
                    np.float32,
                ),
                rep,
            )
            for l in a.engram_layer_ids
        }
        if a.engram_layer_ids
        else None
    )
    POS = int(os.environ.get("POS", "20000"))
    with mesh:
        for mode in modes:
            J.MOE_MESH = mesh if mode.startswith(("grouped", "kernel")) else None
            if mode.startswith("kernel"):
                _m = mode[len("kernel") :]
                os.environ["V41_KERNEL_ANCHOR_MAX_TOKENS"] = "1000000" if "A" in _m else "32"
                _m = _m.replace("A", "")
                os.environ["V41_MOE_OUT"] = "rep" if _m.endswith("rep") else "dp"
                os.environ["V41_KERNEL_MIN_TOKENS"] = _m.replace("rep", "") or "64"
                if (
                    next(iter(params.values())).ndim
                    and params["layers.2.ffn.experts.w1_rowscale"].ndim == 2
                ):
                    t0_ = time.time()
                    params = SH.to_kernel_layout(params, mesh, a.n_layers)
                    log(f"converted experts to kernel layout in {time.time()-t0_:.0f}s")
                os.environ["V41_MOE"] = "kernel"

            def _f(
                params,
                a,
                tokens,
                pos0,
                states,
                ropes,
                engram_rows=None,
                want_logits=True,
                engram_fn=None,
                active=None,
                cap=None,
                _m=mode,
            ):
                return EN.chunk_forward(
                    params,
                    a,
                    tokens,
                    pos0,
                    states,
                    ropes,
                    engram_rows,
                    want_logits,
                    engram_fn,
                    active,
                    cap,
                )  # fresh fn per mode: separate jit cache

            stepm = jax.jit(
                _f,
                static_argnames=("a", "want_logits", "engram_fn", "cap", "_m"),
                donate_argnames=("states",),
            )
            for S in shapes:
                pos = jax.device_put(np.full(B, POS, np.int32), rep)

                def go(i):
                    global states
                    lg, states = stepm(
                        params,
                        a,
                        toks(S, i),
                        pos,
                        states,
                        ropes,
                        erows(S),
                        active=act,
                        cap=cap,
                        _m=mode,
                    )
                    jax.block_until_ready(lg)

                t = time.time()
                go(0)
                log(f"[{mode}] S={S} first call (compile+run) {time.time()-t:.1f}s")
                ts = []
                for i in range(3):
                    t = time.time()
                    go(i + 1)
                    ts.append(time.time() - t)
                log(
                    f"[{mode}] S={S} step median {np.median(ts)*1000:.1f} ms (min {min(ts)*1000:.1f})"
                )
                # pipelined: 4 calls dispatched back to back, one block at the end (separates device time from dispatch/sync latency)
                t = time.time()
                lgs = []
                for i in range(4):
                    lg, states = stepm(
                        params,
                        a,
                        toks(S, 50 + i),
                        pos,
                        states,
                        ropes,
                        erows(S),
                        active=act,
                        cap=cap,
                        _m=mode,
                    )
                    lgs.append(lg)
                t_disp = time.time() - t
                jax.block_until_ready(lgs)
                t_all = time.time() - t
                log(
                    f"[{mode}] S={S} pipelined x4: {t_all/4*1000:.1f} ms/call (dispatch of 4 calls took {t_disp*1000:.1f} ms)"
                )
                d = f"/tmp/prof_{mode}_{S}_b{B}"
                if os.environ.get("PROF", "0") == "1":
                    with jax.profiler.trace(d):
                        for i in range(3):
                            go(10 + i)

log("DONE")
