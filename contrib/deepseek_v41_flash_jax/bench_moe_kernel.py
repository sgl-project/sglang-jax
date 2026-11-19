"""SGL-JAX fused_ep_moe_v2 (Pallas, EP=32, fp8 per-channel) vs our MoE paths at real V4.1 shapes, one layer.
Run the SAME command on every host:  python bench_moe_kernel.py [tokens,...]   (env RANK/NPROC/HEAD_IP as in profrun)
Checks numerics against our dense fp8-rowscale path and prints median step times (compile excluded)."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
import model_ref as J
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

rank, nproc = int(os.environ.get("RANK", "0")), int(os.environ.get("NPROC", "8"))
if nproc > 1:
    jax.distributed.initialize(f"{os.environ.get('HEAD_IP', '127.0.0.1')}:5000", nproc, rank)
log = lambda *m: print(f"[{rank}] {time.strftime('%H:%M:%S')}", *m, flush=True)
E, D, F, K = 384, 5120, 2304, 6
mesh = Mesh(mesh_utils.create_device_mesh((4, 8)), ("dp", "tp"))
EP = P(("dp", "tp"))
sh = lambda *spec: NamedSharding(mesh, P(*spec))
rng = jax.random.PRNGKey(0)

INT8 = bool(
    os.environ.get("INT8")
)  # INT8=1: experts stored as int8 (range +-96, like exact FP4*2^spread) instead of fp8
WDT = jnp.int8 if INT8 else jnp.float8_e4m3fn


def make(shape, dtype, scale, spec):
    def f(i):
        k = jax.random.fold_in(rng, i)
        if dtype == jnp.int8:
            return jax.random.randint(k, shape, -96, 97, jnp.int32).astype(jnp.int8)
        return (jax.random.normal(k, shape, jnp.float32) * scale).astype(dtype)

    return jax.jit(f, out_shardings=sh(*spec))(0)


# ours: w1,w3 [E,F,D], w2 [E,D,F] fp8 + row scales [E,F]/[E,F]/[E,D]
w1 = make((E, F, D), WDT, 1.0, (("dp", "tp"), None, None))
w3 = make((E, F, D), WDT, 1.0, (("dp", "tp"), None, None))
w2 = make((E, D, F), WDT, 1.0, (("dp", "tp"), None, None))
s1 = make((E, F), jnp.float32, 0.0, (("dp", "tp"), None)) + 0.01
s3 = make((E, F), jnp.float32, 0.0, (("dp", "tp"), None)) + 0.01
s2 = make((E, D), jnp.float32, 0.0, (("dp", "tp"), None)) + 0.01
# kernel layout: w1,w3 [E,D,F], w2 [E,F,D]; scales [E,1,1,out]
tr = jax.jit(lambda a: jnp.swapaxes(a, 1, 2), out_shardings=sh(("dp", "tp"), None, None))
kw1, kw3, kw2 = tr(w1), tr(w3), tr(w2)
ks = lambda s: jax.jit(
    lambda a: a[:, None, None, :], out_shardings=sh(("dp", "tp"), None, None, None)
)(s)
ks1, ks3, ks2 = ks(s1), ks(s3), ks(s2)
jax.block_until_ready((kw1, kw3, kw2, ks1, ks3, ks2))
log("weights built (both layouts)")

from sgl_jax.srt.kernels.fused_moe.v2.kernel import FusedMoEBlockConfig, fused_ep_moe_v2
from sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs import (
    get_tuned_fused_moe_v2_block_config,
)

LIMIT = 10.0
a = type(
    "A",
    (),
    dict(
        dim=D,
        n_routed_experts=E,
        n_activated_experts=K,
        route_scale=1.5,
        swiglu_limit=LIMIT,
        score_func="sqrtsoftplus",
        norm_topk_prob=True,
        gate_temp=1.0,
    ),
)()
shared = {
    k: make(s, jnp.bfloat16, 0.02, (None, None))
    for k, s in (("w1", (F, D)), ("w2", (D, F)), ("w3", (F, D)))
}
gate_w = make((E, D), jnp.float32, 0.05, (None, None))
gate_b = make((E,), jnp.float32, 0.0, (None,))
P_ours = {
    "gate.weight": gate_w,
    "gate.bias": gate_b,
    "experts.w1": w1,
    "experts.w3": w3,
    "experts.w2": w2,
    "experts.w1_rowscale": s1,
    "experts.w3_rowscale": s3,
    "experts.w2_rowscale": s2,
    "shared_experts.w1.weight": shared["w1"],
    "shared_experts.w2.weight": shared["w2"],
    "shared_experts.w3.weight": shared["w3"],
}

KW = {"w1": kw1, "w2": kw2, "w3": kw3, "s1": ks1, "s2": ks2, "s3": ks3}
rep = sh()


def route(x, gate_w, gate_b):
    sc = jnp.sqrt(jax.nn.softplus(x.astype(jnp.float32) @ gate_w.T))
    _, idx = jax.lax.top_k(sc + gate_b, K)
    w = jnp.take_along_axis(sc, idx, axis=1)
    w = w / (w.sum(-1, keepdims=True) + 1e-20) * 1.5
    return w, idx.astype(jnp.int32)


def kernel_moe(x, w, idx, kw, cfg):
    tok_sh = NamedSharding(mesh, P(("dp", "tp"), None))
    x_, w_, i_ = (jax.lax.with_sharding_constraint(v, tok_sh) for v in (x, w, idx))
    out = fused_ep_moe_v2(
        mesh,
        x_,
        kw["w1"],
        kw["w2"],
        kw["w3"],
        w_,
        i_,
        K,
        act_fn="silu",
        swiglu_limit=LIMIT,
        block_config=cfg,
        quant_block_k=None,
        w1_scale=kw["s1"],
        w2_scale=kw["s2"],
        w3_scale=kw["s3"],
        direct_scaled_dot=True,
        dp_axis_name="dp",
        tp_axis_name="tp",
    )
    return jax.lax.with_sharding_constraint(out, NamedSharding(mesh, P(None, None)))


def ours(x, Pm, mode):
    J.MOE_MESH = mesh if mode != "dense" else None
    os.environ["V41_MOE"] = "loop"
    return J.moe(Pm, x[None], a)[0]


def timeit(fn, *args, n=8):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t)
    return np.median(ts) * 1000


j_route = jax.jit(route, out_shardings=(rep, rep))
j_dense = jax.jit(lambda x, Pm: ours(x, Pm, "dense"), out_shardings=rep)
j_loop = jax.jit(lambda x, Pm: ours(x, Pm, "loop"), out_shardings=rep)
j_kernel = jax.jit(kernel_moe, out_shardings=rep, static_argnames=("cfg",))
CONFIGS = [
    FusedMoEBlockConfig(bt=bt, bf=bf, btc=btc, bse=256)
    for bt, bf, btc in (
        (128, 256, 128),
        (128, 768, 128),
        (64, 768, 64),
        (256, 768, 256),
        (128, 1152, 128),
        (32, 768, 32),
        (16, 768, 16),
        (8, 768, 8),
    )
]
j_shared = jax.jit(
    lambda x, sw: J.expert(sw["w1"], sw["w2"], sw["w3"], x, LIMIT), out_shardings=rep
)
j_cmp = jax.jit(
    lambda a_, b_: (
        jnp.max(jnp.abs(a_.astype(jnp.float32) - b_.astype(jnp.float32))),
        jnp.max(jnp.abs(b_.astype(jnp.float32))),
    ),
    out_shardings=(rep, rep),
)


def chain(x, wgt, idx, kw, cfg, L):
    out = x
    for _ in range(L):
        out = out + kernel_moe(out, wgt, idx, kw, cfg).astype(out.dtype)
    return out


if os.environ.get("CHAIN"):
    N0 = int(os.environ["CHAIN"])
    x0 = jax.device_put(
        (np.random.default_rng(1).standard_normal((N0, D)) * 0.5)
        .astype(np.float32)
        .astype(jnp.bfloat16),
        rep,
    )
    w0, i0 = j_route(x0, gate_w, gate_b)
    with mesh:
        for L in (1, 2, 4, 8):
            try:
                jc = jax.jit(chain, out_shardings=rep, static_argnames=("cfg", "L"))
                ma = jc.lower(x0, w0, i0, KW, cfg=CONFIGS[1], L=L).compile().memory_analysis()
                log(
                    f"chain N={N0} L={L}: temp {ma.temp_size_in_bytes/1e9:.3f} GB  args {ma.argument_size_in_bytes/1e9:.3f} GB"
                )
            except Exception as e:
                log(f"chain N={N0} L={L}: FAILED {str(e)[:200]}")
    log("DONE")
    sys.exit(0)


if os.environ.get("PADTEST"):
    cfg = CONFIGS[4]  # bt=128, bf=1152, btc=128
    with mesh:
        for N_, n_real in ((256, 3), (256, 64), (1024, 16), (1024, 256), (4096, 64), (4096, 1024)):
            x = jax.device_put(
                (np.random.default_rng(N_).standard_normal((N_, D)) * 0.5)
                .astype(np.float32)
                .astype(jnp.bfloat16),
                rep,
            )
            w0, i0 = j_route(x, gate_w, gate_b)
            w0 = np.asarray(jax.jit(lambda a_: a_, out_shardings=rep)(w0))
            i0 = np.asarray(jax.jit(lambda a_: a_, out_shardings=rep)(i0))
            variants = {}
            wm, idm = w0.copy(), i0.copy()
            wm[n_real:] = 0
            idm[n_real:] = -1
            variants["id=-1"] = (wm, idm)
            wd, idd = w0.copy(), i0.copy()
            wd[n_real:] = 0
            idd[n_real:] = i0[0]
            variants["dup-first"] = (wd, idd)
            for name, (wv, iv) in variants.items():
                log(f"N={N_} n_real={n_real} pad={name}: running ...")
                try:
                    wv_ = jax.device_put(wv.astype(np.float32), rep)
                    iv_ = jax.device_put(iv.astype(np.int32), rep)
                    out = j_kernel(x, wv_, iv_, KW, cfg=cfg)
                    jax.block_until_ready(out)
                    t = timeit(lambda *a_: j_kernel(*a_, cfg=cfg), x, wv_, iv_, KW)
                    log(
                        f"N={N_} n_real={n_real} pad={name}: OK {t:.3f} ms/layer, finite={bool(np.isfinite(np.asarray(jax.jit(lambda a_: a_[:n_real], out_shardings=rep)(out), np.float32)).all())}"
                    )
                except Exception as e:
                    log(
                        f"N={N_} n_real={n_real} pad={name}: FAILED {type(e).__name__}: {str(e)[:200]}"
                    )
    log("DONE")
    sys.exit(0)

for N in [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "32,128,512,2048").split(",")]:
    x = jax.device_put(
        (np.random.default_rng(N).standard_normal((N, D)) * 0.5)
        .astype(np.float32)
        .astype(jnp.bfloat16),
        rep,
    )
    wgt, idx = j_route(x, gate_w, gate_b)
    with mesh:
        ref = j_dense(x, P_ours)
        shared_out = j_shared(
            x, shared
        )  # our moe() adds the shared expert; add it to the kernel's routed output
        for cfg0 in CONFIGS[:2]:
            try:
                ma = j_kernel.lower(x, wgt, idx, KW, cfg=cfg0).compile().memory_analysis()
                log(
                    f"N={N:5d} cfg bt={cfg0.bt} bf={cfg0.bf}: compile-time temp {ma.temp_size_in_bytes/1e9:.3f} GB, out {ma.output_size_in_bytes/1e9:.3f} GB, args {ma.argument_size_in_bytes/1e9:.3f} GB"
                )
            except Exception as e:
                log(f"N={N:5d} memory_analysis failed: {str(e)[:150]}")
        best = None
        for cfg in CONFIGS[:1] if not INT8 else []:
            try:
                ker = j_kernel(x, wgt, idx, KW, cfg=cfg) + shared_out
                dm, rm = j_cmp(ker, ref)
                tk = timeit(lambda *a_: j_kernel(*a_, cfg=cfg), x, wgt, idx, KW)
                log(
                    f"N={N:5d} cfg bt={cfg.bt} bf={cfg.bf} btc={cfg.btc}: kernel {tk:.3f} ms | vs dense max|d|={float(dm):.3e} rel={float(dm)/(float(rm)+1e-9):.2e}"
                )
                if best is None or tk < best[0]:
                    best = (tk, cfg)
            except Exception as e:
                log(
                    f"N={N:5d} cfg bt={cfg.bt} bf={cfg.bf} btc={cfg.btc}: FAILED {type(e).__name__}: {str(e)[:160]}"
                )
        if not INT8:
            log(
                f"N={N:5d} BEST kernel {best[0]:.3f} ms ({best[1]})"
                if best
                else f"N={N:5d} no working config"
            )
        td = timeit(j_dense, x, P_ours)
        os.environ["V41_SPARSE_MAX_TOKENS"] = "1000000"
        tl = (
            timeit(jax.jit(lambda x_, Pm: ours(x_, Pm, "loop"), out_shardings=rep), x, P_ours)
            if N <= (4096 if INT8 else 64)
            else float("nan")
        )
        log(
            f"N={N:5d} [{'int8' if INT8 else 'fp8'} weights] dense path {td:.3f} ms | loop path {tl:.3f} ms   (one layer, routed+shared)"
        )
log("DONE")
