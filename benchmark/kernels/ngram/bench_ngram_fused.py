"""Compare the default PLE path with the explicit Pallas branch on TPU.

Each command runs BOTH baseline and fused, with the same donated BF16 pool.
Projections are included; host hash/lookup/H2D and transformer layers are not.

    # Released Qwen4Exp PLE, decode B=256, 1024 state slots, baseline vs fused.
    python benchmark/kernels/ngram/bench_ngram_fused.py --mode decode --tokens 256 --batch 256 --slots 1024
    # Released Qwen4Exp PLE, prefill T=8192/B=4, baseline vs fused.
    python benchmark/kernels/ngram/bench_ngram_fused.py --mode extend --tokens 8192 --batch 4 --slots 1024
"""

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np
from bench_ngram_device import DECODE_ARGS, EXTEND_ARGS, build, inputs, timeit
from jax.sharding import AxisType, Mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("decode", "extend"), default="decode")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--hlo", action="store_true", help="print pool-sized copies/transposes")
    args = parser.parse_args()
    if args.mode == "decode" and args.tokens != args.batch:
        parser.error("decode requires tokens == batch")
    if args.slots <= args.batch:
        parser.error("reserve slot zero: slots must exceed batch")
    mesh = Mesh(
        np.array(jax.devices())[:1].reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    layer = build(mesh, np.random.default_rng(0))
    data = inputs(mesh, np.random.default_rng(1), args.tokens, args.batch, args.slots)
    data["state_indices"] = data["state_indices"] + 1
    method = layer.forward_decode if args.mode == "decode" else layer.forward_extend
    names = DECODE_ARGS if args.mode == "decode" else EXTEND_ARGS
    argv = [data[name] for name in names]
    # Validate both outputs before timing; no donation during this comparison.
    reference = jax.jit(method)(*argv)
    candidate = jax.jit(functools.partial(method, use_pallas=True))(*argv)
    for label, actual, expected in zip(("delta", "state"), candidate, reference, strict=True):
        actual, expected = actual.astype(jnp.float32), expected.astype(jnp.float32)
        err = jnp.max(jnp.abs(actual - expected))
        rms = jnp.sqrt(jnp.mean((actual - expected) ** 2))
        print(f"{label}: max_abs={float(err):.6g} rms={float(rms):.6g}", flush=True)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0.025, rtol=0.025)
    print(
        f"{jax.devices()[0].device_kind} jax={jax.__version__} {args.mode} T={args.tokens} B={args.batch} slots={args.slots}",
        flush=True,
    )
    timings = {}
    for label, use_pallas in (("baseline", False), ("fused", True)):
        fn = jax.jit(functools.partial(method, use_pallas=use_pallas), donate_argnums=(2,))
        timings[label] = timeit(fn, argv, donated=2, reps=args.reps)
        compiled = fn.lower(*argv).compile().as_text()
        pallas = [
            line.strip()
            for line in compiled.splitlines()
            if "custom-call(" in line and "ngram_norm_gate_conv" in line
        ]
        print(
            f"{label}: median_ms={timings[label]:.3f} PLE_pallas_calls={len(pallas)}",
            flush=True,
        )
        if args.hlo:
            pool_shapes = (f"[{args.slots},10240,9]", f"[{args.slots},9,10240]")
            for line in compiled.splitlines():
                if any(shape in line for shape in pool_shapes) and any(
                    op in line for op in ("copy(", "transpose(", "custom-call(")
                ):
                    print(line.strip().split(", backend_config=")[0], flush=True)
    print(f"speedup={timings['baseline'] / timings['fused']:.3f}x", flush=True)


if __name__ == "__main__":
    main()
