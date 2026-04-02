"""
Benchmark quantized linear kernels vs BF16 baseline.

Compares various quantization configurations (FP8/INT8, per-channel/blockwise,
with/without activation quantization) against BF16 baseline for linear layers.
Supports tensor parallelism (column/row split) benchmarking.

Default shape from MiMo-V2-Flash: hidden_size=4096, intermediate_size=16384.

Usage:
    # Default: MiMo-V2-Flash shape, FP8 per-channel, no act quant, tp=1
    python -m benchmark.quantization.bench_quantized_linear

    # Compare FP8 vs INT8, per-channel vs block-128
    python -m benchmark.quantization.bench_quantized_linear \
        --weight-dtype fp8 int8 --block-size 0 128

    # Test TP column/row split
    python -m benchmark.quantization.bench_quantized_linear \
        --tp-size 1 4 --parallel-mode column row

    # Custom shape
    python -m benchmark.quantization.bench_quantized_linear \
        --tokens 1 512 2048 --hidden-size 8192 --intermediate-size 28672
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
import string
import traceback
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.utils.perf import _load_trace
from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

DTYPE_MAP = {
    "fp8": jnp.float8_e4m3fn,
    "int8": jnp.int8,
}

ACT_DTYPE_MAP = {
    "none": None,
    "fp8": jnp.float8_e4m3fn,
    "int8": jnp.int8,
}


@dataclass(frozen=True)
class QuantConfig:
    name: str
    weight_dtype: jnp.dtype | None  # None = BF16 baseline
    activation_dtype: jnp.dtype | None
    weight_block_size: tuple[int, int] | None


@dataclass(frozen=True)
class ShapeConfig:
    tokens: int
    hidden_size: int
    intermediate_size: int


@dataclass(frozen=True)
class TPConfig:
    tp_size: int
    parallel_mode: str  # "column" or "row"

    @property
    def kernel_axes(self) -> tuple[str | None, str | None]:
        if self.tp_size <= 1:
            return (None, None)
        if self.parallel_mode == "column":
            return (None, "tensor")
        return ("tensor", None)


def _make_quant_config(
    weight_dtype_str: str,
    act_dtype_str: str,
    block_size: int,
) -> QuantConfig:
    w_dtype = DTYPE_MAP[weight_dtype_str]
    a_dtype = ACT_DTYPE_MAP[act_dtype_str]
    block = (block_size, block_size) if block_size > 0 else None

    parts = [weight_dtype_str]
    if block:
        parts.append(f"block{block_size}")
    else:
        parts.append("per_channel")
    if a_dtype is not None:
        parts.append(f"w8a8_{act_dtype_str}")
    else:
        parts.append("w8")
    name = "_".join(parts)

    return QuantConfig(
        name=name,
        weight_dtype=w_dtype,
        activation_dtype=a_dtype,
        weight_block_size=block,
    )


BF16_BASELINE = QuantConfig("bf16_baseline", None, None, None)


def _extract_kernel_durations_ms(trace: dict, scope_name: str) -> list[float]:
    """Extract kernel-level device durations by matching scope_name in tf_op field."""
    durations: list[float] = []
    for e in trace.get("traceEvents", []):
        args = e.get("args", {})
        tf_op = args.get("tf_op", "")
        if scope_name in tf_op and "device_duration_ps" in args:
            durations.append(float(args["device_duration_ps"]) / 1e9)
    return durations


def build_mesh(tp_size: int) -> Mesh:
    devices = jax.devices()[:tp_size]
    if len(devices) < tp_size:
        raise RuntimeError(
            f"Requested tp_size={tp_size} but only {len(jax.devices())} devices available"
        )
    return create_device_mesh(
        ici_parallelism=[1, tp_size],
        dcn_parallelism=[1, 1],
        devices=devices,
        mesh_axes=("data", "tensor"),
    )


def benchmark_single(
    shape: ShapeConfig,
    quant: QuantConfig,
    tp: TPConfig,
    warmup: int,
    tries: int,
) -> dict:
    mesh = build_mesh(tp.tp_size)
    kernel_axes = tp.kernel_axes

    is_baseline = quant.weight_dtype is None

    with jax.set_mesh(mesh):
        linear = LinearBase(
            input_size=shape.hidden_size,
            output_size=shape.intermediate_size,
            mesh=mesh,
            use_bias=False,
            kernel_axes=kernel_axes,
            params_dtype=jnp.bfloat16,
        )

        if is_baseline:
            layer = linear
        else:
            layer = QuantizedLinear.from_linear(
                linear,
                weight_dtype=quant.weight_dtype,
                activation_dtype=quant.activation_dtype,
                weight_block_size=quant.weight_block_size,
            )

        layer_def, layer_state = nnx.split(layer)
        state_leaves, state_treedef = jax.tree_util.tree_flatten(layer_state)

        # Input sharding: for row-parallel, input is sharded on last dim
        if tp.parallel_mode == "row" and tp.tp_size > 1:
            x_pspec = P(None, "tensor")
        else:
            x_pspec = P(None, None)
        x_sharding = NamedSharding(mesh, x_pspec)

        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (shape.tokens, shape.hidden_size),
            dtype=jnp.bfloat16,
            out_sharding=x_sharding,
        )

        @partial(jax.jit, static_argnames=("state_treedef",))
        def forward(x, *, state_treedef, state_leaves):
            state = jax.tree_util.tree_unflatten(state_treedef, state_leaves)
            layer = nnx.merge(layer_def, state)
            out, _ = layer(x)
            return out

        # Use the layer's scope_name to match kernel-level time from trace
        # BF16 LinearBase -> "linear_base", FP8 QuantizedLinear -> "quantized_linear_base"
        scope_name = "linear_base" if is_baseline else "quantized_linear_base"

        def compute():
            return forward(x, state_treedef=state_treedef, state_leaves=state_leaves)

        # Manual warmup
        for _ in range(warmup):
            out = compute()
            jax.block_until_ready(out)

        # Profile without MARKER scope — extract kernel time via tf_op matching
        trace_tag = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        trace_dir = f"/tmp/quant_linear_trace/{scope_name}_{trace_tag}"
        os.makedirs(trace_dir, exist_ok=True)

        with jax.profiler.trace(trace_dir):
            for _ in range(tries):
                out = compute()
                jax.block_until_ready(out)

        trace = _load_trace(trace_dir)
        times = _extract_kernel_durations_ms(trace, scope_name)

    if len(times) > 1:
        times = times[1:]
    mean_ms = float(np.mean(times)) if times else float("nan")

    flops = 2.0 * shape.tokens * shape.hidden_size * shape.intermediate_size
    tflops = flops / (mean_ms / 1000.0) / 1e12 if mean_ms > 0 else 0.0

    return {
        "config": quant.name,
        "tokens": shape.tokens,
        "hidden": shape.hidden_size,
        "inter": shape.intermediate_size,
        "tp_size": tp.tp_size,
        "parallel_mode": tp.parallel_mode,
        "latency_ms": mean_ms,
        "tflops": tflops,
        "samples": times,
    }


def run_benchmark(
    shapes: list[ShapeConfig],
    quant_configs: list[QuantConfig],
    tp_configs: list[TPConfig],
    warmup: int,
    tries: int,
) -> None:
    # Always include BF16 baseline at the front
    all_quants = [BF16_BASELINE] + [q for q in quant_configs if q.weight_dtype is not None]

    print("QUANTIZED LINEAR BENCHMARK")
    print("=" * 80)
    print(f"Configs: {[q.name for q in all_quants]}")
    print(f"Shapes: {[(s.tokens, s.hidden_size, s.intermediate_size) for s in shapes]}")
    print(f"TP: {[(t.tp_size, t.parallel_mode) for t in tp_configs]}")
    print(f"Warmup: {warmup}, Tries: {tries}")
    print()

    for shape, tp in itertools.product(shapes, tp_configs):
        print(
            f"Shape: [{shape.tokens}, {shape.hidden_size}] x [{shape.hidden_size}, {shape.intermediate_size}]"
            f"  |  TP: {tp.parallel_mode} (tp={tp.tp_size})"
        )
        print("-" * 80)
        print(f"{'Config':<35} {'Latency(ms)':>12} {'TFLOPS':>8} {'vs BF16':>8}")
        print("-" * 80)

        baseline_ms = None
        for quant in all_quants:
            try:
                result = benchmark_single(shape, quant, tp, warmup, tries)
            except Exception as e:
                print(f"{quant.name:<35} {'ERROR':>12}   {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            ms = result["latency_ms"]
            tflops = result["tflops"]

            if quant.name == "bf16_baseline":
                baseline_ms = ms
                speedup_str = "1.00x"
            elif baseline_ms and baseline_ms > 0 and np.isfinite(ms):
                speedup_str = f"{baseline_ms / ms:.2f}x"
            else:
                speedup_str = "N/A"

            print(f"{quant.name:<35} {ms:>11.3f}  {tflops:>8.1f} {speedup_str:>8}")

        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark quantized linear kernels vs BF16 baseline."
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 128, 1024, 4096],
        help="Token counts (default: 1 128 1024 4096)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden size (default: 4096, MiMo-V2-Flash)",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=16384,
        help="Intermediate size (default: 16384, MiMo-V2-Flash)",
    )
    parser.add_argument(
        "--weight-dtype",
        type=str,
        nargs="+",
        default=["fp8"],
        choices=["fp8", "int8"],
        help="Weight quantization dtype (default: fp8)",
    )
    parser.add_argument(
        "--act-dtype",
        type=str,
        nargs="+",
        default=["none"],
        choices=["none", "fp8", "int8"],
        help="Activation quantization dtype (default: none)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs="+",
        default=[0],
        help="Block size for blockwise quant (0=per-channel, default: 0)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        nargs="+",
        default=[1],
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--parallel-mode",
        type=str,
        nargs="+",
        default=["column"],
        choices=["column", "row"],
        help="TP parallel mode (default: column)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--tries", type=int, default=10, help="Benchmark iterations (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    shapes = [
        ShapeConfig(
            tokens=t, hidden_size=args.hidden_size, intermediate_size=args.intermediate_size
        )
        for t in args.tokens
    ]

    # Build quant configs from cartesian product of CLI params
    quant_configs = []
    seen = set()
    for w_dtype, a_dtype, bs in itertools.product(
        args.weight_dtype, args.act_dtype, args.block_size
    ):
        cfg = _make_quant_config(w_dtype, a_dtype, bs)
        if cfg.name not in seen:
            seen.add(cfg.name)
            quant_configs.append(cfg)

    tp_configs = []
    seen_tp = set()
    for tp_size, mode in itertools.product(args.tp_size, args.parallel_mode):
        key = (tp_size, mode)
        if key not in seen_tp:
            seen_tp.add(key)
            tp_configs.append(TPConfig(tp_size=tp_size, parallel_mode=mode))

    run_benchmark(shapes, quant_configs, tp_configs, args.warmup, args.tries)
