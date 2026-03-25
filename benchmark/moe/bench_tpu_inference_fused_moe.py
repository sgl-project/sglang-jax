"""
Benchmark the tpu-inference fused_moe kernel (blogs/tpu-inference/fused_moe/v1/kernel.py).

Unlike the sglang-jax kernel benchmarked by bench_fused_moe.py, the tpu-inference
kernel fuses top-k routing inside the Pallas kernel, takes raw gating_output logits,
and expects w1 in a fused [gate, up] layout: (num_experts, 2, hidden_size, intermediate_size).

Usage (timing mode):
    python -m benchmark.moe.bench_tpu_inference_fused_moe \\
        --num-experts 256 --top-k 8 --hidden-size 8192 --intermediate-size 2048 \\
        --scoring-fn sigmoid --num-tokens 128 256 --iters 3

Usage (profiling mode):
    LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true \\
        --xla_xprof_register_llo_debug_info=true" \\
    python -m benchmark.moe.bench_tpu_inference_fused_moe \\
        --profile --profile-dir ./profile_tpu_inf_moe \\
        --num-experts 256 --top-k 8 --hidden-size 8192 --intermediate-size 2048 \\
        --scoring-fn sigmoid --num-tokens 128 --iters 3
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Import shim: make the tpu-inference kernel importable.
#
# The kernel at blogs/tpu-inference/fused_moe/v1/kernel.py imports from
# tpu_inference.* which does not exist in this repo.  We register mock
# modules so the original import statements work unmodified.
# ---------------------------------------------------------------------------
import importlib.util
import logging
import os
import sys
import types

_V1_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "blogs", "tpu-inference", "fused_moe", "v1")
)


def _setup_tpu_inference_mocks():
    """Register mock tpu_inference.* modules so the blog kernel code can be imported."""
    # 1. tpu_inference.logger — provides init_logger() with warning_once support
    tpu_inference = types.ModuleType("tpu_inference")
    tpu_inference.__path__ = []
    tpu_inference_logger = types.ModuleType("tpu_inference.logger")

    def init_logger(name):
        log = logging.getLogger(name)
        _seen: set[str] = set()
        _orig_warning = log.warning

        def warning_once(msg, *a, **kw):
            if msg not in _seen:
                _seen.add(msg)
                _orig_warning(msg, *a, **kw)

        log.warning_once = warning_once  # type: ignore[attr-defined]
        return log

    tpu_inference_logger.init_logger = init_logger  # type: ignore[attr-defined]
    tpu_inference.logger = tpu_inference_logger  # type: ignore[attr-defined]
    sys.modules.setdefault("tpu_inference", tpu_inference)
    sys.modules.setdefault("tpu_inference.logger", tpu_inference_logger)

    # 2. Intermediate package chain for the tuned_block_sizes import
    for pkg in [
        "tpu_inference.kernels",
        "tpu_inference.kernels.fused_moe",
        "tpu_inference.kernels.fused_moe.v1",
    ]:
        m = types.ModuleType(pkg)
        m.__path__ = []  # type: ignore[attr-defined]
        sys.modules.setdefault(pkg, m)

    # 3. Load tuned_block_sizes.py from disk and register
    tbs_path = os.path.join(_V1_DIR, "tuned_block_sizes.py")
    spec = importlib.util.spec_from_file_location(
        "tpu_inference.kernels.fused_moe.v1.tuned_block_sizes", tbs_path
    )
    assert spec and spec.loader
    tbs_mod = importlib.util.module_from_spec(spec)
    sys.modules["tpu_inference.kernels.fused_moe.v1.tuned_block_sizes"] = tbs_mod
    spec.loader.exec_module(tbs_mod)


_setup_tpu_inference_mocks()

# 4. Load the kernel module itself
_kernel_spec = importlib.util.spec_from_file_location(
    "tpu_inference_fused_moe_kernel", os.path.join(_V1_DIR, "kernel.py")
)
assert _kernel_spec and _kernel_spec.loader
_kernel_mod = importlib.util.module_from_spec(_kernel_spec)
_kernel_spec.loader.exec_module(_kernel_mod)
fused_ep_moe = _kernel_mod.fused_ep_moe

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import argparse
import faulthandler
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from benchmark.moe.utils import (
    DEFAULT_NUM_TOKENS,
    MoEBenchmarkCase,
    MoEImbalanceSimulator,
    make_moe_cases,
    select_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace

EP_AXIS_NAME = "model"


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def build_tpu_inference_mesh(ep_size: int) -> jax.sharding.Mesh:
    """Build a 2D mesh for the tpu-inference kernel.

    The kernel hardcodes ``jax.lax.axis_index("data")`` internally, so the
    mesh must have axes ``("data", ep_axis_name)`` with ``data`` size 1.
    """
    devices = jax.devices()[:ep_size]
    return jax.sharding.Mesh(np.array(devices).reshape(1, ep_size), ("data", EP_AXIS_NAME))


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_inputs(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    weight_dtype: jnp.dtype,
    mesh: jax.sharding.Mesh,
):
    """Allocate and shard inputs for the tpu-inference kernel."""
    ep_sharding_2d = NamedSharding(mesh, P(EP_AXIS_NAME, None))
    tokens = jax.device_put(
        jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16), ep_sharding_2d
    )
    w1 = jax.device_put(
        jnp.zeros((num_experts, 2, hidden_size, intermediate_size), dtype=weight_dtype),
        NamedSharding(mesh, P(EP_AXIS_NAME, None, None, None)),
    )
    w2 = jax.device_put(
        jnp.zeros((num_experts, intermediate_size, hidden_size), dtype=weight_dtype),
        NamedSharding(mesh, P(EP_AXIS_NAME, None, None)),
    )
    gating_output = jax.device_put(
        jnp.zeros((num_tokens, num_experts), dtype=jnp.bfloat16), ep_sharding_2d
    )
    return tokens, w1, w2, gating_output


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_all(  # noqa: PLR0913
    *,
    iters: int,
    warmup_iters: int,
    weight_dtype: jnp.dtype,
    num_tokens_list: list[int],
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str,
    scoring_fn: str,
    renormalize_topk_logits: bool,
    imbalance_mode: str,
    alpha: float,
    zipf_s: float,
    hotspot_ratio: float,
    hotspot_count: int,
    zero_expert_count: int,
    non_hotspot_alpha: float,
    profile: bool,
    profile_dir: str,
    # Optional block size overrides (None = auto-tune)
    bt: int | None,
    bf: int | None,
    bd1: int | None,
    bd2: int | None,
    btc: int | None,
    bfc: int | None,
    bd1c: int | None,
    bd2c: int | None,
):
    cases = make_moe_cases(
        num_tokens=num_tokens_list,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        renormalize_topk_logits=renormalize_topk_logits,
    )
    cases = select_cases(cases)

    for case in cases:
        # The tpu-inference kernel requires tp_size == 1
        if case.tp_size != 1:
            print(
                f"skip [case={case.name}] tp_size={case.tp_size} != 1 "
                "(tpu-inference kernel requires all non-EP axes to have size 1)"
            )
            continue

        ep_size = case.ep_size
        mesh = build_tpu_inference_mesh(ep_size)
        mesh_ep = mesh.shape[EP_AXIS_NAME]

        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}, "
            f"ep_size={ep_size}"
        )

        tokens, w1, w2, gating_output = prepare_inputs(
            case.num_tokens, case.num_experts, case.hidden_size, case.intermediate_size,
            weight_dtype, mesh,
        )

        # Generate imbalance-controlled gating logits
        target_counts = MoEImbalanceSimulator.generate_counts(
            case.num_tokens,
            case.top_k,
            case.num_experts,
            mode=imbalance_mode,
            alpha=alpha,
            zipf_s=zipf_s,
            hotspot_ratio=hotspot_ratio,
            hotspot_count=hotspot_count,
            zero_expert_count=zero_expert_count,
            non_hotspot_alpha=non_hotspot_alpha,
        )
        custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
            case.num_tokens, case.num_experts, case.top_k, target_counts
        )
        gating_output = jax.device_put(
            custom_logits.astype(jnp.bfloat16),
            NamedSharding(mesh, P(EP_AXIS_NAME, None)),
        )
        print(f"  imbalance_mode={imbalance_mode}")

        task = "fused-moe-k_.*"

        def _compute(
            _tokens=tokens,
            _w1=w1,
            _w2=w2,
            _gating=gating_output,
        ):
            return fused_ep_moe(
                mesh=mesh,
                tokens=_tokens,
                w1=_w1,
                w2=_w2,
                gating_output=_gating,
                top_k=case.top_k,
                renormalize_topk_logits=case.renormalize_topk_logits,
                act_fn=case.activation,
                scoring_fn=scoring_fn,
                ep_axis_name=EP_AXIS_NAME,
                bt=bt,
                bf=bf,
                bd1=bd1,
                bd2=bd2,
                btc=btc,
                bfc=bfc,
                bd1c=bd1c,
                bd2c=bd2c,
            )

        try:
            if profile:
                profile_case_dir = os.path.join(
                    profile_dir,
                    f"case_{case.num_tokens}t_{case.num_experts}e_ep{mesh_ep}",
                )
                os.makedirs(profile_case_dir, exist_ok=True)
                print(f"  Profiling to: {profile_case_dir}")
                for _ in range(warmup_iters):
                    out = _compute()
                    jax.block_until_ready(out)
                with jax.profiler.trace(profile_case_dir):
                    for step in range(iters):
                        with jax.profiler.StepTraceAnnotation(task, step_num=step):
                            out = _compute()
                            jax.block_until_ready(out)
                print(f"  Profile saved to: {profile_case_dir}")
            else:
                times = multiple_iteration_timeit_from_trace(
                    compute_func=_compute,
                    data_generator=lambda: (),
                    task=task,
                    tries=iters,
                    warmup=warmup_iters,
                )
                if len(times) > 1:
                    times = times[1:]
                mean_ms = float(np.mean(times)) if times else float("nan")
                print(f"  tpu-inference fused_moe: {mean_ms:.3f} ms (trace) | samples={times}")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the tpu-inference fused MoE kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Benchmark control
    parser.add_argument("--iters", type=int, default=3, help="Number of benchmark iterations.")
    parser.add_argument("--warmup-iters", type=int, default=1, help="Warmup iterations before measurement.")

    # Model shape
    parser.add_argument("--num-tokens", type=int, nargs="+", default=None, help="Token counts to benchmark.")
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=2048)

    # Kernel options
    parser.add_argument("--weight-dtype", type=str, default="bfloat16", choices=["bfloat16", "float8_e4m3fn"])
    parser.add_argument("--activation", type=str, default="silu", choices=["silu", "gelu", "swigluoai"])
    parser.add_argument("--scoring-fn", type=str, default="sigmoid", choices=["softmax", "sigmoid"])
    parser.add_argument("--renormalize-topk-logits", action=argparse.BooleanOptionalAction, default=True)

    # Block size overrides (None = auto-tune)
    for param in ["bt", "bf", "bd1", "bd2", "btc", "bfc", "bd1c", "bd2c"]:
        parser.add_argument(f"--{param}", type=int, default=None, help=f"Override block size {param}.")

    # Imbalance simulation
    parser.add_argument("--imbalance-mode", type=str, default="balanced",
                        choices=["balanced", "dirichlet", "zipf", "hotspot", "sparse_hotspot"])
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration.")
    parser.add_argument("--zipf-s", type=float, default=1.1, help="Zipf exponent.")
    parser.add_argument("--hotspot-ratio", type=float, default=0.5, help="Fraction of tokens to hotspot experts.")
    parser.add_argument("--hotspot-count", type=int, default=1, help="Number of hotspot experts.")
    parser.add_argument("--zero-expert-count", type=int, default=0, help="Experts with zero load.")
    parser.add_argument("--non-hotspot-alpha", type=float, default=100.0, help="Dirichlet alpha for non-hotspot.")

    # Profiling
    parser.add_argument("--profile", action="store_true", help="Enable JAX profiling (dump trace).")
    parser.add_argument("--profile-dir", type=str, default="profile_tpu_inference_fused_moe",
                        help="Output directory for profiling traces.")

    # Misc
    parser.add_argument("--compilation-cache-dir", type=str, default=None,
                        help="JAX compilation cache directory.")

    return parser.parse_args()


def main():
    faulthandler.enable()
    args = parse_args()

    if args.compilation_cache_dir:
        from jax.experimental.compilation_cache import compilation_cache as _cc
        _cc.set_cache_dir(args.compilation_cache_dir)

    weight_dtype = {"bfloat16": jnp.bfloat16, "float8_e4m3fn": jnp.float8_e4m3fn}[args.weight_dtype]
    num_tokens_list = args.num_tokens or list(DEFAULT_NUM_TOKENS)

    print(f"Devices: {jax.device_count()} x {jax.devices()[0].platform}")
    print(
        f"Model: experts={args.num_experts}, top_k={args.top_k}, "
        f"hidden={args.hidden_size}, intermediate={args.intermediate_size}, "
        f"weight_dtype={args.weight_dtype}, scoring_fn={args.scoring_fn}"
    )

    run_all(
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        weight_dtype=weight_dtype,
        num_tokens_list=num_tokens_list,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        activation=args.activation,
        scoring_fn=args.scoring_fn,
        renormalize_topk_logits=args.renormalize_topk_logits,
        imbalance_mode=args.imbalance_mode,
        alpha=args.alpha,
        zipf_s=args.zipf_s,
        hotspot_ratio=args.hotspot_ratio,
        hotspot_count=args.hotspot_count,
        zero_expert_count=args.zero_expert_count,
        non_hotspot_alpha=args.non_hotspot_alpha,
        profile=args.profile,
        profile_dir=args.profile_dir,
        bt=args.bt,
        bf=args.bf,
        bd1=args.bd1,
        bd2=args.bd2,
        btc=args.btc,
        bfc=args.bfc,
        bd1c=args.bd1c,
        bd2c=args.bd2c,
    )


if __name__ == "__main__":
    main()
