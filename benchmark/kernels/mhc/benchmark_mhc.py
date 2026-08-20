#!/usr/bin/env python3
"""Fair operator-level mHC comparison against TPU-Inference on one TPU.

TPU-Inference and vLLM are optional external checkouts. Point
``TPU_INFERENCE_ROOT`` and ``VLLM_ROOT`` at their pinned revisions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

SGLANG_ROOT = Path(__file__).resolve().parents[3]
TPU_INFERENCE_ROOT = Path(
    os.environ.get("TPU_INFERENCE_ROOT", SGLANG_ROOT.parent / "tpu-inference")
)
VLLM_ROOT = Path(os.environ.get("VLLM_ROOT", SGLANG_ROOT.parent / "vllm"))

os.environ.setdefault("VLLM_TARGET_DEVICE", "tpu")
for source_root in (VLLM_ROOT, TPU_INFERENCE_ROOT, SGLANG_ROOT / "python"):
    sys.path.insert(0, str(source_root))

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import sgl_jax.srt.kernels.mhc.mhc as sglang_source
import torchax
import tpu_inference.layers.vllm.custom_ops.mhc as tpu_source
import vllm.model_executor.layers.mhc as vllm_source
from sgl_jax.srt.kernels.mhc import (
    mhc_head_collapse_fused,
    mhc_post_fused,
    mhc_pre_fused,
)
from torchax.interop import jax_view, torch_view
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CompilationConfig
from vllm.model_executor.layers.mhc import HCHeadOp, MHCPostOp, MHCPreOp
from vllm.platforms import current_platform

HC = 4
HIDDEN = 4096
DOT_PRECISION = jax.lax.Precision.DEFAULT
DOT_PRECISION_NAME = "DEFAULT"
ROWS = (2 + HC) * HC
RTOL = 2e-2
ATOL = 1e-2
ROTATING_BUFFERS = 4
WARMUPS = 3
TIMED_RUNS = 10
TIMED_RUNS_PER_BLOCK = TIMED_RUNS // 2
MEASUREMENT_ORDER = (0, 1, 1, 0)
INPUT_SEED = 20260825


def _inside(module, root: Path) -> bool:
    return Path(module.__file__).resolve().is_relative_to(root.resolve())


if not _inside(sglang_source, SGLANG_ROOT):
    raise RuntimeError(
        f"loaded SGLang source outside snapshot: {sglang_source.__file__}"
    )
if not _inside(tpu_source, TPU_INFERENCE_ROOT):
    raise RuntimeError(f"loaded TPU-inference outside snapshot: {tpu_source.__file__}")
if not _inside(vllm_source, VLLM_ROOT):
    raise RuntimeError(f"loaded vLLM outside snapshot: {vllm_source.__file__}")
if not current_platform.is_tpu():
    raise RuntimeError("TPU-inference did not select its TPU platform")


class _DispatchConfig:
    compilation_config = CompilationConfig()


with set_current_vllm_config(_DispatchConfig()):
    _TPU_PRE = MHCPreOp()
    _TPU_POST = MHCPostOp()
    _TPU_HEAD = HCHeadOp()


def _check_dispatch(op, method: str) -> None:
    if type(op).__module__ != "tpu_inference.layers.vllm.custom_ops.mhc":
        raise RuntimeError(f"unexpected TPU implementation: {type(op)}")
    if op._forward_method.__qualname__ != method:
        raise RuntimeError(
            f"unexpected TPU dispatch: {op._forward_method.__qualname__}"
        )


_check_dispatch(_TPU_PRE, "VllmMHCPreOp.forward_tpu")
_check_dispatch(_TPU_POST, "VllmMHCPostOp.forward_tpu")
_check_dispatch(_TPU_HEAD, "VllmHCHeadOp.forward_tpu")


@partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "post_multiplier",
        "sinkhorn_iterations",
    ),
)
def sglang_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    *,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    post_multiplier=2.0,
    sinkhorn_iterations=20,
):
    if hc_pre_eps != hc_sinkhorn_eps or post_multiplier != 2.0:
        raise ValueError(
            "the common ABI requires equal eps values and post_multiplier=2"
        )
    layer_input, post_mix, comb_mix = mhc_pre_fused(
        residual,
        fn,
        hc_scale,
        hc_base,
        hc_mult=HC,
        sinkhorn_iters=sinkhorn_iterations,
        norm_eps=rms_eps,
        hc_eps=hc_pre_eps,
        dot_precision=DOT_PRECISION,
    )
    return layer_input, post_mix[..., None], comb_mix


@partial(
    jax.jit,
    static_argnames=(
        "rms_eps",
        "hc_pre_eps",
        "hc_sinkhorn_eps",
        "post_multiplier",
        "sinkhorn_iterations",
    ),
)
def tpu_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    *,
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    post_multiplier=2.0,
    sinkhorn_iterations=20,
):
    post_mix, comb_mix, layer_input = _TPU_PRE(
        torch_view(residual),
        torch_view(fn),
        torch_view(hc_scale),
        torch_view(hc_base),
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        post_multiplier,
        sinkhorn_iterations,
    )
    return jax_view(layer_input), jax_view(post_mix), jax_view(comb_mix)


@jax.jit
def sglang_post(x, residual, post_mix, comb_mix):
    return mhc_post_fused(
        x,
        residual,
        post_mix[..., 0],
        comb_mix,
        precision=DOT_PRECISION,
    )


@jax.jit
def tpu_post(x, residual, post_mix, comb_mix):
    return jax_view(
        _TPU_POST(
            torch_view(x),
            torch_view(residual),
            torch_view(post_mix),
            torch_view(comb_mix),
        )
    )


@partial(jax.jit, static_argnames=("rms_eps", "hc_eps"))
def sglang_head(
    residual,
    fn,
    hc_scale,
    hc_base,
    *,
    rms_eps=1e-6,
    hc_eps=1e-6,
):
    return mhc_head_collapse_fused(
        residual,
        fn,
        hc_scale,
        hc_base,
        hc_mult=HC,
        norm_eps=rms_eps,
        hc_eps=hc_eps,
        dot_precision=DOT_PRECISION,
    )


@partial(jax.jit, static_argnames=("rms_eps", "hc_eps"))
def tpu_head(
    residual,
    fn,
    hc_scale,
    hc_base,
    *,
    rms_eps=1e-6,
    hc_eps=1e-6,
):
    return jax_view(
        _TPU_HEAD(
            torch_view(residual),
            torch_view(fn),
            torch_view(hc_scale),
            torch_view(hc_base),
            rms_eps,
            hc_eps,
        )
    )


def _inputs(tokens: int, seed: int) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    comb = np.exp(rng.standard_normal((tokens, HC, HC), dtype=np.float32) * 0.1)
    comb /= comb.sum(axis=-1, keepdims=True)
    data = {
        "residual": jnp.asarray(
            rng.standard_normal((tokens, HC, HIDDEN), dtype=np.float32) * 0.1
        ).astype(jnp.bfloat16),
        "fn": jnp.asarray(
            rng.standard_normal((ROWS, HC * HIDDEN), dtype=np.float32) * 0.01
        ),
        "scale": jnp.asarray([0.7, 1.1, 0.9], jnp.float32),
        "base": jnp.asarray(rng.standard_normal(ROWS, dtype=np.float32) * 0.05),
        "x": jnp.asarray(
            rng.standard_normal((tokens, HIDDEN), dtype=np.float32) * 0.1
        ).astype(jnp.bfloat16),
        "post": jnp.asarray(
            2.0
            / (
                1.0
                + np.exp(-rng.standard_normal((tokens, HC, 1), dtype=np.float32) * 0.1)
            )
        ),
        "comb": jnp.asarray(comb),
        "head_fn": jnp.asarray(
            rng.standard_normal((HC, HC * HIDDEN), dtype=np.float32) * 0.01
        ),
        "head_scale": jnp.asarray([0.8], jnp.float32),
        "head_base": jnp.asarray(rng.standard_normal(HC, dtype=np.float32) * 0.05),
    }
    jax.block_until_ready(tuple(data.values()))
    return data


def _args(op: str, data: dict[str, jax.Array]):
    if op == "pre":
        return data["residual"], data["fn"], data["scale"], data["base"]
    if op == "post":
        return data["x"], data["residual"], data["post"], data["comb"]
    if op == "head":
        return (
            data["residual"],
            data["head_fn"],
            data["head_scale"],
            data["head_base"],
        )
    raise ValueError(op)


def _correctness(left, right) -> dict[str, float]:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if left_tree != right_tree:
        raise AssertionError(f"output structure differs: {left_tree} != {right_tree}")
    max_abs = 0.0
    max_rel = 0.0
    for lhs, rhs in zip(left_leaves, right_leaves):
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            raise AssertionError(
                f"output ABI differs: {lhs.shape}/{lhs.dtype} != {rhs.shape}/{rhs.dtype}"
            )
        lhs_np = np.asarray(lhs, np.float32)
        rhs_np = np.asarray(rhs, np.float32)
        diff = np.abs(lhs_np - rhs_np)
        max_abs = max(max_abs, float(diff.max(initial=0.0)))
        relative = diff / np.maximum(np.abs(rhs_np), ATOL)
        max_rel = max(max_rel, float(relative.max(initial=0.0)))
        np.testing.assert_allclose(lhs_np, rhs_np, rtol=RTOL, atol=ATOL)
    return {"max_abs": max_abs, "max_rel": max_rel, "rtol": RTOL, "atol": ATOL}


def _invoke(function, op: str, data: dict[str, jax.Array]):
    result = function(*_args(op, data))
    jax.block_until_ready(result)
    return result


def _profile_block(function, op: str, buffers, position: int):
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    with tempfile.TemporaryDirectory() as directory:
        with jax.profiler.trace(directory, profiler_options=options):
            for index in range(TIMED_RUNS_PER_BLOCK):
                buffer = buffers[(position + index) % ROTATING_BUFFERS]
                _invoke(function, op, buffer)
        profile_paths = list(Path(directory).glob("plugins/profile/**/*.xplane.pb"))
        if len(profile_paths) != 1:
            raise RuntimeError(f"expected one TPU profile, found {len(profile_paths)}")
        profile = jax.profiler.ProfileData.from_file(str(profile_paths[0]))

    device = profile.find_plane_with_name("/device:TPU:0")
    if device is None:
        raise RuntimeError("TPU profile contained no /device:TPU:0 plane")
    intervals = []
    event_count = 0
    for line in device.lines:
        if line.name != "XLA Modules":
            continue
        for event in line.events:
            intervals.append((event.start_ns, event.end_ns))
            event_count += 1
    if not intervals:
        raise RuntimeError("TPU profile contained no XLA module events")

    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    if len(merged) != TIMED_RUNS_PER_BLOCK:
        raise RuntimeError(
            f"expected {TIMED_RUNS_PER_BLOCK} device calls, found {len(merged)}"
        )
    return {
        "active_ms": sum(end - begin for begin, end in merged) / 1e6,
        "module_events": event_count,
        "merged_intervals": len(merged),
    }


def _profile_pair(left, right, op: str, buffers):
    functions = (left, right)
    for function in functions:
        _invoke(function, op, buffers[0])
        for index in range(WARMUPS):
            _invoke(function, op, buffers[index % ROTATING_BUFFERS])

    totals = [
        {"active_ms": 0.0, "module_events": 0, "merged_intervals": 0},
        {"active_ms": 0.0, "module_events": 0, "merged_intervals": 0},
    ]
    positions = [0, 0]
    for backend in MEASUREMENT_ORDER:
        function = functions[backend]
        position = positions[backend]
        _invoke(function, op, buffers[position % ROTATING_BUFFERS])
        position += 1
        block = _profile_block(function, op, buffers, position)
        positions[backend] = position + TIMED_RUNS_PER_BLOCK
        for key, value in block.items():
            totals[backend][key] += value

    for total in totals:
        if total["merged_intervals"] != TIMED_RUNS:
            raise RuntimeError(
                f"expected {TIMED_RUNS} total device calls, "
                f"found {total['merged_intervals']}"
            )
        total["mean_ms"] = total.pop("active_ms") / TIMED_RUNS
    return totals


def _revision(path: Path, override_env: str | None = None) -> str:
    if override_env and (revision := os.environ.get(override_env)):
        return revision
    return subprocess.check_output(
        ("git", "-C", str(path), "rev-parse", "HEAD"), text=True
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance():
    files = {
        "sglang_mhc": SGLANG_ROOT / "python/sgl_jax/srt/kernels/mhc/mhc.py",
        "sglang_tune": SGLANG_ROOT / "python/sgl_jax/srt/kernels/mhc/tune.py",
        "tpu_dispatch": TPU_INFERENCE_ROOT
        / "tpu_inference/layers/vllm/custom_ops/mhc.py",
        "vllm_math": VLLM_ROOT / "vllm/model_executor/kernels/mhc/torch.py",
        "benchmark": Path(__file__).resolve(),
    }
    return {
        "revisions": {
            "sglang-jax": _revision(SGLANG_ROOT, "SGLANG_JAX_REVISION"),
            "tpu-inference": _revision(TPU_INFERENCE_ROOT),
            "vllm": _revision(VLLM_ROOT),
        },
        "sha256": {name: _sha256(path) for name, path in files.items()},
        "loaded_sources": {
            "sglang-jax": str(Path(sglang_source.__file__).resolve()),
            "tpu-inference": str(Path(tpu_source.__file__).resolve()),
            "vllm": str(Path(vllm_source.__file__).resolve()),
        },
    }


def _markdown(payload) -> str:
    lines = [
        "# mHC fair operator comparison",
        "",
        (
            f"Configuration: hidden_size={payload['configuration']['hidden_size']}, "
            f"dot_precision={payload['configuration']['dot_precision']}."
        ),
        "",
        (
            "The same dot_precision is applied to both implementations for pre, "
            "post, and head."
        ),
        "",
        (
            "Four independent input buffers rotate through every measurement. Both "
            "implementations receive the same JAX array objects. Results use three "
            "warmups and two five-call blocks per implementation in ABBA order. Device "
            "time is the TPU XLA-module span; speedup is TPU-inference / SGLang-JAX."
        ),
        "",
        "| N | op | SGL device ms | TPU device ms | speedup | max abs | max rel |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        for op, item in result["operators"].items():
            lines.append(
                f"| {result['tokens']} | {op} | "
                f"{item['sglang']['device']['mean_ms']:.4f} | "
                f"{item['tpu_inference']['device']['mean_ms']:.4f} | "
                f"{item['speedup']:.3f}x | "
                f"{item['correctness']['max_abs']:.3e} | "
                f"{item['correctness']['max_rel']:.3e} |"
            )
    return "\n".join(lines) + "\n"


def main():
    global DOT_PRECISION, DOT_PRECISION_NAME, HIDDEN

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=(1, 128, 256, 512, 1024, 2048, 4096, 8192),
    )
    parser.add_argument("--hidden-size", type=int, choices=(4096, 7168), default=4096)
    parser.add_argument(
        "--dot-precision", choices=("default", "highest"), default="default"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("MHC_BENCHMARK_OUTPUT", "/tmp/mhc-results")),
    )
    args = parser.parse_args()

    HIDDEN = args.hidden_size
    DOT_PRECISION_NAME = args.dot_precision.upper()
    DOT_PRECISION = getattr(jax.lax.Precision, DOT_PRECISION_NAME)
    jax.config.update("jax_default_matmul_precision", args.dot_precision)

    if jax.default_backend() != "tpu":
        raise SystemExit("a physical TPU is required")
    devices = jax.devices()
    if len(devices) != 1:
        raise SystemExit(
            f"this benchmark requires one TPU device, found {len(devices)}"
        )
    if any(tokens <= 0 for tokens in args.tokens):
        raise SystemExit("token counts must be positive")
    if inspect.signature(sglang_pre) != inspect.signature(tpu_pre):
        raise RuntimeError("pre ABI mismatch")
    if inspect.signature(sglang_post) != inspect.signature(tpu_post):
        raise RuntimeError("post ABI mismatch")
    if inspect.signature(sglang_head) != inspect.signature(tpu_head):
        raise RuntimeError("head ABI mismatch")

    implementations = {
        "pre": (sglang_pre, tpu_pre),
        "post": (sglang_post, tpu_post),
        "head": (sglang_head, tpu_head),
    }
    results = []
    with torchax.default_env():
        for tokens in args.tokens:
            print(
                f"N={tokens}: allocating {ROTATING_BUFFERS} rotating buffers",
                flush=True,
            )
            buffers = [
                _inputs(tokens, seed=INPUT_SEED + index)
                for index in range(ROTATING_BUFFERS)
            ]
            operators = {}
            for op, (sglang, tpu) in implementations.items():
                print(f"  {op}: correctness + timing", flush=True)
                sg_out = _invoke(sglang, op, buffers[0])
                tpu_out = _invoke(tpu, op, buffers[0])
                correctness = _correctness(sg_out, tpu_out)
                sg_device, tpu_device = _profile_pair(sglang, tpu, op, buffers)
                operators[op] = {
                    "sglang": {"device": sg_device},
                    "tpu_inference": {"device": tpu_device},
                    "speedup": tpu_device["mean_ms"] / sg_device["mean_ms"],
                    "correctness": correctness,
                }
            results.append({"tokens": tokens, "operators": operators})
            del buffers
            gc.collect()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(devices[0]),
        "device_kind": devices[0].device_kind,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "configuration": {
            "hidden_size": HIDDEN,
            "dot_precision": DOT_PRECISION_NAME,
        },
        "protocol": {
            "rotating_buffers": ROTATING_BUFFERS,
            "warmups": WARMUPS,
            "timed_runs": TIMED_RUNS,
            "statistic": "arithmetic mean",
            "order": "ABBA blocks with an excluded settle call before each block",
            "boundary": "TPU XLA-module span from device inputs through completed device outputs",
            "shared_input_objects": True,
        },
        "provenance": _provenance(),
        "results": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output.resolve() / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "summary.md").write_text(_markdown(payload))
    print("\n" + (output / "summary.md").read_text())
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
