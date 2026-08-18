"""Profile gate-fed JAX top-k against the biased top-k Pallas kernel on TPU.

The gate matmul and sigmoid stay outside the measured scopes so both routing
paths receive the layout produced by the real MiMo gate.
"""

import argparse
import functools
import glob
import gzip
import json
import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.biased_topk import biased_topk_pallas

TRACE_ROOT = os.environ.get("TOPK_TRACE_ROOT", "/tmp/tpu_logs/biased_topk_bench")
HIDDEN_SIZE = 6144
EXPERTS = 384
TOPK = 8
SCOPE_JAX = "BIASED_TOPK_JAX"
SCOPE_PALLAS = "BIASED_TOPK_PALLAS"


def reference_biased_topk(router_logits, correction_bias, *, topk):
    scores = router_logits.astype(jnp.float32) + correction_bias.astype(jnp.float32)
    ids = jax.lax.top_k(scores, topk)[1]
    weights = jnp.take_along_axis(router_logits.astype(jnp.float32), ids, axis=1)
    return weights, ids


def make_router_logits(gate_weight):
    def prepare(hidden_states):
        return jax.nn.sigmoid(
            jnp.dot(hidden_states, gate_weight, precision=jax.lax.Precision.HIGHEST)
        )

    return prepare


def make_jax_route(correction_bias):
    def route(router_logits):
        with jax.named_scope(SCOPE_JAX):
            return reference_biased_topk(
                router_logits,
                correction_bias,
                topk=TOPK,
            )

    return route


def make_pallas_route(correction_bias, block_tokens):
    def route(router_logits):
        with jax.named_scope(SCOPE_PALLAS):
            return biased_topk_pallas(
                router_logits,
                correction_bias,
                topk=TOPK,
                block_tokens=block_tokens,
                interpret=False,
            )

    return route


def _scope_device_us(run_fn, scope, tag, *, warmup=5, iters=20):
    for _ in range(warmup):
        jax.block_until_ready(run_fn())
    trace_root = os.path.join(
        TRACE_ROOT,
        f"{re.sub(r'[^A-Za-z0-9]', '_', tag)}_{os.getpid()}_{int(time.time() * 1000)}",
    )
    os.makedirs(trace_root, exist_ok=True)
    with jax.profiler.trace(trace_root):
        for step in range(iters):
            with jax.profiler.StepTraceAnnotation("biased_topk", step_num=step):
                jax.block_until_ready(run_fn())

    profile_dirs = glob.glob(os.path.join(trace_root, "plugins", "profile", "*"))
    if not profile_dirs:
        return float("nan"), trace_root
    latest = max(profile_dirs, key=os.path.getmtime)
    events = []
    for trace_file in sorted(glob.glob(os.path.join(latest, "*.trace.json.gz"))):
        with gzip.open(trace_file) as file:
            events.extend(json.load(file).get("traceEvents", []))

    process_names = {}
    thread_names = {}
    for event in events:
        if event.get("ph") != "M":
            continue
        args = event.get("args", {})
        if event["name"] == "process_name":
            process_names[event["pid"]] = args.get("name", "")
        elif event["name"] == "thread_name":
            thread_names[(event["pid"], event["tid"])] = args.get("name", "")

    module_count = 0
    scope_total_us = 0.0
    for event in events:
        if event.get("ph") != "X":
            continue
        if process_names.get(event["pid"]) != "/device:TPU:0":
            continue
        thread = thread_names.get((event["pid"], event["tid"]), "")
        if thread == "XLA Modules":
            module_count += 1
            continue
        if thread != "XLA Ops":
            continue
        blob = event["name"] + " " + json.dumps(event.get("args", {}))
        if scope not in blob:
            continue
        duration_ps = event.get("args", {}).get("device_duration_ps")
        if duration_ps is not None:
            scope_total_us += float(duration_ps) / 1e6
    if module_count == 0:
        return float("nan"), trace_root
    return scope_total_us / module_count, trace_root


def benchmark_one(tokens, block_tokens):
    gate_weight = jax.random.normal(
        jax.random.key(0),
        (HIDDEN_SIZE, EXPERTS),
        dtype=jnp.float32,
    )
    correction_bias = (
        jax.random.normal(
            jax.random.key(1),
            (EXPERTS,),
            dtype=jnp.float32,
        )
        * 0.1
    )
    hidden_states = jax.random.normal(
        jax.random.key(2),
        (tokens, HIDDEN_SIZE),
        dtype=jnp.bfloat16,
    )
    prepare_router_logits = jax.jit(make_router_logits(gate_weight))
    router_logits = prepare_router_logits(hidden_states)
    jax.block_until_ready(router_logits)
    jax_fn = jax.jit(make_jax_route(correction_bias))
    pallas_fn = jax.jit(make_pallas_route(correction_bias, block_tokens))

    expected_weights, expected_ids = jax_fn(router_logits)
    actual_weights, actual_ids = pallas_fn(router_logits)
    np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
    np.testing.assert_array_equal(
        np.asarray(actual_weights),
        np.asarray(expected_weights),
    )

    jax_us, jax_trace = _scope_device_us(
        functools.partial(jax_fn, router_logits),
        SCOPE_JAX,
        f"jax_t{tokens}",
    )
    pallas_us, pallas_trace = _scope_device_us(
        functools.partial(pallas_fn, router_logits),
        SCOPE_PALLAS,
        f"pallas_t{tokens}",
    )
    return jax_us, pallas_us, jax_trace, pallas_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        default="64,128,256,512,1024,2048,4096,8192,16384,32768",
    )
    parser.add_argument("--block-tokens", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()

    block_tokens = args.block_tokens if args.block_tokens == "auto" else int(args.block_tokens)
    print(f"JAX {jax.__version__} | {jax.devices()[0].device_kind}")
    print("MiMo gate-fed routing: H=6144 E=384 k=8; exact device_duration_ps")
    print(f"{'T':>7} {'jax_us':>10} {'pallas_us':>10} {'speedup':>9}")
    measurements = []
    for tokens in (int(value) for value in args.tokens.split(",")):
        jax_us, pallas_us, jax_trace, pallas_trace = benchmark_one(
            tokens,
            block_tokens,
        )
        speedup = jax_us / pallas_us if pallas_us > 0 else float("nan")
        print(f"{tokens:7d} {jax_us:10.2f} {pallas_us:10.2f} {speedup:8.2f}x")
        print(f"  traces: {jax_trace} {pallas_trace}")
        common = {
            "phase": "benchmark",
            "tokens": tokens,
            "experts": EXPERTS,
            "topk": TOPK,
            "block_tokens": block_tokens,
        }
        measurements.extend(
            [
                {
                    **common,
                    "variant": "jax",
                    "device_duration_us": jax_us,
                    "latency_ms": jax_us / 1000.0,
                },
                {
                    **common,
                    "variant": "pallas",
                    "device_duration_us": pallas_us,
                    "latency_ms": pallas_us / 1000.0,
                    "speedup_vs_jax": speedup,
                },
            ]
        )
    if args.output:
        with open(args.output, "w") as output:
            output.writelines(json.dumps(row, sort_keys=True) + "\n" for row in measurements)


if __name__ == "__main__":
    main()
