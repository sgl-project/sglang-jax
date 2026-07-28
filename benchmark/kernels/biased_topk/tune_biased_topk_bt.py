"""Sweep biased-topk token block sizes on TPU."""

import argparse
import functools
import json

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.kernels.biased_topk.bench_biased_topk import (
    _scope_device_us,
    reference_biased_topk,
)
from sgl_jax.srt.kernels.biased_topk import biased_topk_pallas
from sgl_jax.srt.kernels.biased_topk.tuned_block_sizes import _device_name

EXPERTS = 384
TOPK = 8


def _candidates(tokens):
    candidates = {
        value
        for value in (64, 128, 256, 512, 1024, 2048, 4096)
        if value <= tokens and tokens % value == 0
    }
    if tokens <= 2048:
        candidates.add(tokens)
    return sorted(candidates)


def tune_one(tokens, *, interpret):
    logits = jax.nn.sigmoid(
        jax.random.normal(
            jax.random.key(tokens),
            (tokens, EXPERTS),
            dtype=jnp.float32,
        )
    )
    bias = (
        jax.random.normal(
            jax.random.key(tokens + 1),
            (EXPERTS,),
            dtype=jnp.float32,
        )
        * 0.1
    )
    expected_weights, expected_ids = reference_biased_topk(
        logits,
        bias,
        topk=TOPK,
    )
    rows = []
    measurements = []
    print(f"\nT={tokens}")
    print(f"{'BT':>6} {'status':>12} {'device_us':>12}")
    for block_tokens in _candidates(tokens):
        fn = jax.jit(
            functools.partial(
                biased_topk_pallas,
                topk=TOPK,
                block_tokens=block_tokens,
                interpret=interpret,
            )
        )
        try:
            actual_weights, actual_ids = fn(logits, bias)
            jax.block_until_ready((actual_weights, actual_ids))
            np.testing.assert_array_equal(np.asarray(actual_ids), np.asarray(expected_ids))
            np.testing.assert_array_equal(
                np.asarray(actual_weights),
                np.asarray(expected_weights),
            )
            if interpret:
                device_us = float("nan")
            else:
                device_us, _ = _scope_device_us(
                    functools.partial(fn, logits, bias),
                    "biased-topk",
                    f"tune_t{tokens}_bt{block_tokens}",
                )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            status = (
                "OOM" if "vmem" in message.lower() or "RESOURCE_EXHAUSTED" in message else "FAIL"
            )
            print(f"{block_tokens:6d} {status:>12} {'-':>12}")
            measurements.append(
                {
                    "phase": "tune",
                    "variant": "pallas",
                    "tokens": tokens,
                    "experts": EXPERTS,
                    "topk": TOPK,
                    "block_tokens": block_tokens,
                    "status": status.lower(),
                    "error": message,
                }
            )
            continue
        print(f"{block_tokens:6d} {'ok':>12} {device_us:12.2f}")
        rows.append((block_tokens, device_us))
        measurements.append(
            {
                "phase": "tune",
                "variant": "pallas",
                "tokens": tokens,
                "experts": EXPERTS,
                "topk": TOPK,
                "block_tokens": block_tokens,
                "status": "ok",
                "device_duration_us": device_us,
                "latency_ms": device_us / 1000.0,
            }
        )
    if not rows:
        return None, measurements
    if interpret:
        return rows[-1][0], measurements
    return min(rows, key=lambda row: row[1])[0], measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        default="64,128,256,512,1024,2048,4096,8192,16384,32768",
    )
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    print(f"JAX {jax.__version__} | {jax.devices()[0].device_kind}")
    best = {}
    measurements = []
    for tokens in (int(value) for value in args.tokens.split(",")):
        block_tokens, token_measurements = tune_one(tokens, interpret=args.interpret)
        measurements.extend(token_measurements)
        if block_tokens is not None:
            best[(tokens, EXPERTS, TOPK)] = block_tokens

    print(f"\n# paste into TUNED_BT[{_device_name()!r}]")
    for key, block_tokens in best.items():
        print(f"{key!r}: {block_tokens},")
    if args.output:
        with open(args.output, "w") as output:
            output.writelines(json.dumps(row, sort_keys=True) + "\n" for row in measurements)


if __name__ == "__main__":
    main()
