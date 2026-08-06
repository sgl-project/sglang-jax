"""Sweep exact SparseCore radix top-k configs on TPU.

The lookup key intentionally contains only ``(score_size, topk)``. The batch
dimension is fixed to one while tuning because the radix parameters control how
one score row is partitioned across SparseCore windows and digits.
"""

# ruff: noqa: E402

import argparse
import functools
import json
import os

os.environ.setdefault("TOPK_TRACE_ROOT", "/tmp/tpu_logs/radix_topk_tune")

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.kernels.biased_topk.bench_biased_topk import _scope_device_us
from sgl_jax.srt.kernels.radix_topk import radix_topk_pallas
from sgl_jax.srt.kernels.radix_topk.tuned_configs import (
    RadixTopKConfig,
    _device_name,
    make_radix_topk_config,
)

SCOPE = "RADIX_TOPK_TUNE"


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _tc_tiling_options(value: str) -> tuple[bool, ...]:
    if value == "both":
        return (False, True)
    return (value == "true",)


def _parse_digit_configs(value: str) -> tuple[tuple[int, int], ...]:
    configs = []
    for item in value.split(","):
        digit_width, num_digits = item.lower().split("x", maxsplit=1)
        configs.append((int(digit_width), int(num_digits)))
    return tuple(configs)


def _candidate_configs(args) -> list[RadixTopKConfig]:
    return [
        make_radix_topk_config(
            num_seq_windows=num_windows,
            digit_width=digit_width,
            num_digits=num_digits,
            use_tc_tiling_on_sc=tc_tiling,
        )
        for num_windows in _parse_ints(args.num_seq_windows)
        for digit_width, num_digits in _parse_digit_configs(args.digit_configs)
        for tc_tiling in _tc_tiling_options(args.tc_tiling)
        if digit_width != 4 or not tc_tiling
    ]


def _config_label(config: RadixTopKConfig) -> str:
    return (
        f"w{config.num_seq_windows}_d{config.digit_width}x{config.num_digits}"
        f"_tc{int(config.use_tc_tiling_on_sc)}"
    )


def _pad_scores(scores: jax.Array, config: RadixTopKConfig) -> jax.Array:
    padding = (-scores.shape[-1]) % config.input_alignment
    return jnp.pad(scores, ((0, 0), (0, padding)), constant_values=-jnp.inf)


def _make_run(config: RadixTopKConfig, *, topk: int):
    def run(scores):
        with jax.named_scope(SCOPE):
            return radix_topk_pallas(
                scores,
                k=topk,
                use_approx_top_k=False,
                num_seq_windows=config.num_seq_windows,
                digit_width=config.digit_width,
                num_digits=config.num_digits,
                use_tc_tiling_on_sc=config.use_tc_tiling_on_sc,
            )

    return jax.jit(run)


def _assert_exact(
    expected_values: jax.Array,
    actual_values: jax.Array,
):
    np.testing.assert_array_equal(
        np.sort(np.asarray(actual_values), axis=-1),
        np.sort(np.asarray(expected_values), axis=-1),
    )


def tune_one(score_size: int, topk: int, configs: list[RadixTopKConfig]):
    scores = jax.random.normal(jax.random.key(score_size + topk), (1, score_size), jnp.float32)
    expected_values, _ = jax.lax.top_k(scores, topk)
    jax.block_until_ready(expected_values)

    rows = []
    measurements = []
    print(f"\nscore_size={score_size} topk={topk}")
    print(f"{'config':>20} {'padded_N':>10} {'status':>10} {'device_us':>12}")
    for config in configs:
        label = _config_label(config)
        padded_scores = _pad_scores(scores, config)
        run = _make_run(config, topk=topk)
        try:
            actual_values, actual_indices = run(padded_scores)
            jax.block_until_ready((actual_values, actual_indices))
            _assert_exact(expected_values, actual_values)
            device_us, _ = _scope_device_us(
                functools.partial(run, padded_scores),
                SCOPE,
                f"radix_n{score_size}_k{topk}_{label}",
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            status = (
                "OOM" if "vmem" in message.lower() or "RESOURCE_EXHAUSTED" in message else "FAIL"
            )
            print(f"{label:>20} {padded_scores.shape[-1]:10d} {status:>10} {'-':>12}")
            measurements.append(
                {
                    "phase": "tune",
                    "variant": "radix",
                    "score_size": score_size,
                    "padded_score_size": padded_scores.shape[-1],
                    "topk": topk,
                    "config": vars(config),
                    "status": status.lower(),
                    "error": message,
                }
            )
            continue

        print(f"{label:>20} {padded_scores.shape[-1]:10d} {'ok':>10} {device_us:12.2f}")
        rows.append((config, device_us))
        measurements.append(
            {
                "phase": "tune",
                "variant": "radix",
                "score_size": score_size,
                "padded_score_size": padded_scores.shape[-1],
                "topk": topk,
                "config": vars(config),
                "status": "ok",
                "device_duration_us": device_us,
                "latency_ms": device_us / 1000.0,
            }
        )

    if not rows:
        return None, measurements
    return min(rows, key=lambda row: row[1])[0], measurements


def _print_config(score_size: int, topk: int, config: RadixTopKConfig):
    print(f"    ({score_size}, {topk}): RadixTopKConfig(")
    print(f"        num_seq_windows={config.num_seq_windows},")
    print(f"        digit_width={config.digit_width},")
    print(f"        num_digits={config.num_digits},")
    print(f"        use_tc_tiling_on_sc={config.use_tc_tiling_on_sc},")
    print("    ),")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-sizes", default="135168")
    parser.add_argument("--topks", default="2048")
    parser.add_argument("--num-seq-windows", default="1")
    parser.add_argument("--digit-configs", default="8x4,4x8")
    parser.add_argument("--tc-tiling", choices=["false", "true", "both"], default="both")
    parser.add_argument("--output")
    args = parser.parse_args()

    print(f"JAX {jax.__version__} | {jax.devices()[0].device_kind}")
    configs = _candidate_configs(args)
    best = {}
    measurements = []
    for score_size in _parse_ints(args.score_sizes):
        for topk in _parse_ints(args.topks):
            if not 1 <= topk <= score_size:
                print(f"skip invalid shape: score_size={score_size}, topk={topk}")
                continue
            config, shape_measurements = tune_one(score_size, topk, configs)
            measurements.extend(shape_measurements)
            if config is not None:
                best[(score_size, topk)] = config

    print(f"\n# paste into TUNED_RADIX_TOPK_CONFIGS[{_device_name()!r}]")
    for (score_size, topk), config in best.items():
        _print_config(score_size, topk, config)
    if args.output:
        with open(args.output, "w") as output:
            output.writelines(json.dumps(row, sort_keys=True) + "\n" for row in measurements)


if __name__ == "__main__":
    main()
