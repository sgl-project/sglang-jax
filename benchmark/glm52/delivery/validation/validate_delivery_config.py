"""Validate the exact fused-MoE v2 delivery hot-shape tune entries."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import yaml

CHANNEL_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "python/sgl_jax/srt/utils/quantization/configs/"
    "fp8_glm52_static_per_channel_moe_w8a8_linear_w8a16.yaml"
)
TUNED_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "python/sgl_jax/srt/kernels/fused_moe/v2/tuned_block_configs.py"
)

EXPECTED = {
    (8, "blockwise"): {
        32: (8, 512, 8, 128, 8),
        32768: (128, 1024, 64, 1024, 128),
    },
    (16, "blockwise"): {
        64: (8, 512, 8, 128, 8),
        65536: (128, 1024, 32, 1024, 160),
    },
    (8, "channelwise"): {
        32: (8, 1024, 8, 512, 8),
        32768: (128, 1024, 128, 1024, 128),
    },
    (16, "channelwise"): {
        64: (8, 512, 8, 512, 8),
        65536: (64, 1024, 128, 1024, 128),
    },
}


def _validate_channel_config() -> None:
    config = yaml.safe_load(CHANNEL_CONFIG.read_text())["quantization"]
    assert config["is_static_checkpoint"] is True
    assert config["weight_block_size"] is None
    assert config["per_channel_matmul_backend"] == "pallas"
    assert config["moe"] == {
        "weight_dtype": "float8_e4m3fn",
        "activation_dtype": "float8_e4m3fn",
    }
    rules = config["linear"]["rules"]
    assert rules
    assert all(rule["weight_dtype"] == "float8_e4m3fn" for rule in rules)
    assert all(rule["activation_dtype"] is None for rule in rules)


def _load_tuned_configs() -> dict[str, dict[tuple[Any, ...], tuple[int, ...]]]:
    tree = ast.parse(TUNED_CONFIG.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TUNED_BLOCK_CONFIGS"
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"TUNED_BLOCK_CONFIGS not found in {TUNED_CONFIG}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-chips", type=int, choices=(8, 16), required=True)
    parser.add_argument(
        "--quantization", choices=("blockwise", "channelwise"), required=True
    )
    args = parser.parse_args()

    if args.quantization == "channelwise":
        _validate_channel_config()

    ep_size = 16 if args.physical_chips == 8 else 32
    tuned_configs = _load_tuned_configs()["TPU v7"]
    actual = {}
    expected = EXPECTED[(args.physical_chips, args.quantization)]
    for tokens in expected:
        key = (
            "bfloat16",
            "float8_e4m3fn",
            tokens,
            256,
            8,
            6144,
            2048,
            ep_size,
            True,
            False,
            True,
        )
        if args.quantization == "channelwise":
            key += ("per_channel",)
        actual[tokens] = tuned_configs.get(key)
    if actual != expected:
        raise SystemExit(f"tuned config mismatch: actual={actual}, expected={expected}")
    print(
        "GLM52_DELIVERY_TUNE_CONFIG_OK",
        f"physical_chips={args.physical_chips}",
        f"quantization={args.quantization}",
        actual,
    )


if __name__ == "__main__":
    main()
