#!/usr/bin/env python3
"""
Inspect grok2 safetensors weights to understand the actual TP sharding format.
"""

import glob
import os
import sys

from safetensors import safe_open


def inspect_safetensors(model_path: str):
    """Inspect all safetensors files and print MoE expert weight shapes."""

    weights_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    if not weights_files:
        print(f"No safetensors files found in {model_path}")
        return

    print(f"Found {len(weights_files)} safetensors files")
    print("=" * 80)

    # Track MoE expert weights across files
    moe_weights = {}

    for file_idx, st_file in enumerate(weights_files):
        filename = os.path.basename(st_file)
        print(f"\n📦 File {file_idx}: {filename}")
        print("-" * 80)

        with safe_open(st_file, framework="pt") as f:
            keys = list(f.keys())

            # Filter MoE expert weights
            moe_keys = [k for k in keys if "block_sparse_moe.experts" in k and ".weight" in k]

            if moe_keys:
                print(f"  Found {len(moe_keys)} MoE expert weights")

                # Show first layer's first few experts as example
                layer0_experts = [k for k in moe_keys if "layers.0." in k]

                for key in sorted(layer0_experts)[:9]:  # Show first 3 experts (w1, w2, w3 each)
                    tensor = f.get_tensor(key)
                    shape = tuple(tensor.shape)
                    dtype = tensor.dtype

                    # Track shapes across files
                    if key not in moe_weights:
                        moe_weights[key] = []
                    moe_weights[key].append((file_idx, filename, shape, dtype))

                    print(f"    {key}: {shape} ({dtype})")

    # Summary: show how weights are distributed across files
    print("\n" + "=" * 80)
    print("📊 SUMMARY: Weight distribution across TP shards")
    print("=" * 80)

    if moe_weights:
        # Pick one example weight to analyze
        example_key = sorted(moe_weights.keys())[0]  # e.g., layers.0.experts.0.w1.weight

        print(f"\n🔍 Example: {example_key}")
        print(f"   Appears in {len(moe_weights[example_key])} files:")

        for file_idx, filename, shape, dtype in moe_weights[example_key]:
            print(f"     File {file_idx} ({filename}): shape={shape}, dtype={dtype}")

        # Analyze the sharding pattern
        shapes = [shape for _, _, shape, _ in moe_weights[example_key]]
        if len(set(shapes)) == 1:
            print(f"\n   ✅ All shards have the SAME shape: {shapes[0]}")
            print(
                f"   This suggests: Each file contains a complete copy, OR all shards are identical"
            )
        else:
            print(f"\n   ⚠️  Shards have DIFFERENT shapes:")
            for i, (_, filename, shape, _) in enumerate(moe_weights[example_key]):
                print(f"     Shard {i}: {shape}")

        # Check w1, w2, w3 patterns
        print("\n" + "=" * 80)
        print("🔬 Analyzing w1, w2, w3 shapes for layer 0, expert 0:")
        print("=" * 80)

        for weight_name in ["w1", "w2", "w3"]:
            key = f"model.layers.0.block_sparse_moe.experts.0.{weight_name}.weight"
            if key in moe_weights:
                shapes = [shape for _, _, shape, _ in moe_weights[key]]
                print(f"\n  {weight_name}:")
                print(f"    Num shards: {len(shapes)}")
                print(f"    Shard shapes: {shapes[0]} (assuming all equal)")

                # If TP=8, infer the concat dimension
                if len(shapes) == 8:
                    shape = shapes[0]
                    print(f"    💡 Inference (TP=8):")
                    if shape[0] != shape[1]:
                        # One dimension is much smaller
                        if len(shape) == 2:
                            for axis in [0, 1]:
                                concat_shape = list(shape)
                                concat_shape[axis] *= 8
                                print(
                                    f"       If concat on axis {axis}: final shape = {tuple(concat_shape)}"
                                )
    else:
        print("No MoE expert weights found!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_grok_weights.py <model_path>")
        print("Example: python inspect_grok_weights.py /path/to/grok2-model")
        sys.exit(1)

    model_path = sys.argv[1]
    inspect_safetensors(model_path)
