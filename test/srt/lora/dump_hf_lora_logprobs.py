#!/usr/bin/env python3
# Copyright 2023-2024 SGLang-Jax Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Dump HuggingFace LoRA log probabilities for comparison with sglang-jax.
Attention: Need install torch, transformer and sglang package first.

This script:
1. Runs HuggingFace with LoRA adapters
2. Collects log probabilities (prefill and decode)
3. Saves the results to a JSON file for use in sglang-jax comparison

Usage:
    python dump_hf_lora_logprobs.py --model meta-llama/Llama-2-7b-hf \
        --lora-path yushengsu/sglang_lora_logprob_diff_without_tuning \
        --output hf_logprobs.json

    # Multiple prompts and LoRA adapters
    python dump_hf_lora_logprobs.py \
        --model meta-llama/Llama-2-7b-hf \
        --lora-path adapter1 adapter2 \
        --prompt "Hello" "How are you?" \
        --max-new-tokens 64 \
        --output hf_logprobs.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from misc.hf_runner import HFRunner

# Default configuration
DEFAULT_TEST_PROMPTS = [
    "Chatgpt is a chat bot",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]


def prepare_lora_paths_per_prompt(lora_paths: List[str], num_prompts: int) -> List[Optional[str]]:
    """
    Prepare LoRA paths for each prompt by cycling through available LoRAs.

    Args:
        lora_paths: List of available LoRA adapter paths
        num_prompts: Number of prompts to generate LoRA paths for

    Returns:
        List of LoRA paths (one per prompt), or None values if no LoRAs
    """
    if not lora_paths:
        return [None] * num_prompts

    return [lora_paths[i % len(lora_paths)] for i in range(num_prompts)]


def run_hf_with_lora(
    model_path: str,
    lora_paths: List[str],
    prompts: List[str],
    max_new_tokens: int,
    torch_dtype: torch.dtype,
    use_cpu: bool = False,
    num_layers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run HuggingFace with LoRA and return log probabilities.

    Args:
        model_path: Path to the base model
        lora_paths: List of LoRA adapter paths
        prompts: List of input prompts
        max_new_tokens: Maximum number of tokens to generate
        torch_dtype: PyTorch data type for model

    Returns:
        Dictionary containing logprobs and outputs
    """
    print(f"\n{'='*80}")
    print("Running HuggingFace with LoRA")
    print(f"{'='*80}")
    print(f"  Model: {model_path}")
    print(f"  LoRA paths: {lora_paths}")
    print(f"  Number of prompts: {len(prompts)}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Data type: {torch_dtype}")
    print(f"  Device: {'CPU' if use_cpu else 'GPU'}")
    print(f"  Num layers: {num_layers if num_layers is not None else 'All'}")

    lora_paths_per_prompt = prepare_lora_paths_per_prompt(lora_paths, len(prompts))

    print(f"\nLoRA assignments per prompt:")
    for i, (prompt, lora) in enumerate(zip(prompts, lora_paths_per_prompt)):
        print(f"  [{i}] LoRA: {lora}")
        print(f"      Prompt: {prompt[:60]}...")

    with HFRunner(
        model_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        patch_model_do_sample_false=True,
        use_cpu=use_cpu,
        num_layers=num_layers,
    ) as hf_runner:
        hf_outputs = hf_runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=lora_paths_per_prompt,
        )

    return {
        "top_input_logprobs": hf_outputs.top_input_logprobs,
        "top_output_logprobs": hf_outputs.top_output_logprobs,
        "output_strs": hf_outputs.output_strs,
        "lora_paths": lora_paths_per_prompt,
    }


def convert_to_serializable(obj):
    """
    Convert numpy/torch objects to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def save_logprobs_to_json(
    data: Dict[str, Any],
    output_path: str,
    model_path: str,
    prompts: List[str],
    max_new_tokens: int,
):
    """
    Save log probabilities to JSON file.

    Args:
        data: Data dictionary from HF runner
        output_path: Output file path
        model_path: Model path used
        prompts: Input prompts used
        max_new_tokens: Max new tokens setting
    """
    output_data = {
        "metadata": {
            "model": model_path,
            "num_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
            "prompts": prompts,
        },
        "results": [],
    }

    # Process each prompt
    for i in range(len(prompts)):
        result = {
            "prompt_idx": i,
            "prompt": prompts[i],
            "lora_path": data["lora_paths"][i],
            "output_str": data["output_strs"][i],
            "prefill_logprobs": convert_to_serializable(data["top_input_logprobs"][i]),
            "decode_logprobs": convert_to_serializable(data["top_output_logprobs"][i]),
            "prefill_shape": list(np.array(data["top_input_logprobs"][i]).shape),
            "decode_shape": list(np.array(data["top_output_logprobs"][i]).shape),
        }
        output_data["results"].append(result)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Saved log probabilities to: {output_path}")
    print(f"{'='*80}")
    print(f"Total prompts: {len(prompts)}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def print_summary(data: Dict[str, Any], prompts: List[str]):
    """
    Print a summary of the collected log probabilities.

    Args:
        data: Data dictionary from HF runner
        prompts: Input prompts
    """
    print(f"\n{'='*80}")
    print("Summary of Collected Log Probabilities")
    print(f"{'='*80}")

    for i in range(len(prompts)):
        print(f"\n{'-'*40}")
        print(f"Prompt {i+1}: {prompts[i][:50]}...")
        print(f"{'-'*40}")
        print(f"  LoRA adapter: {data['lora_paths'][i]}")

        prefill_shape = np.array(data["top_input_logprobs"][i]).shape
        decode_shape = np.array(data["top_output_logprobs"][i]).shape

        print(f"  Prefill logprobs shape: {prefill_shape}")
        print(f"  Decode logprobs shape:  {decode_shape}")

        output_str = data["output_strs"][i]
        display_len = 100
        output_display = (
            output_str[:display_len] + "..." if len(output_str) > display_len else output_str
        )
        print(f"  Generated text: {output_display}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump HuggingFace LoRA log probabilities for sglang-jax comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID (e.g., meta-llama/Llama-2-7b-hf)",
    )

    parser.add_argument(
        "--lora-path",
        type=str,
        nargs="+",
        default=[],
        help="LoRA adapter path(s). Can specify multiple paths.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=None,
        help="Input prompts. If not specified, will use default test prompts.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1,
        help="Maximum number of tokens to generate (default: 1)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="hf_lora_logprobs.json",
        help="Output JSON file path (default: hf_lora_logprobs.json)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Model data type (default: float16)",
    )

    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Run on CPU instead of GPU",
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers to forward (e.g., 1 for only first layer). None means all layers.",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Use provided prompts or defaults
    prompts = args.prompt if args.prompt else DEFAULT_TEST_PROMPTS

    # Validate LoRA paths
    if not args.lora_path:
        print("Warning: No LoRA adapters specified. Running without LoRA.")

    # Run HF with LoRA
    try:
        data = run_hf_with_lora(
            model_path=args.model,
            lora_paths=args.lora_path,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            torch_dtype=torch_dtype,
            use_cpu=args.use_cpu,
            num_layers=args.num_layers,
        )

        # Print summary
        print_summary(data, prompts)

        # Save to JSON
        save_logprobs_to_json(
            data=data,
            output_path=args.output,
            model_path=args.model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )

        print("\nDone! You can now use this file for comparison with sglang-jax.")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
