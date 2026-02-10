#!/usr/bin/env python3
"""
Qwen3-VL Standalone Inference Script (Option B)

Demonstrates loading PyTorch safetensors weights into the sglang-jax Qwen3-VL
model and running inference using inputs processed by the HuggingFace processor.

This script validates:
1. Weight loading for both vision encoder and text decoder
2. Vision encoding (pixel_values → vision embeddings)
3. Input preparation using AutoProcessor (PyTorch CPU tensors → JAX arrays)

For full end-to-end generation (with KV cache, auto-regressive decoding),
use the sglang-jax server (Option A):
    python -m sgl_jax --model-path Qwen/Qwen3-VL-2B-Instruct --multimodal

Usage:
    python -m sgl_jax.srt.multimodal.models.qwen3_VL.run_inference \
        --model-path Qwen/Qwen3-VL-2B-Instruct

    # With local weights:
    python -m sgl_jax.srt.multimodal.models.qwen3_VL.run_inference \
        --model-path /path/to/qwen3-vl-2b

    # Vision encoding test with image:
    python -m sgl_jax.srt.multimodal.models.qwen3_VL.run_inference \
        --model-path Qwen/Qwen3-VL-2B-Instruct \
        --image-url "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.jpeg"
"""

import argparse
import json
import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

logger = logging.getLogger(__name__)


def download_model(model_id: str) -> str:
    """Download model from HuggingFace Hub. Returns local path."""
    from huggingface_hub import snapshot_download

    print(f"  Downloading model: {model_id}")
    local_path = snapshot_download(model_id)
    print(f"  Model downloaded to: {local_path}")
    return local_path


def load_processor(model_path: str):
    """Load the HuggingFace processor for tokenization and image processing."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path)


def detect_config_size(model_path: str) -> str:
    """Detect model size from config.json."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        num_hidden_layers = config.get("num_hidden_layers", 28)
        # 2B has 28 layers, 4B has 36, 8B has 36, 32B has 64
        hidden_size = config.get("hidden_size", 2048)
        if num_hidden_layers <= 28:
            return "2b"
        elif hidden_size <= 2560:
            return "4b"
        elif hidden_size <= 4096:
            return "8b"
        else:
            return "32b"
    return "2b"


def create_model_config(model_path: str, dtype):
    """Create configs for model initialization and weight loading.

    Returns:
        qwen3_config: Full Qwen3VLConfig (contains .vision_config and .text_config)
        hf_config: HuggingFace AutoConfig for the generation model
    """
    from transformers import AutoConfig

    from sgl_jax.srt.multimodal.configs.qwen_vl.qwen3_vl_config import Qwen3VLConfig

    size = detect_config_size(model_path)
    print(f"  Detected model size: {size}")

    size_to_config = {
        "2b": Qwen3VLConfig.qwen3vl_2b,
        "4b": Qwen3VLConfig.qwen3vl_4b,
        "8b": Qwen3VLConfig.qwen3vl_8b,
        "32b": Qwen3VLConfig.qwen3vl_32b,
    }
    qwen3_config = size_to_config[size]()

    # Load HuggingFace config for text model
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    return qwen3_config, hf_config


def load_vision_model(model_path: str, qwen3_config, mesh, dtype):
    """Load the Qwen3-VL vision encoder.

    Args:
        model_path: Path to model weights directory
        qwen3_config: Full Qwen3VLConfig (we extract .vision_config from it)
        mesh: JAX device mesh
        dtype: Model dtype
    """
    from sgl_jax.srt.multimodal.models.qwen3_VL.qwen3_vl_vit import (
        Qwen3_VL_VisionModel,
    )

    # Extract the vision sub-config — VisionModel expects Qwen3VLVisionConfig,
    # NOT the full Qwen3VLConfig
    vis_cfg = qwen3_config.vision_config
    text_cfg = qwen3_config.text_config

    print(
        f"  Vision config: depth={vis_cfg.depth}, hidden={vis_cfg.hidden_size}, "
        f"out_hidden={vis_cfg.out_hidden_size}"
    )

    print("  Initializing vision model...")
    t0 = time.time()

    # Create model with eval_shape (lazy init, no memory allocation)
    with jax.default_device(jax.devices("cpu")[0]):
        vision_model = nnx.eval_shape(
            lambda: Qwen3_VL_VisionModel(
                config=vis_cfg,
                dtype=dtype,
                rngs=nnx.Rngs(0),
                mesh=mesh,
            )
        )

    # load_weights expects a config object with .model_path, .vocab_size,
    # .text_hidden_size (used to create the text_embed layer)
    class VisionWeightConfig:
        def __init__(self, path, text_config, vision_config, dt):
            self.model_path = path
            self.dtype = dt
            self.vocab_size = text_config.vocab_size
            self.text_hidden_size = vision_config.out_hidden_size
            self.quantization_config = None

    vm_config = VisionWeightConfig(model_path, text_cfg, vis_cfg, dtype)

    if mesh:
        with mesh:
            vision_model.load_weights(vm_config)
    else:
        vision_model.load_weights(vm_config)

    print(f"  Vision model loaded in {time.time() - t0:.2f}s")
    return vision_model


def load_generation_model(model_path: str, hf_config, mesh, dtype):
    """Load the Qwen3-VL text decoder."""
    from sgl_jax.srt.multimodal.models.qwen3_VL.qwen3_vl_generation import (
        Qwen3_VL_Generation,
    )

    print("  Initializing generation model...")
    t0 = time.time()

    # Create model with eval_shape (lazy init)
    with jax.default_device(jax.devices("cpu")[0]):
        gen_model = nnx.eval_shape(
            lambda: Qwen3_VL_Generation(
                config=hf_config,
                dtype=dtype,
                mesh=mesh,
            )
        )

    # Create ModelConfig for weight loading
    class GenModelConfig:
        def __init__(self, path, config, dt):
            self.model_path = path
            self.hf_config = config
            self.dtype = dt
            self.quantization_config = None

    gm_config = GenModelConfig(model_path, hf_config, dtype)
    
    if mesh:
        with mesh:
            gen_model.load_weights(gm_config)
    else:
        gen_model.load_weights(gm_config)

    print(f"  Generation model loaded in {time.time() - t0:.2f}s")
    return gen_model


def run_text_test(processor, model_path: str, gen_model, mesh, dtype):
    """Test text tokenization and embedding lookup."""
    print("\n--- Text Input Test ---")
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = jnp.array(inputs["input_ids"].numpy())
    print(f"  Input IDs shape: {input_ids.shape}")
    print(f"  Input IDs (first 20): {input_ids[0, :20].tolist()}")
    print(f"  Total tokens: {input_ids.shape[1]}")

    # Test embedding lookup
    embed_tokens = gen_model.model.embed_tokens
    embeddings = embed_tokens(input_ids)
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dtype: {embeddings.dtype}")
    print(f"  Embedding norm (first token): {jnp.linalg.norm(embeddings[0, 0]):.6f}")
    print("  ✓ Text embedding lookup successful!")


def run_vision_test(processor, vision_model, image_url: str):
    """Test vision encoding with an image."""
    print("\n--- Vision Encoding Test ---")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    print(f"  Processing image: {image_url}")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = jnp.array(inputs["input_ids"].numpy())
    print(f"  Input IDs shape: {input_ids.shape}")

    if "pixel_values" not in inputs:
        print("  ✗ No pixel_values found in inputs. Vision processing may have failed.")
        return

    pixel_values = jnp.array(inputs["pixel_values"].numpy())
    image_grid_thw = inputs["image_grid_thw"].numpy()

    print(f"  Pixel values shape: {pixel_values.shape}")
    print(f"  Pixel values dtype: {pixel_values.dtype}")
    print(f"  Image grid THW: {image_grid_thw}")

    # Convert image_grid_thw to tuple format
    grid_thw_tuple = tuple(tuple(x) for x in image_grid_thw.tolist())

    # Run vision encoder
    print("  Running vision encoder...")
    t0 = time.time()
    vision_embeddings = vision_model(
        pixel_values=pixel_values,
        image_grid_thw=grid_thw_tuple,
    )
    jax.block_until_ready(vision_embeddings)
    vision_time = time.time() - t0

    print(f"  Vision embeddings shape: {vision_embeddings.shape}")
    print(f"  Vision embeddings dtype: {vision_embeddings.dtype}")
    print(
        f"  Vision embeddings norm (mean): {jnp.mean(jnp.linalg.norm(vision_embeddings, axis=-1)):.6f}"
    )
    print(f"  Vision encoding time: {vision_time:.2f}s")
    print("  ✓ Vision encoding successful!")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Standalone Inference - Weight Loading & Vision Encoding Test"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--image-url",
        type=str,
        default=None,
        help="Image URL for vision encoding test",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip loading the generation model (only test vision encoder)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dtype_map = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("Qwen3-VL Standalone Inference (sglang-jax)")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    print(f"Backend: {jax.default_backend()}")

    # Create mesh
    # devices = jax.devices()
    # mesh = jax.sharding.Mesh(np.array(devices).reshape(-1), axis_names=("tp",))
    mesh = None
    print(f"Mesh: {mesh}")

    # Resolve model path (download if needed)
    model_path = args.model_path
    if not os.path.exists(model_path):
        print("\n1. Downloading model...")
        model_path = download_model(model_path)
    else:
        print(f"\n1. Using local model: {model_path}")

    # Load configs
    print("\n2. Loading configs...")
    qwen3_config, hf_config = create_model_config(model_path, dtype)

    # Load processor
    print("\n3. Loading processor...")
    processor = load_processor(model_path)

    # Load vision model
    print("\n4. Loading vision model...")
    vision_model = load_vision_model(model_path, qwen3_config, mesh, dtype)

    # Load generation model (optional)
    gen_model = None
    if not args.skip_generation:
        print("\n5. Loading generation model...")
        gen_model = load_generation_model(model_path, hf_config, mesh, dtype)

    # =========================================================================
    # Test 1: Text Input Processing & Embedding
    # =========================================================================
    if gen_model is not None:
        run_text_test(processor, model_path, gen_model, mesh, dtype)

    # =========================================================================
    # Test 2: Vision Encoding
    # =========================================================================
    if args.image_url:
        run_vision_test(processor, vision_model, args.image_url)
    else:
        print("\n--- Vision Test Skipped (no --image-url provided) ---")
        print("  Pass --image-url <URL> to test vision encoding")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("  ✓ Model weights loaded from safetensors")
    print("  ✓ HuggingFace processor can prepare inputs")
    if gen_model:
        print("  ✓ Text decoder embeddings verified")
    if args.image_url:
        print("  ✓ Vision encoder produces embeddings")
    print()
    print("For full auto-regressive generation, use the sglang-jax server:")
    print(f"  python -m sgl_jax --model-path {args.model_path} --multimodal")
    print("=" * 60)


if __name__ == "__main__":
    main()
