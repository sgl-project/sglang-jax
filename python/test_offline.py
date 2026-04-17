#!/usr/bin/env python3
"""
Offline test script for sglang-jax with Qwen/Qwen-7B-Chat model.

Usage:
    # On TPU
    python test_offline_qwen7b.py --device tpu --tp-size 4

    # On CPU (for debugging)
    python test_offline_qwen7b.py --device cpu --tp-size 1

    # On GPU
    python test_offline_qwen7b.py --device gpu --tp-size 1
"""

import argparse
import time
from typing import List

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.sampling.sampling_params import SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Offline test for Ling2.5-Mini with sglang-jax")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/models/gpu-ckpt-ling2.5/safetensors/mnt/pro/moe-lite/jobs/moe-mini-v25-e256-0520-fp8-20T-hl-mla-baseline/iter_0023842_hf",
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="tpu",
        choices=["tpu", "cpu", "gpu"],
        help="Device to use for inference",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.5,
        help="Memory fraction for static allocation",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="/tmp",
        help="Directory to download model weights",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--enable-single-process",
        action="store_true",
        help="Enable single process mode (useful for debugging)",
    )
    return parser.parse_args()


def create_engine(args):
    """Create and initialize the inference engine."""
    print(f"\n{'='*60}")
    print(f"Initializing engine with model: {args.model_path}")
    print(f"Device: {args.device}, TP size: {args.tp_size}, dtype: {args.dtype}")
    print(f"{'='*60}\n")

    engine = Engine(
        model_path=args.model_path,
        trust_remote_code=True,
        tp_size=args.tp_size,
        device=args.device,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        download_dir=args.download_dir,
        skip_server_warmup=False,
        attention_backend="fa",
        random_seed=42,
        node_rank=0,
        enable_single_process=args.enable_single_process,
        log_level="info",
    )

    # Keep physical embedding/lm_head vocab from config for TP sharding, but
    # restrict logits/sampling to tokenizer's actual usable token range.
    tokenizer = get_tokenizer(args.model_path, trust_remote_code=True)
    effective_vocab_size = len(tokenizer)
    if hasattr(engine, "tp_worker"):
        target = engine.tp_worker
    elif hasattr(engine, "scheduler") and hasattr(engine.scheduler, "tp_worker"):
        target = engine.scheduler.tp_worker
    else:
        target = None
    if target is not None:
        worker = target.worker if hasattr(target, "worker") else target
        worker.model_runner.model.config.effective_vocab_size = effective_vocab_size
        worker.model_runner.model.logits_processor.vocab_size = effective_vocab_size
    print(f"Effective sampling vocab size forced to: {effective_vocab_size}")
    return engine


def test_basic_generation(engine, tokenizer, args):
    """Test basic text generation."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Text Generation")
    print("=" * 60)

    prompts = [
        "The president of the United States is",
        "The capital of France is",
        "The capital of China is",
        "The future of AI is",
        "李白乘舟将欲行",
        "who are u?",
    ]

    sampling_params = engine.get_default_sampling_params()
    sampling_params.max_new_tokens = args.max_new_tokens
    sampling_params.temperature = args.temperature
    sampling_params.stop_token_ids = [tokenizer.eos_token_id]
    sampling_params.skip_special_tokens = True

    sampling_params_dict = sampling_params.convert_to_dict()

    start_time = time.time()
    outputs = engine.generate(
        prompt=prompts,
        sampling_params=[sampling_params_dict] * len(prompts),
    )
    elapsed = time.time() - start_time

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n[Prompt {i+1}]: {prompt}")
        print(f"[Output {i+1}]: {output['text']}")

    print(f"\nGeneration time: {elapsed:.2f}s")
    print(f"Throughput: {sum(len(o['output_ids']) for o in outputs) / elapsed:.2f} tokens/s")


def test_chat_completion(engine, tokenizer, args):
    """Test chat completion with Qwen chat template."""
    print("\n" + "=" * 60)
    print("Test 2: Chat Completion")
    print("=" * 60)

    # Build chat prompt using Qwen's chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the largest planet in our solar system?"},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"\n[Chat prompt]:\n{prompt}")

    sampling_params = engine.get_default_sampling_params()
    sampling_params.max_new_tokens = args.max_new_tokens
    sampling_params.temperature = args.temperature
    sampling_params.stop_token_ids = [tokenizer.eos_token_id]
    sampling_params.skip_special_tokens = True

    start_time = time.time()
    output = engine.generate(
        prompt=prompt,
        sampling_params=sampling_params.convert_to_dict(),
    )
    elapsed = time.time() - start_time

    print(f"\n[Response]: {output['text']}")
    print(f"\nGeneration time: {elapsed:.2f}s")


def test_multi_turn_conversation(engine, tokenizer, args):
    """Test multi-turn conversation."""
    print("\n" + "=" * 60)
    print("Test 3: Multi-turn Conversation")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    questions = [
        "Hello! What's your name?",
        "Can you tell me a fun fact about the moon?",
        "How far is the moon from Earth?",
    ]

    sampling_params = engine.get_default_sampling_params()
    sampling_params.max_new_tokens = 64
    sampling_params.temperature = args.temperature
    sampling_params.stop_token_ids = [tokenizer.eos_token_id]
    sampling_params.skip_special_tokens = True

    for i, question in enumerate(questions):
        messages.append({"role": "user", "content": question})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print(f"\n[Turn {i+1}] User: {question}")

        start_time = time.time()
        output = engine.generate(
            prompt=prompt,
            sampling_params=sampling_params.convert_to_dict(),
        )
        elapsed = time.time() - start_time

        response = output["text"]
        print(f"[Turn {i+1}] Assistant: {response}")
        print(f"Generation time: {elapsed:.2f}s")

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": response})


def test_batch_generation(engine, tokenizer, args):
    """Test batch generation with different sampling parameters."""
    print("\n" + "=" * 60)
    print("Test 4: Batch Generation with Different Parameters")
    print("=" * 60)

    prompts = [
        "Translate to English: Bonjour, comment allez-vous?",
        "Translate to Chinese: Hello, how are you?",
        "What is 2 + 2?",
    ]

    # Different sampling params for each prompt
    sampling_params_list = []
    for i, prompt in enumerate(prompts):
        sp = engine.get_default_sampling_params()
        sp.max_new_tokens = 32
        sp.temperature = 0.0 if i == 2 else 0.7  # Greedy for math, sampling for translations
        sp.stop_token_ids = [tokenizer.eos_token_id]
        sp.skip_special_tokens = True
        sampling_params_list.append(sp.convert_to_dict())

    start_time = time.time()
    outputs = engine.generate(
        prompt=prompts,
        sampling_params=sampling_params_list,
    )
    elapsed = time.time() - start_time

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n[Prompt {i+1}]: {prompt}")
        print(f"[Output {i+1}]: {output['text']}")

    print(f"\nTotal batch generation time: {elapsed:.2f}s")


def test_streaming_generation(engine, tokenizer, args):
    """Test streaming generation."""
    print("\n" + "=" * 60)
    print("Test 5: Streaming Generation")
    print("=" * 60)

    prompt = "Write a short story about a robot learning to paint:"

    print(f"\n[Prompt]: {prompt}")
    print("[Response (streaming)]: ", end="", flush=True)

    sampling_params = engine.get_default_sampling_params()
    sampling_params.max_new_tokens = 128
    sampling_params.temperature = 0.8
    sampling_params.stop_token_ids = [tokenizer.eos_token_id]
    sampling_params.skip_special_tokens = True

    start_time = time.time()
    generator = engine.generate(
        prompt=prompt,
        sampling_params=sampling_params.convert_to_dict(),
        stream=True,
    )

    full_text = ""
    for chunk in generator:
        text = chunk["text"]
        delta = text[len(full_text):]
        print(delta, end="", flush=True)
        full_text = text

    elapsed = time.time() - start_time
    print(f"\n\nStreaming time: {elapsed:.2f}s")


def test_input_ids_generation(engine, tokenizer, args):
    """Test generation with input_ids instead of text."""
    print("\n" + "=" * 60)
    print("Test 6: Generation with Input IDs")
    print("=" * 60)

    prompts = [
        "The quick brown fox",
        "In the beginning",
    ]

    # Tokenize prompts
    input_ids_list = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        print(f"\n[Prompt]: {prompt}")
        print(f"[Input IDs]: {ids[:10]}..." if len(ids) > 10 else f"[Input IDs]: {ids}")
        input_ids_list.append(ids)

    sampling_params = engine.get_default_sampling_params()
    sampling_params.max_new_tokens = 32
    sampling_params.temperature = 0.0
    sampling_params.stop_token_ids = [tokenizer.eos_token_id]
    sampling_params.skip_special_tokens = True

    start_time = time.time()
    outputs = engine.generate(
        input_ids=input_ids_list,
        sampling_params=[sampling_params.convert_to_dict()] * len(prompts),
    )
    elapsed = time.time() - start_time

    for i, output in enumerate(outputs):
        print(f"\n[Output {i+1}]: {output['text']}")

    print(f"\nGeneration time: {elapsed:.2f}s")


def main():
    args = parse_args()

    # Create engine
    engine = create_engine(args)

    # Get tokenizer
    tokenizer = get_tokenizer(
        args.model_path,
        trust_remote_code=True,
    )

    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    print(f"BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    try:
        # Run tests
        test_basic_generation(engine, tokenizer, args)
        # test_chat_completion(engine, tokenizer, args)
        # test_multi_turn_conversation(engine, tokenizer, args)
        # test_batch_generation(engine, tokenizer, args)
        # test_streaming_generation(engine, tokenizer, args)
        # test_input_ids_generation(engine, tokenizer, args)

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    finally:
        # Cleanup
        print("\nShutting down engine...")
        engine.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()