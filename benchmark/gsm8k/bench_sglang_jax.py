"""GSM8K benchmark for sglang-jax.

Sends concurrent requests to a running sglang-jax server's /generate endpoint
and measures accuracy and throughput on the GSM8K (or GSM8K Platinum) dataset.

Usage:
    # Start server first:
    #   python3 -m sgl_jax.launch_server --model-path <model> --port 30000 ...

    # Run benchmark:
    python bench_sglang_jax.py --base-url http://localhost:30000 --num-questions 200
"""

import argparse
import ast
import asyncio
import json
import os
import re
import tempfile
import time
import urllib.request

import aiohttp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

INVALID = -9999999


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def download_and_cache_file(url):
    cache_dir = os.path.join(tempfile.gettempdir(), "sgl_jax_bench_cache")
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split("/")[-1]
    cache_path = os.path.join(cache_dir, filename)
    if not os.path.isfile(cache_path):
        print(f"Downloading {url} to {cache_path}...")
        urllib.request.urlretrieve(url, cache_path)
    return cache_path


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


async def send_request(session, base_url, text, sampling_params, semaphore, pbar):
    payload = {
        "text": text,
        "sampling_params": sampling_params,
        "stream": False,
    }
    async with semaphore:
        timeout = aiohttp.ClientTimeout(total=300)
        async with session.post(f"{base_url}/generate", json=payload, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Request failed with status {response.status}: {error_text}")
            result = await response.json()
            pbar.update(1)
            return result


async def run_batch(base_url, questions, sampling_params, parallel):
    semaphore = asyncio.Semaphore(parallel)
    pbar = tqdm(total=len(questions), desc="Generating")

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, base_url, q, sampling_params, semaphore, pbar) for q in questions
        ]
        results = await asyncio.gather(*tasks)

    pbar.close()
    return results


def main(args):
    # Load tokenizer if enable_thinking is set
    tokenizer = None
    if args.enable_thinking:
        from transformers import AutoTokenizer

        assert (
            args.tokenizer_path is not None
        ), "--tokenizer-path is required when --enable-thinking is set"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # Read data
    if args.platinum:
        print("Loading GSM8K Platinum dataset from HuggingFace...")
        dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
        lines = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
    else:
        data_path = args.data_path
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        if not os.path.isfile(data_path):
            data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        raw_question = few_shot_examples + get_one_example(lines, i, False)
        if tokenizer is not None:
            messages = [{"role": "user", "content": raw_question}]
            raw_question = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        questions.append(raw_question)
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(label != INVALID for label in labels)

    # Sampling parameters
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "stop": ["Question", "Assistant:", "<|separator|>"],
    }

    # Run requests
    print(
        f"Running {len(questions)} requests against {args.base_url} "
        f"(parallelism={args.parallel})..."
    )
    tic = time.perf_counter()
    results = asyncio.run(run_batch(args.base_url, questions, sampling_params, args.parallel))
    latency = time.perf_counter() - tic

    # Extract predictions
    preds = []
    for r in results:
        preds.append(get_answer_value(r["text"]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(r["meta_info"]["completion_tokens"] for r in results)
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Dump raw outputs
    if args.output_file:
        with open(args.output_file, "w") as f:
            for i, r in enumerate(results):
                f.write(f"=== Question {i} ===\n")
                f.write(questions[i] + "\n")
                f.write("=== Answer ===\n")
                f.write(r["text"] + "\n")
                f.write(f"=== Prediction: {preds[i]}, Label: {labels[i]} ===\n\n")
        print(f"Raw outputs saved to {args.output_file}")

    # Dump results
    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k-platinum" if args.platinum else "gsm8k",
            "backend": "sgl-jax",
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")
    print(f"Results appended to {args.result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K benchmark for sglang-jax")
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000",
        help="Base URL of the sglang-jax server",
    )
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--parallel", type=int, default=64, help="Max concurrent requests")
    parser.add_argument(
        "--result-file",
        type=str,
        default="bench_results.jsonl",
        help="Path to append JSON result summary",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to write detailed per-question outputs",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode by wrapping prompts with chat template",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer (required when --enable-thinking is set)",
    )
    parser.add_argument(
        "--platinum",
        action="store_true",
        help="Use GSM8K Platinum dataset (drop-in replacement with corrected labels)",
    )
    args = parser.parse_args()
    main(args)
