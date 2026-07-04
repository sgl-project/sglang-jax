"""MMLU-Pro benchmark for sglang-jax.

Sends concurrent requests to a running sglang-jax server's /generate endpoint
and measures multiple-choice accuracy on TIGER-Lab/MMLU-Pro.

Usage:
    python benchmark/mmlu_pro/bench_sglang_jax.py \
      --base-url http://localhost:30000 \
      --tokenizer-path /models/MiMo-V2-Flash \
      --enable-thinking \
      --num-questions 200 \
      --parallel 128
"""

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

INVALID = "<invalid>"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
THINK_CLOSE_RE = re.compile(r"^.*?</think>", flags=re.DOTALL)


def strip_reasoning(text: str) -> str:
    if not text:
        return text
    text = THINK_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = THINK_CLOSE_RE.sub("", text, count=1)
    return text.strip()


def normalize_response_text(result) -> str:
    text = result.get("text", "")
    if isinstance(text, list):
        return "".join(str(x) for x in text)
    return str(text)


def extract_answer(text: str, valid_letters: set[str]) -> str:
    stripped = strip_reasoning(text)
    suffix = stripped[-1200:]
    valid = "".join(sorted(valid_letters))

    indicator_patterns = [
        rf"(?i)(?:final\s+answer|answer)\s*(?:is|:)?\s*[\(\[]?\s*([{valid}])\s*[\)\].,:\s]",
        rf"(?i)(?:therefore|so|thus)[^\n]{{0,120}}?\b([{valid}])\b",
    ]
    for pattern in indicator_patterns:
        matches = re.findall(pattern, suffix)
        if matches:
            return matches[-1].upper()

    standalone = re.findall(rf"(?<![A-Za-z])([{valid}])(?![A-Za-z])", suffix)
    if standalone:
        return standalone[-1].upper()
    return INVALID


def format_prompt(item) -> tuple[str, list[str], str]:
    options = list(item["options"])
    if len(options) > len(LETTERS):
        raise ValueError(f"Too many options: {len(options)}")

    answer_index = int(item["answer_index"])
    label = LETTERS[answer_index]
    lines = [
        "Answer the following multiple choice question.",
        "Think step by step, then put the final answer on the last line as: Answer: <letter>.",
        "",
        f"Question: {item['question']}",
        "",
        "Options:",
    ]
    for i, option in enumerate(options):
        lines.append(f"{LETTERS[i]}. {option}")
    return "\n".join(lines), options, label


def apply_chat_template(tokenizer, prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking:
        kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


async def send_request(session, base_url, prompt, sampling_params, semaphore, pbar):
    payload = {
        "text": prompt,
        "sampling_params": sampling_params,
        "stream": False,
    }
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=1800)
    async with semaphore:
        async with session.post(f"{base_url}/generate", json=payload, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Request failed with status {response.status}: {error_text}")
            result = await response.json()
            pbar.update(1)
            return result


async def run_batch(base_url, prompts, sampling_params, parallel):
    semaphore = asyncio.Semaphore(parallel)
    connector = aiohttp.TCPConnector(limit=max(parallel * 2, 100), ttl_dns_cache=300)
    pbar = tqdm(total=len(prompts), desc="Generating")

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            send_request(session, base_url, prompt, sampling_params, semaphore, pbar)
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)

    pbar.close()
    return results


def choose_examples(dataset, num_questions: int | None, seed: int):
    indices = list(range(len(dataset)))
    if num_questions is not None and num_questions < len(indices):
        rng = random.Random(seed)
        indices = rng.sample(indices, num_questions)
    return [dataset[i] for i in indices]


def completion_tokens(result) -> int:
    meta = result.get("meta_info", {}) or {}
    for key in ("completion_tokens", "output_tokens", "num_completion_tokens"):
        value = meta.get(key)
        if value is not None:
            return int(value)
    return 0


def main(args):
    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print(f"Loading {args.dataset_name} split={args.split}...")
    dataset = load_dataset(args.dataset_name, split=args.split)
    examples = choose_examples(dataset, args.num_questions, args.seed)

    prompts = []
    labels = []
    raw_prompts = []
    option_counts = Counter()
    for item in examples:
        prompt, options, label = format_prompt(item)
        option_counts[len(options)] += 1
        raw_prompts.append(prompt)
        if tokenizer is not None:
            prompt = apply_chat_template(tokenizer, prompt, args.enable_thinking)
        prompts.append(prompt)
        labels.append(label)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "stop": ["Question:", "\n\nQuestion"],
    }

    print(
        f"Running {len(prompts)} MMLU-Pro requests against {args.base_url} "
        f"(parallelism={args.parallel}, option_counts={dict(option_counts)})..."
    )
    tic = time.perf_counter()
    results = asyncio.run(run_batch(args.base_url, prompts, sampling_params, args.parallel))
    latency = time.perf_counter() - tic

    preds = []
    rows = []
    total_completion_tokens = 0
    for i, (item, result) in enumerate(zip(examples, results)):
        text = normalize_response_text(result)
        total_completion_tokens += completion_tokens(result)
        valid_letters = set(LETTERS[: len(item["options"])])
        pred = extract_answer(text, valid_letters)
        preds.append(pred)
        rows.append(
            {
                "question_id": item.get("question_id"),
                "category": item.get("category"),
                "answer": labels[i],
                "prediction": pred,
                "correct": pred == labels[i],
                "invalid": pred == INVALID,
                "output": text,
                "meta_info": result.get("meta_info", {}),
            }
        )

    correct = sum(row["correct"] for row in rows)
    invalid = sum(row["invalid"] for row in rows)
    accuracy = correct / len(rows) if rows else 0.0
    invalid_rate = invalid / len(rows) if rows else 0.0
    output_throughput = total_completion_tokens / latency if latency > 0 else 0.0

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Invalid: {invalid_rate:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    if args.sample_file:
        sample_path = Path(args.sample_file)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_path.open("w") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Per-sample JSONL saved to {sample_path}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fout:
            for i, row in enumerate(rows):
                fout.write(f"=== Question {i} id={row['question_id']} category={row['category']} ===\n")
                fout.write(raw_prompts[i] + "\n")
                fout.write("=== Output ===\n")
                fout.write(row["output"] + "\n")
                fout.write(
                    f"=== Prediction: {row['prediction']}, Label: {row['answer']}, "
                    f"Correct: {row['correct']} ===\n\n"
                )
        print(f"Raw outputs saved to {output_path}")

    summary = {
        "task": "mmlu-pro",
        "backend": "sgl-jax",
        "dataset": args.dataset_name,
        "split": args.split,
        "latency": round(latency, 3),
        "accuracy": round(accuracy, 3),
        "invalid": round(invalid_rate, 3),
        "num_requests": len(rows),
        "output_throughput": round(output_throughput, 3),
        "other": {
            "parallel": args.parallel,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "option_counts": dict(option_counts),
        },
    }
    with open(args.result_file, "a") as fout:
        fout.write(json.dumps(summary) + "\n")
    print(f"Results appended to {args.result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMLU-Pro benchmark for sglang-jax")
    parser.add_argument("--base-url", type=str, default="http://localhost:30000")
    parser.add_argument("--dataset-name", type=str, default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--parallel", type=int, default=128)
    parser.add_argument("--result-file", type=str, default="mmlu_pro_results.jsonl")
    parser.add_argument("--sample-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    main(parser.parse_args())
