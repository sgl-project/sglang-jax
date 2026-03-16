import argparse
import random
import time
from dataclasses import dataclass

import pandas as pd
import requests
from transformers import AutoTokenizer

from test.srt.eval.simple_eval_gpqa import GPQA_SYSTEM_MESSAGE_SUFFIX, format_gpqa_question


SYSTEM_PROMPT = (
    "You are MiMo, an AI assistant developed by Xiaomi. "
    "Today is March 12, 2026 Wednesday. "
    "Your knowledge cutoff date is December 2024.\n\n"
    + GPQA_SYSTEM_MESSAGE_SUFFIX
)


@dataclass
class PromptCase:
    name: str
    user_text: str
    prompt_tokens: int


def build_tokenizer():
    return AutoTokenizer.from_pretrained("/models/MiMo-V2-Flash", trust_remote_code=True)


def build_client(base_url: str, timeout: float) -> dict:
    session = requests.Session()
    session.trust_env = False
    return {"base_url": base_url.rstrip("/"), "timeout": timeout, "session": session}


def chat_ids(tokenizer, system_prompt: str, user_text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def synthetic_text(prefix_repeats: int, tail_word: str, tail_repeats: int) -> str:
    prefix = ("common " * prefix_repeats).strip()
    tail = ((tail_word + " ") * tail_repeats).strip()
    return (prefix + " " + tail).strip()


def search_synthetic(tokenizer, target_shared: int, target_total: int, limit: int) -> None:
    prefix_candidates = []
    for prefix_repeats in range(40, 260):
        primer_ids = chat_ids(tokenizer, SYSTEM_PROMPT, synthetic_text(prefix_repeats, "primer", 1))
        target_ids = chat_ids(tokenizer, SYSTEM_PROMPT, synthetic_text(prefix_repeats, "target", 1))
        shared = common_prefix_len(primer_ids, target_ids)
        if abs(shared - target_shared) <= 16:
            prefix_candidates.append((prefix_repeats, shared))

    scored = []
    for prefix_repeats, shared in prefix_candidates:
        primer_lengths = []
        for primer_repeats in range(1, 32):
            primer_ids = chat_ids(tokenizer, SYSTEM_PROMPT, synthetic_text(prefix_repeats, "primer", primer_repeats))
            primer_lengths.append((primer_repeats, len(primer_ids)))

        target_lengths = []
        for target_repeats in range(220, 620):
            target_ids = chat_ids(tokenizer, SYSTEM_PROMPT, synthetic_text(prefix_repeats, "target", target_repeats))
            target_lengths.append((target_repeats, len(target_ids)))

        for primer_repeats, primer_len in primer_lengths:
            for target_repeats, target_len in target_lengths:
                if target_len < target_total - 40:
                    continue
                score = (
                    abs(shared - target_shared)
                    + abs(target_len - target_total) * 0.2
                    + abs(primer_len - 348) * 0.05
                )
                scored.append(
                    (
                        score,
                        prefix_repeats,
                        primer_repeats,
                        target_repeats,
                        primer_len,
                        target_len,
                        shared,
                    )
                )
    scored.sort()
    for row in scored[:limit]:
        print(row)


def request_once(client: dict, model: str, user_text: str, max_tokens: int, temperature: float, top_p: float):
    started = time.time()
    print({"phase": "request_start", "user_chars": len(user_text), "max_tokens": max_tokens}, flush=True)
    resp = client["session"].post(
        client["base_url"] + "/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        timeout=client["timeout"],
    )
    resp.raise_for_status()
    elapsed = time.time() - started
    payload = resp.json()
    text = payload["choices"][0]["message"]["content"]
    return elapsed, text


def run_synthetic_repro(
    tokenizer,
    client: dict,
    model: str,
    prefix_repeats: int,
    primer_repeats: int,
    target_repeats: int,
    max_tokens: int,
) -> None:
    primer_text = synthetic_text(prefix_repeats, "primer", primer_repeats)
    target_text = synthetic_text(prefix_repeats, "target", target_repeats)
    primer_ids = chat_ids(tokenizer, SYSTEM_PROMPT, primer_text)
    target_ids = chat_ids(tokenizer, SYSTEM_PROMPT, target_text)
    print(
        {
            "primer_tokens": len(primer_ids),
            "target_tokens": len(target_ids),
            "shared_prefix": common_prefix_len(primer_ids, target_ids),
            "primer_repeats": primer_repeats,
            "target_repeats": target_repeats,
            "prefix_repeats": prefix_repeats,
        }
    )
    elapsed, text = request_once(client, model, primer_text, max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    print({"phase": "primer", "elapsed_s": round(elapsed, 3), "text": (text or "")[:120]})
    elapsed, text = request_once(client, model, target_text, max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    print({"phase": "target", "elapsed_s": round(elapsed, 3), "text": (text or "")[:120]})


def load_gpqa_cases(tokenizer) -> list[PromptCase]:
    examples = pd.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv")
    rng = random.Random(0)
    records = [row.to_dict() for _, row in examples.iterrows()]
    records = rng.sample(records, 10)
    records = [record | {"permutation": rng.sample(range(4), 4)} for record in records]
    cases = []
    for i, example in enumerate(records[:10], start=1):
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        choices = [choices[j] for j in example["permutation"]]
        row = {
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "Question": example["Question"],
        }
        user_text = format_gpqa_question(row)
        ids = chat_ids(tokenizer, SYSTEM_PROMPT, user_text)
        cases.append(PromptCase(name=f"sample_{i}", user_text=user_text, prompt_tokens=len(ids)))
    return cases


def describe_gpqa_prefixes(tokenizer) -> None:
    cases = load_gpqa_cases(tokenizer)
    prev_ids: list[list[int]] = []
    for case in cases:
        ids = chat_ids(tokenizer, SYSTEM_PROMPT, case.user_text)
        shared = 0
        if prev_ids:
            shared = max(common_prefix_len(ids, other) for other in prev_ids)
        print({"name": case.name, "prompt_tokens": len(ids), "max_shared_prefix": shared})
        prev_ids.append(ids)


def run_gpqa_subset(
    tokenizer,
    client: dict,
    model: str,
    indices: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    cases = load_gpqa_cases(tokenizer)
    seen_ids: list[list[int]] = []
    for idx in indices:
        case = cases[idx - 1]
        ids = chat_ids(tokenizer, SYSTEM_PROMPT, case.user_text)
        shared = 0
        if seen_ids:
            shared = max(common_prefix_len(ids, other) for other in seen_ids)
        print(
            {
                "name": case.name,
                "sample_index": idx,
                "prompt_tokens": len(ids),
                "max_shared_prefix": shared,
            }
        )
        elapsed, text = request_once(
            client,
            model,
            case.user_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        print(
            {
                "name": case.name,
                "phase": "done",
                "elapsed_s": round(elapsed, 3),
                "text": (text or "")[:120],
            }
        )
        seen_ids.append(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["search-synthetic", "run-synthetic", "describe-gpqa", "run-gpqa-subset"],
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30271/v1")
    parser.add_argument("--model", default="/models/MiMo-V2-Flash")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--target-shared", type=int, default=186)
    parser.add_argument("--target-total", type=int, default=659)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--prefix-repeats", type=int, default=61)
    parser.add_argument("--primer-repeats", type=int, default=2)
    parser.add_argument("--target-repeats", type=int, default=229)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--indices", default="2,9")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    tokenizer = build_tokenizer()

    if args.mode == "search-synthetic":
        search_synthetic(tokenizer, args.target_shared, args.target_total, args.limit)
        return
    if args.mode == "describe-gpqa":
        describe_gpqa_prefixes(tokenizer)
        return
    if args.mode == "run-gpqa-subset":
        client = build_client(args.base_url, args.timeout)
        indices = [int(x) for x in args.indices.split(",") if x]
        run_gpqa_subset(
            tokenizer,
            client,
            args.model,
            indices,
            args.max_tokens,
            args.temperature,
            args.top_p,
        )
        return

    client = build_client(args.base_url, args.timeout)
    run_synthetic_repro(
        tokenizer,
        client,
        args.model,
        args.prefix_repeats,
        args.primer_repeats,
        args.target_repeats,
        args.max_tokens,
    )


if __name__ == "__main__":
    main()
