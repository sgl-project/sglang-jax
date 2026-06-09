# Adapted from https://github.com/openai/simple-evals/

"""
AIME 2026 - American Invitational Mathematics Examination 2026
Dataset: MathArena/aime_2026
https://huggingface.co/datasets/MathArena/aime_2026

Logic ported from sglang `origin/brayden/fix-aime-26-eval` branch
(`python/sglang/test/simple_eval_aime26.py`): boxed-answer extraction with a
matharena strict_parsing=False fallback to the last bare integer, then
normalization to the 0..999 integer range.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval.simple_eval_common as common
from eval.simple_eval_common import (
    HTML_JINJA,
    ChatCompletionSampler,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    make_report,
)

QUERY_TEMPLATE = """Put your final answer within \\boxed{{}}.
The answer is an integer between 0 and 999 inclusive.

{question}"""


def normalize_aime_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = str(answer).strip()
    try:
        num = int(float(answer))
        if 0 <= num <= 999:
            return str(num)
    except (ValueError, TypeError):
        pass
    return answer


def extract_boxed_answer(text: str) -> str | None:
    """Return the content of the last \\boxed{...} or \\fbox{...} with balanced braces."""
    if not text:
        return None
    markers = ("\\boxed{", "\\fbox{")
    last_content: str | None = None
    i = 0
    while i < len(text):
        next_idx = -1
        next_marker_len = 0
        for marker in markers:
            j = text.find(marker, i)
            if j != -1 and (next_idx == -1 or j < next_idx):
                next_idx = j
                next_marker_len = len(marker)
        if next_idx == -1:
            break
        start = next_idx + next_marker_len
        depth = 1
        k = start
        while k < len(text) and depth > 0:
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            k += 1
        if depth == 0:
            last_content = text[start : k - 1]
            i = k
        else:
            break
    return last_content


def extract_last_integer(text: str) -> str | None:
    if not text:
        return None
    matches = re.findall(r"\b\d+\b", text)
    return matches[-1] if matches else None


def extract_aime_answer(text: str) -> str | None:
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer.strip()
    return extract_last_integer(text)


class AIME26Eval(Eval):
    def __init__(
        self,
        num_examples: int | None = None,
        num_threads: int = 64,
    ):
        from datasets import load_dataset

        dataset = load_dataset("MathArena/aime_2026", split="train")
        examples = [{"question": row["problem"], "answer": str(row["answer"])} for row in dataset]
        if num_examples:
            examples = examples[: min(num_examples, len(examples))]
        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=QUERY_TEMPLATE.format(question=row["question"]), role="user"
                )
            ]
            response_text = sampler(prompt_messages) or ""

            extracted_answer = extract_aime_answer(response_text)
            normalized_extracted = normalize_aime_answer(extracted_answer)
            normalized_correct = normalize_aime_answer(row["answer"])

            score = (
                1.0
                if normalized_extracted is not None and normalized_extracted == normalized_correct
                else 0.0
            )

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={"aime26": score},
            )

        results = common.map_with_progress(fn, self.examples, num_threads=self.num_threads)
        return common.aggregate_results(results)


def get_openai_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        base_url = args.base_url.rstrip("/")
    else:
        base_url = f"http://{args.host}:{args.port}"
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def build_chat_completion_sampler(args: argparse.Namespace) -> ChatCompletionSampler:
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k

    return ChatCompletionSampler(
        base_url=get_openai_base_url(args),
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        extra_body=extra_body or None,
    )


def run_eval(args: argparse.Namespace) -> dict[str, float]:
    common.set_ulimit()
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    eval_obj = AIME26Eval(args.num_examples, args.num_threads)
    sampler = build_chat_completion_sampler(args)

    tic = time.perf_counter()
    result = eval_obj(sampler)
    latency = time.perf_counter() - tic

    metrics = result.metrics | {"score": result.score}
    model_name = sampler.model.replace("/", "_")
    report_filename = f"/tmp/aime26_{model_name}.html"
    with open(report_filename, "w") as fh:
        fh.write(make_report(result))
    print(f"Writing report to {report_filename}")

    result_filename = f"/tmp/aime26_{model_name}.json"
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    print(metrics)
    print(f"Total latency: {latency:.3f} s")
    print(f"Score: {metrics['score']:.3f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AIME 2026 simple eval.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server base URL, with or without a trailing /v1.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
