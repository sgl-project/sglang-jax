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

import re

import eval.simple_eval_common as common
from eval.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
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
