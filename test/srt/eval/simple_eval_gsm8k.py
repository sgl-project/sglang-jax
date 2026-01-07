# Adapted from https://github.com/openai/simple-evals/

"""
GSM8K: Grade School Math 8K dataset.
Training Verifiers to Solve Math Word Problems
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
https://arxiv.org/abs/2110.14168
"""

import json
import re
import urllib.request

import eval.simple_eval_common as common
from eval.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

INSTRUCTION_TEMPLATE = """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}"""


def parse_answer(answer: str) -> str:
    """Extract the numeric answer from the response."""
    if "Answer:" not in answer and "answer:" not in answer:
        return ""

    # Get text after "Answer:" (case insensitive)
    answer_text = re.split(r"[Aa]nswer\s*:", answer)[-1].strip()

    # Find all numbers (including negative and decimals) in the string
    numbers = re.findall(r"-?\d+\.?\d*", answer_text.replace(",", ""))

    # Return the first number, or empty string if no numbers found
    return numbers[0].rstrip(".") if numbers else ""


def extract_answer_from_target(target: str) -> str:
    """Extract the numeric answer from GSM8K target format (#### followed by number)."""
    match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", target)
    if match:
        return match.group(1).replace(",", "")
    return ""


def score_gsm8k(target: str, prediction: str) -> bool:
    """Compare target and prediction answers."""
    # Normalize: remove trailing zeros after decimal, remove commas
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")
    if "." in target:
        target = target.rstrip("0").rstrip(".")

    target = target.replace(",", "")
    prediction = prediction.replace(",", "")

    return target == prediction


def get_examples(num_examples: int | None = None) -> list[dict[str, str]]:
    """Load GSM8K test examples."""
    examples = []
    with urllib.request.urlopen(GSM8K_URL) as f:
        for line in f.read().decode("utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            question = data["question"]
            answer = data["answer"]
            target = extract_answer_from_target(answer)
            examples.append(
                {
                    "question": question,
                    "answer": answer,
                    "target": target,
                }
            )
            if num_examples and len(examples) >= num_examples:
                break
    return examples


class GSM8KEval(Eval):
    def __init__(
        self,
        num_examples: int | None = None,  # None means use all examples
        num_threads: int = 64,
    ):
        self._num_threads = num_threads
        self.examples = get_examples(num_examples)

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(example: dict[str, str]):
            question = example["question"]
            correct_answer = example["target"]

            prompt_messages = [
                sampler._pack_message(
                    content=INSTRUCTION_TEMPLATE.format(input=question), role="user"
                )
            ]

            try:
                response_text = sampler(prompt_messages)
            except Exception:
                response_text = ""

            extracted_answer = parse_answer(response_text)
            score = score_gsm8k(correct_answer, extracted_answer)

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={"gsm8k": score},
            )

        results = common.map_with_progress(fn, self.examples, num_threads=self._num_threads)
        return common.aggregate_results(results, default_stats=("mean", "std"))
