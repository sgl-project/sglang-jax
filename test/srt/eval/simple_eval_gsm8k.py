# Adapted from https://github.com/openai/simple-evals/

"""
GSM8K: Grade School Math 8K dataset.
Training Verifiers to Solve Math Word Problems
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
https://arxiv.org/abs/2110.14168
"""

import ast
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
INVALID = -9999999

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


def get_one_example(example: dict[str, str], include_answer: bool) -> str:
    prompt = f"Question: {example['question']}\nAnswer:"
    if include_answer:
        prompt += f" {example['answer']}"
    return prompt


def get_few_shot_examples(examples: list[dict[str, str]]) -> str:
    return "".join(get_one_example(example, include_answer=True) + "\n\n" for example in examples)


def get_answer_value(answer: str) -> int | float:
    numbers = re.findall(r"-?\d+\.?\d*", answer.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


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
        num_shots: int = 0,
    ):
        self._num_threads = num_threads
        self._num_shots = num_shots
        load_count = None if not num_examples else num_examples + num_shots
        examples = get_examples(load_count)
        self._few_shot_prompt = get_few_shot_examples(examples[:num_shots])
        self.examples = examples[num_shots:]
        if num_examples:
            self.examples = self.examples[:num_examples]

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(example: dict[str, str]):
            question = example["question"]
            if self._num_shots:
                correct_answer = get_answer_value(example["answer"])
                prompt = self._few_shot_prompt + get_one_example(example, include_answer=False)
            else:
                correct_answer = example["target"]
                prompt = INSTRUCTION_TEMPLATE.format(input=question)

            prompt_messages = [sampler._pack_message(content=prompt, role="user")]

            try:
                response_text = sampler(prompt_messages)
            except Exception:
                response_text = ""

            if self._num_shots:
                extracted_answer = get_answer_value(response_text)
                score = extracted_answer == correct_answer
            else:
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
