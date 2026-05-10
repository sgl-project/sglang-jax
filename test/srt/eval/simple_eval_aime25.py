# Adapted from https://github.com/openai/simple-evals/ in sglang-jax

import json
import re
import urllib.request
import os

import eval.simple_eval_common as common
from eval.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

AIME_I_URL = "https://modelscope.cn/datasets/opencompass/AIME2025/resolve/master/aime2025-I.jsonl"
AIME_II_URL = "https://modelscope.cn/datasets/opencompass/AIME2025/resolve/master/aime2025-II.jsonl"

INSTRUCTION_TEMPLATE = """Solve the following math problem step by step. Put your answer inside \\boxed{{}}.
{question}
Remember to put your answer inside \\boxed{{}}."""


def parse_answer(answer: str) -> str:
    """Extract the answer inside \\boxed{}."""
    # Look for the last \boxed{} in the text
    matches = re.findall(r"\\boxed\{([^{}]*)\}", answer)
    if matches:
        return matches[-1].strip()
    return ""


def score_aime(target: str, prediction: str) -> bool:
    """Compare target and prediction answers."""
    # Normalize: remove white spaces and commas
    target = target.strip().replace(" ", "").replace(",", "")
    prediction = prediction.strip().replace(" ", "").replace(",", "")
    return target == prediction


def get_examples(num_examples: int | None = None) -> list[dict[str, str]]:
    """Load AIME2025 test examples from ModelScope resolve URLs."""
    examples = []
    urls = [AIME_I_URL, AIME_II_URL]

    for url in urls:
        try:
            with urllib.request.urlopen(url) as f:
                for line in f.read().decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    examples.append(
                        {
                            "question": data["question"],
                            "answer": data["answer"],
                        }
                    )
                    if num_examples and len(examples) >= num_examples:
                        return examples
        except Exception as e:
            print(f"Error loading from {url}: {e}")

    return examples


class AIME25Eval(Eval):
    def __init__(
        self,
        num_examples: int | None = None,
        num_threads: int = 64,
    ):
        self._num_threads = num_threads
        self.examples = get_examples(num_examples)

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(example: dict[str, str]):
            question = example["question"]
            correct_answer = example["answer"]

            prompt_messages = [
                sampler._pack_message(
                    content=INSTRUCTION_TEMPLATE.format(question=question), role="user"
                )
            ]

            try:
                response_text = sampler(prompt_messages)
            except Exception as e:
                print(f"Error in sampling: {e}")
                response_text = ""

            extracted_answer = parse_answer(response_text)
            score = score_aime(correct_answer, extracted_answer)

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
                metrics={"aime25": score},
            )

        results = common.map_with_progress(fn, self.examples, num_threads=self._num_threads)
        return common.aggregate_results(results, default_stats=("mean", "std"))
