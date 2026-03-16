# Adapted from https://github.com/openai/simple-evals/

"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re

import eval.simple_eval_common as common
import pandas
from eval.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

GPQA_QUERY_TEMPLATE = """
You will be given one multiple choice question.
Your entire visible response must be exactly one line in the format:
Answer: $LETTER
where LETTER is one of ABCD.
Do not include any explanation, reasoning, markdown, or extra text.

Example:
Question: What is 2 + 2?
A) 3
B) 4
C) 5
D) 6
Answer: B

Now answer this question:

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

GPQA_SYSTEM_MESSAGE_SUFFIX = """
For multiple choice questions, your entire visible response must be exactly one line in the format:
Answer: X
where X is one of A, B, C, or D.
Do not output reasoning, explanations, markdown, or any extra text.
""".strip()

GPQA_ANSWER_PATTERNS = (
    r"(?i)Answer\s*:\s*([A-D])",
    r"(?i)(?:final|correct)?\s*answer\s*(?:is|:)\s*[*_(\\[]*([A-D])",
    r"(?i)(?:choose|choice|option)\s*(?:is|:)?\s*[*_(\\[]*([A-D])",
    r"(?i)\bletter\s*([A-D])\b",
)


def format_gpqa_question(row: dict) -> str:
    return GPQA_QUERY_TEMPLATE.format(**row)


def extract_gpqa_answer(response_text: str) -> str | None:
    for pattern in GPQA_ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, response_text))
        if matches:
            return matches[-1].group(1).upper()

    for line in reversed(response_text.splitlines()):
        stripped = re.sub(r"[*_`\\s]", "", line)
        if stripped in {"A", "B", "C", "D"}:
            return stripped

    return None


class GPQAEval(Eval):
    def __init__(
        self,
        filename: str,
        num_examples: int | None,
        num_threads: int,
        n_repeats: int = 1,
    ):
        df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.n_repeats = n_repeats
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0],
                B=choices[1],
                C=choices[2],
                D=choices[3],
                Question=row["Question"],
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_gpqa_question(choices_dict), role="user"
                )
            ]
            response_text = sampler(prompt_messages)
            extracted_answer = extract_gpqa_answer(response_text)
            score = 1.0 if extracted_answer == correct_answer else 0.0
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
                metrics={"chars": len(response_text)},
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
