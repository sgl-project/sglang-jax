import random
import re

import eval.simple_eval_common as common
import pandas
from eval.simple_eval_common import Eval, EvalResult, SamplerBase, SingleEvalResult


class SglangMMLUEval(Eval):
    """
    Replicates the SGLang benchmark logic for MMLU:
    - Few-shot prompting (default 5 shots).
    - No Chain-of-Thought (direct answer).
    - Expects model to output just the answer letter (we set max_tokens=1 if possible).
    """

    def __init__(self, filename: str, num_examples: int | None, num_threads: int, n_shots: int = 5):
        df = pandas.read_csv(filename)
        self.n_shots = n_shots

        # Group by subject to get shots from the same subject
        from collections import defaultdict

        subject_examples = defaultdict(list)
        for _, row in df.iterrows():
            subject_examples[row["Subject"]].append(row.to_dict())

        self.shots = {}
        self.test_examples = []

        for subject, examples in subject_examples.items():
            if len(examples) >= n_shots and n_shots > 0:
                self.shots[subject] = examples[:n_shots]
                self.test_examples.extend(examples[n_shots:])
            else:
                self.shots[subject] = []
                self.test_examples.extend(examples)

        if num_examples:
            self.test_examples = random.Random(0).sample(
                self.test_examples, min(num_examples, len(self.test_examples))
            )

        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            subject = row["Subject"]
            shots = self.shots.get(subject, [])

            # Construct prompt identical to SGLang style
            prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
            for shot in shots:
                prompt += f"{shot['Question']}\n"
                prompt += f"A. {shot['A']}\nB. {shot['B']}\nC. {shot['C']}\nD. {shot['D']}\n"
                prompt += f"Answer: {shot['Answer']}\n\n"

            prompt += f"{row['Question']}\n"
            prompt += f"A. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n"
            prompt += "Answer:"

            # Use raw completions to bypass chat templates
            try:
                response = sampler.client.completions.create(
                    model=sampler.model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=1,
                )
                response_text = response.choices[0].text
            except Exception as e:
                # Fallback to chat completions if raw fails
                prompt_messages = [{"role": "user", "content": prompt}]
                response_text = sampler(prompt_messages)

            # Direct answer extraction: take the first non-whitespace character
            extracted_answer = response_text.strip()[0] if len(response_text.strip()) > 0 else None
            if extracted_answer:
                extracted_answer = extracted_answer.upper()

            score = 1.0 if extracted_answer == row["Answer"] else 0.0

            from eval.simple_eval_mmlu import subject2category

            category = subject2category.get(subject, "other")

            return SingleEvalResult(
                html=f"<p>Prompt: {prompt}</p><p>Response: {response_text}</p><p>Extracted: {extracted_answer}</p><p>Correct Answer: {row['Answer']}</p>",
                score=score,
                metrics={category: score},
                convo=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text},
                ],
            )

        results = common.map_with_progress(fn, self.test_examples, self.num_threads)
        return common.aggregate_results(results)
