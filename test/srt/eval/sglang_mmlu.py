import math
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
    - Scores the next-token probabilities of the four answer letters.
    - Optionally wraps the prompt in a non-thinking chat template.
    """

    def __init__(
        self,
        filename: str,
        num_examples: int | None,
        num_threads: int,
        n_shots: int = 5,
        use_chat_template: bool = False,
    ):
        df = pandas.read_csv(filename)
        self.n_shots = n_shots
        self.use_chat_template = use_chat_template

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
        if self.use_chat_template:
            from transformers import AutoTokenizer

            base_url = str(sampler.client.base_url).rstrip("/").removesuffix("/v1")
            config = sampler.client.get(f"{base_url}/get_server_info", cast_to=dict[str, object])
            tokenizer = AutoTokenizer.from_pretrained(
                config.get("tokenizer_path") or config["model_path"],
                revision=config.get("revision"),
                use_fast=config.get("tokenizer_mode") != "slow",
            )

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

            if self.use_chat_template:
                instruction = (
                    "Answer the final multiple-choice question with exactly one letter: "
                    "A, B, C, or D."
                )
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction + "\n\n" + prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt += "Answer:"
            response = sampler.client.completions.create(
                model=sampler.model,
                prompt=prompt,
                temperature=0,
                max_tokens=1,
                logprobs=20,
            )
            response_text = response.choices[0].text
            (scores,) = response.choices[0].logprobs.top_logprobs
            if not scores or any(v is None or not math.isfinite(v) for v in scores.values()):
                raise ValueError("Expected finite next-token logprobs")
            choices = {letter: scores[" " + letter] for letter in "ABCD" if " " + letter in scores}
            # Missing choices cannot beat a returned choice above the top-k cutoff.
            if not choices or (len(choices) < 4 and max(choices.values()) <= min(scores.values())):
                raise ValueError("Top logprobs do not identify the best A/B/C/D answer")
            extracted_answer = max(choices, key=choices.get)

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
