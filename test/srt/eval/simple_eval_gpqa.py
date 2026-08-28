# Adapted from https://github.com/openai/simple-evals/

"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import hashlib
import json
import os
import random
import re
import threading
from pathlib import Path

import eval.simple_eval_common as common
import pandas
from eval.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    format_multichoice_question,
    strip_reasoning,
)


class GPQAEval(Eval):
    def __init__(
        self,
        filename: str,
        num_examples: int | None,
        num_threads: int,
        n_repeats: int = 1,
        checkpoint_path: str | None = None,
        resume_unextracted_only: bool = False,
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
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.resume_unextracted_only = resume_unextracted_only
        self._checkpoint_lock = threading.Lock()

        if self.resume_unextracted_only and self.checkpoint_path is None:
            raise ValueError("resume_unextracted_only requires checkpoint_path")

    @staticmethod
    def _example_id(prompt_messages: list[dict]) -> str:
        prompt = json.dumps(prompt_messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _prepare_example(self, row: dict, sampler: SamplerBase) -> dict:
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
                content=format_multichoice_question(choices_dict), role="user"
            )
        ]
        return {
            "example_id": self._example_id(prompt_messages),
            "prompt_messages": prompt_messages,
            "correct_answer": correct_answer,
        }

    def _load_checkpoint(self) -> dict[str, dict]:
        records = {}
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            return records
        with self.checkpoint_path.open() as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid GPQA checkpoint line {line_number}: {error}"
                    ) from error
                if "example_id" not in record:
                    record["example_id"] = self._example_id(
                        record["prompt_messages"]
                    )
                records[record["example_id"]] = record
        return records

    def _append_checkpoint(self, record: dict) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._checkpoint_lock:
            with self.checkpoint_path.open("a") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())

    @staticmethod
    def _to_eval_result(prepared: dict, record: dict) -> SingleEvalResult:
        response_text = record.get("response_text", "")
        extracted_answer = record.get("extracted_answer")
        correct_answer = prepared["correct_answer"]
        score = 1.0 if extracted_answer == correct_answer else 0.0
        html = common.jinja_env.from_string(HTML_JINJA).render(
            prompt_messages=prepared["prompt_messages"],
            next_message=dict(content=response_text, role="assistant"),
            score=score,
            correct_answer=correct_answer,
            extracted_answer=extracted_answer,
        )
        convo = prepared["prompt_messages"] + [
            dict(content=response_text, role="assistant")
        ]
        return SingleEvalResult(
            html=html,
            score=score,
            convo=convo,
            metrics={
                "chars": int(record.get("chars", len(response_text))),
                "answer_extracted": 1.0 if extracted_answer is not None else 0.0,
            },
        )

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        prepared_examples = [self._prepare_example(row, sampler) for row in self.examples]
        checkpoint_records = self._load_checkpoint()

        def fn(prepared: dict):
            response_text = sampler(prepared["prompt_messages"])
            match = re.search(ANSWER_PATTERN_MULTICHOICE, strip_reasoning(response_text))
            extracted_answer = match.group(1) if match else None
            record = {
                "example_id": prepared["example_id"],
                "prompt_messages": prepared["prompt_messages"],
                "correct_answer": prepared["correct_answer"],
                "response_text": response_text,
                "extracted_answer": extracted_answer,
                "chars": len(response_text),
            }
            self._append_checkpoint(record)
            return record

        if self.resume_unextracted_only:
            pending = [
                prepared
                for prepared in prepared_examples
                if checkpoint_records.get(prepared["example_id"], {}).get(
                    "extracted_answer"
                )
                is None
            ]
        else:
            pending = prepared_examples

        if pending:
            new_records = common.map_with_progress(fn, pending, self.num_threads)
            checkpoint_records.update(
                {record["example_id"]: record for record in new_records}
            )

        missing = [
            prepared["example_id"]
            for prepared in prepared_examples
            if prepared["example_id"] not in checkpoint_records
        ]
        if missing:
            raise RuntimeError(f"GPQA checkpoint is missing {len(missing)} examples")

        results = [
            self._to_eval_result(prepared, checkpoint_records[prepared["example_id"]])
            for prepared in prepared_examples
        ]
        return common.aggregate_results(results)
