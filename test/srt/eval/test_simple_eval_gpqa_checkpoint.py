import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval.simple_eval_gpqa import GPQAEval


class _Sampler:
    def __init__(self, leave_unextracted_question=None):
        self.leave_unextracted_question = leave_unextracted_question
        self.calls = []

    def _pack_message(self, role, content):
        return {"role": role, "content": content}

    def __call__(self, messages):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        if (
            self.leave_unextracted_question
            and self.leave_unextracted_question in prompt
        ):
            return "Still thinking"
        for letter in "ABCD":
            if f"{letter}) correct-" in prompt:
                return f"Answer: {letter}"
        raise AssertionError("correct choice missing from prompt")


class TestGPQACheckpoint(unittest.TestCase):
    def test_resume_runs_only_unextracted_and_merges_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "gpqa.csv"
            checkpoint = root / "checkpoint.jsonl"
            with dataset.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "Question",
                        "Correct Answer",
                        "Incorrect Answer 1",
                        "Incorrect Answer 2",
                        "Incorrect Answer 3",
                    ],
                )
                writer.writeheader()
                for index in range(2):
                    writer.writerow(
                        {
                            "Question": f"question-{index}",
                            "Correct Answer": f"correct-{index}",
                            "Incorrect Answer 1": f"wrong-{index}-1",
                            "Incorrect Answer 2": f"wrong-{index}-2",
                            "Incorrect Answer 3": f"wrong-{index}-3",
                        }
                    )

            first_sampler = _Sampler(leave_unextracted_question="question-0")
            first = GPQAEval(
                str(dataset),
                num_examples=2,
                num_threads=1,
                checkpoint_path=str(checkpoint),
            )(first_sampler)
            self.assertEqual(len(first_sampler.calls), 2)
            self.assertEqual(first.metrics["answer_extracted"], 0.5)

            second_sampler = _Sampler()
            second = GPQAEval(
                str(dataset),
                num_examples=2,
                num_threads=1,
                checkpoint_path=str(checkpoint),
                resume_unextracted_only=True,
            )(second_sampler)
            self.assertEqual(len(second_sampler.calls), 1)
            self.assertIn("question-0", second_sampler.calls[0])
            self.assertEqual(second.metrics["answer_extracted"], 1.0)
            self.assertEqual(second.score, 1.0)

            records = [json.loads(line) for line in checkpoint.read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertEqual(len({record["example_id"] for record in records}), 2)


if __name__ == "__main__":
    unittest.main()
