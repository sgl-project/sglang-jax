import array
import sys
import types
import unittest
from unittest.mock import patch

from benchmark.glm52.prepare_longbench_v2 import (
    CODE_REPO_QA,
    FINANCIAL,
    BuildConfig,
    Candidate,
    _candidate_windows,
    _format_suffix,
    _load_tokenizer,
    _select_candidates,
)


class _WhitespaceTokenizer:
    name_or_path = "fake"
    vocab_size = 65_536
    model_max_length = 1_000_000

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [index + 1 for index, _ in enumerate(text.split())]


def _row(sub_domain=CODE_REPO_QA, source_id="source-1"):
    return {
        "_id": source_id,
        "domain": "domain",
        "sub_domain": sub_domain,
        "difficulty": "hard",
        "length": "long",
        "question": "What is the answer?",
        "choice_A": "one",
        "choice_B": "two",
        "choice_C": "three",
        "choice_D": "four",
        "answer": "A",
        "context": " ".join(f"token-{index}" for index in range(80)),
    }


def _candidate(sub_domain, priority, source_id):
    return Candidate(
        priority=priority,
        source_id=source_id,
        domain="domain",
        sub_domain=sub_domain,
        difficulty="hard",
        length_bucket="long",
        question="question",
        choices={"A": "a", "B": "b", "C": "c", "D": "d"},
        answer="A",
        window_index=0,
        context_token_start=0,
        context_token_end=4,
        source_context_tokens=4,
        suffix_tokens=2,
        source_context_sha256="0" * 64,
        input_ids=(1, 2, 3, 4, 5, 6),
    )


class PrepareLongBenchV2Test(unittest.TestCase):
    def test_load_tokenizer_uses_sgl_jax_runtime_loader(self):
        calls = []
        expected_tokenizer = object()

        def fake_get_tokenizer(tokenizer_path, **kwargs):
            calls.append((tokenizer_path, kwargs))
            return expected_tokenizer

        fake_module = types.ModuleType("sgl_jax.srt.hf_transformers_utils")
        fake_module.get_tokenizer = fake_get_tokenizer

        with patch.dict(
            sys.modules,
            {"sgl_jax.srt.hf_transformers_utils": fake_module},
        ):
            tokenizer = _load_tokenizer("/models/GLM5.2-fp8-channel-wise")

        self.assertIs(tokenizer, expected_tokenizer)
        self.assertEqual(
            calls,
            [
                (
                    "/models/GLM5.2-fp8-channel-wise",
                    {
                        "trust_remote_code": True,
                        "use_fast": True,
                        "local_files_only": True,
                    },
                )
            ],
        )

    def test_suffix_contains_question_choices_and_answer_marker(self):
        suffix = _format_suffix(_row())
        self.assertIn("Question: What is the answer?", suffix)
        self.assertIn("D. four", suffix)
        self.assertTrue(suffix.endswith("Answer:"))

    def test_windows_have_exact_prefix_plus_extend_length(self):
        config = BuildConfig(
            prefix_len=16,
            extend_len=16,
            output_len=8,
            code_quota=1,
            financial_quota=1,
        )
        candidates, reason = _candidate_windows(
            _row(), _WhitespaceTokenizer(), config
        )
        self.assertIsNone(reason)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(
            all(len(candidate.input_ids) == config.total_input_len for candidate in candidates)
        )
        self.assertTrue(
            all(
                isinstance(candidate.input_ids, array.array)
                and candidate.input_ids.typecode == "I"
                for candidate in candidates
            )
        )
        self.assertTrue(all(candidate.suffix_tokens <= config.extend_len for candidate in candidates))
        self.assertEqual(candidates[0].context_token_start, 0)
        self.assertEqual(
            candidates[1].context_token_start,
            candidates[0].context_token_end,
        )

    def test_context_shorter_than_tokenizer_budget_is_rejected(self):
        row = _row()
        row["context"] = "too short"
        candidates, reason = _candidate_windows(
            row,
            _WhitespaceTokenizer(),
            BuildConfig(prefix_len=16, extend_len=16),
        )
        self.assertEqual(candidates, [])
        self.assertEqual(reason, "context_too_short")

    def test_selection_is_balanced_and_uses_lowest_priority(self):
        candidates = [
            _candidate(CODE_REPO_QA, 30, "code-30"),
            _candidate(CODE_REPO_QA, 10, "code-10"),
            _candidate(CODE_REPO_QA, 20, "code-20"),
            _candidate(FINANCIAL, 50, "financial-50"),
            _candidate(FINANCIAL, 40, "financial-40"),
        ]
        selected = _select_candidates(
            candidates, {CODE_REPO_QA: 2, FINANCIAL: 1}
        )
        self.assertEqual(
            {(candidate.sub_domain, candidate.priority) for candidate in selected},
            {(CODE_REPO_QA, 10), (CODE_REPO_QA, 20), (FINANCIAL, 40)},
        )


if __name__ == "__main__":
    unittest.main()
