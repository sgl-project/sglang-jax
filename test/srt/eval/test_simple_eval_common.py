# ruff: noqa: E402
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    ChatCompletionSampler,
    CompletionSampler,
    strip_reasoning,
)
from eval.simple_eval_gsm8k import GSM8KEval, get_answer_value
from run_eval import build_extra_body, build_sampler, get_gsm8k_num_shots
from test_step3p5_mtp_e2e import (
    _DEFAULT_GSM8K_FLOOR,
    _DEFAULT_GSM8K_MAX_TOKENS,
    _GSM8K_MAX_TOKENS,
    _build_gsm8k_eval_args,
)


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(text="The final answer is 42")], usage=None)


class _EventuallySuccessfulCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= 6:
            raise RuntimeError("persistent failure")
        return SimpleNamespace(choices=[SimpleNamespace(text="too late")], usage=None)


class _RecordingSampler:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def _pack_message(self, role, content):
        return {"role": role, "content": content}

    def __call__(self, messages):
        self.prompts.append(messages[0]["content"])
        return next(self.responses)


def _extract_answer(response_text: str, *, strip: bool) -> str | None:
    # Mirror exactly what simple_eval_{gpqa,mmlu} do at answer-extraction time.
    text = strip_reasoning(response_text) if strip else response_text
    match = re.search(ANSWER_PATTERN_MULTICHOICE, text)
    return match.group(1) if match else None


class TestSimpleEvalCommon(unittest.TestCase):
    def test_full_think_block_must_be_stripped_before_extraction(self):
        # A reasoning model writes a mid-thought guess inside <think> and the
        # real answer after it. ANSWER_PATTERN_MULTICHOICE + re.search take the
        # FIRST "Answer: X", so without stripping the trace the scratch-work
        # letter ("A") wins and a correct ("C") response is scored wrong.
        response = (
            "<think>\n"
            "First glance suggests Answer: A from the surface reading.\n"
            "Reconsidering the kinetics, that is wrong - it must be C.\n"
            "</think>\n"
            "Answer: C"
        )
        self.assertEqual(_extract_answer(response, strip=False), "A")
        self.assertEqual(_extract_answer(response, strip=True), "C")

    def test_bare_closing_think_trace_must_be_stripped_before_extraction(self):
        # When the opening <think> is dropped (tokenizer/template), everything
        # up to </think> is reasoning; the only "Answer:" before it is the
        # scratch guess ("B"), not the final answer ("D").
        response = "Working it out, Answer: B seems plausible.</think>\nAnswer: D"
        self.assertEqual(_extract_answer(response, strip=False), "B")
        self.assertEqual(_extract_answer(response, strip=True), "D")

    def test_extraction_unaffected_when_no_reasoning_tags_present(self):
        # Most eval outputs have no <think> tag; stripping must not touch them.
        self.assertEqual(strip_reasoning("Answer: D"), "Answer: D")
        self.assertEqual(strip_reasoning(""), "")
        self.assertEqual(_extract_answer("Answer: D", strip=True), "D")

    def test_build_extra_body_routes_sglang_params_and_chat_template(self):
        # SGLang-only sampling params + chat_template_kwargs ride in extra_body;
        # the OpenAI client merges them into the request body top level. top_p is
        # a native kwarg handled by the sampler, so it must NOT leak into here.
        args = SimpleNamespace(
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            frequency_penalty=0.5,
            repetition_penalty=1.0,
            seed=17,
            chat_template_kwargs={"enable_thinking": True},
        )

        extra_body = build_extra_body(args)

        self.assertEqual(
            extra_body,
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "repetition_penalty": 1.0,
                "frequency_penalty": 0.5,
                "seed": 17,
            },
        )
        self.assertNotIn("top_p", extra_body)

    def test_build_extra_body_is_none_when_nothing_set(self):
        self.assertIsNone(build_extra_body(SimpleNamespace()))
        self.assertIsNone(
            build_extra_body(SimpleNamespace(top_p=0.9, temperature=0.7, max_tokens=64))
        )

    def test_build_extra_body_keeps_chat_template_only(self):
        args = SimpleNamespace(chat_template_kwargs={"enable_thinking": False})

        self.assertEqual(
            build_extra_body(args),
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "EMPTY"})
    def test_build_sampler_selects_completion_and_forwards_raw_prompt(self):
        args = SimpleNamespace(
            api="completion",
            model="test-model",
            temperature=0.0,
            max_tokens=512,
        )

        sampler = build_sampler(args, "http://127.0.0.1:30000/v1")
        fake_completions = _FakeCompletions()
        sampler.client = SimpleNamespace(completions=fake_completions)
        text = sampler([{"role": "user", "content": "Question: 6 * 7\nAnswer:"}])

        self.assertIsInstance(sampler, CompletionSampler)
        self.assertEqual(text, "The final answer is 42")
        self.assertEqual(
            fake_completions.kwargs,
            {
                "model": "test-model",
                "prompt": "Question: 6 * 7\nAnswer:",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 512,
                "stop": ["Question", "Assistant:", "<|separator|>"],
            },
        )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "EMPTY"})
    def test_build_sampler_defaults_to_chat_and_rejects_unknown_api(self):
        args = SimpleNamespace(model="test-model", max_tokens=None)
        sampler = build_sampler(args, "http://127.0.0.1:30000/v1")
        self.assertIsInstance(sampler, ChatCompletionSampler)
        self.assertEqual(sampler.max_tokens, 2048)

        args.api = "native"
        with self.assertRaisesRegex(ValueError, "Unsupported eval API"):
            build_sampler(args, "http://127.0.0.1:30000/v1")

    @patch("eval.simple_eval_common.time.sleep")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "EMPTY"})
    def test_completion_sampler_stops_after_six_transport_failures(self, sleep):
        sampler = build_sampler(
            SimpleNamespace(api="completion", model="test-model"),
            "http://127.0.0.1:30000/v1",
        )
        completions = _EventuallySuccessfulCompletions()
        sampler.client = SimpleNamespace(completions=completions)

        self.assertEqual(sampler([{"role": "user", "content": "prompt"}]), "")
        self.assertEqual(completions.calls, 6)
        self.assertEqual(sleep.call_count, 5)

    def test_gsm8k_few_shot_is_only_enabled_for_completion_api(self):
        self.assertEqual(get_gsm8k_num_shots(SimpleNamespace(num_shots=5)), 0)
        self.assertEqual(
            get_gsm8k_num_shots(SimpleNamespace(api="completion", num_shots=5)),
            5,
        )

    def test_gsm8k_five_shot_uses_held_out_examples_and_last_number(self):
        examples = [
            {
                "question": f"Question {i}",
                "answer": f"reasoning {i} #### {i}",
                "target": str(i),
            }
            for i in range(1, 8)
        ]
        sampler = _RecordingSampler(
            ["A misleading intermediate value is 100; final answer 6", "work 3 then 7"]
        )

        with patch("eval.simple_eval_gsm8k.get_examples", return_value=examples) as load:
            result = GSM8KEval(num_examples=2, num_threads=1, num_shots=5)(sampler)

        load.assert_called_once_with(7)
        self.assertEqual(result.score, 1.0)
        self.assertEqual(len(sampler.prompts), 2)
        self.assertEqual(sampler.prompts[0].count("Question:"), 6)
        self.assertIn("Question: Question 5\nAnswer: reasoning 5 #### 5", sampler.prompts[0])
        self.assertTrue(sampler.prompts[0].endswith("Question: Question 6\nAnswer:"))
        self.assertNotIn("Question: Question 7\nAnswer:", sampler.prompts[0])
        self.assertEqual(get_answer_value("first 100, then -42.5"), -42.5)

    def test_gsm8k_zero_shot_keeps_chat_instruction(self):
        examples = [{"question": "What is 6 * 7?", "answer": "#### 42", "target": "42"}]
        sampler = _RecordingSampler(["Answer: 42"])

        with patch("eval.simple_eval_gsm8k.get_examples", return_value=examples):
            result = GSM8KEval(num_examples=1, num_threads=1)(sampler)

        self.assertEqual(result.score, 1.0)
        self.assertTrue(sampler.prompts[0].startswith("Solve this math problem."))
        self.assertNotIn("Question:", sampler.prompts[0])

    def test_gsm8k_zero_num_examples_preserves_all_examples(self):
        examples = [
            {"question": f"Question {i}", "answer": f"#### {i}", "target": str(i)}
            for i in range(1, 8)
        ]

        with patch("eval.simple_eval_gsm8k.get_examples", return_value=examples) as load:
            eval_obj = GSM8KEval(num_examples=0, num_threads=1, num_shots=5)

        load.assert_called_once_with(None)
        self.assertEqual(len(eval_obj.examples), 2)

    def test_step3p5_gsm8k_uses_upstream_completion_task(self):
        args = _build_gsm8k_eval_args()

        self.assertEqual(args.api, "completion")
        self.assertEqual(args.num_shots, 5)
        self.assertEqual(args.max_tokens, _GSM8K_MAX_TOKENS)
        self.assertEqual(_DEFAULT_GSM8K_MAX_TOKENS, 512)
        self.assertEqual(_DEFAULT_GSM8K_FLOOR, 0.83)


if __name__ == "__main__":
    unittest.main()
