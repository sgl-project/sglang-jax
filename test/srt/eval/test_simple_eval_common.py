# ruff: noqa: E402
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.simple_eval_common import ANSWER_PATTERN_MULTICHOICE, strip_reasoning
from run_eval import build_extra_body


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


if __name__ == "__main__":
    unittest.main()
