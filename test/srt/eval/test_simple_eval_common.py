# ruff: noqa: E402
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.sglang_mmlu import SglangMMLUEval
from eval.sglang_mmlu_chat import SglangMMLUChatEval
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


class TestSglangMMLUChat(unittest.TestCase):
    def setUp(self):
        row = dict(Subject="anatomy", Question="Q", A="a", B="b", C="c", D="d", Answer="B")
        self.client = Mock(base_url="http://localhost:32000/v1/")
        self.client.get.return_value = {"model_path": "model"}
        self.scores = [[-5.0, 6, None], [-2.0, 4, None], [-3.0, 3, None], [-4.0, 5, None]]
        self.meta = {"output_token_ids_logprobs": [self.scores]}
        self.client.post.return_value = {"text": " AD", "meta_info": self.meta}
        self.client.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(text=" AD")]
        )
        self.sampler = SimpleNamespace(model="model", client=self.client)
        # A real, offline tokenizer with BOS insertion and SentencePiece-style decoding.
        vocab = {t: i for i, t in enumerate(["<unk>", "<s>", "▁Answer:", "▁A", "▁B", "▁C", "▁D"])}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Metaspace()]
        )
        backend.decoder = decoders.Metaspace()
        backend.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", 1)]
        )
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="<s>",
            unk_token="<unk>",
            chat_template="{{ bos_token }}user\n{{ messages[0]['content'] }}\nassistant{{ '\\n' }}",
        )
        for target, value in (
            ("transformers.AutoTokenizer.from_pretrained", self.tokenizer),
            ("eval.sglang_mmlu.pandas.read_csv", pandas.DataFrame([row])),
        ):
            self.enterContext(patch(target, return_value=value))

    def evaluation(self):
        return SglangMMLUChatEval("unused.csv", None, 1)(self.sampler)

    def test_chat_prompt_uses_one_token_choice_scoring(self):
        raw = (
            "The following are multiple choice questions (with answers) about anatomy.\n\n"
            "Q\nA. a\nB. b\nC. c\nD. d\nAnswer:"
        )
        with patch.object(
            self.tokenizer, "apply_chat_template", wraps=self.tokenizer.apply_chat_template
        ) as format_prompt:
            result = self.evaluation()
        self.assertEqual(result.score, 1.0)  # B wins by logprob, not the emitted AD.
        user_prompt = (
            "Answer the final multiple-choice question with exactly one letter: "
            "A, B, C, or D.\n\n" + raw
        )
        format_prompt.assert_called_once_with(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt = "<s>user\n" + user_prompt + "\nassistant\nAnswer:"
        self.client.post.assert_called_once_with(
            "http://localhost:32000/generate",
            cast_to=dict[str, object],
            body={
                "input_ids": self.tokenizer.encode(prompt, add_special_tokens=False),
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "token_ids_logprob": [3, 4, 5, 6],
            },
        )
        self.client.completions.create.assert_not_called()
        input_ids = self.client.post.call_args.kwargs["body"]["input_ids"]
        self.assertEqual(self.tokenizer.encode(prompt)[:2], [1, 1])  # Old text path doubles BOS.
        self.assertEqual(input_ids.count(self.tokenizer.bos_token_id), 1)
        self.assertEqual(self.tokenizer.batch_decode([[i] for i in [3, 4, 5, 6]]), list("ABCD"))

    def test_request_and_template_errors_propagate(self):
        for method in ("post", "apply_chat_template"):
            target = self.client if method == "post" else self.tokenizer
            with patch.object(target, method, side_effect=RuntimeError("request failed")):
                with self.assertRaisesRegex(RuntimeError, "request failed"):
                    self.evaluation()

    def test_raw_evaluator_keeps_greedy_answer_extraction(self):
        result = SglangMMLUEval("unused.csv", None, 1)(self.sampler)
        request = self.client.completions.create.call_args.kwargs
        self.assertNotIn("logprobs", request)
        self.assertTrue(request["prompt"].endswith("Answer:"))
        self.assertEqual(result.score, 0.0)  # The emitted AD is scored as A.
        self.client.get.assert_not_called()
        self.client.post.assert_not_called()

    def test_incomplete_or_invalid_logprobs_fail(self):
        invalid = [[], [self.scores, self.scores], [[]], [self.scores[:-1]]]
        for value in (None, float("nan"), float("inf")):
            invalid.append([[[value, 6, None]] + self.scores[1:]])
        for scores in invalid:
            with self.subTest(scores=scores):
                self.meta["output_token_ids_logprobs"] = scores
                with self.assertRaises(ValueError):
                    self.evaluation()

    def test_answers_must_be_distinct_single_token_continuations(self):
        for completed in ([1, 2, 3, 4], [9, 7], [1, 2]):
            with self.subTest(completed=completed):
                # Multi-token continuation, changed prefix, or identical answer tokens.
                with patch.object(self.tokenizer, "encode", side_effect=[[1]] + [completed] * 4):
                    with self.assertRaises(ValueError):
                        self.evaluation()
                self.client.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
