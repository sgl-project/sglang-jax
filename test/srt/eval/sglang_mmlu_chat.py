import math

import eval.simple_eval_common as common
from eval.sglang_mmlu import SglangMMLUEval
from eval.simple_eval_common import EvalResult, SamplerBase, SingleEvalResult
from eval.simple_eval_mmlu import subject2category

QUESTION_TEMPLATE = "{Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"


class SglangMMLUChatEval(SglangMMLUEval):
    """Non-thinking chat MMLU with choice scoring and the existing few-shot split."""

    def __call__(self, sampler: SamplerBase) -> EvalResult:
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
            prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
            for shot in self.shots.get(subject, []):
                prompt += QUESTION_TEMPLATE.format(**shot) + f" {shot['Answer']}\n\n"
            prompt += QUESTION_TEMPLATE.format(**row)

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
            # Chat templates already include any required special tokens.
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            choices = {}
            for letter in "ABCD":
                completed = tokenizer.encode(prompt + " " + letter, add_special_tokens=False)
                if completed[:-1] != input_ids:
                    raise ValueError(f"Answer {letter!r} must be a single-token continuation")
                choices[letter] = completed[-1]
            if len(set(choices.values())) != 4:
                raise ValueError("A/B/C/D must have distinct token IDs")

            response = sampler.client.post(
                f"{base_url}/generate",
                cast_to=dict[str, object],
                body={
                    "input_ids": input_ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                    "return_logprob": True,
                    "return_text_in_logprobs": False,
                    "token_ids_logprob": list(choices.values()),
                },
            )
            (token_scores,) = response["meta_info"]["output_token_ids_logprobs"]
            scores = {token_id: value for value, token_id, _ in token_scores}
            if any(scores.get(t) is None or not math.isfinite(scores[t]) for t in choices.values()):
                raise ValueError("Expected finite logprobs for all A/B/C/D token IDs")
            extracted_answer = max(choices, key=lambda letter: scores[choices[letter]])
            response_text = response["text"]

            score = 1.0 if extracted_answer == row["Answer"] else 0.0

            return SingleEvalResult(
                html=f"<p>Prompt: {prompt}</p><p>Response: {response_text}</p><p>Extracted: {extracted_answer}</p><p>Correct Answer: {row['Answer']}</p>",
                score=score,
                metrics={subject2category.get(subject, "other"): score},
                convo=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_text},
                ],
            )

        results = common.map_with_progress(fn, self.test_examples, self.num_threads)
        return common.aggregate_results(results)
