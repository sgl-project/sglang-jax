import math

import eval.simple_eval_common as common
from eval.sglang_mmlu import SglangMMLUEval
from eval.simple_eval_common import EvalResult, SamplerBase, SingleEvalResult


class SglangMMLUChatEval(SglangMMLUEval):
    """Non-thinking chat MMLU with next-token A/B/C/D choice scoring.

    Reuses the raw evaluator's subject split and few-shot selection.
    """

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
