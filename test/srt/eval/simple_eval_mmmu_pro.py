"""
MMMU-Pro evaluation for VLMs using the run_eval simple-evals interface.
"""

from __future__ import annotations

import ast
import base64
import io
import math
import os
import random
import re
from typing import Any

import eval.simple_eval_common as common
from datasets import load_dataset
from eval.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    strip_reasoning,
)
from PIL import Image

_DOMAIN_PREFIX = "domain:"
_CATEGORY_PREFIX = "category:"


DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

_SUB_CAT2DOMAIN = {
    sub_cat: domain for domain, sub_cats in DOMAIN_CAT2SUB_CAT.items() for sub_cat in sub_cats
}


PROMPTS = {
    "cot": {
        "vision": (
            "Write out the multiple-choice question in the image and then solve it. "
            "The last line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
            "Think step by step before answering."
        ),
        "standard": (
            "Answer the preceding multiple choice question. The last line of your "
            "response should be of the following format: 'Answer: $LETTER' "
            "(without quotes) where LETTER is one of options. Think step by step "
            "before answering."
        ),
    },
    "direct": {
        "vision": (
            "Answer with the option letter from the given choices directly. The last "
            "line of your response should be of the following format: "
            "'Answer: $LETTER' (without quotes) where LETTER is one of options."
        ),
        "standard": "Answer with the option letter from the given choices directly.",
    },
}


def _parse_options(options: Any) -> list[str]:
    if isinstance(options, list):
        return [str(option) for option in options]
    return [str(option) for option in ast.literal_eval(str(options))]


def _build_mc_mapping(options: list[str]) -> tuple[dict[str, str], list[str]]:
    index2ans = {}
    all_choices = []
    for idx, option in enumerate(options):
        letter = chr(ord("A") + idx)
        index2ans[letter] = option
        all_choices.append(letter)
    return index2ans, all_choices


def _replace_image_tokens(text: str) -> tuple[str, list[int]]:
    image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", text)]
    return re.sub(r"<image\s+\d+>", "<image>", text), image_order


def _split_prompt_for_image(prompt: str) -> tuple[str, str]:
    if "<" in prompt and ">" in prompt:
        return prompt.split("<", 1)[0], prompt.split(">", 1)[1]
    return prompt, ""


def _extract_subset_name(sample_id: str) -> str:
    split = sample_id.split("_")[0]
    match = re.search(rf"^{split}_(.+?)_\d+$", sample_id)
    return match.group(1) if match else "Unknown"


def _cap_image(image: Image.Image, cap_px: int) -> Image.Image:
    image = image.convert("RGB")
    if cap_px <= 0:
        return image
    width, height = image.size
    if width * height <= cap_px:
        return image
    scale = math.sqrt(cap_px / (width * height))
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BICUBIC,
    )


def _image_to_data_uri(image: Image.Image, cap_px: int) -> str:
    buffer = io.BytesIO()
    _cap_image(image, cap_px).save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _build_chat_messages(
    prompt: str, images: list[Image.Image], cap_px: int
) -> list[dict[str, Any]]:
    segments = prompt.split("<image>")
    content: list[dict[str, Any]] = []

    if len(segments) - 1 == len(images):
        for idx, segment in enumerate(segments):
            if segment:
                content.append({"type": "text", "text": segment})
            if idx < len(images):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_to_data_uri(images[idx], cap_px)},
                    }
                )
    elif len(images) == 1:
        prefix, suffix = _split_prompt_for_image(prompt)
        if prefix:
            content.append({"type": "text", "text": prefix})
        content.append(
            {"type": "image_url", "image_url": {"url": _image_to_data_uri(images[0], cap_px)}}
        )
        if suffix:
            content.append({"type": "text", "text": suffix})
    else:
        content.append({"type": "text", "text": prompt})
        for image in images:
            content.append(
                {"type": "image_url", "image_url": {"url": _image_to_data_uri(image, cap_px)}}
            )

    return [{"role": "user", "content": content}]


def _parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
    rng: random.Random | None = None,
) -> str:
    response = strip_reasoning(response or "")

    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching_options = [choice for choice in all_choices if choice in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        for index, answer in index2ans.items():
            if answer and answer.lower() in response.lower():
                candidates.append(index)

    if not candidates:
        return (rng or random).choice(all_choices)
    if len(candidates) == 1:
        return candidates[0]

    starts = []
    for candidate in candidates:
        position = response.rfind(f"({candidate})")
        if position == -1:
            position = response.rfind(f" {candidate} ")
        if position == -1 and index2ans.get(candidate):
            position = response.lower().rfind(index2ans[candidate].lower())
        starts.append(position)
    return candidates[int(max(range(len(starts)), key=lambda i: starts[i]))]


class MMMUProVLMEval(Eval):

    def __init__(
        self,
        num_examples: int | None = None,
        num_threads: int = 16,
        setting: str | None = None,
        mode: str | None = None,
        cap_px: int | None = None,
        seed: int = 1,
    ):
        self.num_threads = num_threads
        self.setting = setting or os.getenv("MMMU_SETTING", "vision")
        self.mode = mode or os.getenv("MMMU_MODE", "cot")
        self.cap_px = cap_px if cap_px is not None else int(os.getenv("MMMU_CAP_PX", "1000000"))
        self.seed = seed
        self.samples = self._prepare_samples(num_examples)

    def _prepare_samples(self, num_examples: int | None) -> list[dict[str, Any]]:
        dataset = list(load_dataset("MMMU/MMMU_Pro", self.setting, split="test"))
        if num_examples:
            dataset = random.Random(self.seed).sample(dataset, num_examples)

        prompt_config = PROMPTS[self.mode]
        samples = []
        for example in dataset:
            options = _parse_options(example["options"])
            index2ans, all_choices = _build_mc_mapping(options)

            if "standard (10 options)" in self.setting:
                choices_text = "\n".join(
                    f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(options)
                )
                prompt = f"{example['question']}\n{choices_text}\n{prompt_config['standard']}"
                prompt, image_order = _replace_image_tokens(prompt)
                images = [example[f"image_{idx}"] for idx in image_order]
            elif self.setting == "vision":
                prompt = prompt_config["vision"]
                images = [example["image"]]
            else:
                raise ValueError(f"Unsupported MMMU-Pro setting: {self.setting}")

            category = example.get("subdomain") or _extract_subset_name(example["id"])
            samples.append(
                {
                    "id": example["id"],
                    "prompt": prompt,
                    "images": images,
                    "answer": str(example["answer"]).strip().upper(),
                    "index2ans": index2ans,
                    "all_choices": all_choices,
                    "category": category,
                    "domain": _SUB_CAT2DOMAIN.get(category, "Unknown"),
                }
            )
        return samples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        rng = random.Random(self.seed)

        def fn(sample: dict[str, Any]) -> SingleEvalResult:
            prompt_messages = _build_chat_messages(sample["prompt"], sample["images"], self.cap_px)
            response_text = sampler(prompt_messages) or ""
            extracted_answer = _parse_multi_choice_response(
                response_text,
                sample["all_choices"],
                sample["index2ans"],
                rng,
            )
            score = 1.0 if extracted_answer == sample["answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message={"content": response_text, "role": "assistant"},
                score=score,
                correct_answer=sample["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [{"content": response_text, "role": "assistant"}]
            return SingleEvalResult(
                html=html,
                score=score,
                metrics={
                    f"{_CATEGORY_PREFIX}{sample['category']}": score,
                    f"{_DOMAIN_PREFIX}{sample['domain']}": score,
                },
                convo=convo,
            )

        results = common.map_with_progress(fn, self.samples, self.num_threads)
        return common.aggregate_results(results)
