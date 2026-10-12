# Copyright 2023 VLMEvalKit Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 (see repository LICENSE)
# Adapted from open-compass/VLMEvalKit at d21c5e969983dea6741a98eb2f7ec566ab82ee91:
# vlmeval/utils/matching_util.py and vlmeval/vlm/qwen3_vl/prompt.py.
# Changes: retain only Qwen3-VL MCQ text and exact-matching answer extraction;
# use the standard logger instead of VLMEvalKit's logger. No LLM judge fallback.

import copy as cp
import logging
import os
import re
import string

import pandas as pd

logger = logging.getLogger(__name__)
_VERBOSE_ANSWER_RE = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**")


def choices_for(row):
    return {
        letter: row[letter]
        for letter in string.ascii_uppercase
        if letter in row and not pd.isna(row[letter])
    }


def prompt_for(row):
    options = choices_for(row)
    hint = row.get("hint")
    prompt = f"Hint: {hint}\n" if hint is not None and not pd.isna(hint) else ""
    prompt += f"Question: {row['question']}\n"
    if options:
        prompt += "Options:\n"
        prompt += "".join(f"{letter}. {text}\n" for letter, text in options.items())
        prompt += "Answer with the option letter only."
    return prompt.rstrip()


def can_infer_option(answer, choices):
    verbose = os.environ.get("VERBOSE", 0)
    # Choices is a dictionary
    if "Failed to obtain answer via API" in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = ".()[],:;!*#{}"
    for c in chars:
        answer_mod = answer_mod.replace(c, " ")

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if "A" in splits and len(splits) > 3 and verbose:
                logger.info("A might be a quantifier in the string: %s.", answer)
                return False
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"

    match = _VERBOSE_ANSWER_RE.search(answer or "")
    if match and match.group(1).upper() in choices:
        return match.group(1).upper()

    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)
