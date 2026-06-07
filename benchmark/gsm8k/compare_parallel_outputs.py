"""Compare greedy GSM8K outputs from two concurrency settings."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUESTION_RE = re.compile(
    r"^=== Question (?P<idx>\d+) ===\n(?P<body>.*?)(?=^=== Question \d+ ===\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
PRED_RE = re.compile(
    r"^=== Prediction:\s*(?P<pred>.*?), Label:\s*(?P<label>.*?) ===$", re.MULTILINE
)
PROMPT_QUESTION_RE = re.compile(r"Question:\s*(?P<question>.*?)(?=\nAnswer:)", re.DOTALL)


@dataclass
class ParsedCase:
    idx: int
    question: str
    generated: str
    prediction: str
    label: str

    @property
    def correct(self) -> bool:
        return normalize_answer(self.prediction) == normalize_answer(self.label)


def normalize_answer(value: str) -> str:
    return value.strip().replace(",", "")


def compact_text(value: str, limit: int = 500) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def parse_output(path: Path) -> dict[int, ParsedCase]:
    text = path.read_text()
    cases: dict[int, ParsedCase] = {}
    for match in QUESTION_RE.finditer(text):
        idx = int(match.group("idx"))
        body = match.group("body")
        if "=== Answer ===" not in body:
            raise ValueError(f"{path}: question {idx} is missing generated answer marker")
        prompt_part, after_answer = body.split("=== Answer ===", 1)
        pred_match = PRED_RE.search(after_answer)
        if pred_match is None:
            raise ValueError(f"{path}: question {idx} is missing prediction line")
        generated = after_answer[: pred_match.start()].strip("\n")
        questions = [m.group("question").strip() for m in PROMPT_QUESTION_RE.finditer(prompt_part)]
        question = questions[-1] if questions else ""
        cases[idx] = ParsedCase(
            idx=idx,
            question=compact_text(question),
            generated=generated,
            prediction=pred_match.group("pred").strip(),
            label=pred_match.group("label").strip(),
        )
    if not cases:
        raise ValueError(f"{path}: no GSM8K cases parsed")
    return cases


class TextTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def load_tokenizer(model_path: str | None):
    if model_path is None:
        return TextTokenizer(), "char-fallback"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer, "hf-tokenizer"
    except Exception as exc:  # pragma: no cover - exercised in remote env if tokenizer is absent
        print(f"WARNING: failed to load tokenizer from {model_path}: {exc}")
        return TextTokenizer(), "char-fallback"


def first_diff(left: list[int], right: list[int]) -> int | None:
    for idx, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def first_char_diff(left: str, right: str) -> int | None:
    for idx, (left_ch, right_ch) in enumerate(zip(left, right)):
        if left_ch != right_ch:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def token_obj(tokenizer: Any, token_id: int | None) -> dict[str, Any] | None:
    if token_id is None:
        return None
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        text = chr(token_id)
    return {"id": token_id, "text": text}


def snippet(text: str, pos: int | None, radius: int = 180) -> str:
    if pos is None:
        return compact_text(text, limit=radius * 2)
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def classify_diff(diff_idx: int | None) -> str:
    if diff_idx is None:
        return "same_generation"
    if diff_idx == 0:
        return "first_token_diff"
    return "decode_diff_after_shared_prefix"


def compare_cases(
    baseline: dict[int, ParsedCase],
    candidate: dict[int, ParsedCase],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common_ids = sorted(set(baseline) & set(candidate))
    if set(baseline) != set(candidate):
        missing_left = sorted(set(candidate) - set(baseline))
        missing_right = sorted(set(baseline) - set(candidate))
        raise ValueError(
            "case id mismatch: "
            f"missing_baseline={missing_left[:10]}, missing_candidate={missing_right[:10]}"
        )

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "total": len(common_ids),
        "baseline_correct": 0,
        "candidate_correct": 0,
        "both_correct": 0,
        "baseline_correct_candidate_wrong": 0,
        "baseline_wrong_candidate_correct": 0,
        "both_wrong": 0,
        "candidate_wrong": 0,
        "candidate_wrong_first_token_diff": 0,
        "candidate_wrong_decode_diff_after_shared_prefix": 0,
        "candidate_wrong_same_generation": 0,
    }

    for idx in common_ids:
        left = baseline[idx]
        right = candidate[idx]
        left_ids = tokenizer.encode(left.generated)
        right_ids = tokenizer.encode(right.generated)
        diff_idx = first_diff(left_ids, right_ids)
        char_idx = first_char_diff(left.generated, right.generated)
        category = classify_diff(diff_idx)
        left_correct = left.correct
        right_correct = right.correct

        summary["baseline_correct"] += int(left_correct)
        summary["candidate_correct"] += int(right_correct)
        summary["both_correct"] += int(left_correct and right_correct)
        summary["baseline_correct_candidate_wrong"] += int(left_correct and not right_correct)
        summary["baseline_wrong_candidate_correct"] += int(not left_correct and right_correct)
        summary["both_wrong"] += int(not left_correct and not right_correct)
        summary["candidate_wrong"] += int(not right_correct)
        if not right_correct:
            summary[f"candidate_wrong_{category}"] += 1

        if not right_correct:
            rows.append(
                {
                    "idx": idx,
                    "question": left.question,
                    "label": left.label,
                    "baseline_prediction": left.prediction,
                    "candidate_prediction": right.prediction,
                    "baseline_correct": left_correct,
                    "candidate_correct": right_correct,
                    "diff_category": category,
                    "first_diff_token_index": diff_idx,
                    "first_diff_char_index": char_idx,
                    "baseline_first_token": token_obj(tokenizer, left_ids[0] if left_ids else None),
                    "candidate_first_token": token_obj(
                        tokenizer, right_ids[0] if right_ids else None
                    ),
                    "baseline_diff_token": token_obj(
                        tokenizer,
                        (
                            left_ids[diff_idx]
                            if diff_idx is not None and diff_idx < len(left_ids)
                            else None
                        ),
                    ),
                    "candidate_diff_token": token_obj(
                        tokenizer,
                        (
                            right_ids[diff_idx]
                            if diff_idx is not None and diff_idx < len(right_ids)
                            else None
                        ),
                    ),
                    "baseline_snippet": snippet(left.generated, char_idx),
                    "candidate_snippet": snippet(right.generated, char_idx),
                    "baseline_generated": left.generated,
                    "candidate_generated": right.generated,
                }
            )

    summary["baseline_accuracy"] = summary["baseline_correct"] / summary["total"]
    summary["candidate_accuracy"] = summary["candidate_correct"] / summary["total"]
    return rows, summary


def write_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    baseline_name: str,
    candidate_name: str,
    tokenizer_kind: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "parallel_compare_summary.json").write_text(
        json.dumps(
            {
                "baseline_name": baseline_name,
                "candidate_name": candidate_name,
                "tokenizer": tokenizer_kind,
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with (out_dir / "parallel_compare_wrong_cases.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# GSM8K Parallel Output Comparison",
        "",
        f"Baseline: `{baseline_name}`",
        f"Candidate: `{candidate_name}`",
        f"Tokenizer: `{tokenizer_kind}`",
        "",
        "## Summary",
        "",
        f"- Total aligned cases: {summary['total']}",
        f"- Baseline accuracy: {summary['baseline_accuracy']:.3f}",
        f"- Candidate accuracy: {summary['candidate_accuracy']:.3f}",
        f"- Candidate wrong cases: {summary['candidate_wrong']}",
        f"- Baseline correct but candidate wrong: {summary['baseline_correct_candidate_wrong']}",
        f"- Candidate wrong with first-token diff: {summary['candidate_wrong_first_token_diff']}",
        (
            "- Candidate wrong with later decode diff: "
            f"{summary['candidate_wrong_decode_diff_after_shared_prefix']}"
        ),
        f"- Candidate wrong with same generation text: {summary['candidate_wrong_same_generation']}",
        "",
        "## Wrong Candidate Cases",
        "",
        (
            "| idx | label | baseline pred | candidate pred | baseline ok | "
            "diff category | first diff token | baseline token | candidate token |"
        ),
        "|---:|---:|---:|---:|:---:|---|---:|---|---|",
    ]
    for row in rows:
        left_tok = row["baseline_diff_token"] or row["baseline_first_token"]
        right_tok = row["candidate_diff_token"] or row["candidate_first_token"]
        lines.append(
            "| {idx} | {label} | {baseline_prediction} | {candidate_prediction} | "
            "{baseline_correct} | {diff_category} | {first_diff_token_index} | "
            "`{left}` | `{right}` |".format(
                idx=row["idx"],
                label=row["label"],
                baseline_prediction=row["baseline_prediction"],
                candidate_prediction=row["candidate_prediction"],
                baseline_correct="Y" if row["baseline_correct"] else "N",
                diff_category=row["diff_category"],
                first_diff_token_index=row["first_diff_token_index"],
                left=compact_text((left_tok or {}).get("text", ""), 40).replace("|", "\\|"),
                right=compact_text((right_tok or {}).get("text", ""), 40).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "Full snippets and generated strings are in `parallel_compare_wrong_cases.jsonl`.",
            "",
        ]
    )
    (out_dir / "parallel_compare_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-output", required=True, type=Path)
    parser.add_argument("--candidate-output", required=True, type=Path)
    parser.add_argument("--baseline-name", default="parallel1")
    parser.add_argument("--candidate-name", default="parallel64")
    parser.add_argument("--model-path")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    tokenizer, tokenizer_kind = load_tokenizer(args.model_path)
    baseline = parse_output(args.baseline_output)
    candidate = parse_output(args.candidate_output)
    rows, summary = compare_cases(baseline, candidate, tokenizer)
    write_report(
        args.out_dir,
        rows,
        summary,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        tokenizer_kind=tokenizer_kind,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
