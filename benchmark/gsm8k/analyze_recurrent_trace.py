"""Analyze env-gated recurrent trace logs for a GSM8K greedy run."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

INVALID = "-9999999"

QUESTION_RE = re.compile(
    r"^=== Question (?P<idx>\d+) ===\n(?P<body>.*?)(?=^=== Question \d+ ===\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
RID_RE = re.compile(r"^=== Request ID:\s*(?P<rid>.*?) ===$", re.MULTILINE)
PRED_RE = re.compile(
    r"^=== Prediction:\s*(?P<pred>.*?), Label:\s*(?P<label>.*?) ===$", re.MULTILINE
)


@dataclass
class OutputCase:
    idx: int
    rid: str | None
    prediction: str
    label: str

    @property
    def correct(self) -> bool:
        return normalize(self.prediction) == normalize(self.label)


def normalize(value: str) -> str:
    return value.strip().replace(",", "")


def parse_outputs(path: Path) -> dict[int, OutputCase]:
    text = path.read_text()
    cases = {}
    for match in QUESTION_RE.finditer(text):
        idx = int(match.group("idx"))
        body = match.group("body")
        rid_match = RID_RE.search(body)
        pred_match = PRED_RE.search(body)
        if pred_match is None:
            raise ValueError(f"{path}: missing prediction for question {idx}")
        cases[idx] = OutputCase(
            idx=idx,
            rid=rid_match.group("rid").strip() if rid_match else None,
            prediction=pred_match.group("pred").strip(),
            label=pred_match.group("label").strip(),
        )
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def read_trace(paths: list[str]) -> list[dict[str, Any]]:
    events = []
    for pattern in paths:
        for path in glob.glob(pattern):
            with Path(path).open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    event["_trace_file"] = path
                    events.append(event)
    events.sort(key=lambda item: item.get("ts", 0.0))
    return events


def load_tokenizer(model_path: str | None):
    if model_path is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def decode_ids(tokenizer, token_ids: list[int] | None) -> str:
    if not token_ids:
        return ""
    if tokenizer is None:
        return " ".join(str(x) for x in token_ids)
    return tokenizer.decode(token_ids)


def first(events: list[dict[str, Any]], rid: str, event_name: str, mode: str | None = None):
    for event in events:
        if event.get("rid") != rid or event.get("event") != event_name:
            continue
        if mode is not None and mode not in str(event.get("mode")):
            continue
        return event
    return None


def last_before(
    events: list[dict[str, Any]],
    rid: str,
    event_name: str,
    mode: str | None,
    before_ts: float | None,
):
    matched = None
    for event in events:
        if event.get("rid") != rid or event.get("event") != event_name:
            continue
        if mode is not None and mode not in str(event.get("mode")):
            continue
        if before_ts is not None and float(event.get("ts", 0.0)) > before_ts:
            continue
        matched = event
    return matched


def digest_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return math.inf
    return float(np.max(np.abs(left_arr - right_arr)))


def build_rows(cases: dict[int, OutputCase], events: list[dict[str, Any]], tokenizer):
    rows = []
    for case in cases.values():
        if case.correct:
            continue
        rid = case.rid
        if rid is None:
            continue
        decode_state = first(events, rid, "state_digest", "DECODE")
        first_decode = first(events, rid, "future_token_resolve", "DECODE")
        prefill_state = last_before(
            events,
            rid,
            "state_digest",
            "EXTEND",
            float(first_decode.get("ts")) if first_decode else None,
        )
        rows.append(
            {
                "idx": case.idx,
                "rid": rid,
                "prediction": case.prediction,
                "label": case.label,
                "first_decode_after_ids": (
                    first_decode.get("input_ids_after_resolve") if first_decode else None
                ),
                "first_decode_after_text": decode_ids(
                    tokenizer,
                    first_decode.get("input_ids_after_resolve") if first_decode else None,
                ),
                "first_decode_before_ids": (
                    first_decode.get("input_ids_before_resolve") if first_decode else None
                ),
                "req_pool_idx": first_decode.get("req_pool_idx") if first_decode else None,
                "recurrent_idx": first_decode.get("recurrent_idx") if first_decode else None,
                "prefill_bid": prefill_state.get("bid") if prefill_state else None,
                "decode_bid": decode_state.get("bid") if decode_state else None,
                "prefill_recurrent_after": (
                    prefill_state.get("recurrent_after") if prefill_state else None
                ),
                "decode_recurrent_before": (
                    decode_state.get("recurrent_before") if decode_state else None
                ),
                "recurrent_delta_prefill_after_to_decode_before": digest_delta(
                    prefill_state.get("recurrent_after") if prefill_state else None,
                    decode_state.get("recurrent_before") if decode_state else None,
                ),
                "prefill_conv_after": prefill_state.get("conv_after") if prefill_state else None,
                "decode_conv_before": decode_state.get("conv_before") if decode_state else None,
                "conv_delta_prefill_after_to_decode_before": digest_delta(
                    prefill_state.get("conv_after") if prefill_state else None,
                    decode_state.get("conv_before") if decode_state else None,
                ),
            }
        )
    return rows


def write_report(rows: list[dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "recurrent_trace_wrong_cases.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "wrong_cases": len(rows),
        "first_decode_missing": sum(row["first_decode_after_ids"] is None for row in rows),
        "state_digest_missing": sum(
            row["recurrent_delta_prefill_after_to_decode_before"] is None for row in rows
        ),
    }
    (out_dir / "recurrent_trace_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Recurrent Trace Wrong Cases",
        "",
        f"- Wrong cases: {summary['wrong_cases']}",
        f"- Missing first decode trace: {summary['first_decode_missing']}",
        f"- Missing state digest: {summary['state_digest_missing']}",
        "",
        "| idx | rid | pred | label | first decode text | recurrent delta | conv delta |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {idx} | `{rid}` | {prediction} | {label} | `{text}` | {rd} | {cd} |".format(
                idx=row["idx"],
                rid=row["rid"],
                prediction=row["prediction"],
                label=row["label"],
                text=row["first_decode_after_text"].replace("\n", "\\n"),
                rd=row["recurrent_delta_prefill_after_to_decode_before"],
                cd=row["conv_delta_prefill_after_to_decode_before"],
            )
        )
    (out_dir / "recurrent_trace_report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    cases = parse_outputs(args.output_file)
    events = read_trace(args.trace)
    rows = build_rows(cases, events, tokenizer)
    write_report(rows, args.out_dir)


if __name__ == "__main__":
    main()
