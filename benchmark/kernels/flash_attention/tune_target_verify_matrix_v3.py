"""Run a representative target-verify RPA v3 tuning matrix."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

FULL_CASES = (
    # q4 production buckets.
    ("full-q4-bs4-p16k", 4, 4, 16384),
    ("full-q4-bs8-p16k", 8, 4, 16384),
    ("full-q4-bs16-p16k", 16, 4, 16384),
    ("full-q4-bs32-p16k", 32, 4, 16384),
    ("full-q4-bs64-p16k", 64, 4, 16384),
    # Other speculative lengths at the same total-Q=128 bucket.
    ("full-q2-bs64-p16k", 64, 2, 16384),
    ("full-q8-bs16-p16k", 16, 8, 16384),
    # Context sensitivity for the production q4/bs32 bucket.
    ("full-q4-bs32-p4k", 32, 4, 4096),
    ("full-q4-bs32-p8k", 32, 4, 8192),
    ("full-q4-bs32-p17k", 32, 4, 17408),
    ("full-q4-bs32-p32k", 32, 4, 32768),
)

FULL_CONTEXT_CASES = FULL_CASES[-4:]

SWA_CASES = (
    ("swa128-q4-bs4-p16k", 4, 4, 16384),
    ("swa128-q4-bs8-p16k", 8, 4, 16384),
    ("swa128-q4-bs16-p16k", 16, 4, 16384),
    ("swa128-q4-bs32-p16k", 32, 4, 16384),
    ("swa128-q4-bs64-p16k", 64, 4, 16384),
    ("swa128-q2-bs64-p16k", 64, 2, 16384),
    ("swa128-q8-bs16-p16k", 16, 8, 16384),
)

SWA_CONTEXT_CASES = (
    ("swa128-q4-bs32-p4k", 32, 4, 4096),
    ("swa128-q4-bs32-p8k", 32, 4, 8192),
    ("swa128-q4-bs32-p17k", 32, 4, 17408),
    ("swa128-q4-bs32-p32k", 32, 4, 32768),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=(
            "full",
            "swa",
            "tokens16-full",
            "tokens16-swa",
            "context-full",
            "context-swa",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = []
    if args.suite in ("full", "all"):
        cases.extend((name, bs, q, prefix, 0) for name, bs, q, prefix in FULL_CASES)
    if args.suite in ("swa", "all"):
        cases.extend((name, bs, q, prefix, 128) for name, bs, q, prefix in SWA_CASES)
    if args.suite == "tokens16-full":
        name, bs, q, prefix = FULL_CASES[0]
        cases.append((name, bs, q, prefix, 0))
    if args.suite == "tokens16-swa":
        name, bs, q, prefix = SWA_CASES[0]
        cases.append((name, bs, q, prefix, 128))
    if args.suite == "context-full":
        cases.extend((name, bs, q, prefix, 0) for name, bs, q, prefix in FULL_CONTEXT_CASES)
    if args.suite == "context-swa":
        cases.extend((name, bs, q, prefix, 128) for name, bs, q, prefix in SWA_CONTEXT_CASES)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuner = Path(__file__).with_name("tune_target_verify_v3.py")
    aggregate_path = args.output_dir / "metrics.jsonl"
    with aggregate_path.open("w") as aggregate:
        for name, batch_size, draft_token_num, prefix_len, sliding_window in cases:
            case_path = args.output_dir / f"{name}.jsonl"
            command = [
                sys.executable,
                str(tuner),
                "--batch-size",
                str(batch_size),
                "--draft-token-num",
                str(draft_token_num),
                "--prefix-len",
                str(prefix_len),
                "--tries",
                str(args.tries),
                "--grid",
                "focused",
                "--output",
                str(case_path),
            ]
            if sliding_window:
                command.extend(["--sliding-window", str(sliding_window)])
            print(f"\n## CASE {name}", flush=True)
            subprocess.run(command, check=True)
            for line in case_path.read_text().splitlines():
                record = json.loads(line)
                record["case"] = name
                aggregate.write(json.dumps(record) + "\n")
                aggregate.flush()


if __name__ == "__main__":
    main()
