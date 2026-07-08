#!/usr/bin/env bash
set -euo pipefail

# Wrapper around sgl_jax.bench_serving: sweeps batch sizes and writes a summary.csv.
# Pass the server's .venv/bin/python via --python-bin so sgl_jax imports resolve.
#
# Thin by design: this wrapper owns ONLY dataset-agnostic concerns (the bs sweep,
# num_prompts = bs * multiplier, the common flags, and summary.csv extraction). It does
# NOT translate any dataset-specific flags. Pass everything dataset-specific
# (--random-input-len, --sharegpt-output-len, --gsp-*, --mooncake-*, ...) verbatim after
# `--`; it is forwarded to bench_serving unchanged. This avoids silently dropping flags.
# Do NOT pass --max-concurrency / --num-prompts after `--`; the wrapper owns those.

OUT="/tmp/sgl_jax_speed_benchmark"
HOST="127.0.0.1"
PORT="30000"
MODEL=""
DATASET_NAME="random"
DATASET_PATH=""
BATCH_SIZES="2 4 8 16"
NUM_PROMPTS_MULTIPLIER="3"
BACKEND="sgl-jax"
REQUEST_RATE="inf"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CLEAR_OUTPUT="1"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Run sgl_jax.bench_serving speed benchmarks and write a summary.csv.

Defaults:
  --host 127.0.0.1
  --port 30000
  --model "" (auto-discover from /v1/models)
  --dataset-name random
  --dataset-path "" (required by sharegpt/custom/image/mmmu/mooncake)
  --out /tmp/sgl_jax_speed_benchmark
  --batch-sizes "2 4 8 16"
  --num-prompts-multiplier 3
  --backend sgl-jax

Options (dataset-agnostic; owned by the wrapper):
  --host VALUE
  --port VALUE
  --model VALUE
  --dataset-name VALUE             sharegpt | random | random-ids | generated-shared-prefix
                                   | mmmu | image | mooncake | custom
  --dataset-path VALUE
  --out VALUE
  --batch-sizes VALUE              Quoted space-separated list, e.g. "32 64 128"
  --num-prompts-multiplier VALUE   num_prompts = bs * multiplier
  --backend VALUE
  --request-rate VALUE
  --python-bin VALUE
  --append-output                  Append to existing result.jsonl instead of replacing it
  -h, --help
  --                               Pass dataset-specific flags directly to bench_serving

Dataset-specific flags go after `--` and are forwarded verbatim. Do NOT pass
--max-concurrency / --num-prompts after `--`; the wrapper sets them per batch point.

Examples (see the SKILL.md "Per-dataset recipes" table for the full set):
  # random / random-ids (fixed shapes, the canonical speed workload)
  scripts/speed_benchmark.sh --dataset-name random \
    -- --random-input-len 16384 --random-output-len 1024 --random-range-ratio 1

  # sharegpt
  scripts/speed_benchmark.sh --dataset-name sharegpt --dataset-path /data/sharegpt.json \
    -- --sharegpt-output-len 1024

  # generated-shared-prefix
  scripts/speed_benchmark.sh --dataset-name generated-shared-prefix \
    -- --gsp-num-groups 64 --gsp-prompts-per-group 16 --gsp-system-prompt-len 2048 \
       --gsp-question-len 128 --gsp-output-len 256
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --dataset-path) DATASET_PATH="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --num-prompts-multiplier) NUM_PROMPTS_MULTIPLIER="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --append-output) CLEAR_OUTPUT="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$OUT"
SUMMARY="$OUT/summary.csv"
printf "Max Concurrency,Input tok/s,Output tok/s,Peak output tok/s,Total tok/s,Mean E2E ms,Mean TTFT ms,Mean TPOT ms,Mean ITL ms,P99 ITL ms\n" > "$SUMMARY"

echo "Running speed benchmark"
echo "host=$HOST"
echo "port=$PORT"
echo "model=$MODEL"
echo "dataset_name=$DATASET_NAME"
echo "dataset_path=$DATASET_PATH"
echo "out=$OUT"
echo "batch_sizes=$BATCH_SIZES"

FAILED_BS=()
COMPLETED_BS=()

for bs in $BATCH_SIZES; do
  num_prompts=$((bs * NUM_PROMPTS_MULTIPLIER))
  run_dir="$OUT/bs_${bs}"
  result_file="$run_dir/result.jsonl"
  mkdir -p "$run_dir"
  if [[ "$CLEAR_OUTPUT" == "1" ]]; then
    : > "$result_file"
  fi

  echo "========================================"
  echo "Running sgl-jax benchmark"
  echo "BS=${bs}, num_prompts=${num_prompts}"
  echo "Output: ${result_file}"
  echo "========================================"

  cmd=(
    "$PYTHON_BIN" -m sgl_jax.bench_serving
    --backend "$BACKEND"
    --host "$HOST"
    --port "$PORT"
    --dataset-name "$DATASET_NAME"
    --max-concurrency "$bs"
    --num-prompts "$num_prompts"
    --request-rate "$REQUEST_RATE"
    --output-file "$result_file"
  )
  if [[ -n "$DATASET_PATH" ]]; then
    cmd+=(--dataset-path "$DATASET_PATH")
  fi
  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  # Dataset-specific flags, forwarded verbatim from after `--`.
  cmd+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

  # A failing batch point must not abort the whole sweep: skip it, keep the
  # points already completed, and record it so the run can be reported as partial.
  if ! "${cmd[@]}"; then
    echo "BS=${bs} FAILED (bench_serving exited non-zero); keeping completed points and continuing." >&2
    FAILED_BS+=("$bs")
    continue
  fi

  # Extract one metric row from this bs's result.jsonl and append to summary.csv.
  if ! "$PYTHON_BIN" - "$result_file" "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

result_file, summary = sys.argv[1:3]
path = Path(result_file)
rows = []
if path.exists():
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

def pick(*names):
    for row in reversed(rows):
        for name in names:
            value = row.get(name)
            if value is not None:
                return value
    return ""

values = [
    pick("max_concurrency"),
    pick("input_throughput"),
    pick("output_throughput"),
    pick("max_output_tokens_per_s"),
    pick("total_throughput"),
    pick("mean_e2e_latency_ms"),
    pick("mean_ttft_ms"),
    pick("mean_tpot_ms"),
    pick("mean_itl_ms"),
    pick("p99_itl_ms"),
]
if not rows:
    raise SystemExit(f"No valid JSON rows found in {result_file}")
if any(value == "" for value in values):
    missing = [
        name
        for name, value in zip(
            [
                "max_concurrency",
                "input_throughput",
                "output_throughput",
                "max_output_tokens_per_s",
                "total_throughput",
                "mean_e2e_latency_ms",
                "mean_ttft_ms",
                "mean_tpot_ms",
                "mean_itl_ms",
                "p99_itl_ms",
            ],
            values,
        )
        if value == ""
    ]
    raise SystemExit(f"Missing required metrics in {result_file}: {', '.join(missing)}")
with open(summary, "a", encoding="utf-8") as f:
    f.write(",".join(map(str, values)) + "\n")
PY
  then
    echo "BS=${bs}: metric extraction failed; result.jsonl kept but no summary row." >&2
    FAILED_BS+=("$bs")
    continue
  fi

  COMPLETED_BS+=("$bs")
  echo "Finished BS=${bs}"
done

echo
echo "Speed benchmark finished."
echo "Output root: $OUT"
echo "Summary: $SUMMARY"
cat "$SUMMARY"

echo
echo "Completed batch points: ${COMPLETED_BS[*]:-none}"
if [[ ${#FAILED_BS[@]} -gt 0 ]]; then
  echo "PARTIAL: failed batch points: ${FAILED_BS[*]}" >&2
  exit 3
fi
