#!/usr/bin/env bash
set -euo pipefail

OUT="/tmp/sgl_jax_speed_benchmark"
HOST="127.0.0.1"
PORT="30000"
MODEL=""
DATASET_NAME="random"
BATCH_SIZES="2 4 8 16"
NUM_PROMPTS_MULTIPLIER="3"
INPUT_LEN="1024"
OUTPUT_LEN="4096"
RANDOM_RANGE_RATIO="1"
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
  --out /tmp/sgl_jax_speed_benchmark
  --batch-sizes "2 4 8 16"
  --num-prompts-multiplier 3
  --input-len 1024
  --output-len 4096
  --backend sgl-jax

Options:
  --host VALUE
  --port VALUE
  --model VALUE
  --dataset-name VALUE
  --out VALUE
  --batch-sizes VALUE              Quoted space-separated list, e.g. "32 64 128"
  --num-prompts-multiplier VALUE   num_prompts = bs * multiplier
  --input-len VALUE
  --output-len VALUE
  --random-range-ratio VALUE
  --backend VALUE
  --request-rate VALUE
  --python-bin VALUE
  --append-output                  Append to existing result.jsonl instead of replacing it
  -h, --help
  --                               Pass remaining args directly to bench_serving

Examples:
  scripts/speed_benchmark.sh --batch-sizes "32 64 128"
  scripts/speed_benchmark.sh --model mistralai/Mistral-7B-Instruct-v0.3 --out /tmp/mistral_speed
  scripts/speed_benchmark.sh --dataset-name sharegpt --output-len 1024
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --num-prompts-multiplier) NUM_PROMPTS_MULTIPLIER="$2"; shift 2 ;;
    --input-len) INPUT_LEN="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --random-range-ratio) RANDOM_RANGE_RATIO="$2"; shift 2 ;;
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
echo "out=$OUT"
echo "batch_sizes=$BATCH_SIZES"

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
  if [[ "$DATASET_NAME" == "random" ]]; then
    cmd+=(
      --random-input-len "$INPUT_LEN"
      --random-output-len "$OUTPUT_LEN"
      --random-range-ratio "$RANDOM_RANGE_RATIO"
    )
  fi
  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  cmd+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

  "${cmd[@]}"

  "$PYTHON_BIN" - "$result_file" "$SUMMARY" <<'PY'
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
with open(summary, "a", encoding="utf-8") as f:
    f.write(",".join(map(str, values)) + "\n")
PY

  echo "Finished BS=${bs}"
done

echo
echo "Speed benchmark finished."
echo "Output root: $OUT"
echo "Summary: $SUMMARY"
cat "$SUMMARY"
