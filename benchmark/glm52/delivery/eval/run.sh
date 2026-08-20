#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

if [[ $# -ne 1 ]]; then
  printf 'usage: %s {gsm8k|aime26}\n' "$0" >&2
  exit 2
fi
DATASET="$1"

if ! command -v sgl-eval >/dev/null 2>&1; then
  printf 'sgl-eval is not installed; see benchmark/glm52/delivery/README.md\n' >&2
  exit 2
fi

BASE_URL="${BASE_URL:-http://localhost:${PORT:-30000}/v1}"
MODEL_PATH="${MODEL_PATH:-/models/GLM5.2-fp8-channel-wise}"
OUT_ROOT="${OUT_ROOT:-$PWD/artifacts/eval/glm52}"
EVAL_SCOPE="${EVAL_SCOPE:-quick}"

case "$EVAL_SCOPE" in
  quick|full) ;;
  *)
    printf 'unsupported EVAL_SCOPE: %s (expected quick or full)\n' "$EVAL_SCOPE" >&2
    exit 2
    ;;
esac

case "$DATASET" in
  gsm8k)
    if [[ "$EVAL_SCOPE" == "full" ]]; then
      DEFAULT_NUM_EXAMPLES=1319
    else
      DEFAULT_NUM_EXAMPLES=200
    fi
    NUM_EXAMPLES="${NUM_EXAMPLES:-$DEFAULT_NUM_EXAMPLES}"
    NUM_THREADS="${NUM_THREADS:-128}"
    TEMPERATURE="${TEMPERATURE:-0.0}"
    TOP_P="${TOP_P:-1.0}"
    MAX_TOKENS="${MAX_TOKENS:-4096}"
    ;;
  aime26)
    NUM_EXAMPLES="${NUM_EXAMPLES:-30}"
    NUM_THREADS="${NUM_THREADS:-16}"
    TEMPERATURE="${TEMPERATURE:-1.0}"
    TOP_P="${TOP_P:-0.95}"
    MAX_TOKENS="${MAX_TOKENS:-163840}"
    ;;
  *)
    printf 'unsupported dataset: %s (expected gsm8k or aime26)\n' "$DATASET" >&2
    exit 2
    ;;
esac

if [[ ! "$NUM_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'NUM_EXAMPLES must be a positive integer, got: %s\n' "$NUM_EXAMPLES" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-$OUT_ROOT/$DATASET/$EVAL_SCOPE}"
mkdir -p "$OUT_DIR"

printf 'GLM52_DELIVERY_EVAL dataset=%s scope=%s examples=%s threads=%s model=%s output=%s\n' \
  "$DATASET" "$EVAL_SCOPE" "$NUM_EXAMPLES" "$NUM_THREADS" "$MODEL_PATH" "$OUT_DIR"

EVAL_CMD=(sgl-eval run "$DATASET" \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --num-examples "$NUM_EXAMPLES" \
  --num-threads "$NUM_THREADS" \
  --n-repeats 1 \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --seed "${SEED:-3}" \
  --max-tokens "$MAX_TOKENS" \
  --thinking \
  --chat-template-kwarg enable_thinking=true \
  --out-dir "$OUT_DIR")

if [[ -z "${MIN_SCORE:-}" ]]; then
  exec "${EVAL_CMD[@]}"
fi

"${EVAL_CMD[@]}"
python3 "$REPO_ROOT/benchmark/glm52/delivery/validation/validate_accuracy_metrics.py" \
  --root "$OUT_DIR" \
  --min-score "$MIN_SCORE" \
  --expected-examples "$NUM_EXAMPLES"
