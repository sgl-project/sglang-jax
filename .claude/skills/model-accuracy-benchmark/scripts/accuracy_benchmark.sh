#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the sglang-jax repo's test/srt/run_eval.py accuracy runner.
# Runs against an OpenAI-compatible endpoint and prints the eval score.
# Use only as a fallback when evalscope cannot be installed/invoked on the target host.

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_DIR="${REPO_DIR:-.}"
RUN_EVAL="test/srt/run_eval.py"

BASE_URL="http://127.0.0.1:30000"
MODEL=""
EVAL_NAME="gsm8k"
NUM_EXAMPLES=""
NUM_THREADS="8"
MAX_TOKENS="4096"
TEMPERATURE="0.0"
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Run test/srt/run_eval.py accuracy benchmark against an OpenAI-compatible endpoint.

Defaults:
  --base-url http://127.0.0.1:30000
  --model "" (run_eval auto-discovers from /v1/models when empty)
  --eval-name gsm8k
  --num-examples "" (empty means the full dataset)
  --num-threads 8
  --max-tokens 4096
  --temperature 0.0
  --repo-dir . (directory containing test/srt/run_eval.py)
  --python-bin python3

Options:
  --base-url VALUE
  --model VALUE
  --eval-name VALUE
  --num-examples VALUE
  --num-threads VALUE
  --max-tokens VALUE
  --temperature VALUE
  --repo-dir VALUE       sglang-jax repo root (run_eval.py is resolved under it)
  --python-bin VALUE     Python executable, e.g. the server's .venv/bin/python
  -h, --help
  --                     Pass remaining args directly to run_eval.py

Examples:
  scripts/accuracy_benchmark.sh --num-examples 30
  scripts/accuracy_benchmark.sh --repo-dir ~/sglang-jax --python-bin ~/sglang-jax/.venv/bin/python
  scripts/accuracy_benchmark.sh --base-url http://127.0.0.1:40000 --model /models/mistralai/Mistral-7B-Instruct-v0.3
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --eval-name) EVAL_NAME="$2"; shift 2 ;;
    --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
    --num-threads) NUM_THREADS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "$REPO_DIR/$RUN_EVAL" ]]; then
  echo "run_eval.py not found at $REPO_DIR/$RUN_EVAL; pass --repo-dir <sglang-jax root>." >&2
  exit 2
fi

cmd=(
  "$PYTHON_BIN" "$RUN_EVAL"
  --base-url "$BASE_URL"
  --eval-name "$EVAL_NAME"
  --num-threads "$NUM_THREADS"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
)
if [[ -n "$NUM_EXAMPLES" ]]; then
  cmd+=(--num-examples "$NUM_EXAMPLES")
fi
if [[ -n "$MODEL" ]]; then
  cmd+=(--model "$MODEL")
fi
cmd+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "Running accuracy benchmark"
echo "repo_dir=$REPO_DIR"
echo "base_url=$BASE_URL"
echo "model=${MODEL:-<auto from /v1/models>}"
echo "eval_name=$EVAL_NAME"
echo "num_examples=${NUM_EXAMPLES:-<full dataset>}"
echo "command=${cmd[*]}"

cd "$REPO_DIR"
exec "${cmd[@]}"
