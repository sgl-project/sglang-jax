#!/usr/bin/env bash
set -euo pipefail

: "${GLM52_PHYSICAL_CHIPS:?set by a delivery benchmark wrapper}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

case "$GLM52_PHYSICAL_CHIPS" in
  8)
    CONCURRENCY=32
    DP_SIZE=16
    PREFIX_MODE=shared
    ;;
  16)
    CONCURRENCY=64
    DP_SIZE=32
    PREFIX_MODE=unique
    ;;
  *)
    printf 'unsupported physical chip count: %s (expected 8 or 16)\n' \
      "$GLM52_PHYSICAL_CHIPS" >&2
    exit 2
    ;;
esac

BASE_URL="${BASE_URL:-http://localhost:${PORT:-30000}}"
SERVER_LOG="${SERVER_LOG:-/tmp/glm52-${QUANTIZATION:-channelwise}-${GLM52_PHYSICAL_CHIPS}chip-rank0.log}"
OUTPUT="${OUTPUT:-$PWD/glm52-${QUANTIZATION:-channelwise}-${GLM52_PHYSICAL_CHIPS}chip-benchmark.jsonl}"
if [[ ! -f "$SERVER_LOG" ]]; then
  printf 'SERVER_LOG is required for batch-shape validation and was not found: %s\n' \
    "$SERVER_LOG" >&2
  exit 2
fi

printf 'GLM52_DELIVERY_BENCHMARK physical_chips=%s concurrency=%s dp=%s prefix_mode=%s output=%s\n' \
  "$GLM52_PHYSICAL_CHIPS" "$CONCURRENCY" "$DP_SIZE" "$PREFIX_MODE" "$OUTPUT"

BENCH_ARGS=(
  --base-url "$BASE_URL" \
  --server-log "$SERVER_LOG" \
  --concurrency "$CONCURRENCY" \
  --dp-size "$DP_SIZE" \
  --expected-requests-per-dp 2 \
  --prefix-mode "$PREFIX_MODE" \
  --prefix-len 131072 \
  --extend-len 1024 \
  --output-len 1024 \
  --random-seed 3 \
  --variant "${QUANTIZATION:-channelwise}_${GLM52_PHYSICAL_CHIPS}chip_c${CONCURRENCY}_${PREFIX_MODE}_128k_1k_1k" \
  --cache-hit-tolerance 64
)
if [[ -n "${PROFILE_OUTPUT_DIR:-}" ]]; then
  BENCH_ARGS+=(
    --profile-output-dir "$PROFILE_OUTPUT_DIR"
    --profile-host-tracer-level "${PROFILE_HOST_TRACER_LEVEL:-0}"
    --profile-python-tracer-level "${PROFILE_PYTHON_TRACER_LEVEL:-0}"
    --profile-num-steps "${PROFILE_NUM_STEPS:-3}"
    --profile-by-stage
    --profile-stages prefill decode
  )
fi
BENCH_ARGS+=(--output "$OUTPUT")
exec python3 "$REPO_ROOT/benchmark/glm52/bench_dsa_cache_hit.py" \
  "${BENCH_ARGS[@]}" "$@"
