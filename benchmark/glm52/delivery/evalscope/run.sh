#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s officeqa\n' "$0" >&2
  exit 2
fi
DATASET="$1"
if [[ "$DATASET" != "officeqa" ]]; then
  printf 'unsupported EvalScope dataset: %s (expected officeqa)\n' "$DATASET" >&2
  exit 2
fi
if ! command -v evalscope >/dev/null 2>&1; then
  printf 'evalscope is not installed; initialize third_party/evalscope first\n' >&2
  exit 2
fi

BASE_URL="${BASE_URL:-http://localhost:${PORT:-30000}/v1}"
MODEL_PATH="${MODEL_PATH:-/models/GLM5.2-fp8-channel-wise}"
MODEL_ID="${EVALSCOPE_MODEL_ID:-glm52-channelwise-8chip}"
DATASET_DIR="${EVALSCOPE_DATASET_DIR:-/models/evalscope/officeqa}"
OUT_ROOT="${OUT_ROOT:-$PWD/artifacts/evalscope/glm52}"
LIMIT="${EVALSCOPE_LIMIT:-16}"
MAX_STEPS="${EVALSCOPE_MAX_STEPS:-15}"
EVAL_BATCH_SIZE="${EVALSCOPE_BATCH_SIZE:-16}"
COMMAND_TIMEOUT="${EVALSCOPE_COMMAND_TIMEOUT:-60}"
API_TIMEOUT="${EVALSCOPE_API_TIMEOUT:-3600}"
MAX_TOKENS="${EVALSCOPE_MAX_TOKENS:-4096}"
SEED="${EVALSCOPE_SEED:-3}"
ENABLE_THINKING="${EVALSCOPE_ENABLE_THINKING:-true}"
DEBUG="${EVALSCOPE_DEBUG:-1}"

for value_name in LIMIT MAX_STEPS EVAL_BATCH_SIZE COMMAND_TIMEOUT API_TIMEOUT MAX_TOKENS; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got: %s\n' "$value_name" "$value" >&2
    exit 2
  fi
done
case "$ENABLE_THINKING" in
  true|false) ;;
  *) printf 'EVALSCOPE_ENABLE_THINKING must be true or false, got: %s\n' "$ENABLE_THINKING" >&2; exit 2 ;;
esac
case "$DEBUG" in
  0|1) ;;
  *) printf 'EVALSCOPE_DEBUG must be 0 or 1, got: %s\n' "$DEBUG" >&2; exit 2 ;;
esac

OUT_DIR="${EVALSCOPE_OUT_DIR:-$OUT_ROOT/$DATASET/smoke-$LIMIT}"
WORK_DIR="$OUT_DIR/run"
REASONING_PREFLIGHT_DIR="$OUT_DIR/reasoning-preflight"
PREFLIGHT_DIR="$OUT_DIR/tool-call-preflight"
CACHE_PREFLIGHT_DIR="$OUT_DIR/prefix-cache-preflight"
PROVENANCE_DIR="$OUT_DIR/provenance"
mkdir -p "$DATASET_DIR" "$WORK_DIR" "$REASONING_PREFLIGHT_DIR" "$PREFLIGHT_DIR" \
  "$CACHE_PREFLIGHT_DIR" "$PROVENANCE_DIR"

WRITE_PROBE="$DATASET_DIR/.falcon-officeqa-write-probe-${FALCON_EXP_ID:-local}-$$"
printf 'falcon OfficeQA persistent cache write probe\n' > "$WRITE_PROBE"
rm -f "$WRITE_PROBE"

snapshot_cache() {
  local output="$1"
  python3 - "$DATASET_DIR" "$output" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
files = 0
bytes_total = 0
for path in root.rglob('*'):
    if not path.is_file():
        continue
    files += 1
    try:
        bytes_total += path.stat().st_size
    except OSError:
        pass
output.write_text(json.dumps({
    'path': str(root),
    'storage': 'Falcon GCSFuse mount under /models',
    'captured_at': datetime.now(timezone.utc).isoformat(),
    'file_count': files,
    'bytes': bytes_total,
}, indent=2, sort_keys=True) + '\n')
PY
}

snapshot_cache "$PROVENANCE_DIR/dataset-cache-before.json"

AGENT_CONFIG="$(python3 - "$MAX_STEPS" "$COMMAND_TIMEOUT" <<'PY'
import json
import sys
print(json.dumps({
    'mode': 'native',
    'strategy': 'function_calling',
    'max_steps': int(sys.argv[1]),
    'command_timeout': int(sys.argv[2]),
}, separators=(',', ':')))
PY
)"
GENERATION_CONFIG="$(python3 - "$MAX_TOKENS" "$SEED" "$ENABLE_THINKING" "$API_TIMEOUT" <<'PY'
import json
import sys
print(json.dumps({
    'temperature': 0.0,
    'top_p': 1.0,
    'max_tokens': int(sys.argv[1]),
    'seed': int(sys.argv[2]),
    'timeout': int(sys.argv[4]),
    'extra_body': {
        'chat_template_kwargs': {
            'enable_thinking': sys.argv[3] == 'true',
        },
    },
}, separators=(',', ':')))
PY
)"

python3 - "$PROVENANCE_DIR/run-manifest.json" "$BASE_URL" "$MODEL_PATH" "$MODEL_ID" \
  "$DATASET_DIR" "$LIMIT" "$EVAL_BATCH_SIZE" "$AGENT_CONFIG" "$GENERATION_CONFIG" <<'PY'
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

output = pathlib.Path(sys.argv[1])
payload = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'falcon': {
        'exp_id': os.environ.get('FALCON_EXP_ID'),
        'job_id': os.environ.get('FALCON_JOB_ID'),
        'rank': os.environ.get('FALCON_RANK', os.environ.get('FALCON_JAX_PROCESS_ID', '0')),
    },
    'source': {
        'sglang_jax_commit': os.environ.get('SOURCE_COMMIT'),
        'evalscope_commit': os.environ.get('EVALSCOPE_COMMIT'),
    },
    'service': {
        'api_url': sys.argv[2],
        'model_path': sys.argv[3],
        'model_id': sys.argv[4],
        'tool_call_parser': os.environ.get('GLM52_TOOL_CALL_PARSER', 'glm47'),
        'reasoning_parser': os.environ.get('GLM52_REASONING_PARSER', 'glm45'),
        'dp_schedule_policy': os.environ.get('GLM52_DP_SCHEDULE_POLICY', 'cache_aware'),
        'radix_cache_enabled': True,
        'cache_report_enabled': True,
    },
    'evaluation': {
        'framework': 'EvalScope',
        'dataset': 'officeqa',
        'subset': 'officeqa_pro',
        'dataset_dir': sys.argv[5],
        'dataset_storage': 'persistent GCSFuse mount',
        'limit': int(sys.argv[6]),
        'eval_batch_size': int(sys.argv[7]),
        'agent_config': json.loads(sys.argv[8]),
        'bash_output_policy': 'unmodified',
        'generation_config': json.loads(sys.argv[9]),
        'debug': os.environ.get('EVALSCOPE_DEBUG', '1') == '1',
        'collect_perf': True,
        'progress_tracker': True,
        'judge_strategy': 'rule',
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY

printf 'GLM52_EVALSCOPE_REASONING_PREFLIGHT model=%s base_url=%s output=%s\n' \
  "$MODEL_PATH" "$BASE_URL" "$REASONING_PREFLIGHT_DIR"
python3 benchmark/glm52/delivery/validation/validate_openai_reasoning.py \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --output-dir "$REASONING_PREFLIGHT_DIR"

printf 'GLM52_EVALSCOPE_TOOL_PREFLIGHT model=%s base_url=%s output=%s\n' \
  "$MODEL_PATH" "$BASE_URL" "$PREFLIGHT_DIR"
python3 benchmark/glm52/delivery/validation/validate_openai_tool_call.py \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --output-dir "$PREFLIGHT_DIR"

printf 'GLM52_EVALSCOPE_PREFIX_CACHE_PREFLIGHT model=%s base_url=%s output=%s\n' \
  "$MODEL_PATH" "$BASE_URL" "$CACHE_PREFLIGHT_DIR"
python3 benchmark/glm52/delivery/validation/validate_prefix_cache_hit.py \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --output-dir "$CACHE_PREFLIGHT_DIR"

if [[ -z "${SERVER_LOG:-}" || ! -f "$SERVER_LOG" ]]; then
  printf 'SERVER_LOG must point to the active server log for cache auditing\n' >&2
  exit 2
fi
SERVER_LOG_START_LINE="$(wc -l < "$SERVER_LOG")"
printf '%s\n' "$SERVER_LOG_START_LINE" > "$PROVENANCE_DIR/eval-server-log-start-line.txt"

printf 'GLM52_EVALSCOPE_RUN dataset=%s subset=officeqa_pro limit=%s batch=%s model=%s cache=%s output=%s\n' \
  "$DATASET" "$LIMIT" "$EVAL_BATCH_SIZE" "$MODEL_PATH" "$DATASET_DIR" "$WORK_DIR"

EVAL_ARGS=(
  eval
  --model "$MODEL_PATH"
  --model-id "$MODEL_ID"
  --api-url "$BASE_URL"
  --api-key EMPTY
  --datasets officeqa
  --dataset-dir "$DATASET_DIR"
  --agent-config "$AGENT_CONFIG"
  --limit "$LIMIT"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --work-dir "$WORK_DIR"
  --no-timestamp
  --generation-config "$GENERATION_CONFIG"
  --seed "$SEED"
  --judge-strategy rule
  --collect-perf
  --enable-progress-tracker
)
if [[ "$DEBUG" == "1" ]]; then
  EVAL_ARGS+=(--debug)
fi
evalscope "${EVAL_ARGS[@]}"

python3 benchmark/glm52/delivery/validation/audit_server_prefix_cache.py \
  --server-log "$SERVER_LOG" \
  --start-line "$SERVER_LOG_START_LINE" \
  --expected-min-hits "$LIMIT" \
  --output "$OUT_DIR/server-prefix-cache-audit.json"

snapshot_cache "$PROVENANCE_DIR/dataset-cache-after.json"
TRACE_AUDIT_ARGS=(
  --work-dir "$WORK_DIR"
  --expected-samples "$LIMIT"
  --expected-max-steps "$MAX_STEPS"
  --require-tools
  --output "$OUT_DIR/agent-trace-audit.json"
)
if [[ "$ENABLE_THINKING" == "true" ]]; then
  TRACE_AUDIT_ARGS+=(--require-reasoning-separation)
fi
python3 benchmark/glm52/delivery/validation/audit_evalscope_agent_trace.py \
  "${TRACE_AUDIT_ARGS[@]}"

python3 - "$OUT_DIR" "$OUT_DIR/artifact-inventory.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
files = []
for path in sorted(root.rglob('*')):
    if path.is_file() and path != output:
        files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size})
output.write_text(json.dumps({'root': str(root), 'files': files}, indent=2, sort_keys=True) + '\n')
PY

printf 'GLM52_EVALSCOPE_OK dataset=%s samples=%s trace_audit=%s\n' \
  "$DATASET" "$LIMIT" "$OUT_DIR/agent-trace-audit.json"
