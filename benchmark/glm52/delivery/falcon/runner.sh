#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

if [[ -z "${SOURCE_COMMIT:-}" && -z "${SOURCE_SHA256:-}" ]]; then
  printf 'Falcon manifest must pin SOURCE_COMMIT or SOURCE_SHA256\n' >&2
  exit 2
fi
: "${GLM52_PHYSICAL_CHIPS:?expected 8 or 16}"
: "${GLM52_QUANTIZATION:?expected blockwise or channelwise}"
RUN_MODE="${RUN_MODE:-benchmark}"
case "$RUN_MODE" in
  benchmark|profile|eval|agent_eval) ;;
  *) printf 'unsupported RUN_MODE=%s\n' "$RUN_MODE" >&2; exit 2 ;;
esac

export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/tpu_logs/jax-compilation-cache}"

RANK="${FALCON_JAX_PROCESS_ID:-${FALCON_RANK:-${JOB_COMPLETION_INDEX:-0}}}"
case "$GLM52_PHYSICAL_CHIPS" in
  8) WORLD="${FALCON_WORLD_SIZE:-2}" ;;
  16) WORLD="${FALCON_WORLD_SIZE:-4}" ;;
  *) printf 'unsupported GLM52_PHYSICAL_CHIPS=%s\n' "$GLM52_PHYSICAL_CHIPS" >&2; exit 2 ;;
esac

POD_HOST="$(hostname)"
JOB_BASE="${POD_HOST%-*}"
MASTER_ADDR="$JOB_BASE-0.$JOB_BASE"
OUT="${ARTIFACT_LOCAL_DIR:-/tmp/glm52-delivery-artifact}"
RANK_OUT="$OUT/rank-$RANK"
LOCAL_ROOT="/tmp/tpu_logs/glm52-${GLM52_QUANTIZATION}-${GLM52_PHYSICAL_CHIPS}chip-${RUN_MODE}-rank$RANK"
SERVER_LOG="$LOCAL_ROOT/server.log"
SERVER_PID=0
TAIL_PID=0

case "$GLM52_QUANTIZATION" in
  blockwise)
    MODEL_PATH="${MODEL_PATH:-/models/GLM-5.2-FP8}"
    ;;
  channelwise)
    MODEL_PATH="${MODEL_PATH:-/models/GLM5.2-fp8-channel-wise}"
    ;;
  *)
    printf 'unsupported GLM52_QUANTIZATION=%s\n' "$GLM52_QUANTIZATION" >&2
    exit 2
    ;;
esac

mkdir -p "$RANK_OUT" "$OUT/workload" "$OUT/profiling" "$LOCAL_ROOT" \
  "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$JAX_COMPILATION_CACHE_DIR"
if [[ "$RANK" == "0" ]]; then
  rm -f "$OUT/ALL_DONE"
fi

cd /workspace/sglang-jax
if [[ -n "${SOURCE_COMMIT:-}" ]]; then
  test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
  git rev-parse HEAD > "$RANK_OUT/source-revision.txt"
  SOURCE_ID="$SOURCE_COMMIT"
else
  printf 'sha256:%s\n' "$SOURCE_SHA256" > "$RANK_OUT/source-revision.txt"
  SOURCE_ID="sha256:$SOURCE_SHA256"
fi
if [[ "$RUN_MODE" == "agent_eval" ]]; then
  : "${EVALSCOPE_COMMIT:?Falcon manifest must pin EVALSCOPE_COMMIT for agent_eval}"
  EVALSCOPE_GITLINK="$(git ls-tree HEAD third_party/evalscope | awk '{print $3}')"
  if [[ "$EVALSCOPE_GITLINK" != "$EVALSCOPE_COMMIT" ]]; then
    printf 'EvalScope gitlink mismatch: manifest=%s source=%s\n' \
      "$EVALSCOPE_COMMIT" "$EVALSCOPE_GITLINK" >&2
    exit 2
  fi
  printf '%s\n' "$EVALSCOPE_GITLINK" > "$RANK_OUT/evalscope-gitlink.txt"
fi

python3 -m pip install -q --upgrade pip uv
uv venv --python 3.12 --seed "$LOCAL_ROOT/venv"
. "$LOCAL_ROOT/venv/bin/activate"
uv pip install -q -e './python[tpu]'
if [[ "$RUN_MODE" == "eval" ]]; then
  SGL_EVAL_COMMIT="${SGL_EVAL_COMMIT:-32fa49229575e433629c37379821b5a589a2e422}"
  uv pip install -q "sgl-eval @ git+https://github.com/sgl-project/sgl-eval.git@${SGL_EVAL_COMMIT}"
fi
export PYTHONPATH=/workspace/sglang-jax/python${PYTHONPATH:+:$PYTHONPATH}

python3 - "$MODEL_PATH" "$GLM52_QUANTIZATION" > "$RANK_OUT/model-check.json" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
quantization = sys.argv[2]
config_path = root / "config.json"
index_path = root / "model.safetensors.index.json"
missing = [str(path) for path in (config_path, index_path) if not path.is_file()]
if missing:
    raise SystemExit(f"incomplete checkpoint: {missing}")
config = json.loads(config_path.read_text())
index = json.loads(index_path.read_text())
shards = sorted(set(index["weight_map"].values()))
missing_shards = [name for name in shards if not (root / name).is_file()]
if missing_shards:
    raise SystemExit(f"missing checkpoint shards: {missing_shards[:10]}")
print(json.dumps({
    "root": str(root),
    "delivery_quantization": quantization,
    "checkpoint_quantization_config": config.get("quantization_config"),
    "max_position_embeddings": config.get("max_position_embeddings"),
    "shard_count": len(shards),
}, indent=2, sort_keys=True))
PY

stop_process() {
  local pid="$1"
  if (( pid <= 0 )); then
    return
  fi
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  set +e
  if [[ "$RANK" == "0" ]]; then
    touch "$OUT/ALL_DONE"
  fi
  stop_process "$SERVER_PID"
  stop_process "$TAIL_PID"
  cp "$SERVER_LOG" "$RANK_OUT/server.log" 2>/dev/null || true
}
trap cleanup EXIT

if [[ "$RUN_MODE" == "profile" ]]; then
  export SGLANG_PROFILE_MAX_HOSTS=1
  export SGLANG_PROFILE_NUM_CHIPS_PER_TASK=1
  export SGLANG_PROFILE_NUM_SPARSE_CORES_TO_TRACE=1
  export SGLANG_PROFILE_NUM_SPARSE_CORE_TILES_TO_TRACE=1
fi

MOE_BACKEND="${GLM52_MOE_BACKEND:-fused_v2}"
case "$MOE_BACKEND" in
  fused_v2) ;;
  fused_rs)
    if [[ "$GLM52_QUANTIZATION" != "channelwise" || "$GLM52_PHYSICAL_CHIPS" != "16" ]]; then
      printf 'fused_rs delivery is only validated for channelwise 16-chip GLM-5.2\n' >&2
      exit 2
    fi
    ;;
  *)
    printf 'unsupported GLM52_MOE_BACKEND=%s\n' "$MOE_BACKEND" >&2
    exit 2
    ;;
esac
SERVE_SCRIPT="benchmark/glm52/delivery/serve/${GLM52_QUANTIZATION}_${GLM52_PHYSICAL_CHIPS}chip.sh"
SERVE_EXTRA_ARGS=()
if [[ "$RUN_MODE" == "agent_eval" ]]; then
  SERVE_EXTRA_ARGS+=(
    --dp-schedule-policy "${GLM52_DP_SCHEDULE_POLICY:-cache_aware}"
    --enable-cache-report
    --tool-call-parser "${GLM52_TOOL_CALL_PARSER:-glm47}"
    --reasoning-parser "${GLM52_REASONING_PARSER:-glm45}"
  )
fi
printf 'GLM52_FALCON_DELIVERY_START quantization=%s physical_chips=%s mode=%s rank=%s world=%s source=%s moe_backend=%s serve_script=%s\n' \
  "$GLM52_QUANTIZATION" "$GLM52_PHYSICAL_CHIPS" "$RUN_MODE" "$RANK" "$WORLD" \
  "$SOURCE_ID" "$MOE_BACKEND" "$SERVE_SCRIPT"
GLM52_SERVER_LOG=/dev/stdout \
GLM52_MOE_BACKEND="$MOE_BACKEND" \
WORLD="$WORLD" RANK="$RANK" MASTER_ADDR="$MASTER_ADDR" MODEL_PATH="$MODEL_PATH" \
  "$SERVE_SCRIPT" "${SERVE_EXTRA_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
tail -n +1 -F "$SERVER_LOG" &
TAIL_PID=$!

if [[ "$RANK" != "0" ]]; then
  while [[ ! -f "$OUT/ALL_DONE" ]]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      printf 'GLM52_FALCON_DELIVERY_SERVER_DIED rank=%s\n' "$RANK" >&2
      exit 1
    fi
    sleep 15
  done
  printf 'GLM52_FALCON_DELIVERY_RANK_COMPLETE rank=%s\n' "$RANK"
  exit 0
fi

READY=0
for _ in $(seq 1 1080); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'GLM52_FALCON_DELIVERY_SERVER_DIED_BEFORE_READY rank=0\n' >&2
    exit 1
  fi
  if curl -sf http://localhost:30000/get_server_info > "$OUT/server-info-ready.json"; then
    READY=1
    break
  fi
  sleep 10
done
if [[ "$READY" != "1" ]]; then
  printf 'GLM52_FALCON_DELIVERY_SERVER_READY_TIMEOUT seconds=10800\n' >&2
  exit 1
fi
if [[ "$RUN_MODE" == "agent_eval" ]]; then
  python3 - "$OUT/server-info-ready.json" "$OUT/evalscope-cache-config-audit.json" \
    "${GLM52_CONTEXT_LENGTH:-135168}" "${MEM_FRACTION_STATIC:-0.88}" \
    "${GLM52_MAX_RUNNING_REQUESTS:-32}" "${GLM52_KV_HEADROOM_TOKENS:-8192}" <<'PY'
import json
import pathlib
import sys

server_info_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
expected_context_length = int(sys.argv[3])
expected_mem_fraction_static = float(sys.argv[4])
expected_max_running_requests = int(sys.argv[5])
minimum_kv_headroom_tokens = int(sys.argv[6])
server_info = json.loads(server_info_path.read_text())
dp_size = int(server_info.get("dp_size") or 0)
max_total_num_tokens = int(server_info.get("max_total_num_tokens") or 0)
per_dp_token_capacity = max_total_num_tokens // dp_size if dp_size else 0
report = {
    "dp_schedule_policy": server_info.get("dp_schedule_policy"),
    "disable_radix_cache": server_info.get("disable_radix_cache"),
    "enable_cache_report": server_info.get("enable_cache_report"),
    "context_length": server_info.get("context_length"),
    "expected_context_length": expected_context_length,
    "mem_fraction_static": server_info.get("mem_fraction_static"),
    "expected_mem_fraction_static": expected_mem_fraction_static,
    "max_running_requests": server_info.get("max_running_requests"),
    "expected_max_running_requests": expected_max_running_requests,
    "max_req_input_len": server_info.get("max_req_input_len"),
    "max_total_num_tokens": max_total_num_tokens,
    "dp_size": dp_size,
    "per_dp_token_capacity": per_dp_token_capacity,
    "minimum_kv_headroom_tokens": minimum_kv_headroom_tokens,
    "kv_headroom_tokens": per_dp_token_capacity - expected_context_length,
}
report["passed"] = (
    report["dp_schedule_policy"] == "cache_aware"
    and report["disable_radix_cache"] is False
    and report["enable_cache_report"] is True
    and report["context_length"] == expected_context_length
    and abs(float(report["mem_fraction_static"]) - expected_mem_fraction_static) < 1e-9
    and report["max_running_requests"] == expected_max_running_requests
    and int(report["max_req_input_len"] or 0) >= expected_context_length - 8
    and report["kv_headroom_tokens"] >= minimum_kv_headroom_tokens
)
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print("GLM52_EVALSCOPE_CACHE_CONFIG_AUDIT", json.dumps(report, sort_keys=True))
if not report["passed"]:
    raise SystemExit("agent_eval server configuration or KV headroom audit failed")
PY

  LONG_CONTEXT_DIR="$OUT/long-context-preflight"
  printf 'GLM52_EVALSCOPE_LONG_CONTEXT_PREFLIGHT target_tokens=%s max_tokens=%s output=%s\n' \
    "${GLM52_LONG_CONTEXT_PROBE_TOKENS:-150000}" "${EVALSCOPE_MAX_TOKENS:-4096}" \
    "$LONG_CONTEXT_DIR"
  python3 benchmark/glm52/delivery/validation/validate_openai_long_context.py \
    --base-url "http://localhost:${PORT:-30000}/v1" \
    --model "$MODEL_PATH" \
    --output-dir "$LONG_CONTEXT_DIR" \
    --target-prompt-tokens "${GLM52_LONG_CONTEXT_PROBE_TOKENS:-150000}" \
    --max-tokens "${EVALSCOPE_MAX_TOKENS:-4096}" \
    --timeout "${EVALSCOPE_API_TIMEOUT:-3600}" \
    --flush-url "http://localhost:${PORT:-30000}/flush_cache"
fi

if [[ "$RUN_MODE" == "profile" ]]; then
  env \
    QUANTIZATION="$GLM52_QUANTIZATION" \
    SERVER_LOG="$SERVER_LOG" \
    OUTPUT="$OUT/workload/profile-metrics.jsonl" \
    PROFILE_OUTPUT_DIR="$OUT/profiling/glm52-${GLM52_QUANTIZATION}-${GLM52_PHYSICAL_CHIPS}chip" \
    "benchmark/glm52/delivery/benchmark/run_${GLM52_PHYSICAL_CHIPS}chip.sh" \
    2>&1 | tee "$OUT/workload/$RUN_MODE.log"
elif [[ "$RUN_MODE" == "benchmark" ]]; then
  env \
    QUANTIZATION="$GLM52_QUANTIZATION" \
    SERVER_LOG="$SERVER_LOG" \
    OUTPUT="$OUT/workload/benchmark-metrics.jsonl" \
    "benchmark/glm52/delivery/benchmark/run_${GLM52_PHYSICAL_CHIPS}chip.sh" \
    2>&1 | tee "$OUT/workload/$RUN_MODE.log"
elif [[ "$RUN_MODE" == "eval" ]]; then
  env \
    MODEL_PATH="$MODEL_PATH" \
    OUT_ROOT="$OUT/eval" \
    EVAL_SCOPE="${EVAL_SCOPE:-quick}" \
    "benchmark/glm52/delivery/eval/run.sh" "${EVAL_DATASET:-gsm8k}" \
    2>&1 | tee "$OUT/workload/$RUN_MODE.log"
else
  EVALSCOPE_VENV="$LOCAL_ROOT/evalscope-venv"
  mkdir -p "$OUT/evalscope/provenance"
  git submodule sync -- third_party/evalscope
  git submodule update --init --recursive -- third_party/evalscope
  test "$(git -C third_party/evalscope rev-parse HEAD)" = "$EVALSCOPE_COMMIT"
  git -C third_party/evalscope rev-parse HEAD > "$OUT/evalscope/provenance/evalscope-revision.txt"
  git submodule status third_party/evalscope > "$OUT/evalscope/provenance/submodule-status.txt"

  uv venv --python 3.12 --seed "$EVALSCOPE_VENV"
  uv pip install -q --python "$EVALSCOPE_VENV/bin/python" -e third_party/evalscope
  "$EVALSCOPE_VENV/bin/python" -m pip freeze > "$OUT/evalscope/provenance/pip-freeze.txt"
  "$EVALSCOPE_VENV/bin/python" - <<'PY' > "$OUT/evalscope/provenance/evalscope-version.txt"
import evalscope
print(evalscope.__version__)
PY

  env \
    PATH="$EVALSCOPE_VENV/bin:$PATH" \
    PYTHONPATH= \
    MODEL_PATH="$MODEL_PATH" \
    OUT_ROOT="$OUT/evalscope" \
    BASE_URL="http://localhost:${PORT:-30000}/v1" \
    SERVER_LOG="$SERVER_LOG" \
    GLM52_DP_SCHEDULE_POLICY="${GLM52_DP_SCHEDULE_POLICY:-cache_aware}" \
    "benchmark/glm52/delivery/evalscope/run.sh" "${EVALSCOPE_DATASET:-officeqa}" \
    2>&1 | tee "$OUT/workload/$RUN_MODE.log"
fi

curl -sf http://localhost:30000/get_server_info > "$OUT/server-info-final.json"
if [[ "$RUN_MODE" == "benchmark" || "$RUN_MODE" == "profile" ]]; then
  python3 - "$SERVER_LOG" "$OUT/workload/kernel-lookup-audit.json" <<'PY'
import json
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
known_zero_grid_key = "('decode', 'bfloat16', 'bfloat16', 64, 512, 64, 64, 2048)"
misses = [
    line for line in log_path.read_text(errors="replace").splitlines()
    if "LOOKUP MISS" in line
]
unexpected = [line for line in misses if known_zero_grid_key not in line]
result = {
    "lookup_miss_count": len(misses),
    "known_zero_grid_decode_mnt2048_count": len(misses) - len(unexpected),
    "unexpected_lookup_misses": unexpected,
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("GLM52_DELIVERY_KERNEL_LOOKUP_AUDIT", json.dumps(result, sort_keys=True))
if unexpected:
    raise SystemExit(f"unexpected kernel lookup misses: {unexpected}")
PY
fi

if [[ "$RUN_MODE" == "profile" ]]; then
  TRACE_FILES="$(find "$OUT/profiling" -type f \( -name '*.xplane.pb' -o -name '*.trace.json.gz' \) | wc -l)"
  test "$TRACE_FILES" -gt 0
  printf 'GLM52_FALCON_DELIVERY_PROFILE_TRACE_FILES count=%s\n' "$TRACE_FILES"
fi
printf 'GLM52_FALCON_DELIVERY_OK quantization=%s physical_chips=%s mode=%s\n' \
  "$GLM52_QUANTIZATION" "$GLM52_PHYSICAL_CHIPS" "$RUN_MODE"
