#!/bin/bash
set -euxo pipefail

pkill -f "sgl_jax.launch_server|sglang::scheduler|sglang-jax::detokenizer" || true
sleep 2
pkill -9 -f "sgl_jax.launch_server|sglang::scheduler|sglang-jax::detokenizer" || true
sleep 2

for _ in $(seq 1 30); do
  if ! ss -ltnp 2>/dev/null | grep -q ':30271 '; then
    break
  fi
  sleep 1
done

if ss -ltnp 2>/dev/null | grep -q ':30271 '; then
  echo "port 30271 is still occupied after cleanup" >&2
  ss -ltnp 2>/dev/null | grep ':30271 ' >&2 || true
  exit 1
fi

cd ~/sky_workdir
REPO_DIR=~/sky_workdir/sgl-jax

rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude 'sgl-jax/' \
  ./ "${REPO_DIR}/"

cd "${REPO_DIR}"
source .venv/bin/activate

export PYTHONPATH="${REPO_DIR}/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export SGLANG_HEALTH_CHECK_TIMEOUT="${SGLANG_HEALTH_CHECK_TIMEOUT:-1800}"
export SGLANG_WARMUP_REQUEST_TIMEOUT="${SGLANG_WARMUP_REQUEST_TIMEOUT:-3600}"
export SGLANG_WAIT_FOR_MODEL_INFO_TIMEOUT="${SGLANG_WAIT_FOR_MODEL_INFO_TIMEOUT:-1800}"
export SGLANG_MODEL_INFO_HTTP_TIMEOUT="${SGLANG_MODEL_INFO_HTTP_TIMEOUT:-30}"
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}
WARMUPS=${WARMUPS-}

MODEL_PATH=${MODEL_PATH:-/models/MiMo-V2-Flash}
MODEL_PATH_FALLBACK_REPO_ID=${MODEL_PATH_FALLBACK_REPO_ID:-XiaomiMiMo/MiMo-V2-Flash}
AUTO_FALLBACK_TO_MODELSCOPE=${AUTO_FALLBACK_TO_MODELSCOPE:-1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-mimo-v2-flash}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-30271}
TP_SIZE=${TP_SIZE:-16}
EP_SIZE=${EP_SIZE:-16}
MOE_BACKEND=${MOE_BACKEND:-epmoe}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-128}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
PAGE_SIZE=${PAGE_SIZE:-1}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-}
WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT:-3600}
DISABLE_PRECOMPILE=${DISABLE_PRECOMPILE:-1}
SKIP_SERVER_WARMUP=${SKIP_SERVER_WARMUP:-1}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-}
SPECULATIVE_ALGORITHM=${SPECULATIVE_ALGORITHM:-}
SPECULATIVE_NUM_STEPS=${SPECULATIVE_NUM_STEPS:-3}
SPECULATIVE_EAGLE_TOPK=${SPECULATIVE_EAGLE_TOPK:-1}
SPECULATIVE_NUM_DRAFT_TOKENS=${SPECULATIVE_NUM_DRAFT_TOKENS:-4}

NUM_NODES=${SKYPILOT_NUM_NODES:-1}
NODE_RANK=${SKYPILOT_NODE_RANK:-0}
HEAD_IP=$(echo "${SKYPILOT_NODE_IPS:-}" | head -1 | awk '{print $1}')
HEAD_IP=${HEAD_IP:-$(hostname -I | awk '{print $1}')}
LOG=${HOME}/mimo_sink_debug_node${NODE_RANK}.log

rm -f "${LOG}"

if [[ "${AUTO_FALLBACK_TO_MODELSCOPE}" == "1" ]] \
  && [[ "${MODEL_PATH}" == /* ]] \
  && [[ ! -f "${MODEL_PATH}/config.json" ]] \
  && [[ -n "${MODEL_PATH_FALLBACK_REPO_ID}" ]]; then
  export SGLANG_USE_MODELSCOPE=true
  MODEL_PATH="${MODEL_PATH_FALLBACK_REPO_ID}"
  echo "node_rank=${NODE_RANK} local model missing, falling back to ModelScope repo ${MODEL_PATH}"
fi

server_args=(
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --tp-size "${TP_SIZE}"
  --ep-size "${EP_SIZE}"
  --moe-backend "${MOE_BACKEND}"
  --nnodes "${NUM_NODES}"
  --node-rank "${NODE_RANK}"
  --dist-init-addr "${HEAD_IP}:10011"
  --host "${HOST}"
  --port "${PORT}"
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --page-size "${PAGE_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --watchdog-timeout "${WATCHDOG_TIMEOUT}"
  --log-level info
)

if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  server_args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi

if [[ -n "${MAX_PREFILL_TOKENS}" ]]; then
  server_args+=(--max-prefill-tokens "${MAX_PREFILL_TOKENS}")
fi

if [[ -n "${REASONING_PARSER}" ]]; then
  server_args+=(--reasoning-parser "${REASONING_PARSER}")
fi

if [[ -n "${TOOL_CALL_PARSER}" ]]; then
  server_args+=(--tool-call-parser "${TOOL_CALL_PARSER}")
fi

if [[ "${DISABLE_PRECOMPILE}" == "1" ]]; then
  server_args+=(--disable-precompile)
fi

if [[ "${SKIP_SERVER_WARMUP}" == "1" ]]; then
  server_args+=(--skip-server-warmup)
fi

if [[ -n "${WARMUPS}" ]]; then
  server_args+=(--warmups "${WARMUPS}")
fi

if [[ -n "${SPECULATIVE_ALGORITHM}" ]]; then
  server_args+=(
    --speculative-algorithm "${SPECULATIVE_ALGORITHM}"
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    --disable-overlap-schedule
  )
fi

if [[ -n "${EXTRA_SERVER_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_server_args=(${EXTRA_SERVER_ARGS})
  server_args+=("${extra_server_args[@]}")
fi

nohup python -u -m sgl_jax.launch_server \
  "${server_args[@]}" \
  > "${LOG}" 2>&1 < /dev/null &

echo $! > "${HOME}/mimo_sink_debug_node${NODE_RANK}.pid"
echo "started node_rank=${NODE_RANK} nnodes=${NUM_NODES} head=${HEAD_IP} pid=$(cat "${HOME}/mimo_sink_debug_node${NODE_RANK}.pid") log=${LOG}"
