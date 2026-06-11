#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-restart}"
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/sglang-jax}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/pc}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-10.16.2.6:30042}"
LOG_PREFIX="${LOG_PREFIX:-/tmp/spec-relay-server}"
PRECOMPILE_TOKEN_PADDINGS="${PRECOMPILE_TOKEN_PADDINGS:-8192}"
IFS=', ' read -r -a PRECOMPILE_TOKEN_PADDING_ARGS <<<"${PRECOMPILE_TOKEN_PADDINGS}"

PODS=(
  "perf-16-0-jgb5c"
  "perf-16-1-zhn6d"
  "perf-16-2-hs7kc"
  "perf-16-3-vqw2x"
)

FILES_TO_SYNC=(
  "python/sgl_jax/srt/managers/scheduler.py"
  "python/sgl_jax/srt/managers/scheduler_output_processor_mixin.py"
  "python/sgl_jax/srt/managers/schedule_batch.py"
  "python/sgl_jax/srt/layers/attention/flashattention_backend.py"
  "python/sgl_jax/srt/speculative/draft_extend_fused.py"
  "python/sgl_jax/srt/speculative/base_worker.py"
  "python/sgl_jax/srt/speculative/eagle_util.py"
  "python/sgl_jax/srt/speculative/eagle_worker.py"
  "python/sgl_jax/srt/speculative/overlap_worker.py"
  "python/sgl_jax/srt/speculative/relay_buffer.py"
)

SERVER_ARGS=(
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --tp-size 16
  --dp-size 4
  --ep-size 16
  --moe-backend epmoe
  --nnodes 4
  --dist-init-addr "${DIST_INIT_ADDR}"
  --host 0.0.0.0
  --port 30271
  --page-size 256
  --context-length 262144
  --disable-radix-cache
  --chunked-prefill-size 2048
  --dtype bfloat16
  --mem-fraction-static 0.88
  --swa-full-tokens-ratio 0.2
  --skip-server-warmup
  --log-level info
  --decode-log-interval 1
  --max-running-requests 128
  --dp-schedule-policy round_robin
  --speculative-algorithm NEXTN
  --speculative-eagle-topk 1
  --speculative-num-steps 3
  --speculative-num-draft-tokens 4
  --precompile-bs-paddings 128
  --precompile-token-paddings "${PRECOMPILE_TOKEN_PADDING_ARGS[@]}"
)

usage() {
  cat <<'USAGE'
Usage:
  scripts/spec_relay_pod_server.sh restart
  scripts/spec_relay_pod_server.sh start
  scripts/spec_relay_pod_server.sh stop
  scripts/spec_relay_pod_server.sh status
  scripts/spec_relay_pod_server.sh sync

Environment overrides:
  MODEL_PATH=/data/pc
  DIST_INIT_ADDR=10.16.2.6:30042
  LOG_PREFIX=/tmp/spec-relay-server
  PRECOMPILE_TOKEN_PADDINGS=8192
  REMOTE_ROOT=/tmp/sglang-jax
  PYTHON_BIN=/opt/venv/bin/python
  SGL_JAX_SPEC_RELAY_VERIFY_DEBUG=0
USAGE
}

kubectl_exec() {
  local pod="$1"
  shift
  kubectl exec -i "${pod}" -- "$@"
}

sync_code() {
  local pod file
  for pod in "${PODS[@]}"; do
    for file in "${FILES_TO_SYNC[@]}"; do
      echo "[sync] ${file} -> ${pod}:${REMOTE_ROOT}/${file}"
      kubectl cp "${file}" "${pod}:${REMOTE_ROOT}/${file}" >/dev/null
    done
  done
}

stop_one() {
  local pod="$1"
  echo "[stop] ${pod}"
  kubectl_exec "${pod}" bash -s -- <<'REMOTE'
set -euo pipefail
pids="$(
  ps -ww -eo pid=,args= \
    | awk '
      /\/opt\/venv\/bin\/python -u -m sgl_jax\.launch_server/ {print $1}
      /bash -lc cd \/tmp\/sglang-jax .*sgl_jax\.launch_server.*nohup/ {print $1}
    ' \
    | sort -u
)"
if [ -n "${pids}" ]; then
  echo "${pids}" | xargs -r kill -9
fi
REMOTE
}

stop_all() {
  local pod
  for pod in "${PODS[@]}"; do
    stop_one "${pod}" || true
  done
}

start_one() {
  local pod="$1"
  local rank="$2"
  local log="${LOG_PREFIX}-rank${rank}.log"
  echo "[start] rank${rank} ${pod} log=${log}"
  kubectl_exec "${pod}" bash -s -- \
    "${REMOTE_ROOT}" "${log}" "${PYTHON_BIN}" "${rank}" \
    "${SGL_JAX_SPEC_RELAY_VERIFY_DEBUG:-0}" \
    "${SERVER_ARGS[@]}" <<'REMOTE'
set -euo pipefail
remote_root="$1"
log="$2"
python_bin="$3"
rank="$4"
verify_debug="$5"
shift 5
cd "${remote_root}"
export PYTHONPATH="${remote_root}/python"
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
export JAX_EXPLAIN_CACHE_MISSES=TRUE
export SGL_JAX_SPEC_RELAY_VERIFY_DEBUG="${verify_debug}"
rm -f "${log}"
nohup "${python_bin}" -u -m sgl_jax.launch_server \
  --node-rank "${rank}" \
  "$@" >"${log}" 2>&1 </dev/null &
echo "$!" > "${log}.launcher.pid"
REMOTE
}

start_all() {
  local i
  for i in "${!PODS[@]}"; do
    start_one "${PODS[$i]}" "${i}"
  done
}

status_all() {
  local i pod log
  for i in "${!PODS[@]}"; do
    pod="${PODS[$i]}"
    log="${LOG_PREFIX}-rank${i}.log"
    echo "===== ${pod} rank${i} ====="
    kubectl_exec "${pod}" bash -s -- "${log}" <<'REMOTE' || true
set -euo pipefail
log="$1"
ps -ww -eo pid=,args= \
  | awk '/\/opt\/venv\/bin\/python -u -m sgl_jax\.launch_server/ {print}'
if [ -f "${log}" ]; then
  tail -n 12 "${log}"
else
  echo "missing log: ${log}"
fi
REMOTE
  done
}

case "${ACTION}" in
  restart)
    sync_code
    stop_all
    sleep 2
    start_all
    sleep 2
    status_all
    ;;
  start)
    start_all
    sleep 2
    status_all
    ;;
  stop)
    stop_all
    ;;
  status)
    status_all
    ;;
  sync)
    sync_code
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 2
    ;;
esac
