#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to a model visible on both hosts}"
: "${PREFILL_HOST:?set PREFILL_HOST to the prefill host DNS name or IP}"

ROLE=${ROLE:?set ROLE to prefill or decode}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
PREFILL_PORT=${PREFILL_PORT:-10000}
DECODE_PORT=${DECODE_PORT:-10001}
ROUTER_PORT=${ROUTER_PORT:-30000}
MAX_INFLIGHT=${MAX_INFLIGHT:-2}
DP_SIZE=${DP_SIZE:-1}
TP_SIZE=${TP_SIZE:-1}

if (( DP_SIZE < 1 || TP_SIZE < 1 || TP_SIZE % DP_SIZE != 0 )); then
  printf 'TP_SIZE (%s) must be positive and divisible by DP_SIZE (%s)\n' \
    "${TP_SIZE}" "${DP_SIZE}" >&2
  exit 2
fi

wait_for_health() {
  local url=$1
  for _ in $(seq 1 180); do
    curl -fsS "${url}" >/dev/null && return 0
    sleep 2
  done
  return 1
}

common_args=(
  --model-path "${MODEL_PATH}"
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --page-size 128
  --disable-radix-cache
  --disaggregation-bootstrap-url "http://${PREFILL_HOST}:${BOOTSTRAP_PORT}"
  --disaggregation-use-raiden
  --disaggregation-max-inflight-transfers "${MAX_INFLIGHT}"
)

if [[ "${ROLE}" = prefill ]]; then
  python -m sgl_jax.srt.disaggregation.run_bootstrap \
    --host 0.0.0.0 --port "${BOOTSTRAP_PORT}" &
  python -m sgl_jax.launch_server "${common_args[@]}" \
    --host 0.0.0.0 --port "${PREFILL_PORT}" \
    --disaggregation-mode prefill &
  wait_for_health "http://localhost:${PREFILL_PORT}/health"
  wait
fi

if [[ "${ROLE}" != decode ]]; then
  printf 'ROLE must be prefill or decode\n' >&2
  exit 2
fi

wait_for_health "http://${PREFILL_HOST}:${BOOTSTRAP_PORT}/health"
wait_for_health "http://${PREFILL_HOST}:${PREFILL_PORT}/health"
python -m sgl_jax.launch_server "${common_args[@]}" \
  --host 0.0.0.0 --port "${DECODE_PORT}" \
  --disaggregation-mode decode &
wait_for_health "http://localhost:${DECODE_PORT}/health"

python -m sgl_jax.srt.disaggregation.launch_router \
  --pd-disaggregation --mini-lb \
  --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
  --decode "http://localhost:${DECODE_PORT}" \
  --prefill-bootstrap-host "${PREFILL_HOST}" \
  --max-concurrent-requests "${MAX_INFLIGHT}" \
  --host 0.0.0.0 --port "${ROUTER_PORT}" &
wait_for_health "http://localhost:${ROUTER_PORT}/health"

curl -fsS "http://localhost:${ROUTER_PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8}}'
