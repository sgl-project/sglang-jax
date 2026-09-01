#!/usr/bin/env bash
#
# Launch a local, CPU-only EPD topology (N encoder servers + 1 combined
# language server) with --simulate-compute so device forward and the Raiden
# embedding transfer are replaced by modeled sleeps. Lets you reproduce and
# profile EPD orchestration latency on a laptop without TPU / Falcon.
#
# Usage:
#   MODEL_PATH=/path/to/qwen2.5-vl ./scripts/disaggregation/run_epd_cpu_sim.sh
#
# Then, in another shell, drive + profile it:
#   python scripts/disaggregation/profile_epd_cpu_sim.py --image <url-or-path>
#
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to an in-model multimodal arch (e.g. Qwen2.5-VL); weights are dummy}"

NUM_ENCODERS=${NUM_ENCODERS:-1}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}
# The language scheduler builds a (dp, tp/dp) mesh, so the CPU device count must
# equal tp*dp exactly. Default to that; override only if you know why.
DEVICE_COUNT=${DEVICE_COUNT:-$((TP_SIZE * DP_SIZE))}
ENCODER_PORT_BASE=${ENCODER_PORT_BASE:-31001}
LANG_PORT=${LANG_PORT:-30000}
PROFILER_DIR=${PROFILER_DIR:-/tmp/epd-sim-profile}

# Linear sleep coefficients (ms). Default 0 => no artificial latency (pure
# orchestration). Calibrate from a real TPU run to model realistic timing.
SIM_ENC_BASE_MS=${SIM_ENC_BASE_MS:-0}
SIM_ENC_MS_PER_TOKEN=${SIM_ENC_MS_PER_TOKEN:-0}
SIM_PREFILL_BASE_MS=${SIM_PREFILL_BASE_MS:-0}
SIM_PREFILL_MS_PER_TOKEN=${SIM_PREFILL_MS_PER_TOKEN:-0}
SIM_DECODE_BASE_MS=${SIM_DECODE_BASE_MS:-0}
SIM_DECODE_MS_PER_SEQ=${SIM_DECODE_MS_PER_SEQ:-0}
SIM_TRANSFER_MS_PER_MB=${SIM_TRANSFER_MS_PER_MB:-0}
SIM_NET_RTT_MS=${SIM_NET_RTT_MS:-0}

# CPU device simulation. MUST be exported before any process imports jax.
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=${DEVICE_COUNT} ${XLA_FLAGS:-}"
export SGLANG_JAX_PROFILER_DIR="${PROFILER_DIR}"
mkdir -p "${PROFILER_DIR}"

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local url=$1
  for _ in $(seq 1 240); do
    curl -fsS "${url}" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "timed out waiting for ${url}" >&2
  return 1
}

sim_args=(
  --simulate-compute
  --simulate-compute-encoder-base-ms "${SIM_ENC_BASE_MS}"
  --simulate-compute-encoder-ms-per-token "${SIM_ENC_MS_PER_TOKEN}"
  --simulate-compute-prefill-base-ms "${SIM_PREFILL_BASE_MS}"
  --simulate-compute-prefill-ms-per-token "${SIM_PREFILL_MS_PER_TOKEN}"
  --simulate-compute-decode-base-ms "${SIM_DECODE_BASE_MS}"
  --simulate-compute-decode-ms-per-seq "${SIM_DECODE_MS_PER_SEQ}"
  --simulate-transfer-ms-per-mb "${SIM_TRANSFER_MS_PER_MB}"
  --simulate-network-rtt-ms "${SIM_NET_RTT_MS}"
)

common_args=(
  --model-path "${MODEL_PATH}"
  --tp-size "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --device cpu
  --load-format dummy
  --dtype bfloat16
  --attention-backend native
  --trust-remote-code
  --disaggregation-host-ip 127.0.0.1
)

ENCODER_URLS=()
for ((i = 0; i < NUM_ENCODERS; i++)); do
  port=$((ENCODER_PORT_BASE + i))
  echo ">> starting encoder ${i} on :${port}"
  python -m sgl_jax.launch_server "${common_args[@]}" "${sim_args[@]}" \
    --encoder-only --disable-precompile \
    --host 127.0.0.1 --port "${port}" &
  PIDS+=($!)
  ENCODER_URLS+=("http://127.0.0.1:${port}")
done

for url in "${ENCODER_URLS[@]}"; do
  wait_for_health "${url}/health"
  echo ">> encoder healthy: ${url}"
done

echo ">> starting language server on :${LANG_PORT}"
python -m sgl_jax.launch_server "${common_args[@]}" "${sim_args[@]}" \
  --language-only --encoder-urls "${ENCODER_URLS[@]}" \
  --host 127.0.0.1 --port "${LANG_PORT}" &
PIDS+=($!)
wait_for_health "http://127.0.0.1:${LANG_PORT}/health"

echo ""
echo "=========================================================="
echo "EPD CPU sim ready."
echo "  language:  http://127.0.0.1:${LANG_PORT}"
echo "  encoders:  ${ENCODER_URLS[*]}"
echo "  profiles:  ${PROFILER_DIR}  (encoder/ + prefill/ + decode/)"
echo ""
echo "Profile it from another shell:"
echo "  python scripts/disaggregation/profile_epd_cpu_sim.py \\"
echo "    --lang-url http://127.0.0.1:${LANG_PORT} \\"
echo "    --encoder-url ${ENCODER_URLS[0]} \\"
echo "    --image <url-or-path>"
echo "=========================================================="
echo "Ctrl-C to stop."
wait
