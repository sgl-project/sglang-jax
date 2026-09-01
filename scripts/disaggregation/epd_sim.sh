#!/usr/bin/env bash
#
# One-shot EPD CPU simulation: launch the tiers, drive requests, capture a
# profile, render the flame graph + single-request timeline, open the timeline,
# and tear everything down. One command, no second terminal.
#
#   MODEL_PATH=/path/to/qwen2.5-vl ./scripts/disaggregation/epd_sim.sh
#
# MODEL_PATH is OPTIONAL: if unset, a cached VLM (config+processor) is
# auto-discovered from the HuggingFace cache. Weights are never loaded.
#
# Coefficients (env, all optional; defaults give a readable illustrative graph):
#   SIM_ENC_BASE_MS SIM_ENC_MS_PER_TOKEN SIM_PREFILL_MS_PER_TOKEN
#   SIM_DECODE_MS_PER_SEQ SIM_TRANSFER_MS_PER_MB SIM_NET_RTT_MS
# Topology / workload (env, optional):
#   NUM_ENCODERS TP_SIZE DP_SIZE N_REQUESTS MAX_TOKENS IMAGE PROFILER_DIR PY_TRACER
#
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"

NUM_ENCODERS=${NUM_ENCODERS:-1}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}
DEVICE_COUNT=$((TP_SIZE * DP_SIZE))   # language mesh needs #devices == tp*dp
ENCODER_PORT_BASE=${ENCODER_PORT_BASE:-31001}
LANG_PORT=${LANG_PORT:-30000}
PROFILER_DIR=${PROFILER_DIR:-/tmp/epd-sim-profile}
N_REQUESTS=${N_REQUESTS:-20}
MAX_TOKENS=${MAX_TOKENS:-24}
CONCURRENCY=${CONCURRENCY:-1}   # >1 exercises prefill/decode batching
# Let the scheduler actually batch concurrent requests together.
MAX_RUNNING=${MAX_RUNNING:-$((CONCURRENCY > 8 ? CONCURRENCY : 8))}
PY_TRACER=${PY_TRACER:-0}   # 0 = clean stage view (good for flame graph + timeline)
IMAGE=${IMAGE:-}

SIM_ENC_BASE_MS=${SIM_ENC_BASE_MS:-10}
SIM_ENC_MS_PER_TOKEN=${SIM_ENC_MS_PER_TOKEN:-0.2}
SIM_PREFILL_MS_PER_TOKEN=${SIM_PREFILL_MS_PER_TOKEN:-1.0}
SIM_DECODE_MS_PER_SEQ=${SIM_DECODE_MS_PER_SEQ:-3}
SIM_TRANSFER_MS_PER_MB=${SIM_TRANSFER_MS_PER_MB:-10}
SIM_NET_RTT_MS=${SIM_NET_RTT_MS:-30}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PY=python
[ -x "${ROOT}/.venv/bin/python" ] && PY="${ROOT}/.venv/bin/python"

# Resolve a model directory (config + tokenizer + processor; weights unused).
# Prefer $MODEL_PATH, else auto-discover a cached VLM from the HF cache.
if [ -z "${MODEL_PATH}" ]; then
  MODEL_PATH=$("${PY}" - <<'PY'
import glob, json, os
best, best_score = None, (-1, 1 << 30)
for cfg in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*/snapshots/*/config.json")):
    try:
        d = json.load(open(cfg))
    except Exception:
        continue
    mt = str(d.get("model_type", "")).lower()
    arch = ",".join(d.get("architectures") or [])
    is_vlm = ("vision_config" in d) or ("image_token_id" in d) or ("vl" in mt) or ("VL" in arch)
    if not is_vlm:
        continue
    layers = d.get("num_hidden_layers") or (d.get("text_config") or {}).get("num_hidden_layers") or 1 << 20
    score = (2 if "vl" in mt else 1, layers)  # prefer *vl* model_type, then fewer layers
    if (score[0], -score[1]) > (best_score[0], -best_score[1]):
        best, best_score = os.path.dirname(cfg), score
print(best or "")
PY
)
  if [ -z "${MODEL_PATH}" ]; then
    echo "no cached VLM found in ~/.cache/huggingface/hub; set MODEL_PATH=/path/to/vlm" >&2
    exit 2
  fi
  echo ">> auto-discovered model config: ${MODEL_PATH}"
fi

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=${DEVICE_COUNT} ${XLA_FLAGS:-}"
export SGLANG_JAX_PROFILER_DIR="${PROFILER_DIR}"
rm -rf "${PROFILER_DIR}"; mkdir -p "${PROFILER_DIR}"

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  echo ">> stopping servers"
  for pid in "${PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_for_health() {
  for _ in $(seq 1 120); do
    curl -fsS "$1" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "timed out waiting for $1" >&2; exit 1
}

sim_args=(
  --simulate-compute
  --simulate-compute-encoder-base-ms "${SIM_ENC_BASE_MS}"
  --simulate-compute-encoder-ms-per-token "${SIM_ENC_MS_PER_TOKEN}"
  --simulate-compute-prefill-ms-per-token "${SIM_PREFILL_MS_PER_TOKEN}"
  --simulate-compute-decode-ms-per-seq "${SIM_DECODE_MS_PER_SEQ}"
  --simulate-transfer-ms-per-mb "${SIM_TRANSFER_MS_PER_MB}"
  --simulate-network-rtt-ms "${SIM_NET_RTT_MS}"
)
common_args=(
  --model-path "${MODEL_PATH}" --tp-size "${TP_SIZE}" --dp-size "${DP_SIZE}"
  --device cpu --load-format dummy --dtype bfloat16 --attention-backend native
  --trust-remote-code --disaggregation-host-ip 127.0.0.1
)

ENCODER_URLS=()
for ((i = 0; i < NUM_ENCODERS; i++)); do
  port=$((ENCODER_PORT_BASE + i))
  echo ">> starting encoder ${i} on :${port}"
  "${PY}" -m sgl_jax.launch_server "${common_args[@]}" "${sim_args[@]}" \
    --encoder-only --disable-precompile --host 127.0.0.1 --port "${port}" \
    > "${PROFILER_DIR}/encoder_${i}.log" 2>&1 &
  PIDS+=($!)
  ENCODER_URLS+=("http://127.0.0.1:${port}")
done

echo ">> starting language server on :${LANG_PORT}"
"${PY}" -m sgl_jax.launch_server "${common_args[@]}" "${sim_args[@]}" \
  --language-only --encoder-urls "${ENCODER_URLS[@]}" \
  --context-length 4096 --max-running-requests "${MAX_RUNNING}" --mem-fraction-static 0.1 \
  --host 127.0.0.1 --port "${LANG_PORT}" > "${PROFILER_DIR}/language.log" 2>&1 &
PIDS+=($!)

echo ">> waiting for health (first run compiles; ~30-60s)"
for url in "${ENCODER_URLS[@]}"; do wait_for_health "${url}/health"; done
wait_for_health "http://127.0.0.1:${LANG_PORT}/health"
echo ">> all healthy"

# Auto-generate a small test image if none supplied (works offline).
if [ -z "${IMAGE}" ]; then
  IMAGE="${PROFILER_DIR}/_test.png"
  "${PY}" - "${IMAGE}" <<'PY'
import sys
try:
    from PIL import Image
    import random
    img = Image.new("RGB", (112, 112), (random.randint(0,255),)*3)
    img.save(sys.argv[1])
except Exception as e:
    sys.exit(f"could not create test image ({e}); pass IMAGE=/path/to.jpg")
PY
fi

enc_flags=()
for url in "${ENCODER_URLS[@]}"; do enc_flags+=(--encoder-url "${url}"); done

# Footgun guard: python_tracer_level=1 records every Python call; the trace->JSON
# converter caps at ~1M events and silently truncates in Perfetto. Keep level-1
# captures to a tiny slice (1 request, few tokens, no concurrency).
if [ "${PY_TRACER}" -ge 1 ] && { [ "$((N_REQUESTS * MAX_TOKENS))" -gt 40 ] || [ "${CONCURRENCY}" -gt 1 ]; }; then
  echo "WARNING: PY_TRACER=1 with this workload will likely TRUNCATE the trace at ~1M"
  echo "  events (Perfetto shows it cut). For un-truncated function detail use e.g."
  echo "  PY_TRACER=1 CONCURRENCY=1 N_REQUESTS=1 MAX_TOKENS=8 ...; for concurrency/large"
  echo "  workloads use PY_TRACER=0 (stage annotations, never near the cap)."
fi

echo ">> profiling ${N_REQUESTS} requests (concurrency ${CONCURRENCY})"
"${PY}" "${SCRIPT_DIR}/profile_epd_cpu_sim.py" \
  --lang-url "http://127.0.0.1:${LANG_PORT}" "${enc_flags[@]}" \
  --image "${IMAGE}" --n-requests "${N_REQUESTS}" --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --warmup 1 --python-tracer-level "${PY_TRACER}" --profiler-dir "${PROFILER_DIR}"

echo ">> rendering flame graph + timeline"
"${PY}" "${SCRIPT_DIR}/trace_to_flamegraph.py" --profiler-dir "${PROFILER_DIR}"
"${PY}" "${SCRIPT_DIR}/trace_to_timeline_html.py" --profiler-dir "${PROFILER_DIR}" --rtt-ms "${SIM_NET_RTT_MS}"

# Level-1 traces carry the stdlib/framework firehose; auto-slim to project
# functions so the .slim.trace.json.gz is a readable Perfetto middle ground.
if [ "${PY_TRACER}" -ge 1 ]; then
  echo ">> slimming level-1 trace (project functions only)"
  "${PY}" "${SCRIPT_DIR}/trace_slim.py" --profiler-dir "${PROFILER_DIR}" || true
fi

HTML="${PROFILER_DIR}/epd_timeline.html"
echo ""
echo "=========================================================="
echo "Done. Artifacts in ${PROFILER_DIR}:"
echo "  epd_timeline.html   <- single-request critical path (open this first)"
echo "  epd_flamegraph.svg  <- CPU self-time flame graph"
if [ "${PY_TRACER}" -ge 1 ]; then
  echo "  {encoder_0,language}.slim.trace.json.gz  <- Perfetto (project funcs, de-noised)"
fi
echo "  {encoder_*,language}/plugins/profile/.../*.trace.json.gz  <- Perfetto (raw)"
echo "=========================================================="
if [ "${CONCURRENCY}" -gt 1 ]; then
  echo ""
  echo "NOTE (concurrency ${CONCURRENCY}):"
  echo "  * Requests now batch in the scheduler (see #running-req in language.log)."
  echo "  * The single-request TIMELINE assumes sequential drive; under concurrency"
  echo "    read the FLAME GRAPH + Perfetto instead (decode spans cover the batch)."
  echo "  * Decode is modeled as base_ms + per_seq_ms*batch. With the default"
  echo "    (base 0) decode grows linearly with batch, so batching looks harmful."
  echo "    For realistic batched decode set the FIXED cost via SIM_DECODE_BASE_MS"
  echo "    (dominant) and keep SIM_DECODE_MS_PER_SEQ small, e.g."
  echo "    SIM_DECODE_BASE_MS=20 SIM_DECODE_MS_PER_SEQ=0.5"
fi
if command -v open >/dev/null 2>&1; then open "${HTML}"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "${HTML}" >/dev/null 2>&1 || true
else echo "open ${HTML} in a browser"; fi
